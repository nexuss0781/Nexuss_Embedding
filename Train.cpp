// =============================================================================
// Train.cpp — HFAQE Training Loop
// =============================================================================
#ifndef HFAQE_TRAIN_CPP
#define HFAQE_TRAIN_CPP
//   - EthioBBPE Tokenizer (Component-1.1, via Input.cpp bridge)
//   - HFAQE Core (Core.cpp) forward + backward + weight update
//
// Spec coverage:
//   §2.3  Backward Pass  — sparse gradient accumulation
//   §2.5  Initialization — weight init before training
//   §3.3  mmap layout    — pin hot tier at training start
//   §5.6  Training Step  — single backward updates Q_H/S_H/A/B
//                          without O(V·d) memory spike
//
// Design:
//   1. TrainConfig        — all hyperparameters
//   2. LossFn             — cross-entropy loss with HFAQE LM head
//   3. TrainStep          — single minibatch forward/backward/update
//   4. Trainer            — epoch loop, logging, checkpoint save/load
//   5. train_demo()       — runnable demonstration
// =============================================================================

#include "Input.cpp"   // transitively includes Core.cpp

#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <fstream>
#include <stdexcept>

// =============================================================================
// TrainConfig — all training hyperparameters
// =============================================================================
struct TrainConfig {
    // Optimiser
    float lr           = 1e-3f;   // learning rate (SGD; extend to AdamW as needed)
    float lr_decay     = 0.99f;   // per-epoch LR decay
    float grad_clip    = 1.0f;    // global gradient norm clip threshold

    // Batch
    int   batch_size   = 32;      // sequences per batch
    int   max_seq_len  = 128;     // max tokens per sequence (truncation)
    int   epochs       = 3;

    // Logging / checkpoint
    int   log_every    = 50;      // steps between console log
    int   save_every   = 500;     // steps between checkpoint saves
    std::string ckpt_path = "hfaqe_checkpoint.bin";

    // Zipf frequency estimation
    // In production: supply real corpus frequencies.
    // Here we use Zipf(s=1) as a prior to build the hot/cold split.
    float zipf_s = 1.0f;
};

// =============================================================================
// GradAccumulator — thread-safe (single-thread here) fp32 gradient store
// Wraps the three gradient tensors inside HFAQE and adds clipping.
// =============================================================================
struct GradAccumulator {
    HFAQE& model;
    explicit GradAccumulator(HFAQE& m) : model(m) {}

    // Global L2 gradient norm across all parameters
    float global_norm() const {
        double sq = 0.0;
        for (float v : model.grad_Q) sq += (double)v*v;
        for (float v : model.grad_S) sq += (double)v*v;
        for (float v : model.grad_A) sq += (double)v*v;
        for (float v : model.grad_B) sq += (double)v*v;
        return static_cast<float>(std::sqrt(sq));
    }

    // Clip all gradients so global norm ≤ threshold
    void clip(float threshold) {
        float gnorm = global_norm();
        if (gnorm <= threshold || gnorm < 1e-8f) return;
        float scale = threshold / gnorm;
        for (float& v : model.grad_Q) v *= scale;
        for (float& v : model.grad_S) v *= scale;
        for (float& v : model.grad_A) v *= scale;
        for (float& v : model.grad_B) v *= scale;
    }
};

// =============================================================================
// CrossEntropyLoss — computes loss and ∂L/∂h from LM-head logits
//
// For a sequence T = [t_0, t_1, ..., t_{n-1}], the language-model loss
// is the mean negative log-likelihood of the next token:
//
//   L = -1/n · Σ_{i=0}^{n-2} log softmax(logits_i)[t_{i+1}]
//
// where logits_i = lm_head(X[i,:]).
//
// Backward w.r.t. logits:
//   ∂L/∂logits_i[j] = (softmax_j - 1{j == t_{i+1}}) / n
//
// We return:
//   loss         — scalar cross-entropy
//   dL_dX        — ∂L/∂X[n×d]  computed via LM-head Jacobian
// =============================================================================
struct LossOutput {
    float              loss;
    std::vector<float> dL_dX;   // [n × d]
};

static LossOutput cross_entropy_lm_loss(
    const HFAQE&              model,
    const std::vector<float>& X,        // embeddings [n × d] fp32
    const std::vector<int>&   token_ids // length n
)
{
    int n = static_cast<int>(token_ids.size());
    int d = model.cfg.d;
    int V = model.cfg.V;
    if (n < 2) {
        // Need at least 2 tokens for a next-token prediction step
        return {0.0f, std::vector<float>(static_cast<size_t>(n)*d, 0.0f)};
    }

    LossOutput out;
    out.dL_dX.assign(static_cast<size_t>(n)*d, 0.0f);
    float total_loss = 0.0f;
    int n_steps = n - 1;

    // Pre-allocate buffers
    std::vector<float> logits(V);
    std::vector<float> softmax_probs(V);

    for (int i = 0; i < n_steps; ++i) {
        const float* hi = X.data() + static_cast<ptrdiff_t>(i)*d;
        int target = token_ids[i + 1];

        // Compute logits: lm_head(hi)
        model.lm_head(hi, logits.data());

        // Numerically stable softmax
        float max_logit = *std::max_element(logits.begin(), logits.end());
        float sum_exp = 0.0f;
        for (int t = 0; t < V; ++t) {
            softmax_probs[t] = std::exp(logits[t] - max_logit);
            sum_exp += softmax_probs[t];
        }
        float inv_sum = 1.0f / (sum_exp + 1e-10f);
        for (int t = 0; t < V; ++t) softmax_probs[t] *= inv_sum;

        // Loss contribution: -log p(target)
        float p_target = std::max(softmax_probs[target], 1e-10f);
        total_loss += -std::log(p_target);

        // ∂L/∂logits[i][j] = (p_j - 1{j==target}) / n_steps
        std::vector<float> dL_dlogits(V);
        for (int t = 0; t < V; ++t)
            dL_dlogits[t] = (softmax_probs[t] - (t == target ? 1.0f : 0.0f))
                            / static_cast<float>(n_steps);

        // ∂L/∂h[i] via LM head Jacobian: ∂logits/∂h = E^T  (embedding matrix)
        // Hot: ∂logits[t]/∂h[j] = Ê_t[j]  (dequantized row)
        // Cold: ∂logits[t]/∂h[j] = (B · A_t^T)[j]  = Σ_k B[j,k]·A_t[k]
        //
        // ∂L/∂h[i][j] = Σ_t ∂L/∂logits[t] · ∂logits[t]/∂h[j]
        //             = Σ_{t∈H} dL_dlogits[t] · Ê_t[j]
        //             + Σ_{t∈C} dL_dlogits[t] · (B·α_t)[j]
        //
        float* dhi = out.dL_dX.data() + static_cast<ptrdiff_t>(i)*d;

        // Hot contribution
        for (int slot = 0; slot < model.hot.K; ++slot) {
            int gid = model.hot.global_ids[slot];
            float dg  = dL_dlogits[gid];
            if (std::abs(dg) < 1e-12f) continue;
            // Dequantize hot row on the fly (use inline block-wise)
            const int8_t*  qrow = model.hot.row_q(slot);
            const float*   srow = model.hot.row_s(slot);
            int m_blk = model.cfg.m();
            for (int b = 0; b < m_blk; ++b) {
                int start = b * model.cfg.B;
                int end   = std::min(start + model.cfg.B, d);
                float s   = srow[b];
                for (int j = start; j < end; ++j)
                    dhi[j] += dg * s * static_cast<float>(qrow[j]);
            }
        }

        // Cold contribution: precompute z_c = Σ_t∈C dL_dlogits[t]·A[t,k]  for each k
        // then ∂L/∂h[j] += Σ_k z_c[k] · B[j,k]
        std::vector<float> z_c(model.cfg.r, 0.0f);
        for (int cslot = 0; cslot < model.cold.Vc; ++cslot) {
            int gid = model.cold.global_ids[cslot];
            float dg = dL_dlogits[gid];
            if (std::abs(dg) < 1e-12f) continue;
            const fp16* arow = model.cold.row_a(cslot);
            for (int k = 0; k < model.cfg.r; ++k)
                z_c[k] += dg * bf16_to_f32(arow[k]);
        }
        // ∂L/∂h[j] += Σ_k B[j,k] · z_c[k]
        for (int k = 0; k < model.cfg.r; ++k) {
            if (std::abs(z_c[k]) < 1e-12f) continue;
            const fp16* bk = model.cold.basis_col(k);
            float zk = z_c[k];
            for (int j = 0; j < d; ++j)
                dhi[j] += zk * bf16_to_f32(bk[j]);
        }
    }

    out.loss = total_loss / static_cast<float>(n_steps);
    return out;
}

// =============================================================================
// TrainStep — one minibatch forward + backward + gradient update
// Returns: mean cross-entropy loss for this batch
// =============================================================================
struct StepResult {
    float loss;
    float grad_norm_before_clip;
    int   n_tokens;
    double step_ms;
};

static StepResult train_step(
    HFAQE&                            model,
    const std::vector<std::vector<int>>& batch_ids,  // [batch × seq_len]
    const TrainConfig&                cfg,
    GradAccumulator&                  grad_acc)
{
    using Clock = std::chrono::high_resolution_clock;
    auto t0 = Clock::now();

    model.zero_grad();

    float total_loss  = 0.0f;
    int   total_toks  = 0;
    int   n_seq       = static_cast<int>(batch_ids.size());

    for (const auto& ids : batch_ids) {
        if (ids.size() < 2) continue;  // need at least input + target

        int n = static_cast<int>(ids.size());

        // ---- Forward: token IDs → embeddings (SPEC §2.2) ----
        std::vector<float> X = model.forward(ids);

        // ---- Loss + ∂L/∂X computation (cross-entropy LM) ----
        auto loss_out = cross_entropy_lm_loss(model, X, ids);
        total_loss += loss_out.loss * (n - 1);
        total_toks += (n - 1);

        // ---- Backward: ∂L/∂X → ∂L/∂{Q_H,S_H,A,B} (SPEC §2.3) ----
        model.backward(loss_out.dL_dX.data(), ids.data(), n);
    }

    // ---- Gradient clipping (§5.4 explosion guard) ----
    float gnorm = grad_acc.global_norm();
    grad_acc.clip(cfg.grad_clip);

    // ---- Weight update (SGD, §2.5 apply_gradients) ----
    model.apply_gradients(cfg.lr);

    auto t1 = Clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    float mean_loss = (total_toks > 0)
                    ? total_loss / static_cast<float>(total_toks) : 0.0f;
    return {mean_loss, gnorm, total_toks, ms};
}

// =============================================================================
// Checkpoint: save / load model weights to binary file
// Format (minimal):
//   [header: V d r K B] [Q_H int8] [S_H fp32] [A fp16] [Basis fp16]
//   [hot_global_ids int32] [cold_global_ids int32]
// =============================================================================
static bool save_checkpoint(const HFAQE& model, const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    // Header
    int header[5] = {model.cfg.V, model.cfg.d, model.cfg.r,
                     model.cfg.K, model.cfg.B};
    f.write(reinterpret_cast<const char*>(header), sizeof(header));
    // Q_H
    f.write(reinterpret_cast<const char*>(model.hot.Q_H.data()),
            static_cast<std::streamsize>(model.hot.Q_H.size() * sizeof(int8_t)));
    // S_H
    f.write(reinterpret_cast<const char*>(model.hot.S_H.data()),
            static_cast<std::streamsize>(model.hot.S_H.size() * sizeof(float)));
    // A
    f.write(reinterpret_cast<const char*>(model.cold.A.data()),
            static_cast<std::streamsize>(model.cold.A.size() * sizeof(fp16)));
    // Basis
    f.write(reinterpret_cast<const char*>(model.cold.Basis.data()),
            static_cast<std::streamsize>(model.cold.Basis.size() * sizeof(fp16)));
    // hot_global_ids
    f.write(reinterpret_cast<const char*>(model.hot.global_ids.data()),
            static_cast<std::streamsize>(model.hot.global_ids.size() * sizeof(int)));
    // cold_global_ids
    f.write(reinterpret_cast<const char*>(model.cold.global_ids.data()),
            static_cast<std::streamsize>(model.cold.global_ids.size() * sizeof(int)));
    return f.good();
}

static bool load_checkpoint(HFAQE& model, const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    int header[5];
    f.read(reinterpret_cast<char*>(header), sizeof(header));
    if (header[0] != model.cfg.V || header[1] != model.cfg.d
     || header[2] != model.cfg.r || header[3] != model.cfg.K
     || header[4] != model.cfg.B)
        return false; // config mismatch

    f.read(reinterpret_cast<char*>(model.hot.Q_H.data()),
           static_cast<std::streamsize>(model.hot.Q_H.size() * sizeof(int8_t)));
    f.read(reinterpret_cast<char*>(model.hot.S_H.data()),
           static_cast<std::streamsize>(model.hot.S_H.size() * sizeof(float)));
    f.read(reinterpret_cast<char*>(model.cold.A.data()),
           static_cast<std::streamsize>(model.cold.A.size() * sizeof(fp16)));
    f.read(reinterpret_cast<char*>(model.cold.Basis.data()),
           static_cast<std::streamsize>(model.cold.Basis.size() * sizeof(fp16)));
    f.read(reinterpret_cast<char*>(model.hot.global_ids.data()),
           static_cast<std::streamsize>(model.hot.global_ids.size() * sizeof(int)));
    f.read(reinterpret_cast<char*>(model.cold.global_ids.data()),
           static_cast<std::streamsize>(model.cold.global_ids.size() * sizeof(int)));
    // Rebuild idx maps
    model.hot.idx.clear();
    for (int s = 0; s < model.hot.K; ++s)
        model.hot.idx[model.hot.global_ids[s]] = s;
    model.cold.idx.clear();
    for (int s = 0; s < model.cold.Vc; ++s)
        model.cold.idx[model.cold.global_ids[s]] = s;
    return f.good();
}

// =============================================================================
// Trainer — full training loop over a corpus of token-ID sequences
// =============================================================================
class Trainer {
public:
    HFAQE&          model;
    EmbedPipeline&  pipeline;
    TrainConfig     cfg;

    explicit Trainer(HFAQE& m, EmbedPipeline& p, const TrainConfig& c)
        : model(m), pipeline(p), cfg(c) {}

    // -----------------------------------------------------------------
    // train: given a corpus (list of raw text strings), run for cfg.epochs
    // -----------------------------------------------------------------
    void train(const std::vector<std::string>& corpus) {
        // Pin hot tier in RAM for the duration of training (§3.3)
        model.pin_hot_tier();

        // Tokenize full corpus once
        std::printf("[Trainer] Tokenizing %zu documents...\n", corpus.size());
        auto all_ids = pipeline.tokenizer.encode_batch(corpus);

        // Clamp all IDs to [0, V)
        for (auto& seq : all_ids)
            for (auto& id : seq)
                id = std::max(0, std::min(id, model.cfg.V - 1));

        std::printf("[Trainer] Corpus: %zu sequences\n", all_ids.size());

        float lr = cfg.lr;
        int global_step = 0;

        for (int epoch = 0; epoch < cfg.epochs; ++epoch) {
            // Shuffle corpus each epoch
            std::mt19937 rng(epoch + 1);
            std::shuffle(all_ids.begin(), all_ids.end(), rng);

            float epoch_loss = 0.0f;
            int   epoch_toks = 0;
            int   n_batches  = 0;

            // Build minibatches
            int N = static_cast<int>(all_ids.size());
            for (int b_start = 0; b_start < N; b_start += cfg.batch_size) {
                int b_end = std::min(b_start + cfg.batch_size, N);
                std::vector<std::vector<int>> batch(
                    all_ids.begin() + b_start,
                    all_ids.begin() + b_end);

                // Truncate sequences to max_seq_len
                for (auto& seq : batch)
                    if ((int)seq.size() > cfg.max_seq_len)
                        seq.resize(cfg.max_seq_len);

                GradAccumulator gacc(model);
                // Temporarily adjust lr in model (passed to apply_gradients below)
                TrainConfig step_cfg = cfg;
                step_cfg.lr = lr;

                auto res = train_step(model, batch, step_cfg, gacc);
                epoch_loss += res.loss * res.n_tokens;
                epoch_toks += res.n_tokens;
                ++n_batches;
                ++global_step;

                if (global_step % cfg.log_every == 0) {
                    float avg = (epoch_toks > 0) ? epoch_loss / epoch_toks : 0.0f;
                    std::printf(
                        "[Trainer] epoch=%d step=%d loss=%.4f "
                        "gnorm=%.3f step_ms=%.1f\n",
                        epoch + 1, global_step, avg,
                        res.grad_norm_before_clip, res.step_ms);
                }

                if (global_step % cfg.save_every == 0) {
                    if (save_checkpoint(model, cfg.ckpt_path))
                        std::printf("[Trainer] Checkpoint saved → %s\n",
                                    cfg.ckpt_path.c_str());
                }
            }

            float epoch_avg = (epoch_toks > 0)
                            ? epoch_loss / epoch_toks : 0.0f;
            std::printf("[Trainer] === Epoch %d done | avg_loss=%.4f | lr=%.6f ===\n",
                        epoch + 1, epoch_avg, lr);

            // LR decay
            lr *= cfg.lr_decay;
        }

        // Final checkpoint
        if (save_checkpoint(model, cfg.ckpt_path))
            std::printf("[Trainer] Final checkpoint saved → %s\n",
                        cfg.ckpt_path.c_str());
    }

    // -----------------------------------------------------------------
    // evaluate_perplexity: measures cross-entropy on a held-out set
    // SPEC §5.6: WikiText PPL increase < 0.5 vs baseline
    // -----------------------------------------------------------------
    float evaluate_perplexity(const std::vector<std::string>& eval_corpus) {
        auto all_ids = pipeline.tokenizer.encode_batch(eval_corpus);
        for (auto& seq : all_ids)
            for (auto& id : seq)
                id = std::max(0, std::min(id, model.cfg.V - 1));

        double total_nll  = 0.0;
        int    total_toks = 0;
        int V = model.cfg.V;
        std::vector<float> logits(V);

        for (const auto& ids : all_ids) {
            if ((int)ids.size() < 2) continue;
            int n = static_cast<int>(ids.size());
            auto X = model.forward(ids);
            for (int i = 0; i < n - 1; ++i) {
                const float* hi = X.data() + static_cast<ptrdiff_t>(i)*model.cfg.d;
                model.lm_head(hi, logits.data());
                // Stable log-softmax
                float mx = *std::max_element(logits.begin(), logits.end());
                float lse = 0.0f;
                for (float l : logits) lse += std::exp(l - mx);
                lse = mx + std::log(lse + 1e-10f);
                total_nll  -= static_cast<double>(logits[ids[i+1]] - lse);
                total_toks += 1;
            }
        }
        if (total_toks == 0) return 0.0f;
        double avg_nll = total_nll / total_toks;
        return static_cast<float>(std::exp(avg_nll)); // perplexity
    }
};

// =============================================================================
// train_demo — runnable end-to-end training demonstration
// Uses synthetic token-ID sequences mimicking Zipfian Amharic text.
// =============================================================================
static void train_demo() {
    std::printf("\n=== Train.cpp: HFAQE Training Demo ===\n");

    // Model config (small for demo speed)
    HFAQEConfig model_cfg;
    model_cfg.V = 16000;
    model_cfg.d = 128;
    model_cfg.r = 32;
    model_cfg.K = 1024;
    model_cfg.B = 64;

    HFAQE model(model_cfg);
    auto freq = zipf_frequencies(model_cfg.V);
    model.build_frequency_tiers(freq);
    model.initialize_weights(42);

    EmbedPipeline pipeline(&model);

    // Synthetic corpus: 200 short "sentences" as random token-ID sequences
    // with Zipfian distribution (simulating Amharic text)
    std::mt19937_64 rng(77);
    // Zipf sampler via inverse CDF
    auto zipf_sample = [&](int V_vocab) -> int {
        std::vector<double> cdf(V_vocab);
        double H = 0.0;
        for (int t = 1; t <= V_vocab; ++t) H += 1.0/t;
        double s = 0.0;
        for (int t = 0; t < V_vocab; ++t) { s += 1.0/(t+1)/H; cdf[t] = s; }
        double u = std::uniform_real_distribution<double>(0,1)(rng);
        return static_cast<int>(
            std::lower_bound(cdf.begin(), cdf.end(), u) - cdf.begin());
    };

    std::vector<std::string> corpus;
    corpus.reserve(200);
    for (int i = 0; i < 200; ++i) {
        // Build a fake "text" as a space-separated string of ASCII digits
        // (so stub tokenizer maps each char → token ID)
        std::string doc;
        int seq_len = 10 + (rng() % 20);
        for (int k = 0; k < seq_len; ++k) {
            int t = zipf_sample(model_cfg.V);
            doc += std::to_string(t) + " ";
        }
        corpus.push_back(doc);
    }

    // Training config
    TrainConfig tcfg;
    tcfg.lr        = 5e-3f;
    tcfg.epochs    = 2;
    tcfg.batch_size = 16;
    tcfg.max_seq_len = 32;
    tcfg.log_every  = 5;
    tcfg.save_every = 1000; // don't save in demo
    tcfg.ckpt_path  = ""; // no save

    // Split corpus: 80% train, 20% eval
    int n_train = static_cast<int>(corpus.size() * 0.8f);
    std::vector<std::string> train_set(corpus.begin(), corpus.begin() + n_train);
    std::vector<std::string> eval_set(corpus.begin() + n_train, corpus.end());

    Trainer trainer(model, pipeline, tcfg);
    trainer.train(train_set);

    float ppl = trainer.evaluate_perplexity(eval_set);
    std::printf("[Trainer] Eval perplexity = %.2f\n", ppl);
    std::printf("=== Train.cpp Demo Complete ===\n\n");
}


#endif // HFAQE_TRAIN_CPP
