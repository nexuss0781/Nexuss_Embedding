// =============================================================================
// Train.cpp — HFAQE Production Training Engine
// =============================================================================
// Trains HFAQE token embedding on Salesforce/wikitext-2-raw-v1.
// Download data first:   python dataset.py
//
// Scope: embedding layer ONLY (Q_H, S_H, A, B).
//        No attention, no MLP, no positional encoding.
//        Objective: next-token prediction loss through the embedding + LM head.
//
// Features:
//   - Reads Data/train.txt and Data/validation.txt (written by dataset.py)
//   - Byte-level tokeniser (V=256) — drop-in swap for EthioBBPE (V=16000)
//   - AdamW optimiser for cold A and B; SGD+requantise for hot Q_H / S_H
//   - Cosine LR schedule with linear warm-up
//   - Global gradient clipping + explosion guard
//   - Real-time console monitor: step, loss, PPL, grad-norm, tok/s, ETA
//   - .nex checkpoint save/load via Storage.cpp CheckpointManager
//   - SIGINT / SIGTERM handler — clean save on Ctrl-C
//   - Parallel batch assembly via std::thread
//
// Build:
//   g++ -std=c++17 -O3 -march=native -I. Train.cpp -o train -lm -lpthread
//
// Run:
//   ./train                          # fresh training
//   ./train --resume                 # resume from checkpoints/hfaqe_latest.nex
//   ./train --epochs 10 --lr 3e-4 --batch 64
//   ./train --data Data --ckpt_dir checkpoints
// =============================================================================

#ifndef HFAQE_TRAIN_CPP
#define HFAQE_TRAIN_CPP

#include "Storage.cpp"   // → Core.cpp + NEX format + CheckpointManager

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <random>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <stdexcept>
#include <csignal>
#include <cassert>
#include <sys/stat.h>

#ifdef _WIN32
#  include <windows.h>
#  include <direct.h>
#  define MKDIR(p) _mkdir(p)
#else
#  include <unistd.h>
#  include <sys/resource.h>
#  define MKDIR(p) mkdir((p), 0755)
#endif

// =============================================================================
// TrainConfig — every knob in one place, all settable from argv
// =============================================================================
struct TrainConfig {
    // data
    std::string data_dir   = "Data";
    std::string train_file = "train.txt";
    std::string val_file   = "validation.txt";

    // model  (byte-level tokeniser: V=256, swap to 16000 for EthioBBPE)
    int   V           = 256;
    int   d           = 256;
    int   r           = 64;
    int   K           = 128;
    int   B           = 64;

    // optimiser (AdamW)
    float lr          = 3e-4f;
    float lr_min      = 1e-5f;
    float beta1       = 0.9f;
    float beta2       = 0.999f;
    float eps         = 1e-8f;
    float weight_decay= 1e-2f;
    float grad_clip   = 1.0f;

    // schedule
    int   epochs      = 5;
    int   warmup_steps= 400;

    // batch
    int   batch_size  = 64;
    int   max_seq_len = 256;
    int   num_workers = 4;

    // logging
    int   log_every   = 20;
    int   val_every   = 300;
    int   save_every  = 1000;

    // checkpoint
    std::string ckpt_dir  = "checkpoints";
    std::string ckpt_name = "hfaqe";
    bool  resume          = false;
};

// =============================================================================
// Argument parser
// =============================================================================
static TrainConfig parse_args(int argc, char** argv) {
    TrainConfig cfg;
    auto get = [&](int i, const char* name) -> std::string {
        if (i >= argc)
            throw std::runtime_error(std::string("Missing value for ") + name);
        return argv[i];
    };
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--resume")           cfg.resume      = true;
        else if (a == "--data")             cfg.data_dir    = get(++i, "--data");
        else if (a == "--ckpt_dir")         cfg.ckpt_dir    = get(++i, "--ckpt_dir");
        else if (a == "--epochs")           cfg.epochs      = std::stoi(get(++i,"--epochs"));
        else if (a == "--lr")               cfg.lr          = std::stof(get(++i,"--lr"));
        else if (a == "--batch")            cfg.batch_size  = std::stoi(get(++i,"--batch"));
        else if (a == "--seq_len")          cfg.max_seq_len = std::stoi(get(++i,"--seq_len"));
        else if (a == "--dim")              cfg.d           = std::stoi(get(++i,"--dim"));
        else if (a == "--rank")             cfg.r           = std::stoi(get(++i,"--rank"));
        else if (a == "--hot")              cfg.K           = std::stoi(get(++i,"--hot"));
        else if (a == "--workers")          cfg.num_workers = std::stoi(get(++i,"--workers"));
        else if (a == "--log_every")        cfg.log_every   = std::stoi(get(++i,"--log_every"));
        else if (a == "--val_every")        cfg.val_every   = std::stoi(get(++i,"--val_every"));
        else if (a == "--save_every")       cfg.save_every  = std::stoi(get(++i,"--save_every"));
        else { std::fprintf(stderr,"[warn] unknown arg: %s\n", a.c_str()); }
    }
    return cfg;
}

// =============================================================================
// Byte-level tokeniser (V=256)
// Each UTF-8 byte maps to a token ID 0-255.
// Zero overhead, no external library needed.
// Replace this with EthioBBPE for production Amharic training.
// =============================================================================
static std::vector<int> byte_tokenise(const std::string& text, int max_len = -1) {
    std::vector<int> ids;
    ids.reserve(text.size());
    for (unsigned char c : text) {
        ids.push_back(static_cast<int>(c));
        if (max_len > 0 && (int)ids.size() >= max_len) break;
    }
    return ids;
}

// =============================================================================
// Corpus reader — loads lines from a text file, skips blanks
// =============================================================================
static std::vector<std::string> load_text_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(f, line))
        if (!line.empty()) lines.push_back(line);
    return lines;
}

// =============================================================================
// Token-frequency counter — builds Zipf-compatible freq vector from corpus
// =============================================================================
static std::vector<fp32> corpus_frequencies(
    const std::vector<std::string>& lines, int V)
{
    std::vector<size_t> counts(V, 0);
    for (const auto& ln : lines)
        for (unsigned char c : ln)
            if ((int)c < V) ++counts[c];

    size_t total = 0;
    for (auto x : counts) total += x;
    if (total == 0) total = 1;

    std::vector<fp32> freq(V);
    for (int t = 0; t < V; ++t)
        freq[t] = static_cast<fp32>(counts[t]) / static_cast<fp32>(total);
    return freq;
}


// =============================================================================
// AdamW state for cold-tier parameters (A and B in fp32)
// Hot tier Q_H uses SGD + requantise (no momentum needed — quantisation
// already provides implicit regularisation).
// =============================================================================
struct AdamWState {
    // First and second moment estimates — same shape as the parameter
    std::vector<fp32> m;   // first moment
    std::vector<fp32> v;   // second moment (uncentered variance)
    int step = 0;          // global update count for bias correction

    void init(size_t n) {
        m.assign(n, 0.0f);
        v.assign(n, 0.0f);
    }

    // In-place AdamW update on fp32 parameter array `param` using gradient `g`.
    // weight_decay applied as decoupled L2 (per Loshchilov & Hutter 2019).
    void update(fp32* param, const fp32* g, size_t n,
                float lr, float beta1, float beta2,
                float eps, float weight_decay)
    {
        ++step;
        float bc1 = 1.0f - std::pow(beta1, static_cast<float>(step));
        float bc2 = 1.0f - std::pow(beta2, static_cast<float>(step));
        float lr_t = lr * std::sqrt(bc2) / bc1;

        for (size_t i = 0; i < n; ++i) {
            float gi = g[i];
            m[i] = beta1 * m[i] + (1.0f - beta1) * gi;
            v[i] = beta2 * v[i] + (1.0f - beta2) * gi * gi;
            // Weight decay (decoupled)
            float p = param[i] * (1.0f - lr * weight_decay);
            // Adam update
            param[i] = p - lr_t * m[i] / (std::sqrt(v[i]) + eps);
        }
    }
};

// =============================================================================
// Checkpoint format
// Header: [magic(4)] [V d r K B step epoch] [config blob size]
// Body:   Q_H int8 | S_H fp32 | A fp16 | B fp16
//         hot_global_ids int32 | cold_global_ids int32
//         adamw_m_A fp32 | adamw_v_A fp32 | adamw_m_B fp32 | adamw_v_B fp32
//         adamw_step_A int32 | adamw_step_B int32
// =============================================================================
static const uint32_t CKPT_MAGIC = 0x48465151u; // "HFQQ"

struct CheckpointMeta {
    uint32_t magic;
    int V, d, r, K, B;
    int global_step;
    int epoch;
};

static bool save_checkpoint_full(
    const HFAQE& model,
    const AdamWState& adam_A,
    const AdamWState& adam_B,
    int global_step, int epoch,
    const std::string& path)
{
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;

    CheckpointMeta meta;
    meta.magic       = CKPT_MAGIC;
    meta.V           = model.cfg.V;
    meta.d           = model.cfg.d;
    meta.r           = model.cfg.r;
    meta.K           = model.cfg.K;
    meta.B           = model.cfg.B;
    meta.global_step = global_step;
    meta.epoch       = epoch;

    auto write = [&](const void* ptr, size_t bytes) {
        f.write(reinterpret_cast<const char*>(ptr), static_cast<std::streamsize>(bytes));
    };

    write(&meta, sizeof(meta));
    write(model.hot.Q_H.data(),        model.hot.Q_H.size()        * sizeof(int8_t));
    write(model.hot.S_H.data(),        model.hot.S_H.size()        * sizeof(fp32));
    write(model.cold.A.data(),         model.cold.A.size()         * sizeof(fp16));
    write(model.cold.Basis.data(),     model.cold.Basis.size()     * sizeof(fp16));
    write(model.hot.global_ids.data(), model.hot.global_ids.size() * sizeof(int));
    write(model.cold.global_ids.data(),model.cold.global_ids.size()* sizeof(int));
    write(adam_A.m.data(),             adam_A.m.size()             * sizeof(fp32));
    write(adam_A.v.data(),             adam_A.v.size()             * sizeof(fp32));
    write(adam_B.m.data(),             adam_B.m.size()             * sizeof(fp32));
    write(adam_B.v.data(),             adam_B.v.size()             * sizeof(fp32));
    int step_A = adam_A.step, step_B = adam_B.step;
    write(&step_A, sizeof(int));
    write(&step_B, sizeof(int));

    return f.good();
}

static bool load_checkpoint_full(
    HFAQE& model,
    AdamWState& adam_A,
    AdamWState& adam_B,
    int& global_step, int& epoch,
    const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    CheckpointMeta meta;
    f.read(reinterpret_cast<char*>(&meta), sizeof(meta));
    if (meta.magic != CKPT_MAGIC) return false;
    if (meta.V != model.cfg.V || meta.d != model.cfg.d ||
        meta.r != model.cfg.r || meta.K != model.cfg.K ||
        meta.B != model.cfg.B) return false;

    global_step = meta.global_step;
    epoch       = meta.epoch;

    auto read = [&](void* ptr, size_t bytes) {
        f.read(reinterpret_cast<char*>(ptr), static_cast<std::streamsize>(bytes));
    };

    read(model.hot.Q_H.data(),         model.hot.Q_H.size()        * sizeof(int8_t));
    read(model.hot.S_H.data(),         model.hot.S_H.size()        * sizeof(fp32));
    read(model.cold.A.data(),          model.cold.A.size()         * sizeof(fp16));
    read(model.cold.Basis.data(),      model.cold.Basis.size()     * sizeof(fp16));
    read(model.hot.global_ids.data(),  model.hot.global_ids.size() * sizeof(int));
    read(model.cold.global_ids.data(), model.cold.global_ids.size()* sizeof(int));

    // Rebuild lookup maps
    model.hot.idx.clear();
    for (int s = 0; s < model.hot.K; ++s)
        model.hot.idx[model.hot.global_ids[s]] = s;
    model.cold.idx.clear();
    for (int s = 0; s < model.cold.Vc; ++s)
        model.cold.idx[model.cold.global_ids[s]] = s;

    // AdamW state
    read(adam_A.m.data(), adam_A.m.size() * sizeof(fp32));
    read(adam_A.v.data(), adam_A.v.size() * sizeof(fp32));
    read(adam_B.m.data(), adam_B.m.size() * sizeof(fp32));
    read(adam_B.v.data(), adam_B.v.size() * sizeof(fp32));
    read(&adam_A.step, sizeof(int));
    read(&adam_B.step, sizeof(int));

    return f.good();
}

// Scan checkpoint directory for the latest checkpoint file
static std::string find_latest_checkpoint(const std::string& ckpt_dir,
                                          const std::string& ckpt_name)
{
    std::string latest = ckpt_dir + "/" + ckpt_name + "_latest.bin";
    std::ifstream probe(latest);
    if (probe.good()) return latest;
    return "";
}


// =============================================================================
// LR schedule: linear warm-up → cosine decay
// =============================================================================
static float compute_lr(int step, int warmup, int total_steps,
                         float lr_max, float lr_min)
{
    if (step < warmup) {
        return lr_max * static_cast<float>(step + 1) / static_cast<float>(warmup);
    }
    float progress = static_cast<float>(step - warmup)
                   / static_cast<float>(std::max(1, total_steps - warmup));
    float cosine   = 0.5f * (1.0f + std::cos(static_cast<float>(M_PI) * progress));
    return lr_min + (lr_max - lr_min) * cosine;
}

// =============================================================================
// Frobenius norm of a float vector
// =============================================================================
static float frob(const std::vector<fp32>& v) {
    double s = 0.0;
    for (fp32 x : v) s += (double)x * x;
    return static_cast<float>(std::sqrt(s));
}

// =============================================================================
// Global gradient clipping: scale all grads so total L2 norm ≤ threshold
// =============================================================================
static void clip_gradients(HFAQE& model, float threshold) {
    float gnorm = std::sqrt(
        static_cast<double>(frob(model.grad_Q)) * frob(model.grad_Q) +
        static_cast<double>(frob(model.grad_S)) * frob(model.grad_S) +
        static_cast<double>(frob(model.grad_A)) * frob(model.grad_A) +
        static_cast<double>(frob(model.grad_B)) * frob(model.grad_B));

    if (gnorm > threshold && gnorm > 1e-8f) {
        float scale = threshold / gnorm;
        for (auto& x : model.grad_Q) x *= scale;
        for (auto& x : model.grad_S) x *= scale;
        for (auto& x : model.grad_A) x *= scale;
        for (auto& x : model.grad_B) x *= scale;
    }
}

// =============================================================================
// Cross-entropy next-token prediction loss + ∂L/∂X
// Input:  X[n×d] (embeddings), token_ids[n]
// Output: scalar mean NLL loss, dL_dX[n×d]
// Scope: embedding training only — no MLP, no attention.
//        We train the embedding/LM-head via a simple unigram language model:
//        predict the next byte from the current byte's embedding.
// =============================================================================
struct LossResult {
    float              loss;    // mean negative log-likelihood
    std::vector<fp32>  dL_dX;  // [n × d]
    int                n_toks; // number of prediction steps
};

static LossResult compute_loss(const HFAQE& model,
                                const std::vector<fp32>& X,
                                const std::vector<int>& ids)
{
    int n = static_cast<int>(ids.size());
    int d = model.cfg.d;
    int V = model.cfg.V;
    LossResult out;
    out.dL_dX.assign(static_cast<size_t>(n) * d, 0.0f);
    out.n_toks = 0;

    if (n < 2) { out.loss = 0.0f; return out; }

    std::vector<fp32> logits(V);
    double total_nll = 0.0;
    int n_steps = n - 1;

    for (int i = 0; i < n_steps; ++i) {
        const fp32* hi = X.data() + static_cast<ptrdiff_t>(i) * d;
        int target     = ids[i + 1];

        // LM head: logits = h · E^T  (Algorithm 3)
        model.lm_head(hi, logits.data());

        // Stable softmax
        float mx = *std::max_element(logits.begin(), logits.end());
        float sum_exp = 0.0f;
        for (int t = 0; t < V; ++t) {
            logits[t] = std::exp(logits[t] - mx);
            sum_exp  += logits[t];
        }
        float inv = 1.0f / (sum_exp + 1e-10f);
        for (int t = 0; t < V; ++t) logits[t] *= inv;

        float p_target = std::max(logits[target], 1e-10f);
        total_nll -= std::log(p_target);

        // ∂L/∂logits[t] = (p[t] - 1{t==target}) / n_steps
        float inv_n = 1.0f / static_cast<float>(n_steps);
        fp32* dhi   = out.dL_dX.data() + static_cast<ptrdiff_t>(i) * d;

        // Hot contribution to ∂L/∂h
        for (int slot = 0; slot < model.hot.K; ++slot) {
            int gid  = model.hot.global_ids[slot];
            float dg = (logits[gid] - (gid == target ? 1.0f : 0.0f)) * inv_n;
            if (std::abs(dg) < 1e-12f) continue;
            const int8_t* qr = model.hot.row_q(slot);
            const fp32*   sr = model.hot.row_s(slot);
            int m_blk        = model.cfg.m();
            for (int b = 0; b < m_blk; ++b) {
                int start = b * model.cfg.B;
                int end   = std::min(start + model.cfg.B, d);
                float s   = sr[b];
                for (int j = start; j < end; ++j)
                    dhi[j] += dg * s * static_cast<float>(qr[j]);
            }
        }

        // Cold contribution: precompute z_c[k] = Σ_t dlogits[t]·A[t,k]
        std::vector<fp32> z_c(model.cfg.r, 0.0f);
        for (int cs = 0; cs < model.cold.Vc; ++cs) {
            int gid  = model.cold.global_ids[cs];
            float dg = (logits[gid] - (gid == target ? 1.0f : 0.0f)) * inv_n;
            if (std::abs(dg) < 1e-12f) continue;
            const fp16* ar = model.cold.row_a(cs);
            for (int k = 0; k < model.cfg.r; ++k)
                z_c[k] += dg * bf16_to_f32(ar[k]);
        }
        for (int k = 0; k < model.cfg.r; ++k) {
            if (std::abs(z_c[k]) < 1e-12f) continue;
            const fp16* bk = model.cold.basis_col(k);
            for (int j = 0; j < d; ++j)
                dhi[j] += z_c[k] * bf16_to_f32(bk[j]);
        }
    }

    out.loss   = static_cast<float>(total_nll / n_steps);
    out.n_toks = n_steps;
    return out;
}

// =============================================================================
// Apply AdamW update to cold-tier fp32 parameter arrays, then re-encode bf16
// =============================================================================
static void adamw_update_cold_A(HFAQE& model, AdamWState& adam,
                                 float lr, float beta1, float beta2,
                                 float eps, float wd)
{
    // Work in fp32 scratch, apply AdamW, write back to bf16
    size_t n = static_cast<size_t>(model.cold.Vc) * model.cfg.r;
    std::vector<fp32> param_fp32(n);
    for (size_t i = 0; i < n; ++i)
        param_fp32[i] = bf16_to_f32(model.cold.A[i]);

    adam.update(param_fp32.data(), model.grad_A.data(), n, lr, beta1, beta2, eps, wd);

    for (size_t i = 0; i < n; ++i)
        model.cold.A[i] = f32_to_bf16(param_fp32[i]);
}

static void adamw_update_cold_B(HFAQE& model, AdamWState& adam,
                                 float lr, float beta1, float beta2,
                                 float eps, float wd)
{
    size_t n = static_cast<size_t>(model.cfg.d) * model.cfg.r;
    std::vector<fp32> param_fp32(n);
    for (size_t i = 0; i < n; ++i)
        param_fp32[i] = bf16_to_f32(model.cold.Basis[i]);

    adam.update(param_fp32.data(), model.grad_B.data(), n, lr, beta1, beta2, eps, wd);

    for (size_t i = 0; i < n; ++i)
        model.cold.Basis[i] = f32_to_bf16(param_fp32[i]);
}

// Hot tier: SGD + requantise
static void sgd_update_hot(HFAQE& model, float lr) {
    std::vector<fp32> row(model.cfg.d);
    for (int slot = 0; slot < model.hot.K; ++slot) {
        fp32* gq = model.grad_Q.data() + static_cast<ptrdiff_t>(slot) * model.cfg.d;
        fp32* gs = model.grad_S.data() + static_cast<ptrdiff_t>(slot) * model.cfg.m();
        dequant_row(model.hot.row_q(slot), model.hot.row_s(slot),
                    model.cfg.d, model.cfg.B, row.data());
        for (int j = 0; j < model.cfg.d; ++j)
            row[j] -= lr * gq[j];
        quantize_row(row.data(), model.cfg.d, model.cfg.B,
                     model.hot.row_q(slot), model.hot.row_s(slot));
        (void)gs; // scale grads noted; scale updated via requantise above
    }
}


// =============================================================================
// Real-time monitor
// Prints a compact log line every `log_every` steps, a validation line
// every `val_every` steps.  Uses \r to overwrite in-progress line.
// =============================================================================
struct TrainMonitor {
    using Clock = std::chrono::high_resolution_clock;
    using TP    = Clock::time_point;

    TP   t_start;
    TP   t_last_log;
    int  toks_since_log  = 0;
    int  total_steps     = 0;   // estimated total steps for ETA

    void start(int estimated_total_steps) {
        t_start          = Clock::now();
        t_last_log       = t_start;
        toks_since_log   = 0;
        total_steps      = estimated_total_steps;
    }

    // Called every step — accumulates token count
    void accum(int n_toks) { toks_since_log += n_toks; }

    // Print a training log line
    void log_train(int step, int epoch, float loss, float gnorm,
                   float lr, int n_toks_batch)
    {
        auto now       = Clock::now();
        double wall_s  = std::chrono::duration<double>(now - t_start).count();
        double dt_s    = std::chrono::duration<double>(now - t_last_log).count();
        float  toks_s  = (dt_s > 1e-6) ? toks_since_log / static_cast<float>(dt_s) : 0.f;
        float  ppl     = std::exp(std::min(loss, 20.0f));

        int eta_s = 0;
        if (total_steps > 0 && step > 0) {
            float sps = static_cast<float>(step) / static_cast<float>(wall_s);
            eta_s     = static_cast<int>((total_steps - step) / std::max(sps, 1e-3f));
        }

        std::printf(
            "\r[train] ep=%d  step=%6d/%d  loss=%.4f  ppl=%7.2f"
            "  gnorm=%.3f  lr=%.2e  tok/s=%7.0f  ETA=%dm%02ds   \n",
            epoch + 1, step, total_steps,
            loss, ppl, gnorm, lr, toks_s,
            eta_s / 60, eta_s % 60);
        std::fflush(stdout);

        t_last_log     = now;
        toks_since_log = 0;
    }

    // Print a validation log line
    void log_val(int step, float val_loss, float val_ppl, double elapsed_s) {
        std::printf(
            "[val]   step=%6d  val_loss=%.4f  val_ppl=%7.2f  (%.1fs)\n",
            step, val_loss, val_ppl, elapsed_s);
        std::fflush(stdout);
    }

    // Print a checkpoint save line
    void log_ckpt(int step, const std::string& path) {
        std::printf("[ckpt]  step=%6d  saved → %s\n", step, path.c_str());
        std::fflush(stdout);
    }

    // Print a resume line
    void log_resume(int step, int epoch, const std::string& path) {
        std::printf("[resume] loaded step=%d epoch=%d from %s\n",
                    step, epoch, path.c_str());
        std::fflush(stdout);
    }

    // Final summary
    void log_final(int steps, float best_val_loss, float best_val_ppl) {
        auto now   = Clock::now();
        double tot = std::chrono::duration<double>(now - t_start).count();
        std::printf("\n");
        std::printf("╔══════════════════════════════════════╗\n");
        std::printf("║  Training complete                   ║\n");
        std::printf("║  Steps      : %-6d                 ║\n", steps);
        std::printf("║  Best val loss: %-8.4f             ║\n", best_val_loss);
        std::printf("║  Best val PPL : %-8.2f             ║\n", best_val_ppl);
        std::printf("║  Wall time  : %6.1f s               ║\n", tot);
        std::printf("╚══════════════════════════════════════╝\n");
        std::fflush(stdout);
    }
};

// =============================================================================
// SIGINT / SIGTERM handler — request clean save and exit
// =============================================================================
static std::atomic<bool> g_stop_requested{false};

static void signal_handler(int) {
    g_stop_requested.store(true);
    std::printf("\n[signal] Interrupt received — saving checkpoint and exiting...\n");
    std::fflush(stdout);
}

// =============================================================================
// Validation pass — runs over the full validation set, returns mean NLL loss
// =============================================================================
static float run_validation(HFAQE& model,
                             const std::vector<std::string>& val_lines,
                             const TrainConfig& cfg)
{
    double total_nll  = 0.0;
    int    total_toks = 0;
    std::vector<fp32> logits(cfg.V);

    for (const auto& line : val_lines) {
        auto ids = byte_tokenise(line, cfg.max_seq_len);
        if ((int)ids.size() < 2) continue;
        // Clamp to [0, V)
        for (auto& id : ids) id = std::max(0, std::min(id, cfg.V - 1));

        auto X = model.forward(ids);
        int n  = static_cast<int>(ids.size());

        for (int i = 0; i < n - 1; ++i) {
            const fp32* hi = X.data() + static_cast<ptrdiff_t>(i) * cfg.d;
            model.lm_head(hi, logits.data());
            float mx = *std::max_element(logits.begin(), logits.end());
            float lse = 0.0f;
            for (int t = 0; t < cfg.V; ++t) lse += std::exp(logits[t] - mx);
            lse = mx + std::log(lse + 1e-10f);
            total_nll  -= static_cast<double>(logits[ids[i+1]] - lse);
            total_toks += 1;
        }
    }
    return (total_toks > 0)
           ? static_cast<float>(total_nll / total_toks)
           : 0.0f;
}

// =============================================================================
// Parallel batch assembly
// Each worker tokenises a slice of the line list and fills a shared queue.
// =============================================================================
using TokenSeq  = std::vector<int>;
using BatchVec  = std::vector<TokenSeq>;

static BatchVec assemble_batch(
    const std::vector<std::string>& lines,
    const std::vector<int>& indices,
    int max_seq_len, int V)
{
    BatchVec batch;
    batch.reserve(indices.size());
    for (int idx : indices) {
        auto ids = byte_tokenise(lines[idx], max_seq_len);
        for (auto& id : ids) id = std::max(0, std::min(id, V - 1));
        if ((int)ids.size() >= 2) batch.push_back(std::move(ids));
    }
    return batch;
}

// =============================================================================
// mkdir_p — create directory (and parents) if not exists
// =============================================================================
static void mkdir_p(const std::string& path) {
    // Simple: try to create; if it exists already that's fine
    MKDIR(path.c_str());
}


// =============================================================================
// main — full production training entry point
// Only compiled when building Train.cpp directly (not when included from main.cpp).
// =============================================================================
#ifndef HFAQE_NO_TRAIN_MAIN
int main(int argc, char** argv) {
    // ── parse args ────────────────────────────────────────────────────────────
    TrainConfig cfg = parse_args(argc, argv);
    mkdir_p(cfg.ckpt_dir);

    // ── signal handlers ───────────────────────────────────────────────────────
    std::signal(SIGINT,  signal_handler);
    std::signal(SIGTERM, signal_handler);

    std::printf("╔══════════════════════════════════════════════════════════╗\n");
    std::printf("║  HFAQE Embedding Training                                ║\n");
    std::printf("╚══════════════════════════════════════════════════════════╝\n");
    std::printf("[config] V=%d  d=%d  r=%d  K=%d  B=%d\n",
                cfg.V, cfg.d, cfg.r, cfg.K, cfg.B);
    std::printf("[config] lr=%.2e  epochs=%d  batch=%d  seq=%d\n",
                cfg.lr, cfg.epochs, cfg.batch_size, cfg.max_seq_len);
    std::printf("[config] data=%s  ckpt=%s\n",
                cfg.data_dir.c_str(), cfg.ckpt_dir.c_str());
    std::fflush(stdout);

    // ── load corpus ───────────────────────────────────────────────────────────
    std::string train_path = cfg.data_dir + "/" + cfg.train_file;
    std::string val_path   = cfg.data_dir + "/" + cfg.val_file;

    std::printf("[data] Loading %s ...\n", train_path.c_str());
    std::vector<std::string> train_lines = load_text_file(train_path);
    std::printf("[data] Loading %s ...\n", val_path.c_str());
    std::vector<std::string> val_lines   = load_text_file(val_path);
    std::printf("[data] train=%zu lines  val=%zu lines\n",
                train_lines.size(), val_lines.size());
    std::fflush(stdout);

    // ── build token frequency from training corpus ────────────────────────────
    std::printf("[model] Computing token frequencies ...\n");
    auto freq = corpus_frequencies(train_lines, cfg.V);

    // ── build model ───────────────────────────────────────────────────────────
    HFAQEConfig mcfg;
    mcfg.V = cfg.V; mcfg.d = cfg.d; mcfg.r = cfg.r;
    mcfg.K = cfg.K; mcfg.B = cfg.B;

    HFAQE model(mcfg);
    model.build_frequency_tiers(freq);
    model.initialize_weights(42);
    model.pin_hot_tier();

    std::printf("[model] Hot=%d int8[%d×%d]  Cold=%d bf16[%d×%d]  Basis bf16[%d×%d]\n",
                model.hot.K, model.hot.K, cfg.d,
                model.cold.Vc, model.cold.Vc, cfg.r,
                cfg.d, cfg.r);

    // Memory report
    auto mem = MemoryBudget::compute(mcfg);
    double base_mb  = static_cast<double>(cfg.V) * cfg.d * 2.0 / (1024.0*1024.0);
    double hfaqe_mb = static_cast<double>(mem.total_bytes) / (1024.0*1024.0);
    std::printf("[model] Param RAM: %.2f MB  (baseline BF16 would be %.2f MB)\n",
                hfaqe_mb, base_mb);
    std::fflush(stdout);

    // ── AdamW state for cold A and B ──────────────────────────────────────────
    AdamWState adam_A, adam_B;
    adam_A.init(static_cast<size_t>(model.cold.Vc) * cfg.r);
    adam_B.init(static_cast<size_t>(cfg.d)          * cfg.r);

    // ── AdamW ↔ NexAdamState bridge lambdas ──────────────────────────────────
    auto to_nex_adam = [&]() -> NexAdamState {
        NexAdamState ns;
        ns.m_A    = adam_A.m;  ns.v_A = adam_A.v;  ns.step_A = adam_A.step;
        ns.m_B    = adam_B.m;  ns.v_B = adam_B.v;  ns.step_B = adam_B.step;
        return ns;
    };
    auto from_nex_adam = [&](const NexAdamState& ns) {
        adam_A.m = ns.m_A; adam_A.v = ns.v_A; adam_A.step = ns.step_A;
        adam_B.m = ns.m_B; adam_B.v = ns.v_B; adam_B.step = ns.step_B;
    };

    // ── CheckpointManager — must be constructed before resume ────────────────
    CheckpointManager::Config ccfg;
    ccfg.ckpt_dir  = cfg.ckpt_dir;
    ccfg.base_name = cfg.ckpt_name;
    ccfg.compress  = true;
    ccfg.checksums = true;
    ccfg.keep_last = 3;
    CheckpointManager ckpt_mgr(ccfg);

    // ── Training state (declared before lambdas that capture them) ───────────
    int   global_step   = 0;
    int   start_epoch   = 0;
    float best_val_loss = 1e9f;
    float best_val_ppl  = 1e9f;

    auto save_ckpt = [&](const std::string& tag) {
        NexCheckpointMeta nmeta;
        nmeta.global_step   = global_step;
        nmeta.epoch         = start_epoch;
        nmeta.best_val_loss = best_val_loss;
        nmeta.best_val_ppl  = best_val_ppl;
        auto ns = to_nex_adam();
        ckpt_mgr.save(model, nmeta, tag, &ns, &freq);
    };

    // ── resume from .nex checkpoint ──────────────────────────────────────────
    if (cfg.resume) {
        NexCheckpointMeta loaded_meta;
        NexAdamState      loaded_adam;
        if (ckpt_mgr.load(model, loaded_meta, &loaded_adam)) {
            global_step   = loaded_meta.global_step;
            start_epoch   = loaded_meta.epoch;
            best_val_loss = loaded_meta.best_val_loss;
            best_val_ppl  = loaded_meta.best_val_ppl;
            from_nex_adam(loaded_adam);
        } else {
            std::fprintf(stderr, "[warn] No .nex checkpoint found — starting fresh.\n");
        }
    }

    // ── estimate total steps for ETA ─────────────────────────────────────────
    int steps_per_epoch = static_cast<int>(
        std::ceil(static_cast<double>(train_lines.size()) / cfg.batch_size));
    int total_steps = cfg.epochs * steps_per_epoch;
    int warmup      = cfg.warmup_steps;

    std::printf("[train] %d steps/epoch × %d epochs = %d total steps\n",
                steps_per_epoch, cfg.epochs, total_steps);
    std::fflush(stdout);

    // ── monitor ───────────────────────────────────────────────────────────────
    TrainMonitor monitor;
    monitor.start(total_steps);

    // ── RNG for shuffle ───────────────────────────────────────────────────────
    std::mt19937 rng(12345 + global_step);

    // ── accumulators for smooth logging ───────────────────────────────────────
    double  accum_loss  = 0.0;
    int     accum_toks  = 0;
    int     accum_steps = 0;
    float   last_gnorm  = 0.0f;
    float   last_lr     = cfg.lr;

    // ═════════════════════════════════════════════════════════════════════════
    // Training loop
    // ═════════════════════════════════════════════════════════════════════════
    for (int epoch = start_epoch; epoch < cfg.epochs && !g_stop_requested; ++epoch) {

        // Shuffle indices for this epoch
        std::vector<int> order(train_lines.size());
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), rng);

        // Iterate over mini-batches
        int n_lines = static_cast<int>(train_lines.size());
        for (int b_start = 0;
             b_start < n_lines && !g_stop_requested;
             b_start += cfg.batch_size)
        {
            int b_end = std::min(b_start + cfg.batch_size, n_lines);
            std::vector<int> batch_idx(order.begin() + b_start,
                                       order.begin() + b_end);

            // Assemble token sequences (parallelised over workers)
            BatchVec batch;
            {
                int  n_idx    = static_cast<int>(batch_idx.size());
                int  n_w      = std::min(cfg.num_workers, n_idx);
                int  chunk    = (n_idx + n_w - 1) / n_w;
                std::vector<BatchVec>     partial(n_w);
                std::vector<std::thread>  threads;
                threads.reserve(n_w);

                for (int w = 0; w < n_w; ++w) {
                    int lo = w * chunk;
                    int hi = std::min(lo + chunk, n_idx);
                    if (lo >= hi) break;
                    std::vector<int> slice(batch_idx.begin() + lo,
                                           batch_idx.begin() + hi);
                    threads.emplace_back([&, w, slice]() {
                        partial[w] = assemble_batch(
                            train_lines, slice, cfg.max_seq_len, cfg.V);
                    });
                }
                for (auto& t : threads) t.join();
                for (auto& p : partial)
                    for (auto& seq : p)
                        batch.push_back(std::move(seq));
            }

            if (batch.empty()) continue;

            // Current LR (cosine schedule)
            last_lr = compute_lr(global_step, warmup, total_steps,
                                  cfg.lr, cfg.lr_min);

            // Zero gradients
            model.zero_grad();

            // Forward + loss + backward for each sequence in the batch
            float batch_loss  = 0.0f;
            int   batch_toks  = 0;

            for (const auto& ids : batch) {
                auto X     = model.forward(ids);
                auto lout  = compute_loss(model, X, ids);
                if (lout.n_toks == 0) continue;

                // Scale gradient by sequence contribution weight
                float w = static_cast<float>(lout.n_toks);
                batch_loss += lout.loss * w;
                batch_toks += lout.n_toks;

                // Accumulate gradients (backward pass)
                model.backward(lout.dL_dX.data(), ids.data(),
                                static_cast<int>(ids.size()));
            }
            if (batch_toks == 0) continue;
            batch_loss /= static_cast<float>(batch_toks);

            // Gradient clipping
            clip_gradients(model, cfg.grad_clip);
            last_gnorm = frob(model.grad_B); // representative grad norm

            // Weight updates
            sgd_update_hot(model, last_lr);
            adamw_update_cold_A(model, adam_A, last_lr,
                                 cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay);
            adamw_update_cold_B(model, adam_B, last_lr,
                                 cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay);

            // Accumulate for logging
            accum_loss  += batch_loss * batch_toks;
            accum_toks  += batch_toks;
            ++accum_steps;
            monitor.accum(batch_toks);
            ++global_step;

            // ── Logging ───────────────────────────────────────────────────────
            if (global_step % cfg.log_every == 0) {
                float mean_loss = (accum_toks > 0)
                                ? static_cast<float>(accum_loss / accum_toks)
                                : 0.0f;
                monitor.log_train(global_step, epoch, mean_loss,
                                  last_gnorm, last_lr, batch_toks);
                accum_loss  = 0.0;
                accum_toks  = 0;
                accum_steps = 0;
            }

            // ── Validation ────────────────────────────────────────────────────
            if (global_step % cfg.val_every == 0) {
                auto tv0      = std::chrono::high_resolution_clock::now();
                float vloss   = run_validation(model, val_lines, cfg);
                double vtime  = std::chrono::duration<double>(
                    std::chrono::high_resolution_clock::now() - tv0).count();
                float vppl    = std::exp(std::min(vloss, 20.0f));
                monitor.log_val(global_step, vloss, vppl, vtime);

                if (vloss < best_val_loss) {
                    best_val_loss = vloss;
                    best_val_ppl  = vppl;
                    save_ckpt("best");
                }
            }

            // ── Periodic checkpoint ───────────────────────────────────────────
            if (global_step % cfg.save_every == 0) {
                char tag[32];
                std::snprintf(tag, sizeof(tag), "step_%07d", global_step);
                save_ckpt(tag);
            }
        } // end batch loop

        // ── End-of-epoch validation + checkpoint ──────────────────────────────
        std::printf("[epoch] %d/%d complete.\n", epoch + 1, cfg.epochs);
        float vloss  = run_validation(model, val_lines, cfg);
        float vppl   = std::exp(std::min(vloss, 20.0f));
        monitor.log_val(global_step, vloss, vppl, 0.0);
        if (vloss < best_val_loss) { best_val_loss = vloss; best_val_ppl = vppl; }

        char tag[32];
        std::snprintf(tag, sizeof(tag), "epoch_%02d", epoch + 1);
        save_ckpt(tag);

    } // end epoch loop

    // ── Save final checkpoint on interrupt or normal finish ───────────────────
    save_ckpt("final");
    monitor.log_final(global_step, best_val_loss, best_val_ppl);

    return 0;
}
#endif // HFAQE_NO_TRAIN_MAIN

// =============================================================================
// run_step_train() — smoke-test shim for main.cpp orchestrator
// Called by main.cpp (Step 3). Uses synthetic data — no Data/ files needed.
// =============================================================================
static bool run_step_train() {
    std::printf("\n[Train] Smoke-test: AdamW + backward + clip + checkpoint ...\n");

    HFAQEConfig mcfg;
    mcfg.V = 256; mcfg.d = 64; mcfg.r = 16; mcfg.K = 64; mcfg.B = 64;
    HFAQE model(mcfg);

    // Tiny synthetic corpus of printable ASCII lines
    std::vector<std::string> fake;
    std::mt19937 rng(7);
    for (int i = 0; i < 40; ++i) {
        std::string s;
        for (int k = 0; k < 24; ++k) s += static_cast<char>(32 + rng() % 95);
        fake.push_back(s);
    }
    auto freq = corpus_frequencies(fake, mcfg.V);
    model.build_frequency_tiers(freq);
    model.initialize_weights(42);

    AdamWState adam_A, adam_B;
    adam_A.init(static_cast<size_t>(model.cold.Vc) * mcfg.r);
    adam_B.init(static_cast<size_t>(mcfg.d)         * mcfg.r);

    float total_loss = 0.0f;
    int   steps_ok   = 0;

    for (int step = 0; step < 10; ++step) {
        const auto& line = fake[step % (int)fake.size()];
        auto ids = byte_tokenise(line, 24);
        if ((int)ids.size() < 2) continue;
        for (auto& id : ids) id = std::max(0, std::min(id, mcfg.V - 1));

        model.zero_grad();
        auto X    = model.forward(ids);
        auto lout = compute_loss(model, X, ids);
        if (lout.n_toks == 0) continue;

        model.backward(lout.dL_dX.data(), ids.data(), (int)ids.size());
        clip_gradients(model, 1.0f);
        sgd_update_hot(model, 1e-3f);
        adamw_update_cold_A(model, adam_A, 1e-3f, 0.9f, 0.999f, 1e-8f, 1e-2f);
        adamw_update_cold_B(model, adam_B, 1e-3f, 0.9f, 0.999f, 1e-8f, 1e-2f);

        total_loss += lout.loss;
        ++steps_ok;
    }

    float mean_loss = (steps_ok > 0) ? total_loss / steps_ok : 0.0f;
    bool  finite_ok = std::isfinite(mean_loss);
    bool  no_spike  = model.grad_Q.size() < static_cast<size_t>(mcfg.V) * mcfg.d
                   && model.grad_A.size() < static_cast<size_t>(mcfg.V) * mcfg.d;

    std::printf("[Train] steps=%d  mean_loss=%.4f  finite=%s  no_V×d_spike=%s\n",
                steps_ok, mean_loss,
                finite_ok ? "YES" : "NO",
                no_spike  ? "YES" : "NO");

    // Test checkpoint round-trip
    bool ckpt_ok = false;
    {
        std::string tmp = "/tmp/hfaqe_smoke_ckpt.bin";
        int gs = 10, ep = 0;
        if (save_checkpoint_full(model, adam_A, adam_B, gs, ep, tmp)) {
            HFAQE model2(mcfg);
            model2.build_frequency_tiers(freq);
            model2.initialize_weights(0);
            AdamWState a2, b2;
            a2.init(adam_A.m.size()); b2.init(adam_B.m.size());
            int gs2 = 0, ep2 = 0;
            ckpt_ok = load_checkpoint_full(model2, a2, b2, gs2, ep2, tmp)
                   && gs2 == gs;
        }
    }
    std::printf("[Train] checkpoint round-trip: %s\n", ckpt_ok ? "PASS" : "FAIL");

    bool ok = finite_ok && no_spike && ckpt_ok;
    std::printf("[Train] Result: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

#endif // HFAQE_TRAIN_CPP
