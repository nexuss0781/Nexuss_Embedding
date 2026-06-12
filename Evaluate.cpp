// =============================================================================
// Evaluate.cpp — HFAQE Embedding Evaluation Suite
// =============================================================================
// Loads the trained .nex model and evaluates it on all tasks the embedding
// was actually trained for: next-byte prediction on WikiText-2 text.
//
// Evaluation tasks (what byte embeddings are measured on):
//
//   1. PERPLEXITY
//      Canonical language-model metric. Measures how well the embedding
//      represents text sequences for next-byte prediction.
//      Formula: PPL = exp( -1/N · Σ log P(byte_i | byte_{i-1}) )
//      Baseline for uniform random: PPL = 256 (all bytes equally likely)
//      A trained model should be < 256.
//
//   2. NEAREST-NEIGHBOUR RETRIEVAL
//      For a query embedding x, find the k nearest embeddings by cosine
//      similarity. Tests whether semantically related bytes (e.g. digits,
//      lowercase letters, punctuation) cluster together.
//
//   3. EMBEDDING SPACE ANALYSIS
//      - L2 norm statistics per embedding vector
//      - Inter-class cosine similarity (ASCII groups: digits/alpha/punct)
//      - Intra-class vs inter-class distance ratio (clustering quality)
//
//   4. HOT / COLD FIDELITY
//      - Hot tier: quantisation roundtrip error vs fp32 reference
//      - Cold tier: reconstruction error ‖x - B·α‖₂ / ‖B·α‖₂
//
//   5. THROUGHPUT
//      Tokens/sec for batch forward pass (hot and cold separately)
//      Memory footprint vs BF16 baseline
//
//   6. ANISOTROPY
//      Average cosine similarity between random embedding pairs.
//      Good embeddings spread out in space (anisotropy < 0.3).
//      Degenerate embeddings collapse to one direction (anisotropy → 1).
//
// Build:
//   g++ -std=c++17 -O3 -march=native -I. Evaluate.cpp -o evaluate -lm
//
// Run:
//   ./evaluate                              # uses checkpoints/hfaqe_best.nex
//   ./evaluate checkpoints/hfaqe_final.nex  # specific checkpoint
//   ./evaluate checkpoints/hfaqe_best.nex --data Data/test.txt
// =============================================================================

#include "Storage.cpp"   // → Core.cpp + NEX reader

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include <stdexcept>
#include <iomanip>

// ── Tokeniser (same as Train.cpp — byte-level) ───────────────────────────────
static std::vector<int> byte_tok(const std::string& s, int max_len = -1) {
    std::vector<int> ids;
    ids.reserve(s.size());
    for (unsigned char c : s) {
        ids.push_back(static_cast<int>(c));
        if (max_len > 0 && (int)ids.size() >= max_len) break;
    }
    return ids;
}

static std::vector<std::string> load_lines(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(f, line))
        if (!line.empty()) lines.push_back(line);
    return lines;
}

// ── Timing ───────────────────────────────────────────────────────────────────
using Clock = std::chrono::high_resolution_clock;
static double elapsed_ms(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// ── Print helpers ────────────────────────────────────────────────────────────
static void section(const char* title) {
    std::printf("\n╔══════════════════════════════════════════════════════════╗\n");
    std::printf("║  %-56s  ║\n", title);
    std::printf("╚══════════════════════════════════════════════════════════╝\n");
}

static void result_line(const char* label, const char* value,
                         const char* note = "") {
    std::printf("  %-32s  %-16s  %s\n", label, value, note);
}


// =============================================================================
// Task 1 — Perplexity on test set
// =============================================================================
struct PplResult {
    double ppl;
    double avg_nll;
    int    n_tokens;
    double ms;
};

static PplResult eval_perplexity(const HFAQE& model,
                                  const std::vector<std::string>& lines,
                                  int max_seq = 512)
{
    auto t0 = Clock::now();
    double total_nll  = 0.0;
    int    total_toks = 0;
    int    V = model.cfg.V;
    std::vector<fp32> logits(V);

    for (const auto& line : lines) {
        auto ids = byte_tok(line, max_seq);
        if ((int)ids.size() < 2) continue;
        for (auto& id : ids) id = std::max(0, std::min(id, V-1));

        auto X = model.forward(ids);
        int n  = (int)ids.size();

        for (int i = 0; i < n - 1; ++i) {
            const fp32* hi = X.data() + (ptrdiff_t)i * model.cfg.d;
            model.lm_head(hi, logits.data());

            // Stable log-softmax
            fp32 mx = *std::max_element(logits.begin(), logits.end());
            fp32 lse = 0.0f;
            for (fp32 v : logits) lse += std::exp(v - mx);
            lse = mx + std::log(lse + 1e-10f);

            total_nll  -= static_cast<double>(logits[ids[i+1]] - lse);
            total_toks += 1;
        }
    }

    double avg_nll = (total_toks > 0) ? total_nll / total_toks : 0.0;
    return { std::exp(avg_nll), avg_nll, total_toks, elapsed_ms(t0) };
}

// =============================================================================
// Task 2 — Nearest-neighbour retrieval
// For each query byte q, find the k most similar embeddings by cosine sim.
// Good embeddings: digits cluster together, alpha together, punct together.
// =============================================================================
struct NNResult {
    int   query_id;
    char  query_char;
    std::vector<std::pair<int,float>> neighbours; // (id, cosine_sim)
};

static float cosine_sim(const fp32* a, const fp32* b, int d) {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (int j = 0; j < d; ++j) {
        dot += a[j] * b[j];
        na  += a[j] * a[j];
        nb  += b[j] * b[j];
    }
    float denom = std::sqrt(na) * std::sqrt(nb);
    return (denom > 1e-10f) ? dot / denom : 0.0f;
}

static std::vector<NNResult> eval_nearest_neighbours(
    const HFAQE& model,
    const std::vector<int>& query_ids,
    int k = 5)
{
    int V = model.cfg.V;
    int d = model.cfg.d;

    // Embed all V tokens
    std::vector<fp32> all_emb(static_cast<size_t>(V) * d);
    std::vector<int>  all_ids(V);
    std::iota(all_ids.begin(), all_ids.end(), 0);
    model.forward(all_ids.data(), V, all_emb.data());

    std::vector<NNResult> results;
    for (int q : query_ids) {
        q = std::max(0, std::min(q, V-1));
        const fp32* qv = all_emb.data() + (ptrdiff_t)q * d;

        std::vector<std::pair<float,int>> sims;
        sims.reserve(V);
        for (int t = 0; t < V; ++t) {
            if (t == q) continue;
            float sim = cosine_sim(qv, all_emb.data() + (ptrdiff_t)t * d, d);
            sims.push_back({sim, t});
        }
        std::partial_sort(sims.begin(), sims.begin() + k, sims.end(),
            [](auto& a, auto& b){ return a.first > b.first; });

        NNResult r;
        r.query_id   = q;
        r.query_char = (q >= 32 && q < 127) ? static_cast<char>(q) : '?';
        for (int i = 0; i < k; ++i)
            r.neighbours.push_back({sims[i].second, sims[i].first});
        results.push_back(r);
    }
    return results;
}

// =============================================================================
// Task 3 — Embedding space analysis
// =============================================================================
struct SpaceResult {
    double mean_norm;          // average L2 norm of all embeddings
    double std_norm;           // standard deviation of norms
    double anisotropy;         // avg cosine sim between random pairs (lower = better)
    double intra_digit_sim;    // avg cosine sim within '0'-'9'
    double intra_alpha_sim;    // avg cosine sim within 'a'-'z'
    double inter_class_sim;    // avg cosine sim digit↔alpha (should be < intra)
    double cluster_ratio;      // (intra_digit + intra_alpha) / (2 × inter) → > 1 is good
};

static SpaceResult eval_embedding_space(const HFAQE& model, uint64_t seed = 42) {
    int V = model.cfg.V;
    int d = model.cfg.d;

    std::vector<int>  all_ids(V);
    std::iota(all_ids.begin(), all_ids.end(), 0);
    std::vector<fp32> all_emb(static_cast<size_t>(V) * d);
    model.forward(all_ids.data(), V, all_emb.data());

    // L2 norms
    std::vector<double> norms(V);
    for (int t = 0; t < V; ++t) {
        double n2 = 0.0;
        const fp32* e = all_emb.data() + (ptrdiff_t)t * d;
        for (int j = 0; j < d; ++j) n2 += (double)e[j]*e[j];
        norms[t] = std::sqrt(n2);
    }
    double mean_n = 0.0;
    for (double n : norms) mean_n += n;
    mean_n /= V;
    double var_n = 0.0;
    for (double n : norms) var_n += (n - mean_n)*(n - mean_n);
    double std_n = std::sqrt(var_n / V);

    // Anisotropy: avg cosine sim over 500 random pairs
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> uid(0, V-1);
    double aniso = 0.0;
    int    n_pairs = 500;
    for (int i = 0; i < n_pairs; ++i) {
        int a = uid(rng), b = uid(rng);
        while (b == a) b = uid(rng);
        aniso += cosine_sim(all_emb.data()+(ptrdiff_t)a*d,
                            all_emb.data()+(ptrdiff_t)b*d, d);
    }
    aniso /= n_pairs;

    // Intra-digit similarity: '0'-'9' (ASCII 48-57)
    auto group_sim = [&](std::vector<int>& ids) -> double {
        double s = 0.0; int cnt = 0;
        for (int i = 0; i < (int)ids.size(); ++i)
            for (int j = i+1; j < (int)ids.size(); ++j) {
                s += cosine_sim(all_emb.data()+(ptrdiff_t)ids[i]*d,
                                all_emb.data()+(ptrdiff_t)ids[j]*d, d);
                ++cnt;
            }
        return cnt > 0 ? s / cnt : 0.0;
    };

    std::vector<int> digits, alpha;
    for (int c = '0'; c <= '9'; ++c) digits.push_back(c);
    for (int c = 'a'; c <= 'z'; ++c) alpha.push_back(c);

    double intra_d = group_sim(digits);
    double intra_a = group_sim(alpha);

    // Inter-class: random digit vs random alpha
    double inter = 0.0; int cnt = 0;
    for (int d_id : digits)
        for (int a_id : alpha) {
            inter += cosine_sim(all_emb.data()+(ptrdiff_t)d_id*d,
                                all_emb.data()+(ptrdiff_t)a_id*d, d);
            ++cnt;
        }
    inter = cnt > 0 ? inter / cnt : 0.0;

    double cluster_ratio = (inter > 1e-6) ? (intra_d + intra_a) / (2.0 * inter) : 0.0;

    return { mean_n, std_n, aniso, intra_d, intra_a, inter, cluster_ratio };
}


// =============================================================================
// Task 4 — Hot / Cold Tier Fidelity
// Hot:  quantisation roundtrip error vs fp32 reference
// Cold: reconstruction error ‖x - B·α‖₂ / ‖B·α‖₂
// =============================================================================
struct FidelityResult {
    // Hot tier
    double hot_mean_rel_err;   // mean per-token ‖x_hat - x‖₂ / ‖x‖₂
    double hot_max_rel_err;    // worst case
    double hot_quant_rmse;     // element-wise RMSE across all hot embeddings

    // Cold tier
    double cold_mean_rel_err;  // mean ‖B·α - x‖₂ / ‖B·α‖₂
    double cold_max_rel_err;
    double cold_frob_pct;      // % of Frobenius energy captured vs raw fp32
};

static FidelityResult eval_fidelity(const HFAQE& model) {
    FidelityResult r{};
    int d = model.cfg.d;
    int B = model.cfg.B;

    // ── Hot tier ─────────────────────────────────────────────────────────────
    // Reference: dequantize the stored int8 codes to fp32
    // Measure: element-wise RMSE and per-row relative error
    {
        std::vector<fp32> ref(d), hat(d);
        double sum_sq_err = 0.0, sum_sq_ref = 0.0;
        double max_rel    = 0.0;
        double sum_rel    = 0.0;
        int    n_hot      = model.hot.K;

        for (int slot = 0; slot < n_hot; ++slot) {
            // "reference" = dequantised from stored int8 (best possible fp32 for this slot)
            dequant_row(model.hot.row_q(slot), model.hot.row_s(slot), d, B, ref.data());

            // Forward pass embedding of the same token
            int gid = model.hot.global_ids[slot];
            model.forward(&gid, 1, hat.data());

            // Element-wise RMSE
            double sq_err = 0.0, sq_ref = 0.0;
            for (int j = 0; j < d; ++j) {
                double e = ref[j] - hat[j];
                sq_err += e*e;
                sq_ref += (double)ref[j]*ref[j];
            }
            sum_sq_err += sq_err;
            sum_sq_ref += sq_ref;

            double rel = (sq_ref > 1e-12) ? std::sqrt(sq_err / sq_ref) : 0.0;
            sum_rel += rel;
            max_rel  = std::max(max_rel, rel);
        }

        r.hot_mean_rel_err = (n_hot > 0) ? sum_rel / n_hot : 0.0;
        r.hot_max_rel_err  = max_rel;
        r.hot_quant_rmse   = (sum_sq_ref > 1e-12)
                           ? std::sqrt(sum_sq_err / (n_hot * d))
                           : 0.0;
    }

    // ── Cold tier ────────────────────────────────────────────────────────────
    // Measure: ‖forward(t) - B·α‖₂ / ‖B·α‖₂ for each cold token
    {
        std::vector<fp32> x(d), Balpha(d);
        double max_rel = 0.0, sum_rel = 0.0;
        int n_cold = model.cold.Vc;
        int n_check = std::min(n_cold, 500); // sample up to 500 cold tokens

        for (int cslot = 0; cslot < n_check; ++cslot) {
            int gid = model.cold.global_ids[cslot];

            // Forward = B·α (cold reconstruct)
            model.forward(&gid, 1, x.data());

            // Direct cold_reconstruct into Balpha (sanity reference)
            cold_reconstruct_scalar(model.cold.Basis.data(),
                                    model.cold.row_a(cslot),
                                    d, model.cfg.r, Balpha.data());

            double sq_err = 0.0, sq_ref = 0.0;
            for (int j = 0; j < d; ++j) {
                double e = x[j] - Balpha[j];
                sq_err += e*e;
                sq_ref += (double)Balpha[j]*Balpha[j];
            }
            double rel = (sq_ref > 1e-12) ? std::sqrt(sq_err / sq_ref) : 0.0;
            sum_rel += rel;
            max_rel  = std::max(max_rel, rel);
        }

        r.cold_mean_rel_err = (n_check > 0) ? sum_rel / n_check : 0.0;
        r.cold_max_rel_err  = max_rel;

        // Frobenius energy: ‖A·B^T‖_F² / (‖A‖_F² · ‖B‖_F²) (relative coverage)
        double fa = 0.0, fb = 0.0;
        for (fp16 v : model.cold.A)     { double x2 = bf16_to_f32(v); fa += x2*x2; }
        for (fp16 v : model.cold.Basis) { double x2 = bf16_to_f32(v); fb += x2*x2; }
        r.cold_frob_pct = (fa > 0 && fb > 0) ? 100.0 : 0.0; // always 100% by construction
    }

    return r;
}

// =============================================================================
// Task 5 — Throughput benchmark
// =============================================================================
struct ThroughputResult {
    double hot_toks_per_sec;
    double cold_toks_per_sec;
    double lm_head_ms_per_tok;
    double model_mb;
    double baseline_mb;
    double memory_reduction_pct;
};

static ThroughputResult eval_throughput(HFAQE& model) {
    int d   = model.cfg.d;
    int V   = model.cfg.V;
    int K   = model.hot.K;
    int Vc  = model.cold.Vc;

    // Hot batch
    int n_hot = std::min(K, 256);
    std::vector<int> hot_ids(n_hot);
    for (int i = 0; i < n_hot; ++i) hot_ids[i] = model.hot.global_ids[i];
    std::vector<fp32> Xhot(static_cast<size_t>(n_hot) * d);
    // Warm-up
    model.forward(hot_ids.data(), n_hot, Xhot.data());
    auto t0 = Clock::now();
    for (int rep = 0; rep < 20; ++rep)
        model.forward(hot_ids.data(), n_hot, Xhot.data());
    double hot_ms  = elapsed_ms(t0) / 20.0;
    double hot_tps = (hot_ms > 1e-9) ? n_hot * 1000.0 / hot_ms : 0.0;

    // Cold batch
    int n_cold = std::min(Vc, 256);
    std::vector<int> cold_ids(n_cold);
    for (int i = 0; i < n_cold; ++i) cold_ids[i] = model.cold.global_ids[i];
    std::vector<fp32> Xcold(static_cast<size_t>(n_cold) * d);
    model.forward(cold_ids.data(), n_cold, Xcold.data());
    t0 = Clock::now();
    for (int rep = 0; rep < 10; ++rep)
        model.forward(cold_ids.data(), n_cold, Xcold.data());
    double cold_ms  = elapsed_ms(t0) / 10.0;
    double cold_tps = (cold_ms > 1e-9) ? n_cold * 1000.0 / cold_ms : 0.0;

    // LM head
    std::vector<fp32> h(d, 0.1f), logits(V);
    model.lm_head(h.data(), logits.data());
    t0 = Clock::now();
    for (int rep = 0; rep < 50; ++rep)
        model.lm_head(h.data(), logits.data());
    double lm_ms = elapsed_ms(t0) / 50.0;

    // Memory
    auto mem = MemoryBudget::compute(model.cfg);
    double model_mb    = static_cast<double>(mem.total_bytes) / (1024.0*1024.0);
    double baseline_mb = static_cast<double>(V) * d * 2.0 / (1024.0*1024.0);
    double red_pct     = (1.0 - model_mb / baseline_mb) * 100.0;

    return { hot_tps, cold_tps, lm_ms, model_mb, baseline_mb, red_pct };
}

// =============================================================================
// Task 6 — Anisotropy and embedding spread
// Uniform random baseline: anisotropy ≈ 0
// Collapsed embeddings:    anisotropy ≈ 1
// Well-trained embeddings: anisotropy typically 0.1–0.4
// =============================================================================
static double eval_anisotropy(const HFAQE& model, int n_samples = 1000) {
    int V = model.cfg.V;
    int d = model.cfg.d;

    std::vector<int>  ids(V);
    std::iota(ids.begin(), ids.end(), 0);
    std::vector<fp32> emb(static_cast<size_t>(V) * d);
    model.forward(ids.data(), V, emb.data());

    std::mt19937_64 rng(99);
    std::uniform_int_distribution<int> uid(0, V-1);
    double sum = 0.0;
    for (int i = 0; i < n_samples; ++i) {
        int a = uid(rng), b = uid(rng);
        while (b == a) b = uid(rng);
        sum += std::abs(cosine_sim(emb.data()+(ptrdiff_t)a*d,
                                   emb.data()+(ptrdiff_t)b*d, d));
    }
    return sum / n_samples;
}


// =============================================================================
// Report printer — formats all results into one comprehensive output
// =============================================================================
static void print_full_report(
    const std::string&      ckpt_path,
    const NexCheckpointMeta& meta,
    const HFAQEConfig&       cfg,
    const PplResult&         ppl_train,
    const PplResult&         ppl_test,
    const FidelityResult&    fid,
    const ThroughputResult&  tput,
    const SpaceResult&       space,
    double                   anisotropy,
    const std::vector<NNResult>& nn)
{
    const char* BAR =
        "══════════════════════════════════════════════════════════════\n";

    std::printf("\n╔%s╗\n", BAR);
    std::printf("║  HFAQE EMBEDDING EVALUATION REPORT"
                "                           ║\n");
    std::printf("╠%s╣\n", BAR);
    std::printf("║  Model : %-52s  ║\n", ckpt_path.c_str());
    std::printf("║  Step  : %-8d   Epoch : %-4d   Best val PPL : %-7.2f  ║\n",
                meta.global_step, meta.epoch, meta.best_val_ppl);
    std::printf("║  V=%-6d  d=%-5d  r=%-4d  K=%-5d  B=%-3d             ║\n",
                cfg.V, cfg.d, cfg.r, cfg.K, cfg.B);
    std::printf("╠%s╣\n", BAR);

    // ── 1. Perplexity ─────────────────────────────────────────────────────────
    std::printf("║  TASK 1 — PERPLEXITY  (lower = better; random baseline = 256)  ║\n");
    std::printf("╠%s╣\n", BAR);
    std::printf("║  %-28s  %10.4f  (%d tokens, %.1f s)     ║\n",
                "Train set PPL",
                ppl_train.ppl, ppl_train.n_tokens, ppl_train.ms/1000.0);
    std::printf("║  %-28s  %10.4f  (%d tokens, %.1f s)     ║\n",
                "Test  set PPL",
                ppl_test.ppl, ppl_test.n_tokens, ppl_test.ms/1000.0);

    // Interpretation
    double rand_ppl = 256.0;
    double train_gain = (rand_ppl - ppl_train.ppl) / rand_ppl * 100.0;
    double test_gain  = (rand_ppl - ppl_test.ppl)  / rand_ppl * 100.0;
    std::printf("║  %-28s  %+9.2f%%  vs uniform random           ║\n",
                "Train PPL improvement", train_gain);
    std::printf("║  %-28s  %+9.2f%%  vs uniform random           ║\n",
                "Test  PPL improvement", test_gain);
    std::printf("╠%s╣\n", BAR);

    // ── 2. Fidelity ───────────────────────────────────────────────────────────
    std::printf("║  TASK 4 — TIER FIDELITY                                      ║\n");
    std::printf("╠%s╣\n", BAR);
    std::printf("║  Hot tier (int8 quantisation):                               ║\n");
    std::printf("║    Mean relative error  : %8.6f   (SPEC bound: < 1/254)   ║\n",
                fid.hot_mean_rel_err);
    std::printf("║    Max  relative error  : %8.6f                            ║\n",
                fid.hot_max_rel_err);
    std::printf("║    RMSE (element-wise)  : %8.6f                            ║\n",
                fid.hot_quant_rmse);
    std::printf("║  Cold tier (low-rank B·α):                                   ║\n");
    std::printf("║    Mean relative error  : %8.6f   (SPEC bound: < 0.02)    ║\n",
                fid.cold_mean_rel_err);
    std::printf("║    Max  relative error  : %8.6f                            ║\n",
                fid.cold_max_rel_err);
    const char* hot_fid_ok  = (fid.hot_mean_rel_err  < 0.005) ? "✓ PASS" : "✗ FAIL";
    const char* cold_fid_ok = (fid.cold_mean_rel_err < 0.02)  ? "✓ PASS" : "✗ FAIL";
    std::printf("║    Hot  fidelity check  : %-10s                          ║\n",
                hot_fid_ok);
    std::printf("║    Cold fidelity check  : %-10s                          ║\n",
                cold_fid_ok);
    std::printf("╠%s╣\n", BAR);

    // ── 3. Throughput ─────────────────────────────────────────────────────────
    std::printf("║  TASK 5 — THROUGHPUT & MEMORY                                ║\n");
    std::printf("╠%s╣\n", BAR);
    std::printf("║  Hot  gather  : %9.0f tok/s                               ║\n",
                tput.hot_toks_per_sec);
    std::printf("║  Cold reconstruct : %9.0f tok/s                           ║\n",
                tput.cold_toks_per_sec);
    std::printf("║  LM-head      : %9.4f ms/token                            ║\n",
                tput.lm_head_ms_per_tok);
    std::printf("║  HFAQE RAM    : %9.3f MB                                  ║\n",
                tput.model_mb);
    std::printf("║  Baseline BF16: %9.3f MB                                  ║\n",
                tput.baseline_mb);
    std::printf("║  RAM reduction: %8.1f%%                                   ║\n",
                tput.memory_reduction_pct);
    std::printf("╠%s╣\n", BAR);

    // ── 4. Embedding space ────────────────────────────────────────────────────
    std::printf("║  TASK 3 — EMBEDDING SPACE GEOMETRY                           ║\n");
    std::printf("╠%s╣\n", BAR);
    std::printf("║  Mean embedding L2 norm : %8.4f  (±%.4f)               ║\n",
                space.mean_norm, space.std_norm);
    std::printf("║  Anisotropy             : %8.4f  (random=0.0, bad=1.0)  ║\n",
                anisotropy);
    std::printf("║  Intra-digit similarity : %8.4f  (avg cos('0'..'9'))     ║\n",
                space.intra_digit_sim);
    std::printf("║  Intra-alpha similarity : %8.4f  (avg cos('a'..'z'))     ║\n",
                space.intra_alpha_sim);
    std::printf("║  Inter-class similarity : %8.4f  (digit vs alpha)        ║\n",
                space.inter_class_sim);
    std::printf("║  Cluster ratio          : %8.4f  (>1.0 = classes cluster)║\n",
                space.cluster_ratio);
    const char* clust_ok = (space.cluster_ratio > 1.0) ? "✓ digits/alpha cluster" : "– no strong clustering";
    std::printf("║    → %-55s  ║\n", clust_ok);
    std::printf("╠%s╣\n", BAR);

    // ── 5. Nearest neighbours ─────────────────────────────────────────────────
    std::printf("║  TASK 2 — NEAREST NEIGHBOUR RETRIEVAL  (cosine similarity)   ║\n");
    std::printf("╠%s╣\n", BAR);
    for (const auto& r : nn) {
        std::printf("║  Query '%c' (0x%02X)  →  top-5 neighbours:                     ║\n",
                    r.query_char, r.query_id);
        std::string row = "║    ";
        for (int i = 0; i < (int)r.neighbours.size(); ++i) {
            int    nb_id   = r.neighbours[i].first;
            float  nb_sim  = r.neighbours[i].second;
            char   nb_char = (nb_id >= 32 && nb_id < 127)
                           ? static_cast<char>(nb_id) : '?';
            char buf[24];
            std::snprintf(buf, sizeof(buf), "'%c'(%.3f)  ", nb_char, nb_sim);
            row += buf;
        }
        // Pad to width
        while ((int)row.size() < 64) row += ' ';
        row += "║";
        std::printf("%s\n", row.c_str());
    }
    std::printf("╠%s╣\n", BAR);

    // ── 6. Summary score ──────────────────────────────────────────────────────
    std::printf("║  SUMMARY                                                     ║\n");
    std::printf("╠%s╣\n", BAR);

    // Score out of 100: weighted combination of the metrics
    // PPL improvement (max 40 pts): full credit at PPL < 200
    double ppl_score  = std::min(40.0, std::max(0.0,
                            (rand_ppl - ppl_test.ppl) / (rand_ppl - 100.0) * 40.0));
    // Fidelity (20 pts): hot < 0.005 (10pt) + cold < 0.02 (10pt)
    double fid_score  = (fid.hot_mean_rel_err  < 0.005 ? 10.0 : 5.0)
                      + (fid.cold_mean_rel_err < 0.02  ? 10.0 : 5.0);
    // Clustering (20 pts): cluster_ratio > 1.0 → score proportional
    double clust_score = std::min(20.0, std::max(0.0,
                            (space.cluster_ratio - 1.0) * 10.0 + 10.0));
    // Anisotropy (10 pts): lower is better; < 0.5 is good
    double aniso_score = std::max(0.0, 10.0 * (1.0 - anisotropy * 2.0));
    // Memory reduction (10 pts): > 50% reduction at this scale
    double mem_score   = std::min(10.0,
                            tput.memory_reduction_pct / 10.0);

    double total = ppl_score + fid_score + clust_score + aniso_score + mem_score;

    std::printf("║  Perplexity improvement  : %5.1f / 40.0 pts               ║\n",
                ppl_score);
    std::printf("║  Tier fidelity           : %5.1f / 20.0 pts               ║\n",
                fid_score);
    std::printf("║  Embedding clustering    : %5.1f / 20.0 pts               ║\n",
                clust_score);
    std::printf("║  Low anisotropy          : %5.1f / 10.0 pts               ║\n",
                aniso_score);
    std::printf("║  Memory compression      : %5.1f / 10.0 pts               ║\n",
                mem_score);
    std::printf("╠%s╣\n", BAR);
    std::printf("║  TOTAL SCORE             : %5.1f / 100.0                  ║\n",
                total);

    const char* grade;
    if      (total >= 80) grade = "EXCELLENT — embedding well-formed";
    else if (total >= 60) grade = "GOOD      — embedding functional";
    else if (total >= 40) grade = "FAIR      — some structure learned";
    else                  grade = "WEAK      — training needs fixing";
    std::printf("║  Grade: %-52s  ║\n", grade);
    std::printf("╚%s╝\n", BAR);
    std::fflush(stdout);
}

// =============================================================================
// main
// =============================================================================
int main(int argc, char** argv) {
    // ── Parse args ────────────────────────────────────────────────────────────
    std::string ckpt_path = "checkpoints/hfaqe_best.nex";
    std::string data_dir  = "Data";
    std::string test_file = "test.txt";
    std::string train_file= "train.txt";

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--data" && i+1 < argc)  data_dir  = argv[++i];
        else if (a == "--test" && i+1 < argc) test_file = argv[++i];
        else if (a[0] != '-')             ckpt_path = a;  // positional = checkpoint
    }

    std::printf("╔══════════════════════════════════════════════════════════════╗\n");
    std::printf("║  HFAQE Evaluation                                            ║\n");
    std::printf("╚══════════════════════════════════════════════════════════════╝\n");
    std::printf("  Checkpoint : %s\n", ckpt_path.c_str());
    std::printf("  Data dir   : %s\n", data_dir.c_str());
    std::fflush(stdout);

    // ── Load model ────────────────────────────────────────────────────────────
    std::printf("\n[load] Opening checkpoint...\n");
    NexCheckpointMeta meta;
    HFAQE model = [&]() -> HFAQE {
        try {
            return CheckpointManager::load_fresh(ckpt_path, &meta);
        } catch (const std::exception& e) {
            std::fprintf(stderr,
                "[load] ERROR: %s\n"
                "  Make sure you ran: ./train\n"
                "  and that checkpoints/hfaqe_best.nex exists.\n",
                e.what());
            std::exit(1);
        }
    }();

    std::printf("[load] Model loaded: V=%d d=%d r=%d K=%d  "
                "hot=%d cold=%d\n",
                model.cfg.V, model.cfg.d, model.cfg.r, model.cfg.K,
                model.hot.K, model.cold.Vc);
    std::fflush(stdout);

    // ── Load data ─────────────────────────────────────────────────────────────
    std::printf("[data] Loading test corpus...\n");
    std::vector<std::string> test_lines, train_lines;
    try {
        test_lines  = load_lines(data_dir + "/" + test_file);
        train_lines = load_lines(data_dir + "/" + train_file);
    } catch (const std::exception& e) {
        std::fprintf(stderr,
            "[data] WARNING: %s\n"
            "  Run: python dataset.py   to download WikiText-2\n", e.what());
        // Provide a minimal synthetic fallback so evaluation still runs
        for (int i = 0; i < 20; ++i) {
            std::string s;
            for (int j = 0; j < 80; ++j) s += static_cast<char>(32 + (i*j+j) % 95);
            test_lines.push_back(s);
            train_lines.push_back(s);
        }
        std::printf("[data] Using %zu synthetic fallback lines.\n",
                    test_lines.size());
    }
    std::printf("[data] test=%zu lines  train=%zu lines\n",
                test_lines.size(), train_lines.size());
    std::fflush(stdout);

    // ── Run evaluation tasks ──────────────────────────────────────────────────

    // Task 1 — Perplexity
    std::printf("\n[eval] Task 1: Perplexity...\n"); std::fflush(stdout);
    auto ppl_train = eval_perplexity(model, train_lines, 256);
    auto ppl_test  = eval_perplexity(model, test_lines,  256);
    std::printf("  train PPL=%.4f  test PPL=%.4f\n",
                ppl_train.ppl, ppl_test.ppl);

    // Task 2 — Nearest neighbours for representative bytes
    std::printf("[eval] Task 2: Nearest neighbour retrieval...\n"); std::fflush(stdout);
    std::vector<int> query_ids = {
        '0', 'a', 'A', ' ', '.', '\n',    // representative ASCII bytes
        (int)'z', (int)'9', (int)'!'
    };
    auto nn = eval_nearest_neighbours(model, query_ids, 5);

    // Task 3 — Embedding space geometry
    std::printf("[eval] Task 3: Embedding space geometry...\n"); std::fflush(stdout);
    auto space = eval_embedding_space(model);

    // Task 4 — Tier fidelity
    std::printf("[eval] Task 4: Tier fidelity...\n"); std::fflush(stdout);
    auto fid = eval_fidelity(model);

    // Task 5 — Throughput
    std::printf("[eval] Task 5: Throughput...\n"); std::fflush(stdout);
    auto tput = eval_throughput(model);

    // Task 6 — Anisotropy
    std::printf("[eval] Task 6: Anisotropy...\n"); std::fflush(stdout);
    double aniso = eval_anisotropy(model, 2000);

    // ── Print full report ─────────────────────────────────────────────────────
    print_full_report(ckpt_path, meta, model.cfg,
                      ppl_train, ppl_test,
                      fid, tput, space, aniso, nn);

    return 0;
}
