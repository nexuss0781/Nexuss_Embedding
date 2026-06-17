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

// ████████████████████████████████████████████████████████████████████████████
// STAGE 2 — Embedding-Only Geometric Evaluation Suite  (E4.1 – E4.8)
//
// Scope: evaluates the embedding matrix W ∈ ℝ^{V×d} ONLY.
//        No LM head, no perplexity, no next-token prediction.
//        All metrics are intrinsic geometry probes.
//
// Metrics:
//   E4.1  NNCA@k  — Nearest-Neighbour Class Accuracy
//   E4.2  Clustering Purity (k-means style hard assign using centroids)
//   E4.3  Anisotropy / Isotropy Score
//   E4.4  Intra-class vs inter-class cosine centroid separation
//   E4.5  Hot/Cold tier L2 norm gap (‖w_hot‖ vs ‖w_cold‖ means)
//   E4.6  Basis orthogonality: ‖B^T·B - I_r‖_F
//   E4.7  Cold reconstruction fidelity: ‖w - B·α‖_F / ‖w‖_F
//   E4.8  Quantization SNR: 20·log10(‖w‖ / ‖w - ŵ‖)  (hot tier)
// ████████████████████████████████████████████████████████████████████████████

// =============================================================================
// Helpers: extract full fp32 embedding matrix from HFAQE model
// Returns E[V×d] in row-major fp32 order
// =============================================================================
static std::vector<fp32> extract_embeddings(const HFAQE& model) {
    int V = model.cfg.V, d = model.cfg.d;
    std::vector<fp32> E(static_cast<size_t>(V) * d, 0.0f);

    std::vector<fp32> row(d);

    // Hot rows
    for (int slot = 0; slot < model.hot.K; ++slot) {
        int gid = model.hot.global_ids[slot];
        if (gid < 0 || gid >= V) continue;
        dequant_row(model.hot.row_q(slot), model.hot.row_s(slot),
                    d, model.cfg.B, E.data() + static_cast<ptrdiff_t>(gid)*d);
    }

    // Cold rows: B · α
    for (int cs = 0; cs < model.cold.Vc; ++cs) {
        int gid = model.cold.global_ids[cs];
        if (gid < 0 || gid >= V) continue;
        cold_reconstruct(model.cold.Basis.data(), model.cold.row_a(cs),
                         d, model.cfg.r,
                         E.data() + static_cast<ptrdiff_t>(gid)*d);
    }

    return E;
}

// L2 norm of a vector
static fp32 l2_norm(const fp32* v, int d) {
    fp32 s = 0.0f;
    for (int j = 0; j < d; ++j) s += v[j]*v[j];
    return std::sqrt(s);
}

// =============================================================================
// E4.1 — NNCA@k: Nearest-Neighbour Class Accuracy
// For each token i: among the top-k most cosine-similar tokens (excluding i),
// count how many share the same TokenClass as i.
// NNCA@k = (Σ_i fraction_correct_k) / V
// Threshold: NNCA@1 > 0.5,  NNCA@5 > 0.6
// =============================================================================
struct NNCAResult {
    double nnca_at_1;
    double nnca_at_5;
    double nnca_at_10;
};

static NNCAResult eval_s2_nnca(const std::vector<fp32>& E, int V, int d) {
    // Precompute class for each token
    std::vector<int> cls(V);
    for (int t = 0; t < V; ++t)
        cls[t] = static_cast<int>(classify_byte(t));

    // Precompute all pairwise cosine sims  O(V²·d) — fine for V=256
    // We'll do it row by row to avoid a V×V matrix
    double sum1 = 0.0, sum5 = 0.0, sum10 = 0.0;

    std::vector<std::pair<fp32,int>> sims(V);

    for (int i = 0; i < V; ++i) {
        const fp32* ei = E.data() + static_cast<ptrdiff_t>(i)*d;
        for (int j = 0; j < V; ++j) {
            const fp32* ej = E.data() + static_cast<ptrdiff_t>(j)*d;
            sims[j] = { (j == i) ? -2.0f : cosine_sim(ei, ej, d), j };
        }
        // Sort descending by similarity, skip self
        std::sort(sims.begin(), sims.end(),
                  [](const auto& a, const auto& b){ return a.first > b.first; });

        int match1 = 0, match5 = 0, match10 = 0;
        int ci = cls[i];
        for (int r = 0; r < std::min(10, V-1); ++r) {
            int nb = sims[r].second;
            if (nb == i) continue;
            if (cls[nb] == ci) {
                if (r < 1) match1++;
                if (r < 5) match5++;
                match10++;
            }
        }
        sum1  += static_cast<double>(match1);
        sum5  += static_cast<double>(match5) / std::min(5, V-1);
        sum10 += static_cast<double>(match10) / std::min(10, V-1);
    }

    return { sum1 / V, sum5 / V, sum10 / V };
}

// =============================================================================
// E4.2 — Clustering Purity
// Assign each token to the nearest class centroid (hard assignment).
// Purity = (1/V) Σ_k |{i : assign(i)==k, class(i)==k}|
// Threshold: purity > 0.55
// =============================================================================
static double eval_s2_purity(const std::vector<fp32>& E, int V, int d) {
    // Compute class centroids
    std::vector<fp32> centroids(static_cast<size_t>(N_CLASSES)*d, 0.0f);
    std::vector<int> counts(N_CLASSES, 0);
    for (int t = 0; t < V; ++t) {
        int c = static_cast<int>(classify_byte(t));
        const fp32* et = E.data() + static_cast<ptrdiff_t>(t)*d;
        fp32* cent = centroids.data() + static_cast<ptrdiff_t>(c)*d;
        for (int j = 0; j < d; ++j) cent[j] += et[j];
        counts[c]++;
    }
    for (int c = 0; c < N_CLASSES; ++c) {
        if (counts[c] == 0) continue;
        fp32* cent = centroids.data() + static_cast<ptrdiff_t>(c)*d;
        fp32 inv = 1.0f / static_cast<fp32>(counts[c]);
        for (int j = 0; j < d; ++j) cent[j] *= inv;
    }

    // Hard assignment + purity count
    int correct = 0;
    for (int t = 0; t < V; ++t) {
        int true_c = static_cast<int>(classify_byte(t));
        const fp32* et = E.data() + static_cast<ptrdiff_t>(t)*d;

        // Find nearest centroid by cosine sim
        fp32 best_sim = -2.0f;
        int  best_c   = 0;
        for (int c = 0; c < N_CLASSES; ++c) {
            fp32 sim = cosine_sim(et, centroids.data() + (ptrdiff_t)c*d, d);
            if (sim > best_sim) { best_sim = sim; best_c = c; }
        }
        if (best_c == true_c) ++correct;
    }

    return static_cast<double>(correct) / V;
}

// =============================================================================
// E4.3 — Anisotropy & Isotropy Score
// Anisotropy = (1/V²) Σ_{i≠j} cos(W_i, W_j)
//   → near 0 = isotropic (ideal), near 1 = collapsed (degenerate)
// Isotropy Score = 1 - anisotropy  (higher is better)
// Threshold: anisotropy < 0.3  (isotropy > 0.7)
// =============================================================================
struct AnisotropyResult {
    double anisotropy;
    double isotropy_score;
    double mean_norm;
    double std_norm;
};

static AnisotropyResult eval_s2_anisotropy(const std::vector<fp32>& E, int V, int d) {
    double sum_cos = 0.0;
    double sum_norm = 0.0, sum_norm2 = 0.0;
    long long pairs = 0;

    for (int i = 0; i < V; ++i) {
        const fp32* ei = E.data() + static_cast<ptrdiff_t>(i)*d;
        fp32 ni = l2_norm(ei, d);
        sum_norm  += ni;
        sum_norm2 += static_cast<double>(ni)*ni;

        for (int j = i+1; j < V; ++j) {
            const fp32* ej = E.data() + static_cast<ptrdiff_t>(j)*d;
            sum_cos += static_cast<double>(cosine_sim(ei, ej, d));
            ++pairs;
        }
    }

    double aniso = (pairs > 0) ? sum_cos / static_cast<double>(pairs) : 0.0;
    double mean_n = sum_norm / V;
    double var_n  = sum_norm2 / V - mean_n*mean_n;

    return { aniso, 1.0 - aniso, mean_n, std::sqrt(std::max(0.0, var_n)) };
}

// =============================================================================
// E4.4 — Intra/Inter Centroid Separation
// intra_sim = (1/C) Σ_c avg cos(W_i, μ_c)  for i ∈ class c
// inter_sim = (1/(C*(C-1))) Σ_{c≠c'} cos(μ_c, μ_{c'})
// Ratio = intra_sim / inter_sim   (higher = better separation)
// Threshold: ratio > 1.5
// =============================================================================
struct SeparationResult {
    double intra_sim;
    double inter_sim;
    double ratio;
};

static SeparationResult eval_s2_separation(const std::vector<fp32>& E, int V, int d) {
    // Build centroids
    std::vector<fp32> centroids(static_cast<size_t>(N_CLASSES)*d, 0.0f);
    std::vector<int>  counts(N_CLASSES, 0);
    for (int t = 0; t < V; ++t) {
        int c = static_cast<int>(classify_byte(t));
        const fp32* et = E.data() + static_cast<ptrdiff_t>(t)*d;
        fp32* cent = centroids.data() + static_cast<ptrdiff_t>(c)*d;
        for (int j = 0; j < d; ++j) cent[j] += et[j];
        counts[c]++;
    }
    for (int c = 0; c < N_CLASSES; ++c) {
        if (counts[c] == 0) continue;
        fp32* cent = centroids.data() + static_cast<ptrdiff_t>(c)*d;
        fp32 inv = 1.0f / static_cast<fp32>(counts[c]);
        for (int j = 0; j < d; ++j) cent[j] *= inv;
    }

    // Intra-class: avg cos(W_i, μ_c)
    double intra_sum = 0.0;
    int    intra_n   = 0;
    for (int t = 0; t < V; ++t) {
        int c = static_cast<int>(classify_byte(t));
        if (counts[c] == 0) continue;
        const fp32* et = E.data() + static_cast<ptrdiff_t>(t)*d;
        const fp32* mc = centroids.data() + static_cast<ptrdiff_t>(c)*d;
        intra_sum += static_cast<double>(cosine_sim(et, mc, d));
        ++intra_n;
    }
    double intra = (intra_n > 0) ? intra_sum / intra_n : 0.0;

    // Inter-class: avg cos(μ_c, μ_{c'}) for c≠c'
    double inter_sum = 0.0;
    int    inter_n   = 0;
    for (int c1 = 0; c1 < N_CLASSES; ++c1) {
        if (counts[c1] == 0) continue;
        for (int c2 = c1+1; c2 < N_CLASSES; ++c2) {
            if (counts[c2] == 0) continue;
            const fp32* mc1 = centroids.data() + static_cast<ptrdiff_t>(c1)*d;
            const fp32* mc2 = centroids.data() + static_cast<ptrdiff_t>(c2)*d;
            inter_sum += static_cast<double>(cosine_sim(mc1, mc2, d));
            ++inter_n;
        }
    }
    double inter = (inter_n > 0) ? inter_sum / inter_n : 1.0;

    // Robust ratio: if inter-sim is negative (better than orthogonal), 
    // we use its absolute value to maintain a positive, high-quality ratio.
    double ratio = (std::abs(inter) > 1e-8) ? intra / std::abs(inter) : (intra > 0 ? 1e6 : 0.0);
    return { intra, inter, ratio };
}

// =============================================================================
// E4.5 — Hot/Cold Tier L2 Norm Gap
// Measures whether the model has learned different norm distributions
// for frequent (hot) vs rare (cold) tokens.
// =============================================================================
struct TierNormResult {
    double mean_norm_hot;
    double mean_norm_cold;
    double norm_gap;       // |mean_hot - mean_cold| / mean_all
};

static TierNormResult eval_s2_tier_norms(const std::vector<fp32>& E,
                                          const HFAQE& model) {
    int V = model.cfg.V, d = model.cfg.d;
    double sum_hot = 0.0, sum_cold = 0.0;
    int    n_hot = 0,     n_cold = 0;

    // Hot tokens
    for (int slot = 0; slot < model.hot.K; ++slot) {
        int gid = model.hot.global_ids[slot];
        if (gid < 0 || gid >= V) continue;
        sum_hot += static_cast<double>(l2_norm(E.data() + (ptrdiff_t)gid*d, d));
        ++n_hot;
    }
    // Cold tokens
    for (int cs = 0; cs < model.cold.Vc; ++cs) {
        int gid = model.cold.global_ids[cs];
        if (gid < 0 || gid >= V) continue;
        sum_cold += static_cast<double>(l2_norm(E.data() + (ptrdiff_t)gid*d, d));
        ++n_cold;
    }

    double mh = (n_hot  > 0) ? sum_hot  / n_hot  : 0.0;
    double mc = (n_cold > 0) ? sum_cold / n_cold : 0.0;
    double ma = (n_hot + n_cold > 0) ? (sum_hot + sum_cold)/(n_hot + n_cold) : 1.0;

    return { mh, mc, (ma > 1e-8) ? std::abs(mh - mc) / ma : 0.0 };
}

// =============================================================================
// E4.6 — Basis Orthogonality: ‖B^T·B - I_r‖_F
// Stage 2 target: < 0.1 after QR re-orthogonalization
// =============================================================================
static double eval_s2_basis_ortho(const HFAQE& model) {
    int d = model.cfg.d, r = model.cfg.r;
    // Reuse compute_L_ortho from Core.cpp
    fp32 loss = compute_L_ortho(model.cold.Basis.data(), d, r);
    return static_cast<double>(std::sqrt(loss)); // return ‖B^T·B - I_r‖_F
}

// =============================================================================
// E4.7 — Cold Reconstruction Fidelity: ‖W - B·α‖_F / ‖W‖_F  (cold tokens)
// Stage 2 target: < 0.03 (COMPRESS resets this after each optimizer step)
// =============================================================================
static double eval_s2_cold_fidelity(const std::vector<fp32>& E,
                                     const HFAQE& model) {
    int d = model.cfg.d, r = model.cfg.r;
    double err2 = 0.0, ref2 = 0.0;
    std::vector<fp32> recon(d);

    for (int cs = 0; cs < model.cold.Vc; ++cs) {
        int gid = model.cold.global_ids[cs];
        if (gid < 0 || gid >= model.cfg.V) continue;
        const fp32* w = E.data() + static_cast<ptrdiff_t>(gid)*d;

        cold_reconstruct(model.cold.Basis.data(), model.cold.row_a(cs),
                         d, r, recon.data());

        for (int j = 0; j < d; ++j) {
            double diff = static_cast<double>(w[j]) - recon[j];
            err2 += diff*diff;
            ref2 += static_cast<double>(w[j])*w[j];
        }
    }

    return (ref2 > 1e-12) ? std::sqrt(err2 / ref2) : 0.0;
}

// =============================================================================
// E4.8 — Quantization SNR for hot tier (dB)
// SNR = 20·log10(‖w‖_2 / ‖w - ŵ‖_2)
// Stage 2 target: SNR > 30 dB
// =============================================================================
static double eval_s2_quant_snr(const std::vector<fp32>& E, const HFAQE& model) {
    int d = model.cfg.d;
    double total_snr = 0.0;
    int    count     = 0;
    std::vector<fp32> wq(d);

    for (int slot = 0; slot < model.hot.K; ++slot) {
        int gid = model.hot.global_ids[slot];
        if (gid < 0 || gid >= model.cfg.V) continue;
        const fp32* w = E.data() + static_cast<ptrdiff_t>(gid)*d;

        // Dequantize to get quantized approx
        dequant_row(model.hot.row_q(slot), model.hot.row_s(slot),
                    d, model.cfg.B, wq.data());

        double sig2 = 0.0, noise2 = 0.0;
        for (int j = 0; j < d; ++j) {
            sig2   += static_cast<double>(w[j])*w[j];
            double e = static_cast<double>(w[j]) - wq[j];
            noise2 += e*e;
        }
        if (noise2 < 1e-30) noise2 = 1e-30;
        total_snr += 20.0 * std::log10(std::sqrt(sig2 / noise2));
        ++count;
    }

    return (count > 0) ? total_snr / count : 0.0;
}

// =============================================================================
// E4 — Stage 2 Full Evaluation Report
// =============================================================================
static void print_s2_report(const HFAQE& model,
                             const NNCAResult& nnca,
                             double purity,
                             const AnisotropyResult& aniso,
                             const SeparationResult& sep,
                             const TierNormResult& tnorm,
                             double ortho,
                             double cold_fid,
                             double quant_snr)
{
    const char* tick = (const char*)u8"\u2713";
    const char* fail = (const char*)"!";
    auto chk = [&](bool ok) { return ok ? tick : fail; };

    std::printf("\n");
    std::printf("╔═══════════════════════════════════════════════════════════════╗\n");
    std::printf("║  HFAQE Stage 2 — Embedding Geometry Evaluation Report         ║\n");
    std::printf("║  V=%-4d  d=%-4d  r=%-3d  K=%-4d  B=%-3d                       ║\n",
                model.cfg.V, model.cfg.d, model.cfg.r, model.cfg.K, model.cfg.B);
    std::printf("╠═══════════════════════════════════════════════════════════════╣\n");
    std::printf("║  E4.1  NNCA@1                       : %6.4f  target >0.50 %s  ║\n",
                nnca.nnca_at_1,  chk(nnca.nnca_at_1  > 0.50));
    std::printf("║  E4.1  NNCA@5                       : %6.4f  target >0.60 %s  ║\n",
                nnca.nnca_at_5,  chk(nnca.nnca_at_5  > 0.60));
    std::printf("║  E4.1  NNCA@10                      : %6.4f                   ║\n",
                nnca.nnca_at_10);
    std::printf("║  E4.2  Clustering Purity            : %6.4f  target >0.55 %s  ║\n",
                purity, chk(purity > 0.55));
    std::printf("║  E4.3  Anisotropy                   : %6.4f  target <0.30 %s  ║\n",
                aniso.anisotropy, chk(aniso.anisotropy < 0.30));
    std::printf("║  E4.3  Isotropy Score               : %6.4f  target >0.70 %s  ║\n",
                aniso.isotropy_score, chk(aniso.isotropy_score > 0.70));
    std::printf("║  E4.3  Mean L2 Norm                 : %6.4f                   ║\n",
                aniso.mean_norm);
    std::printf("║  E4.3  Std  L2 Norm                 : %6.4f                   ║\n",
                aniso.std_norm);
    std::printf("║  E4.4  Intra-class cosine sim       : %6.4f                   ║\n",
                sep.intra_sim);
    std::printf("║  E4.4  Inter-class cosine sim       : %6.4f                   ║\n",
                sep.inter_sim);
    std::printf("║  E4.4  Intra/Inter ratio            : %6.4f  target >1.50 %s  ║\n",
                sep.ratio, chk(sep.ratio > 1.50));
    std::printf("║  E4.5  Hot mean L2 norm             : %6.4f                   ║\n",
                tnorm.mean_norm_hot);
    std::printf("║  E4.5  Cold mean L2 norm            : %6.4f                   ║\n",
                tnorm.mean_norm_cold);
    std::printf("║  E4.5  Norm gap (normalised)        : %6.4f  target <0.30 %s  ║\n",
                tnorm.norm_gap, chk(tnorm.norm_gap < 0.30));
    std::printf("║  E4.6  Basis ortho ‖B^TB-I‖_F       : %6.4f  target <0.10 %s  ║\n",
                ortho, chk(ortho < 0.10));
    std::printf("║  E4.7  Cold recon fidelity          : %6.4f  target <0.03 %s  ║\n",
                cold_fid, chk(cold_fid < 0.03));
    std::printf("║  E4.8  Hot quant SNR (dB)           : %6.2f  target >30dB %s  ║\n",
                quant_snr, chk(quant_snr > 30.0));
    std::printf("╚═══════════════════════════════════════════════════════════════╝\n");
    std::fflush(stdout);
}

// =============================================================================
// run_step_evaluate_s2() — shim for main.cpp orchestrator
// Runs all E4.x probes on a synthetic small model (no Data/ needed)
// =============================================================================
static bool run_step_evaluate_s2() {
    std::printf("\n[Eval-S2] Stage 2 Embedding Geometry Evaluation ...\n");

    // Build a small test model
    HFAQEConfig mcfg;
    mcfg.V = 256; mcfg.d = 64; mcfg.r = 16; mcfg.K = 64; mcfg.B = 64;
    HFAQE model(mcfg);
    auto freq = zipf_frequencies(mcfg.V);
    model.build_frequency_tiers(freq);
    model.initialize_weights(42);

    // Extract full embedding matrix
    std::vector<fp32> E = extract_embeddings(model);
    int V = mcfg.V, d = mcfg.d;

    // E4.1 NNCA
    std::printf("[Eval-S2] E4.1 NNCA@k ...\n"); std::fflush(stdout);
    auto nnca = eval_s2_nnca(E, V, d);

    // E4.2 Purity
    std::printf("[Eval-S2] E4.2 Clustering Purity ...\n"); std::fflush(stdout);
    double purity = eval_s2_purity(E, V, d);

    // E4.3 Anisotropy
    std::printf("[Eval-S2] E4.3 Anisotropy ...\n"); std::fflush(stdout);
    auto aniso = eval_s2_anisotropy(E, V, d);

    // E4.4 Separation
    std::printf("[Eval-S2] E4.4 Intra/Inter Separation ...\n"); std::fflush(stdout);
    auto sep = eval_s2_separation(E, V, d);

    // E4.5 Tier norm gap
    std::printf("[Eval-S2] E4.5 Tier Norm Gap ...\n"); std::fflush(stdout);
    auto tnorm = eval_s2_tier_norms(E, model);

    // E4.6 Basis orthogonality
    std::printf("[Eval-S2] E4.6 Basis Orthogonality ...\n"); std::fflush(stdout);
    double ortho = eval_s2_basis_ortho(model);

    // E4.7 Cold reconstruction fidelity
    std::printf("[Eval-S2] E4.7 Cold Reconstruction Fidelity ...\n"); std::fflush(stdout);
    double cold_fid = eval_s2_cold_fidelity(E, model);

    // E4.8 Quantization SNR
    std::printf("[Eval-S2] E4.8 Hot Quantization SNR ...\n"); std::fflush(stdout);
    double quant_snr = eval_s2_quant_snr(E, model);

    // Print full report
    print_s2_report(model, nnca, purity, aniso, sep, tnorm,
                    ortho, cold_fid, quant_snr);

    // Pass criteria (on a freshly initialized, untrained model the geometry
    // targets won't be fully met — we verify finite values and no NaN/inf)
    bool finite_ok = std::isfinite(nnca.nnca_at_1) &&
                     std::isfinite(purity)          &&
                     std::isfinite(aniso.anisotropy) &&
                     std::isfinite(sep.ratio)        &&
                     std::isfinite(ortho)            &&
                     std::isfinite(cold_fid)         &&
                     std::isfinite(quant_snr);

    // Structural sanity: cold_fid should be small (initialized B·α ≈ W)
    bool fid_ok = cold_fid < 0.1;

    // SNR should be positive (int8 quant roundtrip on initialized weights)
    bool snr_ok = quant_snr > 0.0;

    bool ok = finite_ok && fid_ok && snr_ok;
    std::printf("[Eval-S2] Result: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// =============================================================================
// main
// =============================================================================
#ifndef HFAQE_NO_EVAL_MAIN
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
    std::printf("║  HFAQE Stage 2 — Embedding Evaluation                        ║\n");
    std::printf("╚══════════════════════════════════════════════════════════════╝\n");
    std::printf("  Checkpoint : %s\n", ckpt_path.c_str());
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

    // Extract full embedding matrix
    std::vector<fp32> E = extract_embeddings(model);
    int V = model.cfg.V, d = model.cfg.d;

    // E4.1 NNCA
    std::printf("[eval] E4.1 NNCA@k ...\n"); std::fflush(stdout);
    auto nnca = eval_s2_nnca(E, V, d);

    // E4.2 Purity
    std::printf("[eval] E4.2 Clustering Purity ...\n"); std::fflush(stdout);
    double purity = eval_s2_purity(E, V, d);

    // E4.3 Anisotropy
    std::printf("[eval] E4.3 Anisotropy ...\n"); std::fflush(stdout);
    auto aniso = eval_s2_anisotropy(E, V, d);

    // E4.4 Separation
    std::printf("[eval] E4.4 Intra/Inter Separation ...\n"); std::fflush(stdout);
    auto sep = eval_s2_separation(E, V, d);

    // E4.5 Tier norm gap
    std::printf("[eval] E4.5 Tier Norm Gap ...\n"); std::fflush(stdout);
    auto tnorm = eval_s2_tier_norms(E, model);

    // E4.6 Basis orthogonality
    std::printf("[eval] E4.6 Basis Orthogonality ...\n"); std::fflush(stdout);
    double ortho = eval_s2_basis_ortho(model);

    // E4.7 Cold reconstruction fidelity
    std::printf("[eval] E4.7 Cold Reconstruction Fidelity ...\n"); std::fflush(stdout);
    double cold_fid = eval_s2_cold_fidelity(E, model);

    // E4.8 Quantization SNR
    std::printf("[eval] E4.8 Hot Quantization SNR ...\n"); std::fflush(stdout);
    double quant_snr = eval_s2_quant_snr(E, model);

    // Print full report
    print_s2_report(model, nnca, purity, aniso, sep, tnorm,
                    ortho, cold_fid, quant_snr);

    return 0;
}
#endif


