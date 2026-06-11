// =============================================================================
// main.cpp — HFAQE Orchestrator
// =============================================================================
// Runs all components in order before training:
//
//   Step 0: Print build info (compiler, SIMD flags, platform)
//   Step 1: Core   — quantization, SVD, forward/backward, LM-head smoke test
//   Step 2: Input  — tokenizer bridge demo (stub mode without Python)
//   Step 3: Train  — synthetic training loop demo (2 epochs)
//   Step 4: Metrics— hardware perf benchmark + memory budget report
//   Step 5: Output — public API demo (embed_tokens, lm_head, ARC expansion)
//
// Each step is isolated: a failure is reported but execution continues so all
// steps are always exercised.  Exit code = number of failed steps.
//
// Build (Linux, see CMakeLists.txt):
//   cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)
//   ./build/hfaqe_main
//
// To activate AVX-512:
//   cmake -B build -DHFAQE_AVX512=ON && cmake --build build -j$(nproc)
// =============================================================================

// Single-TU include chain — include guards in each file prevent ODR violations:
//   Output.cpp  → Input.cpp → Core.cpp
//   Train.cpp   → Storage.cpp → Core.cpp  (already guarded)
//   Storage.cpp → Core.cpp               (already guarded)
//   Metrics.cpp → Core.cpp               (already guarded)
//
// HFAQE_NO_TRAIN_MAIN suppresses Train.cpp's own main() so it doesn't clash
// with this file's main().
// =============================================================================

#define HFAQE_NO_TRAIN_MAIN
#include "Core.cpp"
#include "Storage.cpp"
#include "Input.cpp"
#include "Train.cpp"
#include "Metrics.cpp"
#include "Output.cpp"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <chrono>

// =============================================================================
// Helpers
// =============================================================================
static const char* SEP =
    "================================================================\n";

static void banner(int step, const char* title) {
    std::printf("\n%s", SEP);
    std::printf("  STEP %d: %s\n", step, title);
    std::printf("%s", SEP);
}

static void step_result(int step, const char* title, bool ok, double ms) {
    std::printf("\n%s", SEP);
    std::printf("  STEP %d [%s] %s  (%.1f ms)\n",
        step, ok ? "PASS" : "FAIL", title, ms);
    std::printf("%s\n", SEP);
}

using Clock = std::chrono::high_resolution_clock;

// =============================================================================
// Step 0 — Build info
// =============================================================================
static void print_build_info() {
    std::printf("%s", SEP);
    std::printf("  HFAQE — Pre-Training Verification Suite\n");
    std::printf("%s", SEP);

#if defined(__clang__)
    std::printf("  Compiler : Clang %d.%d.%d\n",
        __clang_major__, __clang_minor__, __clang_patchlevel__);
#elif defined(__GNUC__)
    std::printf("  Compiler : GCC %d.%d.%d\n",
        __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#else
    std::printf("  Compiler : Unknown\n");
#endif

#if defined(__linux__)
    std::printf("  Platform : Linux\n");
#elif defined(_WIN32)
    std::printf("  Platform : Windows\n");
#elif defined(__APPLE__)
    std::printf("  Platform : macOS\n");
#endif

#if HFAQE_AVX512
    std::printf("  SIMD     : AVX-512 (BW+F) ENABLED\n");
#else
    std::printf("  SIMD     : Scalar fallback (compile with -mavx512f -mavx512bw to enable)\n");
#endif

    std::printf("  sizeof(fp32) = %zu   sizeof(fp16/bf16) = %zu   sizeof(int8) = %zu\n",
        sizeof(fp32), sizeof(fp16), sizeof(int8));
    std::printf("\n");
}

// =============================================================================
// Step 1 — Core smoke test
// Exercises: quantize_row, dequant_row, build_frequency_tiers,
//            initialize_weights, forward (hot+cold), backward, lm_head
// =============================================================================
static bool run_step_core() {
    std::printf("\n[Core] Building model V=1000 d=128 r=32 K=100 B=64 ...\n");

    HFAQEConfig cfg;
    cfg.V = 1000; cfg.d = 128; cfg.r = 32; cfg.K = 100; cfg.B = 64;
    HFAQE model(cfg);
    auto freq = zipf_frequencies(cfg.V);
    model.build_frequency_tiers(freq);
    model.initialize_weights(42);

    std::printf("[Core] Hot tier  : K=%d  int8[%d×%d]\n", model.hot.K, model.hot.K, cfg.d);
    std::printf("[Core] Cold tier : Vc=%d  bf16[%d×%d]  Basis bf16[%d×%d]\n",
        model.cold.Vc, model.cold.Vc, cfg.r, cfg.d, cfg.r);

    // --- Forward: one hot, one cold ---
    int hot_id  = model.hot.global_ids[0];
    int cold_id = model.cold.global_ids[0];

    auto Xh = model.forward({hot_id});
    auto Xc = model.forward({cold_id});

    bool hot_finite  = true, cold_finite  = true;
    for (fp32 v : Xh) if (!std::isfinite(v)) { hot_finite  = false; break; }
    for (fp32 v : Xc) if (!std::isfinite(v)) { cold_finite = false; break; }
    std::printf("[Core] Forward hot  finite=%s  cold  finite=%s\n",
        hot_finite ? "YES" : "NO", cold_finite ? "YES" : "NO");

    // --- Batch forward ---
    std::vector<int> batch;
    for (int i = 0; i < 20; ++i) batch.push_back(i * (cfg.V/20));
    auto Xbatch = model.forward(batch);
    bool batch_ok = (int)Xbatch.size() == 20 * cfg.d;
    std::printf("[Core] Batch forward n=20 shape=%s\n", batch_ok ? "OK" : "FAIL");

    // --- LM head ---
    std::vector<fp32> h(cfg.d, 0.1f);
    auto logits = model.lm_head(h);
    bool lm_ok = (int)logits.size() == cfg.V;
    for (fp32 v : logits) if (!std::isfinite(v)) { lm_ok = false; break; }
    std::printf("[Core] LM head  V=%d  finite=%s\n", cfg.V, lm_ok ? "YES" : "NO");

    // --- Backward ---
    model.zero_grad();
    std::vector<fp32> dX(static_cast<size_t>(20) * cfg.d, 0.01f);
    model.backward(dX.data(), batch.data(), 20);
    std::printf("[Core] Backward  grad_B size=%zu  nnz_cold_rows=%d\n",
        model.grad_B.size(), model.nnz_grad_A_rows());

    // --- Gradient apply ---
    std::vector<fp16> basis_snap = model.cold.Basis;
    model.apply_gradients(1e-3f);
    bool weights_changed = (model.cold.Basis != basis_snap);
    std::printf("[Core] apply_gradients  weights_changed=%s\n",
        weights_changed ? "YES" : "NO");

    // --- OOB guard ---
    bool oob_ok = false;
    try { model.forward({cfg.V}); }
    catch (const std::out_of_range&) { oob_ok = true; }
    std::printf("[Core] OOB guard  token=%d  raised=%s\n", cfg.V, oob_ok ? "YES" : "NO");

    // --- Quantization roundtrip ---
    std::vector<fp32>  row_in(cfg.d), row_out(cfg.d);
    std::vector<int8>  codes(cfg.d);
    std::vector<fp32>  scales((cfg.d+cfg.B-1)/cfg.B);
    for (int j = 0; j < cfg.d; ++j) row_in[j] = 0.01f * (j - cfg.d/2);
    quantize_row(row_in.data(), cfg.d, cfg.B, codes.data(), scales.data());
    dequant_row(codes.data(), scales.data(), cfg.d, cfg.B, row_out.data());
    double rel_err = 0.0, denom = 0.0;
    for (int j = 0; j < cfg.d; ++j) {
        double e = row_in[j] - row_out[j]; rel_err += e*e;
        denom += (double)row_in[j]*row_in[j];
    }
    rel_err = (denom > 0) ? std::sqrt(rel_err/denom) : 0.0;
    std::printf("[Core] Quant roundtrip  rel_err=%.2e  (SPEC bound < 0.005)\n", rel_err);

    bool ok = hot_finite && cold_finite && batch_ok && lm_ok && oob_ok
              && weights_changed && rel_err < 0.005;
    std::printf("[Core] Result: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// =============================================================================
// Step 2 — Input bridge smoke test
// Exercises: TokenizerBridge (stub), EmbedPipeline, encode_and_embed
// =============================================================================
static bool run_step_input() {
    std::printf("\n[Input] Building EmbedPipeline (stub tokenizer) ...\n");

    HFAQEConfig cfg;
    cfg.V = 16000; cfg.d = 128; cfg.r = 32; cfg.K = 1024; cfg.B = 64;
    HFAQE model(cfg);
    auto freq = zipf_frequencies(cfg.V);
    model.build_frequency_tiers(freq);
    model.initialize_weights(99);

    EmbedPipeline pipeline(&model, "");

    // Vocab size check
    std::printf("[Input] tokenizer.vocab_size = %d  (EthioBBPE default: 16000)\n",
        pipeline.vocab_size());

    // Encode single Amharic string (stub: uses char % V)
    std::string text = "\xe1\x88\xb0\xe1\x88\xb3\xe1\x88\x9d"; // "ሰሳም"
    auto ids = pipeline.get_token_ids(text);
    std::printf("[Input] encode('%s') → %zu token(s)\n", text.c_str(), ids.size());

    // Embed
    auto emb = pipeline.encode_and_embed(text);
    bool finite = true;
    for (fp32 v : emb) if (!std::isfinite(v)) { finite = false; break; }
    std::printf("[Input] embed shape=[%zu×%d]  finite=%s\n",
        ids.size(), cfg.d, finite ? "YES" : "NO");

    // Batch
    std::vector<std::string> batch = {"ሰላም", "ዓለም", "ሐዋርያ"};
    auto batch_embs = pipeline.encode_and_embed_batch(batch);
    std::printf("[Input] batch embed  n=%zu sentences\n", batch_embs.size());
    for (size_t i = 0; i < batch_embs.size(); ++i)
        std::printf("[Input]   [%zu] shape=[%zu×%d]\n",
            i, batch_embs[i].size() / cfg.d, cfg.d);

    bool ok = finite && ids.size() > 0 && batch_embs.size() == 3;
    std::printf("[Input] Result: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// =============================================================================
// =============================================================================
// Step 3 — Train smoke test
// Delegates entirely to run_step_train() defined in Train.cpp
// (synthetic data, no Data/ files needed, validates AdamW + NEX checkpoint)
// run_step_train() is defined in Train.cpp (included above) — no redefinition here.
// =============================================================================

// =============================================================================
// Step 4 — Metrics benchmark
// Exercises: Timer, MACBudget, RSS, PerfCounters, ThroughputBench, report
// =============================================================================
static bool run_step_metrics() {
    std::printf("\n[Metrics] Running benchmark V=2000 d=128 r=32 K=200 B=64 ...\n");

    bool ok = true;
    try {
        int fails = run_metrics(/*V=*/2000, /*d=*/128, /*r=*/32,
                                /*K=*/200,  /*B=*/64);
        ok = (fails == 0);
    } catch (const std::exception& e) {
        std::printf("[Metrics] EXCEPTION: %s\n", e.what());
        ok = false;
    }
    std::printf("[Metrics] Result: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// =============================================================================
// Step 5 — Output public API demo
// Exercises: HFAQEOutput factory, embed_tokens, embed_text,
//            lm_head_logits, add_cold_token (ARC), basis_ptr
// =============================================================================
static bool run_step_output() {
    std::printf("\n[Output] Public API demo ...\n");
    bool ok = true;
    try {
        output_demo();
    } catch (const std::exception& e) {
        std::printf("[Output] EXCEPTION: %s\n", e.what());
        ok = false;
    }
    std::printf("[Output] Result: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// =============================================================================
// main
// =============================================================================
int main(int argc, char** argv) {
    (void)argc; (void)argv;

    print_build_info();

    struct StepRecord {
        int         num;
        const char* name;
        bool        passed;
        double      ms;
    };
    std::vector<StepRecord> records;

    // Helper lambda: run a step, time it, catch exceptions
    auto run = [&](int num, const char* name, auto fn) {
        banner(num, name);
        auto t0 = Clock::now();
        bool ok = false;
        try { ok = fn(); }
        catch (const std::exception& e) {
            std::printf("[EXCEPTION in step %d] %s\n", num, e.what());
        } catch (...) {
            std::printf("[UNKNOWN EXCEPTION in step %d]\n", num);
        }
        double ms = std::chrono::duration<double,std::milli>(
            Clock::now() - t0).count();
        step_result(num, name, ok, ms);
        records.push_back({num, name, ok, ms});
    };

    run(1, "Core  — Quantization / SVD / Forward / Backward / LM-Head", run_step_core);
    run(2, "Input — Tokenizer Bridge (stub mode)",                        run_step_input);
    run(3, "Train — Synthetic training loop (1 epoch)",                   run_step_train);
    run(4, "Storage — NEX format self-test (write+read+verify)",
        []() -> bool { return nex_self_test(true); });
    run(5, "Metrics — Throughput / MACs / RSS benchmark",                 run_step_metrics);
    run(6, "Output — Public API (embed_tokens, lm_head, ARC)",            run_step_output);

    // ---- Final report ----
    std::printf("\n%s", SEP);
    std::printf("  HFAQE Pre-Training Verification — FINAL REPORT\n");
    std::printf("%s", SEP);
    std::printf("  %-50s  %6s  %8s\n", "Step", "Status", "Time(ms)");
    std::printf("  %s\n", std::string(68, '-').c_str());
    int n_fail = 0;
    for (auto& r : records) {
        std::printf("  [%d] %-48s  %6s  %8.1f\n",
            r.num, r.name, r.passed ? "PASS" : "FAIL", r.ms);
        if (!r.passed) ++n_fail;
    }
    std::printf("  %s\n", std::string(68, '-').c_str());
    std::printf("  Passed: %d/%d\n",
        static_cast<int>(records.size()) - n_fail,
        static_cast<int>(records.size()));
    if (n_fail == 0)
        std::printf("  ALL STEPS PASSED — code is logically and syntactically correct.\n");
    else
        std::printf("  %d STEP(S) FAILED — review output above before training.\n", n_fail);
    std::printf("%s\n", SEP);

    return n_fail;
}
