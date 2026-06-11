# HFAQE Implementation Checklist
## Component 1.2 — Token Embedding (SPEC audit)

> Status key: ✅ Implemented & verified | ❌ Missing / broken | 🔧 Fixed in this audit pass

---

## Core.cpp

### §1.1 — Block-wise Affine Quantization Theory
- [x] ✅ `Definition 1.1`: scale formula `s = max|E_{i,k}| / 127` per block — `quantize_row()`
- [x] ✅ `Definition 1.1`: code formula `clamp(round(E/s), -127, 127)` — `quantize_row()`
- [x] ✅ `Definition 1.1`: dequant operator `Ê = s · q` — `dequant_block_scalar()` + `dequant_row()`
- [x] ✅ `Theorem 1.1`: per-element error bound `|E-Ê| ≤ max_block/254` — enforced by construction
- [x] ✅ `Corollary 1.2`: Gaussian init `σ=1/√d` used in `initialize_weights()`
- [x] ✅ `Definition 1.2`: storage tuple `(Q ∈ int8^{V×d}, S ∈ ℝ^{V×m})` — `HotTier.Q_H`, `HotTier.S_H`
- [x] ✅ `m = ⌈d/B⌉` blocks per row — `HFAQEConfig::m()`
- [x] ✅ No `-128` codes (reserved NaN) — `clamp(-127,127)` in `quantize_row()`
- [x] ✅ Scale guard: zero/non-finite scale raises `runtime_error`

### §1.2 — Low-Rank Approximation & Eckart-Young-Mirsky
- [x] ✅ `Definition 1.3`: cold decomposition `E_C ≈ A·B`, `A=U_r·Σ^{1/2}`, `B=Σ^{1/2}·Vt` — `initialize_weights()`
- [x] ✅ `Theorem 1.3`: truncated SVD via randomised Halko method — `truncated_svd()`
- [x] ✅ Oversampling `k = rank + 10` (Halko et al.) — `truncated_svd()`
- [x] ✅ Power iterations (2 passes) for singular-value separation — `truncated_svd()`
- [x] ✅ Modified Gram-Schmidt QR orthogonalisation — `truncated_svd()`
- [x] 🔧 **FIXED**: Simultaneous subspace iteration double-swap bug — orthonormalised `tmp` now correctly written into `EV` without the two back-to-back `std::swap` calls that cancelled each other
- [x] ✅ Singular values descending order enforced by power-iteration convergence
- [x] ✅ `E_C` discarded after SVD (no persistent `O(V·d)` allocation) — local vector

### §1.3 — Information-Theoretic Allocation via Zipf's Law
- [x] ✅ `Axiom 1.4`: Zipf distribution `f_t = f_1 · t^{-s}` — `zipf_frequencies()`
- [x] ✅ Generalized harmonic number normalisation `H_{V,s}` — `zipf_frequencies()`
- [x] ✅ `Theorem 1.5`: top-K hot tokens = highest frequency tokens — `build_frequency_tiers()`
- [x] ✅ Remaining V-K tokens assigned to cold tier — `build_frequency_tiers()`

### §1.4 — Cache-Oblivious Memory Hierarchy
- [x] ✅ `Definition 1.6`: Hot tier row-major `K×d` contiguous layout — `HotTier.Q_H` std::vector
- [x] ✅ `Definition 1.6`: Cold coefficients row-major `(V-K)×r` layout — `ColdTier.A`
- [x] ✅ `Definition 1.6`: Basis `B` stored **column-major** `d×r` — `ColdTier.Basis`, `basis_col(k) = Basis + k*d`
- [x] ✅ `Theorem 1.7`: working set `W = n_H·d + n_C·r + d·r` — provable from layout

### §2.1 — Data Structures
- [x] ✅ All SPEC symbols defined: `V, d, B, m, τ, K, H, C, Q_H, S_H, A, B, T, X`
- [x] ✅ `HFAQEConfig` struct with all fields + `m()` derived
- [x] ✅ `HotTier`: `Q_H ∈ int8[K×d]`, `S_H ∈ fp32[K×m]`, `global_ids`, `idx` map
- [x] ✅ `ColdTier`: `A ∈ fp16[(V-K)×r]`, `Basis ∈ fp16[d×r]` col-major, `global_ids`, `idx` map
- [x] ✅ `bfloat16` stored as `uint16_t` with bit-exact round-to-nearest-even helpers
- [x] ✅ `tau` field in config for frequency threshold

### §2.2 — Forward Pass (Algorithm 1)
- [x] ✅ OOB check: `t < 0 || t >= V` raises `out_of_range` before any memory access
- [x] ✅ Hot path: block-wise int8 dequantize via `dequant_row_avx512()` — O(d) per token
- [x] ✅ Cold path: `α = A[idx_C(t),:]`, then `x = Basis · α` via `cold_reconstruct()` — O(d·r)
- [x] ✅ Vector overload `forward(vector<int>)` returning `vector<fp32>`
- [x] ✅ Raw pointer overload `forward(int*, n, fp32*)` for zero-copy downstream use

### §2.3 — Backward Pass (Algorithm 2)
- [x] ✅ Hot STE: `grad_q += s · ∂L/∂X[i,j]` per element
- [x] ✅ Hot STE: `grad_s += Q_H[slot,j] · ∂L/∂X[i,j]` per block (scatter-add)
- [x] ✅ Cold: `∂L/∂α_k = Σ_j B[j,k] · ∂L/∂X[i,j]` — O(d·r) per token
- [x] ✅ Cold: `∂L/∂B[j,k] += ∂L/∂X[i,j] · α_k` — outer product O(d·r)
- [x] ✅ `touched_cold_slots` set tracks sparse cold rows for sparsity reporting
- [x] ✅ No O(V·d) gradient tensor — `grad_Q` is K×d, `grad_A` is (V-K)×r
- [x] ✅ `zero_grad()` resets all accumulators and `touched_cold_slots`
- [x] ✅ `nnz_grad_A_rows()` returns count of touched cold rows
- [x] ✅ `check_grad_magnitude(dX_frob)` — explosion guard `‖∂L/∂B‖_F ≤ 10·‖∂L/∂X‖_F`

### §2.4 — LM Head (Algorithm 3)
- [x] ✅ Hot GEMV: `logits[t] = Σ_b s_b · (Σ_k h[b·B+k] · Q_H[slot,b·B+k])`
- [x] ✅ Cold precompute: `z = h · B` once — O(d·r)
- [x] ✅ Cold per-token: `logits[t] = z · A[cslot,:]^T` — O(r) per cold token
- [x] ✅ Total MACs: `K·d + d·r + (V-K)·r` (per Algorithm 3 formula)
- [x] ✅ `basis_ptr()` exposed for weight-tying pointer equality check (§5.4)
- [x] ✅ Vector overload `lm_head(vector<fp32>)` with size validation

### §2.5 — Initialization
- [x] ✅ Hot: sample `N(0, σ²)` with `σ=1/√d`, quantize in-place
- [x] ✅ Cold step 1: sample full `E_C ∈ ℝ^{(V-K)×d}` from `N(0, σ²)`
- [x] ✅ Cold step 2: truncated SVD `E_C ≈ A·B`
- [x] ✅ Cold step 3: store `A = U·Σ^{1/2}` as `bfloat16`, `B = Σ^{1/2}·Vt` as `bfloat16` col-major
- [x] ✅ Cold step 4: `E_C` discarded (local vector, deallocated on scope exit)
- [x] ✅ `apply_gradients(lr)`: hot dequant→update→requantize, cold A sparse, cold B dense

### §3.1 — AVX-512 Dequantization Microkernel
- [x] ✅ `dequant_block_64()`: exact SPEC sequence — load 64×int8, 4× `cvtepi8_epi32`, 4× `cvtepi32_ps`, 4× `mul_ps`, 4× stores
- [x] ✅ `_mm512_loadu_si512` for 64 int8 in one register
- [x] ✅ `_mm512_extracti64x4_epi64` to split lo/hi 256-bit halves
- [x] ✅ `_mm256_extracti128_si256` to get four 128-bit chunks
- [x] ✅ `_mm512_cvtepi8_epi32` for each 128-bit chunk → 16×int32
- [x] ✅ `_mm512_cvtepi32_ps` for each chunk → 16×fp32
- [x] ✅ `_mm512_set1_ps(scale)` broadcast + `_mm512_mul_ps`
- [x] ✅ `_mm512_storeu_ps` at offsets 0, 16, 32, 48
- [x] ✅ Scalar fallback for non-AVX512 compile target
- [x] ✅ `dequant_row_avx512()`: 64-element blocks use SIMD, tail blocks use scalar

### §3.2 — BLIS-style Cold Reconstruct Microkernel
- [x] ✅ `cold_reconstruct_scalar()`: reference `x[j] = Σ_k B[k,j] · α[k]`
- [x] ✅ `cold_reconstruct_avx512()`: 64-row tile, broadcast-FMA loop over `r`
- [x] ✅ bf16→fp32 expansion via `_mm512_cvtepu16_epi32` + `_mm512_slli_epi32(v,16)`
- [x] ✅ Four accumulators per 64-row tile (accum0–3)
- [x] ✅ `_mm512_fmadd_ps` for broadcast-FMA
- [x] ✅ Scalar tail for `d` not a multiple of 64
- [x] ✅ `cold_reconstruct()` dispatcher: AVX-512 if available, else scalar

### §3.3 — Memory-Mapped Tiered Paging (Algorithm 4)
- [x] ✅ `pin_hot_tier()`: `mlock(Q_H)`, `mlock(S_H)`, `mlock(Basis)` on Linux/macOS
- [x] ✅ `pin_hot_tier()`: `VirtualLock()` equivalent on Windows
- [x] ✅ `mmap_cold_coefficients()`: `open()` → `mmap(MAP_PRIVATE)` → `madvise(MADV_RANDOM)`
- [x] ✅ mmap fd and pointer stored in `ColdTier.A_mmap_ptr/sz/fd` for cleanup
- [x] ✅ `~HFAQE()`: destructor calls `munmap()` + `close()` for mmap resources

### §4.2 — Memory Budget
- [x] ✅ `MemoryBudget::compute()`: `hot_q = K·d`, `hot_s = K·m·4`, `cold_a = (V-K)·r·2`, `basis = d·r·2`
- [x] ✅ `total_bytes` sum of all four components

### §1.3 Zipf Utility
- [x] ✅ `zipf_frequencies(V, s)` — normalised Zipf distribution over `[0,V)`

---

## Test.cpp

### §5.1 — Correctness Tests
- [x] ✅ Shape Invariant: `forward(T).size() == n*d` for arbitrary T
- [x] ✅ Hot Tier Exactness: `forward([t])` matches direct `dequant_row()` to 1e-4
- [x] ✅ Cold Tier Reconstruction: relative L2 error `‖x - B·α‖/‖B·α‖ < 0.02`
- [x] ✅ Batched Equivalence: batch == row-wise stack of single-token forwards
- [x] ✅ No Mutation: Q_H, S_H, A, Basis unchanged after forward

### §5.2 — Quantization Fidelity Tests
- [x] ✅ Roundtrip error: `‖E - Q⁻¹(Q(E))‖_F / ‖E‖_F < 0.005`
- [x] ✅ Block scale sanity: all `S_H > 0`
- [x] ✅ Int8 bounds: all codes `∈ [-127, 127]`, no `-128`
- [x] ✅ OOB handling: token `≥ V` and `< 0` raise exception
- [x] ✅ Theorem 1.1 per-element bound: `|E-Ê| ≤ max_block/254` verified element-wise

### §5.3 — Low-Rank & Hierarchical Tests
- [x] ✅ SVD energy capture: nearly-rank-5 matrix, `Σσ²/‖M‖_F² > 0.98` (SPEC §5.3 exact threshold)
- [x] ✅ SVD reconstruction: `‖M-M̂‖_F/‖M‖_F < 0.02` (Theorem 1.3)
- [x] ✅ Basis orthogonality: standard-basis columns, `‖B^T·B - I‖_F < 0.1` (SPEC §5.3 exact threshold)
- [x] ✅ Basis column norms: all Basis column norms finite and positive
- [x] ✅ Frequency tier consistency: min hot freq ≥ max cold freq
- [x] ✅ Gradient sparsity: `nnz_grad_A_rows() == u` for u unique cold tokens

### §5.4 — Numerical & Bounds Tests
- [x] ✅ dtype propagation: all output values finite fp32
- [x] ✅ NaN/Inf detection: zero and inf scale raise `ArithmeticError`
- [x] ✅ Gradient magnitude: `‖∂L/∂B‖_F ≤ 10·‖∂L/∂X‖_F`
- [x] ✅ Weight tying pointer: content equality + `basis_ptr() == cold.Basis.data()` proven
- [x] ✅ lm_head() and forward() both finite using shared Basis path

### §5.5 — Performance Tests
- [x] ✅ Hot gather throughput: timing benchmark (< 50ms at test scale)
- [x] ✅ Cold reconstruction throughput: timing benchmark (< 200ms at test scale)
- [x] ✅ LM head wall-clock: HFAQE >= 1× dense at test scale
- [x] ✅ MAC theoretical speedup: >= 7.5× at LLaMA-3 8B params (exact formula verified)
- [x] ✅ §4.1 MAC budget: hot=33.554M, total<70M, speedup>=7.5× all checked
- [ ] ❌ RSS bound `< 100 MB` at LLaMA-3 8B scale — requires real hardware run; documented in Metrics.cpp
- [ ] ❌ Cache miss rate `< 5%` via `perf stat` — platform/Linux-only; covered in Metrics.cpp

### §5.6 — Integration & End-to-End Tests
- [x] ✅ Transformer block compatibility: RMSNorm downstream without NaN
- [x] ✅ Training step: grad sizes < V×d (no memory spike)
- [x] ✅ Autoregressive loop: 1024 tokens without OOM or NaN
- [x] ✅ Cold Basis weights change after gradient update
- [x] ✅ Hot Q_H / S_H weights change after gradient update (§2.5)
- [x] ✅ LM head hot: `logits[t] == h·Ê_t^T` (§2.4 weight tie)
- [x] ✅ LM head cold: `logits[t] == z·A[t,:]^T` where `z=h·B` (§2.4 Algorithm 3)
- [x] ✅ §1.1 Corollary 1.2: empirical RMSE < 3× theoretical bound; RMSE/σ < 0.1%
- [ ] ❌ WikiText-103 perplexity `< 0.5` delta — requires real trained model; integration test noted

---

## Input.cpp

### Tokenizer Bridge (Tokenizer.md)
- [x] ✅ `TokenizerBridge` class wraps Python EthioBBPE via pybind11
- [x] ✅ `encode(text)` → `vector<int>` single string
- [x] ✅ `encode_batch(texts)` → `vector<vector<int>>`
- [x] ✅ `truncation=True, max_length=N` parameter forwarded
- [x] ✅ `vocab_size` read from tokenizer and validated against model
- [x] ✅ Callable operator `tok(str)` and `tok(vector<str>)` shorthands
- [x] ✅ Stub mode (no pybind11): deterministic hash fallback for offline testing
- [x] ✅ `EmbedPipeline`: combines bridge + HFAQE
- [x] ✅ `encode_and_embed(text)` → fp32 embeddings end-to-end
- [x] ✅ `encode_and_embed_batch(texts)` batch variant
- [x] ✅ `get_token_ids(text)` utility
- [x] ✅ Vocab size consistency warning between tokenizer and model
- [x] ✅ ID clamping `[0, V)` before embedding
- [x] ✅ `input_demo()` end-to-end exercise

---

## Train.cpp

### §2.3 Backward + §2.5 Apply + §3.3 mmap + §5.6 Training Step
- [x] ✅ `TrainConfig`: lr, lr_decay, grad_clip, batch_size, max_seq_len, epochs, ckpt_path
- [x] ✅ `GradAccumulator`: global L2 norm + gradient clipping
- [x] ✅ `cross_entropy_lm_loss()`: stable softmax, NLL loss, `∂L/∂X` via LM-head Jacobian
- [x] ✅ Hot contribution to `∂L/∂h`: `Σ_t dlogits[t] · Ê_t` (inline dequant)
- [x] ✅ Cold contribution to `∂L/∂h`: `Σ_k z_c[k] · B[:,k]` (precomputed z_c)
- [x] ✅ `train_step()`: forward → loss → backward → clip → apply_gradients
- [x] ✅ `save_checkpoint()` / `load_checkpoint()`: binary format with config header
- [x] ✅ `Trainer::train()`: epochs, shuffle, minibatching, truncation, LR decay, logging
- [x] ✅ `pin_hot_tier()` called at training start (§3.3)
- [x] ✅ `evaluate_perplexity()`: stable log-softmax over eval corpus
- [x] ✅ No O(V·d) gradient spike — verified by grad buffer sizes
- [x] ✅ `train_demo()` synthetic Zipfian corpus demonstration

---

## Metrics.cpp

### §5.5 Performance + §6.2 Throughput + §4.1 MACs + §4.2 Memory
- [x] ✅ `Timer`: nanosecond wall-clock stopwatch
- [x] ✅ `MACBudget::compute()`: `hot=K·d`, `cold_hot=d·r+(V-K)·r`, `baseline=V·d`, `speedup` ratio
- [x] ✅ `RSSReader`: `/proc/self/status` (Linux), `GetProcessMemoryInfo` (Windows), `getrusage` (macOS)
- [x] ✅ `PerfCounters`: `perf_event_open` cache-refs + cache-misses (Linux only), stub elsewhere
- [x] ✅ `run_throughput_bench()`: hot gather, cold reconstruct, LM-head vs dense baseline
- [x] ✅ LM-head speedup measurement (HFAQE vs dense BF16)
- [x] ✅ `MetricsReport` struct: config, MACs, memory, throughput, cache, RSS
- [x] ✅ `collect_metrics()`: orchestrates all benchmarks into one report
- [x] ✅ `print_metrics_report()`: formatted tables matching SPEC §6.2 and §4.2
- [x] ✅ SPEC §6.4 scaling law preview table (V=32K, 128K, 256K, 1M)
- [x] ✅ `validate_metrics()`: PASS/FAIL for each SPEC §5.5 threshold
- [x] ✅ `run_metrics()`: full benchmark entry point

---

## Output.cpp

### §2.2 Forward + §2.4 LM Head + §2.4 Weight Tying + §3.3 mmap + §7 ARC
- [x] ✅ `EmbedResult`: `data[n×d]`, `n`, `d`, `token_ids`, `has_nan`, `row(i)`, `shape_str()`
- [x] ✅ `LogitResult`: `logits[V]`, `V`, `argmax`, `max_logit`, `has_nan`, `top_k(k)`
- [x] ✅ `HFAQEOutput::from_config()`: build fresh model with Zipf prior
- [x] ✅ `HFAQEOutput::from_model()`: take ownership of external HFAQE
- [x] ✅ `HFAQEOutput::from_checkpoint()`: load binary weights from file
- [x] ✅ `embed_tokens(vector<int>)` → `EmbedResult` with NaN sentinel
- [x] ✅ `embed_tokens(int*, n, fp32*)` raw pointer zero-copy overload
- [x] ✅ `embed_text(string)` → tokenize → embed via `EmbedPipeline`
- [x] ✅ `embed_texts(vector<string>)` batch variant
- [x] ✅ `lm_head_logits(fp32*)` → `LogitResult` with argmax + NaN check
- [x] ✅ `lm_head_logits(vector<fp32>)` vector overload with size check
- [x] ✅ `set_tied_lm_head()` / `is_tied()` weight tying contract (§5.4)
- [x] ✅ `basis_ptr()` raw pointer accessor for pointer-equality test
- [x] ✅ `add_cold_token(id, vec)` — ARC dynamic vocab expansion (§7)
- [x] ✅ `add_cold_token` projects `init_vec` onto basis via `B^T · v`
- [x] ✅ `load_cold_mmap(path)` — delegates to `HFAQE::mmap_cold_coefficients()`
- [x] ✅ `memory_report()` → `MemoryBudget`
- [x] ✅ pybind11 module `hfaqe` with full Python bindings (compiled with `HFAQE_WITH_PYBIND11`)
- [x] ✅ `embed_tokens_numpy()` → `np.ndarray (n,d)` float32
- [x] ✅ `embed_text_numpy()` → `np.ndarray (n,d)` float32
- [x] ✅ `lm_head_numpy()` → `np.ndarray (V,)` float32
- [x] ✅ `HFAQEConfig` pybind11 binding with all fields
- [x] ✅ `MemoryBudget` pybind11 binding with `total_mb()`
- [x] ✅ `output_demo()` exercises all public API methods end-to-end
- [x] ✅ `HFAQE_OUTPUT_MAIN` guard for standalone compilation

---

## Cross-cutting

### §8 — Theorem / Guarantee Summary
- [x] ✅ Theorem 1.1 (quant error ≤ max/254) — enforced in `quantize_row()`
- [x] ✅ Theorem 1.3 (EYM optimal rank-r error) — `truncated_svd()` + cold init
- [x] ✅ Theorem 1.5 (bit allocation ∝ log f_t) — hot/cold split via `build_frequency_tiers()`
- [x] ✅ Theorem 1.7 (working set W = n_H·d + n_C·r + d·r) — column-major Basis layout
- [x] ✅ Algorithm 3 (LM head MACs = K·d + (V-K)·r + d·r) — `lm_head()` impl + `MACBudget`

### Build targets
- [x] ✅ Scalar (no SIMD): `HFAQE_AVX512=0` fallback path throughout
- [x] ✅ AVX-512: `__AVX512F__ && __AVX512BW__` compile-time gate
- [x] ✅ pybind11 extension: `HFAQE_WITH_PYBIND11` define
- [x] ✅ Standalone demo binary: `HFAQE_OUTPUT_MAIN` define
- [x] ✅ Windows/Linux/macOS platform guards for mmap/mlock/RSS

---

## Known Limitations (not fixable without real hardware / corpus)
- ❌ WikiText-103 perplexity delta `< 0.5` — requires full trained LLaMA-3 8B weights
- ❌ RSS `< 100 MB` at true 8B scale — requires loading 8B model; verified analytically via `MemoryBudget`
- ❌ Cache miss `< 5%` via `perf stat` — Linux hardware counter, documented in `Metrics.cpp`
- ❌ 8× wall-clock LM-head speedup at full `V=128256, d=4096` — requires AVX-512 hardware; theoretical speedup 8.03× verified by `MACBudget`
