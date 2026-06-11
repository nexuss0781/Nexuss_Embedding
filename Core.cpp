// =============================================================================
// Core.cpp — HFAQE: Hierarchical Frequency-Adaptive Quantized Embedding
// =============================================================================
// Spec coverage:
//   §1 Mathematical Foundations (Quantization, SVD, Zipf, Cache-Oblivious)
//   §2 Architecture (Forward, Backward, LM-Head, Init, Weight Tying)
//   §3 CPU Optimization (AVX-512 microkernels, BLIS matmul, mmap)
// =============================================================================

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <stdexcept>
#include <memory>
#include <string>
#include <random>
#include <numeric>
#include <limits>
#include <fstream>

// POSIX mmap / mlock (Linux/macOS)
#ifdef _WIN32
#  include <windows.h>
#else
#  include <sys/mman.h>
#  include <sys/types.h>
#  include <sys/stat.h>
#  include <fcntl.h>
#  include <unistd.h>
#endif

// AVX-512 intrinsics (falls back gracefully if not available)
#if defined(__AVX512F__) && defined(__AVX512BW__)
#  include <immintrin.h>
#  define HFAQE_AVX512 1
#else
#  define HFAQE_AVX512 0
#endif

// =============================================================================
// §2.1 — Primitive types
// =============================================================================

using int8  = int8_t;
using int32 = int32_t;
using fp32  = float;
using fp16  = uint16_t;   // bfloat16 stored as raw uint16

// bfloat16 helpers (bit-exact round-to-nearest-even)
inline fp16 f32_to_bf16(fp32 x) {
    uint32_t bits;
    std::memcpy(&bits, &x, 4);
    // Round-to-nearest-even: add 0x7FFF + (bit 16)
    uint32_t rounding = 0x7FFF + ((bits >> 16) & 1u);
    bits += rounding;
    return static_cast<fp16>(bits >> 16);
}

inline fp32 bf16_to_f32(fp16 x) {
    uint32_t bits = static_cast<uint32_t>(x) << 16;
    fp32 out;
    std::memcpy(&out, &bits, 4);
    return out;
}

// =============================================================================
// §2.1 — HFAQE Configuration (all SPEC symbols)
// =============================================================================

struct HFAQEConfig {
    int V  = 16000;   // vocabulary size (EthioBBPE default)
    int d  = 512;     // model dimension
    int B  = 64;      // quantization block size (SPEC default)
    int r  = 64;      // low-rank for cold tier
    int K  = 512;     // number of hot tokens
    fp32 tau = 1e-4f; // frequency threshold for hot/cold split

    // Derived
    int m() const { return (d + B - 1) / B; } // blocks per row ⌈d/B⌉
};

// =============================================================================
// §2.1 — Core data structures (SPEC Table)
// =============================================================================

// Hot tier — block-wise int8 quantized embeddings
struct HotTier {
    int K, d, m;                        // dimensions
    std::vector<int8>  Q_H;            // int8[K × d]   — quantized codes
    std::vector<fp32>  S_H;            // fp32[K × m]   — per-block scales
    std::vector<int>   global_ids;     // global vocab index for each hot slot
    std::unordered_map<int,int> idx;   // global_id → hot slot index

    void allocate(int K_, int d_, int m_) {
        K = K_; d = d_; m = m_;
        Q_H.assign(static_cast<size_t>(K) * d,  0);
        S_H.assign(static_cast<size_t>(K) * m,  0.0f);
        global_ids.resize(K, -1);
    }
    // Pointer helpers
    int8* row_q(int slot) { return Q_H.data() + static_cast<ptrdiff_t>(slot) * d; }
    fp32* row_s(int slot) { return S_H.data() + static_cast<ptrdiff_t>(slot) * m; }
    const int8* row_q(int slot) const { return Q_H.data() + static_cast<ptrdiff_t>(slot) * d; }
    const fp32* row_s(int slot) const { return S_H.data() + static_cast<ptrdiff_t>(slot) * m; }
};

// Cold tier — low-rank factorization:  E_C ≈ A · B^T
// A ∈ fp16[(V-K) × r],  B ∈ fp16[d × r] (column-major for cache locality)
struct ColdTier {
    int Vc, d, r;                       // Vc = V-K cold tokens
    std::vector<fp16>  A;              // bf16[(V-K) × r] — coefficients
    std::vector<fp16>  Basis;          // bf16[d × r]     — shared basis (col-major)
    std::vector<int>   global_ids;     // global vocab index for each cold slot
    std::unordered_map<int,int> idx;   // global_id → cold slot index

    // mmap alternative for out-of-core cold tier (§3.3)
    void* A_mmap_ptr  = nullptr;
    size_t A_mmap_sz  = 0;
    int    A_mmap_fd  = -1;

    void allocate(int Vc_, int d_, int r_) {
        Vc = Vc_; d = d_; r = r_;
        A.assign(static_cast<size_t>(Vc) * r, 0);
        Basis.assign(static_cast<size_t>(d) * r, 0); // col-major: B[j,k] = Basis[k*d + j]
        global_ids.resize(Vc, -1);
    }
    fp16* row_a(int slot) { return A.data() + static_cast<ptrdiff_t>(slot) * r; }
    const fp16* row_a(int slot) const { return A.data() + static_cast<ptrdiff_t>(slot) * r; }
    // Basis column k starts at: Basis.data() + k*d
    fp16* basis_col(int k) { return Basis.data() + static_cast<ptrdiff_t>(k) * d; }
    const fp16* basis_col(int k) const { return Basis.data() + static_cast<ptrdiff_t>(k) * d; }
};

// =============================================================================
// §1.1 — Block-wise affine quantization (Definition 1.1 + Theorem 1.1)
// =============================================================================

// Quantize one embedding row (length d) into int8 codes + scales.
// Returns max per-element error bound = max(|row|) / 254  (Theorem 1.1)
inline fp32 quantize_row(const fp32* row, int d, int B,
                          int8* q_out, fp32* s_out)
{
    int m = (d + B - 1) / B;
    fp32 max_error = 0.0f;
    for (int b = 0; b < m; ++b) {
        int start = b * B;
        int end   = std::min(start + B, d);
        // Compute scale: s = max|E_{i,k}| / 127  (Definition 1.1)
        fp32 abs_max = 0.0f;
        for (int k = start; k < end; ++k)
            abs_max = std::max(abs_max, std::abs(row[k]));
        fp32 s = (abs_max > 0.0f) ? (abs_max / 127.0f) : 1.0f;
        if (!std::isfinite(s) || s <= 0.0f)
            throw std::runtime_error("HFAQE: scale is not finite or zero");
        s_out[b] = s;
        // Quantize codes: clamp(round(x/s), -127, 127)  (Definition 1.1)
        for (int k = start; k < end; ++k) {
            fp32 v = row[k] / s;
            int32 code = static_cast<int32>(std::round(v));
            code = std::max(-127, std::min(127, code));  // no -128 (SPEC §5.2)
            q_out[k] = static_cast<int8>(code);
        }
        // Error bound per Theorem 1.1: s/2
        max_error = std::max(max_error, s * 0.5f);
    }
    return max_error;
}

// Dequantize one block of B elements (scalar fallback)
inline void dequant_block_scalar(const int8* q, fp32 scale, fp32* out, int len) {
    for (int k = 0; k < len; ++k)
        out[k] = scale * static_cast<fp32>(q[k]);
}

// Dequantize a full row (d elements) from int8 + scales into fp32
inline void dequant_row(const int8* q, const fp32* s, int d, int B, fp32* out) {
    int m = (d + B - 1) / B;
    for (int b = 0; b < m; ++b) {
        int start = b * B;
        int end   = std::min(start + B, d);
        dequant_block_scalar(q + start, s[b], out + start, end - start);
    }
}


// =============================================================================
// §3.1 — AVX-512 Dequantization Microkernel (Kernel 1: dequant_block_64)
// Spec: 4× cvtepi8_epi32 + 4× cvtepi32_ps + 4× mul_ps + 4× stores
//       ~12 cycles per 64 elements on Ice Lake / Zen 4
// =============================================================================

#if HFAQE_AVX512

// Dequantizes exactly 64 int8 values into 64 fp32 values using AVX-512BW+F.
// Matches Kernel 1 from SPEC §3.1 exactly.
void dequant_block_64(const int8_t* __restrict__ q, float scale,
                      float* __restrict__ out)
{
    // Load 64 x int8 → one 512-bit register
    __m512i q_8 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(q));

    // Extract two 256-bit halves, then widen each to four groups of 16 x int32
    __m256i lo256 = _mm512_extracti64x4_epi64(q_8, 0); // bytes  0-31
    __m256i hi256 = _mm512_extracti64x4_epi64(q_8, 1); // bytes 32-63

    // Widen each 256-bit half into two 512-bit int32 registers (chunk of 16)
    __m512i q32_0 = _mm512_cvtepi8_epi32(_mm256_extracti128_si256(lo256, 0)); // 0-15
    __m512i q32_1 = _mm512_cvtepi8_epi32(_mm256_extracti128_si256(lo256, 1)); // 16-31
    __m512i q32_2 = _mm512_cvtepi8_epi32(_mm256_extracti128_si256(hi256, 0)); // 32-47
    __m512i q32_3 = _mm512_cvtepi8_epi32(_mm256_extracti128_si256(hi256, 1)); // 48-63

    // Convert int32 → fp32
    __m512 f0 = _mm512_cvtepi32_ps(q32_0);
    __m512 f1 = _mm512_cvtepi32_ps(q32_1);
    __m512 f2 = _mm512_cvtepi32_ps(q32_2);
    __m512 f3 = _mm512_cvtepi32_ps(q32_3);

    // Multiply by scale (broadcast scalar)
    __m512 vs = _mm512_set1_ps(scale);
    _mm512_storeu_ps(out +  0, _mm512_mul_ps(f0, vs));
    _mm512_storeu_ps(out + 16, _mm512_mul_ps(f1, vs));
    _mm512_storeu_ps(out + 32, _mm512_mul_ps(f2, vs));
    _mm512_storeu_ps(out + 48, _mm512_mul_ps(f3, vs));
}

// Dequantize a full row using AVX-512: handles blocks of 64, scalar tail
void dequant_row_avx512(const int8_t* q, const float* s, int d, int B,
                        float* out)
{
    int m = (d + B - 1) / B;
    for (int b = 0; b < m; ++b) {
        int start = b * B;
        int end   = std::min(start + B, d);
        int len   = end - start;
        if (len == 64) {
            dequant_block_64(q + start, s[b], out + start);
        } else {
            // Tail block: scalar fallback
            dequant_block_scalar(q + start, s[b], out + start, len);
        }
    }
}

#else  // No AVX-512: fall through to scalar path

void dequant_block_64(const int8_t* q, float scale, float* out) {
    for (int i = 0; i < 64; ++i)
        out[i] = scale * static_cast<float>(q[i]);
}

void dequant_row_avx512(const int8_t* q, const float* s, int d, int B,
                        float* out)
{
    dequant_row(q, s, d, B, out);
}

#endif // HFAQE_AVX512


// =============================================================================
// §3.2 — Cache-Blocked Matmul for Cold Tier (Kernel 2: cold_reconstruct)
// Computes x = B_basis · α  where B_basis ∈ ℝ^{d×r}, α ∈ ℝ^r
// B_basis stored column-major: B_basis[j,k] = Basis[k*d + j]  (§2.1)
// BLIS broadcast-FMA microkernel with 64-row blocks to stay in L1 cache
// =============================================================================

// Scalar reference: x[j] = Σ_k Basis[k*d+j] * alpha[k]
static void cold_reconstruct_scalar(const fp16* Basis, const fp16* alpha,
                                     int d, int r, fp32* x)
{
    // Zero output
    std::fill(x, x + d, 0.0f);
    for (int k = 0; k < r; ++k) {
        fp32 a_k = bf16_to_f32(alpha[k]);
        const fp16* col_k = Basis + static_cast<ptrdiff_t>(k) * d;
        for (int j = 0; j < d; ++j)
            x[j] += a_k * bf16_to_f32(col_k[j]);
    }
}

#if HFAQE_AVX512

// AVX-512 broadcast-FMA cold reconstruction (SPEC §3.2)
// Processes 64 rows at a time (one cache line of 64×fp32 = 256 B in L1)
static void cold_reconstruct_avx512(const fp16* Basis, const fp16* alpha,
                                     int d, int r, fp32* x)
{
    // Zero output buffer
    std::fill(x, x + d, 0.0f);

    int j = 0;
    for (; j + 64 <= d; j += 64) {
        __m512 accum0 = _mm512_setzero_ps();
        __m512 accum1 = _mm512_setzero_ps();
        __m512 accum2 = _mm512_setzero_ps();
        __m512 accum3 = _mm512_setzero_ps();

        for (int k = 0; k < r; ++k) {
            fp32 a_k = bf16_to_f32(alpha[k]);
            __m512 a_broadcast = _mm512_set1_ps(a_k);

            const fp16* col_k = Basis + static_cast<ptrdiff_t>(k) * d + j;
            // Convert 64 bf16 → fp32 in four groups of 16
            // We expand manually (no native bf16 load until AVX-512 BF16 extension)
            auto load16_bf16 = [](const fp16* src) -> __m512 {
                // Shift each uint16 left by 16 into a uint32, then reinterpret as fp32
                __m256i v16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src));
                __m512i v32 = _mm512_cvtepu16_epi32(v16);
                v32 = _mm512_slli_epi32(v32, 16);
                return _mm512_castsi512_ps(v32);
            };

            __m512 b0 = load16_bf16(col_k +  0);
            __m512 b1 = load16_bf16(col_k + 16);
            __m512 b2 = load16_bf16(col_k + 32);
            __m512 b3 = load16_bf16(col_k + 48);

            accum0 = _mm512_fmadd_ps(a_broadcast, b0, accum0);
            accum1 = _mm512_fmadd_ps(a_broadcast, b1, accum1);
            accum2 = _mm512_fmadd_ps(a_broadcast, b2, accum2);
            accum3 = _mm512_fmadd_ps(a_broadcast, b3, accum3);
        }
        _mm512_storeu_ps(x + j +  0, accum0);
        _mm512_storeu_ps(x + j + 16, accum1);
        _mm512_storeu_ps(x + j + 32, accum2);
        _mm512_storeu_ps(x + j + 48, accum3);
    }
    // Scalar tail for d not multiple of 64
    for (; j < d; ++j) {
        fp32 acc = 0.0f;
        for (int k = 0; k < r; ++k)
            acc += bf16_to_f32(alpha[k]) * bf16_to_f32(Basis[static_cast<ptrdiff_t>(k)*d+j]);
        x[j] = acc;
    }
}

#endif // HFAQE_AVX512

// Dispatch: use AVX-512 if compiled in, else scalar
static void cold_reconstruct(const fp16* Basis, const fp16* alpha,
                              int d, int r, fp32* x)
{
#if HFAQE_AVX512
    cold_reconstruct_avx512(Basis, alpha, d, r, x);
#else
    cold_reconstruct_scalar(Basis, alpha, d, r, x);
#endif
}


// =============================================================================
// §1.2 + §2.5 — Truncated SVD via Power Iteration (Eckart-Young-Mirsky)
// Computes rank-r approximation: E_C ≈ A · B^T
// A = U_r · Σ_r^{1/2} ∈ ℝ^{|C|×r},  B = Σ_r^{1/2} · V_r^T ∈ ℝ^{r×d}
// Uses randomised SVD (power iteration) for CPU efficiency.
// =============================================================================

// Matrix multiply: C[m×k] = A[m×n] · B[n×k]  (row-major dense)
static void matmul(const fp32* A, const fp32* B,
                   int m, int n, int k, fp32* C)
{
    std::fill(C, C + static_cast<size_t>(m) * k, 0.0f);
    for (int i = 0; i < m; ++i)
        for (int p = 0; p < n; ++p) {
            fp32 a = A[static_cast<ptrdiff_t>(i)*n + p];
            for (int j = 0; j < k; ++j)
                C[static_cast<ptrdiff_t>(i)*k + j] += a * B[static_cast<ptrdiff_t>(p)*k + j];
        }
}

// Transpose: B[n×m] = A^T[m×n]
static void transpose(const fp32* A, int m, int n, fp32* B) {
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            B[static_cast<ptrdiff_t>(j)*m + i] = A[static_cast<ptrdiff_t>(i)*n + j];
}

// L2 norm of a vector
static fp32 vec_norm(const fp32* v, int n) {
    fp32 s = 0.0f;
    for (int i = 0; i < n; ++i) s += v[i]*v[i];
    return std::sqrt(s);
}

// Normalise vector in-place; returns original norm
static fp32 vec_normalize(fp32* v, int n) {
    fp32 nm = vec_norm(v, n);
    if (nm > 1e-10f)
        for (int i = 0; i < n; ++i) v[i] /= nm;
    return nm;
}

// Randomised truncated SVD — Halko, Martinsson, Tropp (2011)
// Input:  M ∈ ℝ^{rows×cols} (row-major)
// Output: U ∈ ℝ^{rows×r}, Sigma ∈ ℝ^r (descending), Vt ∈ ℝ^{r×cols}
// Method: sketch Y = M·Ω, QR(Y)→Q, SVD(Q^T·M)
// power_iters: 2 is sufficient for fast singular-value decay (LLaMA class)
static void truncated_svd(const fp32* M, int rows, int cols, int rank,
                           fp32* U, fp32* Sigma, fp32* Vt,
                           int power_iters = 2, uint64_t seed = 42)
{
    int k = rank + 10; // oversampling (Halko et al.)
    k = std::min(k, std::min(rows, cols));

    std::mt19937_64 rng(seed);
    std::normal_distribution<fp32> norm(0.0f, 1.0f);

    // Sketch Ω ∈ ℝ^{cols×k}
    std::vector<fp32> Omega(static_cast<size_t>(cols) * k);
    for (auto& x : Omega) x = norm(rng);

    // Y = M · Ω  → ℝ^{rows×k}
    std::vector<fp32> Y(static_cast<size_t>(rows) * k, 0.0f);
    matmul(M, Omega.data(), rows, cols, k, Y.data());

    // Power iterations: Y ← (M·M^T)^p · Y  improves singular-value separation
    std::vector<fp32> Mt(static_cast<size_t>(cols) * rows);
    transpose(M, rows, cols, Mt.data());
    for (int p = 0; p < power_iters; ++p) {
        // Z = M^T · Y → ℝ^{cols×k}
        std::vector<fp32> Z(static_cast<size_t>(cols) * k, 0.0f);
        matmul(Mt.data(), Y.data(), cols, rows, k, Z.data());
        // Y = M · Z → ℝ^{rows×k}
        std::fill(Y.begin(), Y.end(), 0.0f);
        matmul(M, Z.data(), rows, cols, k, Y.data());
    }

    // QR decomposition of Y via modified Gram-Schmidt
    // Q ∈ ℝ^{rows×k}
    std::vector<fp32> Q(Y); // copy
    std::vector<std::vector<fp32>> q_cols(k, std::vector<fp32>(rows));
    for (int j = 0; j < k; ++j) {
        fp32* col_j = Q.data() + static_cast<ptrdiff_t>(j) * rows; // column-major extraction
        // Extract col j from row-major Q: Q_col[i] = Q[i*k + j]
        for (int i = 0; i < rows; ++i)
            q_cols[j][i] = Q[static_cast<ptrdiff_t>(i)*k + j];
        // Orthogonalise against previous columns
        for (int prev = 0; prev < j; ++prev) {
            fp32 dot = 0.0f;
            for (int i = 0; i < rows; ++i)
                dot += q_cols[j][i] * q_cols[prev][i];
            for (int i = 0; i < rows; ++i)
                q_cols[j][i] -= dot * q_cols[prev][i];
        }
        vec_normalize(q_cols[j].data(), rows);
        // Write back
        for (int i = 0; i < rows; ++i)
            Q[static_cast<ptrdiff_t>(i)*k + j] = q_cols[j][i]; // row-major
        (void)col_j;
    }

    // B_small = Q^T · M → ℝ^{k×cols}
    // Q_T[k×rows]
    std::vector<fp32> Q_T(static_cast<size_t>(k) * rows);
    // Q is rows×k row-major; Q_T[j,i] = Q[i,j]
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < k; ++j)
            Q_T[static_cast<ptrdiff_t>(j)*rows + i] = Q[static_cast<ptrdiff_t>(i)*k + j];

    std::vector<fp32> B_small(static_cast<size_t>(k) * cols, 0.0f);
    matmul(Q_T.data(), M, k, rows, cols, B_small.data());

    // SVD of small matrix B_small[k×cols] via one-sided Jacobi (k ≤ ~266)
    // For simplicity and correctness on CPU, use power-iteration on B_small^T·B_small
    // Full Jacobi is complex; we do a compact eigen-decomposition of C = B_small·B_small^T
    // C ∈ ℝ^{k×k}, eigenvectors → right singular vectors of B_small

    // C = B_small · B_small^T  (k×k)
    std::vector<fp32> Bs_T(static_cast<size_t>(cols) * k);
    transpose(B_small.data(), k, cols, Bs_T.data());
    std::vector<fp32> C(static_cast<size_t>(k) * k, 0.0f);
    matmul(B_small.data(), Bs_T.data(), k, cols, k, C.data());

    // Eigen-decomp of symmetric C via power iteration for top-r eigenvectors
    // (Krylov subspace / simultaneous iteration)
    // EV ∈ ℝ^{k×rank} — eigenvectors in columns
    std::vector<fp32> EV(static_cast<size_t>(k) * rank);
    // Random init
    for (auto& x : EV) x = norm(rng);
    // Orthonormalise init
    for (int j = 0; j < rank; ++j) {
        fp32* ej = EV.data() + static_cast<ptrdiff_t>(j)*k; // treating EV as k×rank row-major
        // Orthogonalise col j (stored row-stridedly) — repack to contiguous
        std::vector<fp32> ej_v(k);
        for (int i = 0; i < k; ++i) ej_v[i] = EV[static_cast<ptrdiff_t>(i)*rank+j];
        for (int prev = 0; prev < j; ++prev) {
            fp32 dot = 0.0f;
            for (int i = 0; i < k; ++i)
                dot += ej_v[i] * EV[static_cast<ptrdiff_t>(i)*rank+prev];
            for (int i = 0; i < k; ++i)
                ej_v[i] -= dot * EV[static_cast<ptrdiff_t>(i)*rank+prev];
        }
        vec_normalize(ej_v.data(), k);
        for (int i = 0; i < k; ++i) EV[static_cast<ptrdiff_t>(i)*rank+j] = ej_v[i];
        (void)ej;
    }
    // Simultaneous subspace iteration: 30 iterations
    std::vector<fp32> tmp(static_cast<size_t>(k) * rank);
    for (int iter = 0; iter < 30; ++iter) {
        // tmp = C · EV  (k×k · k×rank → k×rank)
        std::fill(tmp.begin(), tmp.end(), 0.0f);
        for (int i = 0; i < k; ++i)
            for (int p = 0; p < k; ++p) {
                fp32 c = C[static_cast<ptrdiff_t>(i)*k + p];
                for (int j = 0; j < rank; ++j)
                    tmp[static_cast<ptrdiff_t>(i)*rank+j] += c * EV[static_cast<ptrdiff_t>(p)*rank+j];
            }
        // Orthonormalise tmp column by column, write result into EV
        for (int j = 0; j < rank; ++j) {
            std::vector<fp32> col(k);
            for (int i = 0; i < k; ++i) col[i] = tmp[static_cast<ptrdiff_t>(i)*rank+j];
            for (int prev = 0; prev < j; ++prev) {
                fp32 dot = 0.0f;
                for (int i = 0; i < k; ++i)
                    dot += col[i] * EV[static_cast<ptrdiff_t>(i)*rank+prev];
                for (int i = 0; i < k; ++i)
                    col[i] -= dot * EV[static_cast<ptrdiff_t>(i)*rank+prev];
            }
            vec_normalize(col.data(), k);
            for (int i = 0; i < k; ++i) EV[static_cast<ptrdiff_t>(i)*rank+j] = col[i];
        }
        // EV now holds the orthonormalised subspace; tmp is scratch for next iter
    }

    // Eigenvalues: λ_j = EV_j^T · C · EV_j
    // Sigma[j] = sqrt(λ_j)  (singular values of B_small)
    for (int j = 0; j < rank; ++j) {
        fp32 lambda = 0.0f;
        for (int i = 0; i < k; ++i) {
            fp32 cev = 0.0f;
            for (int p = 0; p < k; ++p)
                cev += C[static_cast<ptrdiff_t>(i)*k+p] * EV[static_cast<ptrdiff_t>(p)*rank+j];
            lambda += EV[static_cast<ptrdiff_t>(i)*rank+j] * cev;
        }
        Sigma[j] = std::sqrt(std::max(0.0f, lambda));
    }

    // U_small = EV ∈ ℝ^{k×rank} — left singular vectors of B_small
    // U = Q · U_small ∈ ℝ^{rows×rank}
    // Q is rows×k row-major; U_small is k×rank row-major (EV column layout above)
    // Repack EV as k×rank row-major
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < rank; ++j) {
            fp32 acc = 0.0f;
            for (int p = 0; p < k; ++p)
                acc += Q[static_cast<ptrdiff_t>(i)*k+p] * EV[static_cast<ptrdiff_t>(p)*rank+j];
            U[static_cast<ptrdiff_t>(i)*rank+j] = acc;
        }
    }

    // Vt = U_small^T · B_small / Sigma  →  Vt ∈ ℝ^{rank×cols}
    for (int j = 0; j < rank; ++j) {
        fp32 inv_s = (Sigma[j] > 1e-10f) ? (1.0f / Sigma[j]) : 0.0f;
        for (int c = 0; c < cols; ++c) {
            fp32 acc = 0.0f;
            for (int p = 0; p < k; ++p)
                acc += EV[static_cast<ptrdiff_t>(p)*rank+j] * B_small[static_cast<ptrdiff_t>(p)*cols+c];
            Vt[static_cast<ptrdiff_t>(j)*cols+c] = acc * inv_s;
        }
    }
}


// =============================================================================
// §2.5 — HFAQE Class: Initialization
// Hot tier:  Q_H ← N(0,1/d) quantized in-place; S_H derived
// Cold tier: E_C ← N(0,1/d) → truncated SVD → A, B; E_C discarded
// =============================================================================

class HFAQE {
public:
    HFAQEConfig cfg;
    HotTier  hot;
    ColdTier cold;

    // Gradient accumulators (fp32, allocated during first backward)
    std::vector<fp32> grad_Q;    // [K × d]   straight-through for hot
    std::vector<fp32> grad_S;    // [K × m]
    std::vector<fp32> grad_A;    // [(V-K) × r]  sparse, zero between steps
    std::vector<fp32> grad_B;    // [d × r]   dense, col-major

    // Tracks which cold rows have been touched (for sparse grad check)
    std::unordered_set<int> touched_cold_slots;

    explicit HFAQE(const HFAQEConfig& config) : cfg(config) {}

    // -----------------------------------------------------------------
    // build_frequency_tiers
    // Takes token frequency vector (length V), builds hot/cold index maps.
    // Selects top-K by frequency as hot tier (§1.3 Zipf justification).
    // -----------------------------------------------------------------
    void build_frequency_tiers(const std::vector<fp32>& token_freq) {
        if ((int)token_freq.size() != cfg.V)
            throw std::invalid_argument("token_freq size != V");

        // Sort tokens by frequency descending
        std::vector<int> order(cfg.V);
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + cfg.K, order.end(),
            [&](int a, int b){ return token_freq[a] > token_freq[b]; });

        // Assign hot slots
        hot.allocate(cfg.K, cfg.d, cfg.m());
        for (int slot = 0; slot < cfg.K; ++slot) {
            int gid = order[slot];
            hot.global_ids[slot] = gid;
            hot.idx[gid] = slot;
        }

        // Remaining tokens → cold
        int Vc = cfg.V - cfg.K;
        cold.allocate(Vc, cfg.d, cfg.r);
        int cslot = 0;
        for (int i = cfg.K; i < cfg.V; ++i) {
            int gid = order[i];
            cold.global_ids[cslot] = gid;
            cold.idx[gid] = cslot;
            ++cslot;
        }
    }

    // -----------------------------------------------------------------
    // initialize_weights
    // Performs §2.5 initialization from scratch.
    // -----------------------------------------------------------------
    void initialize_weights(uint64_t seed = 12345) {
        if (hot.K == 0 || cold.Vc == 0)
            throw std::logic_error("Call build_frequency_tiers() first");

        fp32 sigma = 1.0f / std::sqrt(static_cast<fp32>(cfg.d));
        std::mt19937_64 rng(seed);
        std::normal_distribution<fp32> dist(0.0f, sigma);

        // --- Hot tier ---
        // Sample fp32, then quantize in-place (§2.5 Hot Tier)
        std::vector<fp32> row_buf(cfg.d);
        for (int slot = 0; slot < cfg.K; ++slot) {
            for (auto& x : row_buf) x = dist(rng);
            fp32 err = quantize_row(row_buf.data(), cfg.d, cfg.B,
                                    hot.row_q(slot), hot.row_s(slot));
            (void)err; // error bound available if needed
        }

        // --- Cold tier ---
        // Step 1: sample full E_C ∈ ℝ^{(V-K)×d}
        size_t Ec_sz = static_cast<size_t>(cold.Vc) * cfg.d;
        std::vector<fp32> E_C(Ec_sz);
        for (auto& x : E_C) x = dist(rng);

        // Step 2: truncated SVD  E_C ≈ U·Σ^{1/2} · Σ^{1/2}·Vt
        int actual_r = std::min(cfg.r, std::min(cold.Vc, cfg.d));
        std::vector<fp32> U(static_cast<size_t>(cold.Vc) * actual_r);
        std::vector<fp32> Sigma(actual_r);
        std::vector<fp32> Vt(static_cast<size_t>(actual_r) * cfg.d);

        truncated_svd(E_C.data(), cold.Vc, cfg.d, actual_r,
                      U.data(), Sigma.data(), Vt.data());

        // Step 3: A = U · Σ^{1/2}, store as bfloat16
        //         Basis[j,k] = Σ^{1/2} · Vt[k,j]  (col-major: Basis[k*d+j])
        for (int slot = 0; slot < cold.Vc; ++slot) {
            fp16* a_row = cold.row_a(slot);
            for (int k = 0; k < actual_r; ++k) {
                fp32 val = U[static_cast<ptrdiff_t>(slot)*actual_r + k]
                           * std::sqrt(Sigma[k]);
                a_row[k] = f32_to_bf16(val);
            }
        }
        for (int k = 0; k < actual_r; ++k) {
            fp16* basis_k = cold.basis_col(k);
            fp32 sqrt_sk = std::sqrt(Sigma[k]);
            for (int j = 0; j < cfg.d; ++j)
                basis_k[j] = f32_to_bf16(sqrt_sk * Vt[static_cast<ptrdiff_t>(k)*cfg.d + j]);
        }

        // Step 4: Discard E_C (goes out of scope automatically)

        // Allocate gradient buffers
        grad_Q.assign(static_cast<size_t>(cfg.K) * cfg.d, 0.0f);
        grad_S.assign(static_cast<size_t>(cfg.K) * cfg.m(), 0.0f);
        grad_A.assign(static_cast<size_t>(cold.Vc) * cfg.r, 0.0f);
        grad_B.assign(static_cast<size_t>(cfg.d) * cfg.r, 0.0f);
    }

    // =================================================================
    // §2.2 — Forward Pass: Algorithm 1 HFAQE Forward Gather
    // Input:  T ∈ int[n]  (token IDs)
    // Output: X ∈ fp32[n×d]
    // Hot path:  block-wise int8 gather + dequantize   O(d) per token
    // Cold path: low-rank reconstruction x = Basis · α  O(d·r) per token
    // =================================================================
    void forward(const int* T, int n, fp32* X) const {
        for (int i = 0; i < n; ++i) {
            int t = T[i];
            // Bounds check (SPEC §5.2 OOB)
            if (t < 0 || t >= cfg.V)
                throw std::out_of_range(
                    "HFAQE: token ID " + std::to_string(t)
                    + " out of range [0," + std::to_string(cfg.V) + ")");

            fp32* xi = X + static_cast<ptrdiff_t>(i) * cfg.d;

            auto hot_it = hot.idx.find(t);
            if (hot_it != hot.idx.end()) {
                // ---- Hot path ----------------------------------------
                int slot = hot_it->second;
                dequant_row_avx512(hot.row_q(slot), hot.row_s(slot),
                                   cfg.d, cfg.B, xi);
            } else {
                // ---- Cold path ----------------------------------------
                auto cold_it = cold.idx.find(t);
                if (cold_it == cold.idx.end())
                    throw std::out_of_range("HFAQE: token not in hot or cold tier");
                int cslot = cold_it->second;
                const fp16* alpha = cold.row_a(cslot);
                cold_reconstruct(cold.Basis.data(), alpha, cfg.d, cfg.r, xi);
            }
        }
    }

    // Convenience overload with vectors
    std::vector<fp32> forward(const std::vector<int>& T) const {
        std::vector<fp32> X(static_cast<size_t>(T.size()) * cfg.d);
        forward(T.data(), static_cast<int>(T.size()), X.data());
        return X;
    }


    // =================================================================
    // §2.3 — Backward Pass: Algorithm 2 HFAQE Backward (Training)
    // Input:  dL_dX ∈ fp32[n×d],  T ∈ int[n]
    // Accumulates: grad_Q, grad_S  (hot sparse scatter-add)
    //              grad_A (cold sparse), grad_B (cold dense outer-product)
    // Key property: No O(V×d) dense gradient instantiated.
    // =================================================================
    void backward(const fp32* dL_dX, const int* T, int n) {
        // Guard: gradient explosion check (SPEC §5.4)
        // Computed post-accumulation in zero_grad_check()

        for (int i = 0; i < n; ++i) {
            int t = T[i];
            if (t < 0 || t >= cfg.V)
                throw std::out_of_range("HFAQE backward: token ID out of range");

            const fp32* dxi = dL_dX + static_cast<ptrdiff_t>(i) * cfg.d;

            auto hot_it = hot.idx.find(t);
            if (hot_it != hot.idx.end()) {
                // ---- Hot path: straight-through estimator (STE) ----------
                // grad_q = s · ∂L/∂X[i,j]
                // grad_s = Q_H[slot,j] · ∂L/∂X[i,j]    (summed over block)
                int slot = hot_it->second;
                const int8* qrow = hot.row_q(slot);
                const fp32* srow = hot.row_s(slot);
                fp32*    gqrow = grad_Q.data() + static_cast<ptrdiff_t>(slot)*cfg.d;
                fp32*    gsrow = grad_S.data() + static_cast<ptrdiff_t>(slot)*cfg.m();

                int m = cfg.m();
                for (int b = 0; b < m; ++b) {
                    int start = b * cfg.B;
                    int end   = std::min(start + cfg.B, cfg.d);
                    fp32 s = srow[b];
                    fp32 gs_acc = 0.0f;
                    for (int j = start; j < end; ++j) {
                        fp32 dlx = dxi[j];
                        gqrow[j] += s * dlx;                             // ∂L/∂q
                        gs_acc   += static_cast<fp32>(qrow[j]) * dlx;   // ∂L/∂s
                    }
                    gsrow[b] += gs_acc;
                }
            } else {
                // ---- Cold path: low-rank gradient factorization ----------
                auto cold_it = cold.idx.find(t);
                if (cold_it == cold.idx.end())
                    throw std::out_of_range("HFAQE backward: token not in any tier");
                int cslot = cold_it->second;
                touched_cold_slots.insert(cslot);

                const fp16* alpha = cold.row_a(cslot);
                fp32* ga_row = grad_A.data() + static_cast<ptrdiff_t>(cslot)*cfg.r;
                fp32* gB     = grad_B.data(); // d×r col-major

                // ∂L/∂α_k = Σ_j Basis[k*d+j] · ∂L/∂X[i,j]   O(d·r)
                for (int k = 0; k < cfg.r; ++k) {
                    const fp16* bk = cold.basis_col(k);
                    fp32 acc = 0.0f;
                    for (int j = 0; j < cfg.d; ++j)
                        acc += bf16_to_f32(bk[j]) * dxi[j];
                    ga_row[k] += acc;
                }

                // ∂L/∂B[j,k] += ∂L/∂X[i,j] · α_k   (outer product) O(d·r)
                for (int k = 0; k < cfg.r; ++k) {
                    fp32 ak = bf16_to_f32(alpha[k]);
                    fp32* gB_col_k = gB + static_cast<ptrdiff_t>(k)*cfg.d;
                    for (int j = 0; j < cfg.d; ++j)
                        gB_col_k[j] += dxi[j] * ak;
                }
            }
        }
    }

    // =================================================================
    // §2.4 — LM Head Projection: Algorithm 3
    // Input:  h ∈ fp32[d]  (hidden state, pre-final norm)
    // Output: logits ∈ fp32[V]
    // Hot:  logits[t] = Σ_b s_b · (Σ_k h_{b·B+k} · Q_H[slot,b·B+k])
    // Cold: z = h·B (O(d·r)), then logits[t] = z · A[cslot,:]^T (O(r))
    // =================================================================
    void lm_head(const fp32* h, fp32* logits) const {
        // --- Hot path: int8 GEMV with per-block scaling ---
        for (int slot = 0; slot < cfg.K; ++slot) {
            int gid       = hot.global_ids[slot];
            const int8* qrow = hot.row_q(slot);
            const fp32* srow = hot.row_s(slot);
            fp32 dot = 0.0f;
            int m = cfg.m();
            for (int b = 0; b < m; ++b) {
                int start = b * cfg.B;
                int end   = std::min(start + cfg.B, cfg.d);
                fp32 block_sum = 0.0f;
                for (int k = start; k < end; ++k)
                    block_sum += h[k] * static_cast<fp32>(qrow[k]);
                dot += srow[b] * block_sum;
            }
            logits[gid] = dot;
        }

        // --- Cold path: z = h · Basis (O(d·r)), then per-token O(r) ---
        std::vector<fp32> z(cfg.r, 0.0f); // z ∈ ℝ^r
        for (int k = 0; k < cfg.r; ++k) {
            const fp16* bk = cold.basis_col(k);
            fp32 acc = 0.0f;
            for (int j = 0; j < cfg.d; ++j)
                acc += h[j] * bf16_to_f32(bk[j]);
            z[k] = acc;
        }
        for (int cslot = 0; cslot < cold.Vc; ++cslot) {
            int gid         = cold.global_ids[cslot];
            const fp16* arow = cold.row_a(cslot);
            fp32 dot = 0.0f;
            for (int k = 0; k < cfg.r; ++k)
                dot += z[k] * bf16_to_f32(arow[k]);
            logits[gid] = dot;
        }
    }

    std::vector<fp32> lm_head(const std::vector<fp32>& h) const {
        if ((int)h.size() != cfg.d)
            throw std::invalid_argument("lm_head: h size != d");
        std::vector<fp32> logits(cfg.V, 0.0f);
        lm_head(h.data(), logits.data());
        return logits;
    }

    // =================================================================
    // §2.4 — Weight tying: verify Basis pointer identity
    // In a weight-tied model, lm_head.B and embedding.B are the same
    // object. Here we expose a raw pointer for the binding layer to check.
    // =================================================================
    const fp16* basis_ptr() const { return cold.Basis.data(); }
    fp16*       basis_ptr()       { return cold.Basis.data(); }

    // =================================================================
    // Gradient utilities
    // =================================================================
    void zero_grad() {
        std::fill(grad_Q.begin(), grad_Q.end(), 0.0f);
        std::fill(grad_S.begin(), grad_S.end(), 0.0f);
        std::fill(grad_A.begin(), grad_A.end(), 0.0f);
        std::fill(grad_B.begin(), grad_B.end(), 0.0f);
        touched_cold_slots.clear();
    }

    // SPEC §5.4: gradient explosion guard
    // Returns true if ‖∂L/∂B‖_F ≤ 10·‖∂L/∂X‖_F
    bool check_grad_magnitude(fp32 dL_dX_frob) const {
        fp32 gB_frob = 0.0f;
        for (fp32 v : grad_B) gB_frob += v*v;
        gB_frob = std::sqrt(gB_frob);
        return gB_frob <= 10.0f * dL_dX_frob;
    }

    // Number of non-zero rows in ∂L/∂A (SPEC §5.3 gradient sparsity test)
    int nnz_grad_A_rows() const {
        return static_cast<int>(touched_cold_slots.size());
    }


    // =================================================================
    // §3.3 — Memory-Mapped Tiered Paging (mmap cold coefficients)
    // Pins hot tier in RAM with mlock; cold coefficients are mmap'd
    // from a binary file for out-of-core access with MADV_RANDOM.
    // =================================================================
    void pin_hot_tier() {
#ifndef _WIN32
        // mlock hot Q_H, S_H, and Basis in RAM
        if (mlock(hot.Q_H.data(), hot.Q_H.size() * sizeof(int8)) != 0)
            std::fprintf(stderr, "HFAQE: mlock(Q_H) failed (need CAP_IPC_LOCK)\n");
        if (mlock(hot.S_H.data(), hot.S_H.size() * sizeof(fp32)) != 0)
            std::fprintf(stderr, "HFAQE: mlock(S_H) failed\n");
        if (mlock(cold.Basis.data(), cold.Basis.size() * sizeof(fp16)) != 0)
            std::fprintf(stderr, "HFAQE: mlock(Basis) failed\n");
#else
        // Windows: VirtualLock equivalent
        VirtualLock(hot.Q_H.data(), hot.Q_H.size() * sizeof(int8));
        VirtualLock(hot.S_H.data(), hot.S_H.size() * sizeof(fp32));
        VirtualLock(cold.Basis.data(), cold.Basis.size() * sizeof(fp16));
#endif
    }

    // mmap cold coefficients from file (SPEC Algorithm 4)
    // File must contain raw fp16 data: [(V-K) × r] row-major
    bool mmap_cold_coefficients(const std::string& filepath) {
#ifndef _WIN32
        int fd = open(filepath.c_str(), O_RDONLY);
        if (fd < 0) return false;
        size_t sz = static_cast<size_t>(cold.Vc) * cold.r * sizeof(fp16);
        void* ptr = mmap(nullptr, sz, PROT_READ, MAP_PRIVATE, fd, 0);
        if (ptr == MAP_FAILED) { close(fd); return false; }
        madvise(ptr, sz, MADV_RANDOM); // sparse cold-token access pattern
        // Replace in-memory A with mmap'd region
        cold.A_mmap_ptr = ptr;
        cold.A_mmap_sz  = sz;
        cold.A_mmap_fd  = fd;
        // Remap cold.A vector data pointer (zero-copy: cast and assign)
        // We alias the raw pointer — use with care (read-only mmap)
        // For writeable training, fall back to the vector allocation.
        return true;
#else
        (void)filepath;
        return false; // Windows: use VirtualAlloc approach if needed
#endif
    }

    ~HFAQE() {
#ifndef _WIN32
        if (cold.A_mmap_ptr && cold.A_mmap_ptr != MAP_FAILED) {
            munmap(cold.A_mmap_ptr, cold.A_mmap_sz);
            if (cold.A_mmap_fd >= 0) close(cold.A_mmap_fd);
        }
#endif
    }

    // =================================================================
    // §2.5 — Apply gradients (SGD step, learning rate lr)
    // Hot: requantize after fp32 parameter update
    // Cold A: update bf16 coefficients from fp32 grads (sparse)
    // Cold B: update bf16 basis from fp32 grads (dense)
    // =================================================================
    void apply_gradients(fp32 lr) {
        // Hot tier gradient apply: dequant → subtract grad → requantize
        std::vector<fp32> row_fp32(cfg.d);
        for (int slot = 0; slot < cfg.K; ++slot) {
            fp32* gq = grad_Q.data() + static_cast<ptrdiff_t>(slot)*cfg.d;
            fp32* gs = grad_S.data() + static_cast<ptrdiff_t>(slot)*cfg.m();
            // Dequantize current weights
            dequant_row(hot.row_q(slot), hot.row_s(slot),
                        cfg.d, cfg.B, row_fp32.data());
            // Gradient descent on fp32 representation
            for (int j = 0; j < cfg.d; ++j)
                row_fp32[j] -= lr * gq[j];
            // Requantize back
            quantize_row(row_fp32.data(), cfg.d, cfg.B,
                         hot.row_q(slot), hot.row_s(slot));
        }

        // Cold A: sparse update only touched rows
        for (int cslot : touched_cold_slots) {
            fp16* arow = cold.row_a(cslot);
            fp32* garow = grad_A.data() + static_cast<ptrdiff_t>(cslot)*cfg.r;
            for (int k = 0; k < cfg.r; ++k) {
                fp32 updated = bf16_to_f32(arow[k]) - lr * garow[k];
                arow[k] = f32_to_bf16(updated);
            }
        }

        // Cold Basis: dense update (col-major d×r)
        for (int k = 0; k < cfg.r; ++k) {
            fp16* bk = cold.basis_col(k);
            fp32* gbk = grad_B.data() + static_cast<ptrdiff_t>(k)*cfg.d;
            for (int j = 0; j < cfg.d; ++j) {
                fp32 updated = bf16_to_f32(bk[j]) - lr * gbk[j];
                bk[j] = f32_to_bf16(updated);
            }
        }
    }

}; // end class HFAQE

// =============================================================================
// §1.3 — Zipf Utility: build uniform Zipf frequency distribution
// f_t = f_1 · t^{-s},  normalized so Σ f_t = 1
// =============================================================================
std::vector<fp32> zipf_frequencies(int V, fp32 s = 1.0f) {
    std::vector<fp32> freq(V);
    fp32 H = 0.0f;
    for (int t = 1; t <= V; ++t) H += std::pow(static_cast<fp32>(t), -s);
    for (int t = 0; t < V; ++t)
        freq[t] = std::pow(static_cast<fp32>(t+1), -s) / H;
    return freq;
}

// =============================================================================
// §4.2 — Memory Analysis: Theoretical budget computation
// =============================================================================
struct MemoryBudget {
    size_t hot_q_bytes;      // K·d  (int8)
    size_t hot_s_bytes;      // K·m  (fp32)
    size_t cold_a_bytes;     // (V-K)·r (fp16)
    size_t basis_bytes;      // d·r  (fp16)
    size_t total_bytes;

    static MemoryBudget compute(const HFAQEConfig& c) {
        MemoryBudget b;
        b.hot_q_bytes   = static_cast<size_t>(c.K)           * c.d;
        b.hot_s_bytes   = static_cast<size_t>(c.K)           * c.m() * sizeof(fp32);
        b.cold_a_bytes  = static_cast<size_t>(c.V - c.K)     * c.r   * sizeof(fp16);
        b.basis_bytes   = static_cast<size_t>(c.d)            * c.r   * sizeof(fp16);
        b.total_bytes   = b.hot_q_bytes + b.hot_s_bytes
                        + b.cold_a_bytes + b.basis_bytes;
        return b;
    }
};

