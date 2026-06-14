// =============================================================================
// Core.cpp — HFAQE: Hierarchical Frequency-Adaptive Quantized Embedding
// =============================================================================
#ifndef HFAQE_CORE_CPP
#define HFAQE_CORE_CPP
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


// =============================================================================
// STAGE 2 STRUCTURES & MATH HELPERS (PRE-DECLARED)
// =============================================================================
// STAGE 2 — High-Quality Embedding Layer Extensions
// §2 Stage 2 Specification: Latent Master W, STE, TierAllocator, COMPRESS,
//   Composite Loss, Gram-Schmidt QR, Stage 2 Backward + Optimizer Step
// ██████████████████████████████████████████████████████████████████████████████
// =============================================================================

// =============================================================================
// C1.1 — MasterLatent: bf16 W[V×d] — sole trainable parameter (Stage 2)
// grad_W: fp32[V×d] — STE gradient accumulator
// After optimizer step, COMPRESS() refreshes (Q_H, S_H, A, Basis) caches.
// =============================================================================
struct MasterLatent {
    int V, d;
    std::vector<fp16> W;         // bf16[V×d]  — latent master embedding matrix
    std::vector<fp32> dW;        // fp32[V×d]  — STE gradient buffer

    void allocate(int V_, int d_) {
        V = V_; d = d_;
        W.assign(static_cast<size_t>(V) * d, 0);
        dW.assign(static_cast<size_t>(V) * d, 0.0f);
    }

    // Row accessors
    fp16*       row_w(int t)       { return W.data()  + static_cast<ptrdiff_t>(t) * d; }
    const fp16* row_w(int t) const { return W.data()  + static_cast<ptrdiff_t>(t) * d; }
    fp32*       row_dw(int t)      { return dW.data() + static_cast<ptrdiff_t>(t) * d; }
    const fp32* row_dw(int t) const { return dW.data() + static_cast<ptrdiff_t>(t) * d; }

    // Zero all STE gradients
    void zero_grad_master() {
        std::fill(dW.begin(), dW.end(), 0.0f);
    }

    // STE accumulate: dW[t] += g[d]   (called from backward_s2)
    void accumulate_grad(int t, const fp32* g) {
        fp32* dw = row_dw(t);
        for (int j = 0; j < d; ++j) dw[j] += g[j];
    }

    // Frobenius norm of the gradient matrix  ‖dW‖_F
    fp32 grad_norm_master() const {
        double s = 0.0;
        for (fp32 v : dW) s += static_cast<double>(v) * v;
        return static_cast<fp32>(std::sqrt(s));
    }

    // Frobenius norm of a single row gradient ‖dW[t]‖_2
    fp32 row_grad_norm(int t) const {
        const fp32* dw = row_dw(t);
        double s = 0.0;
        for (int j = 0; j < d; ++j) s += static_cast<double>(dw[j]) * dw[j];
        return static_cast<fp32>(std::sqrt(s));
    }

    // Get fp32 row of W (expand bf16)
    void get_row_fp32(int t, fp32* out) const {
        const fp16* w = row_w(t);
        for (int j = 0; j < d; ++j) out[j] = bf16_to_f32(w[j]);
    }

    // Set fp32 row of W (compress to bf16)
    void set_row_fp32(int t, const fp32* in) {
        fp16* w = row_w(t);
        for (int j = 0; j < d; ++j) w[j] = f32_to_bf16(in[j]);
    }

    // Initialize from Gaussian N(0, sigma²)
    void init_gaussian(fp32 sigma, uint64_t seed = 42) {
        std::mt19937_64 rng(seed);
        std::normal_distribution<fp32> dist(0.0f, sigma);
        for (int t = 0; t < V; ++t) {
            fp16* w = row_w(t);
            for (int j = 0; j < d; ++j) w[j] = f32_to_bf16(dist(rng));
        }
    }
};

// =============================================================================
// C1.7 — Modified Gram-Schmidt QR in-place on bf16 Basis columns
// Input/Output: Basis[d × r] column-major (col k = Basis + k*d)
// Ensures B^T·B ≈ I_r  (orthonormal columns)
// Complexity: O(d·r²)  — called every T_ortho steps
// =============================================================================
static void gram_schmidt_qr(fp16* Basis, int d, int r) {
    // Work in fp32 scratch to avoid bf16 accumulation error
    std::vector<fp32> Q(static_cast<size_t>(d) * r);

    // Load all columns to fp32
    for (int k = 0; k < r; ++k) {
        fp32* qk = Q.data() + static_cast<ptrdiff_t>(k) * d;
        const fp16* bk = Basis + static_cast<ptrdiff_t>(k) * d;
        for (int j = 0; j < d; ++j) qk[j] = bf16_to_f32(bk[j]);
    }

    // Modified Gram-Schmidt orthonormalisation
    for (int k = 0; k < r; ++k) {
        fp32* qk = Q.data() + static_cast<ptrdiff_t>(k) * d;

        // Orthogonalise against all previous columns
        for (int prev = 0; prev < k; ++prev) {
            const fp32* qp = Q.data() + static_cast<ptrdiff_t>(prev) * d;
            fp32 dot = 0.0f;
            for (int j = 0; j < d; ++j) dot += qk[j] * qp[j];
            for (int j = 0; j < d; ++j) qk[j] -= dot * qp[j];
        }

        // Normalise column k
        fp32 norm2 = 0.0f;
        for (int j = 0; j < d; ++j) norm2 += qk[j] * qk[j];
        fp32 norm = std::sqrt(norm2);
        if (norm > 1e-10f) {
            fp32 inv = 1.0f / norm;
            for (int j = 0; j < d; ++j) qk[j] *= inv;
        }
        // else: degenerate column — leave as-is (rare edge case)
    }

    // Write orthonormal columns back to bf16
    for (int k = 0; k < r; ++k) {
        const fp32* qk = Q.data() + static_cast<ptrdiff_t>(k) * d;
        fp16* bk = Basis + static_cast<ptrdiff_t>(k) * d;
        for (int j = 0; j < d; ++j) bk[j] = f32_to_bf16(qk[j]);
    }
}

// =============================================================================
// C1.8 — TierAllocator: gradient-magnitude–aware dynamic hot/cold migration
// Maintains migration score μ_i = β·freq_i + (1-β)·avg_grad_norm_i
// =============================================================================
struct TierAllocator {
    int V;
    int T_win;                              // history window length
    fp32 beta;                              // freq/grad blend (default 0.3)

    std::vector<fp32> mu;                   // migration scores [V]
    std::vector<fp32> grad_norm_accum;      // running sum of grad norms [V]
    std::vector<int>  grad_norm_count;      // count of accumulations [V]

    void init(int V_, int T_win_ = 300, fp32 beta_ = 0.3f) {
        V = V_; T_win = T_win_; beta = beta_;
        mu.assign(V, 0.0f);
        grad_norm_accum.assign(V, 0.0f);
        grad_norm_count.assign(V, 0);
    }

    // Accumulate per-token gradient norm at each step
    void record_grad_norm(int t, fp32 gnorm) {
        if (t < 0 || t >= V) return;
        grad_norm_accum[t] += gnorm;
        grad_norm_count[t]++;
    }

    // Update migration scores and select new hot set (top-K by μ)
    // Returns sorted list of new hot token IDs (size K)
    std::vector<int> reallocate(const std::vector<fp32>& freq, int K) {
        // Compute μ_i = β·freq_i + (1-β)·avg_grad_norm_i
        for (int t = 0; t < V; ++t) {
            fp32 avg_g = (grad_norm_count[t] > 0)
                         ? grad_norm_accum[t] / static_cast<fp32>(grad_norm_count[t])
                         : 0.0f;
            mu[t] = beta * freq[t] + (1.0f - beta) * avg_g;
        }

        // Select top-K by μ
        std::vector<int> order(V);
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + K, order.end(),
            [&](int a, int b) { return mu[a] > mu[b]; });

        // Reset accumulations for next window
        std::fill(grad_norm_accum.begin(), grad_norm_accum.end(), 0.0f);
        std::fill(grad_norm_count.begin(), grad_norm_count.end(), 0);

        return std::vector<int>(order.begin(), order.begin() + K);
    }
};

// =============================================================================
// C2.1 — Token taxonomy for ASCII byte vocab (V=256)
// Six semantic classes used by the InfoNCE supervised contrastive loss.
// =============================================================================
enum class TokenClass : int {
    DIGIT = 0,   // '0'..'9'  (10 tokens)
    UPPER = 1,   // 'A'..'Z'  (26 tokens)
    LOWER = 2,   // 'a'..'z'  (26 tokens)
    PUNCT = 3,   // printable non-alnum (33 tokens)
    SPACE = 4,   // whitespace: 0x09,0x0A,0x0D,0x20 (4 tokens)
    CTRL  = 5    // everything else (control / high bytes)
};
static constexpr int N_CLASSES = 6;

static TokenClass classify_byte(int id) {
    if (id >= '0' && id <= '9') return TokenClass::DIGIT;
    if (id >= 'A' && id <= 'Z') return TokenClass::UPPER;
    if (id >= 'a' && id <= 'z') return TokenClass::LOWER;
    if (id == 0x09 || id == 0x0A || id == 0x0D || id == 0x20)
        return TokenClass::SPACE;
    if (id >= 33 && id <= 126)  return TokenClass::PUNCT;
    return TokenClass::CTRL;
}

// =============================================================================
// C2.2 — Class centroids from master latent W
// Output: centroids[C × d] in fp32
// =============================================================================
static void compute_class_centroids(const MasterLatent& master,
                                    int V, int d,
                                    fp32* centroids) // [N_CLASSES × d]
{
    std::vector<int> counts(N_CLASSES, 0);
    std::fill(centroids, centroids + N_CLASSES * d, 0.0f);

    std::vector<fp32> row(d);
    for (int t = 0; t < V; ++t) {
        int c = static_cast<int>(classify_byte(t));
        master.get_row_fp32(t, row.data());
        fp32* cent = centroids + static_cast<ptrdiff_t>(c) * d;
        for (int j = 0; j < d; ++j) cent[j] += row[j];
        counts[c]++;
    }
    for (int c = 0; c < N_CLASSES; ++c) {
        if (counts[c] == 0) continue;
        fp32* cent = centroids + static_cast<ptrdiff_t>(c) * d;
        fp32 inv = 1.0f / static_cast<fp32>(counts[c]);
        for (int j = 0; j < d; ++j) cent[j] *= inv;
    }
}

// =============================================================================
// C2.3 — Supervised InfoNCE semantic contrastive loss
// L_semantic = -Σ_c Σ_{i∈S_c} log[ exp(cos(W_i, μ_c)/τ) / Σ_j exp(cos(W_i, W_j)/τ) ]
// Approximation: denominator uses full V tokens (exact for V=256)
// =============================================================================
static fp32 compute_L_semantic(const MasterLatent& master,
                                const fp32* centroids, // [N_CLASSES × d]
                                int V, int d,
                                fp32 tau = 0.05f)
{
    std::vector<fp32> wi(d), wj(d);
    double total_loss = 0.0;

    // Precompute all embeddings for denominator (O(V²) — fine for V=256)
    std::vector<fp32> all_w(static_cast<size_t>(V) * d);
    for (int t = 0; t < V; ++t) master.get_row_fp32(t, all_w.data() + (ptrdiff_t)t*d);

    for (int i = 0; i < V; ++i) {
        const fp32* wi_ptr = all_w.data() + static_cast<ptrdiff_t>(i) * d;
        int ci = static_cast<int>(classify_byte(i));
        const fp32* mu_c = centroids + static_cast<ptrdiff_t>(ci) * d;

        // Cosine similarity Wi·μ_c
        fp32 dot_ic = 0.0f, ni = 0.0f, nc = 0.0f;
        for (int j = 0; j < d; ++j) {
            dot_ic += wi_ptr[j] * mu_c[j];
            ni     += wi_ptr[j] * wi_ptr[j];
            nc     += mu_c[j]   * mu_c[j];
        }
        fp32 denom_ic = std::sqrt(ni) * std::sqrt(nc);
        fp32 cos_ic   = (denom_ic > 1e-10f) ? dot_ic / denom_ic : 0.0f;
        fp32 num_val  = cos_ic / tau;

        // Denominator: Σ_j exp(cos(Wi, Wj) / τ)
        double log_denom = 0.0;
        {
            // Compute max for stable log-sum-exp
            fp32 mx = -1e30f;
            std::vector<fp32> sims(V);
            for (int jj = 0; jj < V; ++jj) {
                if (jj == i) { sims[jj] = 0.0f; continue; }
                const fp32* wj_ptr = all_w.data() + static_cast<ptrdiff_t>(jj) * d;
                fp32 dot_ij = 0.0f, nj2 = 0.0f;
                for (int k = 0; k < d; ++k) {
                    dot_ij += wi_ptr[k] * wj_ptr[k];
                    nj2    += wj_ptr[k] * wj_ptr[k];
                }
                fp32 dij = std::sqrt(ni) * std::sqrt(nj2);
                sims[jj] = (dij > 1e-10f) ? (dot_ij / dij) / tau : 0.0f;
                mx = std::max(mx, sims[jj]);
            }
            double sum_e = 0.0;
            for (int jj = 0; jj < V; ++jj)
                if (jj != i) sum_e += std::exp(static_cast<double>(sims[jj] - mx));
            log_denom = static_cast<double>(mx) + std::log(sum_e + 1e-40);
        }

        total_loss += static_cast<double>(log_denom) - static_cast<double>(num_val);
    }

    return static_cast<fp32>(total_loss / V);
}

// =============================================================================
// C2.4 — Orthogonality regularizer: L_ortho = ‖B^T·B - I_r‖_F²
// =============================================================================
static fp32 compute_L_ortho(const fp16* Basis, int d, int r) {
    // Compute G = B^T·B  (r×r)
    std::vector<fp32> G(static_cast<size_t>(r) * r, 0.0f);
    for (int k1 = 0; k1 < r; ++k1) {
        const fp16* bk1 = Basis + static_cast<ptrdiff_t>(k1) * d;
        for (int k2 = k1; k2 < r; ++k2) {
            const fp16* bk2 = Basis + static_cast<ptrdiff_t>(k2) * d;
            fp32 dot = 0.0f;
            for (int j = 0; j < d; ++j)
                dot += bf16_to_f32(bk1[j]) * bf16_to_f32(bk2[j]);
            G[static_cast<ptrdiff_t>(k1)*r + k2] = dot;
            G[static_cast<ptrdiff_t>(k2)*r + k1] = dot; // symmetric
        }
    }
    // L_ortho = ‖G - I_r‖_F²
    fp32 loss = 0.0f;
    for (int k1 = 0; k1 < r; ++k1)
        for (int k2 = 0; k2 < r; ++k2) {
            fp32 diff = G[static_cast<ptrdiff_t>(k1)*r+k2] - (k1 == k2 ? 1.0f : 0.0f);
            loss += diff * diff;
        }
    return loss;
}

// =============================================================================
// C2.5 — Hot-cold alignment loss: L_align = ‖μ_hot - μ_cold‖²
// =============================================================================
static fp32 compute_L_align(const MasterLatent& master,
                             const std::vector<int>& hot_ids,
                             int V, int d)
{
    int K = static_cast<int>(hot_ids.size());
    int Vc = V - K;
    if (K == 0 || Vc == 0) return 0.0f;

    std::vector<bool> is_hot(V, false);
    for (int id : hot_ids) if (id >= 0 && id < V) is_hot[id] = true;

    std::vector<fp32> mu_hot(d, 0.0f), mu_cold(d, 0.0f);
    std::vector<fp32> row(d);

    int cnt_h = 0, cnt_c = 0;
    for (int t = 0; t < V; ++t) {
        master.get_row_fp32(t, row.data());
        if (is_hot[t]) {
            for (int j = 0; j < d; ++j) mu_hot[j] += row[j];
            ++cnt_h;
        } else {
            for (int j = 0; j < d; ++j) mu_cold[j] += row[j];
            ++cnt_c;
        }
    }
    if (cnt_h > 0) for (fp32& v : mu_hot) v /= static_cast<fp32>(cnt_h);
    if (cnt_c > 0) for (fp32& v : mu_cold) v /= static_cast<fp32>(cnt_c);

    fp32 loss = 0.0f;
    for (int j = 0; j < d; ++j) {
        fp32 diff = mu_hot[j] - mu_cold[j];
        loss += diff * diff;
    }
    return loss;
}

// =============================================================================
// C2.6 — Quantization-friendly regularizer: L_quant
// Penalizes clipping: L_quant = Σ_{i,b} Σ_j max(0, |W_{i,j}| - 127·s_{i,b})²
// where s_{i,b} = max_block(|W_i|) / 127
// =============================================================================
static fp32 compute_L_quant(const MasterLatent& master, int V, int d, int B_blk) {
    fp32 loss = 0.0f;
    std::vector<fp32> row(d);
    int m = (d + B_blk - 1) / B_blk;

    for (int t = 0; t < V; ++t) {
        master.get_row_fp32(t, row.data());
        for (int b = 0; b < m; ++b) {
            int start = b * B_blk;
            int end   = std::min(start + B_blk, d);
            fp32 abs_max = 0.0f;
            for (int j = start; j < end; ++j)
                abs_max = std::max(abs_max, std::abs(row[j]));
            fp32 s = (abs_max > 0.0f) ? abs_max / 127.0f : 1.0f;
            for (int j = start; j < end; ++j) {
                fp32 excess = std::abs(row[j]) - 127.0f * s;
                if (excess > 0.0f) loss += excess * excess;
            }
        }
    }
    return loss;
}

// =============================================================================
// C2.5.1 — Hot-cold alignment gradient: backward_align
// ∂L_align/∂W_i = 2(μ_hot - μ_cold)/K if hot, else -2(μ_hot - μ_cold)/(V-K)
// =============================================================================
static void backward_align(MasterLatent& master,
                            const std::vector<int>& hot_ids,
                            int V, int d, fp32 lambda2)
{
    if (lambda2 <= 0.0f) return;
    int K = static_cast<int>(hot_ids.size());
    int Vc = V - K;
    if (K == 0 || Vc == 0) return;

    std::vector<bool> is_hot(V, false);
    for (int id : hot_ids) if (id >= 0 && id < V) is_hot[id] = true;

    std::vector<fp32> mu_hot(d, 0.0f), mu_cold(d, 0.0f);
    std::vector<fp32> row(d);

    int cnt_h = 0, cnt_c = 0;
    for (int t = 0; t < V; ++t) {
        master.get_row_fp32(t, row.data());
        if (is_hot[t]) {
            for (int j = 0; j < d; ++j) mu_hot[j] += row[j];
            ++cnt_h;
        } else {
            for (int j = 0; j < d; ++j) mu_cold[j] += row[j];
            ++cnt_c;
        }
    }
    if (cnt_h > 0) for (fp32& v : mu_hot) v /= static_cast<fp32>(cnt_h);
    if (cnt_c > 0) for (fp32& v : mu_cold) v /= static_cast<fp32>(cnt_c);

    std::vector<fp32> diff(d);
    for (int j = 0; j < d; ++j) diff[j] = mu_hot[j] - mu_cold[j];

    fp32 scale_h = 2.0f * lambda2 / static_cast<fp32>(K);
    fp32 scale_c = -2.0f * lambda2 / static_cast<fp32>(Vc);

    for (int t = 0; t < V; ++t) {
        fp32* dw = master.row_dw(t);
        fp32 s = is_hot[t] ? scale_h : scale_c;
        for (int j = 0; j < d; ++j)
            dw[j] += s * diff[j];
    }
}

// =============================================================================
// C2.6.1 — Quantization-friendly gradient: backward_quant
// ∂L_quant/∂W_i,j = 2 · max(0, |W_i,j| - 127·s_i,b) · sgn(W_i,j)
// =============================================================================
static void backward_quant(MasterLatent& master, int V, int d, int B_blk, fp32 lambda4) {
    if (lambda4 <= 0.0f) return;
    std::vector<fp32> row(d);
    int m = (d + B_blk - 1) / B_blk;

    for (int t = 0; t < V; ++t) {
        master.get_row_fp32(t, row.data());
        fp32* dw = master.row_dw(t);
        for (int b = 0; b < m; ++b) {
            int start = b * B_blk;
            int end   = std::min(start + B_blk, d);
            fp32 abs_max = 0.0f;
            for (int j = start; j < end; ++j)
                abs_max = std::max(abs_max, std::abs(row[j]));
            fp32 s_limit = abs_max; // 127 * (abs_max / 127)

            for (int j = start; j < end; ++j) {
                fp32 val = row[j];
                fp32 excess = std::abs(val) - s_limit;
                if (excess > 0.0f) {
                    fp32 sgn = (val > 0.0f) ? 1.0f : -1.0f;
                    dw[j] += 2.0f * lambda4 * excess * sgn;
                }
            }
        }
    }
}

// =============================================================================
// C2.7 — Semantic loss gradient: backward_semantic
// Accumulates InfoNCE gradient w.r.t. dW_master (simplified, temperature-scaled)
// grad_W_i += λ1 · (p_pos_i - indicator + weighted_pull_toward_centroid)
// Approximation: use centroid-pull simplified gradient for efficiency.
// =============================================================================
static void backward_semantic(MasterLatent& master,
                               const fp32* centroids, // [N_CLASSES × d]
                               int V, int d,
                               fp32 tau, fp32 lambda1)
{
    if (lambda1 <= 0.0f) return;
    std::vector<fp32> row(d);
    fp32 scale = lambda1 / (tau * static_cast<fp32>(V));

    for (int t = 0; t < V; ++t) {
        int c = static_cast<int>(classify_byte(t));
        const fp32* mu_c = centroids + static_cast<ptrdiff_t>(c) * d;
        master.get_row_fp32(t, row.data());

        // Simplified gradient: push W_t toward μ_c
        // ∂L_semantic/∂W_t ≈ -scale · (μ_c - W_t)  (push toward centroid)
        fp32* dw = master.row_dw(t);
        for (int j = 0; j < d; ++j)
            dw[j] += scale * (row[j] - mu_c[j]); // gradient descend pulls toward μ_c
    }
}

// =============================================================================
// C2.8 — Orthogonality gradient: backward_ortho
// ∂L_ortho/∂B = 4 · B · (B^T·B - I_r)
// Applied to dBasis (fp32 scratch), then written back before COMPRESS
// =============================================================================
static void backward_ortho(const fp16* Basis, fp32* dBasis, int d, int r, fp32 lambda3) {
    if (lambda3 <= 0.0f) return;

    // G = B^T·B  (r×r)
    std::vector<fp32> G(static_cast<size_t>(r) * r, 0.0f);
    for (int k1 = 0; k1 < r; ++k1) {
        const fp16* bk1 = Basis + static_cast<ptrdiff_t>(k1) * d;
        for (int k2 = k1; k2 < r; ++k2) {
            const fp16* bk2 = Basis + static_cast<ptrdiff_t>(k2) * d;
            fp32 dot = 0.0f;
            for (int jj = 0; jj < d; ++jj)
                dot += bf16_to_f32(bk1[jj]) * bf16_to_f32(bk2[jj]);
            G[static_cast<ptrdiff_t>(k1)*r+k2] = dot;
            G[static_cast<ptrdiff_t>(k2)*r+k1] = dot;
        }
    }

    // (B^T·B - I_r)
    for (int k = 0; k < r; ++k)
        G[static_cast<ptrdiff_t>(k)*r+k] -= 1.0f;

    // ∂L/∂B[:,k] = 4 · λ3 · Σ_{k2} B[:,k2] · G[k2,k]
    fp32 coeff = 4.0f * lambda3;
    for (int k1 = 0; k1 < r; ++k1) {
        fp32* dbk1 = dBasis + static_cast<ptrdiff_t>(k1) * d;
        for (int k2 = 0; k2 < r; ++k2) {
            fp32 g = G[static_cast<ptrdiff_t>(k2)*r+k1] * coeff;
            if (std::abs(g) < 1e-12f) continue;
            const fp16* bk2 = Basis + static_cast<ptrdiff_t>(k2) * d;
            for (int jj = 0; jj < d; ++jj)
                dbk1[jj] += g * bf16_to_f32(bk2[jj]);
        }
    }
}

// =============================================================================
// C1.3 / C1.6 — Stage 2 Backward + apply_gradients_s2
// backward_s2: STE — dW_master[t] += dL_dX[i,:]  for every token
//              NO writes to grad_Q / grad_A / grad_B (caches are non-trainable)
// apply_gradients_s2: AdamW step on W_master, then call COMPRESS()
// =============================================================================

// Stage 2 backward: STE accumulation only — O(n·d)
// Per-token gradient clipping (θ_clip=1.0) applied before accumulation.
static void backward_s2(MasterLatent& master,
                         const fp32* dL_dX, const int* T, int n,
                         fp32 theta_clip = 1.0f)
{
    for (int i = 0; i < n; ++i) {
        int t = T[i];
        if (t < 0 || t >= master.V) continue;
        const fp32* dxi = dL_dX + static_cast<ptrdiff_t>(i) * master.d;

        // Per-token gradient norm for clipping
        fp32 gnorm = 0.0f;
        for (int j = 0; j < master.d; ++j) gnorm += dxi[j] * dxi[j];
        gnorm = std::sqrt(gnorm);

        // Clip factor: min(1, θ / ‖g_t‖)
        fp32 clip = (gnorm > theta_clip && gnorm > 1e-8f)
                    ? theta_clip / gnorm
                    : 1.0f;

        // Accumulate STE gradient
        fp32* dw = master.row_dw(t);
        for (int j = 0; j < master.d; ++j)
            dw[j] += clip * dxi[j];
    }
}

// AdamW optimizer state for master latent W[V×d]
struct AdamWMasterState {
    std::vector<fp32> m;   // first moment  [V×d]
    std::vector<fp32> v;   // second moment [V×d]
    int step = 0;

    void init(int V, int d) {
        size_t n = static_cast<size_t>(V) * d;
        m.assign(n, 0.0f);
        v.assign(n, 0.0f);
    }

    // Full AdamW step on W_master (all V tokens at once)
    void update_all(MasterLatent& master,
                    fp32 lr, fp32 beta1 = 0.9f, fp32 beta2 = 0.999f,
                    fp32 eps = 1e-8f, fp32 weight_decay = 0.01f,
                    // Tier-specific LR multipliers
                    const std::vector<bool>* is_hot_set = nullptr,
                    fp32 gamma_hot = 1.0f, fp32 gamma_cold = 2.0f)
    {
        ++step;
        fp32 bc1  = 1.0f - std::pow(beta1, static_cast<fp32>(step));
        fp32 bc2  = 1.0f - std::pow(beta2, static_cast<fp32>(step));
        fp32 lr_t = lr * std::sqrt(bc2) / bc1;

        int V = master.V, d = master.d;

        for (int t = 0; t < V; ++t) {
            // Tier-specific LR multiplier
            fp32 gamma = 1.0f;
            if (is_hot_set != nullptr)
                gamma = (*is_hot_set)[t] ? gamma_hot : gamma_cold;

            fp32* dw = master.row_dw(t);
            fp16* w  = master.row_w(t);
            fp32* mt = m.data() + static_cast<ptrdiff_t>(t) * d;
            fp32* vt = v.data() + static_cast<ptrdiff_t>(t) * d;

            for (int j = 0; j < d; ++j) {
                fp32 g = dw[j];
                mt[j] = beta1 * mt[j] + (1.0f - beta1) * g;
                vt[j] = beta2 * vt[j] + (1.0f - beta2) * g * g;
                // Expand bf16 → fp32, update, repack
                fp32 wj = bf16_to_f32(w[j]);
                // Decoupled weight decay
                wj *= (1.0f - lr * weight_decay);
                // Adam step
                wj -= gamma * lr_t * mt[j] / (std::sqrt(vt[j]) + eps);
                w[j] = f32_to_bf16(wj);
            }
        }
    }

    // Reset momentum for rows that changed tier (optional)
    void reset_momentum(const std::vector<int>& migrated_rows, int d) {
        for (int t : migrated_rows) {
            fp32* mt = m.data() + static_cast<ptrdiff_t>(t) * d;
            fp32* vt = v.data() + static_cast<ptrdiff_t>(t) * d;
            std::fill(mt, mt + d, 0.0f);
            std::fill(vt, vt + d, 0.0f);
        }
    }
};

// =============================================================================
// Utility: compute hot/cold gradient norm ratio (for Gate G5 logging)
// Returns {gnorm_hot, gnorm_cold}
// =============================================================================
static std::pair<fp32,fp32> compute_tier_grad_ratio(
    const MasterLatent& master,
    const std::vector<bool>& is_hot_set)
{
    double gnorm_hot2 = 0.0, gnorm_cold2 = 0.0;
    int V = master.V, d = master.d;
    for (int t = 0; t < V; ++t) {
        const fp32* dw = master.row_dw(t);
        double row_norm2 = 0.0;
        for (int j = 0; j < d; ++j) row_norm2 += static_cast<double>(dw[j]) * dw[j];
        if (is_hot_set[t]) gnorm_hot2  += row_norm2;
        else               gnorm_cold2 += row_norm2;
    }
    return { static_cast<fp32>(std::sqrt(gnorm_hot2)),
             static_cast<fp32>(std::sqrt(gnorm_cold2)) };
}

// =============================================================================


class HFAQE {
public:
    HFAQEConfig cfg;
    HotTier  hot;
    ColdTier cold;

    // Integrated Training State
    bool training_enabled = false;
    MasterLatent master;
    AdamWMasterState adam_W;
    TierAllocator tier_alloc;
    std::vector<bool> is_hot_set;
    std::vector<int> current_hot_ids;
    int global_step = 0;
    int migration_count = 0;

    // Caching token frequencies for dynamic tier reallocation
    std::vector<fp32> token_frequencies;

    // Training Hyperparameters (Default values matching SPEC)
    fp32 s2_beta = 0.3f;
    int s2_T_realloc = 300;
    int s2_T_ortho = 100;
    fp32 s2_theta_clip = 1.0f;
    fp32 s2_tau = 0.05f;
    fp32 s2_lambda_semantic = 0.1f;
    fp32 s2_lambda_align = 0.01f;
    fp32 s2_lambda_ortho = 0.001f;
    fp32 s2_lambda_quant = 0.001f;
    fp32 s2_gamma_hot = 1.0f;
    fp32 s2_gamma_cold = 2.0f;
    fp32 s2_gamma_basis = 0.5f;

    explicit HFAQE(const HFAQEConfig& config) : cfg(config) {}

    // Deprecated flag support (for compatibility)
    bool stage2_enabled = true; 

    void setup_training(fp32 beta = 0.3f, int T_realloc = 300, int T_ortho = 100) {
        training_enabled = true;
        s2_beta = beta;
        s2_T_realloc = T_realloc;
        s2_T_ortho = T_ortho;

        init_master_from_hfaqe(master, *this);
        adam_W.init(cfg.V, cfg.d);
        tier_alloc.init(cfg.V, T_realloc, beta);

        is_hot_set.assign(cfg.V, false);
        for (int slot = 0; slot < hot.K; ++slot) {
            int gid = hot.global_ids[slot];
            if (gid >= 0 && gid < cfg.V) is_hot_set[gid] = true;
        }
        current_hot_ids.assign(hot.global_ids.begin(), hot.global_ids.end());

        global_step = 0;
        migration_count = 0;
    }

    // Compatibility method
    void enable_stage2(fp32 beta = 0.3f, int T_realloc = 300, int T_ortho = 100) {
        setup_training(beta, T_realloc, T_ortho);
    }

    // -----------------------------------------------------------------
    // build_frequency_tiers
    // Takes token frequency vector (length V), builds hot/cold index maps.
    // Selects top-K by frequency as hot tier (§1.3 Zipf justification).
    // -----------------------------------------------------------------
    void build_frequency_tiers(const std::vector<fp32>& token_freq) {
        token_frequencies = token_freq;
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
    // §2.3 — Backward Pass: Integrated STE Backward
    // Input:  dL_dX ∈ fp32[n×d],  T ∈ int[n]
    // Key property: No O(V×d) dense gradient instantiated.
    // =================================================================
    void backward(const fp32* dL_dX, const int* T, int n, fp32 theta_clip = 1.0f) {
        if (!training_enabled) return;

        for (int i = 0; i < n; ++i) {
            int t = T[i];
            if (t < 0 || t >= cfg.V) continue;
            const fp32* dxi = dL_dX + static_cast<ptrdiff_t>(i) * cfg.d;

            // Per-token gradient norm for clipping
            fp32 gnorm = 0.0f;
            for (int j = 0; j < cfg.d; ++j) gnorm += dxi[j] * dxi[j];
            gnorm = std::sqrt(gnorm);

            // Clip factor
            fp32 clip = (gnorm > theta_clip && gnorm > 1e-8f)
                        ? theta_clip / gnorm
                        : 1.0f;

            // Accumulate STE gradient
            fp32* dw = master.row_dw(t);
            for (int j = 0; j < cfg.d; ++j)
                dw[j] += clip * dxi[j];

            // Record gradient norm for TierAllocator
            tier_alloc.record_grad_norm(t, gnorm);
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
        if (training_enabled) master.zero_grad_master();
    }

    // SPEC §5.4: gradient explosion guard
    // Returns true if ‖dW_master‖_F is within reasonable bounds
    bool check_grad_magnitude(fp32 dL_dX_frob) const {
        if (!training_enabled) return true;
        fp32 gW_frob = master.grad_norm_master();
        return gW_frob <= 10.0f * dL_dX_frob;
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
        return true;
#else
        (void)filepath;
        return false; // Windows: use VirtualAlloc approach if needed
#endif
    }

    // =================================================================
    // §7 — ARC: Dynamic vocabulary expansion
    // Adds a new cold token at runtime by learning only an r-dimensional
    // coefficient vector (no hot-tier or basis modification required).
    // =================================================================
    void add_cold_token(int new_id, const fp32* init_vec = nullptr) {
        // 1. Grow vocabulary config
        int new_V = std::max(cfg.V, new_id + 1);
        int d = cfg.d;
        int r = cfg.r;

        // 2. Allocate and initialize coefficient row α ∈ ℝ^r
        std::vector<fp16> alpha(r, fp16(0));
        if (init_vec != nullptr) {
            // Project init_vec onto basis B: α_k = Σ_j B[j,k] · init_vec[j]
            for (int k = 0; k < r; ++k) {
                const fp16* bk = cold.basis_col(k);
                fp32 dot = 0.0f;
                for (int j = 0; j < d; ++j)
                    dot += bf16_to_f32(bk[j]) * init_vec[j];
                alpha[k] = f32_to_bf16(dot);
            }
        }

        // 3. Append to cold tier caches
        int new_cslot = cold.Vc;
        cold.Vc += 1;
        cold.global_ids.push_back(new_id);
        cold.idx[new_id] = new_cslot;
        cold.A.insert(cold.A.end(), alpha.begin(), alpha.end());

        // 4. Handle training state if enabled
        if (training_enabled) {
            // Grow MasterLatent
            master.V = new_V;
            master.W.resize(static_cast<size_t>(new_V) * d, 0);
            master.dW.resize(static_cast<size_t>(new_V) * d, 0.0f);

            // Initialize new row in W
            if (init_vec != nullptr) {
                master.set_row_fp32(new_id, init_vec);
            } else {
                // If no init_vec, reconstruct from the projected alpha (which is zero if no init_vec)
                std::vector<fp32> recon(d, 0.0f);
                cold_reconstruct(cold.Basis.data(), alpha.data(), d, r, recon.data());
                master.set_row_fp32(new_id, recon.data());
            }

            // Grow AdamW states
            size_t old_sz = adam_W.m.size();
            size_t new_sz = static_cast<size_t>(new_V) * d;
            if (new_sz > old_sz) {
                adam_W.m.resize(new_sz, 0.0f);
                adam_W.v.resize(new_sz, 0.0f);
            }

            // Grow helper sets
            is_hot_set.resize(new_V, false);
            if ((int)token_frequencies.size() < new_V)
                token_frequencies.resize(new_V, 0.0f);
        }

        cfg.V = new_V;
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
    // Integrated Apply Gradients: Standard Flow
    // 1. Semantic, Align, Ortho, and Quant gradients
    // 2. AdamW update on MasterLatent
    // 3. Periodic Reallocation and QR
    // 4. Compress to tiered caches
    // =================================================================
    void apply_gradients(fp32 lr) {
        if (!training_enabled) return;

        // 1. Auxiliary gradients
        std::vector<fp32> centroids(N_CLASSES * cfg.d);
        compute_class_centroids(master, cfg.V, cfg.d, centroids.data());
        backward_semantic(master, centroids.data(), cfg.V, cfg.d, s2_tau, s2_lambda_semantic);
        backward_align(master, current_hot_ids, cfg.V, cfg.d, s2_lambda_align);
        backward_quant(master, cfg.V, cfg.d, cfg.B, s2_lambda_quant);

        // 2. Ortho gradient step on B
        {
            std::vector<fp32> dBasis(static_cast<size_t>(cfg.d) * cfg.r, 0.0f);
            backward_ortho(cold.Basis.data(), dBasis.data(), cfg.d, cfg.r, s2_lambda_ortho);
            fp32 basis_lr = lr * s2_gamma_basis;
            for (int k = 0; k < cfg.r; ++k) {
                fp16* bk = cold.basis_col(k);
                fp32* dbk = dBasis.data() + static_cast<ptrdiff_t>(k) * cfg.d;
                for (int j = 0; j < cfg.d; ++j) {
                    fp32 updated = bf16_to_f32(bk[j]) - basis_lr * dbk[j];
                    bk[j] = f32_to_bf16(updated);
                }
            }
        }

        // 3. AdamW step on W_master
        adam_W.update_all(master, lr, 0.9f, 0.999f, 1e-8f, 0.01f, &is_hot_set, s2_gamma_hot, s2_gamma_cold);

        // 4. Periodic QR re-orthogonalization
        if (global_step > 0 && global_step % s2_T_ortho == 0) {
            gram_schmidt_qr(cold.Basis.data(), cfg.d, cfg.r);
        }

        // 5. Periodic tier reallocation
        if (global_step > 0 && global_step % s2_T_realloc == 0) {
            auto old_hot = current_hot_ids;
            current_hot_ids = tier_alloc.reallocate(token_frequencies, cfg.K);

            std::unordered_set<int> old_set(old_hot.begin(), old_hot.end());
            for (int id : current_hot_ids)
                if (old_set.find(id) == old_set.end()) ++migration_count;

            std::fill(is_hot_set.begin(), is_hot_set.end(), false);
            for (int id : current_hot_ids)
                if (id >= 0 && id < cfg.V) is_hot_set[id] = true;

            std::vector<int> migrated;
            for (int id : current_hot_ids)
                if (old_set.find(id) == old_set.end()) migrated.push_back(id);
            if (!migrated.empty()) adam_W.reset_momentum(migrated, cfg.d);
        }

        // 6. Compress master to caches
        compress_master(*this, master, current_hot_ids);

        ++global_step;
        master.zero_grad_master();
    }

}; // end class HFAQE

// C1.5 — COMPRESS: W_master → {Q_H, S_H, A, Basis}
// Called after each optimizer step (or every T_realloc steps).
// Accepts new hot_ids (from TierAllocator::reallocate) or derives from
// existing tier assignment.  Performs:
//   1. Quantize hot rows from W into (Q_H, S_H)
//   2. Hard QR re-orthonormalise Basis
//   3. Project cold rows: A[cslot] = B^T · W[i]  (orthogonal projection)
// Complexity: O(K·d + d·r² + Vc·d·r)  — called rarely (every 300 steps)
// =============================================================================
static void compress_master(HFAQE& model, const MasterLatent& master,
                             const std::vector<int>& new_hot_ids)
{
    int V = model.cfg.V;
    int K = model.cfg.K;
    int d = model.cfg.d;
    int r = model.cfg.r;
    int B = model.cfg.B;

    // ---- 1. Rebuild tier index maps from new hot set -----------------------
    // Build new hot/cold global_ids and lookup maps
    std::vector<bool> is_hot(V, false);
    for (int id : new_hot_ids) {
        if (id >= 0 && id < V) is_hot[id] = true;
    }

    // Rebuild hot tier
    model.hot.global_ids.resize(K, -1);
    model.hot.idx.clear();
    int slot = 0;
    for (int id : new_hot_ids) {
        if (slot >= K) break;
        model.hot.global_ids[slot] = id;
        model.hot.idx[id] = slot;
        ++slot;
    }

    // Rebuild cold tier
    model.cold.global_ids.clear();
    model.cold.idx.clear();
    int cslot = 0;
    for (int t = 0; t < V; ++t) {
        if (!is_hot[t]) {
            model.cold.global_ids.push_back(t);
            model.cold.idx[t] = cslot++;
        }
    }

    // ---- 2. QR re-orthonormalise Basis B  (Hard reset every T_ortho) ------
    gram_schmidt_qr(model.cold.Basis.data(), d, r);

    // ---- 3. Quantize hot rows: W[i] → (Q_H[slot], S_H[slot]) -------------
    std::vector<fp32> row_fp32(d);
    for (int hslot = 0; hslot < K; ++hslot) {
        int gid = model.hot.global_ids[hslot];
        if (gid < 0 || gid >= V) continue;
        master.get_row_fp32(gid, row_fp32.data());
        quantize_row(row_fp32.data(), d, B,
                     model.hot.row_q(hslot), model.hot.row_s(hslot));
    }

    // ---- 4. Project cold rows: a_i = B^T · W[i]  (O(d·r) per cold token) -
    int Vc = model.cold.Vc;
    for (int cs = 0; cs < Vc; ++cs) {
        int gid = model.cold.global_ids[cs];
        if (gid < 0 || gid >= V) continue;
        master.get_row_fp32(gid, row_fp32.data());

        // a_k = Σ_j B[j,k] · W[gid,j]   (B column-major: basis_col(k)[j])
        fp16* a_row = model.cold.row_a(cs);
        for (int k = 0; k < r; ++k) {
            const fp16* bk = model.cold.basis_col(k);
            fp32 dot = 0.0f;
            for (int j = 0; j < d; ++j)
                dot += bf16_to_f32(bk[j]) * row_fp32[j];
            a_row[k] = f32_to_bf16(dot);
        }
    }
}

// =============================================================================


// Utility: Initialize MasterLatent from existing HFAQE tiered caches
// Used when loading a Stage-1 checkpoint and upgrading to Stage-2 training.
// Dequantizes hot rows, reconstructs cold rows into W_master.
// =============================================================================
static void init_master_from_hfaqe(MasterLatent& master, const HFAQE& model) {
    int V = model.cfg.V;
    int d = model.cfg.d;
    master.allocate(V, d);

    std::vector<fp32> row(d);

    // Hot rows: dequantize int8 → fp32 → store as bf16 in W
    for (int slot = 0; slot < model.hot.K; ++slot) {
        int gid = model.hot.global_ids[slot];
        if (gid < 0 || gid >= V) continue;
        dequant_row(model.hot.row_q(slot), model.hot.row_s(slot),
                    d, model.cfg.B, row.data());
        master.set_row_fp32(gid, row.data());
    }

    // Cold rows: B·α reconstruct → store as bf16 in W
    for (int cs = 0; cs < model.cold.Vc; ++cs) {
        int gid = model.cold.global_ids[cs];
        if (gid < 0 || gid >= V) continue;
        cold_reconstruct(model.cold.Basis.data(), model.cold.row_a(cs),
                         d, model.cfg.r, row.data());
        master.set_row_fp32(gid, row.data());
    }
}
// =============================================================================


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

// =============================================================================
// ██████████████████████████████████████████████████████████████████████████████

#endif // HFAQE_CORE_CPP
