// ternary_gemv_tq2_avx2.cpp — TQ2_0_g128 GEMV, AVX2 + FMA + OpenMP.
//
// Block layout (matches rocm-cpp/src/bonsai_tq2_gemv.hip — the on-disk .h1b
// bytes must round-trip between CPU and iGPU without reformatting):
//   [FP16 d : 2 B][qs[32] : 32 B]   total 34 B per 128-weight block
//   Code c ∈ {0,1,2,3} maps to ternary v(c) ∈ {-1, 0, +1, 0}.
//   LSB-first packing: weight j in block lives at
//     byte = qs[j / 4], shift = (j % 4) * 2, code = (byte >> shift) & 3.
//
// SIMD strategy:
//   - 256-bit ymm throughout (AVX2 + FMA, no AVX-512 — Zen5 widens easily but
//     we keep the user base wide; AVX-512 lane is a follow-up).
//   - Outer parallelism: one OpenMP thread per output row. N_out ≫ nthreads
//     (typ. 2048-6912 rows vs 16 threads), so this is ideal load balance, no
//     false sharing on `out` (each thread writes a distinct fp16 element,
//     cache lines are shared but write-once per row).
//   - Inner loop: per 128-weight block, unpack 32 codes at a time via pshufb
//     lane-LUT, convert to ±1.0f / 0.0f in fp32, load 32 fp16 activations via
//     4 × vcvtph2ps ymm, FMA into two fp32×8 accumulators, repeat 4× to cover
//     the block, fold, multiply by the block scale d, add into the row acc.
//
// Bandwidth-bound on weight bytes: 34 B/block × (K/128) × N output rows.
// On LPDDR5X @ ~256 GB/s, 2048×2048 needs ~1.06 MB per matmul → ~250 us/matmul
// peak. Our measured perf is reported by the test harness.
//
// Correctness tolerance vs the scalar reference: fp32 accumulate order differs
// from scalar ref but the scalar ref also uses fp32; per-block 128-wide sums
// of {-1,0,+1} × fp16 convert to roughly 7-bit mantissa ULP at most. Tests
// assert ≤4 bf16 ULP on the final fp16 output.

#include "halo_cpu/ternary_gemv.h"

#include <immintrin.h>
#include <omp.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace {

constexpr int kBlockSize     = 128;   // weights per block
constexpr int kTQ2BlockBytes = 34;    // 2 B d + 32 B codes
constexpr int kCodesOffset   = 2;

// fp16 helpers — use hardware F16C. We spell them out to avoid pulling in
// <x86intrin.h>'s larger surface.
inline float fp16_to_fp32(uint16_t h) noexcept {
    const __m128i v = _mm_cvtsi32_si128(static_cast<int>(h));
    const __m128  f = _mm_cvtph_ps(v);
    return _mm_cvtss_f32(f);
}

inline uint16_t fp32_to_fp16(float f) noexcept {
    const __m128  v = _mm_set_ss(f);
    const __m128i h = _mm_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    return static_cast<uint16_t>(_mm_extract_epi16(h, 0));
}

// Decode 32 2-bit codes (8 packed bytes = 64 bits = one __m256i with 8 active
// lanes replicated) into a __m256 of 8 fp32 values — 8 codes per call.
// We instead call this 4× per 32-byte block, each time consuming 8 codes.
//
// Algorithm per 8 codes: code is 2 bits, we need v = +1 for 0b10, -1 for 0b00,
// 0 for 0b01 and 0b11. That's:
//     v = int(bit1) - int(bit0 == 0 && bit1 == 0) ... nope. Simpler:
//     v = (c == 2) ? +1.0f : ((c == 0) ? -1.0f : 0.0f)
// Branch-free: let lo = c & 1, hi = (c >> 1) & 1.
//   (c,hi,lo) : v
//   (0,0,0)   : -1
//   (1,0,1)   :  0
//   (2,1,0)   : +1
//   (3,1,1)   :  0
// So v = hi - (hi == 0 && lo == 0 ? 0 : ... ) — cleaner: v = (1 - lo) * (2*hi - 1)
// wait (0,0,0): (1-0)*(0-1) = -1 OK; (1,0,1): (0)*anything = 0 OK;
// (2,1,0): (1)*(1) = +1 OK; (3,1,1): (0)*anything = 0 OK. Perfect.
//
// We use a 16-entry pshufb LUT instead: pack 4 codes/byte → expand to 4 bytes
// of {-1, 0, +1, 0} sign8, then widen to fp32. This is the cleanest AVX2
// expansion and takes 2 vpshufb + 1 vpsrlw + 1 vpand to unpack 4 codes per
// source byte into 4 separate signed-byte lanes.

// LUT: index = code ∈ {0,1,2,3}, value = signed i8 ∈ {-1,0,+1,0}.
//      We replicate the 4-entry LUT to all 16 lanes for pshufb (each lane
//      shuffles within its own 128-bit half, so replicate across both halves).
alignas(32) constexpr int8_t kTernaryLUT[32] = {
    -1,  0, +1,  0, -1,  0, +1,  0, -1,  0, +1,  0, -1,  0, +1,  0,
    -1,  0, +1,  0, -1,  0, +1,  0, -1,  0, +1,  0, -1,  0, +1,  0
};

// Expand 8 source bytes (= 32 codes, one block-quarter) into a __m256i of
// 32 signed-int8 ternary values {-1, 0, +1}. Input `src8` holds 8 bytes at
// the low 64 bits of the first lane; upper 24 bytes ignored. We produce:
//   out[ 0.. 7] = v(byte0.codes[0..3]), v(byte1.codes[0..3])
//   out[ 8..15] = byte2, byte3
//   out[16..23] = byte4, byte5
//   out[24..31] = byte6, byte7
// i.e. in-block weight index == out-lane index directly, matching the
// linear activation order act[b*128 + j].
inline __m256i unpack_32codes_i8(const uint8_t* __restrict__ codes8) noexcept {
    // Load 8 source bytes into the low qword of an xmm. Upper qword is
    // zero-extended by MOVQ semantics; that matters because we rely on the
    // upper-half zeros below when interleaving.
    uint64_t raw;
    std::memcpy(&raw, codes8, sizeof(raw));
    const __m128i src64 = _mm_cvtsi64_si128(static_cast<long long>(raw));

    // Extract the 4 code-positions within each byte by shifting the whole
    // 64-bit source by {0,2,4,6} and then masking to 0x03 per byte. For the
    // `k`-th shifted copy, lane b (b ∈ [0,8)) holds codes[j=k] of input byte b.
    const __m128i s0 = src64;
    const __m128i s1 = _mm_srli_epi64(src64, 2);
    const __m128i s2 = _mm_srli_epi64(src64, 4);
    const __m128i s3 = _mm_srli_epi64(src64, 6);
    const __m128i mask = _mm_set1_epi8(0x03);
    const __m128i c0 = _mm_and_si128(s0, mask);
    const __m128i c1 = _mm_and_si128(s1, mask);
    const __m128i c2 = _mm_and_si128(s2, mask);
    const __m128i c3 = _mm_and_si128(s3, mask);

    // Interleave so that output lane (byte*4 + j) carries codes[j] of byte.
    // unpacklo_epi8(c0, c1) → (c0[0], c1[0], c0[1], c1[1], ..., c0[7], c1[7])
    // unpacklo_epi16(lo01, lo23) picks 16-bit-pair lanes from the low half
    // giving bytes 0..3 of the input expanded to 16 codes; unpackhi covers
    // bytes 4..7 → 16 codes. Combined = 32 codes in linear weight order.
    const __m128i lo01 = _mm_unpacklo_epi8(c0, c1);
    const __m128i lo23 = _mm_unpacklo_epi8(c2, c3);
    const __m128i byt_lo = _mm_unpacklo_epi16(lo01, lo23);
    const __m128i byt_hi = _mm_unpackhi_epi16(lo01, lo23);
    const __m256i codes = _mm256_inserti128_si256(
        _mm256_castsi128_si256(byt_lo), byt_hi, 1);

    // pshufb via the 16-entry-replicated LUT maps {0,1,2,3} → {-1,0,+1,0}
    // as signed i8. Each 128-bit half shuffles within itself; the LUT is
    // identical in both halves, so the effective mapping is uniform.
    const __m256i lut = _mm256_load_si256(
        reinterpret_cast<const __m256i*>(kTernaryLUT));
    return _mm256_shuffle_epi8(lut, codes);
}

// One row, one block: accumulate sum_j v(code_j) * act[j] into two parallel
// fp32×8 accumulators. `codes` points to the 32-byte qs of the block.
// `act_f32_lo/hi/.../` are 4 ymm registers of fp32 activations, 32 total.
//
// Implementation strategy: unpack 32 ternary i8 values at once, widen to i16
// and then i32, convert to fp32 in two halves, FMA against the 32 fp32 acts
// in matching halves. That's 4 FMAs total (2 per 16-wide half) per block.
inline void block_mac(const uint8_t* __restrict__ codes,
                      const __m256& a0, const __m256& a1,
                      const __m256& a2, const __m256& a3,
                      __m256& acc_a, __m256& acc_b) noexcept {
    // 32 signed i8 ternary values: lanes 0..31 correspond to weights 0..31.
    // But a block is 128 weights spread across 32 bytes; we split into 4
    // quarters of 8 bytes × 4 codes/byte = 32 codes each.
    //
    // NOTE: this helper covers ONE quarter (32 codes, bytes 0..7 of `codes`).
    // The caller invokes it 4× with codes + 0/8/16/24 and the matching acts.
    const __m256i tern_i8 = unpack_32codes_i8(codes);

    // Sign-extend i8 → i16 (two __m256i), then i16 → i32 (four __m256i).
    const __m128i lo128 = _mm256_castsi256_si128(tern_i8);
    const __m128i hi128 = _mm256_extracti128_si256(tern_i8, 1);
    const __m256i t_i16_lo = _mm256_cvtepi8_epi16(lo128);   // codes 0..15 as i16
    const __m256i t_i16_hi = _mm256_cvtepi8_epi16(hi128);   // codes 16..31

    const __m256i t_i32_0 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(t_i16_lo));
    const __m256i t_i32_1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(t_i16_lo, 1));
    const __m256i t_i32_2 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(t_i16_hi));
    const __m256i t_i32_3 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(t_i16_hi, 1));

    const __m256  t_f32_0 = _mm256_cvtepi32_ps(t_i32_0);
    const __m256  t_f32_1 = _mm256_cvtepi32_ps(t_i32_1);
    const __m256  t_f32_2 = _mm256_cvtepi32_ps(t_i32_2);
    const __m256  t_f32_3 = _mm256_cvtepi32_ps(t_i32_3);

    // Two independent accumulators to hide FMA latency. Even-indexed ymm
    // groups go into acc_a, odd into acc_b.
    acc_a = _mm256_fmadd_ps(t_f32_0, a0, acc_a);
    acc_b = _mm256_fmadd_ps(t_f32_1, a1, acc_b);
    acc_a = _mm256_fmadd_ps(t_f32_2, a2, acc_a);
    acc_b = _mm256_fmadd_ps(t_f32_3, a3, acc_b);
}

// Horizontal-sum a __m256 of 8 fp32 into a scalar.
inline float hsum256_ps(__m256 v) noexcept {
    const __m128 lo = _mm256_castps256_ps128(v);
    const __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);                       // 4 lanes
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));              // 2 lanes (low 64b)
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x1));        // 1 lane
    return _mm_cvtss_f32(s);
}

// Process ONE full 128-weight block for a single row: returns d * sum.
inline float row_block_dot(const uint8_t* __restrict__ block,
                           const uint16_t* __restrict__ act_block) noexcept {
    // Load the fp16 scale d (bytes 0..1 of the block) and convert to fp32.
    uint16_t d_u16;
    std::memcpy(&d_u16, block, sizeof(d_u16));
    const float d = fp16_to_fp32(d_u16);

    const uint8_t* codes = block + kCodesOffset;

    // 128 fp16 activations → 16 × 8 fp32 = 4 × 32-fp32 chunks.
    // Use two alternating fp32 accumulators to hide FMA latency (Zen5 FMA
    // is 4-cycle latency, 2/cycle throughput — 8 in-flight is sweet spot,
    // 2 accumulators × 4 FMAs per block-quarter × 4 quarters = 32 FMAs with
    // 2-way ILP, fine).
    __m256 acc_a = _mm256_setzero_ps();
    __m256 acc_b = _mm256_setzero_ps();

    // 4 quarters, each covers 32 weights. Weights 0..31, 32..63, 64..95, 96..127.
    // Activations: 4 ymm fp32 loads per quarter (8 fp16 per ymm via vcvtph2ps).
    #pragma GCC unroll 4
    for (int q = 0; q < 4; ++q) {
        const uint8_t*  cq = codes + q * 8;
        const uint16_t* aq = act_block + q * 32;

        // vcvtph2ps: __m128i (8 × fp16) → __m256 (8 × fp32). We call it 4×
        // per quarter to cover 32 activations.
        const __m256 a0 = _mm256_cvtph_ps(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(aq +  0)));
        const __m256 a1 = _mm256_cvtph_ps(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(aq +  8)));
        const __m256 a2 = _mm256_cvtph_ps(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(aq + 16)));
        const __m256 a3 = _mm256_cvtph_ps(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(aq + 24)));

        block_mac(cq, a0, a1, a2, a3, acc_a, acc_b);
    }

    const float sum = hsum256_ps(_mm256_add_ps(acc_a, acc_b));
    return d * sum;
}

}  // namespace

// ── Public C API ─────────────────────────────────────────────────────────────

extern "C" void halo_cpu_ternary_gemv_tq2(
    const uint8_t*  packed,
    const uint16_t* act,
    uint16_t*       out,
    int             N_out,
    int             K_in,
    int             num_threads)
{
    if (N_out <= 0 || K_in <= 0)            return;
    if ((K_in & (kBlockSize - 1)) != 0)     return;
    if (!packed || !act || !out)            return;

    const int num_blocks = K_in / kBlockSize;
    const std::size_t row_bytes =
        static_cast<std::size_t>(num_blocks) * static_cast<std::size_t>(kTQ2BlockBytes);

    const int nthreads = (num_threads > 0)
        ? num_threads
        : omp_get_max_threads();

    // Clamp FP16 output to avoid ±Inf on overflow (matches the HIP kernel's
    // hand-rolled saturate — F16C's default rounding would emit Inf).
    constexpr float kFP16Max = 65504.0f;

    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int row = 0; row < N_out; ++row) {
        const uint8_t* row_ptr = packed + static_cast<std::size_t>(row) * row_bytes;
        float acc = 0.0f;
        for (int b = 0; b < num_blocks; ++b) {
            acc += row_block_dot(row_ptr + static_cast<std::size_t>(b) * kTQ2BlockBytes,
                                 act + static_cast<std::size_t>(b) * kBlockSize);
        }
        if (acc >  kFP16Max) acc =  kFP16Max;
        if (acc < -kFP16Max) acc = -kFP16Max;
        out[row] = fp32_to_fp16(acc);
    }
}
