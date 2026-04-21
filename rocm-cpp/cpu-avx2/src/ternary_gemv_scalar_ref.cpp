// ternary_gemv_scalar_ref.cpp — plain-C++ reference for the TQ2_0_g128 GEMV.
//
// One row at a time, no SIMD, no OpenMP. Matches the HIP kernel's decode
// and fp32 accumulate order EXACTLY so the differential test tolerance
// reflects only AVX2 lane-reduction order vs scalar.

#include "halo_cpu/ternary_gemv.h"

#include <immintrin.h>  // F16C scalar fp16↔fp32

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace {

constexpr int kBlockSize     = 128;
constexpr int kTQ2BlockBytes = 34;
constexpr int kCodesOffset   = 2;

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

inline float ternary_value(uint32_t code) noexcept {
    // 00 → -1, 01 → 0, 10 → +1, 11 → 0 (reserved).
    const float pos = (code == 2u) ? 1.0f : 0.0f;
    const float neg = (code == 0u) ? 1.0f : 0.0f;
    return pos - neg;
}

}  // namespace

extern "C" void halo_cpu_ternary_gemv_tq2_scalar_ref(
    const uint8_t*  packed,
    const uint16_t* act,
    uint16_t*       out,
    int             N_out,
    int             K_in)
{
    if (N_out <= 0 || K_in <= 0)            return;
    if ((K_in & (kBlockSize - 1)) != 0)     return;
    if (!packed || !act || !out)            return;

    const int num_blocks = K_in / kBlockSize;
    const std::size_t row_bytes =
        static_cast<std::size_t>(num_blocks) * static_cast<std::size_t>(kTQ2BlockBytes);

    constexpr float kFP16Max = 65504.0f;

    for (int row = 0; row < N_out; ++row) {
        const uint8_t* row_ptr = packed + static_cast<std::size_t>(row) * row_bytes;
        float acc = 0.0f;
        for (int b = 0; b < num_blocks; ++b) {
            const uint8_t* blk = row_ptr + static_cast<std::size_t>(b) * kTQ2BlockBytes;
            uint16_t d_u16;
            std::memcpy(&d_u16, blk, sizeof(d_u16));
            const float d = fp16_to_fp32(d_u16);

            float block_sum = 0.0f;
            for (int byte_idx = 0; byte_idx < 32; ++byte_idx) {
                const uint32_t qbyte = blk[kCodesOffset + byte_idx];
                for (int j = 0; j < 4; ++j) {
                    const uint32_t code = (qbyte >> (2 * j)) & 3u;
                    const float a = fp16_to_fp32(
                        act[static_cast<std::size_t>(b) * kBlockSize
                            + static_cast<std::size_t>(byte_idx) * 4u + j]);
                    block_sum += ternary_value(code) * a;
                }
            }
            acc += d * block_sum;
        }
        if (acc >  kFP16Max) acc =  kFP16Max;
        if (acc < -kFP16Max) acc = -kFP16Max;
        out[row] = fp32_to_fp16(acc);
    }
}
