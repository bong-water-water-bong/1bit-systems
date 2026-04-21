// halo_cpu/ternary_gemv.h — CPU AVX2 ternary GEMV for halo-ai APU aggregator lane.
//
// Drop-in sibling to the gfx1151 HIP kernel in rocm-cpp/src/bonsai_tq2_gemv.hip.
// Consumes the SAME on-disk TQ2_0_g128 block layout (34 B/block: FP16 d @ 0,
// 32 bytes of 2-bit codes @ 2..33). Purpose: short-prompt prefill (L < ~33)
// where iGPU launch overhead dominates, and optional CPU prefill to keep the
// iGPU free for decode.
//
// Zero deps beyond <omp.h> + C++ stdlib. Static lib target libcpu_avx2_ternary.a.

#ifndef HALO_CPU_TERNARY_GEMV_H_
#define HALO_CPU_TERNARY_GEMV_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// TQ2_0_g128 ternary GEMV on CPU (AVX2 + FMA).
///
/// Inputs:
///   packed     : [N_out × (K_in/128) × 34 B] row-major, AoS block-interleaved.
///                Each 128-weight block is [FP16 d : 2 B][qs : 32 B].
///                2-bit code map: 00→-1, 01→0, 10→+1, 11→0 (reserved).
///   act        : [K_in] fp16 activations (bit-pattern in uint16_t).
///   out        : [N_out] fp16 output (bit-pattern in uint16_t). Written.
///   N_out      : output rows.
///   K_in       : input cols, MUST be a multiple of 128.
///   num_threads: OpenMP thread count; 0 or negative → omp_get_max_threads().
///
/// Thread-safe for disjoint `out` buffers. No dynamic allocation in hot path.
void halo_cpu_ternary_gemv_tq2(
    const uint8_t*  packed,
    const uint16_t* act,
    uint16_t*       out,
    int             N_out,
    int             K_in,
    int             num_threads);

/// Scalar reference — one row per call (serial). Used for differential tests.
/// Same inputs + layout as the AVX2 entry point.
void halo_cpu_ternary_gemv_tq2_scalar_ref(
    const uint8_t*  packed,
    const uint16_t* act,
    uint16_t*       out,
    int             N_out,
    int             K_in);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // HALO_CPU_TERNARY_GEMV_H_
