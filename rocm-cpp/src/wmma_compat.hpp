// wmma_compat.hpp — WMMA intrinsic dispatch between RDNA3 (gfx1151) and RDNA4 (gfx1201).
//
// RDNA3 Wave32 WMMA:
//   A/B fragment per lane: half16_t (16 fp16 values)
//   B fragment per lane:   int8x16_t (16 int8 values, for iu8 WMMA)
//   C accumulator:         float8_t / int32x8_t
//   Builtin:
//     float8_t __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(half16_t, half16_t, float8_t)
//     int32x8_t __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(bool, int8x16_t, bool, int8x16_t, int32x8_t, bool)
//   Lane layout: lanes 0-15 own row/col; lanes 16-31 duplicate.
//
// RDNA4 Wave32 WMMA (gfx1201, RX 9070 XT / Navi 48):
//   A/B fragment per lane: half8_t (8 fp16 values — halved vs RDNA3, lanes now pack 2x data)
//   B fragment per lane:   int8x8_t (8 int8 values)
//   C accumulator:         float8_t / int32x8_t (unchanged)
//   Builtin (clang 19+ / ROCm 7+ rename):
//     float8_t __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(half8_t, half8_t, float8_t)
//     int32x8_t __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(bool, int8x8_t, bool, int8x8_t, int32x8_t, bool)
//   Lane layout: lanes 0-15 and 16-31 each own distinct rows/cols (no duplication).
//
// This header exposes:
//   - `wmma_half_frag_t`       per-arch A/B fragment type for fp16 WMMA
//   - `wmma_i8_frag_t`         per-arch A/B fragment type for iu8 WMMA
//   - `WMMA_K_ELEMS`           elements per lane in an A/B fragment (16 on gfx11, 8 on gfx12)
//   - `wmma_f32_f16(a, b, c)`  dispatched fp16→fp32 WMMA
//   - `wmma_i32_iu8(neg_a, a, neg_b, b, c, clamp)` dispatched iu8→i32 WMMA
//
// Kernels using this header can be written once and compile for both arches
// as long as load loops iterate `WMMA_K_ELEMS` instead of a hardcoded 16.
//
// Correctness note: the lane-duplication rule changed between RDNA3 and RDNA4.
// Kernels using WMMA must use `wmma_k_lane_base()` and `wmma_c_row_for_slot()`
// (defined below) to abstract the load-and-store differences. After that,
// `WMMA_K_ELEMS` and the dispatched `wmma_f32_f16` / `wmma_i32_iu8` builtins
// make one kernel source compile correctly on both arches. `prefill_standalone.hip`
// does this (2026-04-22 port); `wmma_peak_probe.hip` is structure-agnostic and
// was already dual-arch.

#pragma once

#include <cstdint>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#if defined(__gfx1200__) || defined(__gfx1201__) || defined(__gfx12__)
    #define ROCM_CPP_WMMA_GFX12 1
#else
    #define ROCM_CPP_WMMA_GFX12 0
#endif

#if ROCM_CPP_WMMA_GFX12
    using wmma_half_frag_t = __attribute__((ext_vector_type(8))) __fp16;
    using wmma_i8_frag_t   = __attribute__((ext_vector_type(8))) int8_t;
    constexpr int WMMA_K_ELEMS = 8;
#else
    using wmma_half_frag_t = __attribute__((ext_vector_type(16))) __fp16;
    using wmma_i8_frag_t   = __attribute__((ext_vector_type(16))) int8_t;
    constexpr int WMMA_K_ELEMS = 16;
#endif

using wmma_f32_acc_t   = __attribute__((ext_vector_type(8))) float;
using wmma_i32_acc_t   = __attribute__((ext_vector_type(8))) int32_t;

// FP16 × FP16 → FP32, 16×16×16 tile, Wave32.
__device__ __forceinline__
wmma_f32_acc_t wmma_f32_f16(wmma_half_frag_t a, wmma_half_frag_t b, wmma_f32_acc_t c) {
#if ROCM_CPP_WMMA_GFX12
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
#else
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#endif
}

// Signed INT8 × signed INT8 → INT32, 16×16×16 tile, Wave32.
//
// `NegA` / `NegB` / `Clamp` are compile-time template args — clang's WMMA
// builtins take these as `ImmTy` (immediate/constant) operands, so they
// can't be passed as runtime bools. Call sites pick the variant via
// explicit template args, e.g. `wmma_i32_iu8<true, true, false>(a, b, c)`.
template<bool NegA, bool NegB, bool Clamp>
__device__ __forceinline__
wmma_i32_acc_t wmma_i32_iu8(wmma_i8_frag_t a, wmma_i8_frag_t b, wmma_i32_acc_t c) {
#if ROCM_CPP_WMMA_GFX12
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(NegA, a, NegB, b, c, Clamp);
#else
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(NegA, a, NegB, b, c, Clamp);
#endif
}

// ---------------------------------------------------------------------------
// Per-arch A/B load + C store helpers.
//
// The A/B load layout differs between arches:
//   gfx11: every lane loads all 16 K elements for its row (lane&15) or col.
//          Lanes 16-31 duplicate lanes 0-15.
//   gfx12: lane group (lane>>4) picks the K half. Lanes 0-15 load K[0..7],
//          lanes 16-31 load K[8..15], for the same row/col.
//
// `wmma_k_lane_base(lane)` returns the first K element this lane owns:
//   gfx11: 0 (always — every lane walks K[0..15])
//   gfx12: (lane>>4) * 8   (0 for lanes 0-15, 8 for lanes 16-31)
//
// Combined with `WMMA_K_ELEMS` (16 on gfx11, 8 on gfx12) a load loop becomes:
//   for(int e = 0; e < WMMA_K_ELEMS; ++e) {
//       int k = k_tile_base + wmma_k_lane_base(lane) + e;
//       frag[e] = ...;
//   }
// and is uniform-looking to both arches.
__device__ __forceinline__
int wmma_k_lane_base(int lane) {
#if ROCM_CPP_WMMA_GFX12
    return (lane >> 4) * 8;
#else
    (void)lane;
    return 0;
#endif
}

// C accumulator row mapping for a given lane + output slot r ∈ [0, 8).
// Column is always `lane & 15` on both arches.
//   gfx11 (interleaved): lane i, slot r -> C[2*r + (i>>4), i&15]
//   gfx12 (blocked):     lane i, slot r -> C[(i>>4)*8 + r, i&15]
__device__ __forceinline__
int wmma_c_row_for_slot(int lane, int r) {
#if ROCM_CPP_WMMA_GFX12
    return (lane >> 4) * 8 + r;
#else
    return 2 * r + (lane >> 4);
#endif
}
