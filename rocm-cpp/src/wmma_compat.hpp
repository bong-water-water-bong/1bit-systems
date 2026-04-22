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
// The RDNA3 prefill kernels in `prefill_standalone.hip` assume the duplication;
// they will need structural rework for RDNA4 (not just an intrinsic swap). Until
// that lands, only `wmma_peak_probe.hip` (which is structure-agnostic — both
// lane halves cooperate on the same M×N tile) compiles on gfx1201. Other files
// stay gfx1151-only and are skipped from the gfx1201 slice via the CMake
// `HIP_SOURCE_PROPERTY_HIP_ARCHITECTURES` filter.

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
