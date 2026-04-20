# Phase 5 Decode — Integration Map

The Phase 5 ternary×INT8 GEMV kernel (wired as `rcpp_ternary_gemv` /
`rcpp_ternary_gemv_halo`) delivers **2.4–7.1× over rocBLAS FP16** and
**3.4–5.5× over the v1 ternary GEMV** on every measured decode shape.
Full-model simulation on BitNet-2B-4T: **238 tok/s** vs Prism GGUF's
**117.7 tok/s** — a **2.03× uplift** if a consumer wires this in end-to-end.

This doc is the integration map for projects that want to consume Phase 5
via `librocm_cpp.so`. It targets **halo-1bit/mlx-engine** specifically as the
nearest-term consumer, but the same pattern applies to any C++ inference engine.

## What Phase 5 needs from the caller

| Input | Device-side format | How the caller produces it |
|---|---|---|
| Packed ternary weights | `uint32[M, K/16]` (Phase 5) or `uint8[M, (K+3)/4]` (halo variant) | Load once at model init |
| INT8 activations | `int8_t[K]` | Quantize FP16/FP32 activations with per-token scale |
| Activation scale | `float` | `max(|A|) / 127`, recomputed per input |
| Row scales | `float[M]` | Loaded with weights |
| Output buffer | `float[M]` | Caller allocates |

## BitNet-2B-4T decode wiring map

For reference, BitNet-2B-4T has 30 transformer layers plus an LM head.
Every ternary linear layer is a decode-side GEMV at batch=1, and every one
can be routed through `rcpp_ternary_gemv_halo` given halo-1bit's existing
uint8 packed format. Mapping from `TernaryLinear::forward(x)` to the C API:

```cpp
// halo-1bit layer shapes (from mlx_loader.cpp)
// q_proj : [nh*hd,  hs]  = [2560, 2560]
// k_proj : [nkv*hd, hs]  = [ 640, 2560]
// v_proj : [nkv*hd, hs]  = [ 640, 2560]
// o_proj : [hs, nh*hd]   = [2560, 2560]
// gate   : [is, hs]      = [6912, 2560]
// up     : [is, hs]      = [6912, 2560]
// down   : [hs, is]      = [2560, 6912]
// lm_head: [vocab, hs]   = [128256, 2560]

void phase5_forward(const TernaryLinear& tl,
                    const float* x_fp_dev,   // [K] activation
                    float* y_dev,            // [M] output
                    int8_t* scratch_i8_dev,  // [K] reusable INT8 scratch
                    void* stream) {
    // 1) Compute per-vector activation scale + quantize to INT8
    //    Must be a separate tiny kernel (or call into an MLX reduction).
    float x_scale = reduce_absmax_fp16_to_fp32(x_fp_dev, K) / 127.0f;
    quantize_fp32_to_i8(x_fp_dev, scratch_i8_dev, x_scale, K, stream);

    // 2) Single call into librocm_cpp
    rcpp_ternary_gemv_halo(
        /*packed=*/     tl.packed_raw_ptr,   // pull from mx::array.data<void>()
        /*x_i8=*/       scratch_i8_dev,
        /*x_scale=*/    x_scale,
        /*row_scales=*/ tl.scales_raw_ptr,
        /*y=*/          y_dev,
        /*M=*/          tl.rows,
        /*K=*/          tl.cols,
        stream);
}
```

## Required halo-1bit changes

1. **Link librocm_cpp**: add `find_package(rocm_cpp)` or an imported target in
   `~/halo-1bit/mlx-engine/CMakeLists.txt`, link `rocm_cpp` into
   `halo1bit-mlx`. Pre-built `librocm_cpp.so` is staged at
   `~/rocm-cpp/install/lib/`.

2. **Pull raw device pointers from MLX arrays**. MLX exposes `mx::array::data<T>()`
   which returns the device-side raw pointer after the array has been evaluated.
   Ensure the array is contiguous (`mx::contiguous(arr)`) before taking pointers.

3. **Add a small activation-quant kernel**. ~40 lines of HIP that does:
   - Block-reduction for `max(|x|)` across K
   - Broadcast the reciprocal scale
   - Round-to-nearest quantize to INT8
   - Output both the INT8 buffer and the scalar scale

4. **Modify `TernaryLinear::forward`** to dispatch to Phase 5 when HIP is
   available and MLX is in eager mode (we need synchronization to get the
   raw pointer). Keep the MLX dequant-matmul path as a fallback for non-ROCm
   backends.

5. **Rebuild** halo-1bit/mlx-engine. The existing build already handles HIP
   linkage; adding librocm_cpp is a one-line CMake change.

## Why this is deferred rather than done in-session

The kernel side is complete and measured. Full halo-1bit wiring requires
MLX-side knowledge — specifically how `mx::array` pins its HIP device
pointer, how eager vs lazy mode affects pointer stability, and how the
MLX ROCm backend coordinates streams with third-party HIP code.

Getting that exactly right is 1–2 sessions of careful code; rushing it in
this session risks shipping a broken integration that gives worse numbers
than the kernel measurements promise.

## What's shippable today

- `librocm_cpp.so` at `~/rocm-cpp/install/lib/` with both Phase 5 variants
  exposed through the C header.
- A standalone benchmark (`build/test_ck_gemm`) proving correctness + perf.
- The full `tools/bench_ternary` v1 reference for A/B comparison.
- This document as the roadmap for integrators.

Adoption takes time. Meeting it halfway with clear docs and a stable ABI.
