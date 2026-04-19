//! Raw `extern "C"` bindings for the rocm-cpp C API.
//!
//! Signatures are taken verbatim from
//! `/home/bcloud/repos/rocm-cpp/include/rocm_cpp/ck_gemm.h`.
//!
//! **Pointer conventions (matching the C header):**
//!   * `*const c_void` / `*mut c_void` are **device** pointers unless the
//!     comment says "host". Callers must have produced them via `hipMalloc`
//!     (or equivalent) on the same device that runs the kernel.
//!   * `*mut c_void` for a stream is a `hipStream_t`. Passing `null_mut()`
//!     means "default (null) stream" in HIP.
//!
//! **ABI:** every function here has C linkage. `rcpp_status_t` is a C enum
//! whose underlying type is `int` on every platform that builds rocm-cpp
//! (verified: the header uses a plain `typedef enum { ... } rcpp_status_t;`
//! with values 0..4 — sign does not matter for the 5 enumerators).
//!
//! This module is a leaf: no safety wrappers, no `Result`s, no logging.
//! The public API in `lib.rs` is the place for that.

#![allow(non_camel_case_types)]
#![allow(dead_code)]
// The canonical documentation for every `rcpp_*` symbol lives in
// `rocm-cpp/include/rocm_cpp/ck_gemm.h`; duplicating it here would go
// stale. The safe wrappers in `lib.rs` carry the Rust-facing docs.
#![allow(missing_docs)]

use core::ffi::{c_char, c_float, c_int, c_void};
use core::ffi::c_uint;

// -----------------------------------------------------------------------------
// HIP runtime bindings — a narrow subset of `hip/hip_runtime_api.h` sufficient
// for halo-router's device-memory lifecycle (alloc / copy / synchronize) and
// device enumeration. Symbols live in `libamdhip64.so`, which is already
// linked alongside `librocm_cpp.so` (see build.rs).
//
// `hipError_t` is an int-valued enum; 0 == hipSuccess. We fold any non-zero
// return into `RcppError::HipError` at the safe-wrapper boundary.
// `hipMemcpyKind` is likewise an int-valued enum with the familiar constants.
// -----------------------------------------------------------------------------
pub type hip_error_t = c_int;

/// Matches `hipMemcpyKind` from hip/hip_runtime_api.h.
pub const HIP_MEMCPY_HOST_TO_HOST: c_uint = 0;
pub const HIP_MEMCPY_HOST_TO_DEVICE: c_uint = 1;
pub const HIP_MEMCPY_DEVICE_TO_HOST: c_uint = 2;
pub const HIP_MEMCPY_DEVICE_TO_DEVICE: c_uint = 3;
pub const HIP_MEMCPY_DEFAULT: c_uint = 4;

/// `hipSuccess = 0`.
pub const HIP_SUCCESS: hip_error_t = 0;

#[cfg(feature = "link-rocm")]
unsafe extern "C" {
    pub fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> hip_error_t;
    pub fn hipFree(ptr: *mut c_void) -> hip_error_t;
    pub fn hipMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: c_uint,
    ) -> hip_error_t;
    pub fn hipMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: c_uint,
        stream: *mut c_void,
    ) -> hip_error_t;
    pub fn hipMemset(dst: *mut c_void, value: c_int, count: usize) -> hip_error_t;
    pub fn hipMemsetAsync(
        dst: *mut c_void,
        value: c_int,
        count: usize,
        stream: *mut c_void,
    ) -> hip_error_t;
    pub fn hipDeviceSynchronize() -> hip_error_t;
    pub fn hipGetDeviceCount(count: *mut c_int) -> hip_error_t;
    pub fn hipSetDevice(device_id: c_int) -> hip_error_t;
    pub fn hipGetLastError() -> hip_error_t;
    pub fn hipGetErrorString(err: hip_error_t) -> *const c_char;
}

#[cfg(not(feature = "link-rocm"))]
mod hip_stub {
    use super::*;
    #[inline]
    #[allow(unused_variables)]
    pub unsafe extern "C" fn hipMalloc(_ptr: *mut *mut c_void, _size: usize) -> hip_error_t { 1 }
    #[inline]
    #[allow(unused_variables)]
    pub unsafe extern "C" fn hipFree(_ptr: *mut c_void) -> hip_error_t { 1 }
    #[inline]
    #[allow(unused_variables)]
    pub unsafe extern "C" fn hipMemcpy(
        _dst: *mut c_void, _src: *const c_void, _count: usize, _kind: c_uint,
    ) -> hip_error_t { 1 }
    #[inline]
    #[allow(unused_variables)]
    pub unsafe extern "C" fn hipMemcpyAsync(
        _dst: *mut c_void, _src: *const c_void, _count: usize, _kind: c_uint,
        _stream: *mut c_void,
    ) -> hip_error_t { 1 }
    #[inline]
    #[allow(unused_variables)]
    pub unsafe extern "C" fn hipMemset(_dst: *mut c_void, _value: c_int, _count: usize) -> hip_error_t { 1 }
    #[inline]
    #[allow(unused_variables)]
    pub unsafe extern "C" fn hipMemsetAsync(
        _dst: *mut c_void, _value: c_int, _count: usize, _stream: *mut c_void,
    ) -> hip_error_t { 1 }
    #[inline]
    pub unsafe extern "C" fn hipDeviceSynchronize() -> hip_error_t { 1 }
    #[inline]
    #[allow(unused_variables)]
    pub unsafe extern "C" fn hipGetDeviceCount(count: *mut c_int) -> hip_error_t {
        // Safe to write zero into a valid pointer without a feature flag, but
        // we prefer to just report failure on CI hosts.
        1
    }
    #[inline]
    #[allow(unused_variables)]
    pub unsafe extern "C" fn hipSetDevice(_device_id: c_int) -> hip_error_t { 1 }
    #[inline]
    pub unsafe extern "C" fn hipGetLastError() -> hip_error_t { 0 }
    #[inline]
    #[allow(unused_variables)]
    pub unsafe extern "C" fn hipGetErrorString(_err: hip_error_t) -> *const c_char {
        core::ptr::null()
    }
}

#[cfg(not(feature = "link-rocm"))]
pub use hip_stub::*;


/// Mirrors `rcpp_status_t` from `ck_gemm.h`. Kept as a `#[repr(C)] i32` so
/// the compiler uses the same integer width the C side emits for an enum.
pub type rcpp_status_t = c_int;

pub const RCPP_OK: rcpp_status_t = 0;
pub const RCPP_INVALID_ARG: rcpp_status_t = 1;
pub const RCPP_UNSUPPORTED: rcpp_status_t = 2;
pub const RCPP_HIP_ERROR: rcpp_status_t = 3;
pub const RCPP_INTERNAL: rcpp_status_t = 4;

/// Opaque CK-GEMM handle. Callers pass a pointer; they never deref it.
#[repr(C)]
pub struct rcpp_ck_gemm_handle_t {
    _private: [u8; 0],
}

// -----------------------------------------------------------------------------
// CK GEMM (prefill)
// -----------------------------------------------------------------------------
#[cfg(feature = "link-rocm")]
unsafe extern "C" {
    pub fn rcpp_ck_gemm_create(
        M: c_int,
        N: c_int,
        K: c_int,
        handle_out: *mut *mut rcpp_ck_gemm_handle_t,
    ) -> rcpp_status_t;

    pub fn rcpp_ck_gemm_destroy(handle: *mut rcpp_ck_gemm_handle_t);

    pub fn rcpp_ck_gemm_run(
        handle: *mut rcpp_ck_gemm_handle_t,
        A_dev: *const c_void,
        B_dev_packed: *const c_void,
        C_dev: *mut c_void,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    pub fn rcpp_ternary_pack_pk_i4(
        ternary_host: *const i8,
        packed_host: *mut i8,
        K: c_int,
        N: c_int,
    ) -> rcpp_status_t;

    pub fn rcpp_ck_gemm_instance_string(
        handle: *const rcpp_ck_gemm_handle_t,
    ) -> *const c_char;

    // -------------------------------------------------------------------------
    // Phase 5 decode GEMV
    // -------------------------------------------------------------------------
    pub fn rcpp_ternary_gemv(
        packed_weights_dev: *const c_void,
        activations_i8_dev: *const c_void,
        activation_scale: c_float,
        row_scales_dev: *const c_void,
        output_dev: *mut c_void,
        M: c_int,
        K: c_int,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    pub fn rcpp_ternary_gemv_halo(
        packed_weights_dev: *const c_void,
        activations_i8_dev: *const c_void,
        activation_scale: c_float,
        row_scales_dev: *const c_void,
        output_dev: *mut c_void,
        M: c_int,
        K: c_int,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    pub fn rcpp_ternary_gemv_halo_f16(
        packed_weights_dev: *const c_void,
        activations_i8_dev: *const c_void,
        activation_scale: c_float,
        row_scales_dev: *const c_void,
        output_f16_dev: *mut c_void,
        M: c_int,
        K: c_int,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    pub fn rcpp_ternary_gemv_sherry_f16(
        packed_weights_dev: *const c_void,
        activations_i8_dev: *const c_void,
        activation_scale: c_float,
        row_scales_dev: *const c_void,
        output_f16_dev: *mut c_void,
        M: c_int,
        K: c_int,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    pub fn rcpp_ternary_gemv_tq1_halo_f16(
        packed_weights_dev: *const c_void,
        activations_i8_dev: *const c_void,
        activation_scale: c_float,
        row_scales_dev: *const c_void,
        output_f16_dev: *mut c_void,
        M: c_int,
        K_padded: c_int,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    // -------------------------------------------------------------------------
    // Primitive kernels
    // -------------------------------------------------------------------------
    pub fn rcpp_quantize_fp16_to_i8(
        x_fp16_dev: *const c_void,
        x_i8_dev: *mut c_void,
        scale_dev: *mut c_float,
        K: c_int,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    pub fn rcpp_rmsnorm_fp16(
        x_dev: *const c_void,
        weight_dev: *const c_void,
        y_dev: *mut c_void,
        eps: c_float,
        K: c_int,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    pub fn rcpp_rope_fp16(
        x_dev: *mut c_void,
        pos: c_int,
        theta: c_float,
        num_heads: c_int,
        head_dim: c_int,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    pub fn rcpp_silu_glu_fp16(
        up_dev: *const c_void,
        gate_dev: *const c_void,
        y_dev: *mut c_void,
        N: c_int,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    pub fn rcpp_relu2_glu_fp16(
        gate_dev: *const c_void,
        up_dev: *const c_void,
        y_dev: *mut c_void,
        N: c_int,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    pub fn rcpp_relu2_glu_rmsnorm_fp16(
        gate_dev: *const c_void,
        up_dev: *const c_void,
        ffn_sub_norm_dev: *const c_void,
        y_dev: *mut c_void,
        eps: c_float,
        N: c_int,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    pub fn rcpp_embedding_lookup_fp16(
        embedding_dev: *const c_void,
        token_id: c_int,
        y_dev: *mut c_void,
        hidden: c_int,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    pub fn rcpp_residual_add_fp16(
        y_dev: *mut c_void,
        src_dev: *const c_void,
        N: c_int,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    pub fn rcpp_residual_add_fp32_from_fp16(
        x_fp32_dev: *mut c_void,
        src_fp16_dev: *const c_void,
        N: c_int,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    pub fn rcpp_rmsnorm_fp32_in_fp16_out(
        x_fp32_dev: *const c_void,
        weight_dev: *const c_void,
        y_fp16_dev: *mut c_void,
        eps: c_float,
        K: c_int,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    pub fn rcpp_argmax_fp32(
        logits_dev: *const c_void,
        out_idx_dev: *mut c_void,
        V: c_int,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    pub fn rcpp_fp16_gemv(
        W_dev: *const c_void,
        x_dev: *const c_void,
        y_dev: *mut c_void,
        M: c_int,
        K: c_int,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    pub fn rcpp_fp32_to_fp16(
        x_fp32_dev: *const c_void,
        y_fp16_dev: *mut c_void,
        N: c_int,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    pub fn rcpp_top_k_fp32(
        logits_dev: *mut c_void,
        k: c_int,
        V: c_int,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    pub fn rcpp_softmax_fp32(
        logits_dev: *mut c_void,
        V: c_int,
        temperature: c_float,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    pub fn rcpp_sample_multinomial_fp32(
        probs_dev: *const c_void,
        r: c_float,
        out_idx_dev: *mut c_void,
        V: c_int,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    // -------------------------------------------------------------------------
    // Phase 6 — KV cache attention
    // -------------------------------------------------------------------------
    pub fn rcpp_kv_cache_attn_decode(
        Q_dev: *const c_void,
        K_dev: *const c_void,
        V_dev: *const c_void,
        out_dev: *mut c_void,
        num_q_heads: c_int,
        num_kv_heads: c_int,
        head_dim: c_int,
        seq_len: c_int,
        scale: c_float,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    pub fn rcpp_kv_cache_attn_decode_fd(
        Q_dev: *const c_void,
        K_dev: *const c_void,
        V_dev: *const c_void,
        out_dev: *mut c_void,
        num_q_heads: c_int,
        num_kv_heads: c_int,
        head_dim: c_int,
        seq_len: c_int,
        scale: c_float,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    pub fn rcpp_kv_cache_attn_prefill(
        Q_dev: *const c_void,
        K_dev: *const c_void,
        V_dev: *const c_void,
        out_dev: *mut c_void,
        num_q_heads: c_int,
        num_kv_heads: c_int,
        head_dim: c_int,
        seq_len: c_int,
        scale: c_float,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    pub fn rcpp_kv_cache_attn_decode_i8(
        Q_dev: *const c_void,
        K_i8_dev: *const c_void,
        V_i8_dev: *const c_void,
        K_scales_fp16_dev: *const c_void,
        V_scales_fp16_dev: *const c_void,
        out_dev: *mut c_void,
        num_q_heads: c_int,
        num_kv_heads: c_int,
        head_dim: c_int,
        seq_len: c_int,
        scale: c_float,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    pub fn rcpp_kv_cache_attn_prefill_i8(
        Q_dev: *const c_void,
        K_i8_dev: *const c_void,
        V_i8_dev: *const c_void,
        K_scales_fp16_dev: *const c_void,
        V_scales_fp16_dev: *const c_void,
        out_dev: *mut c_void,
        num_q_heads: c_int,
        num_kv_heads: c_int,
        head_dim: c_int,
        seq_len: c_int,
        scale: c_float,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    pub fn rcpp_quantize_fp16_to_i8_rowscale(
        x_fp16_dev: *const c_void,
        out_i8_dev: *mut c_void,
        scale_fp16_out_dev: *mut c_void,
        num_rows: c_int,
        row_len: c_int,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    // -------------------------------------------------------------------------
    // Standalone (CK-free) prefill
    // -------------------------------------------------------------------------
    pub fn rcpp_standalone_gemm(
        A_dev: *const c_void,
        B_dev_packed: *const c_void,
        C_dev: *mut c_void,
        M: c_int,
        N: c_int,
        K: c_int,
        stream: *mut c_void,
    ) -> rcpp_status_t;
}

// -----------------------------------------------------------------------------
// Stub shims (no `link-rocm` feature) — every symbol returns RCPP_UNSUPPORTED.
//
// These have identical signatures to the extern block above, so the rest of
// the crate can call them without cfg gating at every call site. The linker
// sees pure Rust symbols and never references the native .so.
// -----------------------------------------------------------------------------
#[cfg(not(feature = "link-rocm"))]
mod stub {
    use super::*;

    macro_rules! stub_fn {
        ($name:ident ( $( $arg:ident : $ty:ty ),* $(,)? ) -> $ret:ty) => {
            #[inline]
            #[allow(unused_variables, non_snake_case)]
            pub unsafe extern "C" fn $name( $( $arg : $ty ),* ) -> $ret {
                RCPP_UNSUPPORTED
            }
        };
        ($name:ident ( $( $arg:ident : $ty:ty ),* $(,)? )) => {
            #[inline]
            #[allow(unused_variables, non_snake_case)]
            pub unsafe extern "C" fn $name( $( $arg : $ty ),* ) {}
        };
    }

    stub_fn!(rcpp_ck_gemm_create(
        M: c_int, N: c_int, K: c_int,
        handle_out: *mut *mut rcpp_ck_gemm_handle_t
    ) -> rcpp_status_t);

    stub_fn!(rcpp_ck_gemm_destroy(handle: *mut rcpp_ck_gemm_handle_t));

    stub_fn!(rcpp_ck_gemm_run(
        handle: *mut rcpp_ck_gemm_handle_t,
        A_dev: *const c_void, B_dev_packed: *const c_void, C_dev: *mut c_void,
        stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_ternary_pack_pk_i4(
        ternary_host: *const i8, packed_host: *mut i8, K: c_int, N: c_int
    ) -> rcpp_status_t);

    #[inline]
    #[allow(unused_variables)]
    pub unsafe extern "C" fn rcpp_ck_gemm_instance_string(
        _handle: *const rcpp_ck_gemm_handle_t,
    ) -> *const c_char {
        core::ptr::null()
    }

    stub_fn!(rcpp_ternary_gemv(
        packed_weights_dev: *const c_void, activations_i8_dev: *const c_void,
        activation_scale: c_float, row_scales_dev: *const c_void,
        output_dev: *mut c_void, M: c_int, K: c_int, stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_ternary_gemv_halo(
        packed_weights_dev: *const c_void, activations_i8_dev: *const c_void,
        activation_scale: c_float, row_scales_dev: *const c_void,
        output_dev: *mut c_void, M: c_int, K: c_int, stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_ternary_gemv_halo_f16(
        packed_weights_dev: *const c_void, activations_i8_dev: *const c_void,
        activation_scale: c_float, row_scales_dev: *const c_void,
        output_f16_dev: *mut c_void, M: c_int, K: c_int, stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_ternary_gemv_sherry_f16(
        packed_weights_dev: *const c_void, activations_i8_dev: *const c_void,
        activation_scale: c_float, row_scales_dev: *const c_void,
        output_f16_dev: *mut c_void, M: c_int, K: c_int, stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_ternary_gemv_tq1_halo_f16(
        packed_weights_dev: *const c_void, activations_i8_dev: *const c_void,
        activation_scale: c_float, row_scales_dev: *const c_void,
        output_f16_dev: *mut c_void, M: c_int, K_padded: c_int,
        stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_quantize_fp16_to_i8(
        x_fp16_dev: *const c_void, x_i8_dev: *mut c_void,
        scale_dev: *mut c_float, K: c_int, stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_rmsnorm_fp16(
        x_dev: *const c_void, weight_dev: *const c_void, y_dev: *mut c_void,
        eps: c_float, K: c_int, stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_rope_fp16(
        x_dev: *mut c_void, pos: c_int, theta: c_float,
        num_heads: c_int, head_dim: c_int, stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_silu_glu_fp16(
        up_dev: *const c_void, gate_dev: *const c_void, y_dev: *mut c_void,
        N: c_int, stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_relu2_glu_fp16(
        gate_dev: *const c_void, up_dev: *const c_void, y_dev: *mut c_void,
        N: c_int, stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_relu2_glu_rmsnorm_fp16(
        gate_dev: *const c_void, up_dev: *const c_void,
        ffn_sub_norm_dev: *const c_void, y_dev: *mut c_void,
        eps: c_float, N: c_int, stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_embedding_lookup_fp16(
        embedding_dev: *const c_void, token_id: c_int, y_dev: *mut c_void,
        hidden: c_int, stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_residual_add_fp16(
        y_dev: *mut c_void, src_dev: *const c_void, N: c_int,
        stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_residual_add_fp32_from_fp16(
        x_fp32_dev: *mut c_void, src_fp16_dev: *const c_void, N: c_int,
        stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_rmsnorm_fp32_in_fp16_out(
        x_fp32_dev: *const c_void, weight_dev: *const c_void,
        y_fp16_dev: *mut c_void, eps: c_float, K: c_int, stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_argmax_fp32(
        logits_dev: *const c_void, out_idx_dev: *mut c_void, V: c_int,
        stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_fp16_gemv(
        W_dev: *const c_void, x_dev: *const c_void, y_dev: *mut c_void,
        M: c_int, K: c_int, stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_fp32_to_fp16(
        x_fp32_dev: *const c_void, y_fp16_dev: *mut c_void, N: c_int,
        stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_top_k_fp32(
        logits_dev: *mut c_void, k: c_int, V: c_int, stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_softmax_fp32(
        logits_dev: *mut c_void, V: c_int, temperature: c_float,
        stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_sample_multinomial_fp32(
        probs_dev: *const c_void, r: c_float, out_idx_dev: *mut c_void,
        V: c_int, stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_kv_cache_attn_decode(
        Q_dev: *const c_void, K_dev: *const c_void, V_dev: *const c_void,
        out_dev: *mut c_void,
        num_q_heads: c_int, num_kv_heads: c_int, head_dim: c_int,
        seq_len: c_int, scale: c_float, stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_kv_cache_attn_decode_fd(
        Q_dev: *const c_void, K_dev: *const c_void, V_dev: *const c_void,
        out_dev: *mut c_void,
        num_q_heads: c_int, num_kv_heads: c_int, head_dim: c_int,
        seq_len: c_int, scale: c_float, stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_kv_cache_attn_prefill(
        Q_dev: *const c_void, K_dev: *const c_void, V_dev: *const c_void,
        out_dev: *mut c_void,
        num_q_heads: c_int, num_kv_heads: c_int, head_dim: c_int,
        seq_len: c_int, scale: c_float, stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_kv_cache_attn_decode_i8(
        Q_dev: *const c_void,
        K_i8_dev: *const c_void, V_i8_dev: *const c_void,
        K_scales_fp16_dev: *const c_void, V_scales_fp16_dev: *const c_void,
        out_dev: *mut c_void,
        num_q_heads: c_int, num_kv_heads: c_int, head_dim: c_int,
        seq_len: c_int, scale: c_float, stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_kv_cache_attn_prefill_i8(
        Q_dev: *const c_void,
        K_i8_dev: *const c_void, V_i8_dev: *const c_void,
        K_scales_fp16_dev: *const c_void, V_scales_fp16_dev: *const c_void,
        out_dev: *mut c_void,
        num_q_heads: c_int, num_kv_heads: c_int, head_dim: c_int,
        seq_len: c_int, scale: c_float, stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_quantize_fp16_to_i8_rowscale(
        x_fp16_dev: *const c_void, out_i8_dev: *mut c_void,
        scale_fp16_out_dev: *mut c_void,
        num_rows: c_int, row_len: c_int, stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_standalone_gemm(
        A_dev: *const c_void, B_dev_packed: *const c_void, C_dev: *mut c_void,
        M: c_int, N: c_int, K: c_int, stream: *mut c_void
    ) -> rcpp_status_t);
}

#[cfg(not(feature = "link-rocm"))]
pub use stub::*;
