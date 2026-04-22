//! FFI declarations mirroring `cpp/shim.h`.
//!
//! These are `extern "C"` prototypes for the `onebit_kokoro_*` functions
//! the shim exposes. They are resolved at link time only when the
//! `real-kokoro` feature is active (see build.rs). Under the default
//! `stub` feature the symbols are *not* linked — callers must route
//! through the stub module and never hit this FFI surface.

#![allow(non_camel_case_types)]
#![allow(missing_docs)]

use core::ffi::{c_char, c_int};

/// Opaque handle type. The Rust side never dereferences it.
#[repr(C)]
pub struct KokoroCtx {
    _private: [u8; 0],
}

unsafe extern "C" {
    pub fn onebit_kokoro_init(model_path: *const c_char) -> *mut KokoroCtx;

    pub fn onebit_kokoro_free(ctx: *mut KokoroCtx);

    /// Synthesize `text` with `voice` at `speed`. On success, writes at
    /// most `out_cap` s16le samples to `out_pcm` and returns the number
    /// of samples written. On failure returns a negative error code
    /// (buffer too small, null ctx, unknown voice, onnxruntime error).
    pub fn onebit_kokoro_synthesize(
        ctx: *mut KokoroCtx,
        text: *const c_char,
        voice: *const c_char,
        speed: f32,
        out_pcm: *mut i16,
        out_cap: usize,
    ) -> i64;
}

// Silence an unused-import warning in stub-only builds: `c_int` may not
// be referenced through this module when the real-kokoro surface is off.
#[allow(dead_code)]
const _: Option<c_int> = None;
