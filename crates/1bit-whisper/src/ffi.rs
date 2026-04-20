//! FFI declarations mirroring `cpp/shim.h`.
//!
//! These are `extern "C"` prototypes for the four `onebit_whisper_*`
//! functions the shim exposes. They are resolved at link time only when
//! the `real-whisper` feature is active (see build.rs). Under the default
//! `stub` feature the symbols are *not* linked — callers must route
//! through the stub module and never hit this FFI surface.

#![allow(non_camel_case_types)]
#![allow(missing_docs)]

use core::ffi::{c_char, c_int};

/// Opaque handle type. The Rust side never dereferences it.
#[repr(C)]
pub struct WhisperCtx {
    _private: [u8; 0],
}

unsafe extern "C" {
    pub fn onebit_whisper_init(model_path: *const c_char) -> *mut WhisperCtx;

    pub fn onebit_whisper_free(ctx: *mut WhisperCtx);

    pub fn onebit_whisper_feed(
        ctx: *mut WhisperCtx,
        pcm: *const i16,
        n_samples: usize,
    ) -> i32;

    pub fn onebit_whisper_drain(
        ctx: *mut WhisperCtx,
        out_buf: *mut c_char,
        out_len: usize,
    ) -> i32;
}

// Silence an unused-import warning in stub-only builds: `c_int` may not
// be referenced through this module when the real-whisper surface is off.
#[allow(dead_code)]
const _: Option<c_int> = None;
