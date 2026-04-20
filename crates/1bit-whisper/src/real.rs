//! Real engine — libwhisper-backed, compiled only with the `real-whisper`
//! feature.
//!
//! The heavy lifting (sliding-window scheduler, `whisper_full` calls,
//! segment harvesting) lives on the C++ side in `cpp/shim.cpp`. This
//! module is a thin `unsafe`-wrapping layer that converts Rust slices /
//! paths into pointers the shim can consume and marshals the text-blob
//! response back into `Partial`s.

use crate::ffi;
use crate::{Partial, WhisperError};
use std::ffi::CString;
use std::path::Path;
use std::ptr;

/// Size of the scratch buffer we hand `onebit_whisper_drain`. 16 KiB fits a
/// handful of seconds of English transcript comfortably; overflow truncates
/// with a NUL terminator, so we never read past it.
const DRAIN_BUF: usize = 16 * 1024;

/// Tracks the last text we observed, so `drain_partials` can emit only the
/// *new* portion each call instead of re-reporting the entire rolling
/// window.
pub struct RealEngine {
    ctx: *mut ffi::WhisperCtx,
    last_text: String,
    // Running ms counter for the start/end of each drained segment. The
    // shim doesn't surface timestamps yet (future work: bubble up
    // `whisper_full_get_segment_t0/t1`), so we approximate from the
    // sample clock on the Rust side.
    samples_fed: u64,
}

// SAFETY: the underlying CtxImpl has an internal mutex guarding all
// mutable state, and `WhisperCtx*` is just a pointer to that impl. We
// expose `&mut self` on every Rust method, so the Rust type system
// already enforces single-threaded access from the Rust side; the mutex
// is belt-and-braces protection against a future `Arc<WhisperEngine>`
// pattern.
unsafe impl Send for RealEngine {}

impl RealEngine {
    /// Load a whisper ggml model from disk via the shim.
    pub fn new<P: AsRef<Path>>(model: P) -> Result<Self, WhisperError> {
        let path = model.as_ref();
        let path_str = path.to_string_lossy().into_owned();
        let c_path = CString::new(path_str.clone())
            .map_err(|_| WhisperError::InvalidPath)?;

        // SAFETY: `onebit_whisper_init` reads the NUL-terminated string
        // from `c_path.as_ptr()` and returns either NULL or a valid
        // opaque pointer. We keep `c_path` alive until after the call
        // returns.
        let ctx = unsafe { ffi::onebit_whisper_init(c_path.as_ptr()) };
        if ctx.is_null() {
            return Err(WhisperError::ModelLoadFailed { path: path_str });
        }

        Ok(Self {
            ctx,
            last_text: String::new(),
            samples_fed: 0,
        })
    }

    /// Push mono 16 kHz s16le PCM into the shim's ring buffer.
    pub fn feed(&mut self, pcm: &[i16]) -> Result<(), WhisperError> {
        let ptr = if pcm.is_empty() {
            ptr::null()
        } else {
            pcm.as_ptr()
        };
        // SAFETY: `ptr` is either null (with `n_samples == 0`, accepted
        // by the shim) or points at `pcm.len()` contiguous `i16` values
        // owned by the caller. The shim reads them, copies to f32, and
        // drops the reference before returning.
        let rc = unsafe { ffi::onebit_whisper_feed(self.ctx, ptr, pcm.len()) };
        if rc < 0 {
            return Err(WhisperError::ShimError {
                op: "onebit_whisper_feed",
                code: rc,
            });
        }
        self.samples_fed = self.samples_fed.saturating_add(pcm.len() as u64);
        Ok(())
    }

    /// Harvest the latest segment text and emit any new `Partial`s.
    ///
    /// First-pass implementation: we report the full rolling-window text
    /// as one `Partial` whenever it changes. The full streaming design
    /// (per-segment emission with `t0`/`t1` timestamps and dedup) lands
    /// in a follow-up PR; the scaffold just needs the plumbing.
    pub fn drain_partials(&mut self) -> Result<Vec<Partial>, WhisperError> {
        let mut buf = vec![0u8; DRAIN_BUF];
        // SAFETY: `buf.as_mut_ptr()` is a valid writable region of
        // `DRAIN_BUF` bytes; the shim writes at most `DRAIN_BUF` bytes
        // and always NUL-terminates within that range.
        let n = unsafe {
            ffi::onebit_whisper_drain(
                self.ctx,
                buf.as_mut_ptr() as *mut core::ffi::c_char,
                buf.len(),
            )
        };
        if n < 0 {
            return Err(WhisperError::ShimError {
                op: "onebit_whisper_drain",
                code: n,
            });
        }
        buf.truncate(n as usize);
        let text = String::from_utf8(buf).map_err(WhisperError::Utf8)?;

        if text == self.last_text || text.is_empty() {
            return Ok(Vec::new());
        }

        // Approximate stream-time ms from the running sample count. 16 kHz
        // means 16 samples per ms. Report one Partial spanning "since we
        // last emitted" → "now".
        let now_ms: i64 = (self.samples_fed / 16) as i64;
        let prev_ms: i64 = (self.last_text.len() as i64).min(now_ms);
        let start_ms = now_ms.saturating_sub(prev_ms);

        self.last_text = text.clone();
        Ok(vec![Partial {
            text,
            start_ms,
            end_ms: now_ms,
        }])
    }
}

impl Drop for RealEngine {
    fn drop(&mut self) {
        // SAFETY: `self.ctx` was produced by `onebit_whisper_init` and
        // has not been freed yet (this is the only owner). Passing it
        // back to `onebit_whisper_free` exactly once satisfies the
        // shim's ownership contract.
        if !self.ctx.is_null() {
            unsafe { ffi::onebit_whisper_free(self.ctx) };
            self.ctx = ptr::null_mut();
        }
    }
}
