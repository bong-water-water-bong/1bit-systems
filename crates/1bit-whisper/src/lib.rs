//! 1bit-whisper — safe Rust surface over libwhisper (whisper.cpp 1.8.x).
//!
//! The crate is structured as two swappable implementations behind the
//! same public API:
//!
//! - **`stub` feature (default)**: `WhisperEngine` is backed by
//!   [`stub::StubEngine`]; every method returns
//!   [`WhisperError::UnsupportedStub`]. No native dependency, no link
//!   step, safe for CI hosts without whisper.cpp.
//! - **`real-whisper` feature**: `WhisperEngine` is backed by
//!   [`real::RealEngine`], which owns a `WhisperCtx*` allocated by the
//!   C++ shim in `cpp/shim.cpp`. The shim does the sliding-window
//!   `whisper_full` dance behind a 500 ms scheduler; Rust just feeds PCM
//!   and drains text.
//!
//! The two features are mutually exclusive from a behavioural standpoint
//! — if both are enabled `real-whisper` wins (see [`EngineImpl`]). Cargo
//! does not allow us to mark them as mutually exclusive declaratively,
//! so downstream crates should pick one via
//! `default-features = false, features = ["real-whisper"]`.
//!
//! See `docs/wiki/Halo-Whisper-Streaming-Plan.md` for the full streaming
//! design this scaffold lives under.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod ffi;

#[cfg(feature = "stub")]
pub mod stub;

#[cfg(feature = "real-whisper")]
pub mod real;

use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::Path;

// -----------------------------------------------------------------------------
// Public types
// -----------------------------------------------------------------------------

/// One transcription segment handed back to the caller.
///
/// The time range is measured in milliseconds from the start of the stream.
/// `text` is always UTF-8.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Partial {
    /// Transcribed text for this segment.
    pub text: String,
    /// Segment start in stream time (ms).
    pub start_ms: i64,
    /// Segment end in stream time (ms).
    pub end_ms: i64,
}

/// Error surface for the whisper engine.
#[derive(Debug)]
pub enum WhisperError {
    /// The `stub` feature is active and the caller tried to do real work.
    /// Build with `--features real-whisper --no-default-features` to enable
    /// the libwhisper-backed path.
    UnsupportedStub,
    /// The model file failed to load (bad path, corrupt weights, OOM).
    ModelLoadFailed {
        /// Path passed to the engine.
        path: String,
    },
    /// The underlying shim returned a non-zero status code.
    ShimError {
        /// Name of the shim entry point that failed.
        op: &'static str,
        /// Raw status returned.
        code: i32,
    },
    /// Could not convert the caller-supplied model path to a C string.
    InvalidPath,
    /// Internal decode of the shim's output text was not valid UTF-8.
    Utf8(std::string::FromUtf8Error),
}

impl fmt::Display for WhisperError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedStub => f.write_str(
                "1bit-whisper built with `stub` feature; rebuild with \
                 --features real-whisper --no-default-features",
            ),
            Self::ModelLoadFailed { path } => {
                write!(f, "failed to load whisper model at {path:?}")
            }
            Self::ShimError { op, code } => {
                write!(f, "shim call {op} failed with code {code}")
            }
            Self::InvalidPath => f.write_str("model path contained an interior NUL byte"),
            Self::Utf8(e) => write!(f, "shim returned non-UTF-8 text: {e}"),
        }
    }
}

impl std::error::Error for WhisperError {}

// -----------------------------------------------------------------------------
// Public engine — dispatch to stub or real based on active feature
// -----------------------------------------------------------------------------

/// Internal enum kept private; `WhisperEngine` forwards to it.
enum EngineImpl {
    #[cfg(feature = "real-whisper")]
    Real(real::RealEngine),
    #[cfg(all(feature = "stub", not(feature = "real-whisper")))]
    Stub(stub::StubEngine),
}

/// Safe Rust handle for a whisper streaming session.
///
/// Create with [`WhisperEngine::new`], push PCM with [`WhisperEngine::feed`],
/// harvest text with [`WhisperEngine::drain_partials`].
pub struct WhisperEngine {
    inner: EngineImpl,
}

impl WhisperEngine {
    /// Load a whisper `ggml` model from disk.
    ///
    /// Under the default `stub` feature this always returns
    /// [`WhisperError::UnsupportedStub`] — it does not touch the filesystem.
    pub fn new<P: AsRef<Path>>(model: P) -> Result<Self, WhisperError> {
        #[cfg(feature = "real-whisper")]
        {
            let eng = real::RealEngine::new(model)?;
            Ok(Self {
                inner: EngineImpl::Real(eng),
            })
        }
        #[cfg(all(feature = "stub", not(feature = "real-whisper")))]
        {
            let eng = stub::StubEngine::new(model)?;
            Ok(Self {
                inner: EngineImpl::Stub(eng),
            })
        }
        #[cfg(not(any(feature = "stub", feature = "real-whisper")))]
        {
            let _ = model;
            Err(WhisperError::UnsupportedStub)
        }
    }

    /// Push mono 16 kHz `s16le` PCM into the engine.
    pub fn feed(&mut self, pcm: &[i16]) -> Result<(), WhisperError> {
        match &mut self.inner {
            #[cfg(feature = "real-whisper")]
            EngineImpl::Real(e) => e.feed(pcm),
            #[cfg(all(feature = "stub", not(feature = "real-whisper")))]
            EngineImpl::Stub(e) => e.feed(pcm),
        }
    }

    /// Drain any partial transcripts that have landed since the last call.
    ///
    /// Returns an empty `Vec` when no new text is available. Always returns
    /// quickly; the expensive `whisper_full` call happens inside the shim
    /// during `feed`.
    pub fn drain_partials(&mut self) -> Result<Vec<Partial>, WhisperError> {
        match &mut self.inner {
            #[cfg(feature = "real-whisper")]
            EngineImpl::Real(e) => e.drain_partials(),
            #[cfg(all(feature = "stub", not(feature = "real-whisper")))]
            EngineImpl::Stub(e) => e.drain_partials(),
        }
    }
}

// -----------------------------------------------------------------------------
// In-crate tests (>=3 required by workspace convention)
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Stub mode: `new` on a missing file returns `UnsupportedStub`, not
    /// a panic, and not a filesystem-layer error.
    #[cfg(all(feature = "stub", not(feature = "real-whisper")))]
    #[test]
    fn stub_new_on_missing_file_returns_unsupported() {
        match WhisperEngine::new("/does/not/exist.bin") {
            Ok(_) => panic!("stub engine should never return Ok"),
            Err(WhisperError::UnsupportedStub) => { /* expected */ }
            Err(other) => panic!("expected UnsupportedStub, got {other:?}"),
        }
    }

    /// `Partial` serde roundtrip — proves the serde derives compile and
    /// preserve field values.
    #[test]
    fn partial_serde_roundtrip() {
        let p = Partial {
            text: "hello world".into(),
            start_ms: 1000,
            end_ms: 1750,
        };
        let j = serde_json::to_string(&p).expect("serialize");
        let p2: Partial = serde_json::from_str(&j).expect("deserialize");
        assert_eq!(p, p2);
        assert!(j.contains("\"text\":\"hello world\""));
        assert!(j.contains("\"start_ms\":1000"));
        assert!(j.contains("\"end_ms\":1750"));
    }

    /// Error `Display` renders without panic and mentions the relevant
    /// context — a cheap guard that error messages stay useful.
    #[test]
    fn error_display_is_informative() {
        let e = WhisperError::UnsupportedStub;
        let s = format!("{e}");
        assert!(s.contains("stub"));

        let e = WhisperError::ShimError {
            op: "onebit_whisper_feed",
            code: -1,
        };
        let s = format!("{e}");
        assert!(s.contains("onebit_whisper_feed"));
        assert!(s.contains("-1"));
    }
}
