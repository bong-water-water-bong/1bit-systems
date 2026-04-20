//! Stub engine — used when the `stub` feature is active (the default).
//!
//! Every entry point returns `WhisperError::UnsupportedStub` rather than
//! panicking. This lets the workspace build and test green on hosts
//! without libwhisper installed (CI, laptops, etc.) while keeping the
//! real-whisper code path a feature-flag flip away.

use crate::{Partial, WhisperError};
use std::path::Path;

/// Stub engine. Holds no state; every method returns the stub error.
pub struct StubEngine;

impl StubEngine {
    /// Pretend to load a model. Always errors.
    pub fn new<P: AsRef<Path>>(_model: P) -> Result<Self, WhisperError> {
        Err(WhisperError::UnsupportedStub)
    }

    /// Pretend to feed PCM. Always errors.
    pub fn feed(&mut self, _pcm: &[i16]) -> Result<(), WhisperError> {
        Err(WhisperError::UnsupportedStub)
    }

    /// Pretend to drain partials. Always errors.
    pub fn drain_partials(&mut self) -> Result<Vec<Partial>, WhisperError> {
        Err(WhisperError::UnsupportedStub)
    }
}
