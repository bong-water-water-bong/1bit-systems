//! Stub engine — used when the `stub` feature is active (the default).
//!
//! Every entry point returns `KokoroError::UnsupportedStub` rather than
//! panicking. This lets the workspace build and test green on hosts
//! without kokoro.cpp / onnxruntime installed (CI, laptops, etc.) while
//! keeping the real-kokoro code path a feature-flag flip away.

use crate::KokoroError;
use std::path::Path;

/// Stub engine. Holds no state; every method returns the stub error.
pub struct StubEngine;

impl StubEngine {
    /// Pretend to load a model. Always errors with `UnsupportedStub`.
    pub fn new<P: AsRef<Path>>(_model: P) -> Result<Self, KokoroError> {
        Err(KokoroError::UnsupportedStub)
    }

    /// Pretend to synthesize speech. Always errors with `UnsupportedStub`.
    /// Input validation has already happened at the `KokoroEngine` layer
    /// by the time we get here.
    pub fn synthesize(
        &mut self,
        _text: &str,
        _voice: &str,
        _speed: f32,
    ) -> Result<Vec<i16>, KokoroError> {
        Err(KokoroError::UnsupportedStub)
    }
}
