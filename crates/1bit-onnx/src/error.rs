//! Error surface for the ONNX lane.
//!
//! We keep this typed so callers (notably `1bit-server`) can distinguish
//! "artifact missing on disk" (operator problem) from "ORT C++ library could
//! not be loaded" (deployment problem) from "VitisAI EP not registered"
//! (platform problem) without resorting to string matching.

use std::io;
use std::path::PathBuf;

/// Every failure mode the ONNX lane surfaces.
#[derive(Debug, thiserror::Error)]
pub enum OnnxError {
    /// A required file in the artifact directory is missing.
    ///
    /// Typical culprits: the caller pointed us at the raw Hugging Face
    /// directory instead of the OGA output directory, or the output
    /// directory is partially synced.
    #[error("missing artifact file: {0}")]
    MissingArtifact(PathBuf),

    /// The artifact directory exists but does not look like an OGA output
    /// directory (no `model.onnx`).
    #[error("directory at {0} is not an OGA model bundle (no model.onnx)")]
    NotAnArtifactDir(PathBuf),

    /// `genai_config.json` could not be parsed.
    #[error("genai_config.json parse error: {0}")]
    InvalidGenAiConfig(#[from] serde_json::Error),

    /// The ONNX Runtime C++ library could not be located or loaded.
    ///
    /// With the `load-dynamic` feature enabled on the `ort` crate, this
    /// fires when `ORT_DYLIB_PATH` is unset AND the bundled loader fails
    /// to find `libonnxruntime.so` on `LD_LIBRARY_PATH`.
    #[error("ORT runtime not available: {0}")]
    OrtRuntimeUnavailable(String),

    /// VitisAI EP was requested but not registered by the loaded ORT build.
    ///
    /// This is the expected state on Linux STX-H as of 2026-04-21; see
    /// `project_npu_path_analysis.md`. Callers can drop back to the CPU EP.
    #[error("VitisAI EP not available (expected on Linux STX-H until AMD ships the provider): {0}")]
    VitisaiUnavailable(String),

    /// Session creation failed for reasons other than provider availability.
    #[error("ORT session creation failed: {0}")]
    SessionInit(String),

    /// Tokenizer construction from `tokenizer.json` failed.
    #[error("tokenizer load failed: {0}")]
    TokenizerLoad(String),

    /// Underlying filesystem I/O error.
    #[error("io: {0}")]
    Io(#[from] io::Error),
}
