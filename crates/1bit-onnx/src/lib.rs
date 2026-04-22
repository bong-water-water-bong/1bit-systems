//! `onebit-onnx` — ONNX Runtime lane for the 1bit-systems stack.
//!
//! Loads `.onnx` artifacts produced by the ONNX Runtime GenAI Model Builder
//! (`python -m onnxruntime_genai.models.builder -p int4 -e cpu`). First real
//! artifact on disk: `1bit-halo-core/artifacts/trilm-3.9b-int4-n4/`
//! (TriLM_3.9B_Unpacked, LLaMA-arch, 151 MatMulNBits ops, bits=4, block=32).
//!
//! **Scope of this crate** (deliberately minimal):
//!
//! * Artifact discovery + config parsing (`model.rs`).
//! * A thin `ort::Session` wrapper that picks the execution provider from a
//!   compile-time feature set (`session.rs`).
//! * Error surface (`error.rs`).
//!
//! What this crate does **not** do yet:
//!
//! * No KV-cache driven decode loop. The first pass loads + runs prefill for
//!   a single token batch to validate the graph, nothing more. Full
//!   generation lives behind follow-up work once the NPU EP is in hand — the
//!   `com.microsoft:MatMulNBits` graph is VitisAI's target, but running it on
//!   CPU at 3.9B is a bench datapoint, not a production path.
//! * No HTTP layer. The router / server crates own HTTP. This crate stays
//!   library-only so `1bit-server` can wire it in at its leisure.
//!
//! **NPU note.** Per the workspace `CLAUDE.md` Rule E, the NPU path is ONNX
//! Runtime C++ via the AMD VitisAI Execution Provider. The `ort` crate is a
//! Rust FFI binding around that same ONNX Runtime C++ — so this crate is
//! Rule E compliant. Feature `vitisai` flips the EP list to request
//! VitisAI first, falling back to CPU if the provider isn't registered.
//!
//! On Linux STX-H today (2026-04-21) the VitisAI EP Linux provider for
//! Strix Halo is not shipped by AMD — see `project_npu_path_analysis.md`
//! and `project_amd_hf_npu_models.md`. Enabling the feature is safe: the
//! fallback is to the CPU EP.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod error;
pub mod generate;
pub mod model;
pub mod session;

pub use error::OnnxError;
pub use generate::{GenerateRequest, GenerateResponse};
pub use model::{ArtifactPaths, GenAiConfig};
pub use session::{ExecutionLane, OnnxSession};
