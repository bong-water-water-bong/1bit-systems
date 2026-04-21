//! Four speculative heads that project the backbone's hidden state to
//! per-head logit rows.
//!
//! Scaffolding only — no real weight tensors yet. The retrained-weights
//! pass will populate each head's device-side weight buffer from the
//! mmapped `.h1b` (or equivalent) file.
//!
//! # Head shape (from `parrishcorcoran/MedusaBitNet-2B-4T`)
//!
//! From `docs/wiki/Medusa-Prototype-Status.md §1`:
//!
//! | Field            | Value  |
//! |------------------|--------|
//! | Number of heads  | 4      |
//! | Per-head layers  | 1 residual MLP block + shared `lm_head` |
//! | Hidden dim       | 2560   |
//! | Output vocab     | 128256 (shared across heads) |
//! | Per-head params  | ~3.3 MB fp16 residual block |
//! | Total head file  | 13 MB fp16 |
//! | Acceptance rates | 63.0 / 29.0 / 11.1 / 4.6 % |
//!
//! Heads share the backbone's `lm_head`, so the vocab-projection path is
//! the existing LM-head GEMV. Each head only owns its residual block.
//! The forward pass per head is:
//!
//!   hidden' = head.residual(hidden)          // small-M ternary GEMM
//!   logits  = shared_lm_head.project(hidden')  // existing fp16 GEMV
//!
//! Today this file scaffolds the head *type* — the residual-block
//! dispatch lands in the follow-up pass.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::MedusaConfig;
use super::MedusaError;
use super::loader::MedusaHeadsFile;

/// Number of speculative heads we plan to load. Pinned by the
/// `MedusaBitNet-2B-4T` artifact; changing this without also
/// retraining the heads is an ABI break.
pub const NUM_MEDUSA_HEADS: usize = 4;

/// Per-head hidden dimension. Matches the Microsoft `bitnet-b1.58-2B-4T`
/// backbone — a head whose dim doesn't match the backbone is a bug.
pub const MEDUSA_HIDDEN_DIM: usize = 2560;

/// A single speculative head — scaffolding form.
///
/// The retrained-weights pass adds two device-side fields:
/// * `residual_weights: DeviceBuffer<u8>` — packed ternary (halo-1bit).
/// * `residual_scales: DeviceBuffer<f32>` — per-row weight scale.
///
/// For now the head just remembers which positional index it occupies
/// (0 = t+1, 1 = t+2, ...) and the path its weights will be loaded from.
/// Both fields are used by the scaffold's error reporting; the `expected`
/// acceptance rate is carried for telemetry so ops dashboards can compare
/// the measured per-head acceptance against the upstream card values
/// without re-reading this comment.
#[derive(Debug, Clone)]
pub struct MedusaHead {
    /// Head index (0 = t+1, 1 = t+2, 2 = t+3, 3 = t+4).
    pub index: usize,
    /// Path the weights will be loaded from. Shared across all four
    /// heads today — they live in one file per the upstream artifact.
    pub weights_path: PathBuf,
    /// Expected acceptance rate from the model card (0.0–1.0). Used
    /// for telemetry and sanity-checking the live measurements. The
    /// retrained-weights pass can either keep or overwrite these
    /// depending on the training set.
    pub expected_acceptance: f32,
}

/// The four heads, loaded (or stubbed) together. Array-of-structs so
/// the router's hot-path can iterate in head-index order without a
/// hashmap lookup. Fixed-size [`NUM_MEDUSA_HEADS`] so the borrow
/// checker catches a missing head at compile time.
///
/// After the real-loader pass, `file` holds the mmapped
/// [`MedusaHeadsFile`] wrapped in an `Arc` so the router can share one
/// parse across its backends without re-mmapping. The `#[cfg(test)]`
/// [`MedusaHeads::load_stub`] constructor leaves `file` as `None` for
/// the existing scaffold tests that only exercise path-bookkeeping.
#[derive(Debug, Clone)]
pub struct MedusaHeads {
    /// One slot per head. Metadata only — tensor data lives inside
    /// `file` if the real loader produced this handle.
    pub heads: [MedusaHead; NUM_MEDUSA_HEADS],
    /// Path the heads were loaded from. Kept for error messages and
    /// `/metrics` reporting.
    pub source_path: PathBuf,
    /// The mmapped source file, if the real loader produced it.
    /// `None` only for the `#[cfg(test)]` path-bookkeeping stub.
    /// `Arc` so the router can share one parse across backends and
    /// `MedusaHeads` keeps its `Clone` impl.
    pub file: Option<Arc<MedusaHeadsFile>>,
}

/// Per-head acceptance rates from the `MedusaBitNet-2B-4T` model card
/// (Alpaca 52K training set). Ops dashboards compare live measurements
/// against these; a retrain may overwrite them.
const EXPECTED_ACCEPTANCE: [f32; NUM_MEDUSA_HEADS] = [0.630, 0.290, 0.111, 0.046];

impl MedusaHeads {
    /// Real loader: mmap + parse a `.h1b-medusa` file, validate its
    /// header against the scaffold's pinned constants, and return a
    /// handle whose per-head tensor views are borrowed directly from
    /// the mmap (zero-copy).
    ///
    /// Config is accepted today but unused beyond shape validation —
    /// the [`MedusaConfig`] struct only carries the path, which is
    /// passed separately. The parameter is here so a future revision
    /// that adds shape overrides (e.g. a retrained-vocab field) can
    /// thread them in without changing every call site.
    ///
    /// Returns:
    /// * `Ok(Self)` with all four heads + an owning
    ///   [`Arc<MedusaHeadsFile>`] so the router can share this parse
    ///   across backends.
    /// * `Err(WeightsNotFound { path })` if `path` doesn't exist.
    /// * `Err(LoaderError(..))` for bad magic, version mismatch,
    ///   shape mismatch, file-size mismatch. See [`MedusaHeadsFile::open`].
    pub fn load(path: &Path, _config: &MedusaConfig) -> Result<Self, MedusaError> {
        let file = MedusaHeadsFile::open(path)?;
        let heads = std::array::from_fn(|i| MedusaHead {
            index: i,
            weights_path: path.to_path_buf(),
            expected_acceptance: EXPECTED_ACCEPTANCE[i],
        });
        Ok(Self {
            heads,
            source_path: path.to_path_buf(),
            file: Some(Arc::new(file)),
        })
    }

    /// Scaffold loader — path-bookkeeping only, no mmap, no parse.
    /// Kept behind `#[cfg(test)]` for parity with the pre-loader
    /// scaffold. Real callers must use [`Self::load`]; the main
    /// `mod.rs::tests` suite goes through [`super::MedusaState::from_config`]
    /// which dispatches to `load`. `allow(dead_code)` because the
    /// retained-but-internal-only form keeps the scaffold's shape on
    /// record without triggering a warn-on-warnings CI gate.
    #[cfg(test)]
    #[allow(dead_code)]
    pub fn load_stub(path: &Path) -> Result<Self, MedusaError> {
        if !path.exists() {
            return Err(MedusaError::WeightsNotFound {
                path: path.to_path_buf(),
            });
        }
        let heads = std::array::from_fn(|i| MedusaHead {
            index: i,
            weights_path: path.to_path_buf(),
            expected_acceptance: EXPECTED_ACCEPTANCE[i],
        });

        Ok(Self {
            heads,
            source_path: path.to_path_buf(),
            file: None,
        })
    }

    /// The heads in index order (t+1, t+2, t+3, t+4). Convenience
    /// accessor so the verifier doesn't reach into the array directly.
    pub fn as_slice(&self) -> &[MedusaHead; NUM_MEDUSA_HEADS] {
        &self.heads
    }

    /// Scaffold projection: run each head's residual block against the
    /// given hidden state. Returns [`MedusaError::BadInput`] today —
    /// the small-M ternary GEMM is wired up in `1bit-hip` but the
    /// device-pointer dispatch form (from loaded weights, not host
    /// slices) lands in the follow-up pass.
    ///
    /// Placeholder signature for call-site compile-checking; expect this
    /// to grow a `&HipBackend` borrow and a `&mut [u16]` per-head
    /// logit buffer once the live dispatch lands.
    pub fn project_stub(&self, hidden: &[u16]) -> Result<(), MedusaError> {
        if hidden.len() != MEDUSA_HIDDEN_DIM {
            return Err(MedusaError::BadInput(
                "medusa heads: hidden state length must equal MEDUSA_HIDDEN_DIM (2560)",
            ));
        }
        // Scaffolding: no real dispatch. The follow-up pass replaces
        // this body with per-head `ternary_gemm_smallm` calls.
        Err(MedusaError::BadInput(
            "medusa heads: live projection not wired (scaffold only) — awaiting retrained weights",
        ))
    }
}
