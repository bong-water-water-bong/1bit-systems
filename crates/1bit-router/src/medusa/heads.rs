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

use super::MedusaError;

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
#[derive(Debug, Clone)]
pub struct MedusaHeads {
    /// One slot per head. Scaffold form: each head carries only its
    /// metadata. The retrained-weights pass fills in weight buffers.
    pub heads: [MedusaHead; NUM_MEDUSA_HEADS],
    /// Path the heads were loaded from. Kept for error messages and
    /// `/metrics` reporting.
    pub source_path: PathBuf,
}

impl MedusaHeads {
    /// Scaffold loader. Today this only records the path; the
    /// retrained-weights pass will mmap + parse + upload. Expected
    /// acceptance rates come from `project_medusa_plan.md` + the
    /// upstream model card.
    ///
    /// Returns:
    /// * `Ok(Self)` with all four heads pinned at `path`.
    /// * `Err(WeightsNotFound { path })` if `path` does not exist.
    ///   We re-check the path here even though [`super::MedusaState::from_config`]
    ///   has already verified it — the second check costs one stat and
    ///   keeps this loader safe to call from other code paths.
    pub fn load_stub(path: &Path) -> Result<Self, MedusaError> {
        if !path.exists() {
            return Err(MedusaError::WeightsNotFound {
                path: path.to_path_buf(),
            });
        }
        // Acceptance rates from the MedusaBitNet-2B-4T model card (Alpaca
        // 52K training set). Ops dashboards compare live measurements
        // against these; the follow-up pass may retrain and overwrite.
        const EXPECTED_ACCEPTANCE: [f32; NUM_MEDUSA_HEADS] = [0.630, 0.290, 0.111, 0.046];

        let heads = std::array::from_fn(|i| MedusaHead {
            index: i,
            weights_path: path.to_path_buf(),
            expected_acceptance: EXPECTED_ACCEPTANCE[i],
        });

        Ok(Self {
            heads,
            source_path: path.to_path_buf(),
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
