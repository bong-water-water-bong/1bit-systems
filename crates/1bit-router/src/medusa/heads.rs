//! Four speculative heads that project the backbone's hidden state to
//! per-head logit rows.
//!
//! Scaffolding only — no real weight tensors yet. The retrained-weights
//! pass will populate each head's device-side weight buffer from the
//! mmapped `.h1b` (or equivalent) file.
//!
//! # Head shape (from `parrishcorcoran/MedusaBitNet-2B-4T`
//! `medusa_heads_step2000.pt`)
//!
//! | Field            | Value  |
//! |------------------|--------|
//! | Number of heads  | 4      |
//! | Per-head layers  | 1 residual SiLU-gated block + shared `lm_head` |
//! | Hidden dim       | 2560   |
//! | Output vocab     | 128256 (shared across heads, lives on backbone) |
//! | Per-head tensors | `w_in` + `w_out`, each fp16[2560, 2560] (25 MiB) |
//! | Total head file  | ~100 MiB fp16 |
//! | Acceptance rates | 63.0 / 29.0 / 11.1 / 4.6 % |
//!
//! Heads share the backbone's `lm_head`, so the vocab-projection path is
//! the existing LM-head GEMV. Each head only owns its `w_in` / `w_out`
//! matrices. The forward pass per head is (from the upstream README):
//!
//!   h_out  = h + W_out · SiLU(W_in · h)      // small-M ternary GEMM × 2
//!   logits = backbone.lm_head(h_out)           // existing fp16 GEMV
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
/// * `w_in: DeviceBuffer<u16>` — fp16 pre-SiLU projection, hd × hd.
/// * `w_out: DeviceBuffer<u16>` — fp16 post-SiLU projection, hd × hd.
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

    /// Live host-side projection for a single head.
    ///
    /// Computes `h_out = h + W_out · SiLU(W_in · h)` on the CPU, using
    /// the mmapped fp16 weights from the `.h1b-medusa` file. The
    /// `ternary_gemm_smallm` HIP kernel dispatch is still scaffolding
    /// (`Err(Unsupported)`) today, so this path provides a correct
    /// fallback that lets the Medusa speculative-decode loop run
    /// end-to-end while the kernel is being fitted.
    ///
    /// The heads are **fp16-dense**, not ternary — they're the 13 MiB
    /// `W_in` + `W_out` tensors lifted out of the upstream
    /// `MedusaBitNet-2B-4T` checkpoint. At 2560×2560 × 2 matmuls per
    /// head, a scalar-f32 reduction across the contraction axis lands
    /// in the 10–30 ms range on Zen5 per head — heavy enough that this
    /// path is only suitable for measurement / correctness bring-up.
    /// Moving to the native small-M ternary GEMM (when it grows a
    /// device-pointer wrapper + fp16 variant) is the obvious next step.
    ///
    /// # Inputs
    /// * `hidden` — post-final-norm hidden state for this token,
    ///   `MEDUSA_HIDDEN_DIM` fp16 elements (as `u16`).
    /// * `head_idx` — which head (0..NUM_MEDUSA_HEADS).
    ///
    /// # Output
    /// `h_out` as an owned `Vec<u16>` of length `MEDUSA_HIDDEN_DIM`, fp16.
    pub fn project_one_head_host(
        &self,
        hidden: &[u16],
        head_idx: usize,
    ) -> Result<Vec<u16>, MedusaError> {
        if hidden.len() != MEDUSA_HIDDEN_DIM {
            return Err(MedusaError::BadInput(
                "medusa heads: hidden state length must equal MEDUSA_HIDDEN_DIM (2560)",
            ));
        }
        if head_idx >= NUM_MEDUSA_HEADS {
            return Err(MedusaError::BadInput(
                "medusa heads: head_idx out of range",
            ));
        }
        let file = self.file.as_ref().ok_or(MedusaError::BadInput(
            "medusa heads: no mmapped file (was this a scaffold stub?)",
        ))?;
        let view = file.head(head_idx)?;

        let hd = MEDUSA_HIDDEN_DIM;

        // Decode hidden state once.
        let h_f32: Vec<f32> = hidden
            .iter()
            .map(|&bits| half::f16::from_bits(bits).to_f32())
            .collect();

        // inner = W_in · h  — row-major [hd, hd] × [hd] → [hd]
        let mut inner = vec![0.0f32; hd];
        for row in 0..hd {
            let w_row = &view.w_in[row * hd..(row + 1) * hd];
            let mut acc = 0.0f32;
            for k in 0..hd {
                acc += half::f16::from_bits(w_row[k]).to_f32() * h_f32[k];
            }
            inner[row] = acc;
        }

        // SiLU in-place: silu(x) = x / (1 + exp(-x))
        for v in &mut inner {
            let x = *v;
            *v = x / (1.0 + (-x).exp());
        }

        // h_out = h + W_out · inner
        let mut out_f32 = h_f32.clone();
        for row in 0..hd {
            let w_row = &view.w_out[row * hd..(row + 1) * hd];
            let mut acc = 0.0f32;
            for k in 0..hd {
                acc += half::f16::from_bits(w_row[k]).to_f32() * inner[k];
            }
            out_f32[row] += acc;
        }

        // Quantize back to fp16 for the downstream lm_head GEMV (which
        // takes fp16 inputs on the device side).
        let out_u16: Vec<u16> = out_f32
            .iter()
            .map(|&v| half::f16::from_f32(v).to_bits())
            .collect();

        Ok(out_u16)
    }

    /// Batched variant of [`Self::project_one_head_host`] — runs all
    /// four heads against the same hidden state and returns the
    /// per-head projected hidden `h_out` tensors in index order.
    ///
    /// The four scalar GEMVs are independent; rayon-parallelising them
    /// is an obvious speedup but intentionally left for a follow-up
    /// pass once the accuracy of the projection path is confirmed
    /// against the upstream reference. Today we walk them sequentially.
    pub fn project_all_heads_host(
        &self,
        hidden: &[u16],
    ) -> Result<[Vec<u16>; NUM_MEDUSA_HEADS], MedusaError> {
        let mut out: [Vec<u16>; NUM_MEDUSA_HEADS] =
            std::array::from_fn(|_| Vec::new());
        for i in 0..NUM_MEDUSA_HEADS {
            out[i] = self.project_one_head_host(hidden, i)?;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::loader::{
        MEDUSA_DTYPE_FP16, MEDUSA_FORMAT_VERSION, MEDUSA_HEADER_BYTES, MEDUSA_MAGIC,
        MEDUSA_RESIDUAL_LAYERS, MedusaHeadsFile,
    };
    use std::io::Write;

    /// Hand-rolled round-trip: synthesize a small-shape `.h1b-medusa`
    /// file with all-zero weights, run `project_one_head_host`, and
    /// confirm the output equals the input hidden (W_in·h = 0 → SiLU(0)
    /// = 0 → W_out·0 = 0 → h_out = h).
    #[test]
    fn project_one_head_host_zero_weights_is_identity() {
        // Synthesize a file at the canonical shape (2560) so the loader
        // accepts it. The mmap is sparse-extended with zeros, which is
        // exactly the identity-weight case we want to exercise.
        let num_heads = NUM_MEDUSA_HEADS as u32;
        let hidden_dim = MEDUSA_HIDDEN_DIM as u32;

        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(&MEDUSA_MAGIC).unwrap();
        tmp.write_all(&MEDUSA_FORMAT_VERSION.to_le_bytes()).unwrap();
        tmp.write_all(&num_heads.to_le_bytes()).unwrap();
        tmp.write_all(&hidden_dim.to_le_bytes()).unwrap();
        tmp.write_all(&MEDUSA_RESIDUAL_LAYERS.to_le_bytes()).unwrap();
        tmp.write_all(&MEDUSA_DTYPE_FP16.to_le_bytes()).unwrap();
        tmp.write_all(&[0u8; 40]).unwrap();
        let hd = hidden_dim as usize;
        let per_head = 2 * hd * hd * 2;
        let total = MEDUSA_HEADER_BYTES + per_head * num_heads as usize;
        tmp.as_file_mut().set_len(total as u64).unwrap();
        let path = tmp.path().to_path_buf();

        let file = MedusaHeadsFile::open(&path).expect("synthetic file opens");
        let heads = MedusaHeads {
            heads: std::array::from_fn(|i| MedusaHead {
                index: i,
                weights_path: path.clone(),
                expected_acceptance: EXPECTED_ACCEPTANCE[i],
            }),
            source_path: path,
            file: Some(Arc::new(file)),
        };

        // Hidden state with non-trivial values so an identity check is
        // meaningful (all-zero hidden would pass even on a buggy impl).
        let hidden: Vec<u16> = (0..MEDUSA_HIDDEN_DIM)
            .map(|i| half::f16::from_f32(((i as f32) * 0.01) - 10.0).to_bits())
            .collect();

        let h_out = heads
            .project_one_head_host(&hidden, 0)
            .expect("projection runs against synthetic file");

        assert_eq!(h_out.len(), MEDUSA_HIDDEN_DIM);
        // Zero weights → h_out == h (to fp16 round-trip precision).
        for (i, (&a, &b)) in hidden.iter().zip(h_out.iter()).enumerate() {
            let af = half::f16::from_bits(a).to_f32();
            let bf = half::f16::from_bits(b).to_f32();
            assert!(
                (af - bf).abs() < 1e-3,
                "idx {i}: input {af} output {bf}",
            );
        }
    }

    /// All four heads return a `MEDUSA_HIDDEN_DIM`-length vector under
    /// the batched entry point. Size check only — content identity is
    /// covered by the single-head test above.
    #[test]
    fn project_all_heads_host_returns_four_sized_vectors() {
        let num_heads = NUM_MEDUSA_HEADS as u32;
        let hidden_dim = MEDUSA_HIDDEN_DIM as u32;

        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(&MEDUSA_MAGIC).unwrap();
        tmp.write_all(&MEDUSA_FORMAT_VERSION.to_le_bytes()).unwrap();
        tmp.write_all(&num_heads.to_le_bytes()).unwrap();
        tmp.write_all(&hidden_dim.to_le_bytes()).unwrap();
        tmp.write_all(&MEDUSA_RESIDUAL_LAYERS.to_le_bytes()).unwrap();
        tmp.write_all(&MEDUSA_DTYPE_FP16.to_le_bytes()).unwrap();
        tmp.write_all(&[0u8; 40]).unwrap();
        let hd = hidden_dim as usize;
        let per_head = 2 * hd * hd * 2;
        let total = MEDUSA_HEADER_BYTES + per_head * num_heads as usize;
        tmp.as_file_mut().set_len(total as u64).unwrap();
        let path = tmp.path().to_path_buf();

        let file = MedusaHeadsFile::open(&path).expect("synthetic file opens");
        let heads = MedusaHeads {
            heads: std::array::from_fn(|i| MedusaHead {
                index: i,
                weights_path: path.clone(),
                expected_acceptance: EXPECTED_ACCEPTANCE[i],
            }),
            source_path: path,
            file: Some(Arc::new(file)),
        };

        let hidden = vec![0u16; MEDUSA_HIDDEN_DIM];
        let outs = heads
            .project_all_heads_host(&hidden)
            .expect("batched projection runs");
        for h_out in &outs {
            assert_eq!(h_out.len(), MEDUSA_HIDDEN_DIM);
        }
    }
}
