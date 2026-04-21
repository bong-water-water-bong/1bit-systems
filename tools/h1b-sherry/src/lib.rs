//! h1b-sherry — offline requantizer: TQ1 v4 (halo-1bit v4) -> Sherry v3
//! with the `H1B_FLAG_SHERRY_FP16` bit set so the runtime dispatches to
//! `sherry_ternary_gemv_launch` (clean-room fp16 path).
//!
//! ## What this tool does
//!
//! Reads an input `.h1b` file in [`H1bWeightFormat::TQ1V4`] format, decodes
//! every ternary tensor back to `{-1, 0, +1}` at row granularity, then
//! repacks each row under Sherry's 3:4-sparse, 5-bit-per-group encoding
//! (`rcpp_sherry_pack` layout — matches `rocm-cpp/include/rocm_cpp/sherry.h`
//! and the `pack_sherry_v3` reference in `bench_sherry.cpp`).
//!
//! ## Zero-position heuristic
//!
//! Sherry's contract is that every group of 4 weights contains exactly one
//! zero (3:4 sparsity). TQ1 weights are lossless-ternary — a group can have
//! 0, 1, 2, 3, or 4 zeros. The deterministic rule this tool uses:
//!
//!   1. If the group has ≥ 1 zero, pick the **lowest-index zero** as
//!      `zero_pos`. Remaining three lanes stay as their true ±1 / 0 signs
//!      (any spare zeros get their sign flipped to `+1` at pack time; this
//!      is the lossy frontier — signalled via the "positions-changed-sign"
//!      counter the round-trip test gates on).
//!   2. If the group has **no zero** (all ±1), pick the lane whose
//!      **sign-magnitude is smallest** as `zero_pos` — since every lane has
//!      magnitude 1, this collapses to "lowest-index wins", matching the
//!      `pack_sherry_v3` fallback `zero_pos` bug described in
//!      `rocm-cpp/include/rocm_cpp/sherry.h`:
//!
//!    > "if a group has 0 or >=2 zeros this function packs deterministically
//!    >  by picking the LOWEST-INDEX zero as zero_pos and treating any
//!    >  other zeros as +1 (their weight contribution is then wrong — this
//!    >  is a caller bug, not a packer bug; the zero-choice heuristic lives
//!    >  in the requantizer)."
//!
//! This crate IS that requantizer — i.e. the "caller" the kernel header
//! blames for picking the zero. So our job is to emit **valid**
//! 3:4-sparse groups downstream of our own re-pack (we force one-zero per
//! group by flipping the chosen lane to `0`). The kernel's fallback never
//! trips at run-time on files this tool produced.
//!
//! ## What we don't do
//!
//! * We don't recompute scales. Sherry shares the halo-v3 `[rows] f32`
//!   scale layout, so the original per-row scales are copied verbatim.
//! * We don't touch embedding / final_norm / per-layer norm tensors.
//! * We don't re-train or re-learn zeros. This is a **post-hoc** pack
//!   that intentionally accepts up to 25% accuracy drift per group
//!   (the `--ppl` regression gate in `benchmarks/sherry-ppl.sh` is the
//!   fail-safe).
//!
//! See `project_sherry_spike.md` for the broader rollout plan.

pub mod convert;
pub mod sherry_pack;
pub mod tq1_unpack;

pub use convert::{ConvertStats, convert_file};
pub use sherry_pack::{pack_sherry_row, pick_zero_pos};
pub use tq1_unpack::{tq1_row_bytes, unpack_tq1_row};
