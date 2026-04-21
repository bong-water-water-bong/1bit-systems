//! Back-compat shim for the old `cpu_lane` module.
//!
//! Canonical home moved to [`crate::sampler`] in 2026-04-20 when the
//! sampler offload was promoted off the iGPU critical path. The
//! `cpu_lane` name is kept alive because:
//!
//! * `crates/1bit-server/benches/sampler.rs` imports
//!   `onebit_router::cpu_lane::CpuLane` to run the sampler cost probe.
//! * `crates/1bit-server/tests/sampler_cost_probe.rs` does the same.
//!
//! Both of those are on the "do NOT touch" list in the scaffolding
//! brief. Re-exporting keeps them compiling without edits. New code
//! inside `1bit-router` should import from `crate::sampler::*`.
//!
//! Everything here is a pure re-export — no original items.

pub use crate::sampler::cpu::{
    CPU_THREADS_ENV, CpuLane, CpuLaneError, CpuSampler, PipelinedOutcome,
    RESERVED_CORES_FOR_IGPU_COORD, default_thread_count, global_lane,
};
pub use crate::sampler::{SAMPLER_MODE_ENV, SamplerMode, sampler_mode_from_env};
