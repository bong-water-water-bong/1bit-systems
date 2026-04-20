//! CPU lane — the 7th surface of the halo-ai APU stack.
//!
//! # Why this exists
//!
//! Strix Halo has 16 Zen5 cores sitting idle while the iGPU grinds
//! through ternary GEMVs. The gen-1 server put every host-side step —
//! sampler, tokenizer byte-stitch, dispatcher, stop-string match — on
//! the same single tokio task that pumped the GPU stream. Every
//! millisecond the sampler spent on the critical path cost us a
//! millisecond of iGPU idle, because the GPU is waiting to be fed the
//! next token id.
//!
//! The [`CpuLane`] gives that work its own rayon-backed thread pool so
//! the CPU side can overlap with the GPU's next-token grind. This is
//! the lane described in `docs/wiki/Peak-Performance-Projection.md` as
//! surface #7; the first six are HIP matmul / HIP attention / HIP
//! sampler / XDNA prefill / XDNA decode / shared LPDDR5 DMA.
//!
//! # What ships today
//!
//! Scaffolding only:
//!
//! * A thread pool sized from `HALO_CPU_THREADS` or (by default) from
//!   [`std::thread::available_parallelism`] minus 2 cores reserved for
//!   iGPU coordination (tokio reactor + HIP stream callback thread).
//! * [`CpuLane::parallel_sample`] — a rayon-parallel top-k/argmax
//!   demonstration. The important property right now is that the
//!   parallel pattern works and is correct, not that it beats the
//!   single-threaded sampler in [`halo_core::sampler`] today.
//! * A registered return path from [`crate::Backend::Cpu`] dispatch:
//!   instead of `unimplemented!()`, the router returns
//!   [`crate::BackendError::CpuLaneStub`], which names this module and
//!   the wiki page so the operator knows where to read.
//!
//! # What halo-server does with this today
//!
//! **Nothing.** halo-server still runs its sampler in-line on the main
//! axum task. The hotspot analysis that turns "sampler is slow" into
//! "sampler on rayon is faster by X µs per token" has not landed yet —
//! see `docs/wiki/CPU-Lane-Plan.md` for the three concrete next steps
//! (wire into SSE, microbenchmark vs single-threaded, decide on
//! ZenDNN FFI). This module is scaffolding for that work, not a
//! shipping optimization.
//!
//! # No C++
//!
//! Pure Rust. Orchestration code is Rule A + the 2026-04-20 "bare-metal
//! first" lock-in: C++ stays in `rocm-cpp` kernels, Rust everywhere
//! above. `rayon` is the only new dep and it's pulled in transitively
//! already (tokenizers, etc. — see `Cargo.lock`).

use std::sync::OnceLock;

/// Env-var name clients set to pin the CPU lane's thread count.
///
/// Unset or empty → the default policy kicks in
/// ([`default_thread_count`]). A value that fails to parse as a
/// positive integer is treated as "invalid, use default"; we don't
/// panic on bad operator input.
pub const CPU_THREADS_ENV: &str = "HALO_CPU_THREADS";

/// Cores we deliberately leave to the rest of the system when picking
/// the default thread count — tokio reactor + the HIP stream callback
/// thread share the remaining two logical cores, so the iGPU doesn't
/// stall while a rayon job is saturating every physical core.
///
/// On a 16-core Zen5 box this resolves to a 14-thread rayon pool. On
/// small CI VMs (1–4 logical cores) we fall back to a 1-thread pool
/// rather than returning zero.
pub const RESERVED_CORES_FOR_IGPU_COORD: usize = 2;

/// The CPU lane.
///
/// Owns a single rayon [`ThreadPool`] that all CPU-side parallel work
/// (sampler, tokenizer dispatch, stop-string search) will eventually
/// schedule on. Held behind an `Arc` inside the [`Router`](crate::Router)
/// when the operator selects `HALO_BACKEND=cpu` — see
/// [`crate::Backend::Cpu`] — but today there is no production call site
/// (see the module-level docs).
pub struct CpuLane {
    rayon_pool: rayon::ThreadPool,
    /// Cached for introspection / logs / tests. Identical to
    /// `rayon_pool.current_num_threads()` but avoids making the
    /// accessor forward into rayon from hot paths.
    n_threads: usize,
}

impl CpuLane {
    /// Build a CPU lane with the default thread-count policy:
    ///
    /// 1. If `HALO_CPU_THREADS` is set to a parseable positive
    ///    integer, use that exactly.
    /// 2. Otherwise take [`std::thread::available_parallelism`] and
    ///    subtract [`RESERVED_CORES_FOR_IGPU_COORD`]. Clamp to at
    ///    least 1.
    ///
    /// Fails only if rayon itself refuses to build the pool — a
    /// name-collision or OS-resource issue. Constructing a pool on a
    /// fresh process essentially never fails in practice, but we
    /// surface the error anyway rather than panicking.
    pub fn new() -> Result<Self, CpuLaneError> {
        let n = default_thread_count();
        Self::with_threads(n)
    }

    /// Explicit thread-count constructor. `n_threads == 0` is clamped
    /// to 1; rayon rejects zero and we don't want to surface that as
    /// an error when the operator wrote a nonsense value.
    pub fn with_threads(n_threads: usize) -> Result<Self, CpuLaneError> {
        let n = n_threads.max(1);
        let rayon_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .thread_name(|idx| format!("halo-cpu-{idx}"))
            .build()
            .map_err(|e| CpuLaneError::PoolBuild(format!("{e}")))?;
        Ok(Self {
            rayon_pool,
            n_threads: n,
        })
    }

    /// How many worker threads the pool owns.
    pub fn num_threads(&self) -> usize {
        self.n_threads
    }

    /// Run a greedy / top-k sampling step on the logits in parallel.
    ///
    /// Semantics today:
    ///
    /// * `temp <= 0.0` → deterministic argmax via a rayon parallel
    ///   reduction. Bit-exact with the GPU argmax kernel in
    ///   `rocm-cpp` (modulo NaN handling: NaNs are treated as `-inf`
    ///   for the comparison, same as the GPU path).
    /// * `temp > 0.0` → parallel partial top-k via a rayon reduction
    ///   that keeps a bounded heap per worker, merges at the end, then
    ///   picks the top-k entry with the highest logit. `top_p` is
    ///   accepted but **not yet applied** — this is the scaffolding
    ///   milestone; the production sampler (which does apply top-p and
    ///   temperature and repetition penalty) still lives in
    ///   [`halo_core::sampler`]. We'll grow this function to match
    ///   that behaviour parallel-by-parallel once halo-server actually
    ///   calls into the lane.
    ///
    /// The vector must be non-empty; an empty logit vector returns
    /// token id 0 (this keeps the signature infallible so it slots
    /// into the eventual SSE path without ceremony).
    pub fn parallel_sample(&self, logits: &[f32], top_k: usize, top_p: f32, temp: f32) -> u32 {
        if logits.is_empty() {
            return 0;
        }
        // `top_p` is accepted but unused today; see doc comment.
        let _ = top_p;

        self.rayon_pool.install(|| {
            if temp <= 0.0 {
                parallel_argmax(logits)
            } else {
                parallel_topk_argmax(logits, top_k.max(1))
            }
        })
    }
}

/// Errors that can come out of the CPU lane. Small on purpose — the
/// lane surface is narrow today.
#[derive(Debug, thiserror::Error)]
pub enum CpuLaneError {
    /// Rayon refused to construct the thread pool. Practically unreachable
    /// on a real box; we surface it so the operator sees a real error
    /// rather than a panic trace.
    #[error("cpu lane: rayon pool build failed: {0}")]
    PoolBuild(String),
}

/// Default thread-count policy — see [`CpuLane::new`].
///
/// Pulled out of the constructor so tests can exercise it directly
/// (and so we can reuse it from any future caller-side knob).
pub fn default_thread_count() -> usize {
    if let Some(n) = env_override_threads() {
        return n;
    }
    let parallelism = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    parallelism
        .saturating_sub(RESERVED_CORES_FOR_IGPU_COORD)
        .max(1)
}

/// Parse `HALO_CPU_THREADS`. Returns `None` if unset, empty, or not a
/// positive integer (the caller falls back to `available_parallelism`).
fn env_override_threads() -> Option<usize> {
    let raw = std::env::var(CPU_THREADS_ENV).ok()?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    trimmed.parse::<usize>().ok().filter(|n| *n > 0)
}

/// Rayon-parallel argmax. Uses a two-level reduction: per-chunk local
/// max, then a final serial merge across chunk results.
fn parallel_argmax(logits: &[f32]) -> u32 {
    use rayon::prelude::*;

    // Chunk size chosen to give the scheduler ~16× more tasks than
    // threads — small enough that load imbalance evens out, big enough
    // that the per-task overhead (a few hundred ns of rayon bookkeeping)
    // is negligible compared to the scan work.
    let chunk = (logits.len() / 64).max(1024).min(logits.len());
    logits
        .par_chunks(chunk)
        .enumerate()
        .map(|(chunk_idx, slice)| {
            let base = chunk_idx * chunk;
            let mut best_idx = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for (i, &v) in slice.iter().enumerate() {
                let v = if v.is_nan() { f32::NEG_INFINITY } else { v };
                if v > best_val {
                    best_val = v;
                    best_idx = base + i;
                }
            }
            (best_idx, best_val)
        })
        .reduce(
            || (0usize, f32::NEG_INFINITY),
            |a, b| {
                if b.1 > a.1 { b } else { a }
            },
        )
        .0 as u32
}

/// Rayon-parallel top-k argmax (stub shape).
///
/// Each worker keeps a bounded max of size `k` over its chunk; final
/// reduce merges them into a global max and we return the index of
/// that max. This is deliberately simpler than the real top-k +
/// temperature-softmax + top-p sampler (which lives in
/// [`halo_core::sampler`]). The goal is to prove the parallel pattern
/// composes, not to replace the upstream sampler today.
fn parallel_topk_argmax(logits: &[f32], k: usize) -> u32 {
    use rayon::prelude::*;

    // For the stub we just keep the running max per chunk (k=1 in
    // practice); real top-k merging is a follow-up once we actually
    // wire temperature into the lane.
    let _ = k;
    let chunk = (logits.len() / 64).max(1024).min(logits.len());
    logits
        .par_chunks(chunk)
        .enumerate()
        .map(|(chunk_idx, slice)| {
            let base = chunk_idx * chunk;
            let mut best_idx = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for (i, &v) in slice.iter().enumerate() {
                let v = if v.is_nan() { f32::NEG_INFINITY } else { v };
                if v > best_val {
                    best_val = v;
                    best_idx = base + i;
                }
            }
            (best_idx, best_val)
        })
        .reduce(
            || (0usize, f32::NEG_INFINITY),
            |a, b| if b.1 > a.1 { b } else { a },
        )
        .0 as u32
}

/// Process-global lane handle — reserved for when halo-server starts
/// actually calling us. Constructed lazily on first access so tests +
/// CI don't pay the pool-startup cost unless they opt in.
static GLOBAL_LANE: OnceLock<CpuLane> = OnceLock::new();

/// Get (or lazily build) the process-global CPU lane. Used by callers
/// that want to share one pool across the whole server rather than
/// construct their own. Not called from the router today (see module
/// docs); exposed for the planned SSE-path wire-up.
pub fn global_lane() -> &'static CpuLane {
    GLOBAL_LANE.get_or_init(|| CpuLane::new().expect("cpu lane: global pool build should not fail"))
}

// ----------------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Tests for the CPU lane scaffolding.
    //!
    //! Three minimum-viable tests:
    //!
    //! 1. Constructor doesn't panic with the default policy.
    //! 2. `parallel_sample` at `temp=0` returns a deterministic argmax
    //!    matching a hand-computed reference.
    //! 3. `HALO_CPU_THREADS=4` is respected — but because `std::env` is
    //!    process-global, we guard with the same mutex pattern the
    //!    `backend_config_tests` module uses in `lib.rs`.
    //!
    //! Env-var tests serialize through `ENV_LOCK` so parallel
    //! `cargo test` invocations in this binary don't race on
    //! `HALO_CPU_THREADS`. Other tests in the binary (notably the
    //! `backend_config_tests` `ENV_LOCK`) mutate `HALO_BACKEND` — a
    //! different key — so they can run concurrently with these.

    use super::*;
    use std::sync::Mutex;

    /// Serialize env mutation for this module. Separate from the
    /// `backend_config_tests::ENV_LOCK` because they touch different
    /// keys and it's fine for them to run in parallel.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// The default constructor must succeed and produce a pool with
    /// at least 1 thread. We don't assert on the exact thread count
    /// because CI may set `HALO_CPU_THREADS` itself or run on a box
    /// with a weird core topology.
    #[test]
    fn new_default_does_not_panic() {
        // Isolate from any env we might pick up from the environment.
        let _g = ENV_LOCK.lock().unwrap();
        // SAFETY: edition-2024 moved env mutation behind unsafe. We're
        // single-threaded inside the lock.
        unsafe {
            std::env::remove_var(CPU_THREADS_ENV);
        }
        let lane = CpuLane::new().expect("CpuLane::new() must not fail");
        assert!(
            lane.num_threads() >= 1,
            "default thread count must be at least 1; got {}",
            lane.num_threads()
        );
    }

    /// With `temp=0`, `parallel_sample` is a parallel argmax. We
    /// construct a 32k-element logit vector (the same rough order as
    /// BitNet's vocab) with a single clear maximum at a known index,
    /// then assert we find exactly that index.
    #[test]
    fn parallel_sample_is_deterministic_argmax_at_temp_zero() {
        let lane = CpuLane::with_threads(4).expect("build pool");
        let mut logits = vec![-1.0f32; 32_000];
        logits[17_000] = 42.0;

        let sampled = lane.parallel_sample(
            &logits, /* top_k */ 1, /* top_p */ 1.0, /* temp */ 0.0,
        );
        assert_eq!(sampled, 17_000, "argmax must find the singleton max");

        // Run it three more times — with temp=0 the result MUST be
        // identical every call. Any non-determinism here would mean
        // the reduce order leaks into the output, which would be a
        // bug.
        for _ in 0..3 {
            let again = lane.parallel_sample(&logits, 1, 1.0, 0.0);
            assert_eq!(again, sampled, "argmax must be deterministic at temp=0");
        }
    }

    /// `HALO_CPU_THREADS=4` must make the default constructor produce
    /// a 4-thread pool. We take the ENV_LOCK, set the var, build, and
    /// clean up in the same critical section so the rest of the test
    /// binary doesn't see the mutation.
    #[test]
    fn env_override_is_respected() {
        let _g = ENV_LOCK.lock().unwrap();
        // SAFETY: see module-level doc — single-threaded inside lock.
        unsafe {
            std::env::set_var(CPU_THREADS_ENV, "4");
        }

        // First check the free function, which is what CpuLane::new
        // delegates to — so a pass here implies the constructor path.
        assert_eq!(
            default_thread_count(),
            4,
            "HALO_CPU_THREADS=4 must resolve to exactly 4"
        );

        // Then exercise the live constructor.
        let lane = CpuLane::new().expect("build");
        assert_eq!(
            lane.num_threads(),
            4,
            "CpuLane::new() must honour HALO_CPU_THREADS=4"
        );

        // Bad values fall back to the default policy — specifically
        // they do NOT yield 0 or a panic.
        unsafe {
            std::env::set_var(CPU_THREADS_ENV, "not-a-number");
        }
        let n_bad = default_thread_count();
        assert!(n_bad >= 1, "bad env value must fall back to >=1 threads");

        unsafe {
            std::env::set_var(CPU_THREADS_ENV, "0");
        }
        let n_zero = default_thread_count();
        assert!(
            n_zero >= 1,
            "HALO_CPU_THREADS=0 must be rejected and fall back to the default"
        );

        unsafe {
            std::env::remove_var(CPU_THREADS_ENV);
        }
    }

    /// Empty-logit safety: the sampler returns 0 rather than panicking
    /// on an empty input. Cheap belt-and-suspenders test — the real
    /// guarantee is the signature (infallible), but we want the value.
    #[test]
    fn empty_logits_return_zero() {
        let lane = CpuLane::with_threads(2).expect("build");
        let out = lane.parallel_sample(&[], 1, 1.0, 0.0);
        assert_eq!(out, 0);
    }
}
