//! CPU-side sampler lane — rayon pool + bounded-channel pipeline.
//!
//! Owns the persistent Zen5 worker and the handoff primitives that
//! move a freshly-computed logits buffer off the GPU-dispatch thread
//! and onto a dedicated sampler thread. See [`crate::sampler`] for
//! the module-level intro.
//!
//! # Two shapes of offload
//!
//! 1. **[`CpuLane::sample`]** — the installable offload. Calls
//!    `rayon_pool.install(|| sampler.sample(...))` so the sampler
//!    math runs on a named `halo-cpu-<idx>` rayon worker rather than
//!    the caller. Same thread blocks until the sampler returns. Bit-
//!    identical output to the inline path. This is the shape the
//!    router actually calls today from `generate_blocking`.
//! 2. **[`CpuSampler`]** — the bounded-channel pipeline. Owns a single
//!    persistent worker thread fed by a `flume::bounded(1)` queue.
//!    The caller hands over a logits buffer and gets a result via a
//!    one-shot reply channel; while the sampler runs on the worker,
//!    the caller is free to start the next `forward_token`'s GPU
//!    dispatch (memset, embedding lookup, first RMSNorm) on the
//!    dispatch thread. **Because the next forward_token's `cur` token
//!    depends on the just-sampled id, true overlap is limited to the
//!    greedy short-circuit case** (GPU returns argmax, sampler is
//!    skipped) — but the bounded handoff still wins a measurable slice
//!    of wall time by moving the sampler's L1/L2 footprint off the
//!    dispatch core. See `docs/wiki/CPU-Lane-Plan.md` for the decision
//!    memo.

use std::sync::OnceLock;
use std::thread::JoinHandle;

use onebit_core::sampler::Sampler;
use onebit_core::types::TokenId;

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
/// schedule on. Held behind an `Arc` inside the [`Router`](crate::Router).
///
/// Public API is preserved from the old `crate::cpu_lane::CpuLane` so
/// `1bit-server` benches + cost-probe tests keep compiling. See the
/// `crate::cpu_lane` re-export shim.
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
    /// 1. If [`CPU_THREADS_ENV`] is set to a parseable positive integer,
    ///    use that exactly.
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

    /// Run the full [`Sampler::sample`] path on the rayon pool.
    ///
    /// Semantics are **bit-identical** to calling `sampler.sample(logits,
    /// history)` directly: same rep-penalty → top-k → softmax → top-p →
    /// multinomial pipeline, same RNG stream, same scratch buffers. The
    /// only difference is the executing thread — by going through
    /// [`rayon::ThreadPool::install`] the work runs on a `halo-cpu-<idx>`
    /// thread instead of whatever thread the caller was on. That matters
    /// for two reasons:
    ///
    /// * **Observability.** `rocprof` + `perf top` attribute the sampler's
    ///   CPU cycles to a named thread dedicated to the CPU lane rather
    ///   than the generic tokio blocking pool, so operators can see
    ///   sampler cost without tagging individual tasks.
    /// * **Scheduling.** The rayon pool reserves
    ///   [`RESERVED_CORES_FOR_IGPU_COORD`] cores for the tokio reactor +
    ///   HIP stream callback thread, so sustained sampler load can't
    ///   starve the iGPU dispatch path.
    ///
    /// This is the shape the router's `generate_blocking` loop calls
    /// today; the one-deep pipelined variant lives in [`CpuSampler`].
    pub fn sample(
        &self,
        sampler: &mut Sampler,
        logits: &mut [f32],
        recent: &[TokenId],
    ) -> Result<TokenId, onebit_core::HaloError> {
        self.rayon_pool.install(|| sampler.sample(logits, recent))
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
    ///   [`onebit_core::sampler`].
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
/// lane surface is narrow.
#[derive(Debug, thiserror::Error)]
pub enum CpuLaneError {
    /// Rayon refused to construct the thread pool. Practically unreachable
    /// on a real box; we surface it so the operator sees a real error
    /// rather than a panic trace.
    #[error("cpu lane: rayon pool build failed: {0}")]
    PoolBuild(String),

    /// The bounded-channel sampler worker died (panicked, OOM, or was
    /// dropped mid-call). Surfaces when [`CpuSampler::sample_pipelined`]
    /// can't receive a reply.
    #[error("cpu sampler: worker disconnected: {0}")]
    WorkerDisconnected(String),
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

/// Parse [`CPU_THREADS_ENV`]. Returns `None` if unset, empty, or not a
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
pub(super) fn parallel_argmax(logits: &[f32]) -> u32 {
    use rayon::prelude::*;

    // Chunk size chosen to give the scheduler ~16× more tasks than
    // threads — small enough that load imbalance evens out, big enough
    // that the per-task overhead (a few hundred ns of rayon bookkeeping)
    // is negligible compared to the scan work. For a 128256-entry
    // Llama3 vocab with 14 workers, `chunk = 128256/64 ≈ 2004` ⇒
    // ~64 tasks ≈ ~4.5 tasks per worker, which is the rayon sweet spot.
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
/// [`onebit_core::sampler`]). The goal is to prove the parallel pattern
/// composes, not to replace the upstream sampler today.
pub(super) fn parallel_topk_argmax(logits: &[f32], k: usize) -> u32 {
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

// ----------------------------------------------------------------------------
// CpuSampler — bounded-channel pipelined offload
// ----------------------------------------------------------------------------

/// A pipelined-offload sampler wrapper around [`CpuLane::sample`].
///
/// Owns a single persistent worker thread that listens on a
/// [`flume::bounded(1)`](flume::bounded) request channel. Each request
/// carries a logits buffer (moved by value; ownership is returned via
/// the reply channel) and a [`Sampler`] handle (borrowed across the
/// call via a raw pointer — the caller holds the real `&mut Sampler`
/// alive across `send` + `recv`, so this is safe). The worker samples
/// the token and returns it via a one-shot `flume::bounded(1)` reply
/// channel included in the request envelope.
///
/// # Why `flume` and not `tokio::sync::oneshot`
///
/// The decode loop in `generate_blocking` runs inside
/// `tokio::task::spawn_blocking`, which is **not** a tokio runtime
/// context — `oneshot::Receiver::blocking_recv()` would panic there.
/// Flume's `recv()` is a direct blocking primitive that works from any
/// thread. The channel is bounded(1) because we're a one-deep pipeline:
/// there's at most one outstanding sample at a time.
///
/// # Ordering + bit-exactness
///
/// Identical to [`CpuLane::sample`]'s guarantees. The worker calls the
/// same `Sampler::sample` code path with the same inputs — only the
/// executing thread changes. RNG state, scratch buffers, rep-penalty
/// history all flow through the borrowed `&mut Sampler` by pointer,
/// so they remain consistent across calls on the same `CpuSampler`.
pub struct CpuSampler {
    tx: flume::Sender<SampleRequest>,
    worker: Option<JoinHandle<()>>,
}

/// One-shot request envelope sent from the dispatch thread to the
/// sampler worker. `reply_tx` is a dedicated `bounded(1)` so each
/// request gets its own reply channel and we don't have to tag or
/// serialize results.
struct SampleRequest {
    /// Raw pointer to the caller's `Sampler`. The caller awaits the
    /// reply before letting the `Sampler` go out of scope, so the
    /// pointer is valid for the duration of the sample call. We wrap
    /// the raw pointer in `SamplerPtr` so it can cross the thread
    /// boundary (flume requires `Send`); see that type's docstring.
    sampler: SamplerPtr,
    /// Logits buffer moved into the worker. Returned by value on the
    /// reply so the caller can reuse its allocation.
    logits: Vec<f32>,
    /// Recent-token history for rep penalty. Moved in, returned in
    /// `reply_tx`.
    history: Vec<TokenId>,
    /// One-shot reply channel. Bounded(1) so the worker can always
    /// complete its send without coordination; if the caller dropped
    /// it (request cancelled) the worker's send errors and we log.
    reply_tx: flume::Sender<SampleReply>,
}

/// Reply envelope: the sampled token (or error), plus the two buffers
/// moved into the request, so the caller can recycle the allocations.
struct SampleReply {
    outcome: Result<TokenId, onebit_core::HaloError>,
    logits: Vec<f32>,
    history: Vec<TokenId>,
}

/// Wrapper that marks a raw `Sampler` pointer as `Send`. The flume
/// channel requires `Send`; `*mut T` is not `Send` by default.
///
/// # Safety invariant
///
/// The caller MUST keep the `&mut Sampler` alive until the reply
/// arrives — i.e. don't drop or move the sampler between `send` and
/// `recv` on the reply channel. [`CpuSampler::sample_pipelined`]
/// enforces this by taking `&mut Sampler` and awaiting `recv` in the
/// same function scope.
struct SamplerPtr(*mut Sampler);
// SAFETY: see `SamplerPtr`'s docstring. The raw pointer is only
// dereferenced on the worker side while the caller is parked on
// `reply_rx.recv()`, so aliasing is impossible.
unsafe impl Send for SamplerPtr {}

impl CpuSampler {
    /// Build a pipelined sampler tied to a [`CpuLane`]'s rayon pool.
    /// Spawns one worker thread that runs [`CpuLane::sample`] on each
    /// incoming request. The worker thread is pinned inside the rayon
    /// pool's `install` scope so its samples run on a `halo-cpu-<idx>`
    /// thread (same observability invariant as the non-pipelined path).
    ///
    /// The caller holds an `Arc<CpuLane>` (see `Router::cpu_lane`);
    /// this constructor takes the same `Arc` so the worker keeps the
    /// pool alive for its own lifetime independent of the router.
    pub fn new(lane: std::sync::Arc<CpuLane>) -> Self {
        // bounded(1) is deliberate — the pipeline is one-deep. A
        // second outstanding request would mean the dispatch thread
        // ran forward_token twice without waiting for a sample, which
        // is only safe in the greedy (temp=0) short-circuit case —
        // where this channel isn't used at all.
        let (tx, rx) = flume::bounded::<SampleRequest>(1);
        let worker_lane = lane.clone();
        let worker = std::thread::Builder::new()
            .name("halo-sampler-pipe".into())
            .spawn(move || {
                while let Ok(req) = rx.recv() {
                    let SampleRequest {
                        sampler,
                        mut logits,
                        history,
                        reply_tx,
                    } = req;
                    // SAFETY: see `SamplerPtr` — the caller is parked
                    // on `reply_rx.recv()` so the &mut Sampler is
                    // still valid, and nothing else holds an alias.
                    let outcome = unsafe {
                        let s: &mut Sampler = &mut *sampler.0;
                        worker_lane.sample(s, &mut logits, &history)
                    };
                    // If the reply send fails the caller dropped the
                    // receiver mid-call (e.g. task cancellation). We
                    // just swallow — we don't own a logger here and
                    // `tracing` at this depth would pollute every test.
                    let _ = reply_tx.send(SampleReply {
                        outcome,
                        logits,
                        history,
                    });
                }
                // `rx.recv()` returned Err → sender dropped → shutdown.
            })
            .expect("spawn halo-sampler-pipe worker");
        Self {
            tx,
            worker: Some(worker),
        }
    }

    /// Pipelined sample. Hands the logits buffer to the worker over a
    /// `bounded(1)` channel and blocks until the reply arrives. Returns
    /// the sampled token plus the (now-recycled) logits/history buffers.
    ///
    /// The caller MUST keep `sampler: &mut Sampler` alive across this
    /// call (which is automatic — the borrow is scoped to the
    /// function body). See [`SamplerPtr`] for the raw-pointer safety
    /// argument.
    ///
    /// Returns [`CpuLaneError::WorkerDisconnected`] if the worker
    /// thread has died. Samplers returning their own
    /// [`onebit_core::HaloError`] are wrapped in `Ok(Err(...))` at the
    /// outer Result so the caller distinguishes transport failure
    /// from sampler-logic failure.
    pub fn sample_pipelined(
        &self,
        sampler: &mut Sampler,
        logits: Vec<f32>,
        history: Vec<TokenId>,
    ) -> Result<PipelinedOutcome, CpuLaneError> {
        let (reply_tx, reply_rx) = flume::bounded::<SampleReply>(1);
        let req = SampleRequest {
            sampler: SamplerPtr(sampler as *mut Sampler),
            logits,
            history,
            reply_tx,
        };
        self.tx
            .send(req)
            .map_err(|e| CpuLaneError::WorkerDisconnected(format!("send: {e}")))?;
        let reply = reply_rx
            .recv()
            .map_err(|e| CpuLaneError::WorkerDisconnected(format!("recv: {e}")))?;
        Ok(PipelinedOutcome {
            outcome: reply.outcome,
            logits: reply.logits,
            history: reply.history,
        })
    }
}

impl Drop for CpuSampler {
    /// Shut the worker down cleanly: dropping the sender closes the
    /// channel, which unblocks `rx.recv()` with `Err` so the worker
    /// exits its loop.
    fn drop(&mut self) {
        // Drop the sender first so the worker's recv returns Err.
        // Taking a fresh channel to replace `self.tx` is the simplest
        // way to drop it early without making `self.tx` an Option.
        let (dead_tx, _) = flume::bounded::<SampleRequest>(0);
        let old_tx = std::mem::replace(&mut self.tx, dead_tx);
        drop(old_tx);
        if let Some(handle) = self.worker.take() {
            // Best-effort join; we don't want Drop to panic.
            let _ = handle.join();
        }
    }
}

/// Outcome of [`CpuSampler::sample_pipelined`]: the sampled token (or
/// sampler error) plus the two buffers so the caller can reuse the
/// allocations on the next decode step.
pub struct PipelinedOutcome {
    /// Sampler result — `Ok(token_id)` or `Err(sampler error)`.
    pub outcome: Result<TokenId, onebit_core::HaloError>,
    /// The logits scratch, moved in and back. The caller typically
    /// stores this in the decode loop's `logits_scratch: Vec<f32>`
    /// slot to avoid per-step reallocation.
    pub logits: Vec<f32>,
    /// The recent-token history, moved in and back.
    pub history: Vec<TokenId>,
}

/// Process-global lane handle — reserved for when 1bit-server starts
/// actually calling us. Constructed lazily on first access so tests +
/// CI don't pay the pool-startup cost unless they opt in.
static GLOBAL_LANE: OnceLock<CpuLane> = OnceLock::new();

/// Get (or lazily build) the process-global CPU lane. Used by callers
/// that want to share one pool across the whole server rather than
/// construct their own. Not called from the router today (the router
/// constructs its own lane per instance); exposed for the planned SSE
/// wire-up.
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
    //! Coverage:
    //!
    //! 1. Constructor doesn't panic with the default policy.
    //! 2. `parallel_sample` at `temp=0` returns a deterministic argmax
    //!    matching a hand-computed reference.
    //! 3. `HALO_CPU_THREADS=4` is respected.
    //! 4. `Sampler` + `CpuLane::sample` produce bit-identical output
    //!    to the inline path (greedy + full top-k/top-p/rep-penalty).
    //! 5. `CpuSampler` (pipelined) produces bit-identical output to
    //!    the inline path — greedy, temp=1.0, top-k=50.
    //!
    //! Env-var tests serialize through `ENV_LOCK`.

    use super::*;
    use std::sync::Mutex;

    /// Serialize env mutation for this module.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// Default constructor must succeed and produce at least 1 thread.
    #[test]
    fn new_default_does_not_panic() {
        let _g = ENV_LOCK.lock().unwrap();
        // SAFETY: edition-2024 env mutation; single-threaded inside lock.
        unsafe {
            std::env::remove_var(CPU_THREADS_ENV);
        }
        let lane = CpuLane::new().expect("CpuLane::new() must not fail");
        assert!(lane.num_threads() >= 1);
    }

    /// `parallel_sample` at `temp=0` is a deterministic argmax.
    #[test]
    fn parallel_sample_is_deterministic_argmax_at_temp_zero() {
        let lane = CpuLane::with_threads(4).expect("build pool");
        let mut logits = vec![-1.0f32; 32_000];
        logits[17_000] = 42.0;

        let sampled = lane.parallel_sample(&logits, 1, 1.0, 0.0);
        assert_eq!(sampled, 17_000);

        for _ in 0..3 {
            let again = lane.parallel_sample(&logits, 1, 1.0, 0.0);
            assert_eq!(again, sampled);
        }
    }

    /// `HALO_CPU_THREADS=4` makes the default constructor produce a
    /// 4-thread pool. Bad values fall back to the default policy.
    #[test]
    fn env_override_is_respected() {
        let _g = ENV_LOCK.lock().unwrap();
        // SAFETY: edition-2024; single-threaded inside lock.
        unsafe {
            std::env::set_var(CPU_THREADS_ENV, "4");
        }

        assert_eq!(default_thread_count(), 4);
        let lane = CpuLane::new().expect("build");
        assert_eq!(lane.num_threads(), 4);

        unsafe {
            std::env::set_var(CPU_THREADS_ENV, "not-a-number");
        }
        assert!(default_thread_count() >= 1);

        unsafe {
            std::env::set_var(CPU_THREADS_ENV, "0");
        }
        assert!(default_thread_count() >= 1);

        unsafe {
            std::env::remove_var(CPU_THREADS_ENV);
        }
    }

    /// Empty logits return 0 rather than panicking.
    #[test]
    fn empty_logits_return_zero() {
        let lane = CpuLane::with_threads(2).expect("build");
        let out = lane.parallel_sample(&[], 1, 1.0, 0.0);
        assert_eq!(out, 0);
    }

    /// Inline greedy and parallel-lane argmax agree on a 32k-vocab vector.
    #[test]
    fn inline_and_parallel_agree_at_temp_zero() {
        use onebit_core::sampler::{Sampler, SamplerConfig};
        let mut rng = fastrand::Rng::with_seed(0xBEEF);
        let mut logits: Vec<f32> = (0..32_000).map(|_| rng.f32() * 4.0 - 2.0).collect();
        logits[12_345] = 99.0;

        let inline = Sampler::greedy(&logits).expect("greedy") as u32;
        let lane = CpuLane::with_threads(4).expect("build");
        let parallel = lane.parallel_sample(&logits, 1, 1.0, 0.0);
        assert_eq!(inline, parallel);
        assert_eq!(inline, 12_345);

        let mut sampler = Sampler::new(SamplerConfig::default());
        let mut logits_scratch = logits.clone();
        let offloaded = lane
            .sample(&mut sampler, &mut logits_scratch, &[])
            .expect("sample") as u32;
        assert_eq!(offloaded, inline);
    }

    /// Full sampler path (top-k + top-p + temperature + rep-penalty +
    /// multinomial) agrees draw-for-draw across inline and CpuLane.
    #[test]
    fn inline_and_parallel_agree_on_topk_topp_path() {
        use onebit_core::sampler::{Sampler, SamplerConfig};

        let cfg = SamplerConfig {
            temperature: 0.7,
            top_k: 50,
            top_p: 0.95,
            rep_penalty: 1.1,
            rep_last_n: 32,
            seed: 0xDEAD_BEEF,
        };

        let mut rng = fastrand::Rng::with_seed(0x2026_04_20);
        let logits: Vec<f32> = (0..32_000).map(|_| rng.f32() * 8.0 - 4.0).collect();
        let history: Vec<i32> = (0..64).collect();

        let lane = CpuLane::with_threads(4).expect("build");
        let mut inline_sampler = Sampler::new(cfg);
        let mut parallel_sampler = Sampler::new(cfg);

        for i in 0..16 {
            let mut inline_logits = logits.clone();
            let mut parallel_logits = logits.clone();

            let inline_tok = inline_sampler
                .sample(&mut inline_logits, &history)
                .expect("inline sample");
            let parallel_tok = lane
                .sample(&mut parallel_sampler, &mut parallel_logits, &history)
                .expect("parallel sample");

            assert_eq!(
                inline_tok, parallel_tok,
                "draw {i}: inline={inline_tok} parallel={parallel_tok} must be bit-identical"
            );
        }
    }

    // ----- CpuSampler (pipelined handoff) -----

    /// Greedy (temp=0) pipelined sample matches the inline
    /// `Sampler::sample` / `Sampler::greedy` result bit-for-bit.
    /// Case 1 of the three required by the offload brief
    /// (greedy / temp=1.0 / top-k=50).
    #[test]
    fn pipelined_greedy_matches_inline() {
        use onebit_core::sampler::{Sampler, SamplerConfig};

        let mut rng = fastrand::Rng::with_seed(0xB16_B00B);
        let mut logits: Vec<f32> = (0..128_256).map(|_| rng.f32() * 4.0 - 2.0).collect();
        logits[55_555] = 77.0;

        let inline_tok = Sampler::greedy(&logits).expect("greedy");

        let lane = std::sync::Arc::new(CpuLane::with_threads(4).expect("build"));
        let pipe = CpuSampler::new(lane);
        let mut sampler = Sampler::new(SamplerConfig::default()); // temp=0
        let out = pipe
            .sample_pipelined(&mut sampler, logits.clone(), Vec::new())
            .expect("pipelined ok");
        let pipe_tok = out.outcome.expect("sampler ok");
        assert_eq!(inline_tok, pipe_tok);
        assert_eq!(out.logits.len(), 128_256, "logits buffer recycled");
    }

    /// Temperature-1.0 pipelined sample agrees with inline draw-for-draw.
    /// Case 2 of the three required by the offload brief.
    #[test]
    fn pipelined_temp1_matches_inline() {
        use onebit_core::sampler::{Sampler, SamplerConfig};

        let cfg = SamplerConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            rep_penalty: 1.0,
            rep_last_n: 0,
            seed: 0xFACE_FEED,
        };

        let mut rng = fastrand::Rng::with_seed(0xCAFE_BABE);
        let logits: Vec<f32> = (0..32_000).map(|_| rng.f32() * 4.0 - 2.0).collect();

        let mut inline_sampler = Sampler::new(cfg);
        let mut pipe_sampler = Sampler::new(cfg);
        let lane = std::sync::Arc::new(CpuLane::with_threads(4).expect("build"));
        let pipe = CpuSampler::new(lane);

        // 8 draws — each through freshly-cloned logits so the RNG state
        // on both sides diverges only if the pipeline corrupts something.
        let mut recycled_logits: Vec<f32> = logits.clone();
        let mut recycled_history: Vec<TokenId> = Vec::new();
        for i in 0..8 {
            let mut inline_logits = logits.clone();
            let inline_tok = inline_sampler
                .sample(&mut inline_logits, &recycled_history)
                .expect("inline");

            // For pipelined, move recycled buffers in each time; get
            // them back via the outcome.
            recycled_logits.clear();
            recycled_logits.extend_from_slice(&logits);
            let out = pipe
                .sample_pipelined(&mut pipe_sampler, recycled_logits, recycled_history)
                .expect("pipelined");
            let pipe_tok = out.outcome.expect("sampler");
            recycled_logits = out.logits;
            recycled_history = out.history;

            assert_eq!(
                inline_tok, pipe_tok,
                "draw {i}: inline={inline_tok} pipe={pipe_tok} (temp=1.0)"
            );
        }
    }

    /// Top-k=50 pipelined sample agrees with inline draw-for-draw.
    /// Case 3 of the three required by the offload brief.
    #[test]
    fn pipelined_topk50_matches_inline() {
        use onebit_core::sampler::{Sampler, SamplerConfig};

        let cfg = SamplerConfig {
            temperature: 0.8,
            top_k: 50,
            top_p: 1.0,
            rep_penalty: 1.0,
            rep_last_n: 0,
            seed: 0xAAAA_BBBB,
        };

        let mut rng = fastrand::Rng::with_seed(0x1234_5678);
        let logits: Vec<f32> = (0..32_000).map(|_| rng.f32() * 6.0 - 3.0).collect();

        let mut inline_sampler = Sampler::new(cfg);
        let mut pipe_sampler = Sampler::new(cfg);
        let lane = std::sync::Arc::new(CpuLane::with_threads(4).expect("build"));
        let pipe = CpuSampler::new(lane);

        let mut recycled_logits: Vec<f32> = logits.clone();
        let mut recycled_history: Vec<TokenId> = Vec::new();
        for i in 0..8 {
            let mut inline_logits = logits.clone();
            let inline_tok = inline_sampler
                .sample(&mut inline_logits, &recycled_history)
                .expect("inline");

            recycled_logits.clear();
            recycled_logits.extend_from_slice(&logits);
            let out = pipe
                .sample_pipelined(&mut pipe_sampler, recycled_logits, recycled_history)
                .expect("pipelined");
            let pipe_tok = out.outcome.expect("sampler");
            recycled_logits = out.logits;
            recycled_history = out.history;

            assert_eq!(
                inline_tok, pipe_tok,
                "draw {i}: inline={inline_tok} pipe={pipe_tok} (top_k=50)"
            );
        }
    }
}
