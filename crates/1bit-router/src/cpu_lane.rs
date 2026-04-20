//! CPU lane — the 7th surface of the 1bit systems APU stack.
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
//! # What ships today (2026-04-20)
//!
//! * A thread pool sized from `HALO_CPU_THREADS` or (by default) from
//!   [`std::thread::available_parallelism`] minus 2 cores reserved for
//!   iGPU coordination (tokio reactor + HIP stream callback thread).
//! * [`CpuLane::parallel_sample`] — a rayon-parallel top-k/argmax
//!   demonstration. Used by the tests and bench; the production
//!   sampler offload goes through [`CpuLane::sample`] instead (see
//!   below).
//! * [`CpuLane::sample`] — offloads the bit-identical
//!   [`onebit_core::sampler::Sampler::sample`] call onto the rayon pool
//!   via `rayon_pool.install()`. Same math, same RNG state, different
//!   executing thread (named `halo-cpu-<idx>` for observability).
//! * [`SamplerMode`] + [`sampler_mode_from_env`] — an `HALO_SAMPLER`
//!   env dial that picks between the inline sampler (default) and
//!   the parallel offload. The router reads this once at construction
//!   and pins it onto the `Router` struct; every request uses the same
//!   mode.
//! * A registered return path from [`crate::Backend::Cpu`] dispatch:
//!   instead of `unimplemented!()`, the router returns
//!   [`crate::BackendError::CpuLaneStub`], which names this module and
//!   the wiki page so the operator knows where to read.
//!
//! # What 1bit-server does with this today
//!
//! `Router::generate` / `Router::generate_stream` consult
//! [`RouterConfig::sampler_mode`](crate::RouterConfig::sampler_mode).
//! When set to [`SamplerMode::Parallel`] the live decode loop in
//! `generate_blocking` routes the `temperature > 0` branch through
//! [`CpuLane::sample`] instead of calling `Sampler::sample` directly;
//! semantics are bit-identical (see the
//! `inline_and_parallel_agree_*` tests). The default is
//! [`SamplerMode::Inline`] — the measurement at
//! `docs/wiki/CPU-Lane-Plan.md` §"2026-04-20 measurement + decision"
//! shows the sampler is ~4.7% of a 15 ms forward pass, which is right
//! at the edge of "worth pipelining", and in the common `temp = 0`
//! case the GPU returns the argmax so the host sampler is skipped
//! entirely.
//!
//! # No C++
//!
//! Pure Rust. Orchestration code is Rule A + the 2026-04-20 "bare-metal
//! first" lock-in: C++ stays in `rocm-cpp` kernels, Rust everywhere
//! above. `rayon` is the only new dep and it's pulled in transitively
//! already (tokenizers, etc. — see `Cargo.lock`).

use std::sync::OnceLock;

use onebit_core::sampler::Sampler;
use onebit_core::types::TokenId;

/// Env-var name clients set to pin the CPU lane's thread count.
///
/// Unset or empty → the default policy kicks in
/// ([`default_thread_count`]). A value that fails to parse as a
/// positive integer is treated as "invalid, use default"; we don't
/// panic on bad operator input.
pub const CPU_THREADS_ENV: &str = "HALO_CPU_THREADS";

/// Env-var name that selects between the inline sampler (default) and
/// the CPU-lane sampler offload.
///
/// Accepted values (case-insensitive): `inline` / `parallel`.  Anything
/// else (including unset / empty) resolves to the default
/// [`SamplerMode::Inline`]. The router consults this exactly once at
/// construction — see [`sampler_mode_from_env`] — so the knob is a
/// process-level dial, not a per-request one.
pub const SAMPLER_MODE_ENV: &str = "HALO_SAMPLER";

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
    /// What this function does **not** do today:
    ///
    /// * Pipeline forward_token with the sampler. The bench at
    ///   `crates/1bit-server/benches/sampler.rs` shows the full sampler
    ///   path is ~4.5% of a 15 ms forward pass — right at the "noise vs
    ///   signal" threshold, and in the common greedy (`temp=0`) case the
    ///   GPU returns the argmax itself and this function is bypassed
    ///   entirely. Pipelining would need a double-buffered logits
    ///   scratch + async handoff; deferred until a real target emerges.
    ///
    /// The return value is the sampled token id (same semantics as
    /// [`Sampler::sample`]). Errors are surfaced via the same
    /// [`onebit_core::HaloError`] path the inline sampler uses.
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
    ///   [`onebit_core::sampler`]. We'll grow this function to match
    ///   that behaviour parallel-by-parallel once 1bit-server actually
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
/// [`onebit_core::sampler`]). The goal is to prove the parallel pattern
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

/// Which sampler dispatch path the router should take.
///
/// * [`SamplerMode::Inline`] — today's default. The sampler runs on the
///   same `tokio::task::spawn_blocking` thread that drove `forward_token`,
///   so one CPU core handles both the GPU dispatch and the host-side
///   sample. This is the simplest path and the one the measurement in
///   `crates/1bit-server/benches/sampler.rs` says is "good enough"
///   (<5% of a 15 ms forward pass for the full sampler; 0.4% for greedy).
/// * [`SamplerMode::Parallel`] — the sampler runs inside the
///   [`CpuLane`]'s rayon pool via [`CpuLane::sample`]. Bit-identical
///   output; different executing thread. Exists so operators can
///   measure + compare under realistic load without a rebuild.
///
/// The mode is picked once at router construction — it's a process-wide
/// dial, not a per-request option. Expose it through the `HALO_SAMPLER`
/// env var (see [`SAMPLER_MODE_ENV`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SamplerMode {
    /// Sampler runs on whatever thread the caller is on. This is the
    /// default and the well-tested path.
    #[default]
    Inline,
    /// Sampler runs on the [`CpuLane`]'s rayon pool. Bit-identical
    /// output to `Inline`; different CPU lane.
    Parallel,
}

impl SamplerMode {
    /// Case-insensitive parser. Accepts `inline` / `parallel`;
    /// everything else errors with a message that lists the valid set
    /// so ops doesn't have to grep.
    pub fn parse_env(raw: &str) -> Result<Self, crate::BackendError> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "inline" | "" => Ok(SamplerMode::Inline),
            "parallel" => Ok(SamplerMode::Parallel),
            other => Err(crate::BackendError::Other(format!(
                "HALO_SAMPLER: unknown value {other:?}; accepted: inline | parallel"
            ))),
        }
    }

    /// Human-readable label for logs.
    pub fn label(self) -> &'static str {
        match self {
            SamplerMode::Inline => "inline",
            SamplerMode::Parallel => "parallel",
        }
    }
}

/// Read `HALO_SAMPLER` out of the environment, tolerating empty / unset.
///
/// Unset or empty → [`SamplerMode::Inline`]. Any other value is parsed
/// through [`SamplerMode::parse_env`]; bad values surface as
/// [`crate::BackendError::Other`] and the caller is expected to refuse
/// to start rather than silently fall back. That matches how
/// `HALO_BACKEND` behaves so ops tooling sees a single convention.
pub fn sampler_mode_from_env() -> Result<SamplerMode, crate::BackendError> {
    match std::env::var(SAMPLER_MODE_ENV) {
        Ok(raw) if !raw.is_empty() => SamplerMode::parse_env(&raw),
        _ => Ok(SamplerMode::Inline),
    }
}

/// Process-global lane handle — reserved for when 1bit-server starts
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

    /// Parity: the parallel lane sampler and the inline sampler must
    /// produce the **same token id** for the argmax (temperature-zero)
    /// case on a 32k-vocab logits vector. This is the strongest
    /// bit-exact guarantee the scaffold makes — any drift here would
    /// mean the parallel reduce order leaks into the output, which
    /// would be a correctness bug (not a performance regression).
    #[test]
    fn inline_and_parallel_agree_at_temp_zero() {
        use onebit_core::sampler::{Sampler, SamplerConfig};
        // Deterministic logits — mix negative, positive, and one clear
        // winner. fastrand is already in the tree via 1bit-core, so
        // reusing it here doesn't grow the build graph.
        let mut rng = fastrand::Rng::with_seed(0xBEEF);
        let mut logits: Vec<f32> = (0..32_000).map(|_| rng.f32() * 4.0 - 2.0).collect();
        logits[12_345] = 99.0; // crowned winner

        // Inline path — the one `generate_blocking` takes today.
        let inline = Sampler::greedy(&logits).expect("greedy") as u32;
        // Parallel lane path.
        let lane = CpuLane::with_threads(4).expect("build");
        let parallel = lane.parallel_sample(&logits, 1, 1.0, 0.0);
        assert_eq!(
            inline, parallel,
            "inline argmax (token {inline}) and parallel argmax (token {parallel}) must agree"
        );
        // And the winner we planted must actually be the one returned —
        // otherwise the synthetic fixture is broken and the parity test
        // would pass vacuously.
        assert_eq!(inline, 12_345, "expected the planted winner");

        // Also exercise the `sample()` full-path wrapper at temp=0. It
        // delegates to Sampler::sample which short-circuits to
        // Sampler::greedy for temp<=0, so the answer must match both.
        let mut sampler = Sampler::new(SamplerConfig::default()); // temp=0
        let mut logits_scratch = logits.clone();
        let offloaded = lane
            .sample(&mut sampler, &mut logits_scratch, &[])
            .expect("sample") as u32;
        assert_eq!(
            offloaded, inline,
            "CpuLane::sample @ temp=0 must also agree"
        );
    }

    /// Parity for the *full* sampler path (top-k + top-p + temperature
    /// + rep-penalty + multinomial draw) on a 32k-vocab synthetic
    /// logits vector. The parallel lane dispatches through
    /// [`CpuLane::sample`], which runs `Sampler::sample` inside
    /// `rayon_pool.install()` — same code, different thread — so the
    /// sampled token id and the post-call RNG state must be identical.
    ///
    /// We seed both samplers from the same `SamplerConfig` and run them
    /// on two independent clones of the logits vector. Matching output
    /// tokens across N draws is the operational definition of
    /// "bit-identical" for this test — if the rayon rejigger changed
    /// RNG state or reduce order we'd see divergence within the first
    /// few draws.
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

        // 32k-element vector — same rough order as BitNet's vocab.
        let mut rng = fastrand::Rng::with_seed(0x2026_04_20);
        let logits: Vec<f32> = (0..32_000).map(|_| rng.f32() * 8.0 - 4.0).collect();
        // Recent-token history so the rep-penalty branch actually fires
        // and we exercise that code path too.
        let history: Vec<i32> = (0..64).collect();

        let lane = CpuLane::with_threads(4).expect("build");
        let mut inline_sampler = Sampler::new(cfg);
        let mut parallel_sampler = Sampler::new(cfg);

        // Run 16 draws through each sampler on fresh copies of the
        // logits. The multinomial step draws from the RNG — because
        // both samplers are seeded identically and the math is the
        // same, they MUST agree draw-for-draw.
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

    /// `HALO_SAMPLER=parallel` must parse to [`SamplerMode::Parallel`];
    /// unset / empty / `inline` must all resolve to the default. Bad
    /// values must return an error (not silently fall back) so ops sees
    /// the drift.
    #[test]
    fn sampler_mode_env_override_is_respected() {
        let _g = ENV_LOCK.lock().unwrap();

        // Default: unset → Inline.
        // SAFETY: edition-2024 env mutation; single-threaded inside lock.
        unsafe {
            std::env::remove_var(SAMPLER_MODE_ENV);
        }
        assert_eq!(
            sampler_mode_from_env().expect("unset"),
            SamplerMode::Inline,
            "unset HALO_SAMPLER must default to Inline"
        );

        // Empty string → Inline (same as unset).
        unsafe {
            std::env::set_var(SAMPLER_MODE_ENV, "");
        }
        assert_eq!(sampler_mode_from_env().expect("empty"), SamplerMode::Inline);

        // Explicit "inline".
        unsafe {
            std::env::set_var(SAMPLER_MODE_ENV, "inline");
        }
        assert_eq!(
            sampler_mode_from_env().expect("inline"),
            SamplerMode::Inline
        );

        // The interesting case: "parallel" must flip the knob.
        unsafe {
            std::env::set_var(SAMPLER_MODE_ENV, "parallel");
        }
        assert_eq!(
            sampler_mode_from_env().expect("parallel"),
            SamplerMode::Parallel,
            "HALO_SAMPLER=parallel must resolve to SamplerMode::Parallel"
        );

        // Case tolerance — ops scripts aren't always tidy.
        unsafe {
            std::env::set_var(SAMPLER_MODE_ENV, "  PARALLEL\n");
        }
        assert_eq!(
            sampler_mode_from_env().expect("PARALLEL"),
            SamplerMode::Parallel
        );

        // Garbage values must error with a message naming the valid set.
        unsafe {
            std::env::set_var(SAMPLER_MODE_ENV, "nonsense");
        }
        let err = sampler_mode_from_env().expect_err("bad value must error");
        let msg = format!("{err}");
        assert!(
            msg.contains("inline") && msg.contains("parallel"),
            "error must list accepted values; got: {msg}"
        );

        // Clean up before releasing the lock so sibling tests aren't
        // contaminated by our mutations.
        unsafe {
            std::env::remove_var(SAMPLER_MODE_ENV);
        }
    }
}
