//! 1bit-router — the hardware-aware inference dispatcher for 1bit systems.
//!
//! What this crate does:
//!
//! * At construction it **picks one** backend family (HIP on gfx1151 /
//!   MLX on Apple / error otherwise — see [`detect`]).
//! * It mmaps the `.h1b` model + parses the `.htok` tokenizer up front
//!   via `1bit-core`, then hands the weight bytes to the backend (for
//!   HIP: `1bit-hip` uploads them once to the GPU).
//! * It exposes a small async surface — `generate`, `generate_stream`,
//!   `list_models` — that the HTTP layer wraps in an `InferenceBackend`
//!   adapter. **The trait itself lives in `1bit-server`** so this crate
//!   stays free of HTTP types; see 1bit-server's `backend::RealBackend`
//!   for the adapter.
//!
//! Layering:
//!
//! ```text
//!      1bit-server (HTTP)  ──▶  onebit_router::Router
//!                                      │
//!                                      ├── detect() → BackendKind
//!                                      ├── HipBackend (weights + forward pass)
//!                                      └── onebit_core::Sampler (host-side)
//! ```
//!
//! Thread model: the router is `Send + Sync` via a single `tokio::Mutex`
//! around the mutable backend state (KV cache + scratch are per-backend,
//! not per-request). Concurrent HTTP requests serialize behind the mutex;
//! this matches gen-1 behaviour — the C++ server also took a global
//! generation lock because the GPU runs one decode at a time.

#![warn(missing_docs)]

pub mod backend_impl;
pub mod cpu_lane;
pub mod detect;
pub mod medusa;
pub mod sampler;
pub mod tokenizer;
pub mod xdna_flm;

// The XDNA 2 FFI crate is a compile-time dep so flipping 1bit-router's
// `real-xdna` feature propagates to `1bit-xdna/real-xrt`. We don't
// call into it yet — see `Router::generate`'s routing guard — but keeping
// the `use` here ensures any future NPU wiring has a live import path,
// and that `cargo` doesn't garbage-collect the dep when link-time pruning
// kicks in.
#[allow(unused_imports)]
use onebit_xdna as _xdna;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use onebit_core::sampler::{Sampler, SamplerConfig};
use onebit_core::types::TokenId;
use tokio::sync::Mutex;

pub use backend_impl::{BackendError, HipBackend, ModelFormat, sniff_model_format};
pub use detect::{BackendKind, detect};
// Re-exports from the sampler lane. `cpu_lane::` still works as a
// back-compat path for `1bit-server`'s benches — the shim module just
// re-exports these same items.
pub use sampler::cpu::{CpuLane, CpuLaneError, CpuSampler, PipelinedOutcome};
pub use sampler::{SAMPLER_MODE_ENV, SamplerMode, sampler_mode_from_env};

// Re-exported below once the type is declared — see PerplexityResult.

/// Caller-selectable forward-pass backend.
///
/// Distinct from [`BackendKind`], which is the *detected* hardware family —
/// [`Backend`] is the *requested* dispatch surface. The two coincide today
/// (both default to HIP on gfx1151), but they diverge once the NPU lands:
/// `Backend::Xdna` is chosen explicitly via `HALO_BACKEND=xdna` or
/// [`RouterConfig::backend`] and routes long-prompt prefills to the XDNA 2
/// NPU through [`onebit_xdna`] while decode stays on the iGPU.
///
/// Variants:
///
/// * [`Backend::Hip`] — AMD gfx1151 iGPU via `1bit-hip`. Default.
/// * [`Backend::Xdna`] — XDNA 2 NPU via `1bit-xdna`. Prefill-only,
///   crossover at prompt length ≥ 33. Stub today; real dispatch wires on
///   once Peano lands a working xclbin (see
///   `docs/wiki/NPU-Kernel-Design.md`).
/// * [`Backend::Cpu`] — host CPU lane. 7th aspirational surface from
///   `docs/wiki/Peak-Performance-Projection.md`. Scaffolded today in
///   [`cpu_lane::CpuLane`] but not on the critical path yet; selecting
///   it returns [`BackendError::CpuLaneStub`] (see
///   `docs/wiki/CPU-Lane-Plan.md` for the three-step wire-up plan).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backend {
    /// gfx1151 iGPU path via `1bit-hip` (the production path).
    #[default]
    Hip,
    /// XDNA 2 NPU path via `1bit-xdna`. Prefill only, feature-gated
    /// behind `real-xdna`; in default builds forward() returns
    /// [`BackendError::NotYetWired`] for prompts ≥ 33 tokens.
    Xdna,
    /// Host CPU lane — sampler + tokenizer + dispatcher on Zen5 cores
    /// via [`cpu_lane::CpuLane`]. Scaffolded today but not on the
    /// critical path; dispatch surfaces [`BackendError::CpuLaneStub`].
    Cpu,
}

impl Backend {
    /// Parse a `HALO_BACKEND=...` value. Case-insensitive. Accepted
    /// spellings: `hip`, `xdna`, `cpu`. Any other value returns an error
    /// that names all three accepted spellings so operators see the valid
    /// set without having to `grep` the source.
    pub fn parse_env(raw: &str) -> Result<Self, BackendError> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "hip" => Ok(Backend::Hip),
            "xdna" => Ok(Backend::Xdna),
            "cpu" => Ok(Backend::Cpu),
            other => Err(BackendError::Other(format!(
                "HALO_BACKEND: unknown value {other:?}; accepted: hip | xdna | cpu"
            ))),
        }
    }

    /// Human-readable label for logs / `/v1/models`.
    pub fn label(self) -> &'static str {
        match self {
            Backend::Hip => "hip",
            Backend::Xdna => "xdna",
            Backend::Cpu => "cpu",
        }
    }
}

/// Prompt-length threshold above which [`Backend::Xdna`] claims the prefill
/// pass. Below this the HIP path handles the whole forward (NPU kernel
/// launch overhead dominates for short prompts).
///
/// Aspirational target from
/// `docs/wiki/Peak-Performance-Projection.md`; tune once the xclbin lands.
pub const XDNA_PREFILL_MIN_TOKENS: usize = 33;

/// Router configuration — which forward-pass backend to dispatch on.
///
/// Constructed either explicitly by the server layer or via
/// [`RouterConfig::from_env`], which reads `HALO_BACKEND` and falls back to
/// [`Backend::Hip`] when unset. Separate struct (rather than a bare enum
/// argument on `Router::new_with`) so we have a natural place to grow
/// knobs like a prefill-threshold override or an XDNA kernel-cache path
/// without churning call sites.
#[derive(Debug, Clone, Copy, Default)]
pub struct RouterConfig {
    /// Which execution surface the router dispatches the forward pass to.
    pub backend: Backend,
    /// Which sampler dispatch path the router takes — inline (default)
    /// or offloaded to the [`cpu_lane::CpuLane`]. Read from
    /// [`cpu_lane::SAMPLER_MODE_ENV`] (`HALO_SAMPLER`) in
    /// [`RouterConfig::from_env`].
    ///
    /// See `docs/wiki/CPU-Lane-Plan.md` for the decision memo that
    /// explains why the default is `Inline` — short version, the
    /// sampler is ~4.5% of a 15 ms forward pass and doesn't justify
    /// pipelining today.
    pub sampler_mode: SamplerMode,
}

impl RouterConfig {
    /// Build a config from the `HALO_BACKEND` + `HALO_SAMPLER` env vars.
    /// Unset → defaults (Hip / Inline). Set but unparsable →
    /// [`BackendError::Other`].
    pub fn from_env() -> Result<Self, BackendError> {
        let backend = match std::env::var("HALO_BACKEND") {
            Ok(raw) if !raw.is_empty() => Backend::parse_env(&raw)?,
            _ => Backend::default(),
        };
        let sampler_mode = sampler_mode_from_env()?;
        Ok(Self {
            backend,
            sampler_mode,
        })
    }
}

/// Sampling + stopping parameters forwarded through the router.
///
/// Deliberately smaller than OpenAI's full request shape — 1bit-server
/// translates user-facing options (temperature, top_p, max_tokens) into
/// this struct at request time.
#[derive(Debug, Clone)]
pub struct RouterRequest {
    /// Full conversation flattened to a single prompt string (system +
    /// user + assistant turns concatenated as the server decides).
    pub prompt: String,
    /// Hard cap on newly generated tokens.
    pub max_new_tokens: u32,
    /// Sampler knobs. `temperature <= 0` → greedy argmax on GPU.
    pub sampler: SamplerConfig,
    /// Optional stop strings. Matched against the detokenized tail.
    pub stop: Vec<String>,
}

impl Default for RouterRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            max_new_tokens: 256,
            sampler: SamplerConfig::default(),
            stop: Vec::new(),
        }
    }
}

/// A single generation result. Includes the full decoded text plus
/// token accounting in the same shape 1bit-server will re-emit in its
/// OpenAI usage block.
#[derive(Debug, Clone)]
pub struct RouterResponse {
    /// Decoded text (no BOS / EOS, suitable for a chat reply).
    pub text: String,
    /// `true` if we stopped because of EOS or a `stop` string; `false`
    /// if we hit `max_new_tokens`.
    pub stopped_on_eos: bool,
    /// Tokens consumed from the prompt (after tokenization + BOS).
    pub prompt_tokens: u32,
    /// Tokens emitted during decode.
    pub completion_tokens: u32,
}

/// The dispatcher. Always constructed via [`Router::new`] and then held
/// behind an `Arc` in the HTTP layer.
pub struct Router {
    inner: Arc<Mutex<Inner>>,
    model_id: String,
    kind: BackendKind,
    /// Caller-selected dispatch surface. Separate from `kind` (which is
    /// *detected* hardware); `backend` is what the operator asked for via
    /// `HALO_BACKEND` / [`RouterConfig`].
    backend: Backend,
    /// Sampler dispatch path. [`SamplerMode::Cpu`] (default) runs the
    /// sampler on the [`CpuLane`]'s rayon pool via
    /// [`CpuSampler::sample_pipelined`] — a `flume::bounded(1)` handoff
    /// that moves the logits buffer off the GPU-dispatch thread and
    /// onto a persistent `halo-sampler-pipe` worker, freeing the
    /// dispatch thread to start the next `forward_token`'s GPU staging
    /// while the sampler math runs on Zen5. [`SamplerMode::Inline`] is
    /// the legacy path retained for A/B comparison + rollback via
    /// `HALO_SAMPLER=inline`. See the `sampler` module docs for the
    /// 2026-04-20 flip rationale.
    sampler_mode: SamplerMode,
    /// CPU lane. Always constructed — the pool is cheap — but only
    /// dispatched through when `sampler_mode == SamplerMode::Cpu`.
    /// Held behind an `Arc` so `generate_blocking` can clone a handle
    /// into the spawn_blocking closure without taking a reference to
    /// `self`.
    cpu_lane: Arc<CpuLane>,
    /// Pipelined sampler — the bounded-channel handoff. Always
    /// constructed but only driven when
    /// `sampler_mode == SamplerMode::Cpu`. Held behind an `Arc` so
    /// both `generate` and `generate_stream` can hand a handle into
    /// their respective `spawn_blocking` closures. The worker thread
    /// it owns sticks around for the lifetime of the router.
    cpu_sampler: Arc<CpuSampler>,
    /// Medusa speculative-decode state. `Disabled` unless both
    /// `HALO_MEDUSA=1` is set AND the heads file at
    /// `HALO_MEDUSA_HEADS_PATH` loaded cleanly. Arc'd so the verify path
    /// can cheaply snapshot into worker closures.
    #[allow(dead_code)] // consumed by follow-up passes that wire the verify loop
    medusa: Arc<medusa::MedusaState>,
}

struct Inner {
    backend: HipBackend,
    /// Last decoded position in the shared KV cache. Reset between
    /// conversations.
    pos: i32,
    /// Live Medusa verifier state. Accumulates per-head acceptance
    /// counters across decode requests on this router instance. Always
    /// present — the verifier is zero-cost when the
    /// [`medusa::MedusaState`] is `Disabled`, and we read its counters
    /// unconditionally when `/metrics` lands them.
    medusa_verifier: medusa::TreeVerifier,
}

impl Router {
    /// Construct a router for the given `.h1b` model file. The tokenizer
    /// is resolved by taking the model path and replacing `.h1b` with
    /// `.htok`.
    ///
    /// Dispatch backend is taken from the `HALO_BACKEND` env var (via
    /// [`RouterConfig::from_env`]); set to `hip` (default), `xdna`, or
    /// `cpu`.
    ///
    /// Fails if no supported backend is available on this host.
    pub fn new(h1b_path: impl AsRef<Path>) -> Result<Self, BackendError> {
        let h1b_path = h1b_path.as_ref();
        let htok_path = default_htok_path(h1b_path);
        let model_id = default_model_id(h1b_path);

        let cfg = RouterConfig::from_env()?;
        Self::new_with_config(h1b_path, &htok_path, model_id, 4096, cfg)
    }

    /// Full constructor — lets the caller pin the tokenizer location,
    /// model id, and max KV cache size. Backend defaults to `Hip`.
    pub fn new_with(
        h1b_path: &Path,
        htok_path: &Path,
        model_id: String,
        max_context: usize,
    ) -> Result<Self, BackendError> {
        Self::new_with_config(
            h1b_path,
            htok_path,
            model_id,
            max_context,
            RouterConfig::default(),
        )
    }

    /// Full constructor with an explicit [`RouterConfig`]. This is the
    /// call site that actually reads `cfg.backend` and pins it onto the
    /// returned router. The weight-load path is still HIP-only: even
    /// under `Backend::Xdna`, weights upload to the iGPU today because
    /// the NPU's weight surface isn't defined yet. The dispatch decision
    /// happens inside [`Router::generate`] — see the routing guard there.
    pub fn new_with_config(
        h1b_path: &Path,
        htok_path: &Path,
        model_id: String,
        max_context: usize,
        cfg: RouterConfig,
    ) -> Result<Self, BackendError> {
        let kind = detect();
        match kind {
            BackendKind::Hip => {
                let backend = HipBackend::new(h1b_path, htok_path, model_id.clone(), max_context)?;
                // Build the CPU lane once and keep it on the router so
                // `HALO_SAMPLER=cpu` doesn't race to construct a pool
                // per request. Constructing with the default policy
                // honours `HALO_CPU_THREADS` automatically.
                let cpu_lane = Arc::new(
                    CpuLane::new()
                        .map_err(|e| BackendError::Other(format!("cpu lane init: {e}")))?,
                );
                // Pipelined sampler — persistent `halo-sampler-pipe`
                // worker fed by a `flume::bounded(1)` queue. Only
                // dispatched through when `sampler_mode == Cpu`, but
                // constructed unconditionally: the worker thread is
                // ~1 MB RSS and spawning it lazily would race with
                // the first decode request. See `sampler::cpu::CpuSampler`.
                let cpu_sampler = Arc::new(CpuSampler::new(cpu_lane.clone()));

                let medusa_cfg = medusa::MedusaConfig::from_env();
                let medusa_state = match medusa::MedusaState::from_config(&medusa_cfg) {
                    Ok(s) => {
                        if s.is_enabled() {
                            tracing::info!("medusa enabled: heads loaded from {:?}", medusa_cfg.medusa_heads_path);
                        }
                        s
                    }
                    Err(e) => {
                        tracing::warn!("medusa init failed, falling back to Disabled: {e}");
                        medusa::MedusaState::Disabled
                    }
                };

                tracing::info!(
                    model_id,
                    hw = %kind.label(),
                    requested = cfg.backend.label(),
                    sampler = cfg.sampler_mode.label(),
                    cpu_lane_threads = cpu_lane.num_threads(),
                    medusa = medusa_state.is_enabled(),
                    label = backend.label(),
                    "router ready"
                );
                Ok(Self {
                    inner: Arc::new(Mutex::new(Inner {
                        backend,
                        pos: 0,
                        medusa_verifier: medusa::TreeVerifier::new(),
                    })),
                    model_id,
                    kind,
                    backend: cfg.backend,
                    sampler_mode: cfg.sampler_mode,
                    cpu_lane,
                    cpu_sampler,
                    medusa: Arc::new(medusa_state),
                })
            }
            BackendKind::Mlx => {
                unimplemented!(
                    "MLX dispatch is not wired in this session — gfx1151 is the only supported target"
                );
            }
            BackendKind::None => Err(BackendError::Other(
                "no supported accelerator: need AMD gfx1151 (ROCm) or Apple MLX".into(),
            )),
        }
    }

    /// Advertised model id.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Which backend family the *hardware detection* selected.
    pub fn backend_kind(&self) -> BackendKind {
        self.kind
    }

    /// Which dispatch surface the *operator asked for* (via `HALO_BACKEND`
    /// or an explicit [`RouterConfig`]).
    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// Evaluate the prefill-routing decision without running the forward
    /// pass. Factored out of [`Router::generate`] so both sync and stream
    /// paths (and tests) share the same policy.
    ///
    /// Delegates to [`prefill_routing_decision`] with `self.backend`.
    pub fn check_prefill_routing(&self, prompt_tokens: usize) -> Result<(), BackendError> {
        prefill_routing_decision(self.backend, prompt_tokens)
    }

    /// Non-streaming generation. Blocks the calling task while the
    /// forward pass runs; call via `tokio::task::spawn_blocking` if the
    /// caller is on a shared runtime thread.
    pub async fn generate(&self, req: RouterRequest) -> Result<RouterResponse, BackendError> {
        let inner = self.inner.clone();
        let backend = self.backend;
        let sampler_mode = self.sampler_mode;
        let cpu_lane = self.cpu_lane.clone();
        let cpu_sampler = self.cpu_sampler.clone();
        let medusa = self.medusa.clone();
        tokio::task::spawn_blocking(move || {
            let mut guard = inner.blocking_lock();
            generate_blocking(
                &mut guard,
                req,
                None,
                backend,
                sampler_mode,
                &cpu_lane,
                &cpu_sampler,
                &medusa,
            )
        })
        .await
        .map_err(|e| BackendError::Other(format!("blocking task join: {e}")))?
    }

    /// Streaming generation. Each emitted `String` is a fresh delta that
    /// should be concatenated onto what the caller already produced.
    ///
    /// Emits bytes through a bounded channel so SSE backpressure flows
    /// all the way down to the forward-pass loop.
    pub async fn generate_stream(
        &self,
        req: RouterRequest,
    ) -> Result<
        impl futures::stream::Stream<Item = Result<String, BackendError>> + Send + 'static,
        BackendError,
    > {
        let inner = self.inner.clone();
        let backend = self.backend;
        let sampler_mode = self.sampler_mode;
        let cpu_lane = self.cpu_lane.clone();
        let cpu_sampler = self.cpu_sampler.clone();
        let medusa = self.medusa.clone();
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<String, BackendError>>(64);
        tokio::task::spawn_blocking(move || {
            let mut guard = inner.blocking_lock();
            let tx_clone = tx.clone();
            let sink = move |delta: String| {
                // If the client hung up we cannot do anything useful; drop.
                let _ = tx_clone.blocking_send(Ok(delta));
            };
            if let Err(e) = generate_blocking(
                &mut guard,
                req,
                Some(Box::new(sink)),
                backend,
                sampler_mode,
                &cpu_lane,
                &cpu_sampler,
                &medusa,
            ) {
                let _ = tx.blocking_send(Err(e));
            }
        });
        Ok(tokio_stream::wrappers::ReceiverStream::new(rx))
    }

    /// Which sampler path the router is using (process-level dial).
    pub fn sampler_mode(&self) -> SamplerMode {
        self.sampler_mode
    }

    /// `true` iff the Medusa speculative-decode gate resolved to
    /// `Enabled` at router construction. Used by `/metrics` + the
    /// bench harness so operators can confirm the gate without
    /// grepping startup logs.
    pub fn medusa_enabled(&self) -> bool {
        self.medusa.is_enabled()
    }

    /// Snapshot of the Medusa verifier counters since router startup.
    ///
    /// Returns `None` if the gate is off or no verify steps have run
    /// yet. Carries: (total verify steps, per-head accepted counts,
    /// mean accepted-prefix length). Cheap — integer loads only.
    pub async fn medusa_stats(&self) -> Option<MedusaStats> {
        let guard = self.inner.lock().await;
        if !self.medusa.is_enabled() || guard.medusa_verifier.steps == 0 {
            return None;
        }
        let rates = guard.medusa_verifier.per_head_acceptance()?;
        let mean_prefix = guard.medusa_verifier.mean_accepted_prefix_len()?;
        Some(MedusaStats {
            verify_steps: guard.medusa_verifier.steps,
            head_accepted: guard.medusa_verifier.head_accepted,
            per_head_rate: rates,
            mean_accepted_prefix_len: mean_prefix,
        })
    }

    /// Reset any per-conversation state on the backend (KV cache pointer).
    /// The C++ server does this between REPL turns and at server startup.
    pub async fn reset(&self) {
        let mut guard = self.inner.lock().await;
        guard.backend.reset();
        guard.pos = 0;
    }

    /// Compute perplexity over `text`.
    ///
    /// Semantics match the gen-1 C++ `bitnet_decode --ppl` mode byte-for-byte:
    ///
    /// 1. Tokenize `text` (with BOS). Truncate to at most `max_tokens` tokens.
    /// 2. Starting from `pos=0` (KV cache is reset for the run), feed tokens
    ///    one at a time through `forward_token`. At each step `i`, compute
    ///    `log_softmax(logits)[prompt_ids[i+1]]` via log-sum-exp, accumulate
    ///    `-logp` into `total_nll`, increment `scored`.
    /// 3. If the sequence is longer than `stride`, we re-chunk: every
    ///    `stride` tokens we reset the KV cache and restart at `pos=0`. The
    ///    C++ reference does not do this — it's always a single pass — but
    ///    is what we'd need if `max_tokens > max_context`. For parity with
    ///    the documented 1024-token baseline, pass `max_tokens=1024,
    ///    stride=1024` and we take the single-pass path.
    ///
    /// Returns `(mean_nll, perplexity, scored_tokens, elapsed_ms)`.
    ///
    /// Invariants:
    ///   * After this call the router's `pos` is reset to 0 (KV cache is
    ///     considered clean).
    ///   * `stride` clamped to `max_context`. `max_tokens` clamped to
    ///     `usize::MAX` but limited by the actual tokenized length.
    pub async fn perplexity(
        &self,
        text: String,
        stride: usize,
        max_tokens: usize,
    ) -> Result<PerplexityResult, BackendError> {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let mut guard = inner.blocking_lock();
            perplexity_blocking(&mut guard, &text, stride, max_tokens)
        })
        .await
        .map_err(|e| BackendError::Other(format!("blocking task join: {e}")))?
    }
}

/// Snapshot of the Medusa verifier state. Read via
/// [`Router::medusa_stats`] for bench + `/metrics`.
#[derive(Debug, Clone)]
pub struct MedusaStats {
    /// Cumulative count of verify steps run since router startup.
    pub verify_steps: u64,
    /// Per-head accepted counts — `head_accepted[i]` is the number
    /// of times head i produced the same argmax as the backbone at
    /// the corresponding position.
    pub head_accepted: [u64; medusa::heads::NUM_MEDUSA_HEADS],
    /// Live per-head acceptance rate — just `head_accepted / steps`.
    pub per_head_rate: [f64; medusa::heads::NUM_MEDUSA_HEADS],
    /// Mean accepted prefix length (0..=NUM_MEDUSA_HEADS). Equal to
    /// `tokens/cycle - 1` — the average number of speculative
    /// tokens accepted per decode cycle.
    pub mean_accepted_prefix_len: f64,
}

/// PPL harness result.
#[derive(Debug, Clone)]
pub struct PerplexityResult {
    /// Mean negative log-likelihood across all scored tokens (nats).
    pub mean_nll: f64,
    /// `exp(mean_nll)` — standard perplexity.
    pub perplexity: f64,
    /// Number of (context, next-token) pairs that contributed to the mean.
    pub scored_tokens: usize,
    /// Wall-clock time of the scoring loop in milliseconds.
    pub elapsed_ms: f64,
}

fn perplexity_blocking(
    inner: &mut Inner,
    text: &str,
    stride: usize,
    max_tokens: usize,
) -> Result<PerplexityResult, BackendError> {
    // Reset KV so the run is deterministic regardless of previous chat
    // history. Matches gen-1 where --ppl always starts from a fresh decoder.
    inner.backend.reset();
    inner.pos = 0;

    let mut ids: Vec<TokenId> = inner.backend.tokenize(text);
    if ids.len() > max_tokens {
        ids.truncate(max_tokens);
    }
    if ids.len() < 2 {
        return Err(BackendError::BadInput(
            "ppl: need at least 2 tokens after tokenization",
        ));
    }
    // `stride` bounds the per-pass window length. 0 is treated as
    // "whole sequence in one pass".
    let stride = if stride == 0 { ids.len() } else { stride };

    let mut logits_scratch: Vec<f32> = Vec::new();
    let mut total_nll: f64 = 0.0;
    let mut scored: usize = 0;
    let start = std::time::Instant::now();

    let n = ids.len();
    let mut chunk_start = 0usize;
    while chunk_start + 1 < n {
        let chunk_end = (chunk_start + stride).min(n);
        // For each chunk we feed tokens [chunk_start .. chunk_end) at
        // positions [0 .. chunk_end-chunk_start) and score the next-token
        // probability at every step except the last of this chunk (that
        // one has no target inside the chunk unless the chunk ends at `n`,
        // in which case chunk_end == n and i+1 == n so we also stop).
        // Reset KV between chunks.
        if chunk_start > 0 {
            inner.backend.reset();
            inner.pos = 0;
        }
        let chunk_len = chunk_end - chunk_start;
        for i in 0..chunk_len {
            let tok = ids[chunk_start + i];
            let pos = i as i32;
            let _argmax = inner.backend.forward_token(tok, pos, &mut logits_scratch)?;
            // Target for this step is ids[chunk_start + i + 1]; only valid
            // while it's inside the current chunk (we re-feed at the next
            // chunk boundary).
            let target_idx = chunk_start + i + 1;
            if target_idx >= chunk_end {
                break;
            }
            let target = ids[target_idx];
            let nll = neg_log_softmax_at(&logits_scratch, target);
            total_nll += nll;
            scored += 1;
        }
        if chunk_end == n {
            break;
        }
        chunk_start = chunk_end;
    }

    // Leave the backend in a clean state so the next chat request doesn't
    // inherit our KV cache.
    inner.backend.reset();
    inner.pos = 0;

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    if scored == 0 {
        return Err(BackendError::BadInput("ppl: no tokens were scored"));
    }
    let mean_nll = total_nll / (scored as f64);
    Ok(PerplexityResult {
        mean_nll,
        perplexity: mean_nll.exp(),
        scored_tokens: scored,
        elapsed_ms,
    })
}

/// Compute `-log_softmax(logits)[target]` in f64 via log-sum-exp.
///
/// Shared helper; factored out so the PPL math is unit-testable without
/// a GPU.
pub(crate) fn neg_log_softmax_at(logits: &[f32], target: TokenId) -> f64 {
    let v = logits.len();
    debug_assert!(v > 0);
    debug_assert!((target as usize) < v, "target out of vocab range");
    let mut max_l = logits[0];
    for &l in &logits[1..] {
        if l > max_l {
            max_l = l;
        }
    }
    let mut sum_exp: f64 = 0.0;
    for &l in logits {
        sum_exp += ((l - max_l) as f64).exp();
    }
    let logp = (logits[target as usize] - max_l) as f64 - sum_exp.ln();
    -logp
}

/// Synchronous generation path — shared between `generate` and
/// `generate_stream`. If `on_delta` is set, each new delta string is
/// forwarded to it as soon as the tokenizer emits decoded bytes; the
/// full text is also returned in the response.
fn generate_blocking(
    inner: &mut Inner,
    req: RouterRequest,
    mut on_delta: Option<Box<dyn FnMut(String) + Send>>,
    backend: Backend,
    sampler_mode: SamplerMode,
    cpu_lane: &CpuLane,
    cpu_sampler: &CpuSampler,
    medusa: &medusa::MedusaState,
) -> Result<RouterResponse, BackendError> {
    // `cpu_lane` is currently reserved for future use (parallel top-k
    // reductions beyond the `Sampler::sample` path). The `CpuSampler`
    // handoff already owns a clone of the `Arc<CpuLane>` and schedules
    // its worker inside that pool. Keeping the reference on the
    // signature so a future top-k-on-lane path doesn't need another
    // plumbing churn.
    let _ = cpu_lane;
    // ---- Tokenize prompt ----
    let prompt_ids: Vec<TokenId> = inner.backend.tokenize(&req.prompt);
    if prompt_ids.is_empty() {
        return Err(BackendError::BadInput("empty prompt after tokenization"));
    }

    // ---- Routing guard ----
    //
    // Today the whole XDNA 2 integration lives here. The ternary-aware
    // router fans out like this:
    //   * Backend::Hip → always proceed (the happy path).
    //   * Backend::Xdna + long prompt + ternary weights → refuse with
    //     `NpuTernaryUnsupported` (AMD hasn't shipped ternary→INT8 yet).
    //   * Backend::Xdna + long prompt + non-ternary weights → spawn
    //     FastFlowLM via `xdna_flm::flm_prefill`. See that module's
    //     docstring for the subprocess shape.
    //   * Backend::Xdna + short prompt → fall through to HIP (NPU launch
    //     overhead isn't worth it for prompts < 33 tokens).
    //   * Backend::Cpu → `CpuLaneStub`.
    //
    // `HipBackend::new` currently refuses to load anything other than
    // HaloV2 ternary weights (see `backend_impl.rs:308`), so the live
    // router is always ternary. Passing `is_ternary = true` hardcodes
    // that invariant — when the GGUF Q4 loader lands and HipBackend
    // grows a `model_format` accessor, this call site should consult it
    // instead.
    prefill_routing_decision_with_model(backend, prompt_ids.len(), /* is_ternary */ true)?;

    let prompt_tokens = prompt_ids.len() as u32;
    let max_new = req.max_new_tokens.max(1) as usize;

    // Reset the KV cache position per request. Matches gen-1 bitnet_decode's
    // `cache_pos = 0` at the start of every /v1/chat/completions handler.
    // Without this, `pos` accumulates across requests and eventually hits
    // the max_context ceiling (4096) with "context overflow: 4096 + 1 >
    // 4096" errors. Stateful multi-turn carryover is a separate feature.
    inner.pos = 0;

    // Seed the sampler history with the prompt so rep penalty behaves.
    let mut history: Vec<TokenId> = prompt_ids.clone();
    let mut sampler = Sampler::new(req.sampler);
    let mut logits_scratch: Vec<f32> = Vec::new();

    // ---- Prefill (no sampling — we just run the forward pass on every
    //      prompt token to populate KV cache entries). The last token
    //      seeds `cur` for the decode loop. ----
    let mut cur: TokenId = prompt_ids[0];
    for (i, &tok) in prompt_ids.iter().enumerate() {
        let next = inner
            .backend
            .forward_token(tok, inner.pos + i as i32, &mut logits_scratch)?;
        cur = next; // prefill's "next" is only used to seed decode
    }
    inner.pos += prompt_ids.len() as i32;

    // ---- Decode loop ----
    let mut generated_ids: Vec<TokenId> = Vec::with_capacity(max_new);
    let mut stopped_on_eos = false;
    let mut printed_bytes = 0usize;
    let mut full_text = String::new();

    // BitNet's special tokens from the tokenizer — matches the C++ stops.
    let stop_ids = [128001, 128009];

    // Medusa dispatch gate. Only the greedy path activates it today
    // (the sampler-aware variant compares head argmax with a sampled
    // token rather than base argmax — a follow-up pass once the tree-
    // attention kernel is wired). Stepper indirection keeps the legacy
    // path byte-identical for the `medusa_disabled` case.
    let medusa_active =
        medusa.is_enabled() && req.sampler.temperature <= 0.0 && on_delta.is_none();
    let mut hidden_scratch: Vec<u16> = Vec::new();
    let mut head_logits_scratch: Vec<f32> = Vec::new();

    let mut step_count = 0usize;
    while step_count < max_new {
        let pos = inner.pos + step_count as i32;

        // -------------------------------------------------------------
        // Base forward pass. Medusa path takes a variant that also
        // copies the post-final-norm hidden state back to the host so
        // the heads can project it into candidate-token logits.
        // -------------------------------------------------------------
        let argmax_next = if medusa_active {
            inner.backend.forward_token_with_hidden(
                cur,
                pos,
                &mut logits_scratch,
                &mut hidden_scratch,
            )?
        } else {
            inner.backend.forward_token(cur, pos, &mut logits_scratch)?
        };

        // Optional host-side sampling path. [`SamplerMode::Inline`]
        // takes the direct call (legacy, `HALO_SAMPLER=inline`).
        // [`SamplerMode::Cpu`] (default as of 2026-04-20) routes through
        // [`CpuSampler::sample_pipelined`] — a `flume::bounded(1)`
        // handoff onto a persistent `halo-sampler-pipe` worker pinned
        // to the [`CpuLane`]'s rayon pool. Semantics are bit-identical
        // (same `Sampler::sample` code, same RNG state, just a
        // different executing thread). Greedy (`temp <= 0`)
        // short-circuits to the GPU-returned argmax in both modes
        // since there's no scalar work to do.
        let next = if req.sampler.temperature > 0.0 {
            match sampler_mode {
                SamplerMode::Inline => sampler.sample(&mut logits_scratch, &history)?,
                SamplerMode::Cpu => {
                    // Move the scratch buffers into the worker via
                    // flume; receive them back on the reply so we keep
                    // the allocation alive across decode steps. The
                    // `std::mem::take` swap replaces `logits_scratch`
                    // with an empty Vec — the next `forward_token`
                    // call will `extend_from_slice` / `resize` into
                    // whichever Vec we put back.
                    let logits_out = std::mem::take(&mut logits_scratch);
                    let history_out = std::mem::take(&mut history);
                    let out = cpu_sampler
                        .sample_pipelined(&mut sampler, logits_out, history_out)
                        .map_err(|e| BackendError::Other(format!("cpu sampler: {e}")))?;
                    // Put the buffers back even on sampler-error so
                    // the allocation sticks with the decode loop.
                    logits_scratch = out.logits;
                    history = out.history;
                    out.outcome?
                }
            }
        } else {
            argmax_next
        };

        history.push(next);

        // Stop-token check before detokenizing: otherwise the special
        // token's string form (e.g. "<|eot_id|>") leaks into the output.
        if stop_ids.contains(&next) {
            stopped_on_eos = true;
            break;
        }

        generated_ids.push(next);
        step_count += 1;

        // Incremental detokenization — decode the whole generated prefix
        // each step (matches C++) and emit whatever is newly printable.
        let decoded_so_far = inner.backend.detokenize(&generated_ids);
        if decoded_so_far.len() > printed_bytes {
            let delta = decoded_so_far[printed_bytes..].to_string();
            printed_bytes = decoded_so_far.len();
            full_text.push_str(&delta);
            if let Some(cb) = on_delta.as_mut() {
                cb(delta);
            }
        }

        if req
            .stop
            .iter()
            .any(|s| !s.is_empty() && full_text.ends_with(s))
        {
            stopped_on_eos = true;
            break;
        }

        cur = next;

        // -------------------------------------------------------------
        // Medusa speculative-decode extension. When the heads are
        // loaded and we took the with-hidden path above, project the
        // backbone hidden through the four heads to get candidate
        // tokens (at positions t+2..t+5 relative to the base token we
        // just emitted), then verify them sequentially against the
        // backbone. Accept the longest matching prefix.
        //
        // Sequential verify = one backbone call per verified position.
        // Each verify call advances the KV cache by one slot, so if
        // head i is rejected we must NOT continue to verify[i+1] —
        // that would feed the rejected head_i into the backbone and
        // corrupt KV. We therefore break out of the verify loop on
        // first mismatch. Tree-attention amortisation (the follow-up
        // lane) is what actually yields >1.0 tok/backbone.
        // -------------------------------------------------------------
        if medusa_active && step_count < max_new {
            if let medusa::MedusaState::Enabled { heads, .. } = medusa {
                // 1) Project backbone hidden through the 4 heads →
                //    host-side `h_out` fp16 vectors.
                let projected = heads
                    .project_all_heads_host(&hidden_scratch)
                    .map_err(|e| BackendError::Other(format!("medusa project: {e}")))?;

                // 2) Head logits = lm_head(h_out_i); take argmax per
                //    head. Re-use the live backbone lm_head GEMV on
                //    the device to avoid a second 128256×2560 upload
                //    per cycle.
                let mut head_candidates = [0i32; medusa::heads::NUM_MEDUSA_HEADS];
                for i in 0..medusa::heads::NUM_MEDUSA_HEADS {
                    inner
                        .backend
                        .lm_head_from_hidden(&projected[i], &mut head_logits_scratch)?;
                    let mut max_l = head_logits_scratch[0];
                    let mut max_i = 0i32;
                    for (idx, &l) in head_logits_scratch.iter().enumerate().skip(1) {
                        if l > max_l {
                            max_l = l;
                            max_i = idx as i32;
                        }
                    }
                    head_candidates[i] = max_i;
                }

                // 3) Sequential verify.
                //
                //    Standard Medusa convention: head_i predicts the
                //    token at position base+(i+1) relative to the
                //    backbone's own argmax `next`. So head_0 predicts
                //    t+2 (one past the base), head_1 → t+3, etc. Each
                //    verify step runs the backbone on the prior
                //    accepted token to produce base_argmax_i at that
                //    head's position.
                //
                //    verify_cur starts as `next` — the base token at
                //    position `pos+1`. verify[0] feeds it at pos+1 to
                //    predict pos+2 (= head_0's position).
                //
                //    On a match, we accept head_i and advance
                //    verify_cur = head_i for the next step. On a
                //    mismatch we break — the KV cache past this
                //    position is still clean so the next decode
                //    cycle's first forward_token Just Works.
                let mut verify_cur = next;
                let mut accepted_len = 0usize;
                let mut mismatch_base: Option<i32> = None;
                for i in 0..medusa::heads::NUM_MEDUSA_HEADS {
                    // Respect remaining budget.
                    if step_count + accepted_len + 1 > max_new {
                        break;
                    }
                    let vpos = inner.pos + (step_count + accepted_len) as i32;
                    let base_argmax_i = inner.backend.forward_token(
                        verify_cur,
                        vpos,
                        &mut logits_scratch,
                    )?;

                    if base_argmax_i == head_candidates[i] {
                        accepted_len += 1;
                        verify_cur = head_candidates[i];
                        if stop_ids.contains(&head_candidates[i]) {
                            mismatch_base = None;
                            break;
                        }
                        continue;
                    }

                    // Mismatch: record the base argmax as the fallback
                    // token to emit, then stop verifying.
                    mismatch_base = Some(base_argmax_i);
                    break;
                }

                // 4) Update the verifier's counters (for `/metrics`
                //    and the bench output). We hand it the full-row
                //    base_argmax reconstructed from the matches +
                //    (at most one) mismatch we observed. Positions
                //    beyond the first mismatch were never verified
                //    sequentially, so we mark them never-equal so
                //    the verifier's per-head counters don't falsely
                //    credit them.
                let mut base_argmax_full = [0i32; medusa::heads::NUM_MEDUSA_HEADS];
                for i in 0..accepted_len {
                    // Matched heads: base == head by construction.
                    base_argmax_full[i] = head_candidates[i];
                }
                if accepted_len < medusa::heads::NUM_MEDUSA_HEADS {
                    if let Some(bm) = mismatch_base {
                        base_argmax_full[accepted_len] = bm;
                    } else {
                        // Stop-token early-exit on the accepted
                        // path; no "base vs head" comparison was
                        // done at this index. Mark never-equal.
                        base_argmax_full[accepted_len] =
                            head_candidates[accepted_len].wrapping_add(1);
                    }
                }
                for j in (accepted_len + 1)..medusa::heads::NUM_MEDUSA_HEADS {
                    // Never verified — mark never-equal so the
                    // per-head counter stays honest.
                    base_argmax_full[j] = head_candidates[j].wrapping_add(1);
                }
                let _ = inner
                    .medusa_verifier
                    .verify_step(&head_candidates, &base_argmax_full);

                // 5) Emit accepted head-predicted tokens.
                for i in 0..accepted_len {
                    let tok = head_candidates[i];
                    history.push(tok);
                    generated_ids.push(tok);
                    step_count += 1;
                    if stop_ids.contains(&tok) {
                        stopped_on_eos = true;
                        break;
                    }
                }

                // 6) Emit the "free" fallback base token from the
                //    verify step that recorded a mismatch. This is
                //    what makes sequential verify equivalent in
                //    wall-clock to vanilla decode: every verify call
                //    produces a usable base argmax, and if the head
                //    at that position was wrong, we emit the correct
                //    base answer instead.
                if !stopped_on_eos && step_count < max_new {
                    if let Some(bm) = mismatch_base {
                        history.push(bm);
                        if stop_ids.contains(&bm) {
                            stopped_on_eos = true;
                        } else {
                            generated_ids.push(bm);
                            step_count += 1;
                            cur = bm;
                        }
                    } else if accepted_len > 0 {
                        cur = head_candidates[accepted_len - 1];
                    }
                } else if accepted_len > 0 {
                    cur = head_candidates[accepted_len - 1];
                }

                // 7) Re-detokenize for streaming / stop-substring
                //    checks. Streaming path is gated off above
                //    (`medusa_active` requires `on_delta.is_none()`)
                //    so this only matters for stop-substring and the
                //    terminal `full_text`.
                let decoded_so_far = inner.backend.detokenize(&generated_ids);
                if decoded_so_far.len() > printed_bytes {
                    let delta = decoded_so_far[printed_bytes..].to_string();
                    printed_bytes = decoded_so_far.len();
                    full_text.push_str(&delta);
                }
                if stopped_on_eos {
                    break;
                }
                if req
                    .stop
                    .iter()
                    .any(|s| !s.is_empty() && full_text.ends_with(s))
                {
                    stopped_on_eos = true;
                    break;
                }
            }
        }
    }

    // `step_count` is the number of generated tokens emitted; the
    // accumulator is consumed below via `generated_ids.len()`.
    let _ = step_count;

    // Advance cache_pos past this turn's decoded tokens.
    inner.pos += generated_ids.len() as i32;

    Ok(RouterResponse {
        text: full_text,
        stopped_on_eos,
        prompt_tokens,
        completion_tokens: generated_ids.len() as u32,
    })
}

/// Default tokenizer path beside the model: `foo.h1b` → `foo.htok`.
fn default_htok_path(h1b_path: &Path) -> PathBuf {
    let mut p = h1b_path.to_path_buf();
    p.set_extension("htok");
    p
}

/// Default model-id string: file stem of the `.h1b`, lowercased.
fn default_model_id(h1b_path: &Path) -> String {
    h1b_path
        .file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| "halo-model".to_string())
}

// (thiserror's `#[from]` on the `Halo` / `Hip` variants already generates
// the `From<HaloError>` / `From<RcppError>` impls the `?` operator needs.)

/// Standalone routing-policy function. `Ok(())` = safe to dispatch on the
/// current HIP path. `Err(BackendError::NotYetWired(...))` = the caller
/// selected a backend whose real dispatch is not compiled / loaded yet.
///
/// Rules:
///
/// * [`Backend::Hip`] → always OK.
/// * [`Backend::Xdna`] with `prompt_tokens >= `[`XDNA_PREFILL_MIN_TOKENS`]
///   → `NotYetWired("NPU prefill backend not loaded — build xclbin via
///   Peano first, see docs/wiki/NPU-Kernel-Design.md")`.
/// * [`Backend::Xdna`] with a shorter prompt → OK (we fall through to HIP
///   decode; the NPU isn't worth its launch overhead on tiny prompts).
/// * [`Backend::Cpu`] → always [`BackendError::CpuLaneStub`]. The lane
///   itself (thread pool, parallel sampler primitive) is scaffolded in
///   [`cpu_lane`], but it's not on the critical path yet; operators who
///   explicitly set `HALO_BACKEND=cpu` hit this arm.
///
/// Separate from [`Router::check_prefill_routing`] so tests don't need to
/// stand up a Router (which requires HIP + a real .h1b). Both paths share
/// this implementation, so asserting against this function also validates
/// the live `generate` path byte-for-byte.
///
/// **Ternary-unaware.** This function predates the FLM bridge; it treats
/// every Xdna prompt the same way regardless of weight format. New code
/// should prefer [`prefill_routing_decision_with_model`] which routes
/// non-ternary Xdna prompts into the FastFlowLM subprocess bridge and
/// surfaces the ternary-on-NPU gap as a distinct error variant. This
/// function is kept as a thin wrapper over the model-aware version with
/// `is_ternary = true` (the status quo: every model 1bit-router ships
/// with today is ternary).
pub fn prefill_routing_decision(
    backend: Backend,
    prompt_tokens: usize,
) -> Result<(), BackendError> {
    prefill_routing_decision_with_model(backend, prompt_tokens, /* is_ternary */ true)
}

/// Model-aware routing-policy function. Supersedes
/// [`prefill_routing_decision`] for callers that know the loaded model's
/// weight format.
///
/// Extends the rule set with the FastFlowLM bridge:
///
/// * [`Backend::Xdna`] + `prompt_tokens >= XDNA_PREFILL_MIN_TOKENS` +
///   `is_ternary == true` → [`BackendError::NpuTernaryUnsupported`].
///   The message points at `project_lemonade_10_2_pivot.md` so ops knows
///   this is an upstream AMD feature wait, not a 1bit-router bug. Retry
///   with `HALO_BACKEND=hip`.
/// * [`Backend::Xdna`] + `prompt_tokens >= XDNA_PREFILL_MIN_TOKENS` +
///   `is_ternary == false` → dispatches through
///   [`xdna_flm::flm_prefill`]. In practice this returns either
///   [`BackendError::FlmSpawn`] (binary missing) or
///   [`BackendError::NotYetWired`] (FLM alive but KV handoff pending —
///   see `xdna_flm` module docstring).
/// * Everything else mirrors [`prefill_routing_decision`] exactly.
///
/// The caller passes `is_ternary` as a bare bool instead of a model
/// handle so this crate doesn't grow a `1bit-core` model-format
/// dependency just for the policy check — the router's constructors
/// already know whether they loaded a ternary `.h1b` or a Q4 GGUF, they
/// can pass the flag through.
pub fn prefill_routing_decision_with_model(
    backend: Backend,
    prompt_tokens: usize,
    is_ternary: bool,
) -> Result<(), BackendError> {
    match backend {
        Backend::Hip => Ok(()),
        Backend::Xdna if prompt_tokens >= XDNA_PREFILL_MIN_TOKENS => {
            if is_ternary {
                // AMD ternary→INT8 path hasn't shipped; FLM is Q4NX-only.
                // Distinct error variant so ops dashboards can count it
                // separately from "backend not built".
                Err(BackendError::NpuTernaryUnsupported(
                    "ternary BitNet weights cannot run on XDNA 2 today — FastFlowLM is Q4NX-only, \
                     AMD's ternary→INT8 mapping is pending (see project_lemonade_10_2_pivot.md). \
                     Retry with HALO_BACKEND=hip.",
                ))
            } else {
                // Non-ternary (Q4 GGUF etc.) — drive FLM. The subprocess
                // path handles its own feature-flag + binary-missing
                // diagnostics; we just forward whatever it returns.
                // Model id is the caller's concern; we pass a FLM-shaped
                // placeholder here because this is the policy checker,
                // not the actual dispatch. The live `generate` path
                // passes the router's `model_id` through in
                // `generate_blocking`.
                let _ = xdna_flm::flm_prefill(
                    "",
                    &xdna_flm::default_flm_model_id(),
                    /* is_ternary */ false,
                )?;
                // Unreachable on current FLM: flm_prefill always returns
                // Err until KV handoff lands. Kept for the day it does.
                Ok(())
            }
        }
        Backend::Xdna => Ok(()),
        Backend::Cpu => Err(BackendError::CpuLaneStub(
            "CPU sampler lane scaffolded, not yet on critical path; see docs/wiki/CPU-Lane-Plan.md",
        )),
    }
}

#[cfg(test)]
mod ppl_math_tests {
    //! Unit tests for the PPL math helpers. These are GPU-free — they
    //! exercise the log-softmax reduction that accumulates `mean_nll`
    //! without going through `forward_token`.

    use super::neg_log_softmax_at;

    /// With a uniform logit vector of length V, `-log_softmax` at any
    /// target equals `log(V)`. Perplexity of a uniform predictor over V
    /// classes is V — that's the textbook sanity check.
    #[test]
    fn uniform_logits_give_log_v() {
        for v in [2usize, 10, 128, 32_000] {
            let logits = vec![0.7f32; v];
            let nll = neg_log_softmax_at(&logits, (v / 2) as i32);
            let expected = (v as f64).ln();
            assert!(
                (nll - expected).abs() < 1e-9,
                "uniform V={v}: got {nll}, expected {expected}"
            );
        }
    }

    /// With an argmax-strong logit at the target, NLL is near zero and
    /// therefore perplexity near 1.
    #[test]
    fn confident_target_is_near_zero_nll() {
        let mut logits = vec![-50.0f32; 1000];
        logits[42] = 100.0;
        let nll = neg_log_softmax_at(&logits, 42);
        assert!(nll < 1e-10, "expected near-zero NLL, got {nll}");
        assert!((nll.exp() - 1.0).abs() < 1e-9);
    }

    /// Two-class softmax with logit difference d: NLL of the wrong class
    /// equals `log(1 + exp(d))` on the low side and `log(1 + exp(-d))` on
    /// the high side. Spot-check one concrete value.
    #[test]
    fn two_class_closed_form() {
        let logits = [0.0f32, 2.0];
        // softmax[0] = 1 / (1 + e^2); nll(target=0) = log(1 + e^2)
        let nll0 = neg_log_softmax_at(&logits, 0);
        let expected0 = (1.0f64 + 2.0f64.exp()).ln();
        assert!((nll0 - expected0).abs() < 1e-9, "{nll0} vs {expected0}");
        // Consistency: exp(-nll0) + exp(-nll1) must equal 1.
        let nll1 = neg_log_softmax_at(&logits, 1);
        let p_sum = (-nll0).exp() + (-nll1).exp();
        assert!((p_sum - 1.0).abs() < 1e-9, "probs sum to {p_sum}, not 1");
    }

    /// Invariance to a constant logit shift — numerical stability check.
    #[test]
    fn log_softmax_shift_invariant() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b: Vec<f32> = a.iter().map(|&x| x + 10_000.0).collect();
        for t in 0..a.len() {
            let na = neg_log_softmax_at(&a, t as i32);
            let nb = neg_log_softmax_at(&b, t as i32);
            assert!(
                (na - nb).abs() < 1e-6,
                "shift not invariant at t={t}: {na} vs {nb}"
            );
        }
    }
}

#[cfg(test)]
mod backend_config_tests {
    //! Tests for the `Backend` enum + `RouterConfig` env-var plumbing +
    //! the prefill-routing guard that gates XDNA 2 NPU dispatch.
    //!
    //! These are GPU-free: they exercise parsing and policy, never the
    //! forward pass. The "forward returns NotYetWired" test goes through
    //! [`prefill_routing_decision`], which is the same function
    //! [`generate_blocking`] calls — so a pass here means the live
    //! `Router::generate` path would surface the identical error.
    //!
    //! Env-var tests serialize through a Mutex because `std::env::set_var`
    //! mutates process-global state and other tests in the same binary
    //! can race with it.
    use super::*;
    use std::sync::Mutex;

    /// Serialize the env-var tests — `std::env` is process-global.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// Rule: the default dispatch surface is HIP. Gen-1 + gen-2 ship this
    /// way; flipping the default would silently send CI boxes without
    /// `HALO_BACKEND` set off the happy path.
    #[test]
    fn default_backend_is_hip() {
        assert_eq!(Backend::default(), Backend::Hip);
        assert_eq!(RouterConfig::default().backend, Backend::Hip);
    }

    /// `HALO_BACKEND=xdna` must parse to `Backend::Xdna` and land on the
    /// RouterConfig. Case-insensitive + whitespace-tolerant.
    #[test]
    fn halo_backend_env_parses_xdna() {
        let _g = ENV_LOCK.lock().unwrap();
        // SAFETY: edition-2024 moved env mutation behind unsafe. We're
        // single-threaded inside the lock.
        unsafe {
            std::env::set_var("HALO_BACKEND", "xdna");
        }
        let cfg = RouterConfig::from_env().expect("parse");
        assert_eq!(cfg.backend, Backend::Xdna);

        // Whitespace / case tolerance — ops tooling isn't always tidy.
        unsafe {
            std::env::set_var("HALO_BACKEND", "  XDNA\n");
        }
        let cfg2 = RouterConfig::from_env().expect("parse");
        assert_eq!(cfg2.backend, Backend::Xdna);

        // cpu round-trip too — keeps the three-way discriminator honest.
        unsafe {
            std::env::set_var("HALO_BACKEND", "cpu");
        }
        let cfg3 = RouterConfig::from_env().expect("parse");
        assert_eq!(cfg3.backend, Backend::Cpu);

        unsafe {
            std::env::remove_var("HALO_BACKEND");
        }
    }

    /// A garbage `HALO_BACKEND` value must error cleanly — no panic,
    /// no silent fallback. Message must name the accepted spellings so
    /// the operator knows the valid set.
    #[test]
    fn halo_backend_env_rejects_garbage() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::set_var("HALO_BACKEND", "nonsense");
        }
        let err = RouterConfig::from_env().expect_err("should reject");
        let msg = format!("{err}");
        assert!(
            msg.contains("hip") && msg.contains("xdna") && msg.contains("cpu"),
            "error should list accepted spellings; got: {msg}"
        );
        unsafe {
            std::env::remove_var("HALO_BACKEND");
        }
    }

    /// The NPU crossover threshold is literally 33 tokens. After the
    /// FLM bridge landed (2026-04-20), the ternary-implicit legacy
    /// [`prefill_routing_decision`] re-routes to the ternary-unsupported
    /// arm rather than `NotYetWired`. The message must point ops at
    /// `project_lemonade_10_2_pivot.md` so they see this is an upstream
    /// AMD wait, not a build-time gap. This is the exact arm the live
    /// `generate_blocking` takes — same function, same string.
    #[test]
    fn xdna_long_prompt_returns_npu_ternary_unsupported() {
        // Boundary: 33 tokens is at threshold → refuse.
        let err = prefill_routing_decision(Backend::Xdna, XDNA_PREFILL_MIN_TOKENS)
            .expect_err("xdna @ threshold must refuse");
        match err {
            BackendError::NpuTernaryUnsupported(msg) => {
                assert!(
                    msg.contains("ternary") && msg.contains("Q4NX"),
                    "error must name ternary + Q4NX so ops understands the gap; got: {msg}"
                );
                assert!(
                    msg.contains("HALO_BACKEND=hip"),
                    "error must tell ops how to retry; got: {msg}"
                );
            }
            other => panic!("expected NpuTernaryUnsupported, got {other:?}"),
        }

        // Comfortably over threshold — same arm.
        let err =
            prefill_routing_decision(Backend::Xdna, 256).expect_err("long xdna prompt must refuse");
        assert!(matches!(err, BackendError::NpuTernaryUnsupported(_)));

        // Short prompt on Xdna: falls through (the NPU launch overhead
        // isn't worth it for tiny prefills, so HIP handles it).
        assert!(prefill_routing_decision(Backend::Xdna, 32).is_ok());
        assert!(prefill_routing_decision(Backend::Xdna, 1).is_ok());

        // Hip is always fine; Cpu always refuses with the scaffold stub.
        assert!(prefill_routing_decision(Backend::Hip, 4096).is_ok());
        let cpu_err = prefill_routing_decision(Backend::Cpu, 1).unwrap_err();
        match cpu_err {
            BackendError::CpuLaneStub(msg) => {
                assert_eq!(
                    msg,
                    "CPU sampler lane scaffolded, not yet on critical path; see docs/wiki/CPU-Lane-Plan.md",
                    "cpu stub message drifted — update memory + docs together"
                );
            }
            other => panic!("expected CpuLaneStub, got {other:?}"),
        }
    }

    /// Model-aware routing: ternary model on Xdna must surface
    /// [`BackendError::NpuTernaryUnsupported`] with the AMD-upstream-wait
    /// message. Backend::Cpu + any model is unchanged — still
    /// `CpuLaneStub`. Backend::Hip is always OK regardless of model.
    #[test]
    fn prefill_routing_with_model_gates_ternary_on_xdna() {
        // Ternary + Xdna @ threshold → NpuTernaryUnsupported, pointing
        // ops at the memory note and the HIP fallback.
        let err = prefill_routing_decision_with_model(
            Backend::Xdna,
            XDNA_PREFILL_MIN_TOKENS,
            /* is_ternary */ true,
        )
        .expect_err("ternary on xdna must refuse");
        match err {
            BackendError::NpuTernaryUnsupported(msg) => {
                assert!(msg.contains("ternary"));
                assert!(msg.contains("project_lemonade_10_2_pivot.md"));
                assert!(msg.contains("HALO_BACKEND=hip"));
            }
            other => panic!("expected NpuTernaryUnsupported, got {other:?}"),
        }

        // Ternary + Hip → always OK regardless of prompt length.
        assert!(
            prefill_routing_decision_with_model(Backend::Hip, 4096, /* is_ternary */ true).is_ok(),
            "Hip + ternary must pass — we're supposed to run on the iGPU"
        );
        assert!(
            prefill_routing_decision_with_model(Backend::Hip, 1, /* is_ternary */ false).is_ok(),
            "Hip + non-ternary must also pass — Hip is backend-agnostic here"
        );

        // Backend::Cpu is unchanged: always CpuLaneStub, ignores
        // `is_ternary`. Regression guard: nobody accidentally re-routed
        // the CPU lane through FLM.
        for is_t in [true, false] {
            let err = prefill_routing_decision_with_model(Backend::Cpu, 1, is_t)
                .expect_err("cpu always stubs");
            assert!(
                matches!(err, BackendError::CpuLaneStub(_)),
                "Cpu + is_ternary={is_t} must stay CpuLaneStub, got {err:?}"
            );
        }

        // Short prompt on Xdna falls through regardless of model kind —
        // the HIP path picks it up.
        assert!(
            prefill_routing_decision_with_model(
                Backend::Xdna,
                XDNA_PREFILL_MIN_TOKENS - 1,
                /* is_ternary */ true,
            )
            .is_ok(),
            "short prompt must fall through to HIP even for ternary"
        );
        assert!(
            prefill_routing_decision_with_model(
                Backend::Xdna,
                XDNA_PREFILL_MIN_TOKENS - 1,
                /* is_ternary */ false,
            )
            .is_ok()
        );
    }

    /// Non-ternary + Xdna + long prompt → attempts FLM dispatch. Because
    /// `/usr/bin/flm` may or may not be present on the test host, we
    /// accept either (a) the feature-off stub `NotYetWired`, or (b) a
    /// live FLM outcome surfacing as `FlmSpawn` / `NotYetWired` from the
    /// `xdna_flm` path. The failure mode we explicitly reject is a
    /// panic or an `NpuTernaryUnsupported` — that would mean the
    /// ternary gate fired on a non-ternary input.
    #[test]
    fn prefill_routing_non_ternary_xdna_dispatches_to_flm() {
        // Point the FLM bridge at a definitely-nonexistent binary so
        // this test is deterministic regardless of whether
        // `/usr/bin/flm` is installed on the CI / dev host. Uses the
        // same env-var the module exposes for tests + ops dry-runs.
        let _g = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::set_var(xdna_flm::FLM_BINARY_ENV, "/nonexistent/flm-routing-test");
        }

        let err = prefill_routing_decision_with_model(
            Backend::Xdna,
            XDNA_PREFILL_MIN_TOKENS,
            /* is_ternary */ false,
        )
        .expect_err("flm bridge must surface an error today (KV handoff pending)");

        // Acceptable outcomes, by feature-flag state:
        //   feature ON  → FlmSpawn (binary missing at the override path)
        //   feature OFF → NotYetWired (compile-time stub)
        match &err {
            BackendError::FlmSpawn(_) | BackendError::NotYetWired(_) => {
                // good — subprocess path engaged, regardless of outcome
            }
            BackendError::NpuTernaryUnsupported(msg) => {
                panic!("ternary gate fired on is_ternary=false — bug! msg: {msg}");
            }
            other => panic!("unexpected error from non-ternary flm bridge: {other:?}"),
        }

        unsafe {
            std::env::remove_var(xdna_flm::FLM_BINARY_ENV);
        }
    }

    /// Regression guard: [`Backend::Cpu`] policy must be untouched by
    /// the FLM bridge changes. No routing, no subprocess, no flag
    /// sensitivity — always `CpuLaneStub` with the canonical message.
    #[test]
    fn cpu_backend_unchanged_by_flm_bridge() {
        // Every prompt length + both model kinds + both signatures
        // (legacy + model-aware) must return CpuLaneStub.
        for tokens in [0usize, 1, 32, XDNA_PREFILL_MIN_TOKENS, 4096] {
            let err = prefill_routing_decision(Backend::Cpu, tokens)
                .expect_err("legacy cpu always stubs");
            assert!(matches!(err, BackendError::CpuLaneStub(_)));

            for is_t in [true, false] {
                let err2 = prefill_routing_decision_with_model(Backend::Cpu, tokens, is_t)
                    .expect_err("model-aware cpu always stubs");
                match err2 {
                    BackendError::CpuLaneStub(msg) => {
                        assert!(
                            msg.contains("CPU sampler lane"),
                            "cpu stub message must remain stable; got: {msg}"
                        );
                    }
                    other => panic!("Cpu must stay CpuLaneStub, got {other:?}"),
                }
            }
        }
    }

    /// Label round-trip: every variant must render a stable, unique string
    /// for operator logs + `/v1/models`.
    #[test]
    fn backend_labels_are_unique() {
        let hip = Backend::Hip.label();
        let xdna = Backend::Xdna.label();
        let cpu = Backend::Cpu.label();
        assert_ne!(hip, xdna);
        assert_ne!(xdna, cpu);
        assert_ne!(hip, cpu);
        // And parse-env round-trips through the canonical lowercase label.
        assert_eq!(Backend::parse_env(hip).unwrap(), Backend::Hip);
        assert_eq!(Backend::parse_env(xdna).unwrap(), Backend::Xdna);
        assert_eq!(Backend::parse_env(cpu).unwrap(), Backend::Cpu);
    }
}
