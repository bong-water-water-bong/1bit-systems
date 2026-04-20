//! halo-router — the hardware-aware inference dispatcher for halo-ai.
//!
//! What this crate does:
//!
//! * At construction it **picks one** backend family (HIP on gfx1151 /
//!   MLX on Apple / error otherwise — see [`detect`]).
//! * It mmaps the `.h1b` model + parses the `.htok` tokenizer up front
//!   via `halo-core`, then hands the weight bytes to the backend (for
//!   HIP: `halo-bitnet-hip` uploads them once to the GPU).
//! * It exposes a small async surface — `generate`, `generate_stream`,
//!   `list_models` — that the HTTP layer wraps in an `InferenceBackend`
//!   adapter. **The trait itself lives in `halo-server`** so this crate
//!   stays free of HTTP types; see halo-server's `backend::RealBackend`
//!   for the adapter.
//!
//! Layering:
//!
//! ```text
//!      halo-server (HTTP)  ──▶  halo_router::Router
//!                                      │
//!                                      ├── detect() → BackendKind
//!                                      ├── HipBackend (weights + forward pass)
//!                                      └── halo_core::Sampler (host-side)
//! ```
//!
//! Thread model: the router is `Send + Sync` via a single `tokio::Mutex`
//! around the mutable backend state (KV cache + scratch are per-backend,
//! not per-request). Concurrent HTTP requests serialize behind the mutex;
//! this matches gen-1 behaviour — the C++ server also took a global
//! generation lock because the GPU runs one decode at a time.

#![warn(missing_docs)]

pub mod backend_impl;
pub mod detect;
pub mod tokenizer;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use halo_core::sampler::{Sampler, SamplerConfig};
use halo_core::types::TokenId;
use tokio::sync::Mutex;

pub use backend_impl::{BackendError, HipBackend, ModelFormat, sniff_model_format};
pub use detect::{BackendKind, detect};

// Re-exported below once the type is declared — see PerplexityResult.

/// Sampling + stopping parameters forwarded through the router.
///
/// Deliberately smaller than OpenAI's full request shape — halo-server
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
/// token accounting in the same shape halo-server will re-emit in its
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
}

struct Inner {
    backend: HipBackend,
    /// Last decoded position in the shared KV cache. Reset between
    /// conversations.
    pos: i32,
}

impl Router {
    /// Construct a router for the given `.h1b` model file. The tokenizer
    /// is resolved by taking the model path and replacing `.h1b` with
    /// `.htok`.
    ///
    /// Fails if no supported backend is available on this host.
    pub fn new(h1b_path: impl AsRef<Path>) -> Result<Self, BackendError> {
        let h1b_path = h1b_path.as_ref();
        let htok_path = default_htok_path(h1b_path);
        let model_id = default_model_id(h1b_path);

        Self::new_with(h1b_path, &htok_path, model_id, 4096)
    }

    /// Full constructor — lets the caller pin the tokenizer location,
    /// model id, and max KV cache size.
    pub fn new_with(
        h1b_path: &Path,
        htok_path: &Path,
        model_id: String,
        max_context: usize,
    ) -> Result<Self, BackendError> {
        let kind = detect();
        match kind {
            BackendKind::Hip => {
                let backend = HipBackend::new(h1b_path, htok_path, model_id.clone(), max_context)?;
                tracing::info!(
                    model_id,
                    backend = %kind.label(),
                    label = backend.label(),
                    "router ready"
                );
                Ok(Self {
                    inner: Arc::new(Mutex::new(Inner { backend, pos: 0 })),
                    model_id,
                    kind,
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

    /// Which backend family the router selected.
    pub fn backend_kind(&self) -> BackendKind {
        self.kind
    }

    /// Non-streaming generation. Blocks the calling task while the
    /// forward pass runs; call via `tokio::task::spawn_blocking` if the
    /// caller is on a shared runtime thread.
    pub async fn generate(&self, req: RouterRequest) -> Result<RouterResponse, BackendError> {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let mut guard = inner.blocking_lock();
            generate_blocking(&mut guard, req, None)
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
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<String, BackendError>>(64);
        tokio::task::spawn_blocking(move || {
            let mut guard = inner.blocking_lock();
            let tx_clone = tx.clone();
            let sink = move |delta: String| {
                // If the client hung up we cannot do anything useful; drop.
                let _ = tx_clone.blocking_send(Ok(delta));
            };
            if let Err(e) = generate_blocking(&mut guard, req, Some(Box::new(sink))) {
                let _ = tx.blocking_send(Err(e));
            }
        });
        Ok(tokio_stream::wrappers::ReceiverStream::new(rx))
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
            let _argmax = inner
                .backend
                .forward_token(tok, pos, &mut logits_scratch)?;
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
) -> Result<RouterResponse, BackendError> {
    // ---- Tokenize prompt ----
    let prompt_ids: Vec<TokenId> = inner.backend.tokenize(&req.prompt);
    if prompt_ids.is_empty() {
        return Err(BackendError::BadInput("empty prompt after tokenization"));
    }

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
        let next = inner.backend.forward_token(tok, inner.pos + i as i32, &mut logits_scratch)?;
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

    for step in 0..max_new {
        let pos = inner.pos + step as i32;
        let argmax_next = inner.backend.forward_token(cur, pos, &mut logits_scratch)?;

        // Optional host-side sampling path.
        let next = if req.sampler.temperature > 0.0 {
            sampler.sample(&mut logits_scratch, &history)?
        } else {
            argmax_next
        };

        history.push(next);
        cur = next;

        // Stop-token check before detokenizing: otherwise the special
        // token's string form (e.g. "<|eot_id|>") leaks into the output.
        if stop_ids.contains(&next) {
            stopped_on_eos = true;
            break;
        }

        generated_ids.push(next);

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

        if req.stop.iter().any(|s| !s.is_empty() && full_text.ends_with(s)) {
            stopped_on_eos = true;
            break;
        }
    }

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
            assert!((na - nb).abs() < 1e-6, "shift not invariant at t={t}: {na} vs {nb}");
        }
    }
}
