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

pub use backend_impl::{BackendError, HipBackend};
pub use detect::{BackendKind, detect};

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
