//! Axum router + OpenAI-compatible handlers.
//!
//! Endpoints:
//! * `GET  /healthz`               — always 200, plain text.
//! * `GET  /v1/models`             — backend-supplied model list.
//! * `POST /v1/chat/completions`   — OpenAI chat, SSE when `stream: true`.
//! * `POST /v1/completions`        — classic text completion, SSE when
//!   `stream: true`.
//!
//! SSE wire format matches the legacy C++ `bitnet_decode --server`:
//! each event is `data: {json}\n\n`, the first chunk carries the role
//! (`"role":"assistant"` for chat), subsequent chunks carry `content`
//! deltas, the last chunk carries `finish_reason: "stop"` + empty delta,
//! and the stream is terminated with `data: [DONE]\n\n`.

use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use axum::Json;
use axum::Router;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use futures::StreamExt;
use futures::stream::Stream;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use uuid::Uuid;

use crate::api::{
    ChatChoice, ChatChunkChoice, ChatCompletionChunk, ChatCompletionRequest,
    ChatCompletionResponse, ChatDelta, ChatMessage, CompletionChoice, CompletionChunk,
    CompletionChunkChoice, CompletionRequest, CompletionResponse, ImageGenRequest, ModelList,
    PplRequest, PplResponse,
};
use crate::backend::{GenerationParams, SharedBackend};
use crate::error::ServerError;
use crate::metrics::{Metrics, MetricsSnapshot};

/// Axum shared state — the backend plus a metrics handle. Kept `Clone`
/// because axum stores it behind `with_state` which requires `Clone`.
#[derive(Clone)]
pub struct AppState {
    pub backend: SharedBackend,
    pub metrics: Arc<Metrics>,
    /// Base URL for the Stable Diffusion sidecar server (sd-server).
    /// Layer-A image proxy forwards to `{sd_base_url}/v1/images/generations`.
    /// Plumbed from `$HALO_SD_URL` at `main.rs` startup, default
    /// `http://127.0.0.1:8081`.
    pub sd_base_url: Arc<String>,
    /// Reqwest client for upstream forwarding (image proxy today, could
    /// grow more upstreams later). Held as `Arc` so the 900-second timeout
    /// config is set once and shared across requests.
    pub http_client: reqwest::Client,
}

// ─── Router assembly ─────────────────────────────────────────────────────

/// Build the axum `Router` with all routes, CORS, and tracing wired up.
///
/// The returned router is stateful (an [`AppState`] lives in `State<_>`) but
/// fully `Send + 'static`, so callers can hand it straight to
/// `axum::serve`.
///
/// This convenience form constructs a fresh [`Metrics`] — use
/// [`build_router_with_state`] to share a metrics handle across multiple
/// routers (or to inspect it externally).
pub fn build_router(backend: SharedBackend) -> Router {
    build_router_with_state(AppState {
        backend,
        metrics: Arc::new(Metrics::new()),
        sd_base_url: Arc::new("http://127.0.0.1:8081".to_string()),
        http_client: default_http_client(),
    })
}

/// Build the reqwest client used for upstream forwarding (sd-server today).
///
/// 900 s timeout matches the Caddy reverse-proxy `transport http { …
/// response_header_timeout 15m }` on the strixhalo edge — SDXL at
/// 1024×1024/30 steps takes ~60 s but we allow headroom for batched
/// requests and cold starts.
pub fn default_http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(900))
        .build()
        .expect("reqwest::Client with 900s timeout must build")
}

/// Same as [`build_router`] but takes a pre-built [`AppState`].
pub fn build_router_with_state(state: AppState) -> Router {
    // Permissive CORS for now; tighten to the tailnet + localhost once the
    // webui stabilises. See `project_halo_network` memory for the cidrs.
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .route("/healthz", get(healthz))
        .route("/health", get(healthz)) // C++ compatibility
        .route("/metrics", get(metrics_handler))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/npu/status", get(crate::npu::npu_status))
        // Layer-A image generation proxy → sd-server :8081. `/v2` is the
        // canonical gen-2 route (Caddy's `/v2/*` block already routes here);
        // `/v1` alias keeps parity with the legacy OpenAI surface so plain
        // OpenAI SDKs work without a path override.
        .route("/v2/images/generations", post(images_generations))
        .route("/v1/images/generations", post(images_generations))
        .route("/ppl", post(ppl))
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

// ─── Handlers ────────────────────────────────────────────────────────────

async fn healthz() -> impl IntoResponse {
    (StatusCode::OK, "ok\n")
}

async fn metrics_handler(State(s): State<AppState>) -> Json<MetricsSnapshot> {
    Json(s.metrics.snapshot())
}

async fn list_models(State(s): State<AppState>) -> Json<ModelList> {
    Json(ModelList {
        object: "list",
        data: s.backend.list_models(),
    })
}

/// `POST /v1/chat/completions` — streaming or non-streaming per request.
async fn chat_completions(
    State(s): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, ServerError> {
    if req.messages.is_empty() {
        return Err(ServerError::BadRequest(
            "messages array must not be empty".into(),
        ));
    }

    let params = GenerationParams {
        model: req.model.clone(),
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        frequency_penalty: req.frequency_penalty,
        presence_penalty: req.presence_penalty,
    };

    let id = chat_id();
    let created = now_unix();
    let started = Instant::now();

    if req.stream {
        // Streaming path: we don't see the final usage until the stream
        // drains. Wrap the inner stream so we can tally tokens as they
        // flow past and fire record_request at end-of-stream.
        let stream = s.backend.generate_stream(&req.messages, &params).await?;
        let prompt_tokens = approx_prompt_tokens(&req.messages);
        let counted = count_and_report_stream(stream, s.metrics.clone(), prompt_tokens, started);
        let sse = chat_sse_stream(counted, id, created, req.model);
        return Ok(Sse::new(sse)
            .keep_alive(KeepAlive::default())
            .into_response());
    }

    let (text, usage) = s.backend.generate(&req.messages, &params).await?;
    s.metrics.record_request(
        usage.prompt_tokens,
        usage.completion_tokens,
        started.elapsed(),
    );
    let resp = ChatCompletionResponse {
        id,
        object: "chat.completion",
        created,
        model: req.model,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".into(),
                content: text,
            },
            finish_reason: "stop",
        }],
        usage: usage.into_api(),
    };
    Ok(Json(resp).into_response())
}

/// `POST /v1/completions` — classic text completion (prompt in, text out).
///
/// We implement this by synthesising a single `user` message and
/// delegating to the backend. That matches the semantic of every existing
/// halo backend (the legacy C++ server only had chat anyway) and keeps
/// the surface area small.
async fn completions(
    State(s): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> Result<Response, ServerError> {
    let prompt = req.prompt.first().to_string();
    if prompt.is_empty() {
        return Err(ServerError::BadRequest("prompt must not be empty".into()));
    }

    let params = GenerationParams {
        model: req.model.clone(),
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        ..Default::default()
    };

    let synth = vec![ChatMessage {
        role: "user".into(),
        content: prompt,
    }];
    let id = cmpl_id();
    let created = now_unix();
    let started = Instant::now();

    if req.stream {
        let stream = s.backend.generate_stream(&synth, &params).await?;
        let prompt_tokens = approx_prompt_tokens(&synth);
        let counted = count_and_report_stream(stream, s.metrics.clone(), prompt_tokens, started);
        let sse = completion_sse_stream(counted, id, created, req.model);
        return Ok(Sse::new(sse)
            .keep_alive(KeepAlive::default())
            .into_response());
    }

    let (text, usage) = s.backend.generate(&synth, &params).await?;
    s.metrics.record_request(
        usage.prompt_tokens,
        usage.completion_tokens,
        started.elapsed(),
    );
    let resp = CompletionResponse {
        id,
        object: "text_completion",
        created,
        model: req.model,
        choices: vec![CompletionChoice {
            index: 0,
            text,
            finish_reason: "stop",
        }],
        usage: usage.into_api(),
    };
    Ok(Json(resp).into_response())
}

/// `POST /ppl` — perplexity harness (parity check against gen-1 C++
/// `bitnet_decode --ppl`).
///
/// Body: `{text, stride, max_tokens}`. Response: `{mean_nll, perplexity,
/// tokens, elapsed_ms}`. Delegates to [`InferenceBackend::perplexity`],
/// which in the default `EchoBackend` build returns a clear "not
/// supported" error — only the `--features real-backend` build with a
/// loaded `--model` answers for real.
async fn ppl(
    State(s): State<AppState>,
    Json(req): Json<PplRequest>,
) -> Result<Json<PplResponse>, ServerError> {
    if req.text.is_empty() {
        return Err(ServerError::BadRequest(
            "ppl: text must not be empty".into(),
        ));
    }
    let out = s
        .backend
        .perplexity(req.text, req.stride, req.max_tokens)
        .await?;
    Ok(Json(PplResponse {
        mean_nll: out.mean_nll,
        perplexity: out.perplexity,
        tokens: out.tokens,
        elapsed_ms: out.elapsed_ms,
    }))
}

/// `POST /v2/images/generations` (and `/v1/...` alias) — Layer A image
/// proxy.
///
/// Forwards the OpenAI-compat request body verbatim to
/// `{sd_base_url}/v1/images/generations` (sd-server already speaks this
/// shape), then echoes the upstream 200 JSON response back to the caller.
///
/// Validation is minimal on purpose — sd-server is the authority on
/// sampler knobs, size constraints, and negative prompts. The only thing
/// we reject locally is an empty prompt, because sd-server's error
/// message for that case is opaque ("assertion failed: !prompt.empty()")
/// and surfacing the 400 here gives clients a cleaner signal.
///
/// Defaulting: `n → 1`, `size → "1024x1024"`, `output_format → "png"`.
/// Everything else passes through unchanged via the `extra` flatten
/// field on `ImageGenRequest`.
///
/// Any non-2xx upstream response becomes `ServerError::Upstream(..)`
/// → HTTP 502, carrying the upstream status + body prefix in the
/// message for debugging.
async fn images_generations(
    State(s): State<AppState>,
    Json(req): Json<ImageGenRequest>,
) -> Result<Response, ServerError> {
    if req.prompt.trim().is_empty() {
        return Err(ServerError::BadRequest(
            "prompt must not be empty".into(),
        ));
    }

    let url = format!("{}/v1/images/generations", s.sd_base_url.as_str());
    tracing::debug!(target = %url, n = req.n, size = %req.size, "forwarding image request to sd-server");

    let upstream = s
        .http_client
        .post(&url)
        .json(&req)
        .send()
        .await
        .map_err(|e| ServerError::Upstream(format!("sd-server POST {url}: {e}")))?;

    let status = upstream.status();
    if !status.is_success() {
        let body = upstream
            .text()
            .await
            .unwrap_or_else(|e| format!("<body read failed: {e}>"));
        let snippet: String = body.chars().take(512).collect();
        return Err(ServerError::Upstream(format!(
            "sd-server returned {status}: {snippet}"
        )));
    }

    // Forward the JSON body unchanged. We go through `bytes()` rather than
    // `json::<ImageGenResponse>()` so that any sd-server-specific extra
    // fields survive the round-trip (future-proofing for `seed`, `steps`,
    // etc. that we don't model today).
    let bytes = upstream
        .bytes()
        .await
        .map_err(|e| ServerError::Upstream(format!("sd-server body read: {e}")))?;

    Ok((
        StatusCode::OK,
        [(axum::http::header::CONTENT_TYPE, "application/json")],
        bytes,
    )
        .into_response())
}

// ─── Metrics hooks ───────────────────────────────────────────────────────

/// Whitespace-split token count across all messages — matches the crude
/// heuristic `EchoBackend` uses internally. Good enough for metrics while
/// we wait for a tokenizer-backed count in streaming mode.
fn approx_prompt_tokens(messages: &[ChatMessage]) -> u32 {
    messages
        .iter()
        .map(|m| m.content.split_whitespace().count() as u32)
        .sum()
}

/// Wrap a [`TokenStream`] so that we tally completion-token chunks as they
/// flow through and call [`Metrics::record_request`] exactly once at
/// end-of-stream (whether the stream drained normally, errored, or was
/// dropped mid-flight).
fn count_and_report_stream(
    inner: crate::backend::TokenStream,
    metrics: Arc<Metrics>,
    prompt_tokens: u32,
    started: Instant,
) -> crate::backend::TokenStream {
    // `Reporter` fires `record_request` from `Drop`, so partial / cancelled
    // streams still count. We accumulate chunks in a shared `AtomicU64` so
    // the async stream map can update without borrowing `Reporter` mutably.
    struct Reporter {
        metrics: Arc<Metrics>,
        prompt_tokens: u32,
        completion_tokens: Arc<AtomicU64>,
        started: Instant,
    }
    impl Drop for Reporter {
        fn drop(&mut self) {
            let completion = self.completion_tokens.load(Ordering::Relaxed) as u32;
            self.metrics
                .record_request(self.prompt_tokens, completion, self.started.elapsed());
        }
    }

    let completion_tokens = Arc::new(AtomicU64::new(0));
    let reporter = Arc::new(Reporter {
        metrics,
        prompt_tokens,
        completion_tokens: completion_tokens.clone(),
        started,
    });

    use futures::stream::StreamExt;
    let counter = completion_tokens.clone();
    let _keep_alive = reporter.clone(); // moved into the closure below
    let mapped = inner.map(move |item| {
        // Only count successful chunks. Empty strings (final SSE chunk
        // from the backend) still count as 1 "delta" but not a token, so
        // we treat "non-empty delta" as "one token" — matches what the
        // client sees at the SSE layer.
        if let Ok(ref s) = item {
            if !s.is_empty() {
                counter.fetch_add(1, Ordering::Relaxed);
            }
        }
        // Keep the reporter alive for the lifetime of the stream.
        let _ = &_keep_alive;
        item
    });
    Box::pin(mapped)
}

// ─── SSE stream builders ─────────────────────────────────────────────────

/// Wrap a backend delta stream as OpenAI chat-completion SSE events.
///
/// Emission order:
/// 1. Role-only opener with `{"role":"assistant"}` delta.
/// 2. One event per backend chunk with `{"content":"..."}` delta.
/// 3. Final event with empty delta + `finish_reason: "stop"`.
/// 4. `data: [DONE]` literal (emitted as a raw event; see below).
///
/// `axum::response::sse` serializes each `Event` as `event: …\ndata: …\n\n`
/// already; we use `.data(json_string)` and let it handle framing. The
/// `[DONE]` terminator is the one OpenAI-specific quirk — it's not JSON,
/// so we feed it in as a literal string.
fn chat_sse_stream(
    mut inner: crate::backend::TokenStream,
    id: String,
    created: i64,
    model: String,
) -> impl Stream<Item = Result<Event, Infallible>> {
    async_stream::stream! {
        // 1. Role opener.
        let opener = ChatCompletionChunk {
            id: id.clone(),
            object: "chat.completion.chunk",
            created,
            model: model.clone(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatDelta {
                    role: Some("assistant"),
                    content: None,
                },
                finish_reason: None,
            }],
        };
        yield Ok(Event::default().data(serde_json::to_string(&opener).unwrap()));

        // 2. Per-chunk deltas. Backend errors terminate the stream early;
        //    we surface them as an `error` event so the client sees
        //    something rather than a silent EOF.
        while let Some(item) = inner.next().await {
            match item {
                Ok(delta) => {
                    let chunk = ChatCompletionChunk {
                        id: id.clone(),
                        object: "chat.completion.chunk",
                        created,
                        model: model.clone(),
                        choices: vec![ChatChunkChoice {
                            index: 0,
                            delta: ChatDelta {
                                role: None,
                                content: Some(delta),
                            },
                            finish_reason: None,
                        }],
                    };
                    yield Ok(Event::default()
                        .data(serde_json::to_string(&chunk).unwrap()));
                }
                Err(e) => {
                    let err = serde_json::json!({
                        "error": { "message": e.to_string() }
                    });
                    yield Ok(Event::default().event("error").data(err.to_string()));
                    // [DONE] anyway so clients don't hang.
                    yield Ok(Event::default().data("[DONE]"));
                    return;
                }
            }
        }

        // 3. Final chunk with finish_reason.
        let fin = ChatCompletionChunk {
            id: id.clone(),
            object: "chat.completion.chunk",
            created,
            model: model.clone(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatDelta::default(),
                finish_reason: Some("stop"),
            }],
        };
        yield Ok(Event::default().data(serde_json::to_string(&fin).unwrap()));

        // 4. [DONE] sentinel.
        yield Ok(Event::default().data("[DONE]"));
    }
}

/// Same shape as `chat_sse_stream` but for `/v1/completions` (no role
/// field, top-level `text` instead of `delta.content`).
fn completion_sse_stream(
    mut inner: crate::backend::TokenStream,
    id: String,
    created: i64,
    model: String,
) -> impl Stream<Item = Result<Event, Infallible>> {
    async_stream::stream! {
        while let Some(item) = inner.next().await {
            match item {
                Ok(text) => {
                    let chunk = CompletionChunk {
                        id: id.clone(),
                        object: "text_completion",
                        created,
                        model: model.clone(),
                        choices: vec![CompletionChunkChoice {
                            index: 0,
                            text,
                            finish_reason: None,
                        }],
                    };
                    yield Ok(Event::default()
                        .data(serde_json::to_string(&chunk).unwrap()));
                }
                Err(e) => {
                    let err = serde_json::json!({
                        "error": { "message": e.to_string() }
                    });
                    yield Ok(Event::default().event("error").data(err.to_string()));
                    yield Ok(Event::default().data("[DONE]"));
                    return;
                }
            }
        }
        let fin = CompletionChunk {
            id: id.clone(),
            object: "text_completion",
            created,
            model: model.clone(),
            choices: vec![CompletionChunkChoice {
                index: 0,
                text: String::new(),
                finish_reason: Some("stop"),
            }],
        };
        yield Ok(Event::default().data(serde_json::to_string(&fin).unwrap()));
        yield Ok(Event::default().data("[DONE]"));
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────

fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn chat_id() -> String {
    format!("chatcmpl-{}", Uuid::new_v4().simple())
}

fn cmpl_id() -> String {
    format!("cmpl-{}", Uuid::new_v4().simple())
}

// ─── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::EchoBackend;
    use axum::body::{Body, to_bytes};
    use axum::http::{Request, StatusCode};
    use std::sync::Arc;
    use tower::ServiceExt; // for `oneshot`

    fn app() -> Router {
        build_router(Arc::new(EchoBackend::new()))
    }

    #[tokio::test]
    async fn healthz_ok() {
        let resp = app()
            .oneshot(
                Request::builder()
                    .uri("/healthz")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn list_models_shape() {
        let resp = app()
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["object"], "list");
        assert!(!v["data"].as_array().unwrap().is_empty());
        assert_eq!(v["data"][0]["object"], "model");
        assert_eq!(v["data"][0]["owned_by"], "1bit systems");
    }

    #[tokio::test]
    async fn chat_non_streaming() {
        let body = serde_json::json!({
            "model": "bitnet-b1.58-2b-4t",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": false,
        })
        .to_string();
        let resp = app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["object"], "chat.completion");
        assert_eq!(v["choices"][0]["message"]["role"], "assistant");
        assert_eq!(v["choices"][0]["message"]["content"], "stub");
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
        assert!(v["id"].as_str().unwrap().starts_with("chatcmpl-"));
    }

    #[tokio::test]
    async fn chat_streaming_sse_framing() {
        let body = serde_json::json!({
            "model": "bitnet-b1.58-2b-4t",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
        })
        .to_string();
        let resp = app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let ct = resp
            .headers()
            .get("content-type")
            .and_then(|h| h.to_str().ok())
            .unwrap_or("");
        assert!(
            ct.starts_with("text/event-stream"),
            "content-type should be SSE, got {ct:?}"
        );
        let bytes = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();

        // First frame carries the assistant role.
        assert!(
            text.contains("\"role\":\"assistant\""),
            "missing role opener in:\n{text}"
        );
        // EchoBackend streams "stub" char-by-char → at least 4 content frames.
        let content_frames = text.matches("\"content\":\"").count();
        assert!(
            content_frames >= 4,
            "expected at least 4 content deltas, got {content_frames}:\n{text}"
        );
        // Final frame carries finish_reason: stop.
        assert!(
            text.contains("\"finish_reason\":\"stop\""),
            "missing finish_reason in:\n{text}"
        );
        // And the OpenAI [DONE] terminator.
        assert!(text.contains("data: [DONE]"), "missing [DONE] in:\n{text}");
    }

    #[tokio::test]
    async fn completions_non_streaming() {
        let body = serde_json::json!({
            "model": "bitnet-b1.58-2b-4t",
            "prompt": "hello",
            "stream": false,
        })
        .to_string();
        let resp = app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["object"], "text_completion");
        assert_eq!(v["choices"][0]["text"], "stub");
    }

    #[tokio::test]
    async fn chat_empty_messages_400() {
        let body = serde_json::json!({
            "model": "bitnet-b1.58-2b-4t",
            "messages": [],
        })
        .to_string();
        let resp = app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let bytes = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "invalid_request_error");
    }

    #[tokio::test]
    async fn metrics_endpoint_shape_and_increment() {
        // Fresh router (and therefore fresh Metrics) — `requests` starts at 0.
        let state = AppState {
            backend: Arc::new(EchoBackend::new()),
            metrics: Arc::new(crate::metrics::Metrics::new()),
            sd_base_url: Arc::new("http://127.0.0.1:8081".to_string()),
            http_client: default_http_client(),
        };
        let app = build_router_with_state(state.clone());

        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 4 * 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        for k in [
            "requests",
            "generated_tokens",
            "uptime_secs",
            "tokps_recent",
            "p50_ms",
            "p95_ms",
            "completion_tokens_last_hour",
            "npu_up",
        ] {
            assert!(v.get(k).is_some(), "missing key {k} in {v}");
        }
        assert_eq!(v["requests"], 0);

        // Drive one chat completion through the router → counters bump.
        let body = serde_json::json!({
            "model": "bitnet-b1.58-2b-4t",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": false,
        })
        .to_string();
        let chat_resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(chat_resp.status(), StatusCode::OK);
        let _ = to_bytes(chat_resp.into_body(), 64 * 1024).await.unwrap();

        let resp2 = app
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let bytes2 = to_bytes(resp2.into_body(), 4 * 1024).await.unwrap();
        let v2: serde_json::Value = serde_json::from_slice(&bytes2).unwrap();
        assert_eq!(v2["requests"], 1, "requests should tick to 1, got {v2}");
        assert!(
            v2["generated_tokens"].as_u64().unwrap() >= 1,
            "generated_tokens should be positive: {v2}"
        );
    }

    #[tokio::test]
    async fn ppl_stub_returns_backend_error() {
        // EchoBackend doesn't implement real logits → the default
        // trait method fires and we get a 500 + JSON error body.
        let body = serde_json::json!({
            "text": "hello world how are you",
            "stride": 1024,
            "max_tokens": 1024,
        })
        .to_string();
        let resp = app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/ppl")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let bytes = to_bytes(resp.into_body(), 4 * 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "api_error");
        assert!(
            v["error"]["message"]
                .as_str()
                .unwrap_or("")
                .contains("ppl not supported"),
            "expected ppl-not-supported message, got {v}"
        );
    }

    #[tokio::test]
    async fn ppl_empty_text_is_400() {
        let body = serde_json::json!({ "text": "" }).to_string();
        let resp = app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/ppl")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    /// Fake backend whose `perplexity` returns a fixed, known result.
    /// Exercises the /ppl HTTP plumbing (request deserialization, handler
    /// dispatch, response serialization) without a GPU.
    #[derive(Default)]
    struct StubPplBackend;
    impl crate::backend::InferenceBackend for StubPplBackend {
        fn generate<'a>(
            &'a self,
            _messages: &'a [ChatMessage],
            _params: &'a crate::backend::GenerationParams,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<
                        Output = Result<(String, crate::backend::GenerationUsage), ServerError>,
                    > + Send
                    + 'a,
            >,
        > {
            Box::pin(async move { Ok(("ok".into(), crate::backend::GenerationUsage::default())) })
        }
        fn generate_stream<'a>(
            &'a self,
            _messages: &'a [ChatMessage],
            _params: &'a crate::backend::GenerationParams,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<Output = Result<crate::backend::TokenStream, ServerError>>
                    + Send
                    + 'a,
            >,
        > {
            Box::pin(async move {
                let s: crate::backend::TokenStream = Box::pin(futures::stream::empty());
                Ok(s)
            })
        }
        fn perplexity<'a>(
            &'a self,
            text: String,
            _stride: u32,
            _max_tokens: u32,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<
                        Output = Result<crate::backend::PerplexityOutput, ServerError>,
                    > + Send
                    + 'a,
            >,
        > {
            Box::pin(async move {
                // Deterministic fake numbers — mean_nll = 2.2149 chosen so
                // exp(mean_nll) ≈ 9.1607 matches the gen-1 baseline, which
                // makes this test also validate the ±0.05 tolerance used
                // by the ppl-gen2.sh script.
                Ok(crate::backend::PerplexityOutput {
                    mean_nll: 2.2149,
                    perplexity: 2.2149_f64.exp(),
                    tokens: text.len() as u32,
                    elapsed_ms: 1.0,
                })
            })
        }
    }

    #[tokio::test]
    async fn ppl_stub_backend_roundtrip() {
        let app = build_router(Arc::new(StubPplBackend));
        let body = serde_json::json!({
            "text": "the quick brown fox",
            "stride": 1024,
            "max_tokens": 1024,
        })
        .to_string();
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/ppl")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 4 * 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let mean_nll = v["mean_nll"].as_f64().unwrap();
        let ppl = v["perplexity"].as_f64().unwrap();
        assert!((mean_nll - 2.2149).abs() < 1e-9, "mean_nll={mean_nll}");
        assert!((ppl - 9.1607).abs() < 0.01, "ppl={ppl} (expected ~9.1607)");
        assert_eq!(v["tokens"].as_u64().unwrap(), 19);
    }

    /// End-to-end PPL check against a real ROCm-backed 1bit-server. Gated
    /// behind `#[ignore]` because it needs a live model load + GPU + the
    /// wikitext-103 test file on disk. Run with:
    ///
    /// ```sh
    /// cargo test --release --features real-backend -- \
    ///     ppl_live_wikitext --ignored --nocapture
    /// ```
    #[cfg(feature = "real-backend")]
    #[tokio::test]
    #[ignore]
    async fn ppl_live_wikitext() {
        use crate::backend::RealBackend;
        use std::path::PathBuf;

        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        let h1b = std::env::var("HALO_MODEL_H1B")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(format!("{home}/1bit systems/models/halo-1bit-2b.h1b"))
            });
        if !h1b.exists() {
            eprintln!("skipping: {} not found", h1b.display());
            return;
        }
        let wikitext_path = std::env::var("HALO_WIKITEXT")
            .unwrap_or_else(|_| format!("{home}/1bit systems/datasets/wikitext-103-test.txt"));
        let wikitext = std::fs::read_to_string(&wikitext_path).expect("wikitext file");
        let text: String = wikitext.chars().take(6_000).collect();

        let backend = Arc::new(RealBackend::new(&h1b).expect("real backend init"));
        let app = build_router(backend);
        let body = serde_json::json!({
            "text": text,
            "stride": 1024,
            "max_tokens": 1024,
        })
        .to_string();
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/ppl")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 4 * 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let ppl = v["perplexity"].as_f64().unwrap();
        println!("live wikitext PPL = {ppl:.4} (gen-1 baseline: 9.1607)");
        assert!(
            (ppl - 9.1607).abs() < 0.05,
            "gen-2 PPL diverged from gen-1: {ppl:.4} vs 9.1607"
        );
    }

    #[tokio::test]
    async fn echo_prefix_roundtrip() {
        let app = build_router(Arc::new(EchoBackend::with_prefix("echo:")));
        let body = serde_json::json!({
            "messages": [{"role": "user", "content": "hi there"}],
            "stream": false,
        })
        .to_string();
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["choices"][0]["message"]["content"], "echo:hi there");
    }
}
