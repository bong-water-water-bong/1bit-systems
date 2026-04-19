//! Axum router + OpenAI-compatible handlers.
//!
//! Endpoints:
//! * `GET  /healthz`               — always 200, plain text.
//! * `GET  /v1/models`             — backend-supplied model list.
//! * `POST /v1/chat/completions`   — OpenAI chat, SSE when `stream: true`.
//! * `POST /v1/completions`        — classic text completion, SSE when
//!                                   `stream: true`.
//!
//! SSE wire format matches the legacy C++ `bitnet_decode --server`:
//! each event is `data: {json}\n\n`, the first chunk carries the role
//! (`"role":"assistant"` for chat), subsequent chunks carry `content`
//! deltas, the last chunk carries `finish_reason: "stop"` + empty delta,
//! and the stream is terminated with `data: [DONE]\n\n`.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

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
    CompletionChunkChoice, CompletionRequest, CompletionResponse, ModelList,
};
use crate::backend::{GenerationParams, InferenceBackend, SharedBackend};
// `InferenceBackend` is referenced by the generic bound on `build_router`
// below; the import is load-bearing even though handlers only see the
// `dyn`-erased `SharedBackend`.
use crate::error::ServerError;

// ─── Router assembly ─────────────────────────────────────────────────────

/// Build the axum `Router` with all routes, CORS, and tracing wired up.
///
/// The returned router is stateful (the backend lives in `State<_>`) but
/// fully `Send + 'static`, so callers can hand it straight to
/// `axum::serve`.
pub fn build_router<B: InferenceBackend>(backend: Arc<B>) -> Router {
    // Up-cast to the object-safe form so downstream handlers don't carry
    // the generic `B` parameter.
    let shared: SharedBackend = backend;

    // Permissive CORS for now; tighten to the tailnet + localhost once the
    // webui stabilises. See `project_halo_network` memory for the cidrs.
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .route("/healthz", get(healthz))
        .route("/health", get(healthz)) // C++ compatibility
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(shared)
}

// ─── Handlers ────────────────────────────────────────────────────────────

async fn healthz() -> impl IntoResponse {
    (StatusCode::OK, "ok\n")
}

async fn list_models(State(backend): State<SharedBackend>) -> Json<ModelList> {
    Json(ModelList {
        object: "list",
        data: backend.list_models(),
    })
}

/// `POST /v1/chat/completions` — streaming or non-streaming per request.
async fn chat_completions(
    State(backend): State<SharedBackend>,
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

    if req.stream {
        let stream = backend
            .generate_stream(&req.messages, &params)
            .await?;
        let sse = chat_sse_stream(stream, id, created, req.model);
        return Ok(Sse::new(sse)
            .keep_alive(KeepAlive::default())
            .into_response());
    }

    let (text, usage) = backend.generate(&req.messages, &params).await?;
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
    State(backend): State<SharedBackend>,
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

    if req.stream {
        let stream = backend.generate_stream(&synth, &params).await?;
        let sse = completion_sse_stream(stream, id, created, req.model);
        return Ok(Sse::new(sse)
            .keep_alive(KeepAlive::default())
            .into_response());
    }

    let (text, usage) = backend.generate(&synth, &params).await?;
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
    use tower::ServiceExt; // for `oneshot`

    fn app() -> Router {
        build_router(Arc::new(EchoBackend::new()))
    }

    #[tokio::test]
    async fn healthz_ok() {
        let resp = app()
            .oneshot(Request::builder().uri("/healthz").body(Body::empty()).unwrap())
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
        assert!(v["data"].as_array().unwrap().len() >= 1);
        assert_eq!(v["data"][0]["object"], "model");
        assert_eq!(v["data"][0]["owned_by"], "halo-ai");
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
