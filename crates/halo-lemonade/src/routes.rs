//! Axum router — OpenAI + Lemonade-SDK-compat surface.
//!
//! Routes:
//! * `GET  /_health`              — plain-text "ok"
//! * `GET  /api/v1/health`        — `{ "status": "ok" }` (Lemonade SDK probe)
//! * `GET  /v1/models`            — OpenAI-shape model list
//! * `GET  /api/v0/models`        — Lemonade-shape alias of /v1/models
//! * `POST /v1/chat/completions`  — proxied to halo-server upstream
//! * `POST /v1/completions`       — proxied to halo-server upstream
//!
//! Proxy strategy: we forward the raw JSON body (minus any client-supplied
//! `Host` / `Authorization` headers) to the upstream halo-server URL and
//! stream the response back. Streaming passes SSE chunks through unmodified
//! so clients see byte-for-byte the same event stream they'd see hitting
//! halo-server directly.

use axum::{
    Json, Router,
    body::Body,
    extract::State,
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde_json::{Value, json};
use std::sync::Arc;

use crate::registry::ModelRegistry;

#[derive(Clone)]
pub struct AppState {
    pub registry: Arc<ModelRegistry>,
    /// halo-server URL for chat/completions proxying. No trailing slash.
    pub upstream: Arc<String>,
    /// Shared reqwest client so connection pooling works across requests.
    pub http: reqwest::Client,
}

impl AppState {
    pub fn new(registry: ModelRegistry, upstream: impl Into<String>) -> Self {
        Self {
            registry: Arc::new(registry),
            upstream: Arc::new(upstream.into()),
            http: reqwest::Client::builder()
                .user_agent("halo-lemonade/0.1")
                .build()
                .expect("reqwest client"),
        }
    }
}

pub fn build(state: AppState) -> Router {
    Router::new()
        .route("/_health", get(|| async { (StatusCode::OK, "ok") }))
        .route("/api/v1/health", get(lemonade_health))
        .route("/v1/models", get(list_models))
        .route("/api/v0/models", get(list_models))
        .route("/v1/chat/completions", post(proxy_chat))
        .route("/v1/completions", post(proxy_completion))
        .with_state(state)
}

async fn lemonade_health() -> Json<Value> {
    Json(json!({ "status": "ok", "engine": "halo-lemonade" }))
}

async fn list_models(State(s): State<AppState>) -> Json<Value> {
    let data: Vec<Value> = s
        .registry
        .ids()
        .into_iter()
        .map(|id| {
            let entry = s.registry.get(id).unwrap();
            json!({
                "id": id,
                "object": "model",
                "owned_by": entry.backend.split('/').next().unwrap_or("halo-ai"),
                "capabilities": entry.capabilities,
            })
        })
        .collect();
    Json(json!({ "object": "list", "data": data }))
}

async fn proxy_chat(state: State<AppState>, headers: HeaderMap, body: Body) -> Response {
    proxy_to(state, headers, body, "/v1/chat/completions").await
}

async fn proxy_completion(state: State<AppState>, headers: HeaderMap, body: Body) -> Response {
    proxy_to(state, headers, body, "/v1/completions").await
}

async fn proxy_to(
    State(s): State<AppState>,
    headers: HeaderMap,
    body: Body,
    path: &'static str,
) -> Response {
    let url = format!("{}{path}", s.upstream);
    let bytes = match axum::body::to_bytes(body, 16 * 1024 * 1024).await {
        Ok(b) => b,
        Err(e) => return (StatusCode::BAD_REQUEST, format!("body read: {e}")).into_response(),
    };

    // Forward only a safe subset of headers. Drop Host / Authorization
    // (the upstream bearer is ours, not the caller's) / Content-Length
    // (reqwest sets it from the bytes).
    let mut forward = reqwest::header::HeaderMap::new();
    if let Some(ct) = headers.get(header::CONTENT_TYPE) {
        if let Ok(v) = reqwest::header::HeaderValue::from_bytes(ct.as_bytes()) {
            forward.insert(reqwest::header::CONTENT_TYPE, v);
        }
    }

    let resp = match s.http.post(&url).headers(forward).body(bytes.to_vec()).send().await {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(%url, "upstream post failed: {e}");
            return (StatusCode::BAD_GATEWAY, format!("upstream: {e}")).into_response();
        }
    };

    let status =
        StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let resp_ct = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .cloned()
        .unwrap_or(reqwest::header::HeaderValue::from_static("application/json"));

    // Stream the body through unmodified so SSE chat/completions arrive
    // chunk-by-chunk (flush_interval honoured by the caller's proxy).
    let stream = resp.bytes_stream();
    let ct_str = resp_ct.to_str().unwrap_or("application/json").to_string();
    (
        status,
        [(header::CONTENT_TYPE, ct_str)],
        Body::from_stream(stream),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::ModelEntry;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    fn test_state() -> AppState {
        let mut r = ModelRegistry::new();
        r.insert(
            "halo-1bit-2b",
            ModelEntry::new("local/bitnet-hip", vec!["chat".into(), "completion".into()]),
        );
        AppState::new(r, "http://127.0.0.1:65535")
    }

    #[tokio::test]
    async fn health_ok() {
        let app = build(test_state());
        let res = app
            .oneshot(Request::builder().uri("/_health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn lemonade_health_shape() {
        let app = build(test_state());
        let res = app
            .oneshot(Request::builder().uri("/api/v1/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let body = axum::body::to_bytes(res.into_body(), 4096).await.unwrap();
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["status"], "ok");
        assert_eq!(v["engine"], "halo-lemonade");
    }

    #[tokio::test]
    async fn models_lists_registered() {
        let app = build(test_state());
        let res = app
            .oneshot(Request::builder().uri("/v1/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let body = axum::body::to_bytes(res.into_body(), 4096).await.unwrap();
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["object"], "list");
        assert_eq!(v["data"][0]["id"], "halo-1bit-2b");
    }

    #[tokio::test]
    async fn lemonade_models_alias() {
        let app = build(test_state());
        let res = app
            .oneshot(Request::builder().uri("/api/v0/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let body = axum::body::to_bytes(res.into_body(), 4096).await.unwrap();
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["data"][0]["id"], "halo-1bit-2b");
    }

    #[tokio::test]
    async fn empty_registry_returns_empty_list() {
        let mut s = test_state();
        s.registry = Arc::new(ModelRegistry::new());
        let app = build(s);
        let res = app
            .oneshot(Request::builder().uri("/v1/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let body = axum::body::to_bytes(res.into_body(), 4096).await.unwrap();
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["data"].as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn chat_proxy_upstream_down_returns_bad_gateway() {
        // Upstream is 127.0.0.1:65535, unreachable — expect 502.
        let app = build(test_state());
        let body = r#"{"model":"halo-1bit-2b","messages":[{"role":"user","content":"hi"}]}"#;
        let res = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("Content-Type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::BAD_GATEWAY);
    }
}
