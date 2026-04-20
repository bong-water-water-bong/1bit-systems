//! Axum router — OpenAI + Lemonade-SDK-compat surface.
//!
//! Routes:
//! * `GET  /_health`              — plain-text "ok"
//! * `GET  /api/v1/health`        — `{ "status": "ok" }` (Lemonade SDK probe)
//! * `GET  /v1/models`            — OpenAI-shape model list
//! * `GET  /api/v0/models`        — Lemonade-shape alias of /v1/models
//! * `POST /v1/chat/completions`  — dispatched to the active [`Upstream`]
//! * `POST /v1/completions`       — dispatched to the active [`Upstream`]
//! * `GET  /metrics`              — Prometheus text exposition
//!
//! Proxy strategy: we forward the raw JSON body (minus any client-supplied
//! `Host` / `Authorization` headers) through the [`Upstream`] trait and
//! stream the response back. Streaming passes SSE chunks through unmodified
//! so clients see byte-for-byte the same event stream they'd see hitting
//! halo-server directly.
//!
//! Dispatch abstraction: the routes here never reach for reqwest or a
//! hardcoded URL — they hold an `Arc<dyn Upstream>` and every dispatched
//! call is instrumented in [`crate::metrics::Metrics`] so `/metrics`
//! returns live counters + histograms + an upstream-up gauge.

use axum::{
    Json, Router,
    body::Body,
    extract::State,
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use bytes::Bytes;
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::Instant;

use crate::dispatch::{HaloServer, Upstream, UpstreamRequest};
use crate::metrics::Metrics;
use crate::registry::ModelRegistry;

#[derive(Clone)]
pub struct AppState {
    pub registry: Arc<ModelRegistry>,
    /// Runtime dispatch target. Object-safe trait object so we can swap
    /// halo-server for lemond / flm / a test mock without touching any
    /// handler code. Invariant 3.
    pub upstream: Arc<dyn Upstream>,
    /// Process-wide Prometheus metrics. One per AppState, cloned by `Arc`.
    pub metrics: Arc<Metrics>,
}

impl AppState {
    /// Legacy constructor kept so `main.rs` + existing tests don't have to
    /// thread the trait. Parses `upstream_url` into a [`HaloServer`]; if
    /// parsing fails we log and build a `HaloServer` pointing at the
    /// canonical default, which satisfies invariant 5 (fail open).
    pub fn new(registry: ModelRegistry, upstream_url: impl AsRef<str>) -> Self {
        let upstream = HaloServer::new(upstream_url.as_ref()).unwrap_or_else(|e| {
            tracing::warn!(
                "failed to parse upstream url {:?}: {e}; falling back to 127.0.0.1:8180",
                upstream_url.as_ref()
            );
            HaloServer::new("http://127.0.0.1:8180").expect("default upstream always parses")
        });
        Self::with_upstream(registry, Arc::new(upstream))
    }

    /// Construct with a caller-supplied [`Upstream`] impl. This is the path
    /// used by tests (to inject a mock) and by any future main that wants
    /// a non-halo-server backend.
    pub fn with_upstream(registry: ModelRegistry, upstream: Arc<dyn Upstream>) -> Self {
        Self {
            registry: Arc::new(registry),
            upstream,
            metrics: Arc::new(Metrics::new()),
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
        .route("/metrics", get(metrics_text))
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
    dispatch_post(state, headers, body, "/v1/chat/completions", true).await
}

async fn proxy_completion(state: State<AppState>, headers: HeaderMap, body: Body) -> Response {
    dispatch_post(state, headers, body, "/v1/completions", false).await
}

/// Shared POST dispatcher for the two /v1 completion routes.
///
/// Reads the request body once, hands it to the [`Upstream`] trait, and
/// maps the [`crate::dispatch::UpstreamResponse`] back to an axum
/// [`Response`]. Records one metric sample per request regardless of
/// outcome.
async fn dispatch_post(
    State(s): State<AppState>,
    headers: HeaderMap,
    body: Body,
    route: &'static str,
    is_chat: bool,
) -> Response {
    let started = Instant::now();
    let bytes = match axum::body::to_bytes(body, 16 * 1024 * 1024).await {
        Ok(b) => b,
        Err(e) => {
            s.metrics.record(route, 400, started.elapsed());
            return (StatusCode::BAD_REQUEST, format!("body read: {e}")).into_response();
        }
    };

    let ct = headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(String::from);

    let req = UpstreamRequest {
        content_type: ct,
        body: Bytes::from(bytes.to_vec()),
    };

    let result = if is_chat {
        s.upstream.chat_completions(req).await
    } else {
        s.upstream.completions(req).await
    };

    match result {
        Ok(resp) => {
            let status = StatusCode::from_u16(resp.status).unwrap_or(StatusCode::BAD_GATEWAY);
            s.metrics.record(route, resp.status, started.elapsed());
            (
                status,
                [(header::CONTENT_TYPE, resp.content_type)],
                resp.body,
            )
                .into_response()
        }
        Err(e) => {
            tracing::warn!(route, "upstream error: {e}");
            s.metrics.record(route, 502, started.elapsed());
            (StatusCode::BAD_GATEWAY, format!("upstream: {e}")).into_response()
        }
    }
}

/// `GET /metrics` — Prometheus text exposition. Probes the upstream first
/// so the `halo_lemonade_upstream_up` gauge is fresh for this scrape.
async fn metrics_text(State(s): State<AppState>) -> Response {
    let up = s.upstream.health().await;
    s.metrics.set_upstream_up(up);
    let body = s.metrics.render();
    (
        StatusCode::OK,
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        body,
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch::{Upstream, UpstreamResponse};
    use crate::registry::ModelEntry;
    use async_trait::async_trait;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    /// Mock upstream used by the `/metrics` + dispatch-wiring tests.
    #[derive(Debug)]
    struct MockUpstream {
        status: u16,
        body: String,
        healthy: bool,
    }

    #[async_trait]
    impl Upstream for MockUpstream {
        fn name(&self) -> &str {
            "mock"
        }
        async fn chat_completions(
            &self,
            _req: UpstreamRequest,
        ) -> anyhow::Result<UpstreamResponse> {
            Ok(UpstreamResponse {
                status: self.status,
                content_type: "application/json".into(),
                body: Body::from(self.body.clone()),
            })
        }
        async fn completions(&self, _req: UpstreamRequest) -> anyhow::Result<UpstreamResponse> {
            Ok(UpstreamResponse {
                status: self.status,
                content_type: "application/json".into(),
                body: Body::from(self.body.clone()),
            })
        }
        async fn models(&self) -> anyhow::Result<UpstreamResponse> {
            Ok(UpstreamResponse {
                status: 200,
                content_type: "application/json".into(),
                body: Body::from(r#"{"object":"list","data":[]}"#),
            })
        }
        async fn health(&self) -> bool {
            self.healthy
        }
    }

    fn test_state() -> AppState {
        let mut r = ModelRegistry::new();
        r.insert(
            "halo-1bit-2b",
            ModelEntry::new("local/bitnet-hip", vec!["chat".into(), "completion".into()]),
        );
        // 127.0.0.1:65535 is our "dead upstream" — HaloServer::new is fine
        // with it (URL is well-formed) but any real request will fail, so
        // existing BAD_GATEWAY test still exercises the real dispatch path.
        AppState::new(r, "http://127.0.0.1:65535")
    }

    fn mock_state(status: u16, body: &str) -> AppState {
        let mut r = ModelRegistry::new();
        r.insert(
            "halo-1bit-2b",
            ModelEntry::new("local/bitnet-hip", vec!["chat".into()]),
        );
        let mock: Arc<dyn Upstream> = Arc::new(MockUpstream {
            status,
            body: body.to_string(),
            healthy: true,
        });
        AppState::with_upstream(r, mock)
    }

    #[tokio::test]
    async fn health_ok() {
        let app = build(test_state());
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/_health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn lemonade_health_shape() {
        let app = build(test_state());
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/health")
                    .body(Body::empty())
                    .unwrap(),
            )
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
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
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
            .oneshot(
                Request::builder()
                    .uri("/api/v0/models")
                    .body(Body::empty())
                    .unwrap(),
            )
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
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
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

    /// The /metrics endpoint returns a 200 with Prometheus text exposition
    /// (`# TYPE` markers for counter, histogram, and gauge).
    #[tokio::test]
    async fn metrics_endpoint_serves_prometheus_text() {
        let app = build(mock_state(200, r#"{"ok":true}"#));
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let ct = res
            .headers()
            .get(header::CONTENT_TYPE)
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
        assert!(ct.starts_with("text/plain"), "content-type = {ct}");
        let body = axum::body::to_bytes(res.into_body(), 16_384).await.unwrap();
        let text = std::str::from_utf8(&body).unwrap();
        assert!(
            text.contains("# TYPE halo_lemonade_requests_total counter"),
            "missing counter TYPE marker: {text}"
        );
        assert!(
            text.contains("# TYPE halo_lemonade_request_seconds histogram"),
            "missing histogram TYPE marker: {text}"
        );
        assert!(
            text.contains("# TYPE halo_lemonade_upstream_up gauge"),
            "missing gauge TYPE marker: {text}"
        );
    }

    /// After a successful POST through the mock upstream, /metrics shows a
    /// non-zero sample for the `/v1/chat/completions` route with status
    /// 200. This is the integration-level proof that the dispatch trait
    /// and the metrics layer are wired together.
    #[tokio::test]
    async fn dispatch_records_metrics_for_each_call() {
        let state = mock_state(200, r#"{"id":"cmpl-1","choices":[]}"#);
        let app = build(state.clone());

        // Fire one chat request.
        let res = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("Content-Type", "application/json")
                    .body(Body::from(r#"{"model":"halo-1bit-2b","messages":[]}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);

        let res = app
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let body = axum::body::to_bytes(res.into_body(), 16_384).await.unwrap();
        let text = std::str::from_utf8(&body).unwrap();
        assert!(
            text.contains(
                "halo_lemonade_requests_total{route=\"/v1/chat/completions\",status=\"200\"} 1"
            ),
            "expected counter line for successful POST, got:\n{text}"
        );
        assert!(
            text.contains("halo_lemonade_upstream_up 1"),
            "expected upstream up after healthy mock:\n{text}"
        );
    }

    /// Prove the dispatch trait is actually swappable without touching the
    /// route layer. Same routes, two different backends, two different
    /// bodies — the route code doesn't care.
    #[tokio::test]
    async fn dispatch_swappable_backends() {
        #[derive(Debug)]
        struct BackendA;
        #[derive(Debug)]
        struct BackendB;

        #[async_trait]
        impl Upstream for BackendA {
            fn name(&self) -> &str {
                "a"
            }
            async fn chat_completions(
                &self,
                _req: UpstreamRequest,
            ) -> anyhow::Result<UpstreamResponse> {
                Ok(UpstreamResponse {
                    status: 200,
                    content_type: "application/json".into(),
                    body: Body::from(r#"{"backend":"a"}"#),
                })
            }
            async fn completions(
                &self,
                _req: UpstreamRequest,
            ) -> anyhow::Result<UpstreamResponse> {
                unreachable!()
            }
            async fn models(&self) -> anyhow::Result<UpstreamResponse> {
                unreachable!()
            }
            async fn health(&self) -> bool {
                true
            }
        }

        #[async_trait]
        impl Upstream for BackendB {
            fn name(&self) -> &str {
                "b"
            }
            async fn chat_completions(
                &self,
                _req: UpstreamRequest,
            ) -> anyhow::Result<UpstreamResponse> {
                Ok(UpstreamResponse {
                    status: 200,
                    content_type: "application/json".into(),
                    body: Body::from(r#"{"backend":"b"}"#),
                })
            }
            async fn completions(
                &self,
                _req: UpstreamRequest,
            ) -> anyhow::Result<UpstreamResponse> {
                unreachable!()
            }
            async fn models(&self) -> anyhow::Result<UpstreamResponse> {
                unreachable!()
            }
            async fn health(&self) -> bool {
                true
            }
        }

        let mut reg = ModelRegistry::new();
        reg.insert(
            "x",
            ModelEntry::new("local/x", vec!["chat".into()]),
        );
        let state_a = AppState::with_upstream(reg.clone(), Arc::new(BackendA));
        let state_b = AppState::with_upstream(reg, Arc::new(BackendB));

        // Same axum::build call — zero delta in caller code between
        // backends.
        for (state, expected) in [(state_a, "a"), (state_b, "b")] {
            let app = build(state);
            let res = app
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/v1/chat/completions")
                        .header("Content-Type", "application/json")
                        .body(Body::from(r#"{"model":"x","messages":[]}"#))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(res.status(), StatusCode::OK);
            let body = axum::body::to_bytes(res.into_body(), 4096).await.unwrap();
            let v: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(v["backend"], expected);
        }
    }
}
