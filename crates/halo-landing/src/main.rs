//! `halo-landing` — static marketing page + live status probe on :8190.
//!
//! All assets are embedded at compile time via `include_str!`; there are
//! no filesystem reads at runtime. The only outbound calls are to
//! 127.0.0.1 — `/v1/models`, `/metrics`, `rocm-smi`, `xrt-smi`,
//! `systemctl --user`. See `docs/wiki/Crate-halo-landing.md` invariants.
//!
//! Routes:
//! * `GET /`              — embedded HTML
//! * `GET /style.css`     — embedded CSS
//! * `GET /logo.svg`      — embedded SVG
//! * `GET /_live/status`  — legacy one-shot JSON probe (kept for callers)
//! * `GET /_live/stats`   — SSE stream, one JSON event per ~1.5 s
//! * `GET /_live/services` — SSE stream, service-flip deltas only
//! * `GET /_health`       — plain text "ok"

mod status;
mod telemetry;

use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use axum::Json;
use axum::Router;
use axum::extract::State;
use axum::http::{StatusCode, header};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use futures::Stream;
use tracing::info;
use tracing_subscriber::EnvFilter;

use crate::status::{LiveStatus, probe};
use crate::telemetry::{Sources, Telemetry, service_delta};

const INDEX_HTML: &str = include_str!("../assets/index.html");
const STYLE_CSS: &str = include_str!("../assets/style.css");
const LOGO_SVG: &str = include_str!("../assets/logo.svg");

/// SSE cadence for `/_live/stats` and `/_live/services`. Must stay ≥ the
/// telemetry cache TTL so repeated ticks don't always collide with the
/// cache boundary and miss a refresh.
const SSE_INTERVAL: Duration = Duration::from_millis(1500);

#[derive(Clone)]
struct AppState {
    http: reqwest::Client,
    telemetry: Arc<Telemetry>,
}

pub fn build_router() -> Router {
    build_router_with(Sources::default())
}

pub fn build_router_with(sources: Sources) -> Router {
    let state = AppState {
        http: sources.http.clone(),
        telemetry: Arc::new(Telemetry::new(sources)),
    };

    Router::new()
        .route("/", get(index))
        .route("/style.css", get(style))
        .route("/logo.svg", get(logo))
        .route("/_live/status", get(live_status))
        .route("/_live/stats", get(live_stats_sse))
        .route("/_live/services", get(live_services_sse))
        .route("/_health", get(health))
        .with_state(state)
}

async fn index() -> Response {
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        INDEX_HTML,
    )
        .into_response()
}

async fn style() -> Response {
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/css; charset=utf-8")],
        STYLE_CSS,
    )
        .into_response()
}

async fn logo() -> Response {
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "image/svg+xml; charset=utf-8")],
        LOGO_SVG,
    )
        .into_response()
}

async fn live_status(State(s): State<AppState>) -> Json<LiveStatus> {
    Json(probe(&s.http).await)
}

/// `/_live/stats` — push the full [`Stats`] snapshot every
/// [`SSE_INTERVAL`]. First event fires immediately so the UI doesn't
/// sit on `—` for 1.5 s after connect.
async fn live_stats_sse(
    State(s): State<AppState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let telemetry = s.telemetry.clone();
    let stream = async_stream::stream! {
        // Fire-now cadence: emit, then park for SSE_INTERVAL, loop.
        loop {
            let snap = telemetry.snapshot().await;
            let json = serde_json::to_string(&snap)
                .unwrap_or_else(|_| "{}".to_string());
            yield Ok::<_, Infallible>(Event::default().data(json));
            tokio::time::sleep(SSE_INTERVAL).await;
        }
    };
    Sse::new(stream).keep_alive(KeepAlive::default())
}

/// `/_live/services` — emit one event each time a tracked unit flips
/// between active and inactive. On first connect we push a full snapshot
/// so the client can seed its UI.
async fn live_services_sse(
    State(s): State<AppState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let telemetry = s.telemetry.clone();
    let stream = async_stream::stream! {
        let first = telemetry.snapshot().await;
        let payload = serde_json::json!({
            "kind": "snapshot",
            "services": first.services,
        });
        yield Ok::<_, Infallible>(Event::default().data(payload.to_string()));
        let mut prev: Vec<crate::telemetry::ServiceState> = first.services;
        loop {
            tokio::time::sleep(SSE_INTERVAL).await;
            let next = telemetry.snapshot().await;
            if let Some(delta) = service_delta(&prev, &next.services) {
                let payload = serde_json::json!({
                    "kind": "delta",
                    "services": delta,
                });
                yield Ok::<_, Infallible>(Event::default().data(payload.to_string()));
                prev = next.services;
            }
        }
    };
    Sse::new(stream).keep_alive(KeepAlive::default())
}

async fn health() -> Response {
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
        "ok",
    )
        .into_response()
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,halo_landing=debug")),
        )
        .init();

    let addr: SocketAddr = "127.0.0.1:8190".parse().expect("valid bind addr");
    let app = build_router();

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("bind {addr}"))?;
    info!(%addr, "halo-landing listening");

    axum::serve(listener, app)
        .await
        .context("axum serve failed")?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use std::path::PathBuf;
    use tower::ServiceExt;

    fn broken_sources() -> Sources {
        Sources {
            http: reqwest::Client::new(),
            rocm_smi_bin: PathBuf::from("/nonexistent/rocm-smi"),
            xrt_smi_bin: PathBuf::from("/nonexistent/xrt-smi"),
            accel_dev: PathBuf::from("/nonexistent/accel0"),
            shadow_burnin_jsonl: PathBuf::from("/nonexistent/shadow.jsonl"),
            systemctl_bin: PathBuf::from("/nonexistent/systemctl"),
            services: crate::telemetry::TRACKED_SERVICES,
            halo_server_base: "http://192.0.2.1:1".to_string(),
        }
    }

    #[tokio::test]
    async fn index_serves_html_with_wordmark() {
        let resp = build_router()
            .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let ct = resp
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|h| h.to_str().ok())
            .unwrap_or("");
        assert!(
            ct.starts_with("text/html"),
            "expected text/html, got {ct:?}"
        );
        let bytes = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let body = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(body.contains("halo-ai"), "missing wordmark in body");
    }

    #[tokio::test]
    async fn index_has_live_widget_hooks() {
        // Invariant 4: updates live without page reload. The HTML has to
        // subscribe to /_live/stats and /_live/services or the SSE work
        // is wasted.
        let resp = build_router()
            .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let bytes = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let body = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(body.contains("/_live/stats"), "HTML must wire /_live/stats");
        assert!(
            body.contains("/_live/services"),
            "HTML must wire /_live/services"
        );
        assert!(
            body.contains("EventSource"),
            "HTML must use EventSource (SSE) for live widgets"
        );
    }

    #[tokio::test]
    async fn style_serves_css() {
        let resp = build_router()
            .oneshot(
                Request::builder()
                    .uri("/style.css")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let ct = resp
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|h| h.to_str().ok())
            .unwrap_or("");
        assert!(ct.starts_with("text/css"), "expected text/css, got {ct:?}");
    }

    #[tokio::test]
    async fn health_returns_ok() {
        let resp = build_router()
            .oneshot(
                Request::builder()
                    .uri("/_health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 16).await.unwrap();
        assert_eq!(&bytes[..], b"ok");
    }

    #[tokio::test]
    async fn live_status_shape_is_valid_when_backend_down() {
        let resp = build_router()
            .oneshot(
                Request::builder()
                    .uri("/_live/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 4 * 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        for k in [
            "v2_up",
            "v1_up",
            "model",
            "tokps",
            "p50_ms",
            "p95_ms",
            "requests",
            "generated_tokens",
        ] {
            assert!(v.get(k).is_some(), "missing key {k} in {v}");
        }
    }

    #[tokio::test]
    async fn live_stats_sse_content_type() {
        // Handler must advertise text/event-stream so EventSource clients
        // accept the response. (Invariant 4.)
        let resp = build_router_with(broken_sources())
            .oneshot(
                Request::builder()
                    .uri("/_live/stats")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let ct = resp
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|h| h.to_str().ok())
            .unwrap_or("");
        assert!(
            ct.starts_with("text/event-stream"),
            "expected SSE content-type, got {ct:?}"
        );
    }

    #[tokio::test]
    async fn live_stats_first_event_has_all_keys() {
        // Parse the first SSE frame by framing on the raw bytes.
        use futures::StreamExt;
        let resp = build_router_with(broken_sources())
            .oneshot(
                Request::builder()
                    .uri("/_live/stats")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let mut body = resp.into_body().into_data_stream();
        let mut buf = Vec::new();
        loop {
            let chunk = tokio::time::timeout(Duration::from_secs(5), body.next())
                .await
                .expect("SSE frame timed out")
                .expect("SSE stream ended before a frame")
                .expect("SSE chunk error");
            buf.extend_from_slice(&chunk);
            if find_double_newline(&buf).is_some() {
                break;
            }
        }
        let text = String::from_utf8_lossy(&buf);
        let data_line = text
            .lines()
            .find(|l| l.starts_with("data:"))
            .unwrap_or_else(|| panic!("first SSE frame must have a data line; raw={text:?}"));
        let json_str = data_line.trim_start_matches("data:").trim();
        let v: serde_json::Value =
            serde_json::from_str(json_str).expect("SSE data must be JSON");
        for k in [
            "loaded_model",
            "tok_s_decode",
            "gpu_temp_c",
            "gpu_util_pct",
            "npu_up",
            "shadow_burn_exact_pct",
            "services",
            "stale",
        ] {
            assert!(v.get(k).is_some(), "missing {k} in {v}");
        }
        // With broken sources: stale must be true, model must be empty.
        assert_eq!(v["stale"], serde_json::Value::Bool(true));
    }

    #[tokio::test]
    async fn live_stats_respects_interval_between_emits() {
        // Pull two frames and assert at least SSE_INTERVAL minus slack
        // passes between them. Validates the 1.5 s cadence contract.
        use futures::StreamExt;
        use std::time::Instant;
        let resp = build_router_with(broken_sources())
            .oneshot(
                Request::builder()
                    .uri("/_live/stats")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let mut body = resp.into_body().into_data_stream();
        let mut buf = Vec::new();
        let mut frame_times = Vec::new();
        while frame_times.len() < 2 {
            let chunk = tokio::time::timeout(Duration::from_secs(5), body.next())
                .await
                .expect("timed out waiting for SSE frame")
                .expect("stream ended early")
                .expect("chunk ok");
            buf.extend_from_slice(&chunk);
            while let Some(pos) = find_double_newline(&buf) {
                frame_times.push(Instant::now());
                buf.drain(..pos + 2);
                if frame_times.len() >= 2 {
                    break;
                }
            }
        }
        let gap = frame_times[1].duration_since(frame_times[0]);
        // Allow 200 ms slack below the nominal 1500 ms for scheduler jitter.
        assert!(
            gap >= Duration::from_millis(1300),
            "expected ≥1.3 s between SSE frames, got {:?}",
            gap
        );
    }

    fn find_double_newline(buf: &[u8]) -> Option<usize> {
        buf.windows(2).position(|w| w == b"\n\n")
    }
}
