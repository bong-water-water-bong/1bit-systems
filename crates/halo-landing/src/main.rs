//! `halo-landing` — static marketing page + live status probe on :8190.
//!
//! All assets are embedded at compile time via `include_str!`; there are
//! no filesystem reads at runtime. The only outbound call is a 2 s probe
//! of `http://127.0.0.1:8180/v1/models` driven by `GET /_live/status`.
//!
//! Routes:
//! * `GET /`              — embedded HTML
//! * `GET /style.css`     — embedded CSS
//! * `GET /_live/status`  — JSON `{v2_up, v1_up, model, tokps}`, always 200
//! * `GET /_health`       — plain text "ok"

mod status;

use std::net::SocketAddr;

use anyhow::{Context, Result};
use axum::Json;
use axum::Router;
use axum::extract::State;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use tracing::info;
use tracing_subscriber::EnvFilter;

use crate::status::{LiveStatus, probe};

const INDEX_HTML: &str = include_str!("../assets/index.html");
const STYLE_CSS: &str = include_str!("../assets/style.css");

#[derive(Clone)]
struct AppState {
    http: reqwest::Client,
}

pub fn build_router() -> Router {
    let state = AppState {
        http: reqwest::Client::builder()
            .user_agent("halo-landing/0.1")
            .build()
            .expect("reqwest client"),
    };

    Router::new()
        .route("/", get(index))
        .route("/style.css", get(style))
        .route("/_live/status", get(live_status))
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

async fn live_status(State(s): State<AppState>) -> Json<LiveStatus> {
    Json(probe(&s.http).await)
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
    use tower::ServiceExt;

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
        // No halo-server on :8180 in CI — probe should return offline, 200.
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
        // All four keys must be present regardless of backend state.
        assert!(v.get("v2_up").is_some());
        assert!(v.get("v1_up").is_some());
        assert!(v.get("model").is_some());
        assert!(v.get("tokps").is_some());
    }
}
