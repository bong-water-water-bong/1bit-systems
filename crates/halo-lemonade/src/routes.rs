//! Axum router — OpenAI-compat surface. v0 ships only `/v1/models` and a
//! health probe; chat/completions is proxied to halo-server via the
//! upstream fallback in [`LemonadeConfig`] rather than re-implementing
//! generation here.

use axum::{
    extract::State,
    http::StatusCode,
    routing::get,
    Json, Router,
};
use serde_json::{json, Value};
use std::sync::Arc;

use crate::registry::ModelRegistry;

#[derive(Clone)]
pub struct AppState {
    pub registry: Arc<ModelRegistry>,
}

pub fn build(state: AppState) -> Router {
    Router::new()
        .route("/_health", get(|| async { (StatusCode::OK, "ok") }))
        .route("/v1/models", get(list_models))
        .with_state(state)
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
        AppState { registry: Arc::new(r) }
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
        assert_eq!(v["data"][0]["capabilities"][0], "chat");
    }

    #[tokio::test]
    async fn empty_registry_returns_empty_list() {
        let state = AppState { registry: Arc::new(ModelRegistry::new()) };
        let app = build(state);
        let res = app
            .oneshot(Request::builder().uri("/v1/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let body = axum::body::to_bytes(res.into_body(), 4096).await.unwrap();
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["data"].as_array().unwrap().len(), 0);
    }
}
