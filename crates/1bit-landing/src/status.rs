//! Live-status probe for the gen-2 1bit-server on :8180.
//!
//! The landing page polls `/_live/status` every 3 s and uses the result to
//! drive the pulsing green dot + live tok/s number in the hero. We
//! deliberately never 5xx this endpoint — if 1bit-server is down we just
//! report `v2_up: false` so the UI stays legible.
//!
//! Two probes run in sequence:
//! 1. `GET /v1/models` → `v2_up`, `model`.
//! 2. `GET /metrics` → `tokps`, `p50_ms`, `p95_ms`, `requests`,
//!    `generated_tokens`. Graceful degradation: if the metrics endpoint is
//!    unreachable (older server build, 404), the numeric fields stay at 0
//!    rather than getting a hard-coded placeholder — `index.html` displays
//!    an em-dash in that case.

use std::time::Duration;

use serde::Serialize;

/// Shape returned to the browser. Mirror in `assets/index.html` <script>.
#[derive(Debug, Clone, Serialize)]
pub struct LiveStatus {
    pub v2_up: bool,
    pub v1_up: bool,
    pub model: String,
    pub tokps: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub requests: u64,
    pub generated_tokens: u64,
}

impl LiveStatus {
    pub fn offline() -> Self {
        Self {
            v2_up: false,
            v1_up: false,
            model: String::new(),
            tokps: 0.0,
            p50_ms: 0.0,
            p95_ms: 0.0,
            requests: 0,
            generated_tokens: 0,
        }
    }
}

/// Probe `http://127.0.0.1:8180/metrics` with a 2 s timeout.
///
/// Returns the raw JSON blob on success, `None` on any failure (timeout,
/// connection refused, non-2xx, malformed JSON). Kept `pub` so integration
/// tests can exercise it directly.
pub async fn probe_metrics(http: &reqwest::Client) -> Option<serde_json::Value> {
    let req = http
        .get("http://127.0.0.1:8180/metrics")
        .timeout(Duration::from_secs(2))
        .send()
        .await
        .ok()?;
    if !req.status().is_success() {
        return None;
    }
    req.json::<serde_json::Value>().await.ok()
}

/// Probe `http://127.0.0.1:8180/v1/models` with a 2 s timeout, then merge
/// in live metrics from `/metrics`. Returns `LiveStatus::offline()` if the
/// models probe fails; metrics failures just zero out the numeric fields.
pub async fn probe(client: &reqwest::Client) -> LiveStatus {
    let url = "http://127.0.0.1:8180/v1/models";
    let req = client.get(url).timeout(Duration::from_secs(2)).send().await;

    let resp = match req {
        Ok(r) if r.status().is_success() => r,
        _ => return LiveStatus::offline(),
    };

    let body: serde_json::Value = match resp.json().await {
        Ok(v) => v,
        Err(_) => return LiveStatus::offline(),
    };

    // Pull the first model's id out of `{"object":"list","data":[{"id":...}]}`.
    let model = body
        .get("data")
        .and_then(|d| d.get(0))
        .and_then(|m| m.get("id"))
        .and_then(|i| i.as_str())
        .unwrap_or("unknown")
        .to_string();

    // Merge in /metrics. If that probe fails, leave numeric fields at 0 —
    // the frontend renders that as "—".
    let (tokps, p50_ms, p95_ms, requests, generated_tokens) = match probe_metrics(client).await {
        Some(m) => (
            m.get("tokps_recent")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0),
            m.get("p50_ms").and_then(|v| v.as_f64()).unwrap_or(0.0),
            m.get("p95_ms").and_then(|v| v.as_f64()).unwrap_or(0.0),
            m.get("requests").and_then(|v| v.as_u64()).unwrap_or(0),
            m.get("generated_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
        ),
        None => (0.0, 0.0, 0.0, 0, 0),
    };

    LiveStatus {
        v2_up: true,
        v1_up: false, // we only probe gen-2 for now; gen-1 check is future work
        model,
        tokps,
        p50_ms,
        p95_ms,
        requests,
        generated_tokens,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offline_has_false_flags() {
        let s = LiveStatus::offline();
        assert!(!s.v2_up);
        assert!(!s.v1_up);
        assert_eq!(s.tokps, 0.0);
        assert_eq!(s.p50_ms, 0.0);
        assert_eq!(s.p95_ms, 0.0);
        assert_eq!(s.requests, 0);
        assert_eq!(s.generated_tokens, 0);
        assert!(s.model.is_empty());
    }

    #[test]
    fn offline_serializes_all_fields() {
        let s = LiveStatus::offline();
        let v = serde_json::to_value(&s).unwrap();
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
            assert!(v.get(k).is_some(), "offline snapshot missing field {k}");
        }
    }

    #[tokio::test]
    async fn probe_unreachable_returns_offline() {
        // Nothing on :8180 in-test → should return offline, never panic.
        let client = reqwest::Client::new();
        let s = probe(&client).await;
        // We can't assert v2_up=false unconditionally (someone might have
        // 1bit-server running locally), but we can assert the shape is sane.
        if !s.v2_up {
            assert_eq!(s.tokps, 0.0);
            assert!(s.model.is_empty());
            assert_eq!(s.requests, 0);
        }
    }
}
