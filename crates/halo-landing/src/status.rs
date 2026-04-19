//! Live-status probe for the gen-2 halo-server on :8180.
//!
//! The landing page polls `/_live/status` every 3 s and uses the result to
//! drive the pulsing green dot in the hero. We deliberately never 5xx this
//! endpoint — if halo-server is down we just report `v2_up: false` so the
//! UI stays legible.

use std::time::Duration;

use serde::Serialize;

/// Shape returned to the browser. Mirror in `assets/index.html` <script>.
#[derive(Debug, Clone, Serialize)]
pub struct LiveStatus {
    pub v2_up: bool,
    pub v1_up: bool,
    pub model: String,
    pub tokps: f64,
}

impl LiveStatus {
    pub fn offline() -> Self {
        Self {
            v2_up: false,
            v1_up: false,
            model: String::new(),
            tokps: 0.0,
        }
    }
}

/// Probe `http://127.0.0.1:8180/v1/models` with a 2 s timeout.
///
/// Returns a populated `LiveStatus` on success; `LiveStatus::offline()` on
/// any failure (timeout, connection refused, non-2xx, malformed JSON). The
/// `tokps` field is a placeholder — halo-server doesn't expose live tok/s
/// yet, so we surface the burn-bench headline (83.0) when the server is up
/// and will wire a real metric once /metrics lands.
pub async fn probe(client: &reqwest::Client) -> LiveStatus {
    let url = "http://127.0.0.1:8180/v1/models";
    let req = client
        .get(url)
        .timeout(Duration::from_secs(2))
        .send()
        .await;

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

    LiveStatus {
        v2_up: true,
        v1_up: false, // we only probe gen-2 for now; gen-1 check is future work
        model,
        tokps: 83.0,
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
        assert!(s.model.is_empty());
    }

    #[tokio::test]
    async fn probe_unreachable_returns_offline() {
        // Nothing on :8180 in-test → should return offline, never panic.
        let client = reqwest::Client::new();
        let s = probe(&client).await;
        // We can't assert v2_up=false unconditionally (someone might have
        // halo-server running locally), but we can assert the shape is sane.
        if !s.v2_up {
            assert_eq!(s.tokps, 0.0);
            assert!(s.model.is_empty());
        }
    }
}
