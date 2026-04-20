//! LLM-backed claim derivation.
//!
//! Replaces the "claim := observation verbatim" scaffold in [`super::observe`]
//! with a distillation pass: POST the observation to 1bit-server's
//! `/v1/chat/completions` endpoint and ask it for a JSON array of short
//! claims about the observed peer. On any error (HTTP failure, non-JSON
//! body, empty array, …) we fall back to `[observation_verbatim]` so the
//! dialectic pipeline stays resilient when 1bit-server is down.
//!
//! Gated behind the `llm-derive` feature so the default test run doesn't
//! need a live halo-server. When the feature is OFF, [`super::observe`]
//! keeps the scaffold behaviour (one claim = observation text).

use reqwest::Client;
use serde_json::{Value, json};

/// Default model id exposed by halo-server today. Callers can override via
/// [`derive_claims_with_model`] once multi-model routing lands.
pub const DEFAULT_MODEL: &str = "bitnet-b1.58-2B-4T";

/// Default halo-server chat-completions endpoint. Tests override the base
/// URL via [`derive_claims_with_url`].
pub const DEFAULT_CHAT_URL: &str = "http://127.0.0.1:8180/v1/chat/completions";

/// Build the derivation prompt. Exposed for tests + for docs/wiki so the
/// exact wording is a single source of truth.
///
/// Format matches Honcho's "derivation task" shape: ask for a small number
/// of short claims about the *observed* peer, return as a JSON array, no
/// prose. Temperature is set low (0.2) at the HTTP layer to keep this
/// deterministic enough for downstream ranking.
pub fn derivation_prompt(observation: &str, observed_peer: &str) -> String {
    format!(
        "You are a careful observer. Extract 1-3 short claims about {observed_peer} from the following utterance. Return ONLY a JSON array of strings, no prose.\n\nUtterance: {observation}"
    )
}

/// Parse a `/v1/chat/completions` response body into a claim list.
///
/// Strategy:
/// 1. Read `choices[0].message.content` as a string.
/// 2. Try to parse it as a JSON array of strings.
/// 3. If that fails at any step, return `[fallback.to_string()]`.
///
/// Pulled out of the HTTP path as a pure function so the parse rules are
/// testable without spinning up a mock server.
pub fn parse_claims_from_response(body: &Value, fallback: &str) -> Vec<String> {
    let content = body
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str());
    let Some(content) = content else {
        return vec![fallback.to_string()];
    };
    parse_claims_from_content(content, fallback)
}

/// Parse the raw `message.content` string produced by the LLM.
///
/// Accepts a bare JSON array (`["a","b"]`) or a JSON array wrapped in
/// whitespace / code fences. Anything else → `[fallback]`.
pub fn parse_claims_from_content(content: &str, fallback: &str) -> Vec<String> {
    let trimmed = strip_code_fence(content.trim());
    match serde_json::from_str::<Vec<String>>(trimmed) {
        Ok(v) if !v.is_empty() => v,
        _ => vec![fallback.to_string()],
    }
}

/// Strip a surrounding ```json … ``` or ``` … ``` fence if present. Some
/// chat models wrap JSON even when told not to; be lenient.
fn strip_code_fence(s: &str) -> &str {
    let s = s.trim();
    if let Some(rest) = s.strip_prefix("```json") {
        return rest.trim_start().trim_end_matches("```").trim();
    }
    if let Some(rest) = s.strip_prefix("```") {
        return rest.trim_start().trim_end_matches("```").trim();
    }
    s
}

/// Derive claims from an observation using the default halo-server URL
/// and model. This is the entry point [`super::observe`] uses when the
/// `llm-derive` feature is enabled.
pub async fn derive_claims(
    observation: &str,
    observed_peer: &str,
    client: &Client,
) -> Vec<String> {
    derive_claims_with_url(observation, observed_peer, client, DEFAULT_CHAT_URL, DEFAULT_MODEL).await
}

/// Derivation with explicit base URL + model — used by tests that point
/// at a local TCP mock.
pub async fn derive_claims_with_url(
    observation: &str,
    observed_peer: &str,
    client: &Client,
    url: &str,
    model: &str,
) -> Vec<String> {
    let prompt = derivation_prompt(observation, observed_peer);
    let body = json!({
        "model": model,
        "temperature": 0.2,
        "max_tokens": 256,
        "stream": false,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    });
    let resp = match client.post(url).json(&body).send().await {
        Ok(r) => r,
        Err(_) => return vec![observation.to_string()],
    };
    if !resp.status().is_success() {
        return vec![observation.to_string()];
    }
    let json: Value = match resp.json().await {
        Ok(j) => j,
        Err(_) => return vec![observation.to_string()],
    };
    parse_claims_from_response(&json, observation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::net::TcpListener as StdListener;

    /// Spin up a one-shot mock HTTP server on a free port. Accepts one
    /// connection, reads the request, writes `canned_response_body` with
    /// a 200 OK wrapper, closes. Returns the base URL.
    fn spawn_mock(canned_response_body: &'static str) -> String {
        let listener = StdListener::bind("127.0.0.1:0").expect("bind mock");
        let addr = listener.local_addr().expect("addr").to_string();
        std::thread::spawn(move || {
            // One request, one response. Enough for a single test.
            if let Ok((mut stream, _)) = listener.accept() {
                // Drain the request (ignore — we don't branch on it here).
                let mut buf = [0u8; 4096];
                let _ = std::io::Read::read(&mut stream, &mut buf);
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    canned_response_body.len(),
                    canned_response_body
                );
                let _ = stream.write_all(resp.as_bytes());
                let _ = stream.flush();
            }
        });
        format!("http://{addr}/v1/chat/completions")
    }

    #[tokio::test]
    async fn parse_claims_accepts_valid_json_array() {
        // Pure-parse unit test: proves the happy path without HTTP.
        let body = json!({
            "choices": [{
                "message": {
                    "content": "[\"bob likes CLI\", \"bob hates GUIs\"]"
                }
            }]
        });
        let claims = parse_claims_from_response(&body, "fallback");
        assert_eq!(claims, vec!["bob likes CLI", "bob hates GUIs"]);
    }

    #[tokio::test]
    async fn parse_claims_falls_back_on_prose() {
        // LLM ignored instructions and wrote prose → fall back verbatim.
        let body = json!({
            "choices": [{
                "message": {
                    "content": "Sure! Bob seems to like CLI tools."
                }
            }]
        });
        let claims = parse_claims_from_response(&body, "bob said hi");
        assert_eq!(claims, vec!["bob said hi"]);
    }

    #[tokio::test]
    async fn parse_claims_strips_code_fence() {
        let body = json!({
            "choices": [{
                "message": {
                    "content": "```json\n[\"a\", \"b\"]\n```"
                }
            }]
        });
        let claims = parse_claims_from_response(&body, "fallback");
        assert_eq!(claims, vec!["a", "b"]);
    }

    #[tokio::test]
    async fn derive_claims_against_mock_parses_json_array() {
        // End-to-end with a real reqwest::Client pointed at a tiny
        // TCP-level mock. Proves the full HTTP path + response parsing.
        let canned = r#"{"choices":[{"message":{"content":"[\"bob prefers terse CLI\", \"bob reviews PRs fast\"]"}}]}"#;
        let url = spawn_mock(canned);
        let client = Client::new();
        let claims = derive_claims_with_url("bob: lgtm, ship it", "bob", &client, &url, DEFAULT_MODEL).await;
        assert_eq!(claims, vec!["bob prefers terse CLI", "bob reviews PRs fast"]);
    }

    #[tokio::test]
    async fn derive_claims_against_mock_falls_back_on_prose() {
        // Same wire path, but the LLM content is prose → fallback to verbatim.
        let canned = r#"{"choices":[{"message":{"content":"Bob likes CLI tools."}}]}"#;
        let url = spawn_mock(canned);
        let client = Client::new();
        let claims = derive_claims_with_url("bob: lgtm", "bob", &client, &url, DEFAULT_MODEL).await;
        assert_eq!(claims, vec!["bob: lgtm"]);
    }

    #[tokio::test]
    async fn derive_claims_http_error_falls_back() {
        // Port that refuses connections → network error → fallback.
        let client = Client::new();
        let url = "http://127.0.0.1:1/v1/chat/completions"; // port 1 = almost certainly closed
        let claims =
            derive_claims_with_url("carol: wat", "carol", &client, url, DEFAULT_MODEL).await;
        assert_eq!(claims, vec!["carol: wat"]);
    }
}
