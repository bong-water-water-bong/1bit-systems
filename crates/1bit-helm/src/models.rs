//! Models-pane transport: `GET /v1/models` off the lemonade gateway.
//!
//! OpenAI wire shape:
//!
//! ```json
//! { "data": [ { "id": "1bit-monster-2b", "object": "model", "owned_by": "halo", "created": 0 }, ... ] }
//! ```
//!
//! We keep `id` + `owned_by` + `created` so the card grid has something
//! to show besides a bare identifier. Everything else is dropped on the
//! floor — helm doesn't care about `permission`, `parent`, etc.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// One card's worth of data, rendered in the Models pane grid.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelCard {
    pub id: String,
    #[serde(default)]
    pub owned_by: String,
    #[serde(default)]
    pub created: u64,
}

#[derive(Debug, Deserialize)]
struct ModelsEnvelope {
    #[serde(default)]
    data: Vec<ModelCard>,
}

/// Parse an OpenAI `/v1/models` response body.
pub fn parse_models(body: &str) -> Result<Vec<ModelCard>> {
    let env: ModelsEnvelope = serde_json::from_str(body).context("decode /v1/models response")?;
    Ok(env.data)
}

/// Fetch the live list. Uses `bearer` when present.
pub async fn fetch_models(
    http: &reqwest::Client,
    base_url: &str,
    bearer: Option<&str>,
) -> Result<Vec<ModelCard>> {
    let url = format!("{}/v1/models", base_url.trim_end_matches('/'));
    let mut req = http.get(&url);
    if let Some(tok) = bearer {
        req = req.bearer_auth(tok);
    }
    let resp = req.send().await.with_context(|| format!("GET {url}"))?;
    let status = resp.status();
    let body = resp.text().await.context("read /v1/models body")?;
    if !status.is_success() {
        anyhow::bail!("server {}: {}", status, body);
    }
    parse_models(&body)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_standard_openai_shape() {
        let body = r#"{
            "object": "list",
            "data": [
                { "id": "1bit-monster-2b", "object": "model", "owned_by": "halo", "created": 0 },
                { "id": "qwen3-4b-ternary", "object": "model", "owned_by": "halo", "created": 1 }
            ]
        }"#;
        let cards = parse_models(body).unwrap();
        assert_eq!(cards.len(), 2);
        assert_eq!(cards[0].id, "1bit-monster-2b");
        assert_eq!(cards[0].owned_by, "halo");
        assert_eq!(cards[1].id, "qwen3-4b-ternary");
    }

    #[test]
    fn parses_minimal_shape_without_owned_by() {
        // Some self-hosted backends only emit `id`. We still want a card.
        let body = r#"{ "data": [ { "id": "only-id" } ] }"#;
        let cards = parse_models(body).unwrap();
        assert_eq!(cards.len(), 1);
        assert_eq!(cards[0].id, "only-id");
        assert!(cards[0].owned_by.is_empty());
        assert_eq!(cards[0].created, 0);
    }

    #[test]
    fn empty_data_is_fine() {
        let body = r#"{ "data": [] }"#;
        assert!(parse_models(body).unwrap().is_empty());
    }

    #[test]
    fn rejects_non_json() {
        // serde_json is lenient enough to accept `[]` as an empty-data
        // envelope (it treats the top-level array as empty struct fields
        // with #[serde(default)]); we only contract against genuinely
        // malformed bytes.
        assert!(parse_models("not json").is_err());
        assert!(parse_models("").is_err());
    }
}
