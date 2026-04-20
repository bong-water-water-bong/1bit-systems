//! Session-wide configuration (server URL, auth, default model, system prompt).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SessionConfig {
    pub server_url: String,
    #[serde(default)]
    pub bearer: Option<String>,
    pub default_model: String,
    #[serde(default)]
    pub system_prompt: Option<String>,
}

impl SessionConfig {
    pub fn new(server_url: impl Into<String>, default_model: impl Into<String>) -> Self {
        Self {
            server_url: server_url.into(),
            bearer: None,
            default_model: default_model.into(),
            system_prompt: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_config_serde_roundtrip() {
        let cfg = SessionConfig {
            server_url: "http://127.0.0.1:8080".into(),
            bearer: Some("tok".into()),
            default_model: "bitnet-3b".into(),
            system_prompt: Some("You are Halo.".into()),
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let back: SessionConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg, back);
    }

    #[test]
    fn session_config_defaults_optional_fields() {
        let src = r#"{"server_url":"http://x","default_model":"m"}"#;
        let cfg: SessionConfig = serde_json::from_str(src).unwrap();
        assert!(cfg.bearer.is_none());
        assert!(cfg.system_prompt.is_none());
    }
}
