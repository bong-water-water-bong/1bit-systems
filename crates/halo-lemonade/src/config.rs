//! TOML config for the gateway.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LemonadeConfig {
    pub bind: SocketAddr,
    pub registry_path: PathBuf,
    #[serde(default)]
    pub upstream_fallback: Option<String>,
}

impl LemonadeConfig {
    pub fn from_toml_str(s: &str) -> Result<Self> {
        toml::from_str(s).context("parse LemonadeConfig TOML")
    }

    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("read config {}", path.display()))?;
        Self::from_toml_str(&text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_config() {
        let src = r#"
            bind = "127.0.0.1:8765"
            registry_path = "/etc/halo/models.toml"
        "#;
        let cfg = LemonadeConfig::from_toml_str(src).unwrap();
        assert_eq!(cfg.bind.port(), 8765);
        assert_eq!(cfg.registry_path, PathBuf::from("/etc/halo/models.toml"));
        assert!(cfg.upstream_fallback.is_none());
    }

    #[test]
    fn parses_config_with_upstream_fallback() {
        let src = r#"
            bind = "0.0.0.0:9000"
            registry_path = "models.toml"
            upstream_fallback = "https://api.openai.com"
        "#;
        let cfg = LemonadeConfig::from_toml_str(src).unwrap();
        assert_eq!(
            cfg.upstream_fallback.as_deref(),
            Some("https://api.openai.com")
        );
    }
}
