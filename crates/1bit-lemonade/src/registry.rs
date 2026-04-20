//! Model registry: `model_id` → `ModelEntry { backend, capabilities, params }`.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelEntry {
    pub backend: String,
    pub capabilities: Vec<String>,
    #[serde(default)]
    pub params: Value,
}

impl ModelEntry {
    pub fn new(backend: impl Into<String>, capabilities: Vec<String>) -> Self {
        Self {
            backend: backend.into(),
            capabilities,
            params: Value::Null,
        }
    }

    pub fn has_capability(&self, cap: &str) -> bool {
        self.capabilities.iter().any(|c| c == cap)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelRegistry {
    #[serde(flatten)]
    entries: HashMap<String, ModelEntry>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, id: impl Into<String>, entry: ModelEntry) {
        self.entries.insert(id.into(), entry);
    }

    pub fn get(&self, id: &str) -> Option<&ModelEntry> {
        self.entries.get(id)
    }

    pub fn ids(&self) -> Vec<&str> {
        self.entries.keys().map(String::as_str).collect()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn registry_insert_and_lookup_roundtrip() {
        let mut r = ModelRegistry::new();
        r.insert(
            "bitnet-3b",
            ModelEntry {
                backend: "local/bitnet-hip".into(),
                capabilities: vec!["chat".into(), "completion".into()],
                params: json!({ "ctx": 4096 }),
            },
        );
        let got = r.get("bitnet-3b").unwrap();
        assert_eq!(got.backend, "local/bitnet-hip");
        assert!(got.has_capability("chat"));
        assert!(!got.has_capability("embeddings"));
        assert_eq!(got.params["ctx"], 4096);
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn registry_missing_id_returns_none() {
        let r = ModelRegistry::new();
        assert!(r.get("nope").is_none());
        assert!(r.is_empty());
    }
}
