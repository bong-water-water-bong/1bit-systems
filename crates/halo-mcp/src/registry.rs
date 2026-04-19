//! Tool registry — one MCP tool per `halo_agents::Name` variant.
//!
//! The registry is a thin view over `halo_agents::Name::ALL`. Tool names
//! are the bare specialist name (`anvil`, `muse`, …) so `tools/call` can
//! route directly into `halo_agents::Registry::dispatch` with no rename
//! gymnastics. Description text comes from [`crate::specialists`].

use halo_agents::Name;
use serde_json::{Value, json};

use crate::specialists::description_for;

/// A single MCP tool descriptor.
///
/// In Phase 1 the tool name *is* the `halo_agents` specialist name — the
/// dispatcher hands the tool name straight to `Registry::dispatch`.
#[derive(Debug, Clone)]
pub struct Tool {
    /// MCP tool name (= specialist name, e.g. `anvil`).
    pub name: String,
    /// Human-readable description shown to MCP clients.
    pub description: String,
    /// JSON Schema for tool arguments. Passthrough object for now.
    pub input_schema: Value,
}

/// Passthrough schema: every specialist accepts an arbitrary JSON object
/// and the stub bus echoes it back. Per-specialist schemas are a future
/// concern once the real implementations land.
fn passthrough_schema() -> Value {
    json!({ "type": "object" })
}

impl Tool {
    /// Build the MCP tool entry for a single specialist.
    pub fn from_name(n: Name) -> Self {
        Tool {
            name: n.as_str().to_string(),
            description: description_for(n).to_string(),
            input_schema: passthrough_schema(),
        }
    }
}

/// Ordered tool registry. Order matches `halo_agents::Name::ALL` so clients
/// see a stable listing across runs.
#[derive(Debug, Clone, Default)]
pub struct ToolRegistry {
    tools: Vec<Tool>,
}

impl ToolRegistry {
    /// Build the default registry (one tool per specialist in `Name::ALL`).
    pub fn from_agents() -> Self {
        let tools = Name::ALL.iter().map(|n| Tool::from_name(*n)).collect();
        Self { tools }
    }

    /// Lookup by exact tool name. Case-sensitive.
    pub fn find(&self, name: &str) -> Option<&Tool> {
        self.tools.iter().find(|t| t.name == name)
    }

    /// Iterate tools in registration order.
    pub fn iter(&self) -> impl Iterator<Item = &Tool> {
        self.tools.iter()
    }

    /// Number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// True iff no tools are registered.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_has_seventeen_tools() {
        let r = ToolRegistry::from_agents();
        assert_eq!(r.len(), 17);
        assert_eq!(r.len(), Name::ALL.len());
    }

    #[test]
    fn tool_names_match_name_all() {
        let r = ToolRegistry::from_agents();
        let got: Vec<&str> = r.iter().map(|t| t.name.as_str()).collect();
        let want: Vec<&str> = Name::ALL.iter().map(|n| n.as_str()).collect();
        assert_eq!(got, want);
    }

    #[test]
    fn find_hits_known_and_misses_unknown() {
        let r = ToolRegistry::from_agents();
        assert!(r.find("muse").is_some());
        assert!(r.find("warden").is_some());
        assert!(r.find("anvil").is_some());
        assert!(r.find("does_not_exist").is_none());
        // The old `_call` suffix is gone in Phase 1.
        assert!(r.find("anvil_call").is_none());
    }

    #[test]
    fn schema_is_object_typed() {
        let r = ToolRegistry::from_agents();
        for t in r.iter() {
            assert_eq!(t.input_schema["type"], "object", "tool {}", t.name);
        }
    }

    #[test]
    fn descriptions_are_nonempty() {
        let r = ToolRegistry::from_agents();
        for t in r.iter() {
            assert!(!t.description.is_empty(), "empty desc for {}", t.name);
        }
    }
}
