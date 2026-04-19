//! Tool registry — builds the Phase 0 stub tool set from
//! [`crate::specialists::KNOWN`].
//!
//! Parity target: `/home/bcloud/repos/halo-mcp/src/tool_registry.cpp`.

use serde_json::{Value, json};

use crate::specialists::{KNOWN, Specialist};

/// A single MCP tool descriptor.
///
/// Mirrors the C++ `halo_mcp::Tool` struct. Only the fields we actually
/// serialize into `tools/list` responses are stored publicly; the bus
/// routing fields (`target_agent`, `message_kind`) are kept for Phase 1
/// where `tools/call` dispatches into `halo-agents`.
#[derive(Debug, Clone)]
pub struct Tool {
    /// MCP tool name, e.g. `muse_call`.
    pub name: String,
    /// Human-readable description shown to MCP clients.
    pub description: String,
    /// JSON Schema for tool arguments (Phase 0 uses a passthrough stub).
    pub input_schema: Value,
    /// Target agent name on the halo-agents bus. Routing metadata.
    pub target_agent: String,
    /// Message kind to dispatch on the bus. Routing metadata.
    pub message_kind: String,
    /// Whether the specialist mutates external state.
    pub is_write: bool,
}

/// Phase 0 placeholder schema — every stub tool accepts arbitrary args
/// inside a passthrough `arguments` object. Phase 1 replaces these with
/// specialist-specific schemas derived from the `Message` kinds each
/// specialist handles.
fn placeholder_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "arguments": {
                "type": "object",
                "description": "passthrough payload to the specialist; Phase 1 will constrain"
            }
        },
        "required": [],
        "additionalProperties": true
    })
}

fn stub_description(one_liner: &str) -> String {
    // Matches the C++ suffix verbatim so operators can grep either binary
    // and find the same marker.
    format!(
        "{} [Phase 0 stub — returns 'not implemented' until BusBridge is live.]",
        one_liner
    )
}

impl Tool {
    /// Build a Phase 0 stub tool from a [`Specialist`] entry.
    pub fn stub_from(s: &Specialist) -> Self {
        Tool {
            name: format!("{}_call", s.name),
            description: stub_description(s.one_liner),
            input_schema: placeholder_schema(),
            target_agent: s.name.to_string(),
            message_kind: "mcp_call".to_string(),
            is_write: s.is_write,
        }
    }
}

/// Ordered tool registry. Preserves insertion order so `tools/list` output
/// matches the specialist ordering in `Agents.md` and the C++ server.
#[derive(Debug, Clone, Default)]
pub struct ToolRegistry {
    tools: Vec<Tool>,
}

impl ToolRegistry {
    /// Build the default Phase 0 registry (17 stub tools).
    pub fn default_phase0() -> Self {
        let mut r = ToolRegistry::default();
        for s in KNOWN {
            r.register(Tool::stub_from(s));
        }
        r
    }

    /// Register a tool. Later registrations with the same name replace
    /// the earlier entry in place (mirrors `insert_or_assign` semantics).
    pub fn register(&mut self, tool: Tool) {
        if let Some(existing) = self.tools.iter_mut().find(|t| t.name == tool.name) {
            *existing = tool;
        } else {
            self.tools.push(tool);
        }
    }

    /// Lookup by exact tool name.
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
        let r = ToolRegistry::default_phase0();
        assert_eq!(r.len(), 17);
    }

    #[test]
    fn tool_names_have_call_suffix() {
        let r = ToolRegistry::default_phase0();
        for t in r.iter() {
            assert!(t.name.ends_with("_call"), "name missing _call suffix: {}", t.name);
        }
    }

    #[test]
    fn find_hits_known_and_misses_unknown() {
        let r = ToolRegistry::default_phase0();
        assert!(r.find("muse_call").is_some());
        assert!(r.find("warden_call").is_some());
        assert!(r.find("anvil_call").is_some());
        assert!(r.find("does_not_exist").is_none());
    }

    #[test]
    fn registration_order_matches_specialists() {
        let r = ToolRegistry::default_phase0();
        let names: Vec<&str> = r.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names.first().copied(), Some("muse_call"));
        assert_eq!(names.last().copied(), Some("anvil_call"));
    }

    #[test]
    fn description_includes_phase0_marker() {
        let r = ToolRegistry::default_phase0();
        for t in r.iter() {
            assert!(
                t.description.contains("Phase 0 stub"),
                "missing Phase 0 marker in: {}",
                t.description
            );
        }
    }

    #[test]
    fn register_replaces_existing() {
        let mut r = ToolRegistry::default_phase0();
        let before = r.len();
        r.register(Tool {
            name: "muse_call".into(),
            description: "overridden".into(),
            input_schema: json!({}),
            target_agent: "muse".into(),
            message_kind: "mcp_call".into(),
            is_write: false,
        });
        assert_eq!(r.len(), before, "replacement should not grow the registry");
        assert_eq!(r.find("muse_call").unwrap().description, "overridden");
    }
}
