//! Tool registry — one MCP tool per `halo_agents::Name` variant.
//!
//! The registry is a thin view over `halo_agents::Name::ALL`. Tool names
//! are the bare specialist name (`anvil`, `muse`, …) so `tools/call` can
//! route directly into `halo_agents::Registry::dispatch` with no rename
//! gymnastics.
//!
//! # Description + schema sourcing
//!
//! Starting with the `TypedSpecialist` work, descriptions and input
//! schemas are pulled from the live `halo_agents::Registry` whenever one
//! is available (see [`ToolRegistry::from_agents_registry`]). That way
//! real `JsonSchema`-derived schemas land in MCP `tools/list` without a
//! second source of truth.
//!
//! The legacy [`ToolRegistry::from_agents`] constructor still exists for
//! callers (and tests) that don't have a `Registry` handy — it falls back
//! to the static [`crate::specialists::description_for`] table and a
//! passthrough `{ "type": "object" }` schema.

use halo_agents::{Name, Registry};
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
    /// JSON Schema for tool arguments. Real schemas come from
    /// `TypedSpecialist` impls via `Specialist::input_schema`; plain stubs
    /// fall back to a passthrough `{ "type": "object" }`.
    pub input_schema: Value,
}

/// Passthrough schema: any JSON object. Used when the underlying specialist
/// is still a plain stub (no `TypedSpecialist` impl yet).
fn passthrough_schema() -> Value {
    json!({ "type": "object" })
}

impl Tool {
    /// Build the MCP tool entry for a specialist **without** an agents
    /// registry — description comes from the static table, schema is the
    /// passthrough object. Callers with a live registry should prefer
    /// [`Tool::from_registry`] to get the real JSON Schema.
    pub fn from_name(n: Name) -> Self {
        Tool {
            name: n.as_str().to_string(),
            description: description_for(n).to_string(),
            input_schema: passthrough_schema(),
        }
    }

    /// Build the MCP tool entry for a specialist by asking the live
    /// `halo_agents::Registry` for its description + input schema.
    ///
    /// If the specialist overrides `description()` (non-empty), that wins
    /// over the static table. If the specialist returns an empty
    /// `input_schema()` (defensive), we still fall back to the passthrough
    /// so MCP clients always see a valid JSON Schema object.
    pub fn from_registry(n: Name, agents: &Registry) -> Self {
        let sp = agents.get(n);

        let description = match sp {
            Some(s) => {
                let live = s.description();
                if live.is_empty() {
                    description_for(n).to_string()
                } else {
                    live.to_string()
                }
            }
            None => description_for(n).to_string(),
        };

        let input_schema = match sp {
            Some(s) => {
                let v = s.input_schema();
                if v.is_object() { v } else { passthrough_schema() }
            }
            None => passthrough_schema(),
        };

        Tool { name: n.as_str().to_string(), description, input_schema }
    }
}

/// Ordered tool registry. Order matches `halo_agents::Name::ALL` so clients
/// see a stable listing across runs.
#[derive(Debug, Clone, Default)]
pub struct ToolRegistry {
    tools: Vec<Tool>,
}

impl ToolRegistry {
    /// Static-only constructor. Uses [`description_for`] + passthrough
    /// schemas. Prefer [`ToolRegistry::from_agents_registry`] when a live
    /// `halo_agents::Registry` is available — typed specialists only
    /// surface their real schemas through that path.
    pub fn from_agents() -> Self {
        let tools = Name::ALL.iter().map(|n| Tool::from_name(*n)).collect();
        Self { tools }
    }

    /// Build from a live agents registry. Typed specialists contribute
    /// their real `JsonSchema` here; plain stubs keep the passthrough.
    pub fn from_agents_registry(agents: &Registry) -> Self {
        let tools = Name::ALL
            .iter()
            .map(|n| Tool::from_registry(*n, agents))
            .collect();
        Self { tools }
    }

    /// Append a non-specialist tool (e.g. `skill_manage`). The registry
    /// is `tools/list`-facing only — dispatch is handled by the server's
    /// own routing table.
    pub fn push(&mut self, tool: Tool) {
        self.tools.push(tool);
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
    fn static_schema_is_object_typed() {
        // The schema-less constructor still emits a passthrough object.
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

    #[test]
    fn live_registry_pulls_typed_anvil_schema() {
        // When built from a live agents registry, Anvil's schema should
        // be the real JsonSchema (properties.cmd.type == "string"), not
        // the passthrough object.
        let agents = Registry::default_stubs();
        let r = ToolRegistry::from_agents_registry(&agents);
        let anvil = r.find("anvil").expect("anvil tool");
        assert!(anvil.input_schema.is_object());
        let props = anvil
            .input_schema
            .get("properties")
            .expect("properties present");
        assert_eq!(props["cmd"]["type"], "string");
    }

    #[test]
    fn live_registry_keeps_passthrough_for_stubs() {
        // Non-Anvil specialists are still plain stubs — their schema
        // should remain the passthrough { type: object }.
        let agents = Registry::default_stubs();
        let r = ToolRegistry::from_agents_registry(&agents);
        let muse = r.find("muse").expect("muse tool");
        assert_eq!(muse.input_schema["type"], "object");
        assert!(muse.input_schema.get("properties").is_none());
    }
}
