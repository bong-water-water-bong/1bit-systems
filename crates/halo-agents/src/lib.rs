//! halo-agents — 17 specialists as an async dispatch table.
//!
//! Lean by design: one file. Each specialist is a `Specialist` impl that
//! takes a JSON request and returns a JSON response. The registry is a
//! `HashMap<Name, Arc<dyn Specialist>>` seeded at startup. New specialists
//! get one `impl` block, no framework ceremony.
//!
//! Rust port of `agent-cpp/specialists/*.cpp`. Behaviour parity comes later;
//! for now each specialist returns a shaped stub so the routing, MCP bridge,
//! and Discord relay can be wired end-to-end before implementations land.

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;

/// The 17 specialists. Ordering matches agent-cpp/specialists/ for easy diff.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Name {
    Anvil, Carpenter, Cartograph, EchoEar, EchoMouth, Forge, Gateway, Herald,
    Librarian, Magistrate, Muse, Planner, Quartermaster, Scribe, Sentinel,
    Sommelier, Warden,
}

impl Name {
    pub const ALL: &'static [Name] = &[
        Name::Anvil, Name::Carpenter, Name::Cartograph, Name::EchoEar,
        Name::EchoMouth, Name::Forge, Name::Gateway, Name::Herald,
        Name::Librarian, Name::Magistrate, Name::Muse, Name::Planner,
        Name::Quartermaster, Name::Scribe, Name::Sentinel, Name::Sommelier,
        Name::Warden,
    ];

    pub fn as_str(&self) -> &'static str {
        match self {
            Name::Anvil => "anvil",                Name::Carpenter => "carpenter",
            Name::Cartograph => "cartograph",      Name::EchoEar => "echo_ear",
            Name::EchoMouth => "echo_mouth",       Name::Forge => "forge",
            Name::Gateway => "gateway",            Name::Herald => "herald",
            Name::Librarian => "librarian",        Name::Magistrate => "magistrate",
            Name::Muse => "muse",                  Name::Planner => "planner",
            Name::Quartermaster => "quartermaster",Name::Scribe => "scribe",
            Name::Sentinel => "sentinel",          Name::Sommelier => "sommelier",
            Name::Warden => "warden",
        }
    }

    pub fn from_str(s: &str) -> Option<Name> {
        Name::ALL.iter().find(|n| n.as_str() == s).copied()
    }
}

/// Every specialist implements this. Input + output are JSON to keep the
/// wire format identical to agent-cpp and the MCP bridge.
///
/// DSPy-inspired extensions (`description`, `input_schema`, `output_schema`)
/// have lean defaults so existing stub impls keep compiling. Real specialists
/// override them; halo-mcp's `tools/list` surfaces the description + input
/// schema so MCP clients (Claude Code, Claude Desktop, DSPy) can render
/// typed call UIs without round-tripping through the specialist first.
#[async_trait]
pub trait Specialist: Send + Sync {
    fn name(&self) -> Name;

    /// One-line hint shown to MCP clients and agent planners. Default empty.
    fn description(&self) -> &'static str { "" }

    /// JSON Schema for the input argument to `handle`. Default `{}` (any JSON).
    /// Override with a real schema so clients can validate before calling.
    fn input_schema(&self) -> Value { json!({ "type": "object" }) }

    /// JSON Schema for the `Ok(Value)` returned from `handle`. Default `{}`.
    fn output_schema(&self) -> Value { json!({ "type": "object" }) }

    async fn handle(&self, req: Value) -> Result<Value>;
}

/// Boxed specialist — registry values share this type.
pub type Boxed = Arc<dyn Specialist>;

pub struct Registry {
    map: HashMap<Name, Boxed>,
}

impl Registry {
    /// Build the default registry (all 17 stubs). Callers replace individual
    /// entries with real impls via `insert`.
    pub fn default_stubs() -> Self {
        let mut map: HashMap<Name, Boxed> = HashMap::new();
        for n in Name::ALL {
            map.insert(*n, Arc::new(Stub(*n)));
        }
        Self { map }
    }

    pub fn insert(&mut self, s: Boxed) { self.map.insert(s.name(), s); }

    pub fn get(&self, n: Name) -> Option<&Boxed> { self.map.get(&n) }

    pub async fn dispatch(&self, name: &str, req: Value) -> Result<Value> {
        let n = Name::from_str(name).ok_or_else(|| anyhow!("unknown specialist '{name}'"))?;
        let s = self.map.get(&n).ok_or_else(|| anyhow!("no impl registered for {name}"))?;
        s.handle(req).await
    }

    pub fn names(&self) -> Vec<&'static str> {
        self.map.keys().map(|n| n.as_str()).collect()
    }
}

impl Default for Registry {
    fn default() -> Self { Self::default_stubs() }
}

/// Shape-correct stub. Returns `{specialist, status:"stub", echo:req}` so the
/// wiring pipeline (HTTP → router → MCP → Discord relay) can be validated
/// before real implementations land.
struct Stub(Name);

#[async_trait]
impl Specialist for Stub {
    fn name(&self) -> Name { self.0 }
    async fn handle(&self, req: Value) -> Result<Value> {
        Ok(json!({ "specialist": self.0.as_str(), "status": "stub", "echo": req }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_names_roundtrip() {
        for n in Name::ALL {
            assert_eq!(Name::from_str(n.as_str()), Some(*n));
        }
        assert_eq!(Name::ALL.len(), 17);
    }

    #[tokio::test]
    async fn registry_dispatches_stub() {
        let r = Registry::default_stubs();
        let out = r.dispatch("anvil", json!({"cmd":"ping"})).await.unwrap();
        assert_eq!(out["specialist"], "anvil");
        assert_eq!(out["status"], "stub");
        assert_eq!(out["echo"]["cmd"], "ping");
    }

    #[tokio::test]
    async fn registry_unknown_specialist_errors() {
        let r = Registry::default_stubs();
        assert!(r.dispatch("not_a_specialist", json!({})).await.is_err());
    }

    #[tokio::test]
    async fn registry_allows_override() {
        struct Real;
        #[async_trait]
        impl Specialist for Real {
            fn name(&self) -> Name { Name::Anvil }
            async fn handle(&self, _req: Value) -> Result<Value> {
                Ok(json!({"specialist":"anvil","status":"real"}))
            }
        }
        let mut r = Registry::default_stubs();
        r.insert(Arc::new(Real));
        let out = r.dispatch("anvil", json!({})).await.unwrap();
        assert_eq!(out["status"], "real");
    }
}
