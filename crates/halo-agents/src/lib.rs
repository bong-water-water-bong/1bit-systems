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
//!
//! # Two trait layers
//!
//! * [`Specialist`] is the dyn-safe, JSON-erased trait the registry holds
//!   and the MCP bridge calls. Input / output are `serde_json::Value`.
//! * [`TypedSpecialist`] is the DSPy-inspired typed layer: real specialists
//!   declare their input / output structs with [`schemars::JsonSchema`] +
//!   serde derive, implement `handle_typed`, and the [`Typed`] adapter shim
//!   does the serde + schema work automatically. A blanket
//!   `impl<T: TypedSpecialist> Specialist for Typed<T>` makes them drop-in
//!   replacements for stubs in the registry.
//!
//! `TypedSpecialist` itself is **not** object-safe — it has associated types
//! and uses native `impl Future` in the `handle_typed` return position.
//! That's fine: `Typed<T>` is the object-safe wrapper that goes into the
//! registry. Use `Typed::arc(T)` to register a typed specialist.

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use schemars::{JsonSchema, schema_for};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::Arc;

pub mod sessions;
pub use sessions::{Hit, SessionDb};

pub mod skills;
pub use skills::{Skill, SkillAction, SkillStore};

pub mod memory;
pub use memory::{
    DELIMITER as MEMORY_DELIMITER, MAX_MEMORY_CHARS, MAX_USER_CHARS, MemoryKind, MemoryStore,
};

/// The 17 specialists. Ordering matches agent-cpp/specialists/ for easy diff.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Name {
    Anvil,
    Carpenter,
    Cartograph,
    EchoEar,
    EchoMouth,
    Forge,
    Gateway,
    Herald,
    Librarian,
    Magistrate,
    Muse,
    Planner,
    Quartermaster,
    Scribe,
    Sentinel,
    Sommelier,
    Warden,
}

impl Name {
    pub const ALL: &'static [Name] = &[
        Name::Anvil,
        Name::Carpenter,
        Name::Cartograph,
        Name::EchoEar,
        Name::EchoMouth,
        Name::Forge,
        Name::Gateway,
        Name::Herald,
        Name::Librarian,
        Name::Magistrate,
        Name::Muse,
        Name::Planner,
        Name::Quartermaster,
        Name::Scribe,
        Name::Sentinel,
        Name::Sommelier,
        Name::Warden,
    ];

    pub fn as_str(&self) -> &'static str {
        match self {
            Name::Anvil => "anvil",
            Name::Carpenter => "carpenter",
            Name::Cartograph => "cartograph",
            Name::EchoEar => "echo_ear",
            Name::EchoMouth => "echo_mouth",
            Name::Forge => "forge",
            Name::Gateway => "gateway",
            Name::Herald => "herald",
            Name::Librarian => "librarian",
            Name::Magistrate => "magistrate",
            Name::Muse => "muse",
            Name::Planner => "planner",
            Name::Quartermaster => "quartermaster",
            Name::Scribe => "scribe",
            Name::Sentinel => "sentinel",
            Name::Sommelier => "sommelier",
            Name::Warden => "warden",
        }
    }

    // Intentional API: infallible `Option<Name>` lookup, not `Result<Name, _>`.
    // Not implementing `FromStr` because callers treat "no such specialist" as
    // `None`, not as a parse error.
    #[allow(clippy::should_implement_trait)]
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
///
/// Prefer implementing [`TypedSpecialist`] and wrapping in [`Typed`] —
/// that fills `input_schema` / `output_schema` automatically from
/// `#[derive(JsonSchema)]`.
#[async_trait]
pub trait Specialist: Send + Sync {
    fn name(&self) -> Name;

    /// One-line hint shown to MCP clients and agent planners. Default empty.
    fn description(&self) -> &'static str {
        ""
    }

    /// JSON Schema for the input argument to `handle`. Default `{}` (any JSON).
    /// Override with a real schema so clients can validate before calling.
    fn input_schema(&self) -> Value {
        json!({ "type": "object" })
    }

    /// JSON Schema for the `Ok(Value)` returned from `handle`. Default `{}`.
    fn output_schema(&self) -> Value {
        json!({ "type": "object" })
    }

    async fn handle(&self, req: Value) -> Result<Value>;
}

/// Boxed specialist — registry values share this type.
pub type Boxed = Arc<dyn Specialist>;

/// Typed, DSPy-inspired specialist trait.
///
/// Implementors declare concrete `Input` / `Output` types with serde +
/// [`JsonSchema`] derives, plus a `const NAME` and `const DESCRIPTION`.
/// The [`Typed`] adapter wraps any `TypedSpecialist` and implements the
/// dyn-safe [`Specialist`] trait by driving serde + `schema_for!`.
///
/// Uses native `impl Future` (AFIT) in the return position rather than
/// `#[async_trait]` — this trait is never turned into a trait object, so
/// the AFIT restriction is fine. Erasure happens at the [`Specialist`]
/// level via [`Typed`].
pub trait TypedSpecialist: Send + Sync + 'static {
    type Input: DeserializeOwned + JsonSchema + Send;
    type Output: Serialize + JsonSchema + Send;

    const NAME: Name;
    const DESCRIPTION: &'static str;

    fn handle_typed(
        &self,
        req: Self::Input,
    ) -> impl std::future::Future<Output = Result<Self::Output>> + Send;
}

/// Adapter that erases a [`TypedSpecialist`] into a dyn-safe [`Specialist`].
///
/// Use [`Typed::arc`] to wrap a typed specialist and drop the resulting
/// `Arc<dyn Specialist>` into a [`Registry`]. The blanket
/// `impl<T: TypedSpecialist> Specialist for Typed<T>` below handles all
/// the JSON / schema boilerplate.
pub struct Typed<T: TypedSpecialist>(pub T);

impl<T: TypedSpecialist> Typed<T> {
    /// Wrap a typed specialist and return it as an `Arc<dyn Specialist>`
    /// ready to insert into a [`Registry`].
    pub fn arc(inner: T) -> Arc<dyn Specialist> {
        Arc::new(Typed(inner))
    }
}

#[async_trait]
impl<T: TypedSpecialist> Specialist for Typed<T> {
    fn name(&self) -> Name {
        T::NAME
    }

    fn description(&self) -> &'static str {
        T::DESCRIPTION
    }

    fn input_schema(&self) -> Value {
        // `schema_for!` produces a `schemars::Schema` in 1.x; it derefs to
        // a `serde_json::Value` via `.to_value()`.
        serde_json::to_value(schema_for!(T::Input)).unwrap_or_else(|_| json!({ "type": "object" }))
    }

    fn output_schema(&self) -> Value {
        serde_json::to_value(schema_for!(T::Output)).unwrap_or_else(|_| json!({ "type": "object" }))
    }

    async fn handle(&self, req: Value) -> Result<Value> {
        let typed: T::Input = serde_json::from_value(req)
            .map_err(|e| anyhow!("invalid input for {}: {e}", T::NAME.as_str()))?;
        let out: T::Output = self.0.handle_typed(typed).await?;
        let v = serde_json::to_value(out)
            .map_err(|e| anyhow!("failed to serialize output for {}: {e}", T::NAME.as_str()))?;
        Ok(v)
    }
}

pub struct Registry {
    map: HashMap<Name, Boxed>,
}

impl Registry {
    /// Build the default registry. Seeded with all 17 stubs, except
    /// `Anvil` which is swapped for the [`AnvilSpecialist`] typed demo so
    /// MCP clients see a real input schema out of the box. Callers replace
    /// the remaining stubs with real impls via `insert`.
    pub fn default_stubs() -> Self {
        let mut map: HashMap<Name, Boxed> = HashMap::new();
        for n in Name::ALL {
            map.insert(*n, Arc::new(Stub(*n)));
        }
        // Typed demo: Anvil is the first specialist to publish a real
        // JsonSchema via `Typed<AnvilSpecialist>`. Everything else stays a
        // Stub until a real TypedSpecialist impl lands.
        map.insert(Name::Anvil, Typed::arc(AnvilSpecialist));
        Self { map }
    }

    pub fn insert(&mut self, s: Boxed) {
        self.map.insert(s.name(), s);
    }

    pub fn get(&self, n: Name) -> Option<&Boxed> {
        self.map.get(&n)
    }

    pub async fn dispatch(&self, name: &str, req: Value) -> Result<Value> {
        let n = Name::from_str(name).ok_or_else(|| anyhow!("unknown specialist '{name}'"))?;
        let s = self
            .map
            .get(&n)
            .ok_or_else(|| anyhow!("no impl registered for {name}"))?;
        s.handle(req).await
    }

    pub fn names(&self) -> Vec<&'static str> {
        self.map.keys().map(|n| n.as_str()).collect()
    }
}

impl Default for Registry {
    fn default() -> Self {
        Self::default_stubs()
    }
}

/// Shape-correct stub. Returns `{specialist, status:"stub", echo:req}` so the
/// wiring pipeline (HTTP → router → MCP → Discord relay) can be validated
/// before real implementations land.
struct Stub(Name);

#[async_trait]
impl Specialist for Stub {
    fn name(&self) -> Name {
        self.0
    }
    async fn handle(&self, req: Value) -> Result<Value> {
        Ok(json!({ "specialist": self.0.as_str(), "status": "stub", "echo": req }))
    }
}

// ---------------------------------------------------------------------------
// Concrete typed specialist: Anvil (build/bench watcher).
//
// First real `TypedSpecialist` impl — acts as both a demo and a test
// fixture. `handle_typed` echoes the request; real build/bench wiring lands
// later. The point right now is to prove the schema plumbing: MCP clients
// calling `tools/list` should see a non-trivial JSON Schema for Anvil's
// arguments rather than the passthrough `{"type":"object"}` stub.

/// Anvil input. `cmd` is required; `sha` defaults to HEAD.
#[derive(serde::Deserialize, JsonSchema)]
pub struct AnvilRequest {
    /// Sub-command: "status" | "build" | "bench"
    pub cmd: String,
    /// Optional target commit SHA (defaults to HEAD)
    #[serde(default)]
    pub sha: Option<String>,
}

/// Anvil response — shape stable even when `tok_per_s` is unknown.
#[derive(serde::Serialize, JsonSchema)]
pub struct AnvilResponse {
    pub cmd: String,
    pub status: String,
    pub tok_per_s: Option<f32>,
}

/// Typed Anvil specialist. Stub-echoes for now; real impl lands with the
/// watcher integration in halo-core / halo-server.
pub struct AnvilSpecialist;

impl TypedSpecialist for AnvilSpecialist {
    type Input = AnvilRequest;
    type Output = AnvilResponse;
    const NAME: Name = Name::Anvil;
    const DESCRIPTION: &'static str = "anvil — clone, build, and benchmark a repo end-to-end.";

    async fn handle_typed(&self, req: Self::Input) -> Result<Self::Output> {
        Ok(AnvilResponse {
            cmd: req.cmd,
            status: "stub".to_string(),
            tok_per_s: None,
        })
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
        // Muse is still a plain Stub (Anvil is now typed).
        let out = r.dispatch("muse", json!({"cmd":"ping"})).await.unwrap();
        assert_eq!(out["specialist"], "muse");
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
            fn name(&self) -> Name {
                Name::Anvil
            }
            async fn handle(&self, _req: Value) -> Result<Value> {
                Ok(json!({"specialist":"anvil","status":"real"}))
            }
        }
        let mut r = Registry::default_stubs();
        r.insert(Arc::new(Real));
        let out = r.dispatch("anvil", json!({})).await.unwrap();
        assert_eq!(out["status"], "real");
    }

    // -------- TypedSpecialist tests --------

    #[tokio::test]
    async fn typed_anvil_roundtrips_valid_input() {
        // Typed roundtrip: JSON → Input → handle_typed → Output → JSON.
        let a: Arc<dyn Specialist> = Typed::arc(AnvilSpecialist);
        let out = a
            .handle(json!({"cmd": "bench", "sha": "deadbeef"}))
            .await
            .unwrap();
        assert_eq!(out["cmd"], "bench");
        assert_eq!(out["status"], "stub");
        // `tok_per_s: None` serializes to JSON null.
        assert!(out["tok_per_s"].is_null());
    }

    #[test]
    fn typed_anvil_input_schema_has_cmd_string_property() {
        let a = Typed(AnvilSpecialist);
        let schema = a.input_schema();
        // schemars emits a Draft 2020-12 object schema with a `properties`
        // map. We deliberately only assert on the stable bits — version-
        // specific meta keys (title, $schema, etc.) are allowed to drift.
        assert!(schema.is_object(), "schema must be an object");
        let props = schema
            .get("properties")
            .expect("input_schema must expose properties");
        assert_eq!(
            props["cmd"]["type"], "string",
            "cmd should be typed as string, got {schema}"
        );
        // `required` should include `cmd` (no serde default, no Option).
        let required = schema
            .get("required")
            .and_then(|r| r.as_array())
            .expect("required array");
        assert!(
            required.iter().any(|v| v == "cmd"),
            "required should contain 'cmd', got {required:?}"
        );
    }

    #[tokio::test]
    async fn typed_anvil_schema_valid_input_succeeds() {
        let a: Arc<dyn Specialist> = Typed::arc(AnvilSpecialist);
        // Minimal valid payload: cmd present, sha omitted (serde default).
        let out = a.handle(json!({"cmd": "status"})).await.unwrap();
        assert_eq!(out["cmd"], "status");
        assert_eq!(out["status"], "stub");
    }

    #[tokio::test]
    async fn typed_anvil_schema_invalid_input_errors() {
        let a: Arc<dyn Specialist> = Typed::arc(AnvilSpecialist);
        // `cmd` must be a string. Integer → serde error → anyhow::Error.
        let err = a.handle(json!({"cmd": 42})).await.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("invalid input") && msg.contains("anvil"),
            "expected typed-input error, got: {msg}"
        );
    }

    #[test]
    fn typed_anvil_output_schema_mentions_tok_per_s() {
        let a = Typed(AnvilSpecialist);
        let schema = a.output_schema();
        assert!(schema.is_object());
        let props = schema
            .get("properties")
            .expect("output_schema must expose properties");
        assert!(
            props.get("tok_per_s").is_some(),
            "tok_per_s should appear in output schema, got {schema}"
        );
    }

    #[tokio::test]
    async fn registry_default_wires_anvil_as_typed() {
        // The default registry should route Anvil through Typed<AnvilSpecialist>,
        // so the response shape is the typed one (cmd/status/tok_per_s),
        // NOT the generic Stub shape ({specialist, status, echo}).
        let r = Registry::default_stubs();
        let out = r.dispatch("anvil", json!({"cmd": "build"})).await.unwrap();
        assert_eq!(
            out["cmd"], "build",
            "got stub shape instead of typed: {out}"
        );
        assert!(
            out.get("echo").is_none(),
            "Anvil should no longer echo: {out}"
        );
    }
}
