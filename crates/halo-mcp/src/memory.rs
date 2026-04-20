//! `memory_manage` MCP tool — exposes the halo-agents MEMORY.md / USER.md
//! file-backed memory surface to any MCP client (Hermes Agent, Claude Code, …).
//!
//! One tool, three actions (add / replace / remove), two target kinds
//! (memory / user). Mirrors Hermes' memory tool semantics so conversations
//! roundtrip between the two runtimes.
//!
//! # Argument schema
//!
//! ```json
//! {
//!   "action": "add" | "replace" | "remove",
//!   "kind":   "memory" | "user",      // default "memory"
//!   "entry":  "<text>",               // required for add / replace
//!   "match":  "<substring>"           // required for replace / remove
//! }
//! ```

use anyhow::{Result, anyhow, bail};
use halo_agents::{MemoryKind, MemoryStore};
use serde_json::{Value, json};

use crate::registry::Tool;

pub const TOOL_NAME: &str = "memory_manage";

pub const DESCRIPTION: &str =
    "memory_manage — add / replace / remove entries in halo's MEMORY.md (agent \
     notes) and USER.md (user profile). Hermes-compatible § delimiter.";

pub fn input_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "replace", "remove"],
                "description": "Which mutation to perform."
            },
            "kind": {
                "type": "string",
                "enum": ["memory", "user"],
                "description": "Target file. Default 'memory' (MEMORY.md)."
            },
            "entry": {
                "type": "string",
                "description": "Entry text. Required for add / replace."
            },
            "match": {
                "type": "string",
                "description": "Substring to locate the target entry. Required for replace / remove."
            }
        },
        "required": ["action"]
    })
}

pub fn tool() -> Tool {
    Tool {
        name: TOOL_NAME.to_string(),
        description: DESCRIPTION.to_string(),
        input_schema: input_schema(),
    }
}

fn parse_kind(args: &serde_json::Map<String, Value>) -> MemoryKind {
    match args.get("kind").and_then(|v| v.as_str()) {
        Some("user") => MemoryKind::User,
        _            => MemoryKind::Memory,
    }
}

fn dispatch(store: &std::sync::Mutex<MemoryStore>, args: Value) -> Result<String> {
    let obj = args.as_object()
        .ok_or_else(|| anyhow!("memory_manage: arguments must be a JSON object"))?;
    let action = obj.get("action").and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("memory_manage: missing required field 'action'"))?;
    let kind = parse_kind(obj);

    let guard = store.lock().unwrap_or_else(|p| p.into_inner());

    match action {
        "add" => {
            let entry = obj.get("entry").and_then(|v| v.as_str())
                .ok_or_else(|| anyhow!("memory_manage: action=add requires 'entry'"))?;
            guard.add(kind, entry)?;
            Ok(format!("added entry to {}", kind.filename()))
        }
        "replace" => {
            let needle = obj.get("match").and_then(|v| v.as_str())
                .ok_or_else(|| anyhow!("memory_manage: action=replace requires 'match'"))?;
            let entry  = obj.get("entry").and_then(|v| v.as_str())
                .ok_or_else(|| anyhow!("memory_manage: action=replace requires 'entry'"))?;
            guard.replace(kind, needle, entry)?;
            Ok(format!("replaced matching entry in {}", kind.filename()))
        }
        "remove" => {
            let needle = obj.get("match").and_then(|v| v.as_str())
                .ok_or_else(|| anyhow!("memory_manage: action=remove requires 'match'"))?;
            guard.remove(kind, needle)?;
            Ok(format!("removed matching entry from {}", kind.filename()))
        }
        other => bail!("memory_manage: unknown action '{other}'"),
    }
}

pub fn handle(store: &std::sync::Mutex<MemoryStore>, args: Value) -> Value {
    match dispatch(store, args) {
        Ok(msg) => json!({ "message": msg }),
        Err(e)  => json!({ "error": e.to_string(), "tool": TOOL_NAME }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use tempfile::TempDir;

    fn fresh() -> (TempDir, Mutex<MemoryStore>) {
        let td = TempDir::new().unwrap();
        let store = MemoryStore::with_root(td.path().to_path_buf()).unwrap();
        (td, Mutex::new(store))
    }

    #[test]
    fn tool_descriptor_shape() {
        let t = tool();
        assert_eq!(t.name, TOOL_NAME);
        assert!(!t.description.is_empty());
        let actions = t.input_schema["properties"]["action"]["enum"]
            .as_array().unwrap();
        assert_eq!(actions.len(), 3);
    }

    #[test]
    fn add_writes_to_memory_md() {
        let (td, store) = fresh();
        let resp = handle(&store, json!({"action":"add","entry":"gpu is gfx1151"}));
        assert!(resp.get("error").is_none(), "got {resp}");
        let body = std::fs::read_to_string(td.path().join("MEMORY.md")).unwrap();
        assert!(body.contains("gpu is gfx1151"));
    }

    #[test]
    fn kind_user_writes_to_user_md() {
        let (td, store) = fresh();
        handle(&store, json!({"action":"add","kind":"user","entry":"bcloud, MT"}));
        assert!(td.path().join("USER.md").exists());
        assert!(!td.path().join("MEMORY.md").exists());
    }

    #[test]
    fn replace_swaps_matching_entry() {
        let (td, store) = fresh();
        handle(&store, json!({"action":"add","entry":"GPU is old"}));
        let r = handle(&store, json!({
            "action":"replace","match":"GPU","entry":"GPU gfx1151 128GB LPDDR5"
        }));
        assert!(r.get("error").is_none(), "got {r}");
        let body = std::fs::read_to_string(td.path().join("MEMORY.md")).unwrap();
        assert!(body.contains("128GB"));
        assert!(!body.contains("is old"));
    }

    #[test]
    fn remove_drops_matching_entry() {
        let (td, store) = fresh();
        handle(&store, json!({"action":"add","entry":"keep me"}));
        handle(&store, json!({"action":"add","entry":"drop me"}));
        let r = handle(&store, json!({"action":"remove","match":"drop"}));
        assert!(r.get("error").is_none(), "got {r}");
        let body = std::fs::read_to_string(td.path().join("MEMORY.md")).unwrap();
        assert!(body.contains("keep me"));
        assert!(!body.contains("drop me"));
    }

    #[test]
    fn unknown_action_errors() {
        let (_td, store) = fresh();
        let r = handle(&store, json!({"action":"burninate","entry":"x"}));
        assert!(r.get("error").is_some());
    }

    #[test]
    fn replace_without_match_field_errors() {
        let (_td, store) = fresh();
        let r = handle(&store, json!({"action":"replace","entry":"x"}));
        assert!(r.get("error").is_some());
    }
}
