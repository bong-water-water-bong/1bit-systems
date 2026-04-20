//! `skill_manage` MCP tool — exposes the halo-agents skill
//! self-improvement surface (Create / Patch / Edit / Delete) to any MCP
//! client (Hermes Agent, Claude Code, …).
//!
//! One tool, four actions, discriminated by an `action` field. The handler
//! translates JSON arguments into a [`halo_agents::SkillAction`] and calls
//! [`halo_agents::skills::tool::apply`] against a shared
//! [`halo_agents::SkillStore`].
//!
//! # Why a separate module
//!
//! The 17 specialists come from `halo_agents::Name::ALL` and are routed
//! via `Registry::dispatch`. Skills are not a specialist — they are the
//! self-modification API *for* the agent fleet. Keeping them in their
//! own MCP tool preserves the one-tool-per-specialist invariant the
//! dispatcher relies on and keeps the "who can edit SKILL.md" audit
//! surface obvious to ops.
//!
//! # Argument schema
//!
//! ```json
//! {
//!   "action": "create" | "patch" | "edit" | "delete",
//!   "name": "<skill-name>",        // required for all actions
//!   "category": "<category>",      // create only, optional (default "uncategorized")
//!   "description": "<one-liner>",  // create only, optional
//!   "body": "<markdown>",          // create / edit
//!   "old_string": "<exact>",       // patch
//!   "new_string": "<exact>"        // patch
//! }
//! ```

use anyhow::{Result, anyhow, bail};
use halo_agents::{Skill, SkillAction, SkillStore, skills::tool::apply};
use serde_json::{Value, json};

use crate::registry::Tool;

/// MCP tool name. Clients pass this as `params.name` on `tools/call`.
pub const TOOL_NAME: &str = "skill_manage";

/// Human-readable description shown in `tools/list`.
pub const DESCRIPTION: &str =
    "skill_manage — create / patch / edit / delete SKILL.md files in the \
     halo skill store. Drives the Hermes-compatible self-improvement loop.";

/// JSON Schema for `skill_manage` arguments. Mirrors the four-variant
/// [`SkillAction`] surface. `action` is an enum; `name` is always required;
/// the remaining fields are conditionally required by action but the
/// handler validates that — MCP clients often strip JSON-Schema
/// `oneOf`/`if`/`then` branches when generating UIs, so we keep the
/// published schema flat and enforce the per-action required set in Rust.
pub fn input_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "patch", "edit", "delete"],
                "description": "Which skill mutation to perform."
            },
            "name": {
                "type": "string",
                "description": "Skill name (directory slug, unique across categories)."
            },
            "category": {
                "type": "string",
                "description": "Category directory. `create` only; default 'uncategorized'."
            },
            "description": {
                "type": "string",
                "description": "One-line skill description. `create` only."
            },
            "body": {
                "type": "string",
                "description": "Markdown body. Required for `create` and `edit`."
            },
            "old_string": {
                "type": "string",
                "description": "Exact substring to replace. Required for `patch`."
            },
            "new_string": {
                "type": "string",
                "description": "Replacement text. Required for `patch`."
            }
        },
        "required": ["action", "name"]
    })
}

/// Build the MCP [`Tool`] descriptor. The `ToolRegistry` uses this so
/// `tools/list` surfaces `skill_manage` alongside the 17 specialists.
pub fn tool() -> Tool {
    Tool {
        name: TOOL_NAME.to_string(),
        description: DESCRIPTION.to_string(),
        input_schema: input_schema(),
    }
}

/// Parse JSON arguments into a [`SkillAction`]. Per-action required-field
/// validation lives here — the published JSON Schema only enforces
/// `action` + `name` because the MCP clients we target don't reliably
/// honour `oneOf`.
fn parse_action(args: &Value) -> Result<SkillAction> {
    let obj = args
        .as_object()
        .ok_or_else(|| anyhow!("skill_manage: arguments must be a JSON object"))?;

    let action = obj
        .get("action")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("skill_manage: missing required field 'action'"))?;

    let name = obj
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("skill_manage: missing required field 'name'"))?
        .to_string();

    match action {
        "create" => {
            let description = obj
                .get("description")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let category = obj
                .get("category")
                .and_then(|v| v.as_str())
                .unwrap_or("uncategorized")
                .to_string();
            let body = obj
                .get("body")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let mut skill = Skill::new(name, description);
            skill.metadata_halo.category = category;
            skill.body = body;
            Ok(SkillAction::Create(skill))
        }
        "patch" => {
            let old = obj
                .get("old_string")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    anyhow!("skill_manage: action=patch requires 'old_string'")
                })?
                .to_string();
            let new = obj
                .get("new_string")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    anyhow!("skill_manage: action=patch requires 'new_string'")
                })?
                .to_string();
            Ok(SkillAction::Patch { name, old, new })
        }
        "edit" => {
            let body = obj
                .get("body")
                .and_then(|v| v.as_str())
                .ok_or_else(|| anyhow!("skill_manage: action=edit requires 'body'"))?
                .to_string();
            Ok(SkillAction::Edit { name, body })
        }
        "delete" => Ok(SkillAction::Delete { name }),
        other => bail!("skill_manage: unknown action '{other}'"),
    }
}

/// Execute a `skill_manage` tool call against a shared [`SkillStore`].
///
/// Returns the JSON payload that lands in the MCP `content[0].text` slot:
/// `{"message": "created skill 'foo' in category 'bar'"}` on success or
/// `{"error": "...", "tool": "skill_manage"}` on failure. The server
/// wraps this in the usual `{content: [...], isError: bool}` envelope.
pub fn handle(store: &std::sync::Mutex<SkillStore>, args: Value) -> Value {
    match parse_action(&args) {
        Err(e) => json!({ "error": e.to_string(), "tool": TOOL_NAME }),
        Ok(action) => {
            let mut guard = match store.lock() {
                Ok(g) => g,
                Err(poisoned) => poisoned.into_inner(),
            };
            match apply(&mut guard, action) {
                Ok(msg) => json!({ "message": msg }),
                Err(e) => json!({ "error": e.to_string(), "tool": TOOL_NAME }),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use tempfile::TempDir;

    fn fresh_store() -> (TempDir, Mutex<SkillStore>) {
        let td = TempDir::new().expect("tempdir");
        let store = SkillStore::with_root(td.path().to_path_buf());
        (td, Mutex::new(store))
    }

    #[test]
    fn tool_descriptor_shape() {
        let t = tool();
        assert_eq!(t.name, TOOL_NAME);
        assert!(!t.description.is_empty());
        assert!(t.input_schema.is_object());
        let required = t.input_schema["required"]
            .as_array()
            .expect("required array");
        let names: Vec<&str> = required.iter().filter_map(|v| v.as_str()).collect();
        assert!(names.contains(&"action"));
        assert!(names.contains(&"name"));
        let actions = t.input_schema["properties"]["action"]["enum"]
            .as_array()
            .expect("enum");
        assert_eq!(actions.len(), 4);
    }

    #[test]
    fn create_writes_file_to_disk() {
        let (td, store) = fresh_store();
        let resp = handle(
            &store,
            json!({
                "action": "create",
                "name": "unit-test-skill",
                "category": "tests",
                "description": "scratch skill",
                "body": "# test\n"
            }),
        );
        assert!(resp.get("error").is_none(), "got error: {resp}");
        let msg = resp["message"].as_str().expect("message");
        assert!(msg.contains("created"));
        assert!(msg.contains("unit-test-skill"));

        let on_disk = td.path().join("tests/unit-test-skill/SKILL.md");
        assert!(on_disk.exists(), "SKILL.md not written at {}", on_disk.display());
        let raw = std::fs::read_to_string(&on_disk).unwrap();
        assert!(raw.contains("name: unit-test-skill"));
        assert!(raw.contains("# test"));
    }

    #[test]
    fn patch_replaces_old_string() {
        let (_td, store) = fresh_store();
        // Seed.
        let _ = handle(
            &store,
            json!({
                "action": "create",
                "name": "patch-target",
                "category": "tests",
                "description": "seed",
                "body": "before marker end\n"
            }),
        );
        let resp = handle(
            &store,
            json!({
                "action": "patch",
                "name": "patch-target",
                "old_string": "marker",
                "new_string": "REPLACED"
            }),
        );
        assert!(resp.get("error").is_none(), "got error: {resp}");
        assert!(resp["message"].as_str().unwrap().contains("patched"));

        // Read back through the store to prove persistence.
        let guard = store.lock().unwrap();
        let got = guard.get("patch-target").unwrap().expect("skill present");
        assert!(got.body.contains("REPLACED"));
        assert!(!got.body.contains("marker"));
    }

    #[test]
    fn delete_removes_file_from_disk() {
        let (td, store) = fresh_store();
        let _ = handle(
            &store,
            json!({
                "action": "create",
                "name": "doomed-skill",
                "category": "tests",
                "description": "seed",
                "body": "bye\n"
            }),
        );
        let dir = td.path().join("tests/doomed-skill");
        assert!(dir.exists(), "precondition: dir should exist");

        let resp = handle(
            &store,
            json!({
                "action": "delete",
                "name": "doomed-skill"
            }),
        );
        assert!(resp.get("error").is_none(), "got error: {resp}");
        assert!(resp["message"].as_str().unwrap().contains("deleted"));
        assert!(!dir.exists(), "dir still exists after delete");
    }

    #[test]
    fn edit_replaces_body_and_keeps_frontmatter() {
        let (_td, store) = fresh_store();
        let _ = handle(
            &store,
            json!({
                "action": "create",
                "name": "edit-target",
                "category": "tests",
                "description": "seed",
                "body": "original body\n"
            }),
        );
        let resp = handle(
            &store,
            json!({
                "action": "edit",
                "name": "edit-target",
                "body": "\nbrand-new body\n"
            }),
        );
        assert!(resp.get("error").is_none(), "got error: {resp}");
        assert!(resp["message"].as_str().unwrap().contains("edited"));

        let guard = store.lock().unwrap();
        let got = guard.get("edit-target").unwrap().expect("skill present");
        assert!(got.body.contains("brand-new body"));
        assert!(!got.body.contains("original body"));
        assert_eq!(got.description, "seed");
    }

    #[test]
    fn missing_action_is_surfaced_as_error_not_panic() {
        let (_td, store) = fresh_store();
        let resp = handle(&store, json!({ "name": "foo" }));
        assert!(resp["error"].as_str().unwrap().contains("action"));
        assert_eq!(resp["tool"], TOOL_NAME);
    }

    #[test]
    fn unknown_action_is_rejected_cleanly() {
        let (_td, store) = fresh_store();
        let resp = handle(
            &store,
            json!({ "action": "nuke", "name": "foo" }),
        );
        assert!(resp["error"].as_str().unwrap().contains("nuke"));
    }

    #[test]
    fn patch_without_old_string_errors() {
        let (_td, store) = fresh_store();
        let resp = handle(
            &store,
            json!({ "action": "patch", "name": "foo", "new_string": "x" }),
        );
        assert!(resp["error"].as_str().unwrap().contains("old_string"));
    }
}
