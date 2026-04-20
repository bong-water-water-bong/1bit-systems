//! The `skill_manage` tool surface.
//!
//! Four actions — `Create`, `Patch`, `Edit`, `Delete` — cover the entire
//! Hermes self-improvement loop. An LLM produces one [`SkillAction`], the
//! [`apply`] dispatcher executes it against a [`SkillStore`], and the
//! return is a one-line human-readable confirmation the agent can
//! re-emit as a tool result.
//!
//! No branching / heuristics live here. The "autonomous skill-creation
//! trigger" layer (≥5 successful tool calls, etc.) decides *when* to
//! call `apply`; this module decides *how*.

use anyhow::Result;

use super::format::Skill;
use super::store::SkillStore;

/// One of the four skill-management actions exposed to LLM tool callers.
///
/// This is `#[non_exhaustive]` so we can add `Rename` / `Move` later
/// without breaking existing matchers.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum SkillAction {
    /// Create a new skill.
    Create(Skill),
    /// Targeted `sed`-style replacement inside SKILL.md. Prefer over
    /// `Edit` for small fixes — lower risk, more auditable.
    Patch {
        name: String,
        old: String,
        new: String,
    },
    /// Replace the entire markdown body, keeping frontmatter.
    Edit { name: String, body: String },
    /// Remove the skill directory and all children.
    Delete { name: String },
}

/// Dispatch a [`SkillAction`] against a [`SkillStore`] and return a
/// human-readable confirmation string. Errors bubble up verbatim from
/// the underlying [`SkillStore`] call.
pub fn apply(store: &mut SkillStore, action: SkillAction) -> Result<String> {
    match action {
        SkillAction::Create(skill) => {
            let name = skill.name.clone();
            let category = skill.category().to_string();
            store.create(skill)?;
            Ok(format!("created skill '{name}' in category '{category}'"))
        }
        SkillAction::Patch { name, old, new } => {
            store.patch(&name, &old, &new)?;
            Ok(format!(
                "patched skill '{name}' ({} → {} chars)",
                old.len(),
                new.len()
            ))
        }
        SkillAction::Edit { name, body } => {
            let len = body.len();
            store.edit(&name, body)?;
            Ok(format!("edited skill '{name}' body ({len} chars)"))
        }
        SkillAction::Delete { name } => {
            store.delete(&name)?;
            Ok(format!("deleted skill '{name}'"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skills::format::Metadata;
    use tempfile::TempDir;

    fn mk_skill(name: &str) -> Skill {
        Skill {
            name: name.into(),
            description: "apply-test skill".into(),
            version: "0.1.0".into(),
            platforms: vec!["linux".into()],
            metadata_halo: Metadata {
                tags: vec![],
                category: "test".into(),
                fallback_for_toolsets: vec![],
                requires_toolsets: vec![],
            },
            body: "\noriginal body\n".into(),
        }
    }

    #[test]
    fn apply_create_then_delete_cycle() {
        let td = TempDir::new().unwrap();
        let mut store = SkillStore::with_root(td.path().to_path_buf());

        let msg = apply(&mut store, SkillAction::Create(mk_skill("s1"))).unwrap();
        assert!(msg.contains("created"), "got: {msg}");
        assert!(msg.contains("s1"));
        assert!(td.path().join("test/s1/SKILL.md").exists());

        let msg = apply(
            &mut store,
            SkillAction::Delete {
                name: "s1".into(),
            },
        )
        .unwrap();
        assert!(msg.contains("deleted"));
        assert!(!td.path().join("test/s1").exists());
    }

    #[test]
    fn apply_edit_replaces_body_and_keeps_frontmatter() {
        let td = TempDir::new().unwrap();
        let mut store = SkillStore::with_root(td.path().to_path_buf());
        apply(&mut store, SkillAction::Create(mk_skill("e1"))).unwrap();

        let msg = apply(
            &mut store,
            SkillAction::Edit {
                name: "e1".into(),
                body: "\nbrand new body\n".into(),
            },
        )
        .unwrap();
        assert!(msg.contains("edited"));

        let got = store.get("e1").unwrap().unwrap();
        assert!(got.body.contains("brand new body"));
        assert!(!got.body.contains("original body"));
        // Frontmatter intact.
        assert_eq!(got.name, "e1");
        assert_eq!(got.description, "apply-test skill");
        assert_eq!(got.version, "0.1.0");
        assert_eq!(got.metadata_halo.category, "test");
    }

    #[test]
    fn apply_patch_returns_len_summary() {
        let td = TempDir::new().unwrap();
        let mut store = SkillStore::with_root(td.path().to_path_buf());
        apply(&mut store, SkillAction::Create(mk_skill("p1"))).unwrap();

        let msg = apply(
            &mut store,
            SkillAction::Patch {
                name: "p1".into(),
                old: "original body".into(),
                new: "patched".into(),
            },
        )
        .unwrap();
        assert!(msg.contains("patched"));
        assert!(msg.contains("p1"));
    }
}
