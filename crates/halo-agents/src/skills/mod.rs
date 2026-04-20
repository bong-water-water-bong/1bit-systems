//! halo-agents skills subsystem.
//!
//! This module implements the [Hermes-compatible] skill format + on-disk
//! layout verbatim (`~/.halo/skills/<category>/<name>/SKILL.md`, YAML
//! frontmatter with `metadata.halo.*`). Interop target: agentskills.io.
//!
//! [Hermes-compatible]: https://github.com/NousResearch/hermes-agent
//!
//! # Surface
//!
//! * [`Skill`] — parsed SKILL.md (frontmatter + markdown body).
//! * [`SkillStore`] — CRUD over the on-disk layout. Use
//!   [`SkillStore::new`] for the default `~/.halo/skills/` root, or
//!   [`SkillStore::with_root`] to point at a tempdir in tests.
//! * [`SkillAction`] + [`apply`] — the "skill_manage" tool surface. This
//!   is the **entire** self-improvement-loop API: the LLM produces one
//!   `SkillAction` and we execute it.
//!
//! # Non-goals (intentional gaps)
//!
//! * No FTS5 search here; that lives in a separate module.
//! * No autonomous creation triggers (heuristic layer is upstream of us).
//! * `references/`, `templates/`, `scripts/`, `assets/` subdirs may exist
//!   empty; we neither read nor write them.

pub mod format;
pub mod store;
pub mod tool;

pub use format::{Metadata, Skill};
pub use store::SkillStore;
pub use tool::{SkillAction, apply};
