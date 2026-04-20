//! On-disk skill store.
//!
//! Layout (matches Hermes verbatim, `hermes` → `halo`):
//!
//! ```text
//! ~/.halo/skills/<category>/<name>/
//! ├── SKILL.md           # frontmatter + markdown (this crate reads/writes)
//! ├── references/        # optional — untouched by this crate
//! ├── templates/         # optional — untouched
//! ├── scripts/           # optional — untouched
//! └── assets/            # optional — untouched
//! ```
//!
//! The store keeps zero in-memory state beyond the root path. Every call
//! walks the filesystem fresh — fine at the scale of "a few hundred skills
//! per user" we're aiming at, and it keeps the self-improvement loop free
//! of staleness bugs.

use anyhow::{Context, Result, anyhow, bail};
use std::fs;
use std::path::{Path, PathBuf};

use super::format::Skill;

/// Reads/writes SKILL.md files under a root directory.
///
/// Use [`SkillStore::new`] for the default `~/.halo/skills` root;
/// tests should use [`SkillStore::with_root`] to isolate against a
/// tempdir so they don't mutate the real user home.
#[derive(Debug, Clone)]
pub struct SkillStore {
    root: PathBuf,
}

impl SkillStore {
    /// Create a store rooted at `~/.halo/skills`. The directory is created
    /// lazily on the first write.
    pub fn new() -> Result<Self> {
        let home = dirs::home_dir()
            .ok_or_else(|| anyhow!("could not resolve user home directory"))?;
        Ok(Self::with_root(home.join(".halo").join("skills")))
    }

    /// Create a store rooted at an explicit path. Used by tests + callers
    /// that want to host multiple skill roots on one machine.
    pub fn with_root(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// The configured root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// List every skill under the store. Missing root → empty vec (not
    /// an error — a fresh machine has no skills).
    pub fn list(&self) -> Result<Vec<Skill>> {
        if !self.root.exists() {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        for cat_entry in fs::read_dir(&self.root)
            .with_context(|| format!("read_dir {}", self.root.display()))?
        {
            let cat_entry = cat_entry?;
            if !cat_entry.file_type()?.is_dir() {
                continue;
            }
            for skill_entry in fs::read_dir(cat_entry.path())? {
                let skill_entry = skill_entry?;
                if !skill_entry.file_type()?.is_dir() {
                    continue;
                }
                let skill_md = skill_entry.path().join("SKILL.md");
                if !skill_md.exists() {
                    continue;
                }
                let src = fs::read_to_string(&skill_md)
                    .with_context(|| format!("read {}", skill_md.display()))?;
                match Skill::parse(&src) {
                    Ok(s) => out.push(s),
                    Err(e) => {
                        tracing::warn!(
                            path = %skill_md.display(),
                            error = %e,
                            "skipping malformed SKILL.md"
                        );
                    }
                }
            }
        }
        Ok(out)
    }

    /// Fetch a single skill by `name`. Returns `Ok(None)` if no skill with
    /// that name exists (not an error — callers routinely probe).
    pub fn get(&self, name: &str) -> Result<Option<Skill>> {
        let Some(path) = self.find_path(name)? else {
            return Ok(None);
        };
        let src = fs::read_to_string(&path)
            .with_context(|| format!("read {}", path.display()))?;
        Skill::parse(&src).map(Some)
    }

    /// Write a new skill to disk. Errors if a skill with the same name
    /// already exists — callers that want overwrite semantics should use
    /// [`SkillStore::edit`] after a successful `get`.
    pub fn create(&mut self, skill: Skill) -> Result<()> {
        if self.find_path(&skill.name)?.is_some() {
            bail!("skill '{}' already exists; use edit to replace", skill.name);
        }
        let dir = self.root.join(skill.category()).join(&skill.name);
        fs::create_dir_all(&dir)
            .with_context(|| format!("mkdir -p {}", dir.display()))?;
        let path = dir.join("SKILL.md");
        let rendered = skill.render()?;
        fs::write(&path, rendered)
            .with_context(|| format!("write {}", path.display()))?;
        Ok(())
    }

    /// String-replace `old` with `new` inside the **rendered** SKILL.md
    /// (frontmatter + body, same as `sed -i`). Errors if `old` is not
    /// present — that is a safety feature: the LLM should fail loud rather
    /// than silently no-op when its target text has moved.
    ///
    /// `old` must match exactly once. On error the file is not modified.
    pub fn patch(&mut self, name: &str, old: &str, new: &str) -> Result<()> {
        if old.is_empty() {
            bail!("patch: old string must be non-empty");
        }
        let path = self
            .find_path(name)?
            .ok_or_else(|| anyhow!("patch: no skill named '{name}'"))?;
        let src = fs::read_to_string(&path)
            .with_context(|| format!("read {}", path.display()))?;
        if !src.contains(old) {
            bail!("patch: old string not found in skill '{name}'");
        }
        let matches = src.matches(old).count();
        if matches > 1 {
            bail!(
                "patch: old string is ambiguous in skill '{name}' \
                 ({matches} occurrences); narrow the snippet"
            );
        }
        let updated = src.replacen(old, new, 1);
        fs::write(&path, updated)
            .with_context(|| format!("write {}", path.display()))?;
        Ok(())
    }

    /// Replace the markdown body of a skill, preserving frontmatter.
    pub fn edit(&mut self, name: &str, new_body: String) -> Result<()> {
        let path = self
            .find_path(name)?
            .ok_or_else(|| anyhow!("edit: no skill named '{name}'"))?;
        let src = fs::read_to_string(&path)
            .with_context(|| format!("read {}", path.display()))?;
        let mut skill = Skill::parse(&src)?;
        skill.body = new_body;
        fs::write(&path, skill.render()?)
            .with_context(|| format!("write {}", path.display()))?;
        Ok(())
    }

    /// Remove a skill's directory entirely (SKILL.md + any references/
    /// templates/scripts/assets). Errors if the skill doesn't exist.
    pub fn delete(&mut self, name: &str) -> Result<()> {
        let path = self
            .find_path(name)?
            .ok_or_else(|| anyhow!("delete: no skill named '{name}'"))?;
        let dir = path
            .parent()
            .ok_or_else(|| anyhow!("delete: SKILL.md has no parent dir"))?;
        fs::remove_dir_all(dir)
            .with_context(|| format!("rm -rf {}", dir.display()))?;
        Ok(())
    }

    /// Locate the SKILL.md for `name`, scanning every category subdir.
    /// Returns `None` if no such skill exists.
    fn find_path(&self, name: &str) -> Result<Option<PathBuf>> {
        if !self.root.exists() {
            return Ok(None);
        }
        for cat_entry in fs::read_dir(&self.root)? {
            let cat_entry = cat_entry?;
            if !cat_entry.file_type()?.is_dir() {
                continue;
            }
            let candidate = cat_entry.path().join(name).join("SKILL.md");
            if candidate.exists() {
                return Ok(Some(candidate));
            }
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skills::format::Metadata;
    use tempfile::TempDir;

    fn sample_skill(name: &str, category: &str) -> Skill {
        Skill {
            name: name.into(),
            description: "test skill".into(),
            version: "0.1.0".into(),
            platforms: vec!["linux".into()],
            metadata_halo: Metadata {
                tags: vec!["test".into()],
                category: category.into(),
                fallback_for_toolsets: vec![],
                requires_toolsets: vec![],
            },
            body: "\n# Body\n\noriginal content\n".into(),
        }
    }

    #[test]
    fn create_then_get_then_list_roundtrip() {
        let td = TempDir::new().unwrap();
        let mut store = SkillStore::with_root(td.path().to_path_buf());

        // Fresh store → empty list, no entry.
        assert!(store.list().unwrap().is_empty());
        assert!(store.get("missing").unwrap().is_none());

        store.create(sample_skill("alpha", "demos")).unwrap();
        store.create(sample_skill("beta", "demos")).unwrap();

        let got = store.get("alpha").unwrap().expect("alpha exists");
        assert_eq!(got.name, "alpha");
        assert_eq!(got.metadata_halo.category, "demos");

        let mut names: Vec<String> =
            store.list().unwrap().into_iter().map(|s| s.name).collect();
        names.sort();
        assert_eq!(names, vec!["alpha", "beta"]);

        // File layout: <root>/demos/alpha/SKILL.md
        assert!(td.path().join("demos/alpha/SKILL.md").exists());
        assert!(td.path().join("demos/beta/SKILL.md").exists());
    }

    #[test]
    fn patch_happy_path_replaces_once() {
        let td = TempDir::new().unwrap();
        let mut store = SkillStore::with_root(td.path().to_path_buf());
        store.create(sample_skill("patchme", "demos")).unwrap();

        store.patch("patchme", "original content", "patched!").unwrap();

        let got = store.get("patchme").unwrap().unwrap();
        assert!(got.body.contains("patched!"));
        assert!(!got.body.contains("original content"));
    }

    #[test]
    fn patch_errors_when_old_string_missing_and_does_not_mutate() {
        let td = TempDir::new().unwrap();
        let mut store = SkillStore::with_root(td.path().to_path_buf());
        store.create(sample_skill("strict", "demos")).unwrap();

        let before =
            std::fs::read_to_string(td.path().join("demos/strict/SKILL.md")).unwrap();

        let err = store
            .patch("strict", "THIS STRING IS NOT PRESENT", "anything")
            .unwrap_err();
        assert!(err.to_string().contains("not found"));

        // No mutation.
        let after =
            std::fs::read_to_string(td.path().join("demos/strict/SKILL.md")).unwrap();
        assert_eq!(before, after);
    }

    #[test]
    fn patch_errors_on_ambiguous_match() {
        let td = TempDir::new().unwrap();
        let mut store = SkillStore::with_root(td.path().to_path_buf());
        let mut s = sample_skill("ambig", "demos");
        s.body = "hi hi hi".into();
        store.create(s).unwrap();

        let err = store.patch("ambig", "hi", "bye").unwrap_err();
        assert!(err.to_string().contains("ambiguous"));
    }

    #[test]
    fn edit_replaces_body_preserves_frontmatter() {
        let td = TempDir::new().unwrap();
        let mut store = SkillStore::with_root(td.path().to_path_buf());
        store.create(sample_skill("editme", "demos")).unwrap();

        store
            .edit("editme", "\n# Replaced\n\nwhole new body\n".into())
            .unwrap();

        let got = store.get("editme").unwrap().unwrap();
        assert!(got.body.contains("whole new body"));
        // Frontmatter survived unchanged.
        assert_eq!(got.name, "editme");
        assert_eq!(got.version, "0.1.0");
        assert_eq!(got.metadata_halo.category, "demos");
    }

    #[test]
    fn delete_removes_on_disk_entry() {
        let td = TempDir::new().unwrap();
        let mut store = SkillStore::with_root(td.path().to_path_buf());
        store.create(sample_skill("doomed", "demos")).unwrap();
        let dir = td.path().join("demos/doomed");
        assert!(dir.exists());

        store.delete("doomed").unwrap();
        assert!(!dir.exists());
        assert!(store.get("doomed").unwrap().is_none());
    }

    #[test]
    fn create_rejects_duplicate_names() {
        let td = TempDir::new().unwrap();
        let mut store = SkillStore::with_root(td.path().to_path_buf());
        store.create(sample_skill("dup", "demos")).unwrap();
        let err = store.create(sample_skill("dup", "demos")).unwrap_err();
        assert!(err.to_string().contains("already exists"));
    }
}
