//! File-backed store for MEMORY.md + USER.md.

use anyhow::{anyhow, Context, Result};
use std::fs;
use std::path::PathBuf;

/// Hermes-compatible cap for MEMORY.md body in characters.
pub const MAX_MEMORY_CHARS: usize = 2200;
/// Hermes-compatible cap for USER.md body in characters.
pub const MAX_USER_CHARS: usize = 1375;
/// Separator between entries. Section-sign on its own line.
pub const DELIMITER: &str = "§";

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MemoryKind {
    Memory,
    User,
}

impl MemoryKind {
    pub fn filename(self) -> &'static str {
        match self { Self::Memory => "MEMORY.md", Self::User => "USER.md" }
    }
    pub fn cap(self) -> usize {
        match self { Self::Memory => MAX_MEMORY_CHARS, Self::User => MAX_USER_CHARS }
    }
}

/// Root-scoped file store. Use `MemoryStore::new()` for `~/.halo/memories/`
/// or `MemoryStore::with_root(p)` for isolated tests.
#[derive(Debug, Clone)]
pub struct MemoryStore {
    root: PathBuf,
}

impl MemoryStore {
    /// Default root: `$HOME/.halo/memories/`. Creates the dir if missing.
    pub fn new() -> Result<Self> {
        let home = dirs::home_dir().context("no home dir")?;
        Self::with_root(home.join(".halo").join("memories"))
    }

    /// Test-visible root. Creates the dir if missing.
    pub fn with_root(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        fs::create_dir_all(&root).with_context(|| format!("mkdir {}", root.display()))?;
        Ok(Self { root })
    }

    fn path(&self, kind: MemoryKind) -> PathBuf { self.root.join(kind.filename()) }

    /// Read every entry for `kind`, split on DELIMITER, trimmed.
    pub fn list(&self, kind: MemoryKind) -> Result<Vec<String>> {
        let p = self.path(kind);
        if !p.exists() { return Ok(Vec::new()); }
        let body = fs::read_to_string(&p).with_context(|| format!("read {}", p.display()))?;
        Ok(split_entries(&body))
    }

    /// Append `entry` to the end of `kind`'s file. Fails if the resulting
    /// body would exceed the cap.
    pub fn add(&self, kind: MemoryKind, entry: &str) -> Result<()> {
        let e = entry.trim();
        if e.is_empty() { return Err(anyhow!("entry is empty")); }
        let mut entries = self.list(kind)?;
        entries.push(e.to_string());
        self.write_all(kind, &entries)
    }

    /// Replace the first entry whose substring-match on `needle` succeeds
    /// with `new_entry`. Errors if no match.
    pub fn replace(&self, kind: MemoryKind, needle: &str, new_entry: &str) -> Result<()> {
        let ne = new_entry.trim();
        if ne.is_empty() { return Err(anyhow!("new_entry is empty")); }
        let mut entries = self.list(kind)?;
        let idx = entries.iter().position(|e| e.contains(needle))
            .ok_or_else(|| anyhow!("no entry contains {:?}", needle))?;
        entries[idx] = ne.to_string();
        self.write_all(kind, &entries)
    }

    /// Remove the first entry whose substring-match on `needle` succeeds.
    /// Errors if no match.
    pub fn remove(&self, kind: MemoryKind, needle: &str) -> Result<()> {
        let mut entries = self.list(kind)?;
        let idx = entries.iter().position(|e| e.contains(needle))
            .ok_or_else(|| anyhow!("no entry contains {:?}", needle))?;
        entries.remove(idx);
        self.write_all(kind, &entries)
    }

    /// Render all entries into the on-disk body using DELIMITER. Enforces
    /// the Hermes-compatible char cap.
    fn write_all(&self, kind: MemoryKind, entries: &[String]) -> Result<()> {
        let body = render(entries);
        if body.chars().count() > kind.cap() {
            return Err(anyhow!(
                "{} body {} chars would exceed cap {}; consolidate first",
                kind.filename(), body.chars().count(), kind.cap()
            ));
        }
        let p = self.path(kind);
        fs::write(&p, body).with_context(|| format!("write {}", p.display()))?;
        Ok(())
    }

    /// Frozen snapshot for prompt injection. Hermes loads once per session
    /// and never refreshes; we follow the same discipline.
    pub fn snapshot(&self, kind: MemoryKind) -> Result<String> {
        let p = self.path(kind);
        if !p.exists() { return Ok(String::new()); }
        fs::read_to_string(&p).with_context(|| format!("read {}", p.display()))
    }
}

fn split_entries(body: &str) -> Vec<String> {
    body.split(&format!("\n{}\n", DELIMITER))
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn render(entries: &[String]) -> String {
    entries.iter()
        .map(|e| e.trim())
        .filter(|e| !e.is_empty())
        .collect::<Vec<_>>()
        .join(&format!("\n{}\n", DELIMITER))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn fresh() -> (tempfile::TempDir, MemoryStore) {
        let d = tempdir().unwrap();
        let s = MemoryStore::with_root(d.path().to_path_buf()).unwrap();
        (d, s)
    }

    #[test]
    fn empty_store_reports_no_entries() {
        let (_d, s) = fresh();
        assert!(s.list(MemoryKind::Memory).unwrap().is_empty());
        assert!(s.list(MemoryKind::User).unwrap().is_empty());
        assert_eq!(s.snapshot(MemoryKind::Memory).unwrap(), "");
    }

    #[test]
    fn add_and_list_roundtrip() {
        let (_d, s) = fresh();
        s.add(MemoryKind::Memory, "first note").unwrap();
        s.add(MemoryKind::Memory, "second note").unwrap();
        let got = s.list(MemoryKind::Memory).unwrap();
        assert_eq!(got, vec!["first note", "second note"]);
    }

    #[test]
    fn delimiter_is_section_sign_on_own_line() {
        let (_d, s) = fresh();
        s.add(MemoryKind::Memory, "a").unwrap();
        s.add(MemoryKind::Memory, "b").unwrap();
        let body = s.snapshot(MemoryKind::Memory).unwrap();
        assert_eq!(body, "a\n§\nb");
    }

    #[test]
    fn replace_hits_first_match() {
        let (_d, s) = fresh();
        s.add(MemoryKind::Memory, "GPU gfx1151").unwrap();
        s.add(MemoryKind::Memory, "CPU Ryzen").unwrap();
        s.replace(MemoryKind::Memory, "gfx1151", "GPU gfx1151, 128GB LPDDR5").unwrap();
        let got = s.list(MemoryKind::Memory).unwrap();
        assert_eq!(got, vec!["GPU gfx1151, 128GB LPDDR5", "CPU Ryzen"]);
    }

    #[test]
    fn replace_errors_when_no_match() {
        let (_d, s) = fresh();
        s.add(MemoryKind::Memory, "a").unwrap();
        assert!(s.replace(MemoryKind::Memory, "not-there", "new").is_err());
        // store unchanged
        assert_eq!(s.list(MemoryKind::Memory).unwrap(), vec!["a"]);
    }

    #[test]
    fn remove_deletes_matching_entry() {
        let (_d, s) = fresh();
        s.add(MemoryKind::Memory, "keep me").unwrap();
        s.add(MemoryKind::Memory, "drop me").unwrap();
        s.remove(MemoryKind::Memory, "drop").unwrap();
        assert_eq!(s.list(MemoryKind::Memory).unwrap(), vec!["keep me"]);
    }

    #[test]
    fn cap_rejects_oversize_body() {
        let (_d, s) = fresh();
        let big = "x".repeat(MAX_MEMORY_CHARS + 1);
        assert!(s.add(MemoryKind::Memory, &big).is_err());
    }

    #[test]
    fn user_cap_is_smaller_than_memory_cap() {
        assert!(MAX_USER_CHARS < MAX_MEMORY_CHARS);
    }

    #[test]
    fn user_kind_writes_to_user_md_not_memory_md() {
        let (d, s) = fresh();
        s.add(MemoryKind::User, "bcloud — Halo maintainer, timezone MT").unwrap();
        assert!(d.path().join("USER.md").exists());
        assert!(!d.path().join("MEMORY.md").exists());
    }

    #[test]
    fn empty_entry_rejected() {
        let (_d, s) = fresh();
        assert!(s.add(MemoryKind::Memory, "   \n").is_err());
    }
}
