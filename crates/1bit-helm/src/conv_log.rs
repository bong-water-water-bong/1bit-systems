//! JSONL conversation logs under `~/.halo/helm/conversations/<ts>.jsonl`.
//!
//! One file per session. One JSON object per turn, `{role, content, ts}`.
//! The server-side FTS5 `sessions` table is the canonical store (per
//! `Crate-halo-helm.md` non-goal: "Not a chat-logging product"); these
//! local files are a debug breadcrumb for when the server was down + the
//! user still wants to copy-paste their last exchange.
//!
//! We write on explicit close — NOT every turn — because the SSE stream
//! often produces 50+ deltas per assistant reply and flushing each one
//! would thrash the filesystem.

use crate::conversation::Conversation;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

/// On-disk log entry. Mirrors `ChatTurn` but kept separate so future
/// schema tweaks don't ripple into the wire type.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LogEntry {
    pub role: String,
    pub content: String,
    pub ts: u64,
}

/// Default root: `~/.halo/helm/conversations/`. Falls back to `./` if
/// HOME is unset (headless CI — we'd rather write into cwd than panic).
pub fn default_root() -> PathBuf {
    if let Some(home) = dirs::home_dir() {
        home.join(".halo").join("helm").join("conversations")
    } else {
        PathBuf::from(".")
    }
}

/// Write `conv` as a JSONL file named `<unix_ts>.jsonl` under `root`.
/// Returns the file path. Creates `root` if missing.
pub fn write_session(root: &Path, conv: &Conversation) -> Result<PathBuf> {
    fs::create_dir_all(root).with_context(|| format!("mkdir {}", root.display()))?;
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let path = root.join(format!("{ts}.jsonl"));
    let mut f = fs::File::create(&path).with_context(|| format!("create {}", path.display()))?;
    for turn in &conv.turns {
        let entry = LogEntry {
            role: turn.role.to_string(),
            content: turn.content.clone(),
            ts: turn.ts,
        };
        let line = serde_json::to_string(&entry).context("serialize log entry")?;
        writeln!(f, "{line}").with_context(|| format!("write {}", path.display()))?;
    }
    Ok(path)
}

/// Parse a JSONL log back into a Vec<LogEntry>. Malformed lines are
/// skipped with a warning — we never want "user's last session" to be
/// gated behind a strict parse.
pub fn read_session(path: &Path) -> Result<Vec<LogEntry>> {
    let body = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut out = Vec::new();
    for (i, line) in body.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<LogEntry>(line) {
            Ok(e) => out.push(e),
            Err(e) => {
                tracing::warn!(
                    path = %path.display(),
                    line = i + 1,
                    err = %e,
                    "skipping malformed log line"
                );
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversation::Conversation;
    use tempfile::TempDir;

    #[test]
    fn roundtrip_write_then_read_preserves_turns() {
        let td = TempDir::new().unwrap();
        let mut conv = Conversation::new();
        conv.push_user("what is 2+2?".into());
        conv.push_assistant("4".into());
        conv.push_user("thanks".into());

        let path = write_session(td.path(), &conv).unwrap();
        assert!(path.exists());
        assert!(path.extension().unwrap() == "jsonl");

        let entries = read_session(&path).unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].role, "user");
        assert_eq!(entries[0].content, "what is 2+2?");
        assert_eq!(entries[1].role, "assistant");
        assert_eq!(entries[1].content, "4");
        assert_eq!(entries[2].role, "user");
        assert_eq!(entries[2].content, "thanks");
    }

    #[test]
    fn write_creates_root_if_missing() {
        let td = TempDir::new().unwrap();
        let nested = td.path().join("does/not/exist/yet");
        let conv = Conversation::new();
        // empty conv still produces a file (zero lines) — useful as a
        // "user opened and closed helm" breadcrumb.
        let path = write_session(&nested, &conv).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn read_skips_malformed_lines() {
        let td = TempDir::new().unwrap();
        let p = td.path().join("mixed.jsonl");
        let mut f = std::fs::File::create(&p).unwrap();
        writeln!(f, r#"{{"role":"user","content":"hi","ts":1}}"#).unwrap();
        writeln!(f, "not json at all").unwrap();
        writeln!(f, r#"{{"role":"assistant","content":"hello","ts":2}}"#).unwrap();
        let entries = read_session(&p).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].role, "user");
        assert_eq!(entries[1].role, "assistant");
    }
}
