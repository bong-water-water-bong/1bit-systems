//! `SessionDb` — thin wrapper around a `rusqlite::Connection` to the
//! session-history + FTS5 index at `~/.halo/state.db`.
//!
//! One connection per [`SessionDb`] instance; the caller is responsible
//! for wrapping in `Arc<Mutex<_>>` if cross-task sharing is needed. Most
//! call sites (CLI one-shot `halo history search`, an axum handler with
//! per-request `spawn_blocking`) won't need that.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rusqlite::Connection;

use super::schema::SCHEMA_SQL;

/// Connection handle to `~/.halo/state.db` (or an explicit path in tests).
///
/// Call [`SessionDb::open_default`] from application code and
/// [`SessionDb::open`] from tests. Both run the full DDL + trigger
/// bootstrap on every open; `CREATE IF NOT EXISTS` makes this safe to
/// call repeatedly.
pub struct SessionDb {
    conn: Connection,
}

impl SessionDb {
    /// Open (creating if absent) a session database at the given path.
    ///
    /// Used by tests with a tempdir path, and by
    /// [`SessionDb::open_default`] with `~/.halo/state.db`. The parent
    /// directory is created eagerly; if it already exists, nothing
    /// happens.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create parent dir for {}", path.display()))?;
        }
        let conn =
            Connection::open(path).with_context(|| format!("open sqlite at {}", path.display()))?;
        conn.execute_batch(SCHEMA_SQL)
            .context("run session-db schema bootstrap")?;
        Ok(Self { conn })
    }

    /// Open `~/.halo/state.db` — the canonical path on real systems.
    ///
    /// Falls back to `./.halo/state.db` if we can't resolve a home dir,
    /// matching the resilience pattern the rest of the crate uses
    /// (see `halo-cli::update::home`).
    pub fn open_default() -> Result<Self> {
        Self::open(default_state_db_path())
    }

    /// Borrow the underlying connection. Handy for joined queries that
    /// don't fit the narrow helper API — tests use this to verify row
    /// counts without us having to expose a counter per table.
    pub fn conn(&self) -> &Connection {
        &self.conn
    }

    /// Insert a new session row. Returns the `id` that was inserted so
    /// callers can chain `insert_turn` against it.
    pub fn insert_session(
        &self,
        id: &str,
        platform: &str,
        started_at: i64,
        parent_id: Option<&str>,
        user_id: Option<&str>,
    ) -> Result<()> {
        self.conn
            .execute(
                "INSERT INTO sessions (id, platform, started_at, parent_id, user_id)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params![id, platform, started_at, parent_id, user_id],
            )
            .with_context(|| format!("insert session {id}"))?;
        Ok(())
    }

    /// Insert a turn under an existing session. Returns the autoincrement
    /// `turns.id`. The INSERT trigger fans this out to `turns_fts`.
    pub fn insert_turn(
        &self,
        session_id: &str,
        role: &str,
        content: &str,
        created_at: i64,
    ) -> Result<i64> {
        self.conn
            .execute(
                "INSERT INTO turns (session_id, role, content, created_at)
                 VALUES (?1, ?2, ?3, ?4)",
                rusqlite::params![session_id, role, content, created_at],
            )
            .with_context(|| format!("insert turn into session {session_id}"))?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Delete a session and (via ON DELETE CASCADE + the DELETE trigger)
    /// its turns and their FTS entries. Returns the number of session
    /// rows deleted (0 if the id was unknown).
    pub fn delete_session(&self, id: &str) -> Result<usize> {
        let n = self
            .conn
            .execute("DELETE FROM sessions WHERE id = ?1", rusqlite::params![id])
            .with_context(|| format!("delete session {id}"))?;
        Ok(n)
    }
}

/// Canonical path `~/.halo/state.db`, with a `./.halo/state.db`
/// fallback if no home directory can be resolved (unusual, but matches
/// the rest of the crate).
fn default_state_db_path() -> PathBuf {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    home.join(".halo").join("state.db")
}
