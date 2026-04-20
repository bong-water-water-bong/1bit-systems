//! `DialecticStore` — persistence layer for the dialectic module.
//!
//! Shape mirrors Honcho's core tables (`peers`, `sessions`, `messages`,
//! `documents`) but flattened for SQLite:
//!
//! | Honcho (Postgres)          | 1bit-agents (SQLite)       |
//! | -------------------------- | -------------------------- |
//! | `peers(name, workspace)`   | `dialectic_peers(id)`      |
//! | `sessions(name)`           | `dialectic_sessions(id)`   |
//! | `messages`                 | `dialectic_observations`   |
//! | `documents(observer, observed, content, embedding)` | `dialectic_inferences(observer, observed, claim)` |
//! | `message_embeddings` (HNSW) | — (deferred, Rule A)        |
//!
//! We deliberately skip `workspace_name` (single-tenant today) and
//! `pgvector` / `hnsw` (we don't ship embeddings yet; keyword ranking
//! is fine until the LLM derivation pass lands).
//!
//! # DB file
//!
//! Default path is `~/.halo/dialectic.db` — a separate file from
//! `~/.halo/state.db` so session-history reads don't contend with
//! dialectic writes. Callers that want to colocate can reuse the same
//! path; schema objects are namespaced with a `dialectic_` prefix so
//! `SessionDb::open(path)` + `SqliteDialecticStore::open(path)` against
//! the same file don't collide.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rusqlite::{Connection, OptionalExtension, params};

use super::{Inference, Observation, ObservationKind};

/// Abstract persistence for the dialectic module. One impl today
/// ([`SqliteDialecticStore`]); kept as a trait so tests + future
/// backends (e.g. an in-memory `HashMap` fake) stay easy.
pub trait DialecticStore {
    /// Insert an observation. Returns the autoincrement row id —
    /// callers cite this id from [`Inference::support_observations`].
    fn insert_observation(&self, obs: &Observation) -> Result<i64>;

    /// Insert an inference tagged against `(observer, observed)`.
    fn insert_inference(
        &self,
        observer_id: &str,
        observed_id: &str,
        inference: &Inference,
    ) -> Result<i64>;

    /// List all inferences for a given `(observer, observed)` pair,
    /// ordered by `created_at` descending (newest first).
    fn list_inferences(&self, observer_id: &str, observed_id: &str) -> Result<Vec<Inference>>;

    /// Look up a single observation by id. Returns `None` if the id
    /// is unknown. Used by callers that want to expand an inference's
    /// `support_observations` back to full rows.
    fn get_observation(&self, id: i64) -> Result<Option<Observation>>;
}

/// DDL for the dialectic SQLite store. Idempotent — safe to run on
/// every open.
///
/// Schema notes:
///
/// * `dialectic_observations` — one row per ingested peer utterance.
///   `kind` stores the `ObservationKind` enum as text.
/// * `dialectic_inferences` — one row per derived claim, keyed by
///   composite `(observer_id, observed_id)`. No foreign keys to the
///   observations table because the LLM pipeline may cite raw text
///   that no longer has a 1:1 observation row (e.g. a summarized
///   rollup). `support_observation_ids` is a JSON array of row ids.
/// * `dialectic_supports` — bridge table materializing the JSON
///   array for query convenience. Two writes per inference, but we
///   gain `WHERE observation_id = ?` without JSON parsing.
const SCHEMA_SQL: &str = r#"
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS dialectic_observations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    peer_id     TEXT NOT NULL,
    text        TEXT NOT NULL,
    timestamp   INTEGER NOT NULL,
    kind        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_dobs_peer ON dialectic_observations(peer_id);
CREATE INDEX IF NOT EXISTS idx_dobs_session ON dialectic_observations(session_id);

CREATE TABLE IF NOT EXISTS dialectic_inferences (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    observer_id  TEXT NOT NULL,
    observed_id  TEXT NOT NULL,
    claim        TEXT NOT NULL,
    created_at   INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_dinf_pair
    ON dialectic_inferences(observer_id, observed_id, created_at DESC);

CREATE TABLE IF NOT EXISTS dialectic_supports (
    inference_id   INTEGER NOT NULL REFERENCES dialectic_inferences(id) ON DELETE CASCADE,
    observation_id INTEGER NOT NULL REFERENCES dialectic_observations(id) ON DELETE CASCADE,
    PRIMARY KEY (inference_id, observation_id)
);

CREATE INDEX IF NOT EXISTS idx_dsup_obs ON dialectic_supports(observation_id);
"#;

/// SQLite-backed [`DialecticStore`].
pub struct SqliteDialecticStore {
    conn: Connection,
}

impl SqliteDialecticStore {
    /// Open (creating if absent) a dialectic database at the given path.
    /// Used by tests with a tempdir path, and by [`Self::open_default`]
    /// with `~/.halo/dialectic.db`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create parent dir for {}", path.display()))?;
        }
        let conn = Connection::open(path)
            .with_context(|| format!("open sqlite at {}", path.display()))?;
        conn.execute_batch(SCHEMA_SQL)
            .context("run dialectic-db schema bootstrap")?;
        Ok(Self { conn })
    }

    /// Open `~/.halo/dialectic.db`. Falls back to `./.halo/dialectic.db`
    /// if no home directory is resolvable (matches `SessionDb::open_default`).
    pub fn open_default() -> Result<Self> {
        Self::open(default_path())
    }

    /// Borrow the underlying connection — handy for test assertions
    /// and for the (future) 1bit-mcp bridge that needs joined queries.
    pub fn conn(&self) -> &Connection {
        &self.conn
    }
}

impl DialecticStore for SqliteDialecticStore {
    fn insert_observation(&self, obs: &Observation) -> Result<i64> {
        let kind = match obs.kind {
            ObservationKind::Explicit => "explicit",
            ObservationKind::Derived => "derived",
        };
        self.conn
            .execute(
                "INSERT INTO dialectic_observations
                   (session_id, peer_id, text, timestamp, kind)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![obs.session_id, obs.peer_id, obs.text, obs.timestamp, kind],
            )
            .context("insert dialectic observation")?;
        Ok(self.conn.last_insert_rowid())
    }

    fn insert_inference(
        &self,
        observer_id: &str,
        observed_id: &str,
        inference: &Inference,
    ) -> Result<i64> {
        self.conn
            .execute(
                "INSERT INTO dialectic_inferences
                   (observer_id, observed_id, claim, created_at)
                 VALUES (?1, ?2, ?3, ?4)",
                params![observer_id, observed_id, inference.claim, inference.created_at],
            )
            .context("insert dialectic inference")?;
        let inf_id = self.conn.last_insert_rowid();
        // Write the supports bridge rows. INSERT OR IGNORE keeps the
        // composite PK honest if a caller ever re-cites the same obs.
        let mut stmt = self
            .conn
            .prepare_cached(
                "INSERT OR IGNORE INTO dialectic_supports (inference_id, observation_id)
                 VALUES (?1, ?2)",
            )
            .context("prepare dialectic_supports insert")?;
        for obs_id in &inference.support_observations {
            stmt.execute(params![inf_id, obs_id])
                .context("insert dialectic_supports row")?;
        }
        Ok(inf_id)
    }

    fn list_inferences(&self, observer_id: &str, observed_id: &str) -> Result<Vec<Inference>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT id, claim, created_at
                   FROM dialectic_inferences
                  WHERE observer_id = ?1 AND observed_id = ?2
                  ORDER BY created_at DESC, id DESC",
            )
            .context("prepare list_inferences")?;
        let rows: Vec<(i64, String, i64)> = stmt
            .query_map(params![observer_id, observed_id], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?))
            })
            .context("query list_inferences")?
            .collect::<rusqlite::Result<Vec<_>>>()
            .context("collect list_inferences")?;

        let mut out = Vec::with_capacity(rows.len());
        let mut supports_stmt = self
            .conn
            .prepare_cached(
                "SELECT observation_id FROM dialectic_supports WHERE inference_id = ?1",
            )
            .context("prepare supports lookup")?;
        for (id, claim, created_at) in rows {
            let supports: Vec<i64> = supports_stmt
                .query_map(params![id], |r| r.get::<_, i64>(0))
                .context("query supports")?
                .collect::<rusqlite::Result<Vec<_>>>()
                .context("collect supports")?;
            out.push(Inference {
                claim,
                support_observations: supports,
                created_at,
            });
        }
        Ok(out)
    }

    fn get_observation(&self, id: i64) -> Result<Option<Observation>> {
        let row = self
            .conn
            .query_row(
                "SELECT session_id, peer_id, text, timestamp, kind
                   FROM dialectic_observations WHERE id = ?1",
                params![id],
                |r| {
                    let kind_s: String = r.get(4)?;
                    let kind = match kind_s.as_str() {
                        "derived" => ObservationKind::Derived,
                        _ => ObservationKind::Explicit,
                    };
                    Ok(Observation {
                        session_id: r.get(0)?,
                        peer_id: r.get(1)?,
                        text: r.get(2)?,
                        timestamp: r.get(3)?,
                        kind,
                    })
                },
            )
            .optional()
            .context("query get_observation")?;
        Ok(row)
    }
}

fn default_path() -> PathBuf {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    home.join(".halo").join("dialectic.db")
}
