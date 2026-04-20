//! FTS5 full-text search over the `turns` table.
//!
//! Uses SQLite's `snippet()` helper against the `turns_fts` external-
//! content table to return a context window around the match. Sort is
//! FTS5's built-in `bm25`-esque `rank` column (ascending = most relevant
//! first), and we expose it as a negated `f64` in [`Hit::rank`] so API
//! consumers can sort by `rank DESC` if they do their own re-ranking
//! (larger = better, matching the convention the Hermes TUI uses).

use anyhow::{Context, Result};
use rusqlite::params;

use super::db::SessionDb;

/// A single FTS5 hit. One row per matched `turns` row.
///
/// `rank` is the FTS5 BM25 score, negated. FTS5 emits it as a raw
/// negative-or-zero number where "more negative = more relevant"; we
/// flip the sign at read time so callers can reason about it as a
/// standard ascending-or-descending score without tripping over
/// SQLite's convention.
#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    pub session_id: String,
    pub turn_id: i64,
    pub role: String,
    pub snippet: String,
    pub rank: f64,
}

impl SessionDb {
    /// Full-text search over turn content. `query` is FTS5 MATCH syntax
    /// (e.g. `kernel` or `kernel AND rocprof`). Returns up to `limit`
    /// hits, ordered by relevance (most relevant first).
    ///
    /// We build a 32-token-window snippet: `snippet(turns_fts, 0, '[',
    /// ']', '…', 32)`. Column `0` is `content` — the only indexed
    /// column on `turns_fts`. `[`/`]` bracket the matched terms, `…`
    /// marks truncation.
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<Hit>> {
        let sql = "
            SELECT
                turns_fts.session_id,
                turns_fts.rowid,
                turns_fts.role,
                snippet(turns_fts, 0, '[', ']', '…', 32),
                rank
            FROM turns_fts
            WHERE turns_fts MATCH ?1
            ORDER BY rank
            LIMIT ?2
        ";

        let mut stmt = self
            .conn()
            .prepare(sql)
            .context("prepare FTS5 search statement")?;

        // SQLite/FTS5 rank is returned as a REAL. It's already signed;
        // we flip it so a larger `Hit::rank` = more relevant.
        let rows = stmt
            .query_map(params![query, limit as i64], |row| {
                Ok(Hit {
                    session_id: row.get(0)?,
                    turn_id: row.get(1)?,
                    role: row.get(2)?,
                    snippet: row.get(3)?,
                    rank: -row.get::<_, f64>(4)?,
                })
            })
            .with_context(|| format!("run FTS5 search for {query:?}"))?;

        let mut hits = Vec::new();
        for r in rows {
            hits.push(r.context("read FTS5 hit row")?);
        }
        Ok(hits)
    }
}
