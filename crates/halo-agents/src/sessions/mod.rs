//! halo-agents session-history + FTS5 search layer.
//!
//! Mirrors [Hermes Agent]'s `~/.hermes/state.db` shape — SQLite with a
//! `sessions` + `turns` split, plus an FTS5 index over turn content —
//! at `~/.halo/state.db`. See `docs/wiki/Hermes-Integration.md` for the
//! wider interop rationale.
//!
//! [Hermes Agent]: https://github.com/NousResearch/hermes-agent
//!
//! # Surface
//!
//! * [`SessionDb`] — connection wrapper. [`SessionDb::open_default`]
//!   opens `~/.halo/state.db`; [`SessionDb::open`] takes any path
//!   (used by tests against a tempdir).
//! * [`Hit`] — one row returned by [`SessionDb::search`], with a 32-
//!   token BM25 snippet.
//! * [`schema::SCHEMA_SQL`] — the DDL + triggers. Re-exported here for
//!   callers that need to run the bootstrap against a foreign handle
//!   (e.g. a migration harness).
//!
//! # Quick start
//!
//! ```no_run
//! use halo_agents::SessionDb;
//!
//! let db = SessionDb::open_default()?;
//! db.insert_session("abc", "cli", 1_700_000_000, None, None)?;
//! db.insert_turn("abc", "user", "how fast is the kernel?", 1_700_000_001)?;
//! let hits = db.search("kernel", 10)?;
//! assert_eq!(hits.len(), 1);
//! # Ok::<_, anyhow::Error>(())
//! ```

pub mod db;
pub mod schema;
pub mod search;

pub use db::SessionDb;
pub use search::Hit;

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn open_tempdb() -> (tempfile::TempDir, SessionDb) {
        let dir = tempdir().expect("create tempdir");
        let path = dir.path().join("state.db");
        let db = SessionDb::open(&path).expect("open session db");
        (dir, db)
    }

    /// 1) Open a fresh db against a tempdir path and confirm the schema
    /// bootstrap actually created `sessions`, `turns`, and `turns_fts`.
    /// Any missing object = the bootstrap regressed.
    #[test]
    fn open_bootstraps_schema() {
        let (_dir, db) = open_tempdb();
        let mut stmt = db
            .conn()
            .prepare("SELECT name FROM sqlite_master WHERE type IN ('table','view') ORDER BY name")
            .unwrap();
        let names: Vec<String> = stmt
            .query_map([], |row| row.get::<_, String>(0))
            .unwrap()
            .map(|r| r.unwrap())
            .collect();
        for needed in ["sessions", "turns", "turns_fts"] {
            assert!(
                names.iter().any(|n| n == needed),
                "missing table {needed} in {names:?}"
            );
        }
    }

    /// 2) Insert a session with three turns and confirm row counts by
    /// direct SELECT — independent of the FTS5 layer.
    #[test]
    fn insert_session_and_turns_roundtrip() {
        let (_dir, db) = open_tempdb();
        db.insert_session("s1", "cli", 1_700_000_000, None, Some("bcloud"))
            .unwrap();
        db.insert_turn("s1", "user", "what's the bench at?", 1_700_000_001).unwrap();
        db.insert_turn("s1", "assistant", "66 tok/s @ 64 tok", 1_700_000_002).unwrap();
        db.insert_turn("s1", "tool", "rocprof log line", 1_700_000_003).unwrap();

        let sessions: i64 = db
            .conn()
            .query_row("SELECT COUNT(*) FROM sessions", [], |r| r.get(0))
            .unwrap();
        assert_eq!(sessions, 1);

        let turns: i64 = db
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM turns WHERE session_id = 's1'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(turns, 3);
    }

    /// 3) FTS5 returns the expected turn for a single-term query, with
    /// the snippet brackets wrapping the match.
    #[test]
    fn fts5_search_finds_exact_word() {
        let (_dir, db) = open_tempdb();
        db.insert_session("s1", "cli", 1_700_000_000, None, None).unwrap();
        let t1 = db
            .insert_turn("s1", "user", "how fast is the kernel?", 1_700_000_001)
            .unwrap();
        db.insert_turn("s1", "assistant", "the cache is warm", 1_700_000_002)
            .unwrap();

        let hits = db.search("kernel", 10).unwrap();
        assert_eq!(hits.len(), 1, "want 1 hit, got {hits:?}");
        let h = &hits[0];
        assert_eq!(h.session_id, "s1");
        assert_eq!(h.turn_id, t1);
        assert_eq!(h.role, "user");
        assert!(
            h.snippet.contains("[kernel]"),
            "snippet should bracket match, got {:?}",
            h.snippet
        );
    }

    /// 4) Multi-word query: confirm BM25 ordering. The turn that matches
    /// *both* terms has to outrank the turn that matches only one, and
    /// our `Hit::rank` is flipped so larger = more relevant.
    #[test]
    fn fts5_search_orders_hits_by_rank() {
        let (_dir, db) = open_tempdb();
        db.insert_session("s1", "cli", 1_700_000_000, None, None).unwrap();
        // Both terms → stronger match.
        let both = db
            .insert_turn(
                "s1",
                "user",
                "rocprof on the ternary kernel shows 92 percent peak",
                1_700_000_001,
            )
            .unwrap();
        // One term only.
        let one = db
            .insert_turn("s1", "assistant", "kernel started cleanly", 1_700_000_002)
            .unwrap();
        // No terms.
        db.insert_turn("s1", "tool", "completely unrelated log", 1_700_000_003)
            .unwrap();

        let hits = db.search("rocprof AND kernel", 10).unwrap();
        assert_eq!(hits.len(), 1, "AND: want 1 hit, got {hits:?}");
        assert_eq!(hits[0].turn_id, both);

        let hits = db.search("rocprof OR kernel", 10).unwrap();
        assert_eq!(hits.len(), 2, "OR: want 2 hits, got {hits:?}");
        assert_eq!(hits[0].turn_id, both, "top hit should match both terms");
        assert_eq!(hits[1].turn_id, one);
        assert!(
            hits[0].rank >= hits[1].rank,
            "rank should be descending, got {} then {}",
            hits[0].rank,
            hits[1].rank,
        );
    }

    /// 5) Deleting a session cascades into `turns` (via the FK) and the
    /// DELETE trigger scrubs `turns_fts`. No orphan FTS hits allowed.
    #[test]
    fn delete_session_cascades_into_fts() {
        let (_dir, db) = open_tempdb();
        db.insert_session("s1", "cli", 1_700_000_000, None, None).unwrap();
        db.insert_turn("s1", "user", "orphanprobe token alpha", 1_700_000_001)
            .unwrap();
        db.insert_turn("s1", "assistant", "orphanprobe token beta", 1_700_000_002)
            .unwrap();
        db.insert_session("s2", "cli", 1_700_000_000, None, None).unwrap();
        db.insert_turn("s2", "user", "keeper token gamma", 1_700_000_003)
            .unwrap();

        // Baseline: 3 turns, 3 FTS rows.
        let pre: i64 = db
            .conn()
            .query_row("SELECT COUNT(*) FROM turns_fts", [], |r| r.get(0))
            .unwrap();
        assert_eq!(pre, 3);

        let n = db.delete_session("s1").unwrap();
        assert_eq!(n, 1, "should delete exactly one session row");

        let turns_left: i64 = db
            .conn()
            .query_row("SELECT COUNT(*) FROM turns", [], |r| r.get(0))
            .unwrap();
        assert_eq!(turns_left, 1, "FK cascade should leave only s2's turn");

        let fts_left: i64 = db
            .conn()
            .query_row("SELECT COUNT(*) FROM turns_fts", [], |r| r.get(0))
            .unwrap();
        assert_eq!(fts_left, 1, "DELETE trigger should scrub FTS rows");

        // And a search for the orphan tokens should now return nothing.
        let hits = db.search("orphanprobe", 10).unwrap();
        assert!(hits.is_empty(), "orphan FTS rows lingered: {hits:?}");

        // Keeper is still findable.
        let hits = db.search("keeper", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].session_id, "s2");
    }
}
