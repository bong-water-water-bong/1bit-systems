//! DDL for the `~/.halo/state.db` session history store.
//!
//! Shape matches Hermes' `~/.hermes/state.db` verbatim so a future import
//! tool can shovel rows across without column-name translation:
//!
//! * `sessions`   — one row per CLI / messaging conversation.
//! * `turns`      — one row per user / assistant / tool message.
//! * `turns_fts`  — FTS5 contentless external-content mirror of `turns.content`,
//!   kept in sync by three triggers (INSERT / UPDATE / DELETE).
//!
//! We use `content=turns, content_rowid=id` so FTS5 stores only the
//! tokenized index (no duplicate content column) and joins back against
//! `turns` by rowid. Triggers must insert into the "writable pseudo-table"
//! `turns_fts(turns_fts, rowid, content)` with the `'delete'` command to
//! keep the index consistent on UPDATE / DELETE — the standard FTS5
//! external-content dance.
//!
//! Note on cascade semantics: SQLite does NOT honor `ON DELETE CASCADE`
//! unless `PRAGMA foreign_keys = ON` is set for the connection; [`open`]
//! does that once at connection time. We define the FK explicitly so
//! deleting a row from `sessions` also deletes its `turns` rows, and the
//! DELETE trigger then cleans `turns_fts` behind them.

/// Pragmas + schema executed on every [`crate::sessions::db::SessionDb::open`] call.
/// `CREATE ... IF NOT EXISTS` + `CREATE TRIGGER IF NOT EXISTS` keep this
/// idempotent across reopens.
pub const SCHEMA_SQL: &str = r#"
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    platform    TEXT NOT NULL,
    started_at  INTEGER NOT NULL,
    ended_at    INTEGER,
    parent_id   TEXT REFERENCES sessions(id),
    user_id     TEXT
);

CREATE TABLE IF NOT EXISTS turns (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);

CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(
    content,
    session_id UNINDEXED,
    role       UNINDEXED,
    content=turns,
    content_rowid=id
);

-- Keep turns_fts in sync with turns. External-content FTS5 requires
-- explicit triggers; the 'delete' command flushes the old tokens from
-- the index before UPDATE re-inserts the new ones.

CREATE TRIGGER IF NOT EXISTS turns_ai AFTER INSERT ON turns BEGIN
    INSERT INTO turns_fts(rowid, content, session_id, role)
        VALUES (new.id, new.content, new.session_id, new.role);
END;

CREATE TRIGGER IF NOT EXISTS turns_ad AFTER DELETE ON turns BEGIN
    INSERT INTO turns_fts(turns_fts, rowid, content, session_id, role)
        VALUES ('delete', old.id, old.content, old.session_id, old.role);
END;

CREATE TRIGGER IF NOT EXISTS turns_au AFTER UPDATE ON turns BEGIN
    INSERT INTO turns_fts(turns_fts, rowid, content, session_id, role)
        VALUES ('delete', old.id, old.content, old.session_id, old.role);
    INSERT INTO turns_fts(rowid, content, session_id, role)
        VALUES (new.id, new.content, new.session_id, new.role);
END;
"#;
