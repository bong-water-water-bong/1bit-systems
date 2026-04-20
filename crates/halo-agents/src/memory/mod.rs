//! `MEMORY.md` + `USER.md` file-backed memory layer.
//!
//! Matches Hermes Agent format byte-for-byte so skill/memory files roundtrip
//! between halo-agents and hermes-agent. See `docs/wiki/Hermes-Integration.md`.
//!
//! * `MEMORY.md` — agent's personal notes. 2200-char cap.
//! * `USER.md`   — user profile (name, role, timezone, prefs). 1375-char cap.
//! * Entries separated by `§` (section-sign, U+00A7) on its own line.
//! * Load-once-per-session semantics. The caller reads at agent start,
//!   writes on `add`/`replace`/`remove`, and does not expect in-session
//!   refresh (matches Hermes' "frozen snapshot" invariant).

pub mod store;

pub use store::{DELIMITER, MAX_MEMORY_CHARS, MAX_USER_CHARS, MemoryKind, MemoryStore};
