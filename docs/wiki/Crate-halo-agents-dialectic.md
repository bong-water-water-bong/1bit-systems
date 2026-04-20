# Crate: 1bit-agents / dialectic

**Phase:** solutioning  
**Status:** scaffold + LLM derivation + embedding-ish ranker landed 2026-04-20  
**Upstream reference:** [plastic-labs/honcho][honcho] (MIT, Python + Postgres)

Rust reimplementation of Honcho's dialectic user-modeling surface as a
module in `1bit-agents`. No Python FFI, no Postgres — SQLite via the
`rusqlite` (`bundled`) dep we already pull in for `sessions`.

[honcho]: https://github.com/plastic-labs/honcho

## What is dialectic user modeling

For every incoming **Observation** (a labelled chunk of a peer's utterance
inside a session), a dialectic pipeline derives structured **Inferences** —
claims about the peer — citing the observations that support them. Claims
are accumulated across sessions keyed by `(observer_peer, observed_peer)`,
so peer A's model of peer B is distinct from peer B's self-model. On
query, claims tagged against the observed peer are filtered by relevance
and returned ranked. Honcho's production pipeline uses an LLM to distill
raw messages into claims ("derivation task") and a second LLM call at
query time ("dialectic chat") to synthesize an answer over the top; the
Rust port stubs both today (ingestion stores the observation verbatim as
a claim, query is case-insensitive keyword overlap) and leaves the LLM
call as a clearly marked `TODO`.

## Rust surface

```
crates/1bit-agents/src/dialectic/
    mod.rs        — public types + observe() / infer() functions + tests
    store.rs      — DialecticStore trait + SqliteDialecticStore impl + DDL
    derive.rs     — LLM-backed claim distillation (feature: llm-derive)
    rank.rs       — 128-dim hash embedding + cosine ranker
```

Re-exports live at `onebit_agents::{UserModel, Observation, Inference,
ObservationKind, DialecticStore, SqliteDialecticStore, observe, infer}`.

## SQL schema (SQLite, 1bit-agents/dialectic)

```sql
CREATE TABLE dialectic_observations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    peer_id     TEXT NOT NULL,
    text        TEXT NOT NULL,
    timestamp   INTEGER NOT NULL,
    kind        TEXT NOT NULL           -- 'explicit' | 'derived'
);

CREATE TABLE dialectic_inferences (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    observer_id  TEXT NOT NULL,
    observed_id  TEXT NOT NULL,
    claim        TEXT NOT NULL,
    created_at   INTEGER NOT NULL
);
CREATE INDEX idx_dinf_pair
    ON dialectic_inferences(observer_id, observed_id, created_at DESC);

CREATE TABLE dialectic_supports (
    inference_id   INTEGER NOT NULL REFERENCES dialectic_inferences(id) ON DELETE CASCADE,
    observation_id INTEGER NOT NULL REFERENCES dialectic_observations(id) ON DELETE CASCADE,
    PRIMARY KEY (inference_id, observation_id)
);
```

DB lives at `~/.halo/dialectic.db`. Separate file from `~/.halo/state.db`
so FTS5 session reads don't contend with dialectic writes. Same-file
colocation is safe — object names are prefixed `dialectic_`.

## Deviation from Honcho upstream

| Honcho upstream                              | 1bit-agents/dialectic                          | Reason                                                                 |
| -------------------------------------------- | ---------------------------------------------- | ---------------------------------------------------------------------- |
| Postgres + pgvector + HNSW                   | SQLite (rusqlite bundled)                      | Zero-dep beyond what `sessions` already pulls; Rule A (no Python) holds. |
| `Workspace` multi-tenant                     | Single-tenant (dropped)                        | One operator, one box; revisit if 1bit-server ever hosts 3rd parties.  |
| `Peer` row with config/metadata              | Peer referenced by string id                   | Peer registry is `1bit-agents::sessions::sessions.user_id`; no need to dup. |
| `Messages` at session scope                  | `dialectic_observations`                       | Separation lets FTS5 session search keep its own row layout.           |
| LLM "derivation task" distills claims        | POST /v1/chat/completions (feature: llm-derive) with verbatim fallback | Landed 2026-04-20. Prompt template below; feature off by default so tests don't need a live halo-server. |
| LLM "dialectic chat" synthesizes answers     | Two-stage: word-substring gate → hash-embedding cosine rerank | Landed 2026-04-20. Upgrade path still open once 1bit-server exposes /v1/embeddings. |
| pgvector cosine similarity over embeddings   | 128-dim FNV-1a hash embeddings + L2 + cosine (with /v1/embeddings probe) | Hash ranker is always on. Probe switches to remote embeddings automatically when halo-server starts serving them. |
| `Document.level` (explicit / derived / …)    | `ObservationKind::{Explicit, Derived}`         | Enum present, only `Explicit` produced today. `Derived` reserved for LLM pass. |
| Async task queue (`queue`, `dream`, reconciler) | — (not in scaffold)                         | Derivation is synchronous today. Revisit when LLM pass lands.          |
| Webhooks                                     | —                                              | Not needed for local single-operator deploy.                           |
| Collections of vector-embedded documents (RAG) | —                                            | `1bit-agents::memory` + `sessions` FTS5 cover our RAG story today.     |

## Top-level API (Rust)

```rust
use onebit_agents::{observe, infer, Observation, ObservationKind, SqliteDialecticStore};

let store = SqliteDialecticStore::open_default()?;       // ~/.halo/dialectic.db
let obs = Observation {
    session_id: "cli-2026-04-20".into(),
    peer_id:    "bob".into(),
    text:       "bob prefers terse CLI output".into(),
    timestamp:  now_unix(),
    kind:       ObservationKind::Explicit,
};
observe(&store, "alice", obs)?;                          // writes + derives
let claims = infer(&store, "alice", "bob", "CLI style")?; // ranked recall
```

## Endpoints we'd need to mirror (future; not wired today)

Reference list for when we expose this to 1bit-mcp / 1bit-server:

- `POST /dialectic/observe` → `observe()`
- `GET  /dialectic/infer?observer=&observed=&q=` → `infer()`
- `GET  /dialectic/user_model/{observer}/{observed}` → `list_inferences()`
- `GET  /dialectic/observation/{id}` → `get_observation()`

Route shape mirrors Honcho's `POST /peers/{peer_id}/chat` + `GET
/peers/.../representation` but scoped at the `(observer, observed)`
pair level, which maps more cleanly to our multi-agent specialist mesh
than Honcho's implicit "assistant = current user" assumption.

## Derivation prompt

The single source of truth lives in
`crates/1bit-agents/src/dialectic/derive.rs::derivation_prompt`. Rendered:

```
You are a careful observer. Extract 1-3 short claims about {observed_peer}
from the following utterance. Return ONLY a JSON array of strings, no prose.

Utterance: {observation}
```

Transport: `POST http://127.0.0.1:8180/v1/chat/completions`, model
`bitnet-b1.58-2B-4T`, `temperature: 0.2`, `max_tokens: 256`,
`stream: false`. Parses `choices[0].message.content` as a JSON array;
also strips ```` ```json ... ``` ```` fences. Any error (HTTP, parse,
empty array) → falls back to `[observation_verbatim]`.

## Tests

Sixteen in-crate unit tests under `dialectic::tests` +
`dialectic::derive::tests` + `dialectic::rank::tests`:

**Module (`dialectic::tests`) — 6 tests:**

1. `empty_store_infer_returns_empty` — fresh store, no panic.
2. `two_observations_same_topic_both_cited` — ranking + citation.
3. `observations_do_not_cross_contaminate_peers` — observed-peer scoping.
4. `observers_do_not_cross_contaminate` — observer-peer scoping.
5. `empty_query_returns_all_newest_first` — default full-representation.
6. `observe_infer_ranked_by_hash_embedding` — end-to-end with new ranker.

**Derive (`dialectic::derive::tests`) — 6 tests:**

7. `parse_claims_accepts_valid_json_array` — pure-parse happy path.
8. `parse_claims_falls_back_on_prose` — pure-parse fallback.
9. `parse_claims_strips_code_fence` — tolerates ```` ```json ... ``` ```` wraps.
10. `derive_claims_against_mock_parses_json_array` — full reqwest → TCP mock.
11. `derive_claims_against_mock_falls_back_on_prose` — full reqwest, prose body.
12. `derive_claims_http_error_falls_back` — connection refused → fallback.

**Rank (`dialectic::rank::tests`) — 4 tests:**

13. `hash_embed_self_similarity_is_one` — L2 normalization sanity.
14. `hash_embed_overlap_beats_disjoint` — shared-vocab > disjoint-vocab.
15. `rank_orders_by_descending_similarity` — ordering contract.
16. `rank_empty_query_returns_all_without_panic` — zero-vector safety.

Total crate tests after port: **78 unit + 1 integration** (was 67 + 1).

## Open work (in order)

1. ~~**LLM derivation task.**~~ **Done 2026-04-20.** `derive_claims` +
   `observe_with_llm` behind `llm-derive` feature flag. Default off so
   offline / CI paths stay hermetic.
2. ~~**Embedding-based ranker.**~~ **Done 2026-04-20.** 128-dim FNV-1a
   hash embedding + L2 + cosine with two-stage gate against hash
   collisions. Probe for `/v1/embeddings` lives behind
   `rank::probe_embeddings_endpoint` so the switch to real embeddings
   is a one-call upgrade once 1bit-core ships one.
3. **1bit-mcp surface.** Expose the four endpoints above through the MCP
   bridge.
4. **1bit-server HTTP surface.** Route to `/v1/dialectic/*` behind the
   same bearer guard the rest of 1bit-server uses.
5. **Remote embeddings upgrade.** When halo-server starts serving
   `/v1/embeddings`, swap `rank_by_hash_embedding` for a remote call
   with the hash embedding as its fallback. Probe + cache already in
   place.
