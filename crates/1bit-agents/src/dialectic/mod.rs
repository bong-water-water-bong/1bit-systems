//! Dialectic user modeling — Rust reimplementation of [Honcho]'s flagship
//! surface.
//!
//! # What this is
//!
//! Honcho is a "memory library for stateful agents" (MIT, Python + Postgres).
//! Its headline feature is *dialectic user modeling*: per-user belief rows
//! that an LLM agent accumulates from raw conversation and queries back as
//! natural language. Upstream stores these in Postgres with pgvector; we
//! mirror the algorithm in SQLite so 1bit-agents stays zero-dep beyond
//! `rusqlite` (which `sessions` already pulls in bundled).
//!
//! # One-paragraph algorithm summary
//!
//! For every incoming [`Observation`] (a labelled chunk of a peer's
//! utterance inside a session), the dialectic pipeline derives structured
//! [`Inference`]s — claims about the peer — citing the observations that
//! support them. Claims are accumulated across sessions keyed by
//! `(observer_peer, observed_peer)`, so peer A's model of peer B is
//! distinct from peer B's model of themself. On query, we retrieve claims
//! tagged against the observed peer, filter by relevance to the question,
//! and return them in a ranked list. The real Honcho pipeline uses an LLM
//! to distill raw messages into claims ("derivation task") and a second
//! LLM call at query time ("dialectic chat") to synthesize an answer;
//! today's Rust port stubs both: ingestion stores the observation verbatim
//! as a claim, and inference is keyword-overlap retrieval over those
//! claims. The LLM-backed passes are left as `TODO` hooks so routing +
//! persistence can be wired end-to-end while prompt templates land later.
//!
//! [Honcho]: https://github.com/plastic-labs/honcho
//!
//! # Module surface
//!
//! * [`UserModel`] — per-peer aggregate: the set of claims we hold.
//! * [`Observation`] — dialectic input: one chunk of peer-authored text.
//! * [`Inference`]  — dialectic output: a claim + the observations that
//!   support it.
//! * [`DialecticStore`] — trait abstracting persistence. One impl today
//!   ([`SqliteDialecticStore`]).
//! * [`observe`] / [`infer`] — the two top-level functions callers use.
//!
//! See `docs/wiki/Crate-1bit-agents-dialectic.md` for the deviation matrix
//! vs upstream and the schema reasoning.

pub mod derive;
pub mod rank;
pub mod store;

pub use store::{DialecticStore, SqliteDialecticStore};

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Per-peer aggregate view — the set of inferences we currently hold
/// about a single `(observer, observed)` pair.
///
/// This is a *read model*. Writes go through [`observe`]; reads go
/// through [`infer`]. The fields here mirror Honcho's `Peer` ×
/// `Collection` pair (one peer × one `observer → observed` scope)
/// collapsed into one struct because we don't yet need multi-workspace.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct UserModel {
    /// Peer whose point of view this is (who is doing the modeling).
    pub observer_id: String,
    /// Peer being modeled.
    pub observed_id: String,
    /// Claims derived from this peer's observations, newest first.
    pub inferences: Vec<Inference>,
}

/// One labelled chunk of peer-authored text. Equivalent to Honcho's
/// `Message` plus its `session_id` / `peer_name` foreign keys.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Observation {
    /// Session this observation was authored in.
    pub session_id: String,
    /// Peer who produced the text (the *observed* peer).
    pub peer_id: String,
    /// Raw text.
    pub text: String,
    /// Unix seconds. Honcho uses pg `timestamp with time zone`; we keep
    /// it as `i64` to match `sessions::turns.created_at`.
    pub timestamp: i64,
    /// Observation classification. Maps to Honcho's `DocumentLevel`
    /// (explicit vs derived). For now we only emit `Explicit`.
    pub kind: ObservationKind,
}

/// Provenance tier. Honcho's pipeline refines `Explicit` observations
/// into `Derived` claims via an LLM pass; we keep the enum ready but
/// only produce `Explicit` today.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ObservationKind {
    /// Raw peer utterance, stored verbatim.
    Explicit,
    /// LLM-synthesized summary of one or more `Explicit` observations.
    /// Unused today; reserved for the LLM pass.
    Derived,
}

/// Dialectic output — one claim and the observations supporting it.
///
/// `support_observations` lists observation row-ids (as returned by
/// [`DialecticStore::insert_observation`]). The store resolves these
/// back to full [`Observation`] rows on demand, so the `Inference`
/// itself is cheap to pass around.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Inference {
    /// The claim text. In the LLM pipeline this would be a distilled
    /// fact ("user prefers terse CLI output"). Today it's just the
    /// underlying observation's text.
    pub claim: String,
    /// Row ids of the observations that support this claim.
    pub support_observations: Vec<i64>,
    /// Unix seconds when the claim was generated.
    pub created_at: i64,
}

/// Ingest an observation — synchronous, no LLM call.
///
/// Writes the observation and stores one inference whose claim is the
/// raw observation text. This is the Rule A-safe fast path: callers
/// that don't want to depend on halo-server (tests, offline operators,
/// anything with `llm-derive` disabled) get a working pipeline.
///
/// When the `llm-derive` feature is enabled, prefer [`observe_with_llm`]
/// for the distillation pass.
pub fn observe<S: DialecticStore>(store: &S, observer_id: &str, obs: Observation) -> Result<i64> {
    let id = store.insert_observation(&obs)?;
    let inference = Inference {
        claim: obs.text.clone(),
        support_observations: vec![id],
        created_at: obs.timestamp,
    };
    store.insert_inference(observer_id, &obs.peer_id, &inference)?;
    Ok(id)
}

/// LLM-backed ingest. Writes the observation, then calls 1bit-server's
/// `/v1/chat/completions` endpoint to distill 1-3 short claims about
/// the observed peer, and stores each claim as an inference citing the
/// original observation row.
///
/// On any LLM error (HTTP failure, non-JSON body, empty array) falls
/// back to storing the observation verbatim as a single claim — same
/// shape as [`observe`]. Callers never see a hard failure from halo-
/// server being down.
///
/// `base_url` is the full `/v1/chat/completions` URL. Pass
/// [`derive::DEFAULT_CHAT_URL`] for the local halo-server default.
#[cfg(feature = "llm-derive")]
pub async fn observe_with_llm<S: DialecticStore>(
    store: &S,
    observer_id: &str,
    obs: Observation,
    client: &reqwest::Client,
    base_url: &str,
    model: &str,
) -> Result<i64> {
    let id = store.insert_observation(&obs)?;
    let claims =
        derive::derive_claims_with_url(&obs.text, &obs.peer_id, client, base_url, model).await;
    for claim in claims {
        let inference = Inference {
            claim,
            support_observations: vec![id],
            created_at: obs.timestamp,
        };
        store.insert_inference(observer_id, &obs.peer_id, &inference)?;
    }
    Ok(id)
}

/// Answer a natural-language question about the observed peer by
/// returning the inferences most relevant to the question.
///
/// Two-stage ranking: a case-insensitive word-substring gate filters
/// the candidate set (same admission bar the old keyword-overlap pass
/// applied) and [`rank::rank_by_hash_embedding`] re-orders the
/// survivors by 128-dim hashed sparse-embedding cosine. The gate
/// prevents hash-collision false positives (``"verbose"`` bucket-
/// colliding with a ``"terse"`` claim) without giving up the richer
/// frequency-aware ordering the embedding gives us once candidates
/// actually share vocabulary.
///
/// An empty `query` returns every inference, newest first — matches
/// Honcho's "no filter → full representation" behaviour. Ties in the
/// cosine score break by `created_at` descending.
pub fn infer<S: DialecticStore>(
    store: &S,
    observer_id: &str,
    observed_id: &str,
    query: &str,
) -> Result<Vec<Inference>> {
    let all = store.list_inferences(observer_id, observed_id)?;
    if query.trim().is_empty() {
        return Ok(all);
    }
    // Stage 1: word-substring gate. Keep claims that contain at least
    // one lowercased query token as a substring. Identical to the old
    // scoring function's admission rule — regression-safe for the
    // five original tests.
    let q_terms: Vec<String> = query
        .split_whitespace()
        .map(|t| t.to_lowercase())
        .collect();
    let candidates: Vec<Inference> = all
        .into_iter()
        .filter(|inf| {
            let claim_lc = inf.claim.to_lowercase();
            q_terms.iter().any(|t| claim_lc.contains(t.as_str()))
        })
        .collect();
    if candidates.is_empty() {
        return Ok(vec![]);
    }
    // Stage 2: hash-embedding cosine re-rank. Pulls frequency-weighted
    // ordering out of the buckets — two claims that both mention "CLI"
    // still get separated by their density of CLI-related tokens.
    let claims: Vec<String> = candidates.iter().map(|i| i.claim.clone()).collect();
    let ranked = rank::rank_by_hash_embedding(query, &claims);
    let mut filtered: Vec<(f32, Inference)> = ranked
        .into_iter()
        .map(|(score, idx)| (score, candidates[idx].clone()))
        .collect();
    // Ties → newest first. `partial_cmp` on f32 is safe here: `hash_embed`
    // never produces NaN (we guard the divide-by-zero).
    filtered.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.1.created_at.cmp(&a.1.created_at))
    });
    Ok(filtered.into_iter().map(|(_, inf)| inf).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn open_tempstore() -> (tempfile::TempDir, SqliteDialecticStore) {
        let dir = tempdir().expect("create tempdir");
        let path = dir.path().join("dialectic.db");
        let store = SqliteDialecticStore::open(&path).expect("open dialectic store");
        (dir, store)
    }

    fn obs(session: &str, peer: &str, text: &str, ts: i64) -> Observation {
        Observation {
            session_id: session.to_string(),
            peer_id: peer.to_string(),
            text: text.to_string(),
            timestamp: ts,
            kind: ObservationKind::Explicit,
        }
    }

    /// 1) Fresh store: `infer` returns empty, no crash, no pre-seeding
    ///    required. Baseline sanity — proves the DDL bootstrap wired up.
    #[test]
    fn empty_store_infer_returns_empty() {
        let (_dir, store) = open_tempstore();
        let out = infer(&store, "alice", "bob", "anything").unwrap();
        assert!(out.is_empty(), "empty store must yield no inferences");
        // And listing directly should be empty too.
        let direct = store.list_inferences("alice", "bob").unwrap();
        assert!(direct.is_empty());
    }

    /// 2) Two observations that share a keyword → both come back for a
    ///    query hitting that keyword, ranked so the higher-overlap one
    ///    wins. This is the minimum "dialectic query works" contract.
    #[test]
    fn two_observations_same_topic_both_cited() {
        let (_dir, store) = open_tempstore();
        let id1 = observe(
            &store,
            "alice",
            obs("s1", "bob", "bob prefers terse CLI output", 1000),
        )
        .unwrap();
        let id2 = observe(
            &store,
            "alice",
            obs("s1", "bob", "bob dislikes verbose logging in CLI tools", 1001),
        )
        .unwrap();
        // A keyword ("CLI") present in both claims should return both.
        let hits = infer(&store, "alice", "bob", "CLI preferences").unwrap();
        assert_eq!(hits.len(), 2, "want both claims, got {hits:?}");
        // Support vectors point at the original observation rows.
        let supports: Vec<i64> = hits.iter().flat_map(|i| i.support_observations.clone()).collect();
        assert!(supports.contains(&id1));
        assert!(supports.contains(&id2));
        // And an unrelated query returns nothing.
        let miss = infer(&store, "alice", "bob", "rocprof kernels").unwrap();
        assert!(miss.is_empty(), "unrelated query should not match: {miss:?}");
    }

    /// 3) Observations on different observed peers don't leak across:
    ///    alice's model of bob and alice's model of carol stay distinct.
    ///    Same observer, different observeds — regression guard for
    ///    the `(observer, observed)` composite key.
    #[test]
    fn observations_do_not_cross_contaminate_peers() {
        let (_dir, store) = open_tempstore();
        observe(
            &store,
            "alice",
            obs("s1", "bob", "bob likes terse CLI output", 1000),
        )
        .unwrap();
        observe(
            &store,
            "alice",
            obs("s2", "carol", "carol loves verbose debug logs", 1001),
        )
        .unwrap();

        let bob_model = store.list_inferences("alice", "bob").unwrap();
        assert_eq!(bob_model.len(), 1, "bob's model: want 1, got {bob_model:?}");
        assert!(bob_model[0].claim.contains("terse"));

        let carol_model = store.list_inferences("alice", "carol").unwrap();
        assert_eq!(carol_model.len(), 1, "carol's model: want 1, got {carol_model:?}");
        assert!(carol_model[0].claim.contains("verbose"));

        // And queries into each stay scoped.
        let q = infer(&store, "alice", "bob", "verbose").unwrap();
        assert!(
            q.is_empty(),
            "carol's 'verbose' claim leaked into bob's model: {q:?}"
        );
    }

    /// 4) A different observer (e.g. self-model vs another agent's model)
    ///    does not see the first observer's claims. Guards the other
    ///    half of the composite key.
    #[test]
    fn observers_do_not_cross_contaminate() {
        let (_dir, store) = open_tempstore();
        observe(
            &store,
            "alice",
            obs("s1", "bob", "bob likes terse CLI output", 1000),
        )
        .unwrap();
        // Different observer, same observed — should be an independent
        // model.
        let other = store.list_inferences("dave", "bob").unwrap();
        assert!(other.is_empty(), "alice's claims leaked to dave: {other:?}");
    }

    /// 5) Empty query returns every inference, newest first. Matches
    ///    Honcho's "no filter → full representation" default.
    #[test]
    fn empty_query_returns_all_newest_first() {
        let (_dir, store) = open_tempstore();
        observe(&store, "alice", obs("s1", "bob", "first observation", 1000)).unwrap();
        observe(&store, "alice", obs("s1", "bob", "second observation", 2000)).unwrap();
        observe(&store, "alice", obs("s1", "bob", "third observation", 3000)).unwrap();

        let out = infer(&store, "alice", "bob", "").unwrap();
        assert_eq!(out.len(), 3);
        // Newest first.
        assert_eq!(out[0].created_at, 3000);
        assert_eq!(out[1].created_at, 2000);
        assert_eq!(out[2].created_at, 1000);
    }

    /// 6) End-to-end: seed several observations across unrelated topics,
    ///    then query. The hash-embedding ranker must return the topical
    ///    claims ahead of the unrelated ones and drop the zero-score
    ///    rocprof claim entirely.
    #[test]
    fn observe_infer_ranked_by_hash_embedding() {
        let (_dir, store) = open_tempstore();
        observe(
            &store,
            "alice",
            obs("s1", "bob", "bob prefers terse CLI output", 1000),
        )
        .unwrap();
        observe(
            &store,
            "alice",
            obs("s1", "bob", "bob dislikes verbose logging in CLI tools", 1001),
        )
        .unwrap();
        observe(
            &store,
            "alice",
            obs("s1", "bob", "rocprof kernels run hot on gfx1151", 1002),
        )
        .unwrap();

        let hits = infer(&store, "alice", "bob", "CLI preferences").unwrap();
        // The rocprof claim has no hash overlap with "CLI preferences"
        // and should be filtered out.
        assert_eq!(hits.len(), 2, "want 2 CLI-topical hits, got {hits:?}");
        for h in &hits {
            assert!(
                h.claim.to_lowercase().contains("cli"),
                "non-CLI claim leaked into results: {h:?}"
            );
        }
    }
}
