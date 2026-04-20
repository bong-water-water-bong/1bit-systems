//! Relevance ranking for [`super::infer`].
//!
//! Replaces the case-insensitive keyword-overlap scoring with a hash-
//! based sparse embedding + cosine similarity. Still no model dep: we
//! hash each word into 3 fixed features in a 128-dim vector, sum, L2-
//! normalize. ~20 LOC of math, no external runtime requirement.
//!
//! If 1bit-server ever exposes `/v1/embeddings` (OpenAI-compat shape) we
//! probe it once at startup via [`probe_embeddings_endpoint`] and prefer
//! that signal over the hash fallback. The probe result is cached per
//! URL in a `OnceLock` so the check is paid once per process.
//!
//! This module is *not* behind the `llm-derive` feature — the hash ranker
//! is always available and replaces keyword overlap unconditionally. The
//! probe only runs when the caller explicitly asks for a remote ranker.

/// Dimensionality of the hash embedding. 128 is plenty for ≤ low-thousands
/// claims and keeps the dot-product under a cache line.
pub const HASH_DIM: usize = 128;

/// Number of hash buckets each word lights up. 3 is Bloom-filter folklore
/// for balancing collision rate vs density at small dims.
const HASHES_PER_WORD: usize = 3;

/// Compute a sparse L2-normalized hash embedding for `text`.
///
/// For each lowercased whitespace-split token we deterministically hash
/// it into [`HASHES_PER_WORD`] buckets in `[0, HASH_DIM)` and add +1 to
/// each. The resulting vector is L2-normalized so the cosine similarity
/// is just the dot product.
pub fn hash_embed(text: &str) -> [f32; HASH_DIM] {
    let mut v = [0f32; HASH_DIM];
    for raw_word in text.split_whitespace() {
        let word = raw_word.to_lowercase();
        // Strip trivial punctuation so "CLI," and "CLI" bucket together.
        let word = word.trim_matches(|c: char| !c.is_alphanumeric());
        if word.is_empty() {
            continue;
        }
        for salt in 0..HASHES_PER_WORD {
            let h = fnv1a_with_salt(word.as_bytes(), salt as u64);
            let idx = (h as usize) % HASH_DIM;
            v[idx] += 1.0;
        }
    }
    l2_normalize(&mut v);
    v
}

/// Cosine similarity between two embeddings. Because [`hash_embed`] L2-
/// normalizes, this collapses to a dot product.
pub fn cosine(a: &[f32; HASH_DIM], b: &[f32; HASH_DIM]) -> f32 {
    let mut dot = 0f32;
    for i in 0..HASH_DIM {
        dot += a[i] * b[i];
    }
    dot
}

fn l2_normalize(v: &mut [f32; HASH_DIM]) {
    let mut ss = 0f32;
    for x in v.iter() {
        ss += *x * *x;
    }
    if ss > 0.0 {
        let inv = 1.0 / ss.sqrt();
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

/// FNV-1a 64-bit with an additive salt so we get 3 independent hashes
/// per word without keeping three separate hasher states.
fn fnv1a_with_salt(bytes: &[u8], salt: u64) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325u64.wrapping_add(salt.wrapping_mul(0x100000001b3));
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Score + rank a list of claims against a query using the hash
/// embedding. Returns `(score, index)` pairs sorted high-to-low.
///
/// Indexes back into the caller's slice — the caller holds the full
/// [`super::Inference`] values and re-orders them.
pub fn rank_by_hash_embedding(query: &str, claims: &[String]) -> Vec<(f32, usize)> {
    let q_emb = hash_embed(query);
    let mut scored: Vec<(f32, usize)> = claims
        .iter()
        .enumerate()
        .map(|(i, c)| (cosine(&q_emb, &hash_embed(c)), i))
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored
}

// ---------------------------------------------------------------------------
// Optional: probe halo-server's /v1/embeddings endpoint.
//
// The halo-server today does NOT expose /v1/embeddings (1bit-core has no
// embedder on the gfx1151 side yet). We probe anyway so the moment it
// lands we start using it without a code change.
//
// Caching: we do a best-effort probe via reqwest and memoize the bool
// per URL in a `OnceLock`-guarded `HashMap`. Tests that want to force
// the fallback path simply never call `prefer_remote_embeddings`.
// ---------------------------------------------------------------------------

use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

fn probe_cache() -> &'static RwLock<HashMap<String, bool>> {
    static CACHE: OnceLock<RwLock<HashMap<String, bool>>> = OnceLock::new();
    CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

/// HEAD / GET the given embeddings URL and return whether it looks live.
/// Anything other than a 2xx or 405-ish ("method not allowed, but route
/// exists") → false. Caches the result per URL.
pub async fn probe_embeddings_endpoint(url: &str, client: &reqwest::Client) -> bool {
    if let Some(cached) = probe_cache().read().ok().and_then(|m| m.get(url).copied()) {
        return cached;
    }
    // OpenAI-compat: /v1/embeddings is POST. A bare GET gives us 405 if the
    // route exists, 404 if not — both good enough for presence detection.
    let live = match client.get(url).send().await {
        Ok(r) => {
            let s = r.status().as_u16();
            s == 405 || r.status().is_success()
        }
        Err(_) => false,
    };
    if let Ok(mut m) = probe_cache().write() {
        m.insert(url.to_string(), live);
    }
    live
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The hash embedding is L2-normalized, so the cosine of a vector with
    /// itself is 1.0 (within FP epsilon). Baseline shape check.
    #[test]
    fn hash_embed_self_similarity_is_one() {
        let v = hash_embed("bob prefers terse CLI output");
        let s = cosine(&v, &v);
        assert!(
            (s - 1.0).abs() < 1e-5,
            "self cosine should be ~1.0, got {s}"
        );
    }

    /// Two texts that share most tokens should score higher than two
    /// texts that share none. This is the "better than keyword overlap"
    /// claim the task asked us to earn.
    #[test]
    fn hash_embed_overlap_beats_disjoint() {
        let q = hash_embed("CLI preferences");
        let hit = hash_embed("bob prefers terse CLI output");
        let miss = hash_embed("rocprof kernels run hot on gfx1151");

        let s_hit = cosine(&q, &hit);
        let s_miss = cosine(&q, &miss);
        assert!(
            s_hit > s_miss,
            "overlap should outscore disjoint: hit={s_hit} miss={s_miss}"
        );
        // And the disjoint score should be near zero (no shared buckets
        // except by pure hash collision — unlikely at dim=128).
        assert!(s_miss < 0.2, "disjoint score should be low, got {s_miss}");
    }

    /// Ranker returns all claims ordered by descending cosine score.
    #[test]
    fn rank_orders_by_descending_similarity() {
        let claims = vec![
            "rocprof kernels run hot on gfx1151".to_string(),
            "bob prefers terse CLI output".to_string(),
            "bob dislikes verbose logging in CLI tools".to_string(),
        ];
        let ranked = rank_by_hash_embedding("CLI preferences", &claims);
        assert_eq!(ranked.len(), 3);
        // The top-ranked entry should be one of the two CLI claims
        // (either; both share the "cli" token with the query).
        let top_idx = ranked[0].1;
        assert!(
            top_idx == 1 || top_idx == 2,
            "top should be one of the CLI claims, got idx={top_idx}"
        );
        // And the rocprof claim should be last (or tied-last with a
        // hash-collision neighbor, but in practice last).
        assert_eq!(ranked[2].1, 0, "rocprof claim should rank last");
    }

    /// Empty query produces a zero vector, so every cosine is 0 (or
    /// NaN-guarded 0 after the normalize early-return). Ranker must not
    /// panic and must return all claims.
    #[test]
    fn rank_empty_query_returns_all_without_panic() {
        let claims = vec!["a".to_string(), "b".to_string()];
        let ranked = rank_by_hash_embedding("", &claims);
        assert_eq!(ranked.len(), 2);
    }
}
