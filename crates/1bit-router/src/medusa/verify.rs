//! Tree-attention verification state machine.
//!
//! Accepts the longest prefix of head-predicted candidate tokens that
//! matches the base-model's argmax at each position. Sequential today;
//! the plan doc (`docs/wiki/Medusa-Integration-Plan.md §2.1`) discusses
//! a dedicated tree-attention kernel for a later pass that batches the
//! per-candidate queries into one attention launch with a sparse mask.
//!
//! # State machine per decode step
//!
//! 1. Backbone runs one forward pass on the last accepted token →
//!    post-final-norm hidden state `h`.
//! 2. Four Medusa heads fire in parallel on `h` → four logit rows.
//!    `sampler.argmax` picks one candidate token per head:
//!    `c = [c_1, c_2, c_3, c_4]` (representing t+1, t+2, t+3, t+4).
//! 3. Base-model runs one **tree-attention forward pass** over the
//!    candidate vector (today: four sequential `forward_token` calls;
//!    tomorrow: one batched launch). This produces four argmax
//!    predictions `a = [a_1, a_2, a_3, a_4]` — what the base model
//!    *would have* produced at each of those positions.
//! 4. Walk `c` vs `a` in index order; accept as long as `c_i == a_i`.
//!    First mismatch → discard the rest. `a` at the mismatch position
//!    is the "fallback" token emitted as the accepted prefix's tail.
//!
//! The state machine lives on the router across decode steps so we can
//! emit per-step acceptance rates into `/metrics` without recomputing
//! windowed averages.

use super::heads::NUM_MEDUSA_HEADS;

/// Outcome of a single verify step. Carried back to the router so it can
/// advance the KV cache by `accepted_len` slots, seed the next iteration
/// with `next_token`, and update its per-head acceptance counters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifyOutcome {
    /// Number of speculative tokens accepted this step (0..=N). `0`
    /// means every head disagreed with the base model at index 0; the
    /// base-model argmax is still emitted as `next_token`, so the step
    /// always advances by at least one token.
    pub accepted_len: usize,
    /// The token to emit as the "next" token after this verify step.
    /// Either the last head token in the accepted prefix (if everything
    /// matched) or the base-model argmax at the first mismatch (if some
    /// prefix was accepted) or the base-model argmax at index 0 (if
    /// nothing was accepted).
    pub next_token: i32,
    /// Per-head match bitmap — `matches[i] = true` iff
    /// `head_candidates[i] == base_argmax[i]`. Stored for telemetry
    /// even past the first mismatch (useful for measuring downstream
    /// head acceptance rates in isolation, e.g. if head 2 happens to
    /// agree after head 1 rejected).
    pub matches: [bool; NUM_MEDUSA_HEADS],
}

/// Rolling state for the verifier. Today this is just acceptance
/// counters. Tomorrow it grows the tree-attention kernel's per-step
/// mask builder + KV rewind state.
#[derive(Debug, Clone)]
pub struct TreeVerifier {
    /// Cumulative count of verify steps run since router startup.
    /// Used to normalize the per-head acceptance counters.
    pub steps: u64,
    /// Per-head accepted-count. Divide by `steps` to get the live
    /// per-head acceptance rate that ops dashboards display alongside
    /// the upstream model-card rates.
    pub head_accepted: [u64; NUM_MEDUSA_HEADS],
    /// Cumulative accepted-prefix length (0..=N per step). Divide by
    /// `steps` to get the average per-step speedup (== mean
    /// `accepted_len + 1`, since each step always emits at least one
    /// base-model token).
    pub total_accepted_prefix_len: u64,
}

impl Default for TreeVerifier {
    fn default() -> Self {
        Self::new()
    }
}

impl TreeVerifier {
    /// Fresh verifier — all counters zero.
    pub const fn new() -> Self {
        Self {
            steps: 0,
            head_accepted: [0; NUM_MEDUSA_HEADS],
            total_accepted_prefix_len: 0,
        }
    }

    /// Run one verify step given the head-predicted candidates and the
    /// base-model's per-position argmax. Accepts the longest matching
    /// prefix.
    ///
    /// Preconditions:
    /// * Both slices must have length [`NUM_MEDUSA_HEADS`]. Violations
    ///   return `None` — the router treats this as a scaffolding bug
    ///   and falls through to the non-Medusa path.
    ///
    /// Side effect: updates `self.steps`, `self.head_accepted`, and
    /// `self.total_accepted_prefix_len`.
    pub fn verify_step(
        &mut self,
        head_candidates: &[i32],
        base_argmax: &[i32],
    ) -> Option<VerifyOutcome> {
        if head_candidates.len() != NUM_MEDUSA_HEADS || base_argmax.len() != NUM_MEDUSA_HEADS {
            return None;
        }

        let mut matches = [false; NUM_MEDUSA_HEADS];
        let mut accepted_len = 0usize;
        let mut first_mismatch: Option<usize> = None;

        for i in 0..NUM_MEDUSA_HEADS {
            let hit = head_candidates[i] == base_argmax[i];
            matches[i] = hit;
            if hit && first_mismatch.is_none() {
                accepted_len += 1;
            } else if first_mismatch.is_none() {
                first_mismatch = Some(i);
            }
        }

        let next_token = match first_mismatch {
            // Everything accepted: the last base-model argmax is the
            // "next token" so the next decode step kicks off from there.
            None => base_argmax[NUM_MEDUSA_HEADS - 1],
            // Partial acceptance: emit the base-model argmax at the
            // first-mismatch position. Head candidates beyond that are
            // thrown out.
            Some(i) => base_argmax[i],
        };

        // Telemetry — applied to every step, even the first-mismatch
        // case (we still want to see how often downstream heads would
        // have agreed if the upstream hadn't rejected).
        self.steps += 1;
        for (i, &m) in matches.iter().enumerate() {
            if m {
                self.head_accepted[i] += 1;
            }
        }
        self.total_accepted_prefix_len += accepted_len as u64;

        Some(VerifyOutcome {
            accepted_len,
            next_token,
            matches,
        })
    }

    /// Mean per-head acceptance rate — used by the `/metrics` endpoint.
    /// Returns `None` until the first step so we don't divide by zero.
    pub fn per_head_acceptance(&self) -> Option<[f64; NUM_MEDUSA_HEADS]> {
        if self.steps == 0 {
            return None;
        }
        let denom = self.steps as f64;
        let mut out = [0.0f64; NUM_MEDUSA_HEADS];
        for (o, &acc) in out.iter_mut().zip(self.head_accepted.iter()) {
            *o = (acc as f64) / denom;
        }
        Some(out)
    }

    /// Mean accepted prefix length per step (0..=N).
    pub fn mean_accepted_prefix_len(&self) -> Option<f64> {
        if self.steps == 0 {
            return None;
        }
        Some((self.total_accepted_prefix_len as f64) / (self.steps as f64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// All four heads match → accepted_len == 4, next_token is the
    /// last head's base argmax.
    #[test]
    fn verify_all_match() {
        let mut v = TreeVerifier::new();
        let heads = [10, 20, 30, 40];
        let base = [10, 20, 30, 40];
        let out = v.verify_step(&heads, &base).expect("valid lengths");
        assert_eq!(out.accepted_len, 4);
        assert_eq!(out.next_token, 40);
        assert_eq!(out.matches, [true; 4]);
        assert_eq!(v.steps, 1);
        assert_eq!(v.head_accepted, [1, 1, 1, 1]);
        assert_eq!(v.total_accepted_prefix_len, 4);
    }

    /// First-head mismatch → accepted_len == 0, next_token is the base
    /// argmax at position 0. Downstream matches still contribute to
    /// per-head telemetry (they would have matched on their own).
    #[test]
    fn verify_first_mismatch() {
        let mut v = TreeVerifier::new();
        let heads = [99, 20, 30, 40];
        let base = [10, 20, 30, 40];
        let out = v.verify_step(&heads, &base).expect("valid lengths");
        assert_eq!(out.accepted_len, 0);
        assert_eq!(out.next_token, 10); // base argmax at the mismatch
        assert_eq!(out.matches, [false, true, true, true]);
        // Telemetry: per-head counters reflect the isolated matches.
        assert_eq!(v.head_accepted, [0, 1, 1, 1]);
        assert_eq!(v.total_accepted_prefix_len, 0);
    }

    /// Middle mismatch — accept 2, emit base argmax at index 2.
    #[test]
    fn verify_partial_then_mismatch() {
        let mut v = TreeVerifier::new();
        let heads = [10, 20, 77, 40];
        let base = [10, 20, 30, 40];
        let out = v.verify_step(&heads, &base).expect("valid lengths");
        assert_eq!(out.accepted_len, 2);
        assert_eq!(out.next_token, 30);
        assert_eq!(out.matches, [true, true, false, true]);
        // Head 3 still matched in isolation.
        assert_eq!(v.head_accepted, [1, 1, 0, 1]);
        assert_eq!(v.total_accepted_prefix_len, 2);
    }

    /// Bad input lengths → None (signal the scaffolding bug without a
    /// panic so the router falls through to the non-Medusa path).
    #[test]
    fn verify_bad_lengths() {
        let mut v = TreeVerifier::new();
        assert!(v.verify_step(&[1, 2, 3], &[1, 2, 3, 4]).is_none());
        assert!(v.verify_step(&[], &[]).is_none());
        assert!(v.verify_step(&[1; 5], &[1; 5]).is_none());
        // Failed calls must not move the counters.
        assert_eq!(v.steps, 0);
    }

    /// Acceptance-rate math matches the counter math.
    #[test]
    fn acceptance_rate_smoke() {
        let mut v = TreeVerifier::new();
        assert!(v.per_head_acceptance().is_none());
        assert!(v.mean_accepted_prefix_len().is_none());

        v.verify_step(&[10, 20, 30, 40], &[10, 20, 30, 40]);
        v.verify_step(&[10, 20, 77, 40], &[10, 20, 30, 40]);
        // Steps: 2. Per-head matches: [2, 2, 1, 2]. Prefix accepted: 4 + 2 = 6.
        let rates = v.per_head_acceptance().unwrap();
        assert!((rates[0] - 1.0).abs() < 1e-12);
        assert!((rates[2] - 0.5).abs() < 1e-12);
        let mean = v.mean_accepted_prefix_len().unwrap();
        assert!((mean - 3.0).abs() < 1e-12);
    }
}
