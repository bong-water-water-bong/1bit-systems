//! Host-side logits sampler.
//!
//! Ported from the `sample_host` lambda in
//! `/home/bcloud/repos/rocm-cpp/tools/bitnet_decode.cpp`. Semantics match
//! byte-for-byte (same application order: repetition penalty → top-k →
//! softmax → top-p → multinomial draw).
//!
//! For greedy argmax (what the C++ fast-path uses when `temperature <= 0`),
//! call [`Sampler::greedy`] directly — it takes the raw logits slice.

use crate::error::HaloError;
use crate::types::TokenId;

/// Sampling hyperparameters.
///
/// Defaults are conservative — `temperature=0` enables greedy argmax
/// (identical to `rcpp_argmax_fp32` on device).
#[derive(Debug, Clone, Copy)]
pub struct SamplerConfig {
    /// Softmax temperature. `<= 0` means "use greedy argmax, ignore the
    /// rest". `1.0` is vanilla softmax.
    pub temperature: f32,
    /// Keep only the top `k` logits (mask the rest to `-inf`). `0` disables.
    pub top_k: u32,
    /// Nucleus sampling: keep the smallest prefix whose cumulative
    /// probability is `>= top_p`. `1.0` disables.
    pub top_p: f32,
    /// Repetition penalty. `1.0` disables. `> 1` discourages repeats.
    pub rep_penalty: f32,
    /// Window size for repetition penalty — how many trailing tokens of
    /// `recent` to penalize. `0` disables (same effect as `rep_penalty=1`).
    pub rep_last_n: u32,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            rep_penalty: 1.0,
            rep_last_n: 64,
            seed: 0xC0FFEE,
        }
    }
}

/// Stateful sampler: owns the RNG so draws are reproducible across turns.
pub struct Sampler {
    cfg: SamplerConfig,
    rng: fastrand::Rng,
    /// Reusable scratch buffer so per-token sampling is alloc-free after warm-up.
    scratch: Vec<f32>,
    index_scratch: Vec<u32>,
}

impl Sampler {
    pub fn new(cfg: SamplerConfig) -> Self {
        Self {
            cfg,
            rng: fastrand::Rng::with_seed(cfg.seed),
            scratch: Vec::new(),
            index_scratch: Vec::new(),
        }
    }

    pub fn config(&self) -> &SamplerConfig {
        &self.cfg
    }

    pub fn set_config(&mut self, cfg: SamplerConfig) {
        // Reseeding the RNG only when the seed actually changes — matches
        // expected REPL behaviour (changing temperature mid-conversation
        // shouldn't reset the stream).
        if cfg.seed != self.cfg.seed {
            self.rng = fastrand::Rng::with_seed(cfg.seed);
        }
        self.cfg = cfg;
    }

    /// Greedy argmax — returns the token id with the largest logit.
    /// Identical to `rcpp_argmax_fp32`. Independent of sampler state so
    /// it's a free function in spirit; kept as a method for API symmetry.
    pub fn greedy(logits: &[f32]) -> Result<TokenId, HaloError> {
        if logits.is_empty() {
            return Err(HaloError::Sampler("empty logits"));
        }
        let (mut best_idx, mut best_val) = (0usize, logits[0]);
        for (i, &v) in logits.iter().enumerate().skip(1) {
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
        }
        Ok(best_idx as TokenId)
    }

    /// Full sampler path (`temperature > 0`).
    ///
    /// `logits` is mutated in-place (rep-penalty + masking + softmax) so
    /// callers that need the raw logits for logging should clone first.
    /// `recent` is the full decode history; we only read the last
    /// `rep_last_n` entries.
    pub fn sample(&mut self, logits: &mut [f32], recent: &[TokenId]) -> Result<TokenId, HaloError> {
        let v = logits.len();
        if v == 0 {
            return Err(HaloError::Sampler("empty logits"));
        }

        // --- Greedy fast path (matches C++ `temperature <= 0` branch) ---
        if self.cfg.temperature <= 0.0 {
            return Self::greedy(logits);
        }

        // --- Repetition penalty ---
        if (self.cfg.rep_penalty - 1.0).abs() > f32::EPSILON && self.cfg.rep_last_n > 0 {
            let n = recent.len();
            let start = n.saturating_sub(self.cfg.rep_last_n as usize);
            for &id in &recent[start..n] {
                if id >= 0 && (id as usize) < v {
                    let l = &mut logits[id as usize];
                    *l = if *l > 0.0 {
                        *l / self.cfg.rep_penalty
                    } else {
                        *l * self.cfg.rep_penalty
                    };
                }
            }
        }

        // --- Top-k: mask everything below the k-th largest to -inf ---
        let top_k = self.cfg.top_k as usize;
        if top_k > 0 && top_k < v {
            self.scratch.clear();
            self.scratch.extend_from_slice(logits);
            // nth_element equivalent: partition so element at (V - k) is the
            // kth-largest; use `select_nth_unstable_by` on a total order.
            let pivot = v - top_k;
            self.scratch.select_nth_unstable_by(pivot, |a, b| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            });
            let thresh = self.scratch[pivot];
            for l in logits.iter_mut() {
                if *l < thresh {
                    *l = f32::NEG_INFINITY;
                }
            }
        }

        // --- Softmax with temperature ---
        let mut max_logit = f32::NEG_INFINITY;
        for &l in logits.iter() {
            if l > max_logit {
                max_logit = l;
            }
        }
        let inv_temp = 1.0 / self.cfg.temperature;
        let mut sum = 0.0f64;
        for l in logits.iter_mut() {
            let e = ((*l - max_logit) * inv_temp).exp();
            *l = e;
            sum += e as f64;
        }
        let inv = if sum > 0.0 { (1.0 / sum) as f32 } else { 1.0 };
        for l in logits.iter_mut() {
            *l *= inv;
        }

        // --- Top-p (nucleus) ---
        if self.cfg.top_p > 0.0 && self.cfg.top_p < 1.0 {
            self.index_scratch.clear();
            self.index_scratch.extend(0..v as u32);
            // Sort indices by descending probability.
            self.index_scratch.sort_unstable_by(|&a, &b| {
                logits[b as usize]
                    .partial_cmp(&logits[a as usize])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut csum = 0.0f32;
            let mut cutoff = v;
            for (i, &idx) in self.index_scratch.iter().enumerate() {
                csum += logits[idx as usize];
                if csum >= self.cfg.top_p {
                    cutoff = i + 1;
                    break;
                }
            }
            // Zero the tail.
            for &idx in &self.index_scratch[cutoff..] {
                logits[idx as usize] = 0.0;
            }
            // Renormalize the kept head.
            let mut keep_sum = 0.0f32;
            for &idx in &self.index_scratch[..cutoff] {
                keep_sum += logits[idx as usize];
            }
            if keep_sum > 0.0 {
                let s = 1.0 / keep_sum;
                for &idx in &self.index_scratch[..cutoff] {
                    logits[idx as usize] *= s;
                }
            }
        }

        // --- Multinomial draw ---
        let r = self.rng.f32();
        let mut acc = 0.0f32;
        for (i, &p) in logits.iter().enumerate() {
            acc += p;
            if acc >= r {
                return Ok(i as TokenId);
            }
        }
        Ok((v - 1) as TokenId)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_argmax() {
        let logits = [0.1, 3.2, -1.0, 0.5];
        assert_eq!(Sampler::greedy(&logits).unwrap(), 1);
    }

    #[test]
    fn temperature_zero_is_greedy() {
        let mut s = Sampler::new(SamplerConfig {
            temperature: 0.0,
            ..SamplerConfig::default()
        });
        let mut l = vec![0.1, 3.2, -1.0, 0.5];
        assert_eq!(s.sample(&mut l, &[]).unwrap(), 1);
    }

    #[test]
    fn top_k_1_is_argmax() {
        let mut s = Sampler::new(SamplerConfig {
            temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
            rep_penalty: 1.0,
            rep_last_n: 0,
            seed: 42,
        });
        let mut l = vec![0.1, 3.2, -1.0, 0.5];
        // With only index 1 unmasked, softmax puts all mass there, so draw
        // always hits index 1 regardless of seed.
        for _ in 0..10 {
            let mut lc = l.clone();
            assert_eq!(s.sample(&mut lc, &[]).unwrap(), 1);
        }
        // Reference the original once to avoid "unused" warnings.
        let _ = &mut l;
    }

    #[test]
    fn rep_penalty_suppresses_recent() {
        // Without rep penalty, argmax picks token 1.
        let mut s = Sampler::new(SamplerConfig {
            temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
            rep_penalty: 10.0,
            rep_last_n: 4,
            seed: 0,
        });
        let mut l = vec![0.1, 3.2, 3.0, 0.5];
        // Mark token 1 as "recent" — rep penalty divides its positive logit
        // by 10, dropping it to 0.32, so top-1 becomes token 2 (3.0).
        let next = s.sample(&mut l, &[1, 1, 1]).unwrap();
        assert_eq!(next, 2, "expected rep penalty to push argmax to token 2");
    }

    #[test]
    fn multinomial_distribution_roughly_correct() {
        // With a two-class softmax where class 1 is 100x more likely,
        // we should almost always pick class 1.
        let mut s = Sampler::new(SamplerConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            rep_penalty: 1.0,
            rep_last_n: 0,
            seed: 7,
        });
        let mut hits_1 = 0;
        for _ in 0..200 {
            let mut l = vec![0.0, (100f32).ln()]; // probs ~ [1/101, 100/101]
            if s.sample(&mut l, &[]).unwrap() == 1 {
                hits_1 += 1;
            }
        }
        assert!(hits_1 > 180, "class-1 hits={hits_1}/200, expected > 180");
    }
}
