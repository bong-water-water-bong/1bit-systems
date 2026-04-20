//! Live server metrics — counters + bounded latency histograms.
//!
//! A single [`Metrics`] instance lives behind an `Arc` alongside the
//! [`SharedBackend`](crate::backend::SharedBackend); every request handler
//! calls [`Metrics::record_request`] after awaiting the backend's `generate*`
//! method, passing the prompt / completion token counts and the wall-clock
//! elapsed for the whole request. `GET /metrics` reads out a
//! [`MetricsSnapshot`] via [`Metrics::snapshot`].
//!
//! Design notes:
//! * Counters are `AtomicU64` for lock-free increments on the hot path.
//! * The two deques (per-request latency and "recent request sums") are
//!   `Mutex<VecDeque<..>>` — contention is negligible (one lock per
//!   completion, not per token) and the bounded size keeps memory flat.
//! * `tokps_recent` is computed as
//!   `sum(completion_tokens_recent) / sum(elapsed_recent).as_secs_f64()`,
//!   which matches the "tok/s the server just produced" intuition better
//!   than a rolling moving-average over individual token inter-arrival
//!   times (we don't see those at the HTTP layer anyway).
//! * No histogram library — we compute p50 / p95 on demand by cloning the
//!   (small, capped) deque and sorting. 1024 × 8 B = 8 KB, sort is O(n log n)
//!   on cold data, < 50 µs. Fine for a 3-second poll.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use serde::Serialize;

/// Cap on the rolling per-token-latency deque (one slot ≈ one completed
/// request). 1024 requests is ~10 minutes of steady traffic at 2 rps, which
/// gives a stable p95 without letting memory drift.
const PER_REQ_TOKPS_CAP: usize = 1024;

/// Cap on the per-request total-latency deque. Tighter than `PER_REQ_TOKPS_CAP`
/// because the p50/p95 numbers are most useful when they reflect the recent
/// past; 256 requests is roughly the last few minutes.
const PER_REQ_LATENCY_CAP: usize = 256;

/// One deque slot — holds the completion-tokens and elapsed pair so we can
/// compute aggregate tok/s without a second pass over the data.
#[derive(Debug, Clone, Copy)]
struct Sample {
    completion_tokens: u64,
    elapsed: Duration,
}

/// Process-wide metrics handle. Construct once at server boot, clone the
/// `Arc` into the router state.
#[derive(Debug)]
pub struct Metrics {
    started: Instant,
    total_requests: AtomicU64,
    total_generated_tokens: AtomicU64,
    /// Rolling samples used for `tokps_recent`.
    tokps_samples: Mutex<VecDeque<Sample>>,
    /// Rolling per-request latencies used for p50 / p95.
    latencies: Mutex<VecDeque<Duration>>,
    /// Gauge — 1 if the most recent `GET /v1/npu/status` saw a live device
    /// node + `amdxdna` loaded, 0 otherwise. Stays at 0 until the endpoint
    /// is polled at least once; that's fine, the endpoint is the source of
    /// truth.
    npu_up: AtomicU64,
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Metrics {
    /// New zeroed metrics, `started` pinned to the construction instant.
    pub fn new() -> Self {
        Self {
            started: Instant::now(),
            total_requests: AtomicU64::new(0),
            total_generated_tokens: AtomicU64::new(0),
            tokps_samples: Mutex::new(VecDeque::with_capacity(PER_REQ_TOKPS_CAP)),
            latencies: Mutex::new(VecDeque::with_capacity(PER_REQ_LATENCY_CAP)),
            npu_up: AtomicU64::new(0),
        }
    }

    /// Set the `halo_npu_up` gauge. Called by the `/v1/npu/status` handler
    /// after probing the hardware.
    pub fn set_npu_up(&self, up: bool) {
        self.npu_up.store(if up { 1 } else { 0 }, Ordering::Relaxed);
    }

    /// Read the `halo_npu_up` gauge.
    pub fn npu_up(&self) -> u64 {
        self.npu_up.load(Ordering::Relaxed)
    }

    /// Record one completed request.
    ///
    /// `prompt_tokens` is counted separately from generated tokens (the
    /// snapshot field is named `generated_tokens` precisely so we don't
    /// conflate the two). `elapsed` is the wall-clock duration the HTTP
    /// handler spent awaiting the backend — i.e. what the user sees.
    pub fn record_request(&self, _prompt_tokens: u32, completion_tokens: u32, elapsed: Duration) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.total_generated_tokens
            .fetch_add(completion_tokens as u64, Ordering::Relaxed);

        let sample = Sample {
            completion_tokens: completion_tokens as u64,
            elapsed,
        };

        if let Ok(mut d) = self.tokps_samples.lock() {
            if d.len() == PER_REQ_TOKPS_CAP {
                d.pop_front();
            }
            d.push_back(sample);
        }
        if let Ok(mut d) = self.latencies.lock() {
            if d.len() == PER_REQ_LATENCY_CAP {
                d.pop_front();
            }
            d.push_back(elapsed);
        }
    }

    /// Take a point-in-time snapshot. Computes p50 / p95 by cloning the
    /// (bounded) latency deque and sorting; cheap enough for a poll-driven
    /// endpoint.
    pub fn snapshot(&self) -> MetricsSnapshot {
        let requests = self.total_requests.load(Ordering::Relaxed);
        let generated_tokens = self.total_generated_tokens.load(Ordering::Relaxed);
        let uptime_secs = self.started.elapsed().as_secs();

        // tok/s over the recent sample window.
        let (recent_tokens, recent_secs) = {
            let d = self.tokps_samples.lock().expect("tokps_samples poisoned");
            let mut toks = 0u64;
            let mut secs = 0.0f64;
            for s in d.iter() {
                toks += s.completion_tokens;
                secs += s.elapsed.as_secs_f64();
            }
            (toks, secs)
        };
        let tokps_recent = if recent_secs > 0.0 {
            recent_tokens as f64 / recent_secs
        } else {
            0.0
        };

        // p50 / p95 over the latency deque.
        let (p50_ms, p95_ms) = {
            let d = self.latencies.lock().expect("latencies poisoned");
            if d.is_empty() {
                (0.0, 0.0)
            } else {
                let mut v: Vec<u128> = d.iter().map(|x| x.as_millis()).collect();
                v.sort_unstable();
                (percentile_ms(&v, 50.0), percentile_ms(&v, 95.0))
            }
        };

        MetricsSnapshot {
            requests,
            generated_tokens,
            uptime_secs,
            tokps_recent,
            p50_ms,
            p95_ms,
            completion_tokens_last_hour: recent_tokens,
            npu_up: self.npu_up.load(Ordering::Relaxed),
        }
    }
}

/// JSON shape returned by `GET /metrics`.
#[derive(Debug, Clone, Serialize)]
pub struct MetricsSnapshot {
    pub requests: u64,
    pub generated_tokens: u64,
    pub uptime_secs: u64,
    pub tokps_recent: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    /// Sum of completion tokens across the rolling window. Named
    /// `last_hour` for the external contract — the window is at most
    /// `PER_REQ_TOKPS_CAP` requests, which is a fair proxy at typical
    /// traffic levels.
    pub completion_tokens_last_hour: u64,
    /// `halo_npu_up` gauge — 1 if the NPU is live, 0 otherwise. Updated
    /// lazily by the `/v1/npu/status` handler; stays 0 until first poll.
    pub npu_up: u64,
}

/// Nearest-rank percentile on a *sorted* `Vec<u128>` of millisecond values.
fn percentile_ms(sorted: &[u128], pct: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    // rank = ceil(pct/100 * n) — 1-based, clamped.
    let n = sorted.len();
    let rank = ((pct / 100.0) * n as f64).ceil() as usize;
    let idx = rank.saturating_sub(1).min(n - 1);
    sorted[idx] as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_shape_empty() {
        let m = Metrics::new();
        let s = m.snapshot();
        assert_eq!(s.requests, 0);
        assert_eq!(s.generated_tokens, 0);
        assert_eq!(s.tokps_recent, 0.0);
        assert_eq!(s.p50_ms, 0.0);
        assert_eq!(s.p95_ms, 0.0);
        assert_eq!(s.completion_tokens_last_hour, 0);
        assert_eq!(s.npu_up, 0);
        // uptime_secs is monotonic but may be 0 immediately after construction.

        // Shape assertion via serde: every field serializes.
        let v = serde_json::to_value(&s).unwrap();
        for k in [
            "requests",
            "generated_tokens",
            "uptime_secs",
            "tokps_recent",
            "p50_ms",
            "p95_ms",
            "completion_tokens_last_hour",
            "npu_up",
        ] {
            assert!(v.get(k).is_some(), "snapshot missing field {k}");
        }
    }

    #[test]
    fn record_request_updates_counters() {
        let m = Metrics::new();
        m.record_request(5, 10, Duration::from_millis(100));
        m.record_request(7, 20, Duration::from_millis(200));

        let s = m.snapshot();
        assert_eq!(s.requests, 2);
        assert_eq!(s.generated_tokens, 30);
        // 30 tokens / 0.3 s = 100 tok/s.
        assert!(
            (s.tokps_recent - 100.0).abs() < 1e-6,
            "tokps_recent = {}",
            s.tokps_recent
        );
        assert_eq!(s.completion_tokens_last_hour, 30);
    }

    #[test]
    fn p95_on_known_distribution() {
        let m = Metrics::new();
        // 100 samples, 10 ms through 1000 ms in 10 ms steps. p50 should
        // fall at the 50th-ranked sample (500 ms), p95 at the 95th (950 ms).
        for i in 1..=100u64 {
            m.record_request(0, 1, Duration::from_millis(i * 10));
        }
        let s = m.snapshot();
        assert_eq!(s.requests, 100);
        assert!(
            (s.p50_ms - 500.0).abs() < 1.0,
            "p50_ms = {} (expected ~500)",
            s.p50_ms
        );
        assert!(
            (s.p95_ms - 950.0).abs() < 1.0,
            "p95_ms = {} (expected ~950)",
            s.p95_ms
        );
    }

    #[test]
    fn deque_caps_hold_under_flood() {
        let m = Metrics::new();
        for _ in 0..(PER_REQ_TOKPS_CAP + PER_REQ_LATENCY_CAP + 100) {
            m.record_request(0, 1, Duration::from_millis(1));
        }
        // Counters keep climbing, deques stay capped.
        assert_eq!(m.tokps_samples.lock().unwrap().len(), PER_REQ_TOKPS_CAP);
        assert_eq!(m.latencies.lock().unwrap().len(), PER_REQ_LATENCY_CAP);
    }
}
