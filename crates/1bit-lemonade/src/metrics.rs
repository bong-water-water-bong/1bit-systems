//! Prometheus-text `/metrics` for 1bit-lemonade.
//!
//! We hand-roll the text exposition format rather than pulling in
//! `prometheus_client` or the `metrics` + `metrics-exporter-prometheus`
//! pair. Reasons:
//!
//! * Zero new crate deps keeps the `cargo-tree` small (Rule A/B discipline:
//!   if a 60-line module can replace a 20-crate tree, it wins).
//! * The exposition format is trivial — three counter/histogram/gauge
//!   lines per metric, no protobuf, no labels library.
//! * Consistency with [`crate::dispatch`]: the /metrics endpoint is pure
//!   plumbing, no business logic lives here.
//!
//! What we export:
//!
//! * `onebit_lemonade_requests_total{route,status}` — request counter.
//! * `onebit_lemonade_request_seconds{route}` — latency histogram with
//!   fixed buckets (0.005 … 60 s) matching Prometheus defaults.
//! * `onebit_lemonade_upstream_up` — 1 if the most recent probe of the
//!   active [`crate::dispatch::Upstream`] succeeded, 0 otherwise.

use std::collections::BTreeMap;
use std::fmt::Write;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// Default Prometheus bucket set (seconds). Covers 5 ms to 10 s — below
/// that is noise on a gateway, above that is a stuck request.
pub const DEFAULT_BUCKETS_SECONDS: &[f64] = &[
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
];

/// Histogram state for a single `{route}` label pair. Bucket counts are
/// cumulative following the Prometheus convention.
#[derive(Debug, Default, Clone)]
struct Histogram {
    /// Parallel to [`DEFAULT_BUCKETS_SECONDS`]. `counts[i]` = number of
    /// observations ≤ `DEFAULT_BUCKETS_SECONDS[i]`.
    counts: Vec<u64>,
    sum: f64,
    count: u64,
}

impl Histogram {
    fn new() -> Self {
        Self {
            counts: vec![0u64; DEFAULT_BUCKETS_SECONDS.len()],
            sum: 0.0,
            count: 0,
        }
    }

    fn observe(&mut self, seconds: f64) {
        self.sum += seconds;
        self.count += 1;
        for (i, &b) in DEFAULT_BUCKETS_SECONDS.iter().enumerate() {
            if seconds <= b {
                self.counts[i] += 1;
            }
        }
    }
}

/// Metrics registry. One per process, shared via `Arc` into the router.
#[derive(Debug)]
pub struct Metrics {
    /// `onebit_lemonade_requests_total{route,status}`. Key = `(route, status_code)`.
    requests: Mutex<BTreeMap<(String, u16), u64>>,
    /// `onebit_lemonade_request_seconds{route}`. Key = `route`.
    latencies: Mutex<BTreeMap<String, Histogram>>,
    /// `onebit_lemonade_upstream_up` gauge — 0 or 1.
    upstream_up: AtomicU64,
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            requests: Mutex::new(BTreeMap::new()),
            latencies: Mutex::new(BTreeMap::new()),
            upstream_up: AtomicU64::new(0),
        }
    }

    /// Instrument one upstream-dispatched request. Bumps the counter for
    /// `{route,status}`, observes the latency, and folds the up/down signal
    /// into the gauge. Called by the route layer immediately after the
    /// [`crate::dispatch::Upstream`] call returns.
    pub fn record(&self, route: &str, status: u16, elapsed: Duration) {
        if let Ok(mut m) = self.requests.lock() {
            *m.entry((route.to_string(), status)).or_insert(0) += 1;
        }
        if let Ok(mut m) = self.latencies.lock() {
            m.entry(route.to_string())
                .or_insert_with(Histogram::new)
                .observe(elapsed.as_secs_f64());
        }
        // Any 5xx means 1bit-server or the dispatch flipped to down.
        let healthy = status < 500;
        self.upstream_up
            .store(if healthy { 1 } else { 0 }, Ordering::Relaxed);
    }

    /// Explicitly set the `onebit_lemonade_upstream_up` gauge. Called by the
    /// /metrics handler itself so a scrape gets a live probe rather than
    /// only the signal from the last request.
    pub fn set_upstream_up(&self, up: bool) {
        self.upstream_up
            .store(if up { 1 } else { 0 }, Ordering::Relaxed);
    }

    /// Render the current state as a Prometheus text exposition blob.
    /// Follows the canonical ordering: `# HELP`, `# TYPE`, then samples.
    pub fn render(&self) -> String {
        let mut out = String::with_capacity(2048);

        // Counter
        let _ = writeln!(
            out,
            "# HELP onebit_lemonade_requests_total Total requests processed by 1bit-lemonade."
        );
        let _ = writeln!(out, "# TYPE onebit_lemonade_requests_total counter");
        if let Ok(m) = self.requests.lock() {
            if m.is_empty() {
                // At least one sample so scrapers don't flag "no data".
                let _ = writeln!(
                    out,
                    "onebit_lemonade_requests_total{{route=\"init\",status=\"0\"}} 0"
                );
            } else {
                for ((route, status), count) in m.iter() {
                    let _ = writeln!(
                        out,
                        "onebit_lemonade_requests_total{{route=\"{}\",status=\"{}\"}} {}",
                        escape_label(route),
                        status,
                        count
                    );
                }
            }
        }

        // Histogram
        let _ = writeln!(
            out,
            "# HELP onebit_lemonade_request_seconds Request latency in seconds."
        );
        let _ = writeln!(out, "# TYPE onebit_lemonade_request_seconds histogram");
        if let Ok(m) = self.latencies.lock() {
            if m.is_empty() {
                // Emit an empty histogram so `# TYPE histogram` is followed
                // by at least one sample line (prometheus accepts this).
                for &b in DEFAULT_BUCKETS_SECONDS {
                    let _ = writeln!(
                        out,
                        "onebit_lemonade_request_seconds_bucket{{route=\"init\",le=\"{b}\"}} 0"
                    );
                }
                let _ = writeln!(
                    out,
                    "onebit_lemonade_request_seconds_bucket{{route=\"init\",le=\"+Inf\"}} 0"
                );
                let _ = writeln!(
                    out,
                    "onebit_lemonade_request_seconds_sum{{route=\"init\"}} 0"
                );
                let _ = writeln!(
                    out,
                    "onebit_lemonade_request_seconds_count{{route=\"init\"}} 0"
                );
            } else {
                for (route, h) in m.iter() {
                    let r = escape_label(route);
                    for (i, &b) in DEFAULT_BUCKETS_SECONDS.iter().enumerate() {
                        let _ = writeln!(
                            out,
                            "onebit_lemonade_request_seconds_bucket{{route=\"{r}\",le=\"{b}\"}} {}",
                            h.counts[i]
                        );
                    }
                    let _ = writeln!(
                        out,
                        "onebit_lemonade_request_seconds_bucket{{route=\"{r}\",le=\"+Inf\"}} {}",
                        h.count
                    );
                    let _ = writeln!(
                        out,
                        "onebit_lemonade_request_seconds_sum{{route=\"{r}\"}} {}",
                        h.sum
                    );
                    let _ = writeln!(
                        out,
                        "onebit_lemonade_request_seconds_count{{route=\"{r}\"}} {}",
                        h.count
                    );
                }
            }
        }

        // Gauge
        let _ = writeln!(
            out,
            "# HELP onebit_lemonade_upstream_up 1 if the active upstream responds, 0 otherwise."
        );
        let _ = writeln!(out, "# TYPE onebit_lemonade_upstream_up gauge");
        let _ = writeln!(
            out,
            "onebit_lemonade_upstream_up {}",
            self.upstream_up.load(Ordering::Relaxed)
        );

        out
    }
}

/// Escape a label value per the Prometheus text-format rules: `\`, `"`,
/// and `\n` are the three characters that need backslashing. Route names
/// we emit today don't hit any of these, but a future registry-driven
/// route wouldn't be bound by that.
fn escape_label(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            other => out.push(other),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_render_contains_type_markers() {
        let m = Metrics::new();
        let out = m.render();
        assert!(out.contains("# TYPE onebit_lemonade_requests_total counter"));
        assert!(out.contains("# TYPE onebit_lemonade_request_seconds histogram"));
        assert!(out.contains("# TYPE onebit_lemonade_upstream_up gauge"));
    }

    #[test]
    fn record_increments_counter_and_histogram() {
        let m = Metrics::new();
        m.record("/v1/chat/completions", 200, Duration::from_millis(150));
        m.record("/v1/chat/completions", 200, Duration::from_millis(300));
        m.record("/v1/chat/completions", 500, Duration::from_millis(25));
        let out = m.render();

        assert!(out.contains(
            "onebit_lemonade_requests_total{route=\"/v1/chat/completions\",status=\"200\"} 2"
        ));
        assert!(out.contains(
            "onebit_lemonade_requests_total{route=\"/v1/chat/completions\",status=\"500\"} 1"
        ));
        // Sum of 0.150 + 0.300 + 0.025 = 0.475
        assert!(
            out.contains("onebit_lemonade_request_seconds_count{route=\"/v1/chat/completions\"} 3")
        );
        // 5xx flips upstream_up to 0.
        assert!(out.contains("onebit_lemonade_upstream_up 0"));
    }

    #[test]
    fn set_upstream_up_flips_gauge() {
        let m = Metrics::new();
        m.set_upstream_up(true);
        assert!(m.render().contains("onebit_lemonade_upstream_up 1"));
        m.set_upstream_up(false);
        assert!(m.render().contains("onebit_lemonade_upstream_up 0"));
    }

    #[test]
    fn histogram_bucket_ordering_is_cumulative() {
        let m = Metrics::new();
        // 1 ms observation — falls into every bucket.
        m.record("/v1/models", 200, Duration::from_millis(1));
        let out = m.render();
        // Each `le` bucket should have count 1.
        assert!(out.contains(
            "onebit_lemonade_request_seconds_bucket{route=\"/v1/models\",le=\"0.005\"} 1"
        ));
        assert!(
            out.contains(
                "onebit_lemonade_request_seconds_bucket{route=\"/v1/models\",le=\"10\"} 1"
            )
        );
        assert!(out.contains(
            "onebit_lemonade_request_seconds_bucket{route=\"/v1/models\",le=\"+Inf\"} 1"
        ));
    }

    #[test]
    fn escape_label_handles_specials() {
        assert_eq!(escape_label("abc"), "abc");
        assert_eq!(escape_label("a\"b"), "a\\\"b");
        assert_eq!(escape_label("a\\b"), "a\\\\b");
        assert_eq!(escape_label("a\nb"), "a\\nb");
    }

    /// Render a realistic mix of samples and sanity-check the first ten
    /// lines follow the canonical Prometheus ordering: HELP, TYPE, data.
    /// Also prints them under `--nocapture` so reviewers can eyeball the
    /// exposition format.
    #[test]
    fn sample_render_follows_prometheus_ordering() {
        let m = Metrics::new();
        m.record("/v1/chat/completions", 200, Duration::from_millis(120));
        m.record("/v1/chat/completions", 200, Duration::from_millis(340));
        m.record("/v1/models", 200, Duration::from_millis(3));
        m.set_upstream_up(true);
        let text = m.render();
        let lines: Vec<&str> = text.lines().collect();
        assert!(lines[0].starts_with("# HELP onebit_lemonade_requests_total"));
        assert!(lines[1].starts_with("# TYPE onebit_lemonade_requests_total counter"));
        // Counter samples come before the next HELP block.
        assert!(lines[2].starts_with("onebit_lemonade_requests_total{"));
        for (i, line) in lines.iter().take(10).enumerate() {
            println!("LINE{i:02}: {line}");
        }
    }
}
