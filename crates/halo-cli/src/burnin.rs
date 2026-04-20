// `halo burnin` — parity analyzer for the shadow-burnin JSONL log.
//
// Strictly read-only: no side effects, no network, no file writes. Points at
// `~/claude output/shadow-burnin.jsonl` by default (quote the space — the
// folder name is literal) and answers four questions:
//
//   halo burnin stats         overall byte-exact pct + counts
//   halo burnin drift         top-N divergence patterns (prompt → v1/v2 delta)
//   halo burnin recent -n N   last N entries with pass/fail glyph
//   halo burnin since <ts>    slice by timestamp (ISO-8601 lexicographic)
//
// With no subcommand `halo burnin` prints a one-line summary and exits 0
// if the byte-exact rate is ≥ 95%, 1 otherwise. That exit code is what the
// launch-readiness script and `halo doctor` consume.
//
// The on-disk schema (one JSON per line; fields observed in strix-burnin
// service output as of 2026-04-20) is:
//
//   { ts, prompt_idx, prompt_snippet, prefix_match_chars, full_match,
//     v1_ms, v2_ms, v1_text, v2_text, v1_tokens, v2_tokens }
//
// `prefix_match_chars` doubles as the first-byte-of-divergence offset: the
// number of leading characters that matched before v1 and v2 diverged.

use anyhow::{Context, Result, bail};
use clap::Subcommand;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

/// Default log path — the space in `claude output` is real.
const DEFAULT_LOG_REL: &str = "claude output/shadow-burnin.jsonl";

/// Pass/fail threshold the zero-subcommand entrypoint uses. Set at 95%
/// because the bar for the launch window is "≥ 95% byte-exact".
const PASS_THRESHOLD_PCT: f64 = 95.0;

// ---------------------------------------------------------------------------
// Subcommand surface

#[derive(Subcommand, Debug)]
pub enum BurninCmd {
    /// Overall byte-exact rate + sample counts.
    Stats {
        /// Path to shadow-burnin JSONL (default: ~/claude output/shadow-burnin.jsonl).
        #[arg(long)]
        log: Option<PathBuf>,
    },
    /// Top-N prompts by number of divergent rounds.
    Drift {
        #[arg(long)]
        log: Option<PathBuf>,
        /// How many prompt buckets to show.
        #[arg(long, default_value_t = 10)]
        top: usize,
    },
    /// Last N entries with a pass/fail glyph.
    Recent {
        #[arg(long)]
        log: Option<PathBuf>,
        /// Tail window.
        #[arg(long, short = 'n', default_value_t = 20)]
        tail: usize,
    },
    /// Only include entries with ts >= <timestamp> (ISO-8601 lex compare).
    Since {
        /// Cutoff timestamp, e.g. `2026-04-20T05:00:00Z`.
        timestamp: String,
        #[arg(long)]
        log: Option<PathBuf>,
    },
}

// ---------------------------------------------------------------------------
// JSONL row

#[derive(Debug, Deserialize, Clone)]
pub struct Row {
    pub ts: String,
    pub prompt_idx: u32,
    #[serde(default)]
    pub prompt_snippet: String,
    /// First-byte-of-divergence offset (characters).
    #[serde(default)]
    pub prefix_match_chars: u32,
    pub full_match: bool,
    #[serde(default)]
    pub v1_ms: u64,
    #[serde(default)]
    pub v2_ms: u64,
    #[serde(default)]
    pub v1_text: String,
    #[serde(default)]
    pub v2_text: String,
}

// ---------------------------------------------------------------------------
// Loader

/// Default log location: `$HOME/claude output/shadow-burnin.jsonl`.
fn default_log_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(DEFAULT_LOG_REL)
}

/// Resolve `--log` override or fall back to the default.
fn resolve_log(explicit: Option<PathBuf>) -> PathBuf {
    explicit.unwrap_or_else(default_log_path)
}

/// Parse every line of a JSONL file into `Row`s. Non-JSON / partial lines are
/// skipped silently (the service may still be mid-write on the last line).
pub fn load_rows(path: &Path) -> Result<Vec<Row>> {
    let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut rows = Vec::new();
    for line in BufReader::new(f).lines() {
        let line = line.with_context(|| format!("read {}", path.display()))?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Ok(r) = serde_json::from_str::<Row>(line) {
            rows.push(r);
        }
    }
    Ok(rows)
}

// ---------------------------------------------------------------------------
// Aggregates

/// Tally of pass/fail + mean latency over a slice of rows.
#[derive(Debug, Clone, Copy, Default)]
pub struct Stats {
    pub total: usize,
    pub pass: usize,
    pub fail: usize,
    pub pct: f64,
    pub mean_v1_ms: f64,
    pub mean_v2_ms: f64,
}

pub fn compute_stats(rows: &[Row]) -> Stats {
    let total = rows.len();
    let pass = rows.iter().filter(|r| r.full_match).count();
    let fail = total - pass;
    let pct = if total == 0 {
        0.0
    } else {
        100.0 * pass as f64 / total as f64
    };
    let (v1_sum, v2_sum): (u64, u64) = rows
        .iter()
        .fold((0, 0), |(a, b), r| (a + r.v1_ms, b + r.v2_ms));
    let mean_v1_ms = if total == 0 {
        0.0
    } else {
        v1_sum as f64 / total as f64
    };
    let mean_v2_ms = if total == 0 {
        0.0
    } else {
        v2_sum as f64 / total as f64
    };
    Stats {
        total,
        pass,
        fail,
        pct,
        mean_v1_ms,
        mean_v2_ms,
    }
}

/// One drift bucket: which prompt produced how many mismatches, what the
/// typical divergence offset was, and a sample v1 vs v2 pair.
#[derive(Debug, Clone)]
pub struct Drift {
    pub prompt_idx: u32,
    pub prompt_snippet: String,
    pub fail_count: usize,
    pub typical_offset: u32,
    pub sample_v1: String,
    pub sample_v2: String,
}

/// Group failures by prompt_idx, sort descending by fail count, take top N.
pub fn compute_drift(rows: &[Row], top: usize) -> Vec<Drift> {
    // Bucket failures by prompt_idx.
    let mut by_idx: HashMap<u32, Vec<&Row>> = HashMap::new();
    for r in rows.iter().filter(|r| !r.full_match) {
        by_idx.entry(r.prompt_idx).or_default().push(r);
    }
    let mut out: Vec<Drift> = by_idx
        .into_iter()
        .map(|(idx, bucket)| {
            // Most common offset (mode) across this prompt's failures.
            let mut offset_hist: HashMap<u32, usize> = HashMap::new();
            for r in &bucket {
                *offset_hist.entry(r.prefix_match_chars).or_insert(0) += 1;
            }
            let typical_offset = offset_hist
                .into_iter()
                .max_by_key(|&(_, n)| n)
                .map(|(o, _)| o)
                .unwrap_or(0);
            let head = bucket[0];
            Drift {
                prompt_idx: idx,
                prompt_snippet: head.prompt_snippet.clone(),
                fail_count: bucket.len(),
                typical_offset,
                sample_v1: truncate(&head.v1_text, 60),
                sample_v2: truncate(&head.v2_text, 60),
            }
        })
        .collect();
    // Primary: most failures. Tiebreak: lower prompt_idx for stable output.
    out.sort_by(|a, b| {
        b.fail_count
            .cmp(&a.fail_count)
            .then(a.prompt_idx.cmp(&b.prompt_idx))
    });
    out.truncate(top);
    out
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(n).collect();
        out.push('…');
        out
    }
}

/// Last `n` rows.
pub fn tail_rows(rows: &[Row], n: usize) -> Vec<Row> {
    let start = rows.len().saturating_sub(n);
    rows[start..].to_vec()
}

/// Rows whose `ts` is lexicographically >= cutoff. ISO-8601 in Z form is
/// lex-sortable so we don't need chrono here.
pub fn filter_since(rows: &[Row], cutoff: &str) -> Vec<Row> {
    rows.iter()
        .filter(|r| r.ts.as_str() >= cutoff)
        .cloned()
        .collect()
}

// ---------------------------------------------------------------------------
// Entry point

pub async fn run(subcmd: Option<BurninCmd>) -> Result<()> {
    match subcmd {
        None => run_default().await,
        Some(BurninCmd::Stats { log }) => run_stats(resolve_log(log)),
        Some(BurninCmd::Drift { log, top }) => run_drift(resolve_log(log), top),
        Some(BurninCmd::Recent { log, tail }) => run_recent(resolve_log(log), tail),
        Some(BurninCmd::Since { timestamp, log }) => run_since(resolve_log(log), &timestamp),
    }
}

async fn run_default() -> Result<()> {
    let path = default_log_path();
    if !path.exists() {
        bail!(
            "shadow-burnin log missing at {} — is strix-burnin.service running?",
            path.display()
        );
    }
    let rows = load_rows(&path)?;
    let s = compute_stats(&rows);
    if s.total == 0 {
        println!("shadow-burnin: 0 rounds logged at {}", path.display());
        std::process::exit(1);
    }
    println!(
        "shadow-burnin: {}/{} byte-exact = {:.2}% (threshold {:.0}%)",
        s.pass, s.total, s.pct, PASS_THRESHOLD_PCT
    );
    if s.pct >= PASS_THRESHOLD_PCT {
        Ok(())
    } else {
        std::process::exit(1);
    }
}

fn run_stats(path: PathBuf) -> Result<()> {
    let rows = load_rows(&path)?;
    let s = compute_stats(&rows);
    println!("log:            {}", path.display());
    println!("rounds:         {}", s.total);
    println!("byte-exact:     {} ({:.2}%)", s.pass, s.pct);
    println!("divergent:      {} ({:.2}%)", s.fail, 100.0 - s.pct);
    println!("mean v1 ms:     {:.1}", s.mean_v1_ms);
    println!("mean v2 ms:     {:.1}", s.mean_v2_ms);
    if let (Some(first), Some(last)) = (rows.first(), rows.last()) {
        println!("window:         {} → {}", first.ts, last.ts);
    }
    Ok(())
}

fn run_drift(path: PathBuf, top: usize) -> Result<()> {
    let rows = load_rows(&path)?;
    let drift = compute_drift(&rows, top);
    if drift.is_empty() {
        println!(
            "no divergent rounds in {} — nothing to report",
            path.display()
        );
        return Ok(());
    }
    let total_fail: usize = rows.iter().filter(|r| !r.full_match).count();
    println!(
        "top {} drift buckets  ({} total divergent rounds)",
        drift.len(),
        total_fail
    );
    println!();
    for (rank, d) in drift.iter().enumerate() {
        let pct_of_fail = 100.0 * d.fail_count as f64 / total_fail.max(1) as f64;
        println!(
            "{:>2}. idx={:<3}  fails={:<5} ({:>5.1}% of mismatches)  offset={}",
            rank + 1,
            d.prompt_idx,
            d.fail_count,
            pct_of_fail,
            d.typical_offset
        );
        println!("    prompt: {}", d.prompt_snippet);
        println!("    v1:     {}", d.sample_v1);
        println!("    v2:     {}", d.sample_v2);
    }
    Ok(())
}

fn run_recent(path: PathBuf, tail: usize) -> Result<()> {
    let rows = load_rows(&path)?;
    let rows = tail_rows(&rows, tail);
    for r in &rows {
        let glyph = if r.full_match { "✓" } else { "✗" };
        println!(
            "{} {} idx={:<3} v1={:>4}ms v2={:>4}ms  {}",
            glyph,
            r.ts,
            r.prompt_idx,
            r.v1_ms,
            r.v2_ms,
            truncate(&r.prompt_snippet, 48)
        );
    }
    Ok(())
}

fn run_since(path: PathBuf, cutoff: &str) -> Result<()> {
    let rows = load_rows(&path)?;
    let rows = filter_since(&rows, cutoff);
    let s = compute_stats(&rows);
    println!("since:          {}", cutoff);
    println!("rounds:         {}", s.total);
    println!("byte-exact:     {} ({:.2}%)", s.pass, s.pct);
    println!("divergent:      {} ({:.2}%)", s.fail, 100.0 - s.pct);
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
//
// Every test writes a synthetic JSONL to a tempfile and runs the pure
// analyzer functions (compute_stats / compute_drift / tail_rows /
// filter_since) plus load_rows. We deliberately skip the `run_*` entry
// points that call `std::process::exit` — the pure layer is what we
// actually want to lock down.

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// Write `lines` to a tempfile and return it. Caller owns the handle so
    /// the file lives at least as long as the test.
    fn tmp_jsonl(lines: &[&str]) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        for l in lines {
            writeln!(f, "{l}").unwrap();
        }
        f.flush().unwrap();
        f
    }

    /// 10-row fixture: 7 pass + 3 fail, with varied prompt_idx for drift.
    /// Timestamps strictly monotonic so `since` tests cleanly slice.
    fn fixture_10() -> NamedTempFile {
        let mk = |i: u32,
                  ts: &str,
                  idx: u32,
                  pass: bool,
                  offset: u32,
                  snippet: &str,
                  v1: &str,
                  v2: &str| {
            format!(
                r#"{{"ts":"{ts}","prompt_idx":{idx},"prompt_snippet":"{snippet}","prefix_match_chars":{offset},"full_match":{pass},"v1_ms":{},"v2_ms":{},"v1_text":"{v1}","v2_text":"{v2}","v1_tokens":1,"v2_tokens":1}}"#,
                100 + i,
                100 + i
            )
        };
        let lines: Vec<String> = vec![
            mk(
                0,
                "2026-04-20T00:00:00Z",
                0,
                true,
                10,
                "capital of France",
                " Paris.",
                " Paris.",
            ),
            mk(1, "2026-04-20T00:10:00Z", 1, true, 12, "2+2", " 4.", " 4."),
            mk(
                2,
                "2026-04-20T01:00:00Z",
                7,
                false,
                0,
                "gold symbol",
                "1",
                "0",
            ),
            mk(
                3,
                "2026-04-20T02:00:00Z",
                7,
                false,
                0,
                "gold symbol",
                "1",
                "0",
            ),
            mk(
                4,
                "2026-04-20T03:00:00Z",
                2,
                true,
                20,
                "planets",
                " Jupiter",
                "Jupiter ",
            ),
            mk(
                5,
                "2026-04-20T04:00:00Z",
                7,
                false,
                0,
                "gold symbol",
                "1",
                "0",
            ),
            mk(
                6,
                "2026-04-20T05:00:00Z",
                3,
                true,
                5,
                "Hamlet",
                " wrote",
                " wrote",
            ),
            mk(
                7,
                "2026-04-20T06:00:00Z",
                4,
                true,
                8,
                "horror",
                " silent",
                " silent",
            ),
            mk(
                8,
                "2026-04-20T07:00:00Z",
                5,
                true,
                3,
                "clouds",
                " drift",
                " drift",
            ),
            mk(
                9,
                "2026-04-20T08:00:00Z",
                6,
                true,
                9,
                "poem",
                " soft",
                " soft",
            ),
        ];
        let refs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();
        tmp_jsonl(&refs)
    }

    #[test]
    fn stats_on_10_line_fixture_reports_70pct() {
        let f = fixture_10();
        let rows = load_rows(f.path()).unwrap();
        assert_eq!(rows.len(), 10, "all 10 rows must parse");
        let s = compute_stats(&rows);
        assert_eq!(s.total, 10);
        assert_eq!(s.pass, 7);
        assert_eq!(s.fail, 3);
        assert!((s.pct - 70.0).abs() < 1e-9, "expected 70.0%, got {}", s.pct);
    }

    #[test]
    fn drift_returns_non_empty_ranking_with_top_bucket_first() {
        let f = fixture_10();
        let rows = load_rows(f.path()).unwrap();
        let d = compute_drift(&rows, 10);
        assert!(
            !d.is_empty(),
            "drift must be non-empty on a fixture with failures"
        );
        // prompt_idx=7 fails 3×, should be rank 1.
        assert_eq!(d[0].prompt_idx, 7);
        assert_eq!(d[0].fail_count, 3);
        // Typical offset across the three 'gold symbol' failures is 0.
        assert_eq!(d[0].typical_offset, 0);
    }

    #[test]
    fn drift_top_cap_truncates() {
        let f = fixture_10();
        let rows = load_rows(f.path()).unwrap();
        let d = compute_drift(&rows, 1);
        assert_eq!(d.len(), 1);
    }

    #[test]
    fn recent_tail_3_returns_exactly_three_rows() {
        let f = fixture_10();
        let rows = load_rows(f.path()).unwrap();
        let r = tail_rows(&rows, 3);
        assert_eq!(r.len(), 3);
        // Must be the last three timestamps from the fixture.
        assert_eq!(r[0].ts, "2026-04-20T06:00:00Z");
        assert_eq!(r[2].ts, "2026-04-20T08:00:00Z");
    }

    #[test]
    fn since_filter_cuts_rows_by_timestamp() {
        let f = fixture_10();
        let rows = load_rows(f.path()).unwrap();
        // Cutoff in the middle of the fixture (05:00:00Z is row index 6).
        let sliced = filter_since(&rows, "2026-04-20T05:00:00Z");
        assert_eq!(sliced.len(), 4, "expected rows 6..=9 to remain");
        for r in &sliced {
            assert!(r.ts.as_str() >= "2026-04-20T05:00:00Z");
        }
        // Boundary: cutoff past the last row → empty.
        let empty = filter_since(&rows, "2027-01-01T00:00:00Z");
        assert!(empty.is_empty());
        // Boundary: cutoff before the first row → all rows.
        let all = filter_since(&rows, "2020-01-01T00:00:00Z");
        assert_eq!(all.len(), rows.len());
    }

    #[test]
    fn load_rows_tolerates_blank_and_malformed_lines() {
        let f = tmp_jsonl(&[
            r#"{"ts":"2026-04-20T00:00:00Z","prompt_idx":0,"prompt_snippet":"ok","prefix_match_chars":1,"full_match":true,"v1_ms":1,"v2_ms":1,"v1_text":"a","v2_text":"a","v1_tokens":1,"v2_tokens":1}"#,
            "",
            "not json at all",
            r#"{"ts":"2026-04-20T00:01:00Z","prompt_idx":1,"prompt_snippet":"also ok","prefix_match_chars":0,"full_match":false,"v1_ms":1,"v2_ms":1,"v1_text":"x","v2_text":"y","v1_tokens":1,"v2_tokens":1}"#,
        ]);
        let rows = load_rows(f.path()).unwrap();
        assert_eq!(rows.len(), 2, "blank + malformed lines must be skipped");
        let s = compute_stats(&rows);
        assert_eq!(s.pass, 1);
        assert_eq!(s.fail, 1);
    }
}
