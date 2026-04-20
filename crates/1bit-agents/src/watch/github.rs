//! GitHub watcher plumbing used by the `1bit-watch-github` binary.
//!
//! Pure, read-only classification + config parsing. The binary wraps these
//! helpers around an `octocrab` client; keeping the logic here means the
//! binary is thin and the classifier is unit-testable without a live GitHub
//! token.

use crate::Name;

/// Default repository list if `HALO_GH_REPOS` is unset.
///
/// Kept as a plain `&[&str]` so the parser is exercised in tests by the
/// same codepath the binary uses at startup.
pub const DEFAULT_REPOS: &[&str] = &[
    "bong-water-water-bong/1bit-systems",
    "bong-water-water-bong/bitnet-mlx.rs",
    "strix-ai-rs/halo-workspace",
];

/// Default poll interval in seconds, if `HALO_GH_POLL_SECONDS` is unset.
pub const DEFAULT_POLL_SECONDS: u64 = 300;

/// Lightweight view of a GitHub issue or PR — enough to classify and
/// route. Built by the binary from the `octocrab` response, fed into
/// [`classify`]. Keeping it string-y lets tests build fixtures without
/// wrestling with the `octocrab` types.
#[derive(Debug, Clone)]
pub struct Event {
    pub title: String,
    pub body: String,
    pub labels: Vec<String>,
    pub author: String,
    pub url: String,
    pub repo: String,
    /// `true` for PRs, `false` for issues. PRs always route to Magistrate
    /// regardless of labels.
    pub is_pr: bool,
}

/// Classify a GitHub event to a specialist.
///
/// Priority (PR > bug > feature > docs > sentinel fallback):
///
/// * Any PR → [`Name::Magistrate`]. Code review is Magistrate's brief.
/// * Labels `bug` OR title contains `error` / `crash` / `fail` (case-
///   insensitive) → [`Name::Sentinel`]. Incidents go to the watchdog.
/// * Labels `enhancement` or `feature` → [`Name::Planner`]. Roadmap work.
/// * Label `documentation` → [`Name::Scribe`].
/// * Anything else → [`Name::Sentinel`] as a safe default — the watchdog
///   is already the on-call specialist, and unrouted issues should still
///   surface somewhere.
pub fn classify(event: &Event) -> Name {
    if event.is_pr {
        return Name::Magistrate;
    }

    let labels_lc: Vec<String> = event.labels.iter().map(|l| l.to_lowercase()).collect();
    let has_label = |needle: &str| labels_lc.iter().any(|l| l == needle);

    let title_lc = event.title.to_lowercase();
    let title_has_fault = ["error", "crash", "fail"]
        .iter()
        .any(|needle| title_lc.contains(needle));

    if has_label("bug") || title_has_fault {
        return Name::Sentinel;
    }
    if has_label("enhancement") || has_label("feature") {
        return Name::Planner;
    }
    if has_label("documentation") {
        return Name::Scribe;
    }
    Name::Sentinel
}

/// Parse a comma-separated repo list. Trims whitespace around each entry,
/// drops empties (trailing commas, doubled commas, all-whitespace chunks),
/// and accepts the input as-is otherwise. No validation of `owner/repo`
/// shape — octocrab will reject bad inputs at call time.
pub fn parse_repos(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

/// Read `HALO_GH_REPOS` or fall back to [`DEFAULT_REPOS`]. Empty / all-
/// whitespace env values fall through to the default, so operators can't
/// accidentally silence the watcher by setting `HALO_GH_REPOS=""`.
pub fn repos_from_env() -> Vec<String> {
    match std::env::var("HALO_GH_REPOS") {
        Ok(s) => {
            let parsed = parse_repos(&s);
            if parsed.is_empty() {
                DEFAULT_REPOS.iter().map(|s| s.to_string()).collect()
            } else {
                parsed
            }
        }
        Err(_) => DEFAULT_REPOS.iter().map(|s| s.to_string()).collect(),
    }
}

/// Read `HALO_GH_POLL_SECONDS` or fall back to [`DEFAULT_POLL_SECONDS`].
/// Invalid / zero values fall through to the default — a zero poll would
/// hammer GitHub and get us rate-limited.
pub fn poll_seconds_from_env() -> u64 {
    match std::env::var("HALO_GH_POLL_SECONDS") {
        Ok(s) => s
            .trim()
            .parse::<u64>()
            .ok()
            .filter(|&n| n > 0)
            .unwrap_or(DEFAULT_POLL_SECONDS),
        Err(_) => DEFAULT_POLL_SECONDS,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(title: &str, labels: &[&str], is_pr: bool) -> Event {
        Event {
            title: title.to_string(),
            body: String::new(),
            labels: labels.iter().map(|s| s.to_string()).collect(),
            author: "tester".to_string(),
            url: "https://example.invalid/1".to_string(),
            repo: "owner/repo".to_string(),
            is_pr,
        }
    }

    #[test]
    fn classify_bug_label_routes_to_sentinel() {
        assert_eq!(
            classify(&ev("weird thing", &["bug"], false)),
            Name::Sentinel
        );
    }

    #[test]
    fn classify_crash_title_routes_to_sentinel_without_label() {
        assert_eq!(
            classify(&ev("server crash on startup", &[], false)),
            Name::Sentinel
        );
    }

    #[test]
    fn classify_error_title_case_insensitive() {
        // Title casing shouldn't matter. "ERROR" should trigger Sentinel.
        assert_eq!(
            classify(&ev("ERROR: segfault in decode", &[], false)),
            Name::Sentinel
        );
    }

    #[test]
    fn classify_enhancement_routes_to_planner() {
        assert_eq!(
            classify(&ev("add streaming API", &["enhancement"], false)),
            Name::Planner
        );
    }

    #[test]
    fn classify_feature_label_routes_to_planner() {
        assert_eq!(
            classify(&ev("new thing", &["feature"], false)),
            Name::Planner
        );
    }

    #[test]
    fn classify_documentation_routes_to_scribe() {
        assert_eq!(
            classify(&ev("fix typo in README", &["documentation"], false)),
            Name::Scribe
        );
    }

    #[test]
    fn classify_pr_always_routes_to_magistrate() {
        // Even if the PR is labeled `bug`, PRs go to Magistrate for review.
        assert_eq!(
            classify(&ev("fix: segfault", &["bug"], true)),
            Name::Magistrate
        );
        assert_eq!(
            classify(&ev("docs: typo", &["documentation"], true)),
            Name::Magistrate
        );
    }

    #[test]
    fn classify_unknown_falls_back_to_sentinel() {
        assert_eq!(classify(&ev("random question", &[], false)), Name::Sentinel);
    }

    #[test]
    fn parse_repos_handles_whitespace_and_trailing_commas() {
        // Covers: leading/trailing whitespace, trailing comma, doubled
        // comma, pure-whitespace chunk. All should drop out cleanly.
        let raw = "  foo/bar , baz/qux,, ,trailing/comma,";
        let got = parse_repos(raw);
        assert_eq!(got, vec!["foo/bar", "baz/qux", "trailing/comma"]);
    }

    #[test]
    fn parse_repos_empty_string_returns_empty_vec() {
        assert!(parse_repos("").is_empty());
        assert!(parse_repos("   ").is_empty());
        assert!(parse_repos(",,,").is_empty());
    }

    #[test]
    fn poll_seconds_default_when_unset() {
        // SAFETY: single-threaded test access; we restore on the way out.
        // We don't rely on any prior value — just assert the default when
        // the variable is removed.
        // Serial guard is not necessary because tests within a single
        // binary run on separate threads but this test only removes + checks.
        // If we see flake on CI we can add a mutex.
        // SAFETY: the `set_var` / `remove_var` APIs are `unsafe` on edition
        // 2024 because they mutate process-global state; tests run in-
        // process but we only touch a variable unique to this suite.
        unsafe { std::env::remove_var("HALO_GH_POLL_SECONDS") };
        assert_eq!(poll_seconds_from_env(), 300);
        assert_eq!(poll_seconds_from_env(), DEFAULT_POLL_SECONDS);
    }

    #[test]
    fn poll_seconds_zero_falls_back_to_default() {
        // Zero is a footgun (would hammer the API); treat it like unset.
        unsafe { std::env::set_var("HALO_GH_POLL_SECONDS", "0") };
        assert_eq!(poll_seconds_from_env(), DEFAULT_POLL_SECONDS);
        unsafe { std::env::remove_var("HALO_GH_POLL_SECONDS") };
    }

    #[test]
    fn repos_default_list_has_three_entries() {
        // Tied to the default list defined above; bump this assertion
        // deliberately if we add canonical repos.
        assert_eq!(DEFAULT_REPOS.len(), 3);
        assert!(DEFAULT_REPOS.contains(&"strix-ai-rs/halo-workspace"));
    }
}
