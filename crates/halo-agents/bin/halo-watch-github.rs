//! halo-watch-github — read-only GitHub poller for halo-agents.
//!
//! Walks the configured repo list (`HALO_GH_REPOS`, comma-separated
//! `owner/repo`) once per invocation, finds issues + PRs opened or updated
//! in the last `HALO_GH_POLL_SECONDS * 2` seconds, classifies each event,
//! and dispatches it to a specialist via the shared [`Registry`].
//!
//! Rule A discipline: this binary is the only authorized Rust process that
//! reaches out to `api.github.com`. It never writes — no issue comments,
//! no labels, no reactions. Read-only scope keeps the PAT harmless even if
//! it leaks.
//!
//! Designed to be called by a systemd oneshot + timer (see
//! `strixhalo/systemd/strix-watch-github.{service,timer}`). Loop behavior
//! is therefore "one pass and exit", not "tokio::select! forever"; the
//! timer handles cadence.

use anyhow::Result;
use halo_agents::Registry;
use halo_agents::watch::github::{
    Event, classify, poll_seconds_from_env, repos_from_env,
};
use serde_json::json;
use std::time::Duration;
use tracing::{info, warn};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false)
        .init();

    let repos = repos_from_env();
    let poll_seconds = poll_seconds_from_env();
    // "opened or updated in the last 2N seconds" — lookback is double the
    // poll interval so a unit that fires a bit late still catches the
    // events that arrived during the previous window.
    let lookback = Duration::from_secs(poll_seconds.saturating_mul(2));

    let token = std::env::var("GITHUB_TOKEN").ok().filter(|s| !s.is_empty());
    if token.is_none() {
        // Rule from the scope doc: no token → log + exit clean. We don't
        // want the systemd unit to churn against the 60 req/hr anon limit
        // when an operator simply hasn't provisioned a PAT yet.
        info!("no GITHUB_TOKEN set, skipping poll");
        return Ok(());
    }

    let mut octo_builder = octocrab::OctocrabBuilder::new();
    if let Some(t) = token {
        octo_builder = octo_builder.personal_token(t);
    }
    let octo = octo_builder.build()?;

    let registry = Registry::default_stubs();
    let since = now_minus(lookback);

    info!(
        repos = repos.len(),
        poll_seconds,
        lookback_seconds = lookback.as_secs(),
        "halo-watch-github starting one-shot poll"
    );

    let mut events_seen = 0usize;
    let mut events_routed = 0usize;

    for repo in &repos {
        let (owner, name) = match split_owner_repo(repo) {
            Some(pair) => pair,
            None => {
                warn!(repo, "invalid repo format, expected owner/repo");
                continue;
            }
        };

        match poll_repo(&octo, owner, name, &since).await {
            Ok(events) => {
                for ev in events {
                    events_seen += 1;
                    let target = classify(&ev);
                    let kind = if ev.is_pr { "pr" } else { "issue" };
                    // One-line summary to journal — this is the only line
                    // operators need to eyeball for correctness.
                    info!(
                        target = target.as_str(),
                        repo = ev.repo.as_str(),
                        kind,
                        author = ev.author.as_str(),
                        url = ev.url.as_str(),
                        "{}",
                        ev.title
                    );
                    let payload = json!({
                        "title":  ev.title,
                        "body":   ev.body,
                        "labels": ev.labels,
                        "author": ev.author,
                        "url":    ev.url,
                        "repo":   ev.repo,
                        "kind":   kind,
                    });
                    match registry.dispatch(target.as_str(), payload).await {
                        Ok(_) => events_routed += 1,
                        Err(e) => warn!(target = target.as_str(), error = %e, "dispatch failed"),
                    }
                }
            }
            Err(e) => {
                warn!(repo, error = %e, "repo poll failed");
            }
        }
    }

    info!(
        events_seen,
        events_routed, "halo-watch-github pass complete"
    );
    Ok(())
}

fn split_owner_repo(s: &str) -> Option<(&str, &str)> {
    let (owner, repo) = s.split_once('/')?;
    if owner.is_empty() || repo.is_empty() {
        return None;
    }
    Some((owner, repo))
}

/// Compute `now - lookback` as a UTC `DateTime<Utc>` suitable for
/// GitHub's `since` parameter. octocrab 0.39 expects `chrono::DateTime`
/// here; we rely on its transitive chrono pull to avoid adding a direct
/// dep for a one-liner.
fn now_minus(lookback: Duration) -> chrono::DateTime<chrono::Utc> {
    let now = std::time::SystemTime::now();
    let since = now - lookback;
    since.into()
}

async fn poll_repo(
    octo: &octocrab::Octocrab,
    owner: &str,
    name: &str,
    since: &chrono::DateTime<chrono::Utc>,
) -> Result<Vec<Event>> {
    // `state=all, since=...` returns both open and recently-closed issues;
    // we want updated-in-window, not just currently-open. GitHub folds PRs
    // into the issues listing — each issue with `pull_request.is_some()`
    // is a PR, which is how we classify without a second call.
    let page = octo
        .issues(owner, name)
        .list()
        .state(octocrab::params::State::All)
        .since(*since)
        .per_page(50)
        .send()
        .await?;

    let repo_tag = format!("{owner}/{name}");
    let mut out = Vec::with_capacity(page.items.len());
    for issue in page.items {
        let is_pr = issue.pull_request.is_some();
        let labels = issue
            .labels
            .into_iter()
            .map(|l| l.name)
            .collect::<Vec<_>>();
        out.push(Event {
            title: issue.title,
            body: issue.body.unwrap_or_default(),
            labels,
            author: issue.user.login,
            url: issue.html_url.to_string(),
            repo: repo_tag.clone(),
            is_pr,
        });
    }
    Ok(out)
}
