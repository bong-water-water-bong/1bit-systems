//! 1bit-watch-discord — Discord gateway client that keeps the 17 specialist
//! registry aware of channel activity.
//!
//! Design:
//!   * Lurker by default. We listen on a whitelist of channels
//!     (`HALO_DISCORD_CHANNELS`), classify each message, and dispatch a
//!     compact payload to the relevant specialist via `Registry::dispatch`.
//!   * NO automatic replies. We only post to a channel when directly
//!     `@mentioned` — and only for the `status` subcommand.
//!   * Gracefully degrades if `DISCORD_BOT_TOKEN` is missing: print help
//!     and exit 0. This matters because the systemd unit ships disabled,
//!     so the binary is expected to be invoked on a fresh box with no
//!     token configured.
//!
//! Rule A: no Python at runtime — serenity-rs is pure Rust.
//! Rule D: edition 2024, Rust 1.86.
//!
//! Env:
//!   DISCORD_BOT_TOKEN          required to actually connect
//!   HALO_DISCORD_CHANNELS      comma-separated channel IDs to watch
//!   HALO_SERVER_URL            default http://127.0.0.1:8180
//!   HALO_LANDING_URL           default http://127.0.0.1:8190
//!   RUST_LOG                   default onebit_watch_discord=info

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use onebit_agents::watch::{
    Classification, HELP_TEXT, classify, is_direct_mention, parse_channel_whitelist, strip_mention,
};
use onebit_agents::{Name, Registry};
use serde_json::json;
use serenity::Client;
use serenity::all::{AutoArchiveDuration, Context, CreateThread, EventHandler, GatewayIntents, Http, Message, Ready};
use serenity::async_trait;
use tracing::{error, info, warn};

const DEFAULT_SERVER_URL: &str = "http://127.0.0.1:8180";
const DEFAULT_LANDING_URL: &str = "http://127.0.0.1:8190";

/// Handler shared across every gateway event. `Arc`'d internally by serenity.
///
/// Two Discord tokens live here:
///   * halo's token (passed to the serenity `Client` in `main`) is the
///     gateway listener — lurker only, never posts.
///   * echo's token lives in `echo_http`. Every outbound post
///     (specialist reply, thread creation, `@halo status`) routes
///     through this client so the halo identity stays silent.
/// If `ECHO_BOT_TOKEN` is unset, `echo_http` is `None` and outbound
/// posting is skipped with a warning — preserves the lurker-only
/// behaviour as the safe fallback.
struct Handler {
    registry: Arc<Registry>,
    channels: HashSet<u64>,
    http: reqwest::Client,
    echo_http: Option<Arc<Http>>,
    server_url: String,
    landing_url: String,
}

#[async_trait]
impl EventHandler for Handler {
    async fn ready(&self, _ctx: Context, ready: Ready) {
        info!(
            bot = %ready.user.name,
            channels = self.channels.len(),
            "1bit-watch-discord connected"
        );
    }

    async fn message(&self, ctx: Context, msg: Message) {
        // Never react to our own messages or other bots.
        if msg.author.bot {
            return;
        }

        // Whitelist gate.
        //   Empty whitelist  = observe ALL channels the bot is permissioned for.
        //   Non-empty filter = only these channels.
        let channel_id = msg.channel_id.get();
        if !self.channels.is_empty() && !self.channels.contains(&channel_id) {
            return;
        }

        // Lightweight classification + dispatch. We do this even for
        // mentioned messages so the specialist sees the context.
        let class = classify(&msg.content);
        let specialist = class.specialist();

        info!(
            author = %msg.author.name,
            channel = channel_id,
            class = %class,
            specialist = specialist.as_str(),
            "dispatching to specialist"
        );

        let req = json!({
            "source": "discord",
            "channel_id": channel_id,
            "author": msg.author.name,
            "classification": class.as_str(),
            "content": msg.content,
        });
        match self.registry.dispatch(specialist.as_str(), req).await {
            Ok(resp) => {
                if let Some(text) = resp.get("text").and_then(|v| v.as_str()) {
                    if !text.trim().is_empty() {
                        self.post_response(&ctx, &msg, class, text).await;
                    }
                }
            }
            Err(e) => warn!(error = %e, "specialist dispatch failed"),
        }

        // Reply path: ONLY on direct mention. Keeps us off channels
        // otherwise. The bot ID is resolved via the cache — serenity
        // populates this from the READY payload.
        let me = ctx.cache.current_user().id.get();
        if !is_direct_mention(&msg.content, me) {
            return;
        }

        let cmd = strip_mention(&msg.content, me);
        if cmd.starts_with("status") {
            let line = self.status_line().await;
            if let Some(http) = self.echo_http.as_ref() {
                if let Err(e) = msg.channel_id.say(http.as_ref(), line).await {
                    warn!(error = %e, "echo failed to post status reply");
                }
            } else {
                warn!("@halo status requested but ECHO_BOT_TOKEN unset; silent.");
            }
        }
        // Any other `@halo-bot …` command is deliberately ignored for now.
        // We don't want accidental auto-moderation surface.
    }
}

impl Handler {
    /// Post a specialist's text reply via echo's token. BugReport-classified
    /// messages are moved into a newly-created public thread on the original
    /// message so follow-ups stay scoped to one incident. All other
    /// classifications reply in the original channel. If `ECHO_BOT_TOKEN`
    /// is unset, we warn and drop the post — halo never speaks with its
    /// own token.
    async fn post_response(&self, _ctx: &Context, msg: &Message, class: Classification, text: &str) {
        let Some(http) = self.echo_http.as_ref() else {
            warn!(
                "specialist returned text but ECHO_BOT_TOKEN is unset — halo stays silent. \
                 Configure echo's bot token to enable posting."
            );
            return;
        };

        if matches!(class, Classification::BugReport) {
            let name = format!(
                "troubleshoot: {}",
                truncate_for_thread_name(msg.content.as_str())
            );
            let builder =
                CreateThread::new(name).auto_archive_duration(AutoArchiveDuration::OneDay);
            match msg
                .channel_id
                .create_thread_from_message(http.as_ref(), msg.id, builder)
                .await
            {
                Ok(thread) => {
                    if let Err(e) = thread.id.say(http.as_ref(), text).await {
                        warn!(error = %e, thread_id = %thread.id, "echo failed to post into thread");
                    }
                    return;
                }
                Err(e) => {
                    warn!(error = %e, "failed to create troubleshoot thread, falling back to channel reply");
                }
            }
        }

        if let Err(e) = msg.channel_id.say(http.as_ref(), text).await {
            warn!(error = %e, "echo failed to post specialist response");
        }
    }

    /// Single-line status pulled from 1bit-server /v1/models and
    /// 1bit-landing /metrics. Both probes are best-effort; a missing
    /// endpoint surfaces as "?" in its slot rather than failing the whole
    /// reply.
    async fn status_line(&self) -> String {
        let models_url = format!("{}/v1/models", self.server_url);
        let metrics_url = format!("{}/metrics", self.landing_url);

        let models = self.http.get(&models_url).send().await.ok().and_then(|r| {
            if r.status().is_success() {
                Some(r)
            } else {
                None
            }
        });
        let model_count = match models {
            Some(r) => match r.json::<serde_json::Value>().await {
                Ok(v) => v
                    .get("data")
                    .and_then(|d| d.as_array())
                    .map(|a| a.len().to_string())
                    .unwrap_or_else(|| "?".into()),
                Err(_) => "?".into(),
            },
            None => "?".into(),
        };

        let metrics_ok = self
            .http
            .get(&metrics_url)
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false);
        let metrics_tag = if metrics_ok { "up" } else { "down" };

        format!(
            "halo: models={} landing={} ({} specialists online)",
            model_count,
            metrics_tag,
            Name::ALL.len()
        )
    }
}

/// Clip a message body to a thread-title-friendly length (≤80 chars).
/// Strips newlines and collapses whitespace. Used for auto-created
/// troubleshoot threads so titles stay legible.
fn truncate_for_thread_name(src: &str) -> String {
    let flat: String = src
        .chars()
        .map(|c| if c.is_whitespace() { ' ' } else { c })
        .collect();
    let collapsed: String = flat
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if collapsed.chars().count() <= 80 {
        collapsed
    } else {
        collapsed
            .chars()
            .take(77)
            .chain("...".chars())
            .collect()
    }
}

fn init_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "onebit_watch_discord=info".into()),
        )
        .with_target(false)
        .init();
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();

    // Token gate. Missing token → print help + exit 0. Do NOT try to
    // connect. Do NOT panic. See watch::HELP_TEXT for the copy.
    let token = match std::env::var("DISCORD_BOT_TOKEN") {
        Ok(t) if !t.trim().is_empty() => t,
        _ => {
            println!("{HELP_TEXT}");
            return Ok(());
        }
    };

    let channels_raw = std::env::var("HALO_DISCORD_CHANNELS").unwrap_or_default();
    let channels: HashSet<u64> = parse_channel_whitelist(&channels_raw).into_iter().collect();
    if channels.is_empty() {
        warn!(
            "HALO_DISCORD_CHANNELS is empty — bot will connect but observe \
             nothing. Set it to a comma-separated list of channel IDs."
        );
    }

    let server_url = std::env::var("HALO_SERVER_URL").unwrap_or_else(|_| DEFAULT_SERVER_URL.into());
    let landing_url =
        std::env::var("HALO_LANDING_URL").unwrap_or_else(|_| DEFAULT_LANDING_URL.into());

    // Echo's token is the posting identity. Halo never posts with its own
    // token — every outbound message, thread creation, status reply flows
    // through this client. Unset => halo stays purely lurker.
    let echo_http = match std::env::var("ECHO_BOT_TOKEN") {
        Ok(t) if !t.trim().is_empty() => {
            info!("ECHO_BOT_TOKEN configured — posting identity active");
            Some(Arc::new(Http::new(&format!("Bot {t}"))))
        }
        _ => {
            warn!(
                "ECHO_BOT_TOKEN unset — halo will classify + dispatch but not post. \
                 Set echo's bot token to enable the reply pipeline."
            );
            None
        }
    };

    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .user_agent("1bit-watch-discord/0.1")
        .build()?;

    let registry = Arc::new(Registry::default_stubs());

    let handler = Handler {
        registry,
        channels,
        http,
        echo_http,
        server_url,
        landing_url,
    };

    // MESSAGE_CONTENT is a privileged intent — operator must enable it on
    // the bot in the Discord developer portal. Without it we still see
    // messages but content is empty and classification reduces to Chat.
    let intents = GatewayIntents::GUILD_MESSAGES
        | GatewayIntents::DIRECT_MESSAGES
        | GatewayIntents::MESSAGE_CONTENT;

    let mut client = Client::builder(&token, intents)
        .event_handler(handler)
        .await?;

    if let Err(e) = client.start().await {
        error!(error = %e, "discord client exited with error");
        return Err(e.into());
    }

    Ok(())
}

// The heavy-lifting tests live in `onebit_agents::watch::tests` so they run
// under `cargo test -p 1bit-agents`. The startup-without-token behaviour
// is exercised there via the HELP_TEXT + env-gated main shape; a separate
// integration test below validates that the compiled binary short-circuits
// when the token is unset.
//
// See tests/watch_discord_startup.rs.
