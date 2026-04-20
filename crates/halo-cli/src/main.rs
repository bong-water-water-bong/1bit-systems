// halo — strix-ai-rs unified CLI.
//
// Rust port of bin/halo (bash). First crate in the halo-workspace to prove
// the cargo workspace + tokio + clap + reqwest stack before we touch kernels
// or agents.

use anyhow::Result;
use clap::{Parser, Subcommand};

mod bench;
mod chat;
mod status;
mod doctor;
mod install;
mod logs;
mod ppl;
mod restart;
mod say;
mod update;
mod version;

/// halo — one command, all halo-ai ops
#[derive(Parser, Debug)]
#[command(name = "halo", about, version, propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// One-line-per-service state snapshot
    Status,
    /// Tail systemd journal for a halo service
    Logs {
        /// Service short name (bitnet, sd, whisper, kokoro, lemonade, agent, ...)
        service: String,
        /// Follow
        #[arg(short = 'f', long)]
        follow: bool,
        /// Last N lines
        #[arg(short = 'n', long, default_value_t = 50)]
        lines: u32,
    },
    /// Restart a halo service
    Restart { service: String },
    /// Comprehensive health check across the stack
    Doctor,
    /// Pull + rebuild + restart touched components
    Update {
        #[arg(long)]
        no_build: bool,
        #[arg(long)]
        no_restart: bool,
    },
    /// halo stack version + component SHAs
    Version,
    /// Install a component from packages.toml (core, agents, voice, sd, ...)
    Install {
        /// Component name; omit with --list to see the full catalogue
        component: Option<String>,
        /// List available components
        #[arg(long)]
        list: bool,
    },
    /// Speak text through halo-kokoro :8083 + the host's audio player
    Say {
        /// The text to synthesize. Quote it for multi-word input.
        text: Vec<String>,
        /// Voice id (see `curl :8083/voices` for the list)
        #[arg(short = 'v', long)]
        voice: Option<String>,
        /// Playback speed (0 < x ≤ 4)
        #[arg(short = 's', long, default_value_t = 1.0)]
        speed: f32,
    },
    /// Interactive one-shot REPL against halo-server :8180
    Chat {
        #[arg(long)]
        url: Option<String>,
        #[arg(long)]
        model: Option<String>,
        #[arg(long, default_value_t = 128)]
        max_tokens: u32,
    },
    /// Shadow-burnin summary (parity vs gen-1 C++ :8080)
    Bench {
        /// If set, run N new rounds before printing the summary
        #[arg(long)]
        rounds: Option<u32>,
        /// Only count rounds after this ISO-8601 timestamp
        #[arg(long)]
        since: Option<String>,
    },
    /// Perplexity against gen-1 wikitext baseline (9.1607)
    Ppl {
        #[arg(long)]
        url: Option<String>,
        #[arg(long, default_value_t = 1024)]
        stride: u32,
        #[arg(long, default_value_t = 1024)]
        max_tokens: u32,
        /// Bytes of wikitext to send (from the start of the file)
        #[arg(long, default_value_t = 6000)]
        bytes: usize,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| "halo=info".into()))
        .with_target(false)
        .init();

    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Status                         => status::run().await,
        Cmd::Logs { service, follow, lines } => logs::run(&service, follow, lines).await,
        Cmd::Restart { service }            => restart::run(&service).await,
        Cmd::Doctor                         => doctor::run().await,
        Cmd::Update { no_build, no_restart } => update::run(no_build, no_restart).await,
        Cmd::Version                        => version::run().await,
        Cmd::Install { component, list }    => {
            if list || component.is_none() { install::list().await }
            else                             { install::run_install(&component.unwrap()).await }
        }
        Cmd::Say { text, voice, speed }     => {
            let phrase = text.join(" ");
            let voice_owned: String = voice.unwrap_or_else(|| say::default_voice().into());
            say::run(&phrase, &voice_owned, speed).await
        }
        Cmd::Chat  { url, model, max_tokens }            => chat::run(url, model, max_tokens).await,
        Cmd::Bench { rounds, since }                     => bench::run(rounds, since).await,
        Cmd::Ppl   { url, stride, max_tokens, bytes }    => ppl::run(url, stride, max_tokens, bytes).await,
    }
}
