// halo — strix-ai-rs unified CLI.
//
// Rust port of bin/halo (bash). First crate in the halo-workspace to prove
// the cargo workspace + tokio + clap + reqwest stack before we touch kernels
// or agents.

use anyhow::Result;
use clap::{Parser, Subcommand};

mod status;
mod doctor;
mod logs;
mod restart;
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
    }
}
