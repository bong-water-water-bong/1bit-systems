//! 1bit-lemonade binary — OpenAI-compat model gateway.
//!
//! v0 scope: serve /v1/models from a TOML registry and /_health. Actual
//! chat/completions forwarding lands when 1bit-router's client API is
//! stable enough to call from a second process without re-loading weights.

use anyhow::{Context, Result};
use clap::Parser;
use onebit_lemonade::routes::{AppState, build};
use onebit_lemonade::{LemonadeConfig, ModelEntry, ModelRegistry};
use std::net::SocketAddr;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "1bit-lemonade", about, version)]
struct Cli {
    /// Bind address
    #[arg(long, default_value = "127.0.0.1:8200")]
    bind: SocketAddr,
    /// Upstream 1bit-server for chat/completions proxying (no trailing /).
    #[arg(
        long,
        default_value = "http://127.0.0.1:8180",
        env = "HALO_LEMONADE_UPSTREAM"
    )]
    upstream: String,
    /// Optional TOML config at this path (registry + upstream_fallback)
    #[arg(long)]
    config: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "onebit_lemonade=info".into()),
        )
        .with_target(false)
        .init();

    let cli = Cli::parse();

    let registry = if let Some(path) = cli.config.as_ref() {
        let _cfg = LemonadeConfig::load(path).context("loading config")?;
        // Registry path is read via cfg.registry_path; for v0 we inline the
        // default halo-1bit-2b so the endpoint returns something useful
        // before the user wires a real registry file.
        default_registry()
    } else {
        default_registry()
    };

    let state = AppState::new(registry, &cli.upstream);
    let app = build(state);
    let listener = tokio::net::TcpListener::bind(cli.bind).await?;
    tracing::info!(bind = %cli.bind, upstream = %cli.upstream, "1bit-lemonade listening");
    axum::serve(listener, app).await?;
    Ok(())
}

fn default_registry() -> ModelRegistry {
    let mut r = ModelRegistry::new();
    r.insert(
        "1bit-monster-2b",
        ModelEntry::new("local/bitnet-hip", vec!["chat".into(), "completion".into()]),
    );
    r
}
