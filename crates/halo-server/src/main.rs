//! `halo-server` binary — OpenAI-compatible HTTP daemon.
//!
//! Replaces `bitnet_decode --server` (C++) and `lemonade-server` (Python).
//! Today it wires the stub [`EchoBackend`]; once `halo-router` implements
//! [`halo_server::InferenceBackend`] the single `Arc::new(...)` line
//! below is the only edit needed.

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Parser;
use tracing::info;
use tracing_subscriber::EnvFilter;

use halo_server::{EchoBackend, build_router, shutdown_signal};

/// halo-server — OpenAI-compatible HTTP front door.
#[derive(Parser, Debug)]
#[command(name = "halo-server", version, about)]
struct Args {
    /// Bind address (host:port). Overrides $HALO_SERVER_BIND.
    #[arg(long, env = "HALO_SERVER_BIND", default_value = "127.0.0.1:8080")]
    bind: SocketAddr,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,halo_server=debug,tower_http=info")),
        )
        .init();

    let args = Args::parse();

    // NOTE: swap this line for `halo_router::Router::new(...)` once the
    // router crate implements `InferenceBackend`. No other code changes.
    let backend = Arc::new(EchoBackend::new());
    let app = build_router(backend);

    let listener = tokio::net::TcpListener::bind(args.bind)
        .await
        .with_context(|| format!("bind {}", args.bind))?;
    info!(addr = %args.bind, "halo-server listening");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("axum serve failed")?;

    info!("halo-server stopped cleanly");
    Ok(())
}
