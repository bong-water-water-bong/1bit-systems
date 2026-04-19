//! `halo-server` binary — OpenAI-compatible HTTP daemon.
//!
//! Replaces `bitnet_decode --server` (C++) and `lemonade-server` (Python).
//! Ships with an [`EchoBackend`] by default; pass `--features real-backend`
//! at build time plus `--model <path>` at launch time to load a real
//! `.h1b` model into `halo_router::Router` and swap it in.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Parser;
use tracing::info;
use tracing_subscriber::EnvFilter;

use halo_server::{EchoBackend, InferenceBackend, build_router, shutdown_signal};

/// halo-server — OpenAI-compatible HTTP front door.
#[derive(Parser, Debug)]
#[command(name = "halo-server", version, about)]
struct Args {
    /// Bind address (host:port). Overrides $HALO_SERVER_BIND.
    #[arg(long, env = "HALO_SERVER_BIND", default_value = "127.0.0.1:8080")]
    bind: SocketAddr,

    /// Path to a `.h1b` model to load via halo-router.
    ///
    /// Ignored when the binary is built without `--features real-backend`.
    /// When set, the tokenizer is resolved by replacing the `.h1b` extension
    /// with `.htok` (matches gen-1's on-disk layout).
    #[arg(long, env = "HALO_SERVER_MODEL")]
    model: Option<PathBuf>,
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

    let backend: Arc<dyn InferenceBackend> = build_backend(&args)?;
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

#[cfg(feature = "real-backend")]
fn build_backend(args: &Args) -> Result<Arc<dyn InferenceBackend>> {
    use halo_server::backend::RealBackend;
    match &args.model {
        Some(path) => {
            info!(model = %path.display(), "real-backend enabled — loading halo-router");
            let real = RealBackend::new(path)
                .with_context(|| format!("halo-router load {}", path.display()))?;
            Ok(Arc::new(real))
        }
        None => {
            // real-backend compiled in but no model path provided — keep the
            // stub so the binary still starts (useful for health checks /
            // reading /v1/models before anything is loaded).
            info!("real-backend compiled but no --model given; using EchoBackend");
            Ok(Arc::new(EchoBackend::new()))
        }
    }
}

#[cfg(not(feature = "real-backend"))]
fn build_backend(args: &Args) -> Result<Arc<dyn InferenceBackend>> {
    if args.model.is_some() {
        tracing::warn!(
            "`--model` was passed but this binary was built without `--features real-backend`; \
             ignoring and using EchoBackend"
        );
    }
    Ok(Arc::new(EchoBackend::new()))
}
