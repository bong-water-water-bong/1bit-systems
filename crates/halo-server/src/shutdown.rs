//! Graceful-shutdown signal future.
//!
//! systemd sends SIGTERM on stop; interactive runs want Ctrl-C. We `select!`
//! on both and return as soon as either fires, so `axum::serve(...)
//! .with_graceful_shutdown(shutdown_signal())` cleanly drains in-flight
//! requests before exiting.
//!
//! On non-Unix targets we fall back to Ctrl-C only — the crate isn't
//! expected to run off Linux right now, but keeping it portable avoids
//! surprise build breakage if we ever cross-compile for, say, macOS dev.

use tracing::info;

pub async fn shutdown_signal() {
    let ctrl_c = async {
        if let Err(e) = tokio::signal::ctrl_c().await {
            tracing::warn!("failed to install Ctrl-C handler: {e}");
            std::future::pending::<()>().await;
        }
    };

    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut s) => {
                s.recv().await;
            }
            Err(e) => {
                tracing::warn!("failed to install SIGTERM handler: {e}");
                std::future::pending::<()>().await;
            }
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c    => info!("received Ctrl-C, shutting down"),
        _ = terminate => info!("received SIGTERM, shutting down"),
    }
}
