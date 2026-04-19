//! halo-server — OpenAI-compatible HTTP server for the halo-ai stack.
//!
//! Replaces the legacy `bitnet_decode --server` (C++) and `lemonade-server`
//! (Python) daemons with a single Rust binary. The crate exposes a library
//! surface so downstream code (tests, integration harnesses, the unified
//! `halo` CLI) can spin up the router without going through the process
//! boundary.
//!
//! ## Layout
//!
//! * [`api`]      — OpenAI wire-format request / response types.
//! * [`backend`]  — the [`InferenceBackend`] trait + a default [`EchoBackend`]
//!                  stub that will be swapped for halo-router once the router
//!                  crate grows a real dispatcher.
//! * [`routes`]   — axum router, handlers, SSE streaming plumbing.
//! * [`shutdown`] — SIGTERM / Ctrl-C graceful-shutdown future (for systemd).
//! * [`error`]    — error type with `IntoResponse` → OpenAI-shaped JSON errors.
//!
//! ## Minimal usage
//!
//! ```no_run
//! use std::sync::Arc;
//! use halo_server::{backend::EchoBackend, routes::build_router, shutdown::shutdown_signal};
//!
//! # async fn run() -> anyhow::Result<()> {
//! let backend = Arc::new(EchoBackend::default());
//! let app = build_router(backend);
//! let listener = tokio::net::TcpListener::bind("127.0.0.1:8080").await?;
//! axum::serve(listener, app)
//!     .with_graceful_shutdown(shutdown_signal())
//!     .await?;
//! # Ok(()) }
//! ```

pub mod api;
pub mod backend;
pub mod error;
pub mod routes;
pub mod shutdown;

pub use backend::{EchoBackend, InferenceBackend, TokenStream};
#[cfg(feature = "real-backend")]
pub use backend::RealBackend;
pub use error::ServerError;
pub use routes::build_router;
pub use shutdown::shutdown_signal;
