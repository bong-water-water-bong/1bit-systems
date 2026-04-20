//! halo-lemonade — OpenAI-compat model gateway scaffold.
//!
//! Rust replacement for the Python `lemonade-server` binary. This crate is
//! the plumbing layer: registry of model IDs, dispatch decision (local
//! halo-router backend vs upstream proxy), and TOML config. HTTP surface
//! (`/v1/chat/completions`, `/v1/models`, `/v1/embeddings`) lands in a
//! follow-up once the router contract is stable.

pub mod config;
pub mod dispatch;
pub mod metrics;
pub mod registry;
pub mod routes;

pub use config::LemonadeConfig;
pub use dispatch::{Dispatch, HaloServer, Upstream, UpstreamRequest, UpstreamResponse};
pub use metrics::Metrics;
pub use registry::{ModelEntry, ModelRegistry};
