//! halo-gaia — desktop client scaffold.
//!
//! Eventually the native Rust replacement for lemonade-sdk's Gaia app. For
//! now this is just the transport + session model: config, conversation
//! buffer, OpenAI-shaped HTTP client. No UI, no TUI — those land in
//! follow-up crates once the core talks to halo-server cleanly.

pub mod client;
pub mod conversation;
pub mod session;

pub use client::GaiaClient;
pub use conversation::{ChatTurn, Conversation, Role};
pub use session::SessionConfig;
