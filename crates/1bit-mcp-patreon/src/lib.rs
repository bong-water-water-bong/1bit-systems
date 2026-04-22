//! onebit-mcp-patreon — library half of the Patreon MCP bridge.
//!
//! The binary (`src/main.rs`) is a thin stdio JSON-RPC loop; all Patreon
//! API semantics live here so they are unit-testable without spawning a
//! child process or hitting the network.
//!
//! Patreon API v2 reference: https://docs.patreon.com/
//!
//! Rule A compliance: caller-side Rust, zero Python. This crate runs at
//! the operator's request, not in the serving hot path.

pub mod client;
pub mod mcp;
pub mod patreon;

pub use client::PatreonClient;
pub use mcp::{handle, tools_json};
