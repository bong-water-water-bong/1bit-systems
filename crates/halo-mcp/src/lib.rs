//! halo-mcp — JSON-RPC 2.0 stdio MCP server exposing halo-agents
//! specialists as tools.
//!
//! Rust port of the C++ `halo-mcp` binary at
//! `/home/bcloud/repos/halo-mcp/`. Phase 0 parity: the 17 specialists
//! are exposed as stub tools that return `{"error": "not implemented"}`
//! on `tools/call`. Phase 1 will wire the call handler into the
//! `halo-agents` bus.
//!
//! Entry points:
//!   * [`StdioServer::phase0`] — prebuilt server with 17 stub tools.
//!   * [`ToolRegistry::default_phase0`] — just the registry, for tests
//!     or custom handlers.
//!
//! Wire format:
//!   * One JSON-RPC object per `\n`-delimited line (Claude Code MCP
//!     convention). No LSP `Content-Length` framing.

pub mod registry;
pub mod server;
pub mod specialists;

pub use registry::{Tool, ToolRegistry};
pub use server::{
    CallHandler, PROTOCOL_VERSION, SERVER_NAME, SERVER_VERSION, StdioServer,
    phase0_not_implemented_handler,
};
pub use specialists::{KNOWN, Specialist};
