//! halo-mcp — JSON-RPC 2.0 stdio MCP server exposing `halo_agents`
//! specialists as tools.
//!
//! Phase 1 wiring: `tools/call` dispatches through
//! [`halo_agents::Registry`]. The tool list is derived from
//! [`halo_agents::Name::ALL`] so the MCP-visible surface and the agents
//! bus are the same single source of truth.
//!
//! Entry points:
//!   * [`StdioServer::with_default_agents`] — prebuilt server with the
//!     default 17-stub `halo_agents::Registry`.
//!   * [`StdioServer::new`] — custom tool registry + shared `Arc<Registry>`.
//!   * [`ToolRegistry::from_agents`] — just the tool list, for tests.
//!
//! Wire format:
//!   * One JSON-RPC object per `\n`-delimited line (Claude Code MCP
//!     convention). No LSP `Content-Length` framing.

pub mod memory;
pub mod registry;
pub mod server;
pub mod skills;
pub mod specialists;

pub use registry::{Tool, ToolRegistry};
pub use server::{PROTOCOL_VERSION, SERVER_NAME, SERVER_VERSION, StdioServer};
pub use specialists::description_for;
