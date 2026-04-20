// 1bit-mcp — Phase 1 entry point.
//
// Flow:
//   1. Build a StdioServer wired to the default 1bit-agents registry
//      (17 stubs via Registry::default_stubs, one Arc shared across
//      requests).
//   2. Drive the JSON-RPC loop on stdin/stdout until EOF.
//
// Signals: SIGTERM / SIGINT close stdin via the kernel and let run()
// return Ok(()) naturally. We don't install handlers yet — the current
// workload is entirely synchronous stubs, so there's nothing to drain.
//
// Environment:
//   HALO_MCP_TIMEOUT_MS — reserved for when real specialists do I/O.
//                         Currently just logged so ops tooling can
//                         verify the plumbing end-to-end.
//   RUST_LOG            — standard tracing filter. Defaults to
//                         "onebit_mcp=info" if unset.

use anyhow::Result;
use tokio::io;
use tracing::info;

use onebit_mcp::StdioServer;

fn load_timeout_ms() -> u64 {
    std::env::var("HALO_MCP_TIMEOUT_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|&ms| ms > 0)
        .unwrap_or(30_000)
}

#[tokio::main]
async fn main() -> Result<()> {
    // Logs go to stderr so they never corrupt the JSON-RPC stream on
    // stdout. Clients talking to us over stdio rely on this.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "onebit_mcp=info".into()),
        )
        .with_writer(std::io::stderr)
        .with_target(false)
        .init();

    let timeout_ms = load_timeout_ms();
    let server = StdioServer::with_default_agents();

    info!(
        version = %onebit_mcp::SERVER_VERSION,
        tools = server.registry().len(),
        timeout_ms,
        "1bit-mcp starting (Phase 1 — tools/call routes via onebit_agents::Registry)"
    );

    server.run(io::stdin(), io::stdout()).await?;

    info!("1bit-mcp stdin closed, exiting 0");
    Ok(())
}
