//! 1bit-mcp-patreon — MCP stdio server binary.
//!
//! Spawn pattern mirrors the rest of the tree: read newline-framed
//! JSON-RPC from stdin, write newline-framed JSON-RPC to stdout, all
//! real work delegated to the `onebit_mcp_patreon` library crate.
//!
//! Env:
//!   PATREON_ACCESS_TOKEN  — creator-scoped OAuth token. Required for any
//!                           `tools/call` request; `tools/list` and
//!                           `initialize` work without it.

use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

use onebit_mcp_patreon::handle;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber_init();

    let stdin = tokio::io::stdin();
    let mut stdin = BufReader::new(stdin);
    let mut stdout = tokio::io::stdout();
    let mut line = String::new();
    loop {
        line.clear();
        let n = stdin.read_line(&mut line).await?;
        if n == 0 {
            break;
        }
        let req: Value = match serde_json::from_str(line.trim_end()) {
            Ok(v) => v,
            Err(e) => {
                let err = serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": Value::Null,
                    "error": { "code": -32700, "message": format!("parse error: {e}") }
                });
                writeln_frame(&mut stdout, &err).await?;
                continue;
            }
        };
        let resp = handle(&req).await;
        writeln_frame(&mut stdout, &resp).await?;
    }
    Ok(())
}

async fn writeln_frame<W: AsyncWriteExt + Unpin>(w: &mut W, v: &Value) -> std::io::Result<()> {
    let mut s = serde_json::to_string(v).map_err(std::io::Error::other)?;
    s.push('\n');
    w.write_all(s.as_bytes()).await?;
    w.flush().await?;
    Ok(())
}

/// Keep tracing wired but silent unless `RUST_LOG` is set — same shape as
/// sibling MCP binaries so operator logs line up.
fn tracing_subscriber_init() {
    // Keep dependency graph light: emit through `tracing::info!` etc. if
    // the caller pulled in a subscriber, otherwise no-op. We avoid pulling
    // tracing-subscriber here to keep cold-start fast (<5ms).
    tracing::dispatcher::get_default(|_| {});
}
