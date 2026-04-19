//! JSON-RPC 2.0 stdio server.
//!
//! Rust port of `/home/bcloud/repos/halo-mcp/src/stdio_server.cpp`.
//!
//! Protocol notes:
//!   * MCP spec revision `2024-11-05` (stable as of Apr 2026).
//!   * One JSON-RPC object per line, `\n`-delimited (Claude Code MCP client
//!     convention — no LSP-style `Content-Length` framing).
//!   * Notifications (missing `id`) still get a response in Phase 0 to
//!     match the C++ implementation; Claude Code never sends pure
//!     notifications on `tools/*` anyway.
//!   * EOF on stdin is graceful — we flush stdout and the caller exits 0.

use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWrite, AsyncWriteExt, BufReader};

use crate::registry::{Tool, ToolRegistry};

/// MCP protocol revision we speak.
pub const PROTOCOL_VERSION: &str = "2024-11-05";

/// Server name reported via `initialize.serverInfo`.
pub const SERVER_NAME: &str = "halo-mcp";

/// Server version reported via `initialize.serverInfo`.
pub const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

// JSON-RPC 2.0 standard error codes.
pub mod err {
    pub const INVALID_REQUEST: i32 = -32600;
    pub const METHOD_NOT_FOUND: i32 = -32601;
    pub const INVALID_PARAMS: i32 = -32602;
    pub const INTERNAL: i32 = -32603;
    /// Server-defined range: -32000 to -32099. Reserved for "unknown tool".
    pub const UNKNOWN_TOOL: i32 = -32001;
}

/// Handler signature for `tools/call`. Takes the matched tool + raw
/// arguments, returns a JSON payload (which may itself carry an `error`
/// key — that gets surfaced via MCP `isError`).
pub type CallHandler = Box<dyn Fn(&Tool, &Value) -> Value + Send + Sync>;

/// The Phase 0 default handler — returns `{"error": "not implemented"}`
/// for every call, matching the behaviour of the C++ binary built
/// without `HALO_MCP_HAS_AGENT_CPP`.
pub fn phase0_not_implemented_handler() -> CallHandler {
    Box::new(|tool: &Tool, _args: &Value| -> Value {
        json!({
            "error": "not implemented",
            "tool": tool.name,
            "target_agent": tool.target_agent,
            "phase": 0,
        })
    })
}

fn make_ok(id: Value, result: Value) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": result,
    })
}

fn make_err(id: Value, code: i32, msg: impl Into<String>) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {
            "code": code,
            "message": msg.into(),
        },
    })
}

/// The JSON-RPC server. Owns the tool registry and call handler; driven
/// by [`run`](StdioServer::run).
pub struct StdioServer {
    registry: ToolRegistry,
    handler: CallHandler,
}

impl StdioServer {
    /// Construct a server from a registry and a call handler.
    pub fn new(registry: ToolRegistry, handler: CallHandler) -> Self {
        Self { registry, handler }
    }

    /// Convenience: Phase 0 server with the default 17-tool registry and
    /// the "not implemented" call handler.
    pub fn phase0() -> Self {
        Self::new(
            ToolRegistry::default_phase0(),
            phase0_not_implemented_handler(),
        )
    }

    /// Borrow the underlying registry (useful for metrics / tests).
    pub fn registry(&self) -> &ToolRegistry {
        &self.registry
    }

    /// Handle a single parsed JSON-RPC request, returning the response.
    pub fn handle_request(&self, req: &Value) -> Value {
        // JSON-RPC allows id to be string, number, or null. Echo what we
        // received; absence means notification.
        let id = req.get("id").cloned().unwrap_or(Value::Null);
        let method = req.get("method").and_then(|m| m.as_str()).unwrap_or("");

        match method {
            "initialize" => make_ok(
                id,
                json!({
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": { "tools": {} },
                    "serverInfo": {
                        "name": SERVER_NAME,
                        "version": SERVER_VERSION,
                    },
                }),
            ),

            "tools/list" => {
                let tools: Vec<Value> = self
                    .registry
                    .iter()
                    .map(|t| {
                        json!({
                            "name": t.name,
                            "description": t.description,
                            "inputSchema": t.input_schema,
                        })
                    })
                    .collect();
                make_ok(id, json!({ "tools": tools }))
            }

            "tools/call" => self.handle_tools_call(id, req),

            other => make_err(
                id,
                err::METHOD_NOT_FOUND,
                format!("method not found: {other}"),
            ),
        }
    }

    fn handle_tools_call(&self, id: Value, req: &Value) -> Value {
        let params = match req.get("params") {
            Some(Value::Object(_)) => req.get("params").unwrap().clone(),
            Some(Value::Null) | None => Value::Object(Default::default()),
            Some(_) => {
                return make_err(id, err::INVALID_PARAMS, "params must be an object");
            }
        };

        let name = params
            .get("name")
            .and_then(|n| n.as_str())
            .unwrap_or("")
            .to_string();

        let args = params
            .get("arguments")
            .cloned()
            .unwrap_or_else(|| Value::Object(Default::default()));

        let Some(tool) = self.registry.find(&name) else {
            return make_err(
                id,
                err::UNKNOWN_TOOL,
                format!("unknown tool: {name}"),
            );
        };

        // The handler is a pure fn in Phase 0 and can't panic through
        // user input — but we still guard Phase 1 bus-bridge paths by
        // catching unwinds is overkill here; we simply let panics abort
        // (panic = "abort" in release profile anyway). If a future
        // handler needs to surface std::io errors, wrap them inside the
        // JSON payload as `{"error": "..."}`.
        let result = (self.handler)(tool, &args);
        let is_error = result.get("error").is_some();

        make_ok(
            id,
            json!({
                "content": [
                    { "type": "text", "text": result.to_string() }
                ],
                "isError": is_error,
            }),
        )
    }

    /// Drive the stdio loop until EOF. Reads newline-delimited JSON-RPC
    /// requests from `input` and writes `\n`-delimited responses to
    /// `output`. Graceful EOF returns `Ok(())` with output flushed.
    pub async fn run<R, W>(&self, input: R, mut output: W) -> anyhow::Result<()>
    where
        R: tokio::io::AsyncRead + Unpin,
        W: AsyncWrite + Unpin,
    {
        let reader = BufReader::new(input);
        let mut lines = reader.lines();

        while let Some(line) = lines.next_line().await? {
            if line.is_empty() {
                continue;
            }

            let resp = match serde_json::from_str::<Value>(&line) {
                Ok(req) => self.handle_request(&req),
                Err(e) => make_err(
                    Value::Null,
                    err::INVALID_REQUEST,
                    format!("parse error: {e}"),
                ),
            };

            // Claude Code MCP client expects one-line JSON, `\n`-delimited.
            let mut bytes = serde_json::to_vec(&resp)?;
            bytes.push(b'\n');
            output.write_all(&bytes).await?;
            output.flush().await?;
        }

        // Graceful EOF — flush any remaining buffered output and exit.
        output.flush().await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::AsyncReadExt;

    fn server() -> StdioServer {
        StdioServer::phase0()
    }

    #[test]
    fn initialize_returns_server_info() {
        let s = server();
        let resp = s.handle_request(&json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {}
        }));
        assert_eq!(resp["jsonrpc"], "2.0");
        assert_eq!(resp["id"], 1);
        assert_eq!(resp["result"]["serverInfo"]["name"], SERVER_NAME);
        assert_eq!(resp["result"]["serverInfo"]["version"], SERVER_VERSION);
        assert_eq!(resp["result"]["protocolVersion"], PROTOCOL_VERSION);
        assert!(resp["result"]["capabilities"]["tools"].is_object());
    }

    #[test]
    fn tools_list_returns_seventeen() {
        let s = server();
        let resp = s.handle_request(&json!({
            "jsonrpc": "2.0",
            "id": "x",
            "method": "tools/list"
        }));
        assert_eq!(resp["id"], "x");
        let tools = resp["result"]["tools"].as_array().expect("tools array");
        assert_eq!(tools.len(), 17);
        // Every tool must have name + description + inputSchema.
        for t in tools {
            assert!(t["name"].is_string());
            assert!(t["description"].is_string());
            assert!(t["inputSchema"].is_object());
        }
    }

    #[test]
    fn tools_list_preserves_order() {
        let s = server();
        let resp = s.handle_request(&json!({
            "jsonrpc": "2.0", "id": 1, "method": "tools/list"
        }));
        let tools = resp["result"]["tools"].as_array().unwrap();
        assert_eq!(tools[0]["name"], "muse_call");
        assert_eq!(tools[16]["name"], "anvil_call");
    }

    #[test]
    fn tools_call_unknown_tool_returns_minus_32001() {
        let s = server();
        let resp = s.handle_request(&json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": { "name": "nope_call", "arguments": {} }
        }));
        assert_eq!(resp["error"]["code"], err::UNKNOWN_TOOL);
    }

    #[test]
    fn tools_call_returns_not_implemented_envelope() {
        let s = server();
        let resp = s.handle_request(&json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": { "name": "muse_call", "arguments": {"hello": "world"} }
        }));
        // The Phase 0 handler writes `{"error": "not implemented", ...}`
        // into the content block and sets isError = true.
        assert_eq!(resp["result"]["isError"], true);
        let text = resp["result"]["content"][0]["text"]
            .as_str()
            .expect("content text");
        assert!(text.contains("not implemented"), "got: {text}");
        assert!(text.contains("muse"));
    }

    #[test]
    fn unknown_method_returns_minus_32601() {
        let s = server();
        let resp = s.handle_request(&json!({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "does/not/exist"
        }));
        assert_eq!(resp["error"]["code"], err::METHOD_NOT_FOUND);
    }

    #[test]
    fn invalid_params_object_rejected() {
        let s = server();
        let resp = s.handle_request(&json!({
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": 42
        }));
        assert_eq!(resp["error"]["code"], err::INVALID_PARAMS);
    }

    #[tokio::test]
    async fn stdio_loop_handles_multiple_requests_and_eof() {
        let s = server();
        let input = b"{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\"}\n\
                      {\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"initialize\"}\n"
            .to_vec();

        let mut output: Vec<u8> = Vec::new();
        s.run(&input[..], &mut output).await.expect("run ok");

        let text = String::from_utf8(output).expect("utf8");
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 2, "expected 2 response lines, got {lines:?}");

        let r1: Value = serde_json::from_str(lines[0]).unwrap();
        let r2: Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(r1["id"], 1);
        assert_eq!(r1["result"]["tools"].as_array().unwrap().len(), 17);
        assert_eq!(r2["id"], 2);
        assert_eq!(r2["result"]["serverInfo"]["name"], SERVER_NAME);
    }

    #[tokio::test]
    async fn stdio_loop_emits_parse_error_on_garbage() {
        let s = server();
        let input = b"this is not json\n".to_vec();
        let mut output: Vec<u8> = Vec::new();
        s.run(&input[..], &mut output).await.expect("run ok");

        let text = String::from_utf8(output).unwrap();
        let v: Value = serde_json::from_str(text.trim()).unwrap();
        assert_eq!(v["error"]["code"], err::INVALID_REQUEST);
        assert_eq!(v["id"], Value::Null);
    }

    #[tokio::test]
    async fn stdio_loop_skips_empty_lines() {
        let s = server();
        let input = b"\n\n{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\"}\n\n"
            .to_vec();
        let mut output: Vec<u8> = Vec::new();
        s.run(&input[..], &mut output).await.unwrap();

        let text = String::from_utf8(output).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 1);
    }

    #[tokio::test]
    async fn stdio_loop_graceful_eof_with_empty_input() {
        let s = server();
        let input: &[u8] = b"";
        let mut output: Vec<u8> = Vec::new();
        s.run(input, &mut output).await.expect("EOF is graceful");
        assert!(output.is_empty());
    }

    // Sanity: verify our helpers produce well-formed JSON-RPC shapes.
    #[test]
    fn make_ok_and_err_shape() {
        let ok = make_ok(json!(1), json!({"x": 1}));
        assert_eq!(ok["jsonrpc"], "2.0");
        assert_eq!(ok["result"]["x"], 1);
        let e = make_err(json!(null), err::INTERNAL, "boom");
        assert_eq!(e["error"]["code"], err::INTERNAL);
        assert_eq!(e["error"]["message"], "boom");
    }

    // Read back a single response line from a Vec<u8> output.
    #[allow(dead_code)]
    async fn read_one_line(buf: &[u8]) -> String {
        let mut r = BufReader::new(buf);
        let mut s = String::new();
        r.read_to_string(&mut s).await.unwrap();
        s
    }
}
