//! JSON-RPC 2.0 stdio MCP server — bridges Claude Code's MCP client to
//! `onebit_agents::Registry`.
//!
//! Rust port of `/home/bcloud/repos/1bit-mcp/src/stdio_server.cpp`.
//!
//! Protocol notes:
//!   * MCP spec revision `2024-11-05` (stable as of Apr 2026).
//!   * One JSON-RPC object per line, `\n`-delimited (Claude Code MCP client
//!     convention — no LSP-style `Content-Length` framing).
//!   * Notifications (missing `id`) still get a response so the C++ and Rust
//!     binaries behave identically on the wire.
//!   * EOF on stdin is graceful — we flush stdout and the caller exits 0.
//!
//! Dispatch model:
//!   `tools/call` looks up the MCP tool by its exact name in the
//!   [`ToolRegistry`], then forwards the call to
//!   [`onebit_agents::Registry::dispatch`]. The registry is shared across
//!   every request via `Arc<Registry>` — no per-request allocation.

use std::sync::{Arc, Mutex};

use onebit_agents::{MemoryStore, Registry, SkillStore};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWrite, AsyncWriteExt, BufReader};

use crate::memory as memory_tool;
use crate::registry::ToolRegistry;
use crate::skills as skill_tool;

/// MCP protocol revision we speak.
pub const PROTOCOL_VERSION: &str = "2024-11-05";

/// Server name reported via `initialize.serverInfo`.
pub const SERVER_NAME: &str = "1bit-mcp";

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

/// The JSON-RPC server. Owns the `ToolRegistry` (for `tools/list` shape),
/// a shared `onebit_agents::Registry` (for specialist `tools/call` dispatch),
/// and a shared `SkillStore` (for the `skill_manage` tool).
pub struct StdioServer {
    tools: ToolRegistry,
    agents: Arc<Registry>,
    skills: Arc<Mutex<SkillStore>>,
    memory: Arc<Mutex<MemoryStore>>,
}

impl StdioServer {
    /// Construct a server from a tool registry, a shared agents registry,
    /// a shared skill store, and a shared memory store. The tool registry
    /// must already include the `skill_manage` + `memory_manage`
    /// descriptors if callers want them surfaced in `tools/list`.
    pub fn new(
        tools: ToolRegistry,
        agents: Arc<Registry>,
        skills: Arc<Mutex<SkillStore>>,
        memory: Arc<Mutex<MemoryStore>>,
    ) -> Self {
        Self {
            tools,
            agents,
            skills,
            memory,
        }
    }

    /// Convenience: default tools derived from `onebit_agents::Name::ALL`,
    /// agents registry seeded with `Registry::default_stubs()`, skill
    /// store rooted at `~/.halo/skills`. This is the wiring the binary
    /// uses on startup.
    ///
    /// Tool schemas are sourced from the live `onebit_agents::Registry`, so
    /// typed specialists (e.g. `Typed<AnvilSpecialist>`) publish their
    /// real `JsonSchema` in `tools/list` automatically. `skill_manage` is
    /// appended after the 17 specialists.
    ///
    /// Falls back to a tempdir-rooted store if `~/.halo/skills` cannot be
    /// resolved (no `$HOME`) — the server still boots, skill writes go
    /// somewhere harmless, and the tool call error propagates as an MCP
    /// `isError` payload rather than a startup crash.
    pub fn with_default_agents() -> Self {
        let agents = Arc::new(Registry::default_stubs());
        let mut tools = ToolRegistry::from_agents_registry(&agents);
        tools.push(skill_tool::tool());
        tools.push(memory_tool::tool());
        let skills = SkillStore::new().unwrap_or_else(|_| {
            SkillStore::with_root(std::env::temp_dir().join("1bit-mcp-skills-fallback"))
        });
        let memory = MemoryStore::new().unwrap_or_else(|_| {
            MemoryStore::with_root(std::env::temp_dir().join("1bit-mcp-memory-fallback"))
                .expect("tempdir memory fallback")
        });
        Self::new(
            tools,
            agents,
            Arc::new(Mutex::new(skills)),
            Arc::new(Mutex::new(memory)),
        )
    }

    /// Test-friendly constructor: wires the default agents registry but
    /// roots the skill + memory stores at arbitrary paths (tempdir in
    /// tests). Adds `skill_manage` + `memory_manage` to the tool registry.
    pub fn with_skill_root(root: std::path::PathBuf) -> Self {
        let agents = Arc::new(Registry::default_stubs());
        let mut tools = ToolRegistry::from_agents_registry(&agents);
        tools.push(skill_tool::tool());
        tools.push(memory_tool::tool());
        let skills = Arc::new(Mutex::new(SkillStore::with_root(root.clone())));
        let memory = Arc::new(Mutex::new(
            MemoryStore::with_root(root.join("memories")).expect("memory tempdir"),
        ));
        Self::new(tools, agents, skills, memory)
    }

    /// Borrow the underlying tool registry (useful for metrics / tests).
    pub fn registry(&self) -> &ToolRegistry {
        &self.tools
    }

    /// Borrow the shared agents registry.
    pub fn agents(&self) -> &Arc<Registry> {
        &self.agents
    }

    /// Borrow the shared skill store.
    pub fn skills(&self) -> &Arc<Mutex<SkillStore>> {
        &self.skills
    }

    /// Borrow the shared memory store.
    pub fn memory(&self) -> &Arc<Mutex<MemoryStore>> {
        &self.memory
    }

    /// Handle a single parsed JSON-RPC request, returning the response.
    pub async fn handle_request(&self, req: &Value) -> Value {
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
                    .tools
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

            "tools/call" => self.handle_tools_call(id, req).await,

            other => make_err(
                id,
                err::METHOD_NOT_FOUND,
                format!("method not found: {other}"),
            ),
        }
    }

    async fn handle_tools_call(&self, id: Value, req: &Value) -> Value {
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

        // Reject names we don't publish in `tools/list`. This keeps the
        // MCP-visible surface identical to the agents registry without
        // trusting the client to stay inside it.
        if self.tools.find(&name).is_none() {
            return make_err(id, err::UNKNOWN_TOOL, format!("unknown tool: {name}"));
        }

        // `skill_manage` + `memory_manage` are not specialists — they
        // mutate on-disk files directly. Route them before the agents
        // dispatch. Errors surface as an MCP `isError` content block,
        // same as the specialist path.
        let payload = if name == skill_tool::TOOL_NAME {
            skill_tool::handle(&self.skills, args)
        } else if name == memory_tool::TOOL_NAME {
            memory_tool::handle(&self.memory, args)
        } else {
            // Forward to the agents registry. A dispatch error here means
            // the specialist itself rejected the payload; surface it as an
            // MCP `isError` content block rather than a JSON-RPC error so
            // Claude Code can show it to the user without aborting.
            match self.agents.dispatch(&name, args).await {
                Ok(v) => v,
                Err(e) => json!({ "error": e.to_string(), "tool": name }),
            }
        };
        let is_error = payload.get("error").is_some();

        make_ok(
            id,
            json!({
                "content": [
                    { "type": "text", "text": payload.to_string() }
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
                Ok(req) => self.handle_request(&req).await,
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

    fn server() -> StdioServer {
        StdioServer::with_default_agents()
    }

    #[tokio::test]
    async fn initialize_returns_server_info() {
        let s = server();
        let resp = s
            .handle_request(&json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {}
            }))
            .await;
        assert_eq!(resp["jsonrpc"], "2.0");
        assert_eq!(resp["id"], 1);
        assert_eq!(resp["result"]["serverInfo"]["name"], SERVER_NAME);
        assert_eq!(resp["result"]["serverInfo"]["version"], SERVER_VERSION);
        assert_eq!(resp["result"]["protocolVersion"], PROTOCOL_VERSION);
        assert!(resp["result"]["capabilities"]["tools"].is_object());
    }

    #[tokio::test]
    async fn tools_list_returns_specialists_plus_skill_manage() {
        use onebit_agents::Name;
        let s = server();
        let resp = s
            .handle_request(&json!({
                "jsonrpc": "2.0",
                "id": "x",
                "method": "tools/list"
            }))
            .await;
        assert_eq!(resp["id"], "x");
        let tools = resp["result"]["tools"].as_array().expect("tools array");
        // 17 specialists + skill_manage + memory_manage.
        assert_eq!(tools.len(), 19);

        let got: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
        let mut want: Vec<&str> = Name::ALL.iter().map(|n| n.as_str()).collect();
        want.push(crate::skills::TOOL_NAME);
        want.push(crate::memory::TOOL_NAME);
        assert_eq!(
            got, want,
            "tools/list order must match Name::ALL + skill_manage + memory_manage"
        );

        for t in tools {
            assert!(t["name"].is_string());
            assert!(t["description"].is_string());
            // Schemas may be the passthrough { "type": "object" } for stubs
            // or a full JsonSchema for TypedSpecialist impls. We only
            // assert it's a JSON object with at least some shape — full
            // structure varies with schemars output across versions.
            let schema = &t["inputSchema"];
            assert!(schema.is_object(), "schema not object for {}", t["name"]);
        }

        // Anvil is the first TypedSpecialist demo — its schema must be a
        // real JsonSchema (has `properties`, not just `{"type":"object"}`).
        let anvil = tools
            .iter()
            .find(|t| t["name"] == "anvil")
            .expect("anvil tool");
        assert!(
            anvil["inputSchema"].get("properties").is_some(),
            "anvil should publish a typed schema with properties, got {}",
            anvil["inputSchema"]
        );
    }

    #[tokio::test]
    async fn tools_call_routes_to_agents_typed_anvil() {
        // Anvil is now a TypedSpecialist — response shape is the typed
        // AnvilResponse { cmd, status, tok_per_s }.
        let s = server();
        let resp = s
            .handle_request(&json!({
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": { "name": "anvil", "arguments": { "cmd": "ping" } }
            }))
            .await;
        assert_eq!(resp["result"]["isError"], false);
        let text = resp["result"]["content"][0]["text"]
            .as_str()
            .expect("content text");
        let inner: Value = serde_json::from_str(text).expect("inner json");
        assert_eq!(inner["cmd"], "ping");
        assert_eq!(inner["status"], "stub");
        assert!(inner["tok_per_s"].is_null());
    }

    #[tokio::test]
    async fn tools_call_routes_to_agents_stub_for_muse() {
        // Plain Stub path (Muse is still a Stub): {specialist, status, echo}.
        let s = server();
        let resp = s
            .handle_request(&json!({
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": { "name": "muse", "arguments": { "cmd": "ping" } }
            }))
            .await;
        assert_eq!(resp["result"]["isError"], false);
        let text = resp["result"]["content"][0]["text"]
            .as_str()
            .expect("content text");
        let inner: Value = serde_json::from_str(text).expect("inner json");
        assert_eq!(inner["status"], "stub");
        assert_eq!(inner["specialist"], "muse");
        assert_eq!(inner["echo"]["cmd"], "ping");
    }

    #[tokio::test]
    async fn tools_call_unknown_tool_returns_minus_32001() {
        let s = server();
        let resp = s
            .handle_request(&json!({
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": { "name": "nope", "arguments": {} }
            }))
            .await;
        assert_eq!(resp["error"]["code"], err::UNKNOWN_TOOL);
        assert!(resp.get("result").is_none());
    }

    #[tokio::test]
    async fn unknown_method_returns_minus_32601() {
        let s = server();
        let resp = s
            .handle_request(&json!({
                "jsonrpc": "2.0",
                "id": 4,
                "method": "does/not/exist"
            }))
            .await;
        assert_eq!(resp["error"]["code"], err::METHOD_NOT_FOUND);
    }

    #[tokio::test]
    async fn invalid_params_object_rejected() {
        let s = server();
        let resp = s
            .handle_request(&json!({
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": 42
            }))
            .await;
        assert_eq!(resp["error"]["code"], err::INVALID_PARAMS);
    }

    #[tokio::test]
    async fn tools_call_missing_arguments_defaults_to_empty_object() {
        // No `arguments` key — the stub should still get called with {}.
        let s = server();
        let resp = s
            .handle_request(&json!({
                "jsonrpc": "2.0",
                "id": 7,
                "method": "tools/call",
                "params": { "name": "muse" }
            }))
            .await;
        let text = resp["result"]["content"][0]["text"].as_str().unwrap();
        let inner: Value = serde_json::from_str(text).unwrap();
        assert_eq!(inner["specialist"], "muse");
        assert_eq!(inner["echo"], json!({}));
    }

    #[tokio::test]
    async fn tools_call_every_specialist_roundtrips() {
        use onebit_agents::Name;
        let s = server();
        for n in Name::ALL {
            // Anvil is typed — schema-validated input (needs `cmd`). All
            // other specialists are plain Stubs and accept `{}`.
            let args = if *n == Name::Anvil {
                json!({ "cmd": "status" })
            } else {
                json!({})
            };
            let resp = s
                .handle_request(&json!({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": { "name": n.as_str(), "arguments": args }
                }))
                .await;
            let text = resp["result"]["content"][0]["text"].as_str().unwrap();
            let inner: Value = serde_json::from_str(text).unwrap();
            assert_eq!(inner["status"], "stub", "bad status for {}", n.as_str());
            if *n == Name::Anvil {
                // Typed response shape — `cmd` echoed, no `specialist` key.
                assert_eq!(inner["cmd"], "status");
            } else {
                assert_eq!(inner["specialist"], n.as_str());
            }
        }
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
        // 17 specialists + skill_manage.
        assert_eq!(r1["result"]["tools"].as_array().unwrap().len(), 19);
        assert_eq!(r2["id"], 2);
        assert_eq!(r2["result"]["serverInfo"]["name"], SERVER_NAME);
    }

    #[tokio::test]
    async fn stdio_loop_tools_call_end_to_end() {
        // Use Muse (still a plain Stub) so we can exercise the stub echo
        // shape end-to-end through the stdio loop.
        let s = server();
        let input = b"{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"muse\",\"arguments\":{\"cmd\":\"ping\"}}}\n"
            .to_vec();
        let mut output: Vec<u8> = Vec::new();
        s.run(&input[..], &mut output).await.expect("run ok");

        let text = String::from_utf8(output).unwrap();
        let resp: Value = serde_json::from_str(text.trim()).unwrap();
        assert_eq!(resp["id"], 1);
        let inner_text = resp["result"]["content"][0]["text"].as_str().unwrap();
        assert!(
            inner_text.contains("\"status\":\"stub\""),
            "got: {inner_text}"
        );
        assert!(
            inner_text.contains("\"echo\":{\"cmd\":\"ping\"}"),
            "got: {inner_text}"
        );
    }

    #[tokio::test]
    async fn stdio_loop_typed_anvil_end_to_end() {
        // Typed specialist: request → schema-checked deserialize →
        // handle_typed → typed response → JSON.
        let s = server();
        let input = b"{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"anvil\",\"arguments\":{\"cmd\":\"build\"}}}\n"
            .to_vec();
        let mut output: Vec<u8> = Vec::new();
        s.run(&input[..], &mut output).await.expect("run ok");

        let text = String::from_utf8(output).unwrap();
        let resp: Value = serde_json::from_str(text.trim()).unwrap();
        let inner_text = resp["result"]["content"][0]["text"].as_str().unwrap();
        let inner: Value = serde_json::from_str(inner_text).unwrap();
        assert_eq!(inner["cmd"], "build");
        assert_eq!(inner["status"], "stub");
        assert!(inner["tok_per_s"].is_null());
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
        let input = b"\n\n{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\"}\n\n".to_vec();
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

    #[test]
    fn make_ok_and_err_shape() {
        let ok = make_ok(json!(1), json!({"x": 1}));
        assert_eq!(ok["jsonrpc"], "2.0");
        assert_eq!(ok["result"]["x"], 1);
        let e = make_err(json!(null), err::INTERNAL, "boom");
        assert_eq!(e["error"]["code"], err::INTERNAL);
        assert_eq!(e["error"]["message"], "boom");
    }

    #[tokio::test]
    async fn agents_registry_is_shared_arc() {
        // Confirm two handle_request calls reuse the same Arc<Registry> —
        // we're not rebuilding per request.
        let s = server();
        let before = Arc::strong_count(s.agents());
        let _ = s
            .handle_request(&json!({
                "jsonrpc":"2.0","id":1,"method":"tools/call",
                "params":{"name":"muse","arguments":{}}
            }))
            .await;
        let after = Arc::strong_count(s.agents());
        assert_eq!(
            before, after,
            "Arc strong_count should be stable across requests"
        );
    }

    // -------- skill_manage wiring tests --------
    //
    // These confirm the full JSON-RPC round-trip lands in the shared
    // `SkillStore` without touching `~/.halo/skills`. Per-action unit
    // tests live in `crate::skills::tests`; this is the integration
    // surface.

    #[tokio::test]
    async fn skill_manage_create_roundtrips_through_tools_call() {
        let td = tempfile::TempDir::new().unwrap();
        let s = StdioServer::with_skill_root(td.path().to_path_buf());
        let resp = s
            .handle_request(&json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "skill_manage",
                    "arguments": {
                        "action": "create",
                        "name": "server-test-skill",
                        "category": "tests",
                        "description": "wiring",
                        "body": "# body\n"
                    }
                }
            }))
            .await;
        assert_eq!(resp["result"]["isError"], false, "resp: {resp}");
        let text = resp["result"]["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("created"), "got: {text}");
        assert!(td.path().join("tests/server-test-skill/SKILL.md").exists());
    }

    #[tokio::test]
    async fn skill_manage_unknown_action_surfaces_as_is_error() {
        let td = tempfile::TempDir::new().unwrap();
        let s = StdioServer::with_skill_root(td.path().to_path_buf());
        let resp = s
            .handle_request(&json!({
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "skill_manage",
                    "arguments": { "action": "explode", "name": "x" }
                }
            }))
            .await;
        // The call still succeeds at the JSON-RPC layer — the error lives
        // in the MCP `isError` content block. This matches the specialist
        // error-reporting convention.
        assert_eq!(resp["result"]["isError"], true);
        let text = resp["result"]["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("explode"), "got: {text}");
    }
}
