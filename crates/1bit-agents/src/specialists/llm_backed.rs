//! LLM-backed specialists — replaces the stub `{}` returns with a real
//! round-trip to 1bit-halo-server's `/v1/chat/completions` endpoint.
//!
//! One struct, [`LlmSpecialist`], parametrised by `(Name, system_prompt)`.
//! Each instance is a single-turn prompt-the-LLM-and-relay-the-text worker.
//! Registered by [`crate::Registry::default_live`] for the four
//! specialists the Discord → echo pipeline actually delivers today:
//!
//! * [`Name::Herald`]        — conversational Q&A + chat-overflow
//! * [`Name::Sentinel`]      — bug-triage in auto-created threads
//! * [`Name::Magistrate`]    — feature-request review
//! * [`Name::Quartermaster`] — GitHub event triage
//!
//! Output contract: every response carries a top-level `"text"` field,
//! even on error. `1bit-watch-discord` reads `resp["text"]` and relays
//! it through echo's Discord client. A network failure must not break
//! the pipeline — instead we surface `(<name> down)` so operators see
//! something in channel rather than silent nothing.

use anyhow::Result;
use async_trait::async_trait;
use reqwest::Client;
use serde_json::{Value, json};
use std::time::Duration;

use crate::{Name, Specialist};

/// Default base URL for 1bit-halo-server on the local box. The full
/// chat-completions path is `{base}/v1/chat/completions`.
pub fn default_base_url() -> String {
    "http://127.0.0.1:8180".to_string()
}

/// Default model id served by 1bit-halo-server today.
pub fn default_model_id() -> String {
    "halo-1bit-2b".to_string()
}

/// Hard cap on how long we'll wait for the LLM. Discord replies need
/// to feel alive; a 10s ceiling is more than enough for a ≤400-tok
/// completion on the strixhalo box.
const HTTP_TIMEOUT: Duration = Duration::from_secs(10);

/// Generic LLM-backed specialist. Holds a pre-built reqwest client so
/// keep-alive reuses the same TCP connection across Discord messages.
pub struct LlmSpecialist {
    name: Name,
    base_url: String,
    model_id: String,
    system_prompt: String,
    client: Client,
}

impl LlmSpecialist {
    /// Construct with explicit wiring. Callers pick the name, prompt,
    /// and server URL. Tests override `base_url`; production uses
    /// [`default_base_url`].
    pub fn new(
        name: Name,
        base_url: impl Into<String>,
        model_id: impl Into<String>,
        system_prompt: impl Into<String>,
    ) -> Self {
        let client = Client::builder()
            .timeout(HTTP_TIMEOUT)
            .build()
            .unwrap_or_else(|_| Client::new());
        Self {
            name,
            base_url: base_url.into(),
            model_id: model_id.into(),
            system_prompt: system_prompt.into(),
            client,
        }
    }

    /// Herald — conversational Q&A voice of 1bit.systems on Discord.
    pub fn herald(base_url: impl Into<String>, model_id: impl Into<String>) -> Self {
        Self::new(Name::Herald, base_url, model_id, HERALD_PROMPT)
    }

    /// Sentinel — bug-triage specialist. Restates, asks for missing
    /// repro info, suggests the most likely known cause.
    pub fn sentinel(base_url: impl Into<String>, model_id: impl Into<String>) -> Self {
        Self::new(Name::Sentinel, base_url, model_id, SENTINEL_PROMPT)
    }

    /// Magistrate — feature-request reviewer. Accept / defer / reject
    /// verdicts tied to the ternary-kernel roadmap.
    pub fn magistrate(base_url: impl Into<String>, model_id: impl Into<String>) -> Self {
        Self::new(Name::Magistrate, base_url, model_id, MAGISTRATE_PROMPT)
    }

    /// Quartermaster — GitHub event triage (issue opened/closed,
    /// PR opened/merged, push).
    pub fn quartermaster(base_url: impl Into<String>, model_id: impl Into<String>) -> Self {
        Self::new(Name::Quartermaster, base_url, model_id, QUARTERMASTER_PROMPT)
    }

    /// Exposed for tests — confirm the wired prompt covers the role.
    pub fn system_prompt(&self) -> &str {
        &self.system_prompt
    }

    fn down_text(&self) -> String {
        format!("({} down)", self.name.as_str())
    }

    fn completions_url(&self) -> String {
        format!("{}/v1/chat/completions", self.base_url.trim_end_matches('/'))
    }

    async fn call_llm(&self, user_content: &str) -> Result<String, String> {
        let body = json!({
            "model": self.model_id,
            "temperature": 0.3,
            "max_tokens": 400,
            "stream": false,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
            ],
        });
        let resp = self
            .client
            .post(self.completions_url())
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("http send: {e}"))?;
        if !resp.status().is_success() {
            return Err(format!("http status: {}", resp.status()));
        }
        let v: Value = resp.json().await.map_err(|e| format!("json decode: {e}"))?;
        extract_content(&v).ok_or_else(|| "no choices[0].message.content".to_string())
    }
}

/// Pull `choices[0].message.content` out of an OpenAI-shaped response.
/// Exposed `pub(crate)` so tests can pin the parse logic without HTTP.
pub(crate) fn extract_content(v: &Value) -> Option<String> {
    v.get("choices")?
        .get(0)?
        .get("message")?
        .get("content")?
        .as_str()
        .map(|s| s.to_string())
}

/// Pull the user-facing content out of a dispatch request. Discord
/// populates `content`; GitHub payloads may not, so we fall back to
/// the whole JSON serialised as a string — Quartermaster's prompt
/// explicitly says "given a single GitHub event JSON".
fn extract_user_content(req: &Value) -> String {
    if let Some(s) = req.get("content").and_then(|v| v.as_str()) {
        if !s.trim().is_empty() {
            return s.to_string();
        }
    }
    // No bare `content` string — serialise the full payload. Used by
    // Quartermaster for GitHub event JSON.
    serde_json::to_string(req).unwrap_or_default()
}

#[async_trait]
impl Specialist for LlmSpecialist {
    fn name(&self) -> Name {
        self.name
    }

    fn description(&self) -> &'static str {
        // Static description — each specialist wires its own role via
        // the prompt, but MCP clients see a generic line here. The
        // load-bearing role text is in `system_prompt`.
        "LLM-backed specialist — calls 1bit-halo-server /v1/chat/completions"
    }

    async fn handle(&self, req: Value) -> Result<Value> {
        let user = extract_user_content(&req);
        match self.call_llm(&user).await {
            Ok(text) => Ok(json!({ "text": text })),
            Err(e) => {
                // Graceful degradation: ALWAYS return a `text` field so
                // echo still has something to post. The `error` field
                // is surfaced alongside for operator logs.
                Ok(json!({
                    "error": e,
                    "text": self.down_text(),
                }))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// System prompts. Kept as module-level `const` so tests can assert against
// them and operators can grep for exact wording.

pub const HERALD_PROMPT: &str = "You are Herald, the conversational voice of 1bit.systems on Discord. \
Answer the user's question concisely in 1-3 short paragraphs. If the answer requires specifics you \
don't have (benchmark numbers, internal paths, ROCm versions), ask ONE follow-up question. Tone: \
calm technical, no emoji, no marketing, no exclamations, no movie quotes. If the question isn't \
about 1bit.systems, say so and redirect to #water-cooler.";

pub const SENTINEL_PROMPT: &str = "You are Sentinel, bug-triage specialist for 1bit.systems. The \
user reported an issue. Your job: (1) restate the bug in one line, (2) list what else you need to \
reproduce (kernel version, ROCm version, commit SHA, minimal repro), (3) suggest the single most \
likely cause from the known-issue list (amdgpu OPTC hang on kernel 7.x, mlock missing, RoPE \
convention mismatch, ROCm gfx1151 Tier-1 absent, Caddy bearer missing, KV-cache mutex contention). \
Output <= 400 words. Calm technical tone.";

pub const MAGISTRATE_PROMPT: &str = "You are Magistrate, feature-request reviewer for 1bit.systems. \
The user proposed a feature or change. Output: (1) restatement in one line, (2) ranked verdict \
[accept, defer, reject] with one-sentence reasoning tied to roadmap priorities (ternary kernels on \
gfx1151 = primary, gfx1201 port = second, Sparse-BitNet retrain = in flight, NPU = deferred until \
XDNA2 unblocks), (3) if accept, suggest where in the codebase it lands. <= 300 words. Calm \
technical tone.";

pub const QUARTERMASTER_PROMPT: &str = "You are Quartermaster, GitHub event triage for 1bit.systems. \
Given a single GitHub event JSON (issue opened/closed, PR opened/merged, push), output a one-line \
human summary of what changed and whether it needs operator attention. <= 60 words. No preamble.";

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener as StdListener;

    /// One-shot TCP mock. Accepts a single connection, reads the
    /// request bytes, writes a canned 200 response. Returns the
    /// base URL (no path) so callers build `{base}/v1/chat/completions`.
    fn spawn_mock(canned_body: &'static str) -> String {
        let listener = StdListener::bind("127.0.0.1:0").expect("bind mock");
        let addr = listener.local_addr().expect("addr").to_string();
        std::thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut buf = [0u8; 8192];
                let _ = stream.read(&mut buf);
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    canned_body.len(),
                    canned_body
                );
                let _ = stream.write_all(resp.as_bytes());
                let _ = stream.flush();
            }
        });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn herald_returns_text_field_from_mock_completion() {
        // Happy-path: mock returns a synthetic OpenAI completion; the
        // specialist's handle() output must carry `"text"` verbatim.
        let canned = r#"{"choices":[{"message":{"content":"Herald here: bench is 83 tok/s on gfx1151."}}]}"#;
        let base = spawn_mock(canned);

        let herald = LlmSpecialist::herald(base, default_model_id());
        let out = herald
            .handle(json!({
                "source": "discord",
                "content": "what's the current tok/s?",
            }))
            .await
            .unwrap();
        assert_eq!(
            out["text"], "Herald here: bench is 83 tok/s on gfx1151.",
            "text field must echo the mock completion, got {out}"
        );
        // Happy path: no `error` key.
        assert!(
            out.get("error").is_none(),
            "no error expected on happy path: {out}"
        );
    }

    #[tokio::test]
    async fn network_error_returns_graceful_down_text() {
        // Port 1 is reserved and refuses connections in practice.
        // Specialist must still return a Value with a `"text"` field
        // containing "down" so echo posts something readable.
        let sentinel = LlmSpecialist::sentinel(
            "http://127.0.0.1:1".to_string(),
            default_model_id(),
        );
        let out = sentinel
            .handle(json!({
                "source": "discord",
                "content": "panic in decode loop",
            }))
            .await
            .unwrap();
        let text = out["text"].as_str().expect("text field must be a string");
        assert!(
            text.contains("(down)") || text.contains("down)"),
            "degraded text should contain '(down)' marker, got: {text}"
        );
        // Error reason should be surfaced for operator logs.
        assert!(
            out.get("error").is_some(),
            "error field should be populated on network failure: {out}"
        );
    }

    #[test]
    fn system_prompts_wire_name_and_role_keyword_through() {
        // Basic construction sanity. Each factory produces an
        // LlmSpecialist whose `system_prompt` mentions both the
        // specialist's own name and a role-identifying keyword — so
        // a typo in factory wiring can't silently ship a blank prompt.
        let h = LlmSpecialist::herald("http://x", "m");
        assert_eq!(h.name(), Name::Herald);
        assert!(h.system_prompt().contains("Herald"));
        assert!(h.system_prompt().to_lowercase().contains("discord"));

        let s = LlmSpecialist::sentinel("http://x", "m");
        assert_eq!(s.name(), Name::Sentinel);
        assert!(s.system_prompt().contains("Sentinel"));
        assert!(s.system_prompt().to_lowercase().contains("bug"));

        let m = LlmSpecialist::magistrate("http://x", "m");
        assert_eq!(m.name(), Name::Magistrate);
        assert!(m.system_prompt().contains("Magistrate"));
        assert!(m.system_prompt().to_lowercase().contains("feature"));

        let q = LlmSpecialist::quartermaster("http://x", "m");
        assert_eq!(q.name(), Name::Quartermaster);
        assert!(q.system_prompt().contains("Quartermaster"));
        assert!(q.system_prompt().to_lowercase().contains("github"));
    }

    #[test]
    fn extract_user_content_prefers_content_field_then_falls_back_to_json() {
        // Discord messages carry `content`; GitHub payloads may not.
        let disc = json!({"content": "hello", "author": "alice"});
        assert_eq!(extract_user_content(&disc), "hello");

        let gh = json!({"event": "pull_request", "action": "opened", "number": 7});
        let got = extract_user_content(&gh);
        assert!(got.contains("pull_request"));
        assert!(got.contains("opened"));
    }

    #[test]
    fn extract_content_parses_openai_shape() {
        let v = json!({"choices":[{"message":{"content":"ok"}}]});
        assert_eq!(extract_content(&v).as_deref(), Some("ok"));

        let missing = json!({"choices":[]});
        assert!(extract_content(&missing).is_none());
    }
}
