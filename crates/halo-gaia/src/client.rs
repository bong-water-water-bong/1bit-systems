//! HTTP client that POSTs a `Conversation` to `{server_url}/v1/chat/completions`.
//!
//! The `reqwest::Client` is injectable so tests and alt-transports (mocks,
//! custom TLS, proxy) can swap it in without touching call sites.

use crate::conversation::Conversation;
use crate::session::SessionConfig;
use crate::stream::{parse_sse_line, SseEvent};
use anyhow::{anyhow, Context, Result};
use async_stream::try_stream;
use futures::stream::Stream;
use futures::StreamExt;
use serde_json::{json, Value};

pub struct GaiaClient {
    pub cfg: SessionConfig,
    pub http: reqwest::Client,
}

impl GaiaClient {
    pub fn new(cfg: SessionConfig) -> Self {
        Self { cfg, http: reqwest::Client::new() }
    }

    pub fn with_http(cfg: SessionConfig, http: reqwest::Client) -> Self {
        Self { cfg, http }
    }

    /// Build the JSON body the server sees. Pure — no I/O, easy to test.
    pub fn build_body(&self, conv: &Conversation) -> Value {
        let mut messages = Vec::with_capacity(conv.turns.len() + 1);
        if let Some(sys) = &self.cfg.system_prompt {
            messages.push(json!({ "role": "system", "content": sys }));
        }
        messages.extend(conv.to_openai_messages());
        json!({ "model": self.cfg.default_model, "messages": messages })
    }

    pub async fn send(&self, conv: &Conversation) -> Result<String> {
        let url = format!("{}/v1/chat/completions", self.cfg.server_url.trim_end_matches('/'));
        let mut req = self.http.post(&url).json(&self.build_body(conv));
        if let Some(tok) = &self.cfg.bearer {
            req = req.bearer_auth(tok);
        }
        let resp = req.send().await.with_context(|| format!("POST {url}"))?;
        let status = resp.status();
        let body: Value = resp.json().await.context("decode response json")?;
        if !status.is_success() {
            return Err(anyhow!("server {}: {}", status, body));
        }
        body["choices"][0]["message"]["content"]
            .as_str()
            .map(str::to_owned)
            .ok_or_else(|| anyhow!("missing choices[0].message.content in {body}"))
    }

    /// Streaming variant. POSTs with `stream: true` and yields one content
    /// chunk at a time by parsing OpenAI-style `data: {...}` SSE lines.
    ///
    /// Uses a manual newline splitter over the raw byte stream rather than
    /// pulling in an EventSource client — SSE for chat completions is trivial
    /// enough (one `data:` field per event, blank-line separated) that the
    /// extra dep isn't worth it. The parser itself lives in `stream.rs` and
    /// is unit-tested against the documented OpenAI shape.
    pub fn send_stream(
        &self,
        conv: &Conversation,
    ) -> impl Stream<Item = Result<String>> + Send + 'static {
        let url = format!("{}/v1/chat/completions", self.cfg.server_url.trim_end_matches('/'));
        let mut body = self.build_body(conv);
        body["stream"] = Value::Bool(true);
        let http = self.http.clone();
        let bearer = self.cfg.bearer.clone();

        try_stream! {
            let mut req = http.post(&url).json(&body);
            if let Some(tok) = bearer {
                req = req.bearer_auth(tok);
            }
            let resp = req.send().await.with_context(|| format!("POST {url}"))?;
            let status = resp.status();
            let mut bytes = if status.is_success() {
                resp.bytes_stream()
            } else {
                let text = resp.text().await.unwrap_or_default();
                Err(anyhow!("server {}: {}", status, text))?;
                unreachable!();
            };
            let mut buf: Vec<u8> = Vec::with_capacity(4096);
            'outer: while let Some(chunk) = bytes.next().await {
                let chunk = chunk.context("sse chunk")?;
                buf.extend_from_slice(&chunk);
                // Split the accumulated buffer on '\n' and keep the trailing
                // partial fragment in `buf`.
                while let Some(nl) = buf.iter().position(|b| *b == b'\n') {
                    let line = buf.drain(..=nl).collect::<Vec<u8>>();
                    let line = &line[..line.len() - 1]; // drop the '\n'
                    let line = std::str::from_utf8(line).unwrap_or("");
                    match parse_sse_line(line) {
                        SseEvent::Delta(s) => yield s,
                        SseEvent::Done => break 'outer,
                        SseEvent::Ignore => {}
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_body_injects_system_prompt_first() {
        let mut cfg = SessionConfig::new("http://x", "m");
        cfg.system_prompt = Some("be brief".into());
        let client = GaiaClient::new(cfg);
        let mut conv = Conversation::new();
        conv.push_user("hi".into());
        let body = client.build_body(&conv);
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "be brief");
        assert_eq!(msgs[1]["role"], "user");
        assert_eq!(body["model"], "m");
    }

    #[test]
    fn build_body_without_system_prompt() {
        let client = GaiaClient::new(SessionConfig::new("http://x", "m"));
        let mut conv = Conversation::new();
        conv.push_user("hi".into());
        let body = client.build_body(&conv);
        assert_eq!(body["messages"].as_array().unwrap().len(), 1);
    }
}
