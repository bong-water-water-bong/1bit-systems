//! `VoicePipeline` — orchestrator: 1bit-server SSE → [`SentenceSplitter`]
//! → kokoro `/tts` POST per sentence → [`VoiceChunk`] stream of WAV bytes.
//!
//! The interleave is the whole point: we emit a TTS request the moment
//! the splitter yields a sentence, in parallel with the LLM still
//! generating. Downstream consumers get their first audio chunk while
//! the LLM is still tokenizing later sentences.
//!
//! Two consumers in mind:
//! 1. `1bit-voice` CLI — pipes audio to `paplay`/`aplay` as chunks arrive.
//! 2. Future `1bit-echo` crate — frames Opus packets over WebSocket to a
//!    browser peer.
//!
//! This crate has no `aplay` dependency. Playback is the caller's job.

use anyhow::{Context, Result};
use bytes::Bytes;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::time::Duration;
use tokio_stream::Stream;

use crate::splitter::SentenceSplitter;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceConfig {
    /// 1bit-server base URL (must expose `/v1/chat/completions`).
    pub llm_url: String,
    /// Model id for the LLM request.
    pub model: String,
    /// Max tokens for the completion.
    pub max_tokens: u32,
    /// Sampling temperature. 0 = greedy.
    pub temperature: f32,
    /// 1bit-halo-kokoro `/tts` endpoint.
    pub tts_url: String,
    /// Voice id passed to kokoro.
    pub voice: String,
    /// Timeout per HTTP call.
    pub timeout_secs: u64,
}

impl Default for VoiceConfig {
    fn default() -> Self {
        Self {
            llm_url: "http://127.0.0.1:8180/v1/chat/completions".into(),
            model: "1bit-monster-2b".into(),
            max_tokens: 256,
            temperature: 0.7,
            tts_url: "http://127.0.0.1:8083/tts".into(),
            voice: "af_sky".into(),
            timeout_secs: 60,
        }
    }
}

/// One audio chunk + the sentence that produced it, for caller-side
/// caption / logging. `index` lets consumers reorder if the TTS task
/// is run out of order (we don't do that today, kept for future).
#[derive(Debug, Clone)]
pub struct VoiceChunk {
    pub index: usize,
    pub sentence: String,
    pub wav: Bytes,
}

pub struct VoicePipeline {
    cfg: VoiceConfig,
    http: reqwest::Client,
}

impl VoicePipeline {
    pub fn new(cfg: VoiceConfig) -> Result<Self> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(cfg.timeout_secs))
            .user_agent("1bit-voice/0.1")
            .build()?;
        Ok(Self { cfg, http })
    }

    /// Drive the pipeline for one user prompt. Returns a stream of
    /// `VoiceChunk`s in sentence order.
    pub fn speak(
        self,
        prompt: impl Into<String>,
    ) -> Pin<Box<dyn Stream<Item = Result<VoiceChunk>> + Send>> {
        let prompt = prompt.into();
        let cfg = self.cfg.clone();
        let http = self.http.clone();

        Box::pin(async_stream::try_stream! {
            let body = serde_json::json!({
                "model": cfg.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": cfg.max_tokens,
                "temperature": cfg.temperature,
                "stream": true,
            });

            let resp = http.post(&cfg.llm_url).json(&body).send().await
                .with_context(|| format!("POST {}", cfg.llm_url))?;
            let resp = match resp.error_for_status_ref() {
                Ok(_) => resp,
                Err(_) => {
                    let status = resp.status();
                    let text = resp.text().await.unwrap_or_default();
                    Err(anyhow::anyhow!("llm {}: {}", status, text))?;
                    unreachable!()
                }
            };

            let mut splitter = SentenceSplitter::new();
            let mut byte_stream = resp.bytes_stream();
            let mut sse_buf = String::new();
            let mut idx = 0usize;

            while let Some(chunk) = byte_stream.next().await {
                let chunk = chunk.context("sse chunk read")?;
                sse_buf.push_str(&String::from_utf8_lossy(&chunk));

                while let Some(nl) = sse_buf.find("\n\n") {
                    let event = sse_buf[..nl].to_string();
                    sse_buf.drain(..nl + 2);
                    if let Some(delta) = parse_sse_delta(&event) {
                        if delta.is_empty() { continue; }
                        for sentence in splitter.feed(&delta) {
                            let wav = synthesize(&http, &cfg, &sentence).await?;
                            yield VoiceChunk { index: idx, sentence, wav };
                            idx += 1;
                        }
                    }
                }
            }

            // Flush any trailing partial sentence (no closing punct).
            if let Some(tail) = splitter.finish() {
                let wav = synthesize(&http, &cfg, &tail).await?;
                yield VoiceChunk { index: idx, sentence: tail, wav };
            }
        })
    }
}

/// Parse a single SSE event block and return the OpenAI-compat
/// `choices[0].delta.content` if present. Ignores `[DONE]` / role-only /
/// malformed lines — returns `None` so the loop skips them.
fn parse_sse_delta(event: &str) -> Option<String> {
    let mut payload: Option<&str> = None;
    for line in event.lines() {
        if let Some(rest) = line.strip_prefix("data: ") {
            payload = Some(rest);
        } else if let Some(rest) = line.strip_prefix("data:") {
            payload = Some(rest);
        }
    }
    let p = payload?.trim();
    if p == "[DONE]" {
        return Some(String::new());
    }
    let v: serde_json::Value = serde_json::from_str(p).ok()?;
    let delta = v.get("choices")?.get(0)?.get("delta")?;
    delta.get("content")?.as_str().map(str::to_string)
}

/// POST a single sentence to 1bit-halo-kokoro `/tts`, return the raw WAV bytes.
///
/// Omits the `speed` field by default (kokoro_tts throws on the default
/// `1.0` via cxxopts; a known upstream bug documented in
/// `project_halo_kokoro.md`). Caller can override by plumbing `speed` in
/// future; v0 keeps it simple.
async fn synthesize(http: &reqwest::Client, cfg: &VoiceConfig, text: &str) -> Result<Bytes> {
    let body = serde_json::json!({
        "text": text,
        "voice": cfg.voice,
    });
    let resp = http
        .post(&cfg.tts_url)
        .json(&body)
        .send()
        .await
        .with_context(|| format!("POST {}", cfg.tts_url))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let msg = resp.text().await.unwrap_or_default();
        anyhow::bail!("kokoro {}: {}", status, msg);
    }
    Ok(resp.bytes().await?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_points_at_local() {
        let c = VoiceConfig::default();
        assert!(c.llm_url.contains("127.0.0.1:8180"));
        assert!(c.tts_url.contains("127.0.0.1:8083"));
        assert_eq!(c.voice, "af_sky");
    }

    #[test]
    fn parse_sse_delta_happy_path() {
        let event = r#"data: {"choices":[{"delta":{"content":"Paris"}}]}"#;
        assert_eq!(parse_sse_delta(event), Some("Paris".into()));
    }

    #[test]
    fn parse_sse_delta_done_marker_empty_content() {
        assert_eq!(parse_sse_delta("data: [DONE]"), Some(String::new()));
        assert_eq!(parse_sse_delta("data:[DONE]"), Some(String::new()));
    }

    #[test]
    fn parse_sse_delta_role_opener_without_content() {
        // First event in an OpenAI stream is role-only, no content yet.
        let event = r#"data: {"choices":[{"delta":{"role":"assistant"}}]}"#;
        assert_eq!(parse_sse_delta(event), None);
    }

    #[test]
    fn parse_sse_delta_malformed_returns_none() {
        assert_eq!(parse_sse_delta("data: not-json"), None);
        assert_eq!(parse_sse_delta("comment: whatever"), None);
        assert_eq!(parse_sse_delta(""), None);
    }

    #[test]
    fn voice_chunk_roundtrips_metadata() {
        let c = VoiceChunk {
            index: 3,
            sentence: "Hi.".into(),
            wav: Bytes::from_static(b"RIFF....WAVE"),
        };
        assert_eq!(c.index, 3);
        assert_eq!(c.sentence, "Hi.");
        assert!(c.wav.starts_with(b"RIFF"));
    }
}
