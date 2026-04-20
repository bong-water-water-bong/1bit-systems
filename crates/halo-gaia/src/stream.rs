//! OpenAI-compatible SSE stream parser.
//!
//! halo-server emits events in the canonical OpenAI chat-completion SSE shape:
//!
//! ```text
//! data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n
//! data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n
//! data: {"choices":[{"delta":{"content":" world"}}]}\n\n
//! data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n
//! data: [DONE]\n\n
//! ```
//!
//! `parse_sse_line` walks one logical line at a time and returns a
//! [`SseEvent`]. Callers feed raw lines in stream order; empty keep-alive
//! lines, comment lines, and non-`data:` fields are silently dropped so the
//! consumer only ever sees content or termination.

use serde_json::Value;

/// One logical parser outcome for a single SSE line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SseEvent {
    /// Content delta to append to the running assistant message.
    Delta(String),
    /// Terminal `[DONE]` sentinel — the server is done with this response.
    Done,
    /// Anything we don't care about (keep-alive blanks, role-only opener,
    /// comments, unknown fields). The caller should skip and keep reading.
    Ignore,
}

/// Parse a single SSE line into an [`SseEvent`].
///
/// The input is *one* line — without the trailing `\n`. The stream-reading
/// loop is responsible for framing; this function is pure.
pub fn parse_sse_line(line: &str) -> SseEvent {
    let line = line.trim_end_matches('\r');
    // Blank keep-alive between events, or SSE comment lines like `: ping`.
    if line.is_empty() || line.starts_with(':') {
        return SseEvent::Ignore;
    }
    // Only `data:` fields carry payload for chat completions. `event:`,
    // `id:`, `retry:` are fine to ignore for this client.
    let Some(payload) = line.strip_prefix("data:") else {
        return SseEvent::Ignore;
    };
    let payload = payload.trim_start();
    if payload == "[DONE]" {
        return SseEvent::Done;
    }
    // Parse the JSON blob. Server-side errors get emitted as `data: {"error":...}`
    // which we treat as "no content" and let the caller notice via Done /
    // EOF; we don't try to surface them here since this is v0.
    let Ok(val) = serde_json::from_str::<Value>(payload) else {
        return SseEvent::Ignore;
    };
    let delta = &val["choices"][0]["delta"]["content"];
    match delta.as_str() {
        Some(s) if !s.is_empty() => SseEvent::Delta(s.to_owned()),
        _ => SseEvent::Ignore,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_well_formed_delta() {
        let line = r#"data: {"choices":[{"delta":{"content":"Hi"}}]}"#;
        assert_eq!(parse_sse_line(line), SseEvent::Delta("Hi".into()));
    }

    #[test]
    fn parses_done_sentinel() {
        assert_eq!(parse_sse_line("data: [DONE]"), SseEvent::Done);
        // Tolerate the no-space variant too — some proxies rewrite this.
        assert_eq!(parse_sse_line("data:[DONE]"), SseEvent::Done);
    }

    #[test]
    fn ignores_non_data_lines() {
        assert_eq!(parse_sse_line(""), SseEvent::Ignore);
        assert_eq!(parse_sse_line(": keep-alive"), SseEvent::Ignore);
        assert_eq!(parse_sse_line("event: message"), SseEvent::Ignore);
        assert_eq!(parse_sse_line("id: 42"), SseEvent::Ignore);
    }

    #[test]
    fn ignores_role_only_opener() {
        // First frame carries only {"role":"assistant"} with no content.
        let line = r#"data: {"choices":[{"delta":{"role":"assistant"}}]}"#;
        assert_eq!(parse_sse_line(line), SseEvent::Ignore);
    }

    #[test]
    fn ignores_malformed_json() {
        assert_eq!(parse_sse_line("data: {not json"), SseEvent::Ignore);
    }

    #[test]
    fn strips_trailing_cr() {
        // Servers that emit CRLF (curl in some modes, nginx buffering).
        let line = "data: [DONE]\r";
        assert_eq!(parse_sse_line(line), SseEvent::Done);
    }
}
