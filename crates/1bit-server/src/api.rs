//! OpenAI-compatible wire types.
//!
//! Shapes match the `/v1/chat/completions`, `/v1/completions`, and
//! `/v1/models` endpoints as emitted by the legacy C++ `bitnet_decode --server`
//! (see `rocm-cpp/tools/bitnet_decode.cpp` — look for `httplib::Server`).
//!
//! Only the fields we actually consume or produce are modelled. Unknown
//! fields on incoming requests are tolerated via `serde`'s default
//! deny-nothing behaviour so newer OpenAI SDKs don't break us.

use serde::{Deserialize, Serialize};

// ─── Shared ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

// ─── /v1/chat/completions ────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    #[serde(default = "default_model")]
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub stop: Option<serde_json::Value>,
}

fn default_model() -> String {
    "bitnet-b1.58-2b-4t".to_string()
}

fn default_max_tokens() -> u32 {
    256
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str, // "chat.completion"
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: &'static str, // "stop" | "length"
}

// Streaming chunk: each SSE `data:` line carries one of these.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str, // "chat.completion.chunk"
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChunkChoice>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatChunkChoice {
    pub index: u32,
    pub delta: ChatDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<&'static str>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

// ─── /v1/completions ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct CompletionRequest {
    #[serde(default = "default_model")]
    pub model: String,
    pub prompt: PromptInput,
    #[serde(default)]
    pub stream: bool,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
}

/// OpenAI allows `prompt` to be either a single string or an array of
/// strings (for batching). We accept both and flatten at handler time.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum PromptInput {
    Single(String),
    Batch(Vec<String>),
}

impl PromptInput {
    pub fn first(&self) -> &str {
        match self {
            PromptInput::Single(s) => s.as_str(),
            PromptInput::Batch(v) => v.first().map(String::as_str).unwrap_or(""),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: &'static str, // "text_completion"
    pub created: i64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize)]
pub struct CompletionChoice {
    pub index: u32,
    pub text: String,
    pub finish_reason: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct CompletionChunk {
    pub id: String,
    pub object: &'static str, // "text_completion"
    pub created: i64,
    pub model: String,
    pub choices: Vec<CompletionChunkChoice>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CompletionChunkChoice {
    pub index: u32,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<&'static str>,
}

// ─── /ppl ────────────────────────────────────────────────────────────────

/// Input for `POST /ppl` — score a text blob against the loaded model.
///
/// Scoring is single-pass: we tokenize the full text (with BOS), truncate
/// to `max_tokens`, then feed each token into the decoder and accumulate
/// `-log_softmax(logits)[next_token]`. See
/// [`onebit_router::Router::perplexity`] for the exact algorithm; it is a
/// straight port of gen-1 `bitnet_decode --ppl`.
#[derive(Debug, Clone, Deserialize)]
pub struct PplRequest {
    /// Raw text to score. Typically a slice of wikitext-103-test.txt.
    pub text: String,
    /// Re-chunk window. `0` or `>= max_tokens` = single pass (the gen-1
    /// default). Larger than the router's `max_context` is clamped.
    #[serde(default = "default_ppl_stride")]
    pub stride: u32,
    /// Upper bound on the number of tokens actually scored — protects
    /// against a huge paste locking the backend for minutes.
    #[serde(default = "default_ppl_max_tokens")]
    pub max_tokens: u32,
}

fn default_ppl_stride() -> u32 {
    1024
}

fn default_ppl_max_tokens() -> u32 {
    8192
}

/// Output of `POST /ppl`. Field names mirror gen-1's `--ppl` JSON line so
/// downstream tooling can parse either transparently.
#[derive(Debug, Clone, Serialize)]
pub struct PplResponse {
    pub mean_nll: f64,
    pub perplexity: f64,
    /// Number of (context, target) pairs averaged into `mean_nll`.
    pub tokens: u32,
    pub elapsed_ms: f64,
}

// ─── /v1/models ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct ModelList {
    pub object: &'static str, // "list"
    pub data: Vec<ModelCard>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelCard {
    pub id: String,
    pub object: &'static str, // "model"
    pub owned_by: &'static str,
}

impl ModelCard {
    pub fn halo(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            object: "model",
            owned_by: "1bit systems",
        }
    }
}
