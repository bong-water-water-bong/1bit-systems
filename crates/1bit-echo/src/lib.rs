//! 1bit-echo wraps [`onebit_voice`] + an Opus encoder + axum WebSocket to
//! ship voice chunks to browsers.
//!
//! Topology:
//!
//! ```text
//!   browser ── ws /ws ──▶ 1bit-echo ── spawn ──▶ 1bit-voice::VoicePipeline
//!                ▲                                         │
//!                └───── preamble + opus/wav frames ◀───────┘
//! ```
//!
//! Protocol:
//!
//! 1. client upgrades, sends one text frame with the prompt;
//! 2. server replies with ONE text frame carrying a JSON preamble:
//!    `{"sample_rate": 48000, "channels": 1, "frame_ms": 20, "codec": "opus"}`
//!    (or `"codec": "wav"` in legacy mode);
//! 3. for each `VoiceChunk` from 1bit-voice, the server either forwards
//!    the RIFF file verbatim (`--codec wav`) or re-encodes it into a
//!    series of 20 ms Opus packets, one binary frame per packet
//!    (`--codec opus`, default in the browser path).
//!
//! Intentionally minimal: no auth, no multi-turn, no reconnect.

pub mod opus;
pub mod server;

pub use server::{Codec, EchoServer};
