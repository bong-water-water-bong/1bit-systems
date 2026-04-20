//! halo-echo wraps [`halo_voice`] + an Opus encoder + axum WebSocket to
//! ship voice chunks to browsers.
//!
//! Today: scaffold only. Real Opus encoding is deferred — we yield raw
//! WAV frames over the WebSocket until `opus-rs` or `symphonia` lands in
//! the tree. The wire format is going to change the moment that dep
//! appears; consumers should treat the binary payloads as opaque audio
//! blobs and not assume WAV forever.
//!
//! Topology:
//!
//! ```text
//!   browser ── ws /ws ──▶ halo-echo ── spawn ──▶ halo-voice::VoicePipeline
//!                ▲                                         │
//!                └────── binary frames (WAV today) ◀───────┘
//! ```
//!
//! The browser sends one text frame (the prompt). halo-echo drives the
//! existing halo-voice sentence-boundary pipeline, forwarding each
//! `VoiceChunk.wav` straight down the socket as a binary frame.
//!
//! Intentionally minimal: no auth, no multi-turn, no reconnect. Those
//! land once the Opus encoder does.

pub mod server;

pub use server::EchoServer;
