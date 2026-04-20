//! Axum WebSocket server for halo-echo.
//!
//! One route: `GET /ws`. Protocol is trivial today:
//!
//! 1. client upgrades, sends a single text frame containing the prompt
//! 2. server drives [`halo_voice::VoicePipeline::speak`] and forwards
//!    each `VoiceChunk.wav` as a binary frame
//! 3. server closes the socket when the stream drains
//!
//! No heartbeat, no reconnect, no auth. Scaffold only.

use anyhow::Result;
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::IntoResponse,
    routing::get,
    Router,
};
use futures::StreamExt;
use std::net::SocketAddr;
use std::sync::Arc;

use halo_voice::{VoiceConfig, VoicePipeline};

/// Entry point for the browser voice service.
///
/// Holds a bind address and the [`VoiceConfig`] that every inbound
/// connection will clone to build its own pipeline. `VoicePipeline` is
/// single-shot (its `speak` consumes `self`), so we build one per socket.
pub struct EchoServer {
    pub bind: SocketAddr,
    pub voice_cfg: VoiceConfig,
}

impl EchoServer {
    /// Default bind (`127.0.0.1:8085`) with a default [`VoiceConfig`].
    ///
    /// Handy for tests and for `EchoServer::default().run()` in one-liners.
    pub fn new() -> Self {
        Self {
            bind: default_bind(),
            voice_cfg: VoiceConfig::default(),
        }
    }

    /// Start axum on `self.bind` and serve `/ws` forever.
    pub async fn run(self) -> Result<()> {
        let state = Arc::new(self.voice_cfg);
        let app = Router::new()
            .route("/ws", get(ws_handler))
            .with_state(state);

        tracing::info!(bind = %self.bind, "halo-echo listening");
        let listener = tokio::net::TcpListener::bind(self.bind).await?;
        axum::serve(listener, app).await?;
        Ok(())
    }
}

impl Default for EchoServer {
    fn default() -> Self {
        Self::new()
    }
}

/// The canonical default bind for halo-echo. Kept as a fn so tests can
/// parse it without duplicating the string.
pub fn default_bind() -> SocketAddr {
    "127.0.0.1:8085"
        .parse()
        .expect("default bind literal is a valid SocketAddr")
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(cfg): State<Arc<VoiceConfig>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, cfg))
}

async fn handle_socket(mut socket: WebSocket, cfg: Arc<VoiceConfig>) {
    // Step 1: pull the prompt. One text frame, then we pivot to server-push.
    let prompt = match socket.recv().await {
        Some(Ok(Message::Text(t))) => t,
        Some(Ok(other)) => {
            tracing::warn!(?other, "expected text prompt frame; closing");
            let _ = socket.close().await;
            return;
        }
        Some(Err(e)) => {
            tracing::warn!(error = %e, "ws recv failed");
            return;
        }
        None => {
            tracing::debug!("client closed before sending prompt");
            return;
        }
    };

    let pipeline = match VoicePipeline::new((*cfg).clone()) {
        Ok(p) => p,
        Err(e) => {
            tracing::error!(error = %e, "failed to build VoicePipeline");
            let _ = socket.send(Message::Text(format!("error: {e}"))).await;
            let _ = socket.close().await;
            return;
        }
    };

    let mut stream = pipeline.speak(prompt);
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(c) => {
                // TODO(opus): when opus-rs/symphonia lands, re-encode c.wav
                // to Opus packets and frame them here. For now we ship the
                // raw WAV bytes as-is and the browser can decode via
                // WebAudio's decodeAudioData.
                if let Err(e) = socket.send(Message::Binary(c.wav.to_vec())).await {
                    tracing::debug!(error = %e, "client went away mid-stream");
                    return;
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "voice pipeline error");
                let _ = socket.send(Message::Text(format!("error: {e}"))).await;
                break;
            }
        }
    }

    let _ = socket.close().await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_bind_is_localhost_8085() {
        let s = EchoServer::new();
        assert_eq!(s.bind.to_string(), "127.0.0.1:8085");
        assert_eq!(s.bind.port(), 8085);
        assert!(s.bind.ip().is_loopback());
    }

    #[test]
    fn voice_config_roundtrips_through_server() {
        let cfg = VoiceConfig {
            llm_url: "http://example.test:9/v1/chat/completions".into(),
            tts_url: "http://example.test:10/tts".into(),
            voice: "bf_emma".into(),
            ..Default::default()
        };

        let s = EchoServer {
            bind: default_bind(),
            voice_cfg: cfg.clone(),
        };
        assert_eq!(s.voice_cfg.llm_url, cfg.llm_url);
        assert_eq!(s.voice_cfg.tts_url, cfg.tts_url);
        assert_eq!(s.voice_cfg.voice, "bf_emma");
    }

    #[test]
    fn construct_with_custom_bind_is_noop_until_run() {
        // Can build an EchoServer on any bind without touching the network.
        let s = EchoServer {
            bind: "127.0.0.1:0".parse().unwrap(),
            voice_cfg: VoiceConfig::default(),
        };
        assert_eq!(s.bind.port(), 0);
        // Default impl lines up with the explicit new().
        let d = EchoServer::default();
        assert_eq!(d.bind, default_bind());
    }
}
