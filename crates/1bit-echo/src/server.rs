//! Axum WebSocket server for 1bit-echo.
//!
//! Wire protocol (see crate docs): client upgrades and sends one text
//! frame (prompt). Server responds with one text frame — a JSON preamble
//! advertising the codec — then a stream of binary audio frames. In
//! `Codec::Wav` each frame is one RIFF file from kokoro verbatim. In
//! `Codec::Opus` each frame is one Opus packet (20 ms, 48 kHz, mono).

use anyhow::Result;
use axum::{
    Router,
    extract::{
        State,
        ws::{Message, WebSocket, WebSocketUpgrade},
    },
    response::IntoResponse,
    routing::get,
};
use futures::StreamExt;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use tokio_stream::Stream;

use onebit_voice::{VoiceChunk, VoiceConfig, VoicePipeline};

use crate::opus::{FRAME_MS, TARGET_SR, encode_wav_to_opus};

/// Wire-format selector. Default is `Wav` for backwards compat; browsers
/// should flip to `Opus`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum Codec {
    #[default]
    Wav,
    Opus,
}

impl std::str::FromStr for Codec {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "wav" | "WAV" => Ok(Codec::Wav),
            "opus" | "OPUS" => Ok(Codec::Opus),
            other => Err(format!("unknown codec `{other}` (want: wav|opus)")),
        }
    }
}

impl Codec {
    fn as_str(self) -> &'static str {
        match self {
            Codec::Wav => "wav",
            Codec::Opus => "opus",
        }
    }
}

/// Entry point for the browser voice service. `VoicePipeline::speak`
/// consumes `self`, so we build one pipeline per inbound socket.
pub struct EchoServer {
    pub bind: SocketAddr,
    pub voice_cfg: VoiceConfig,
    pub codec: Codec,
}

#[derive(Clone)]
struct AppState {
    voice_cfg: Arc<VoiceConfig>,
    codec: Codec,
}

impl EchoServer {
    pub fn new() -> Self {
        Self {
            bind: default_bind(),
            voice_cfg: VoiceConfig::default(),
            codec: Codec::default(),
        }
    }

    /// Start axum on `self.bind` and serve `/ws` forever.
    pub async fn run(self) -> Result<()> {
        let state = AppState {
            voice_cfg: Arc::new(self.voice_cfg),
            codec: self.codec,
        };
        let app = Router::new()
            .route("/ws", get(ws_handler))
            .with_state(state);

        tracing::info!(bind = %self.bind, codec = self.codec.as_str(), "1bit-echo listening");
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

pub fn default_bind() -> SocketAddr {
    "127.0.0.1:8085"
        .parse()
        .expect("default bind literal is a valid SocketAddr")
}

async fn ws_handler(ws: WebSocketUpgrade, State(state): State<AppState>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

/// Build the preamble text frame advertising the chosen codec. WAV mode
/// still carries a preamble so consumers can parse it once and skip
/// per-chunk RIFF reads.
fn preamble_json(codec: Codec) -> String {
    let (sr, frame_ms, name) = match codec {
        Codec::Opus => (TARGET_SR, FRAME_MS, "opus"),
        Codec::Wav => (24_000, 0, "wav"),
    };
    serde_json::json!({
        "sample_rate": sr,
        "channels": 1,
        "frame_ms": frame_ms,
        "codec": name,
    })
    .to_string()
}

async fn handle_socket(mut socket: WebSocket, state: AppState) {
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
        None => return,
    };

    // Step 2: send the codec preamble immediately — a browser can set up
    // its decoder in parallel with first-byte latency.
    if socket
        .send(Message::Text(preamble_json(state.codec)))
        .await
        .is_err()
    {
        return;
    }

    let pipeline = match VoicePipeline::new((*state.voice_cfg).clone()) {
        Ok(p) => p,
        Err(e) => {
            tracing::error!(error = %e, "failed to build VoicePipeline");
            let _ = socket.send(Message::Text(format!("error: {e}"))).await;
            let _ = socket.close().await;
            return;
        }
    };

    let stream = pipeline.speak(prompt);
    forward_chunks(&mut socket, stream, state.codec).await;
    let _ = socket.close().await;
}

/// Drive the chunk stream → ws, racing stream output against the client's
/// recv half. If the peer sends Close, `{"type":"cancel"}`, or drops, we
/// break early — `stream` is dropped on return, which cancels 1bit-voice's
/// outstanding LLM + TTS work.
async fn forward_chunks(
    socket: &mut WebSocket,
    mut stream: Pin<Box<dyn Stream<Item = Result<VoiceChunk>> + Send>>,
    codec: Codec,
) {
    let mut sent: usize = 0;
    loop {
        tokio::select! {
            biased;
            incoming = socket.recv() => match incoming {
                None | Some(Ok(Message::Close(_))) | Some(Err(_)) => {
                    tracing::debug!(chunks = sent, "client dropped mid-stream after {sent} chunks sent");
                    return;
                }
                Some(Ok(Message::Text(t))) if t.contains("\"cancel\"") => {
                    tracing::debug!(chunks = sent, "client dropped mid-stream after {sent} chunks sent (cancel)");
                    return;
                }
                Some(Ok(_)) => continue,
            },
            next = stream.next() => match next {
                Some(Ok(c)) => {
                    if !send_chunk(socket, &c, codec).await { return; }
                    sent += 1;
                }
                Some(Err(e)) => {
                    tracing::warn!(error = %e, "voice pipeline error");
                    let _ = socket.send(Message::Text(format!("error: {e}"))).await;
                    return;
                }
                None => return,
            },
        }
    }
}

async fn send_chunk(socket: &mut WebSocket, c: &VoiceChunk, codec: Codec) -> bool {
    match codec {
        Codec::Wav => socket.send(Message::Binary(c.wav.to_vec())).await.is_ok(),
        Codec::Opus => match encode_wav_to_opus(&c.wav, TARGET_SR) {
            Ok(packets) => {
                for pkt in packets {
                    if socket.send(Message::Binary(pkt)).await.is_err() {
                        return false;
                    }
                }
                true
            }
            Err(e) => {
                tracing::warn!(error = %e, "opus encode failed; dropping chunk");
                let _ = socket.send(Message::Text(format!("error: {e}"))).await;
                true
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

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
            codec: Codec::Opus,
        };
        assert_eq!(s.voice_cfg.llm_url, cfg.llm_url);
        assert_eq!(s.voice_cfg.tts_url, cfg.tts_url);
        assert_eq!(s.voice_cfg.voice, "bf_emma");
        assert_eq!(s.codec, Codec::Opus);
    }

    #[test]
    fn construct_with_custom_bind_is_noop_until_run() {
        let s = EchoServer {
            bind: "127.0.0.1:0".parse().unwrap(),
            voice_cfg: VoiceConfig::default(),
            codec: Codec::Wav,
        };
        assert_eq!(s.bind.port(), 0);
        let d = EchoServer::default();
        assert_eq!(d.bind, default_bind());
        assert_eq!(d.codec, Codec::Wav);
    }

    #[test]
    fn codec_from_str_roundtrip() {
        assert_eq!(Codec::from_str("wav").unwrap(), Codec::Wav);
        assert_eq!(Codec::from_str("opus").unwrap(), Codec::Opus);
        assert!(Codec::from_str("mp3").is_err());
    }

    #[test]
    fn preamble_opus_shape() {
        let v: serde_json::Value = serde_json::from_str(&preamble_json(Codec::Opus)).unwrap();
        assert_eq!(v["codec"], "opus");
        assert_eq!(v["sample_rate"], 48_000);
        assert_eq!(v["channels"], 1);
        assert_eq!(v["frame_ms"], 20);
    }

    /// Protocol smoke test: bring up `EchoServer` with `Codec::Opus` on
    /// an ephemeral port, connect with tokio-tungstenite, send a prompt,
    /// and assert the first inbound frame is the text-JSON preamble. The
    /// voice pipeline is pointed at a dead loopback URL so we don't need
    /// 1bit-server / halo-kokoro — the preamble is sent BEFORE the
    /// pipeline runs, so it still arrives.
    #[tokio::test]
    async fn opus_mode_sends_text_preamble_first() {
        use futures::SinkExt;
        use tokio_tungstenite::tungstenite::Message as TMessage;

        let voice_cfg = VoiceConfig {
            llm_url: "http://127.0.0.1:1/v1/chat/completions".into(),
            tts_url: "http://127.0.0.1:1/tts".into(),
            timeout_secs: 2,
            ..Default::default()
        };
        let state = AppState {
            voice_cfg: Arc::new(voice_cfg),
            codec: Codec::Opus,
        };
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let bound = listener.local_addr().unwrap();
        let app = Router::new()
            .route("/ws", get(ws_handler))
            .with_state(state);
        let server_handle = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        // Connect with a short retry since axum::serve is async to start.
        let url = format!("ws://{bound}/ws");
        let (mut ws, _) = {
            let mut attempt = 0;
            loop {
                match tokio_tungstenite::connect_async(&url).await {
                    Ok(pair) => break pair,
                    Err(_) if attempt < 10 => {
                        attempt += 1;
                        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                    }
                    Err(e) => panic!("ws connect failed: {e}"),
                }
            }
        };

        ws.send(TMessage::Text("hello".into())).await.unwrap();
        let first = tokio::time::timeout(std::time::Duration::from_secs(5), ws.next())
            .await
            .expect("preamble timeout")
            .expect("stream ended")
            .expect("ws error");
        let text = match first {
            TMessage::Text(t) => t,
            other => panic!("expected Text preamble, got {other:?}"),
        };
        let v: serde_json::Value = serde_json::from_str(&text).expect("preamble is valid JSON");
        assert_eq!(v["codec"], "opus");
        assert_eq!(v["sample_rate"], 48_000);

        server_handle.abort();
    }

    /// Client disconnects after the preamble. We assert 1bit-voice's
    /// stream is dropped promptly — `DroppedFlag` flips on drop. Runs with
    /// a synthetic stream (no real backends) so it's always on.
    #[tokio::test]
    async fn cancel_on_client_disconnect() {
        use axum::extract::ws::WebSocketUpgrade;
        use bytes::Bytes;
        use futures::SinkExt;
        use onebit_voice::VoiceChunk;
        use std::sync::atomic::{AtomicBool, Ordering};
        use tokio_tungstenite::tungstenite::Message as TMessage;

        // Drop-detecting sentinel: stream holds it, drop flips the flag.
        struct DropGuard(Arc<AtomicBool>);
        impl Drop for DropGuard {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }

        let dropped = Arc::new(AtomicBool::new(false));
        let dropped_clone = dropped.clone();

        async fn mock_handler(
            ws: WebSocketUpgrade,
            State(guard): State<Arc<AtomicBool>>,
        ) -> impl IntoResponse {
            ws.on_upgrade(move |mut socket| async move {
                let _ = socket.recv().await; // prompt
                let _ = socket
                    .send(Message::Text(preamble_json(Codec::Wav)))
                    .await;
                let sentinel = DropGuard(guard);
                let stream = Box::pin(async_stream::stream! {
                    let _hold = sentinel; // moves into the stream; drops with it
                    // First chunk fires immediately; loop then parks forever.
                    let wav = Bytes::from_static(&[0u8; 44]);
                    yield Ok::<_, anyhow::Error>(VoiceChunk { index: 0, sentence: String::new(), wav });
                    loop { tokio::time::sleep(std::time::Duration::from_secs(60)).await; }
                });
                forward_chunks(&mut socket, stream, Codec::Wav).await;
                let _ = socket.close().await;
            })
        }

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let bound = listener.local_addr().unwrap();
        let app = Router::new()
            .route("/ws", get(mock_handler))
            .with_state(dropped_clone);
        let server_handle = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let url = format!("ws://{bound}/ws");
        let (mut ws, _) = loop {
            match tokio_tungstenite::connect_async(&url).await {
                Ok(p) => break p,
                Err(_) => tokio::time::sleep(std::time::Duration::from_millis(20)).await,
            }
        };
        ws.send(TMessage::Text("hi".into())).await.unwrap();
        // Drain preamble + first chunk, then drop the client.
        let _ = tokio::time::timeout(std::time::Duration::from_secs(2), ws.next()).await;
        let _ = tokio::time::timeout(std::time::Duration::from_secs(2), ws.next()).await;
        drop(ws);

        // Server should notice disconnect and drop the stream quickly.
        for _ in 0..50 {
            if dropped.load(Ordering::SeqCst) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        assert!(
            dropped.load(Ordering::SeqCst),
            "VoicePipeline stream was not dropped after client disconnect"
        );
        server_handle.abort();
    }
}
