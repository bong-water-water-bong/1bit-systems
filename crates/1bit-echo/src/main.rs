//! 1bit-echo — browser-side voice gateway.
//!
//! Runs an axum server on `--bind`. The browser opens a WebSocket to
//! `/ws`, sends a single text frame with the prompt, and gets back a
//! stream of binary audio frames driven by the existing 1bit-voice
//! pipeline. See the library docs for the (deferred) Opus plan.
//!
//! ```bash
//! 1bit-echo --bind 127.0.0.1:8085 \
//!           --llm-url http://127.0.0.1:8180/v1/chat/completions \
//!           --tts-url http://127.0.0.1:8083/tts \
//!           --voice af_sky
//! ```

use anyhow::{Context, Result};
use clap::Parser;
use onebit_echo::{Codec, EchoServer};
use onebit_voice::VoiceConfig;
use std::net::SocketAddr;

#[derive(Parser, Debug)]
#[command(name = "1bit-echo", about, version)]
struct Cli {
    /// Bind address for the WebSocket server.
    #[arg(long, default_value = "127.0.0.1:8085")]
    bind: SocketAddr,
    /// 1bit-server OpenAI-compat chat completions endpoint.
    #[arg(long, default_value = "http://127.0.0.1:8180/v1/chat/completions")]
    llm_url: String,
    /// halo-kokoro `/tts` endpoint.
    #[arg(long, default_value = "http://127.0.0.1:8083/tts")]
    tts_url: String,
    /// Voice id forwarded to kokoro.
    #[arg(long, default_value = "af_sky")]
    voice: String,
    /// Wire codec for audio frames. `wav` forwards the RIFF files from
    /// kokoro verbatim; `opus` re-encodes each chunk into 20 ms Opus
    /// packets at 48 kHz for browsers.
    #[arg(long, value_enum, default_value_t = Codec::Wav)]
    codec: Codec,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "onebit_echo=info".into()),
        )
        .with_target(false)
        .init();

    let cli = Cli::parse();

    let voice_cfg = VoiceConfig {
        llm_url: cli.llm_url,
        tts_url: cli.tts_url,
        voice: cli.voice,
        ..VoiceConfig::default()
    };

    let server = EchoServer {
        bind: cli.bind,
        voice_cfg,
        codec: cli.codec,
    };

    server
        .run()
        .await
        .context("1bit-echo server exited with error")
}
