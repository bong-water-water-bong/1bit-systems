//! 1bit-voice — CLI that takes a prompt on stdin, streams audio
//! chunks to stdout as they arrive. Pipe into `paplay` / `aplay` /
//! `ffplay` for the speakers.
//!
//! ```bash
//! echo "tell me a one-sentence joke" | 1bit-voice --prompt - | paplay --raw
//! ```
//!
//! Or dump to disk:
//!
//! ```bash
//! 1bit-voice --prompt "explain rust" --out /tmp/reply.wav
//! ```

use anyhow::{Context, Result};
use clap::Parser;
use futures::StreamExt;
use onebit_voice::{VoiceConfig, VoicePipeline};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "1bit-voice", about, version)]
struct Cli {
    /// Prompt text. Use `-` to read from stdin.
    #[arg(long)]
    prompt: String,
    /// 1bit-server base URL
    #[arg(long, default_value = "http://127.0.0.1:8180/v1/chat/completions")]
    llm_url: String,
    /// Model id
    #[arg(long, default_value = "1bit-monster-2b")]
    model: String,
    /// halo-kokoro /tts URL
    #[arg(long, default_value = "http://127.0.0.1:8083/tts")]
    tts_url: String,
    /// Voice id
    #[arg(long, default_value = "af_sky")]
    voice: String,
    /// Max generation tokens
    #[arg(long, default_value_t = 256)]
    max_tokens: u32,
    /// Sampling temperature (0 = greedy)
    #[arg(long, default_value_t = 0.7)]
    temperature: f32,
    /// Write each chunk to this file (concatenated WAV). Omit = stdout.
    #[arg(long)]
    out: Option<PathBuf>,
    /// Log sentence-by-sentence latency to stderr
    #[arg(long)]
    timing: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "onebit_voice=info".into()),
        )
        .with_target(false)
        .init();

    let cli = Cli::parse();

    let prompt = if cli.prompt == "-" {
        let mut buf = String::new();
        std::io::stdin().read_to_string(&mut buf)?;
        buf.trim().to_string()
    } else {
        cli.prompt.clone()
    };

    let cfg = VoiceConfig {
        llm_url: cli.llm_url,
        model: cli.model,
        max_tokens: cli.max_tokens,
        temperature: cli.temperature,
        tts_url: cli.tts_url,
        voice: cli.voice,
        timeout_secs: 120,
    };

    let pipeline = VoicePipeline::new(cfg).context("build voice pipeline")?;
    let start = Instant::now();
    let mut stream = pipeline.speak(prompt);

    let mut sink: Box<dyn Write> = if let Some(path) = cli.out.as_ref() {
        Box::new(std::fs::File::create(path).with_context(|| format!("create {}", path.display()))?)
    } else {
        Box::new(std::io::stdout().lock())
    };

    let mut total_bytes = 0usize;
    let mut n_chunks = 0usize;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if cli.timing {
            eprintln!(
                "[1bit-voice] chunk {} ({:>4} ms since start, {} bytes wav) sentence=\"{}\"",
                chunk.index,
                start.elapsed().as_millis(),
                chunk.wav.len(),
                chunk.sentence,
            );
        }
        sink.write_all(&chunk.wav)?;
        sink.flush()?;
        total_bytes += chunk.wav.len();
        n_chunks += 1;
    }
    if cli.timing {
        eprintln!(
            "[1bit-voice] done · {} chunks · {} bytes · {:.2} s wall",
            n_chunks,
            total_bytes,
            start.elapsed().as_secs_f32(),
        );
    }
    Ok(())
}
