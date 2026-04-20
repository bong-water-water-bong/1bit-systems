//! Live end-to-end integration test for `VoicePipeline`.
//!
//! Gated behind both `#[ignore]` and the `real-backend` cargo feature so
//! `cargo test --workspace` on CI / dev boxes without the stack up stays
//! green. Run by hand on strixhalo when both services are live:
//!
//! ```bash
//! cargo test -p halo-voice --release --features real-backend \
//!     -- --ignored --nocapture
//! ```
//!
//! Preconditions (asserted only implicitly via HTTP errors):
//!   * halo-server  listening on 127.0.0.1:8180 with /v1/chat/completions
//!   * halo-kokoro  listening on 127.0.0.1:8083 with /tts
//!
//! The whole future is wrapped in a 60 s `tokio::time::timeout` so a hung
//! backend can't wedge CI if someone accidentally flips the feature on.

#![cfg(feature = "real-backend")]

use std::time::{Duration, Instant};

use futures::StreamExt;
use halo_voice::{VoiceChunk, VoiceConfig, VoicePipeline};

#[tokio::test]
#[ignore = "needs live halo-server + halo-kokoro; run with --features real-backend --ignored"]
async fn speaks_two_sentences_against_live_stack() {
    let fut = async {
        let pipeline = VoicePipeline::new(VoiceConfig::default()).expect("build VoicePipeline");

        let start = Instant::now();
        let mut first_audio_at: Option<Duration> = None;
        let mut chunks: Vec<VoiceChunk> = Vec::new();

        let mut stream = pipeline.speak("Say hello in exactly two short sentences.");
        while let Some(item) = stream.next().await {
            let chunk = item.expect("live pipeline yielded an error");
            if first_audio_at.is_none() {
                first_audio_at = Some(start.elapsed());
            }
            chunks.push(chunk);
        }

        let wall = start.elapsed();
        let total_bytes: usize = chunks.iter().map(|c| c.wav.len()).sum();

        eprintln!("--- halo-voice live pipeline ---");
        eprintln!("chunks:            {}", chunks.len());
        eprintln!(
            "first-audio:       {:?}",
            first_audio_at.unwrap_or_default()
        );
        eprintln!("total wall time:   {:?}", wall);
        eprintln!("total audio bytes: {}", total_bytes);
        for (i, c) in chunks.iter().enumerate() {
            eprintln!("  [{}] {} bytes  sentence={:?}", i, c.wav.len(), c.sentence);
        }

        assert!(
            !chunks.is_empty(),
            "expected >=1 VoiceChunk from live pipeline"
        );
        assert!(
            chunks[0].wav.starts_with(b"RIFF"),
            "first chunk WAV must start with RIFF magic, got {:?}",
            &chunks[0].wav.get(..4)
        );
        assert!(
            total_bytes > 1000,
            "expected >1000 audio bytes total, got {}",
            total_bytes
        );
    };

    tokio::time::timeout(Duration::from_secs(60), fut)
        .await
        .expect("live pipeline test exceeded 60s timeout");
}
