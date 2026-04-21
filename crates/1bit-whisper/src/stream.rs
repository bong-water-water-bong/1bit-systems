//! `WhisperStream` — non-blocking streaming façade over [`WhisperEngine`].
//!
//! This module wraps the existing blocking engine in a background worker
//! thread so that the audio producer (mic, WebSocket, test fixture…)
//! never stalls on `whisper_full`. The audio producer feeds PCM16 into a
//! bounded channel; the worker drains the channel, drives the engine's
//! sliding-window scheduler every ~500 ms, and pushes fresh
//! [`Partial`]s back out through a second bounded channel that
//! [`WhisperStream::try_recv`] polls without blocking.
//!
//! # Threading model
//!
//! ```text
//!  caller                 bounded(pcm_cap)              worker thread
//!  ┌──────────┐  feed()   ┌────────────────┐  recv  ┌──────────────────┐
//!  │feeder    │──────────▶│ flume::bounded │───────▶│ WhisperEngine    │
//!  │(audio    │           └────────────────┘        │  - accum 500 ms  │
//!  │ thread)  │                                     │  - feed() to     │
//!  │          │◀────────┐                           │    real/stub     │
//!  └──────────┘  try_recv│  ┌────────────────┐      │  - drain_partials│
//!                        └──│ flume::bounded │◀─────│                  │
//!                           └────────────────┘      └──────────────────┘
//! ```
//!
//! * **PCM channel** is bounded so backpressure surfaces as `feed`
//!   returning `false` rather than unbounded memory growth. Default
//!   depth is 64 chunks ≈ 8 s at 125 ms per chunk.
//! * **Partial channel** is bounded the same way; if the caller stops
//!   draining, the worker starts dropping oldest partials so it never
//!   blocks on the text-emit side.
//! * **Drop** closes the PCM channel; the worker observes the disconnect
//!   on its next `recv_timeout` and exits cleanly. The [`Drop`] impl
//!   does not `join` the worker — the user's pressure test relies on
//!   the fact that dropping the stream returns immediately even if the
//!   worker is mid-tick. The `JoinHandle` is carried as an `Option` and
//!   dropped without join; the thread self-terminates shortly after.
//!
//! # Scheduling cadence
//!
//! The worker accumulates PCM until it holds ≥ `STEP_SAMPLES` (500 ms at
//! 16 kHz = 8000 samples), then hands the whole accumulated buffer to
//! the inner [`WhisperEngine::feed`] and calls `drain_partials`. This
//! matches the plan's "fixed sliding window" policy
//! (`docs/wiki/Halo-Whisper-Streaming-Plan.md` §2). The real
//! sliding-window dance — prepending `keep_ms` of previous audio and
//! running `whisper_full` — lives in `cpp/shim.cpp`; this module's job
//! is only to pump the right-shaped PCM into the shim at the right
//! cadence and lift partials back into a non-blocking Rust API.
//!
//! # Latency target
//!
//! Voice-latency memory (`project_voice_latency_sharding.md`, 2026-04-20)
//! frames today's mouth-to-ear budget at **1–3 s**, with whisper partials
//! specifically sized to cut STT latency from ~300 ms batch to ~80 ms
//! streamed. The target first-audio-to-first-partial budget for
//! `WhisperStream` is **≤ 1.3 s** end-to-end (STT side of the 1 s
//! Tailscale-LAN mouth-to-ear target from the same memo). Partials beyond
//! the first should land at the ~500 ms step cadence set by
//! [`STEP_SAMPLES`].
//!
//! # Feature interaction
//!
//! Under the default `stub` feature the inner `WhisperEngine::new` never
//! returns `Ok`, so [`WhisperStream::new`] short-circuits with
//! [`WhisperError::UnsupportedStub`] and never spawns a worker. Under
//! `real-whisper` the worker is live. Consumers get the same API either
//! way; the feature choice only decides whether inference happens.

use crate::{Partial, WhisperEngine, WhisperError};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// 16 kHz × 500 ms = 8000 samples. Matches the `step_ms` default in the
/// upstream `whisper.cpp/examples/stream/stream.cpp` and the plan doc.
const STEP_SAMPLES: usize = 8_000;

/// Default PCM queue depth: 64 chunks. At ~125 ms per chunk this is
/// ~8 s of audio before backpressure kicks in.
const PCM_CHANNEL_CAP: usize = 64;

/// Default partial queue depth: 64 partials. A partial is roughly one
/// per 500 ms tick, so this holds ~32 s of emitted text.
const PARTIAL_CHANNEL_CAP: usize = 64;

/// Poll timeout for the PCM channel on the worker side. Short enough
/// that a `Drop` on the main thread is observed within ~100 ms.
const WORKER_RECV_TIMEOUT: Duration = Duration::from_millis(100);

/// Non-blocking streaming handle over a [`WhisperEngine`].
///
/// Spawns a worker thread on construction; the worker terminates when
/// this struct is dropped. See the [module docs](self) for the
/// threading + cadence model and the 1.3 s first-partial latency target
/// cited in the voice-latency memo.
pub struct WhisperStream {
    pcm_tx: Option<flume::Sender<Vec<i16>>>,
    partial_rx: flume::Receiver<Partial>,
    /// Cooperative shutdown flag. Set in [`Drop`]; the worker checks it
    /// on every tick so it exits promptly even if it just received a
    /// chunk.
    shutdown: Arc<AtomicBool>,
    /// `Option` so we can `take()` it out if we ever want to join; today
    /// we deliberately drop without joining (see module docs — the
    /// pressure-loop test depends on this non-blocking drop).
    worker: Option<JoinHandle<()>>,
}

impl WhisperStream {
    /// Open a streaming session backed by a fresh [`WhisperEngine`].
    ///
    /// Under the default `stub` feature this returns
    /// [`WhisperError::UnsupportedStub`] without spawning a worker.
    pub fn new<P: AsRef<Path>>(model: P) -> Result<Self, WhisperError> {
        let engine = WhisperEngine::new(model)?;
        Ok(Self::from_engine(engine))
    }

    /// Wrap an already-constructed [`WhisperEngine`] in a streaming
    /// worker. Exposed separately so tests (and future callers who want
    /// to share a single engine across streams) can drive the plumbing
    /// without going through `WhisperEngine::new`.
    pub fn from_engine(engine: WhisperEngine) -> Self {
        let (pcm_tx, pcm_rx) = flume::bounded::<Vec<i16>>(PCM_CHANNEL_CAP);
        let (partial_tx, partial_rx) = flume::bounded::<Partial>(PARTIAL_CHANNEL_CAP);
        let shutdown = Arc::new(AtomicBool::new(false));

        let worker_shutdown = Arc::clone(&shutdown);
        let worker = thread::Builder::new()
            .name("onebit-whisper-stream".into())
            .spawn(move || worker_loop(engine, pcm_rx, partial_tx, worker_shutdown))
            .expect("spawn whisper stream worker");

        Self {
            pcm_tx: Some(pcm_tx),
            partial_rx,
            shutdown,
            worker: Some(worker),
        }
    }

    /// Push mono 16 kHz s16le PCM into the worker.
    ///
    /// Non-blocking-ish: if the PCM channel is full the call waits for
    /// space for a short bounded window (8 ms) and then gives up. The
    /// return value is `true` if the chunk was accepted, `false` if it
    /// was dropped due to backpressure. The return is deliberately
    /// `bool` rather than `Result` because backpressure drops are a
    /// *policy* decision, not an error — a dropped chunk just means the
    /// model is behind the mic and we'd rather skip audio than stall
    /// the producer.
    ///
    /// Under the stub-only build the channel is always open but the
    /// worker discards everything it receives; feeding is a no-op from
    /// a transcription standpoint.
    pub fn feed(&mut self, pcm: &[i16]) -> bool {
        let Some(tx) = self.pcm_tx.as_ref() else {
            return false;
        };
        // Allocate once per chunk; the worker owns the buffer from here.
        // Channel bound is sample-chunk-count, not sample-count, so large
        // chunks don't distort the queue depth.
        let owned = pcm.to_vec();
        match tx.send_timeout(owned, Duration::from_millis(8)) {
            Ok(()) => true,
            Err(flume::SendTimeoutError::Timeout(_)) => false,
            Err(flume::SendTimeoutError::Disconnected(_)) => false,
        }
    }

    /// Drain one partial if one is ready. Returns `None` immediately if
    /// the worker has not emitted since the last call.
    ///
    /// This is the non-blocking half of the trait — safe to poll at
    /// 60 Hz from the audio thread.
    pub fn try_recv(&mut self) -> Option<Partial> {
        match self.partial_rx.try_recv() {
            Ok(p) => Some(p),
            Err(flume::TryRecvError::Empty) => None,
            Err(flume::TryRecvError::Disconnected) => None,
        }
    }

    /// Drain every partial currently buffered. Convenience for callers
    /// that poll less frequently than the worker emits.
    pub fn drain(&mut self) -> Vec<Partial> {
        let mut out = Vec::new();
        while let Some(p) = self.try_recv() {
            out.push(p);
        }
        out
    }
}

impl Drop for WhisperStream {
    fn drop(&mut self) {
        // Flip the cooperative-shutdown bit so the worker exits on its
        // next poll.
        self.shutdown.store(true, Ordering::Release);
        // Dropping the sender disconnects the PCM channel; the worker's
        // `recv_timeout` will surface `Disconnected` within one
        // `WORKER_RECV_TIMEOUT`. We intentionally do NOT join the
        // worker here — the streaming smoke test needs `drop` to return
        // promptly under pressure, and the worker holds no resources
        // whose cleanup callers depend on synchronously (the Drop impl
        // on WhisperEngine / RealEngine is what calls
        // `onebit_whisper_free`, and it runs on the worker thread once
        // `engine` goes out of scope there).
        self.pcm_tx.take();
        // Leave `worker` as-is; dropping the JoinHandle detaches the
        // thread and it self-terminates.
        let _ = self.worker.take();
    }
}

/// Worker-thread entry point. Owns the [`WhisperEngine`] for its whole
/// lifetime; exits when the PCM channel disconnects or the shutdown
/// flag goes high.
fn worker_loop(
    mut engine: WhisperEngine,
    pcm_rx: flume::Receiver<Vec<i16>>,
    partial_tx: flume::Sender<Partial>,
    shutdown: Arc<AtomicBool>,
) {
    // Accumulator for audio between step ticks. Allocated once.
    let mut accum: Vec<i16> = Vec::with_capacity(STEP_SAMPLES * 2);

    loop {
        if shutdown.load(Ordering::Acquire) {
            break;
        }

        match pcm_rx.recv_timeout(WORKER_RECV_TIMEOUT) {
            Ok(chunk) => accum.extend_from_slice(&chunk),
            Err(flume::RecvTimeoutError::Timeout) => { /* fall through to shutdown + step check */ }
            Err(flume::RecvTimeoutError::Disconnected) => break,
        }

        // Only hand audio to the engine once we've crossed the step
        // boundary — a 500 ms window is what the plan's sliding
        // scheduler (and upstream `stream.cpp`) operates on.
        if accum.len() >= STEP_SAMPLES {
            // The engine's shim owns the sliding-window state; we
            // don't try to overlap `keep_ms` in Rust. Just hand over
            // what we have and clear.
            if let Err(e) = engine.feed(&accum) {
                // Feed failures are typically UnsupportedStub (harmless
                // — this is the stub-feature no-op path) or a shim
                // error. In either case we drop the audio and keep
                // going; a noisy error log on every tick would spam
                // the caller under stub, so only log non-stub errors.
                if !matches!(e, WhisperError::UnsupportedStub) {
                    tracing::warn!(target: "onebit_whisper::stream", error = %e, "feed failed");
                }
            }
            accum.clear();

            match engine.drain_partials() {
                Ok(parts) => {
                    for p in parts {
                        // Non-blocking send: if the consumer is behind,
                        // drop the newest partial rather than blocking
                        // the worker. A stalled consumer is a caller
                        // bug; the worker's job is to stay responsive
                        // to the mic.
                        if let Err(flume::TrySendError::Disconnected(_)) = partial_tx.try_send(p) {
                            return;
                        }
                    }
                }
                Err(WhisperError::UnsupportedStub) => { /* stub feature — silent */ }
                Err(e) => {
                    tracing::warn!(target: "onebit_whisper::stream", error = %e, "drain failed");
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Under the stub feature, constructing a stream surfaces the same
    /// `UnsupportedStub` error as the underlying engine — no worker
    /// thread is spawned, nothing leaks.
    #[cfg(all(feature = "stub", not(feature = "real-whisper")))]
    #[test]
    fn stub_new_returns_unsupported_without_spawn() {
        match WhisperStream::new("/does/not/matter.bin") {
            Ok(_) => panic!("stub stream should never construct successfully"),
            Err(WhisperError::UnsupportedStub) => { /* ok */ }
            Err(other) => panic!("expected UnsupportedStub, got {other:?}"),
        }
    }

    /// Constants sanity — keeps the sliding-window step honest against
    /// the plan doc (500 ms @ 16 kHz = 8000 samples).
    #[test]
    fn step_is_500ms_at_16khz() {
        assert_eq!(STEP_SAMPLES, 16_000 / 2);
    }
}
