---
phase: implementation
owner: anvil
---

# Crate: halo-echo

## Problem

Browser + mobile peers on the mesh need sub-second voice chat with halo-server. REST + polling doesn't cut it — Opus-framed WebSocket does. halo-echo wraps halo-voice's sentence-boundary streaming pipeline behind an axum WebSocket server so `wss://strixhalo.local/audio/ws` is a one-connection voice surface for any client that speaks the Opus-over-WS shape.

## Invariants

1. **First byte of audio lands under 1.5 s of prompt receipt.** Measured at the server wire, not the decoder. Today we hit 1.23 s end-to-end via halo-voice's sentence-boundary interleave.
2. **WAV passthrough is always available.** `--codec wav` flag keeps the raw halo-voice path working as a fallback when Opus encoding hits an edge case.
3. **No Python at runtime.** Opus encoding uses `audiopus` (libopus binding). No pydub, no ffmpeg subprocess.
4. **One connection, one session.** Voice state is connection-scoped; disconnection cancels outstanding halo-voice work.
5. **Bearer-gated.** The `/audio/ws` upgrade request requires a valid `Authorization: Bearer sk-halo-…` header matching `/etc/caddy/bearers.txt`. Caddy terminates TLS and forwards the Authorization header.

## Non-goals

- Not an audio recorder / logger. Audio flows through, never hits disk in the hot path.
- Not a voice-activity-detector (VAD) on the input side — that's halo-whisper's job, when streaming partials land.
- Not a broadcast server. One ws connection = one user session. No pub/sub.
- Not a WebRTC endpoint — plain WebSocket only, simpler, no ICE/STUN/TURN stack needed on the mesh.

## Interface

```rust
pub struct Codec { Wav, Opus }  // --codec flag maps here

pub struct EchoServer {
    pub bind: std::net::SocketAddr,
    pub voice_cfg: halo_voice::VoiceConfig,
    pub codec: Codec,
}

impl EchoServer {
    pub async fn run(self) -> anyhow::Result<()>;
}
```

Wire protocol on the `/ws` route:

- Client → Server:
  - First frame: text, the user prompt. UTF-8 string.
  - Subsequent frames: control (e.g., `{"type":"cancel"}`) — optional.
- Server → Client:
  - First frame: text JSON preamble
    `{"sample_rate":48000,"channels":1,"frame_ms":20,"codec":"opus"}`
  - Subsequent frames: binary, one Opus packet per frame (20 ms, 48 kHz mono, 24 kbps VBR).
  - On error: text JSON `{"error": "..."}` then close.

## Test matrix

| Invariant | Test |
|---|---|
| 1 (latency) | `opus_mode_sends_text_preamble_first` (tokio-tungstenite integration) |
| 2 (WAV fallback) | `codec_from_str_roundtrip`, both feature branches compile |
| 3 (no Python) | ldd on the release binary shows no `libpython*.so` — checked in CI |
| 4 (session scope) | cancellation on disconnect (manual; live-only today) |
| 5 (bearer gate) | Caddy route `/audio/* ` bearer-matcher inherited — tested at Caddy layer, not in the crate |

## TODO

- [x] Scaffold EchoServer + axum route (done 2026-04-20)
- [x] Opus encoder via audiopus + 48 kHz resampler
- [x] Preamble JSON frame
- [x] Binary opus frames
- [ ] Cancellation path on ws disconnect mid-stream — today halo-voice keeps running after client drops
- [ ] Live integration test against real halo-server + halo-kokoro behind `--features real-backend`
- [ ] Demote from implementation → analysis if the ternary-INT8 NPU path changes the latency math enough that we want halo-echo to gate on the NPU backend

## Spec cross-ref

| Spec section | Code file |
|---|---|
| Interface / Codec | `crates/halo-echo/src/lib.rs` |
| Interface / EchoServer | `crates/halo-echo/src/server.rs` |
| Opus encoding | `crates/halo-echo/src/opus.rs` |
| CLI flag plumbing | `crates/halo-echo/src/main.rs` |

## Phase: implementation

Phase promotes to `verified` once:
- Cancellation path closes halo-voice work on disconnect (open TODO above)
- Live integration test passes against running halo-server
- First-byte-of-audio latency measured live and logged into `docs/wiki/Benchmarks.md`

## Cross-refs

- `docs/wiki/SDD-Workflow.md` — phase gates
- `crates/halo-voice/` — upstream pipeline halo-echo wraps
- `crates/halo-kokoro/` — the actual TTS backend
- Deployed at `/audio/` via `strix-echo.service` (tracked systemd unit) on 127.0.0.1:8085
