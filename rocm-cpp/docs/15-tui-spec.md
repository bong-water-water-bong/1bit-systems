# 15 — `bitnet_tui` spec sheet (terminal interface)

Living spec for the FTXUI **terminal / over-SSH** frontend on
`librocm_cpp`. The desktop interface (floating tabs, frosted glass,
GPU-shaded wallpaper) is a companion spec at
[`docs/17-desktop-gui-spec.md`](17-desktop-gui-spec.md) — Dear ImGui +
SDL2, talks to the same HTTP server. One backend, two renderers.

Contributors: pick a milestone, file a PR. Keep scope tight — this doc
is the contract.

## Contribution rules (read first)

If you want to add to this project, these are non-negotiable:

1. **No MLX.** Not as a dep, not as a fallback, not as a test shim, not
   as a "what if we supported both" patch. rocm-cpp is post-MLX by design.
2. **BitNet only.** Ternary / 1.58-bit is the model class this stack
   exists for. No FP16 LLMs, no INT4, no GGUF glue, no quantization
   scheme substitutes. If a feature makes sense for BitNet it ships; if
   it's a generalization to other quant, it goes in another project.
3. **No Python at runtime.** Build-time tooling is fine; inference is C++.
4. **UI stack is fixed.** **FTXUI** for the terminal / SSH interface
   (this spec). **Dear ImGui + SDL2** for the desktop GUI (see spec 17).
   No Qt, no web, no Electron, no browser layer.
5. **The install script stays working.** If your PR breaks `install.sh`
   on a clean Strix Halo box, it doesn't land.

PRs that violate any of the above will be closed without debate. These
are the lane markers, not a style preference — they're what makes this
stack the 1-bit blaster instead of another framework-adapter.

## Two pages, one binary

The TUI has **two distinct experiences**, switched with a page toggle
(default `F1` / `F2`):

| Page | Purpose | Feel |
|---|---|---|
| **Main** | Full controls, all boilerplate — the serious working surface. Inspect state, drive inference, read stats. | Tool. Dense. All-keyboard. |
| **Man Cave** | Pure fun. Full voice-interactive agent (Muse). Conversation, vibes, no knobs. | Experience. Minimal. Voice-first. |

The two pages share the same loaded model, the same KV caches, and the
same `librocm_cpp` backend. Switching pages doesn't reload anything.

## Goals (both pages)

1. One binary, no X11, works over SSH.
2. Wraps the existing `forward_token` decode loop — the TUI does **not**
   reimplement inference.
3. FTXUI only — no Qt, no web, no browser layer.

## Non-goals

- No training, no QAT.
- No multi-model hot-swap in v1.
- No remote control / HTTP server (belongs in a different binary).

## Stack

| Layer | Choice | Why |
|---|---|---|
| UI library | [FTXUI](https://github.com/ArthurSonzogni/FTXUI) | MIT, header-only core, C++20, no X11 |
| Backend | `librocm_cpp.so` via `rocm_cpp/ck_gemm.h` + `rocm_cpp/bitnet_model.h` | our own kernels |
| Build | CMake `FetchContent(FTXUI)` inside the existing top-level CMakeLists.txt | zero extra setup |
| Tokenizer | `halo-1bit/models/*.htok` (LLaMA 3 BPE, load via a small C++ helper) | matches the .h1b model |
| Voice (Man Cave only) | Whisper.cpp (STT) + Kokoro TTS (existing Strix Halo services) | already running in the mesh |

Link: `target_link_libraries(bitnet_tui PRIVATE rocm_cpp ftxui::screen ftxui::dom ftxui::component)`.

## Page 1 — Main (controls / boilerplate)

```
┌─ rocm-cpp // halo-1bit-2b-absmean.h1b ──────────────── 82.5 tok/s ─┐
│                                                                    │
│ ┌─ CHAT ─────────────────────────────────┐┌─ STATS ──────────────┐ │
│ │ > Hello, how are you?                  ││  tok/s      :  82.5  │ │
│ │ I am doing well, thanks for asking.    ││  last tok   : 12.1ms │ │
│ │                                        ││  ctx len    :    17  │ │
│ │                                        ││  vram used  : 1.8 GB │ │
│ │                                        ││                      │ │
│ │                                        ││  LAYER TIMINGS       │ │
│ │                                        ││  input_norm : 0.08ms │ │
│ │                                        ││  q/k/v proj : 0.64ms │ │
│ │                                        ││  attn       : 0.91ms │ │
│ │                                        ││  o proj     : 0.22ms │ │
│ │                                        ││  ffn (fused): 1.83ms │ │
│ │                                        ││  × 30 layers         │ │
│ └────────────────────────────────────────┘└──────────────────────┘ │
│ ┌─ KV CACHE HEATMAP (30 layers × seq) ───────────────────────────┐ │
│ │ L0  ████████▓▓▓▓░░░░░░░░░░░░░░░░░░░░                           │ │
│ │ L15 ██████▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░                           │ │
│ │ L29 ██████▓▓░░░░░░░░░░░░░░░░░░░░░░░░                           │ │
│ └────────────────────────────────────────────────────────────────┘ │
│ > _                                                                │
└─ F1 main · F2 man-cave · TAB pane · / prompt · Ctrl-R reset · q ──┘
```

### Main components

- **ChatPanel** — append-only decoded-token stream, user lines prefixed `> `.
- **StatsPanel** — rolling tok/s, context length, VRAM via `hipMemGetInfo`,
  per-layer timings from `hipEvent` pairs around each phase.
- **KvHeatmap** — 30 rows × seq_len columns, intensity = rolling L2 norm
  of `K_caches[l][pos,:]`. One hipMemcpy per token into a host staging buf.
- **PromptBar** — FTXUI `Input`, on Enter → tokenize → `forward_token` loop.

### Main key bindings

| Key | Action |
|---|---|
| `F1` / `F2` | Switch page (Main / Man Cave) |
| `/` | Focus prompt bar |
| `Enter` (in prompt) | Send, start generation |
| `Ctrl-C` / `Esc` (during gen) | Stop generation, keep KV state |
| `Ctrl-R` | Clear chat + reset KV caches (hipMemset) |
| `Tab` / `Shift-Tab` | Cycle focus between panels |
| `q` | Quit |

## Page 2 — Man Cave (voice / Muse)

Pure fun. Zero controls on screen by default. You walk in, say something,
Muse talks back. That's the whole interface.

```
┌─ MAN CAVE ──────────────────────────────────────── 🎤 listening ──┐
│                                                                    │
│                                                                    │
│                                                                    │
│                   ╭─── MUSE ────╮                                 │
│                   │             │                                 │
│                   │   ~  ~  ~   │   ← waveform while she talks    │
│                   │    ~~~~~~   │                                 │
│                   │             │                                 │
│                   ╰─────────────╯                                 │
│                                                                    │
│        " What's on your mind tonight, architect? "                 │
│                                                                    │
│                                                                    │
│                                                                    │
│                                                                    │
└─ F1 main · hold-space push-to-talk · m mute · q quit ─────────────┘
```

### Man Cave components

- **MuseVisual** — centered animated ASCII "orb" that breathes while
  idle, becomes a waveform while Muse is speaking (driven by TTS
  amplitude samples over a unix socket from Kokoro).
- **CaptionLine** — last Muse utterance rendered in muted text, fades
  out after a few seconds. No scrollback by default (press `h` for
  history overlay).
- **ListeningIndicator** — top-right, shows mic state: idle / listening
  / thinking / speaking.

### Voice loop

```
mic → whisper.cpp STT (localhost) → text prompt
   → bitnet_decode forward_token loop (Muse persona prepended)
   → streamed tokens → Kokoro TTS (localhost) → speaker
```

Muse persona system prompt lives in `tools/muse_prompt.txt` — loaded at
startup, prepended to every conversation. Kept short (< 200 tokens) to
preserve KV budget for actual talk.

### Man Cave key bindings

| Key | Action |
|---|---|
| `F1` | Switch to Main page |
| `space` (hold) | Push-to-talk (if hands-free detection is off) |
| `m` | Mute mic (pause listening) |
| `h` | Toggle history overlay |
| `q` | Quit |

## Data flow (shared)

```
main()
  ├─ rcpp_bitnet_load_h1b()               // same as bitnet_decode
  ├─ TokenStream  (bounded mpmc queue)    // producer: decode thread
  │                                        // consumer: FTXUI render thread
  ├─ LayerTimings (6 × ring<64, float>)   // hipEvent timing per phase
  ├─ ChatModel    (std::vector<Line>)
  ├─ VoiceBridge  (opt. unix sockets to whisper.cpp + Kokoro)
  └─ ScreenInteractive::Loop(...)
       │
       ├─ Render()         60 Hz soft-limit (FTXUI sleeps between frames)
       └─ OnEvent(...)     key handling, prompt submit, page switch
```

Inference runs on a dedicated std::thread, talks to the UI only through
the TokenStream queue and atomic<double> tok/s counter. FTXUI stays on
the main thread. No mutex held across a `hipMemcpy`.

## Milestones (pick one, PR it)

1. **M1 — skeleton** — `bitnet_tui` builds, shows two empty pages
   (Main, Man Cave), `F1`/`F2` switches, loads .h1b, quits cleanly.
2. **M2 — Main: chat** — prompt bar → tokenizer → `forward_token` loop
   → detokenize → append to ChatPanel. No stats.
3. **M3 — Main: stats** — rolling tok/s + per-layer `hipEvent` timings
   into StatsPanel. Requires minor instrumentation hooks in
   `bitnet_decode.cpp` (move the forward loop into a header so the TUI
   can reuse it with a timings callback).
4. **M4 — Main: KV heatmap** — KvHeatmap component, host staging, one
   hipMemcpy per token.
5. **M5 — Man Cave: visuals** — MuseVisual orb + breathing + caption
   line. Still typed input — no voice yet.
6. **M6 — Man Cave: voice loop** — Whisper.cpp STT socket + Kokoro TTS
   socket wiring. Push-to-talk first, continuous listening behind a flag.
7. **M7 — polish** — themes, help overlays, save-chat, keybinding
   remap via a tiny config.

Each milestone is a separate PR. Don't bundle.

## Non-obvious details

- **BPE tokenizer**: `halo-1bit/models/*.htok` stores the LLaMA 3 BPE
  vocab + merges. A single-file C++ implementation (~300 LOC) is in
  scope for the M2 PR; alternatively link against a minimal MIT-licensed
  BPE library. **No Python at runtime** — that bar stays.
- **Thread-safety of `rcpp_*`**: Assume single-thread for v1; all
  librocm_cpp calls from the decode thread only.
- **Voice services**: Whisper.cpp and Kokoro are already running as
  systemd user services on the architect's Strix Halo mesh. The Man
  Cave page talks to them over localhost sockets — it does not embed
  them.
- **Directory**: `tools/bitnet_tui.cpp` + helpers in `tools/tui/`.
  Promote to a top-level `tui/` dir only if it grows past ~2k LOC.

## Open questions

- Should the Muse persona live in this repo or in `halo-ai-core`? Default:
  this repo for now, move later if halo-ai agents unify.
- Do we want sampling knobs (temperature, top-k) in Main page? Yes in M7.
