# 17 — `halo_gui` spec sheet (desktop interface)

Companion spec to [`docs/15-tui-spec.md`](15-tui-spec.md). Same backend
(librocm_cpp HTTP server), different renderer. This is the **desktop /
pixel** interface; the terminal interface stays FTXUI.

## Stack

**Fixed, not up for debate per the stack contribution rules:**

| Component | Choice | Why |
|---|---|---|
| Window / input | SDL2 | zlib, C native, Windows + Linux + macOS, minimal dep |
| UI immediate-mode | Dear ImGui | MIT, header-heavy C++, game-industry battle-tested, GPU-accelerated |
| Renderer | SDL_Renderer w/ SDL_GPU or OpenGL 3.3 (SDL2 built-in) | portability; custom shaders for frost/blur |
| JSON | nlohmann/json | MIT, already in the rocm-cpp tree |
| HTTP client | cpp-httplib | MIT, already in the tree |
| Shaders | GLSL 330 core | broadest driver coverage on gfx1151 + Intel Arc B580 |

No Qt. No Electron. No Tauri. No GTK.

## Landing page — floating agent tabs (the reason this spec exists)

The entry screen is **not** a chat window. It's a top-down view of the
agent colony. Each specialist in agent-cpp is a tile that drifts around
the screen at 0.5–2 char-equivalents per second; click one to focus;
focus expands that tile into a detail pane showing what the service does,
its current status, last message processed, counters.

```
┌─ halo-ai core ─────────────────────────── Ryzen AI Max+ 395 ──┐
│                                                                │
│                                                                │
│         ╭───────╮                        ╭───────────╮         │
│         │ muse  │                        │ cartograph│         │
│         ╰───────╯                        ╰───────────╯         │
│                     ╭──────╮                                   │
│                     │forge │          ╭──────╮                 │
│                     ╰──────╯          │herald│                 │
│                                       ╰──────╯                 │
│    ╭───────╮                                        ╭──────╮   │
│    │planner│                                        │scribe│   │
│    ╰───────╯      ╭─────────╮                       ╰──────╯   │
│                   │ sentinel│                                  │
│                   ╰─────────╯                                  │
│                                                                │
│      wallpaper: dynamic daily (frosted blur behind all tiles)  │
│      click a tile -> opens service detail window               │
└────────────────────────────────────────────────────────────────┘
```

### Visual design

- **Background layer**: daily-rotating wallpaper (pick from `~/Pictures/halo-wallpapers/` or rotate from Unsplash/NASA-APOD with a fallback).
- **Frost layer**: Gaussian or Kawase blur shader over the wallpaper; tiles float on top with ~40% alpha + border.
- **Tiles**: rounded-rect with soft shadow, 140×56 px typical, text centered. Color accent by category (cognition / voice / Discord / GitHub / ops).
- **Motion**: constrained random walk with gentle drift toward center-of-mass; collision avoidance so tiles don't pile up.
- **Focus**: clicking a tile freezes motion, shows detail pane with:
  - One-liner description
  - Running / idle / error status (from `/health` or an agent-cpp query endpoint)
  - Recent log tail
  - Counters (messages handled, last latency, tok/s if LLM-backed)
- **Escape** collapses back to floating mode.

### The chat page

F1 toggles to a chat page (same layout and behavior as `bitnet_tui`). That's the working surface once you know what you want to do. The landing page is the welcome + status view.

### Man Cave page

F2 page is voice-interactive Muse. Same spec as TUI Man Cave (docs/15 M5+), GPU-shaded visual of breathing orb / speech waveform using actual shader. Whisper.cpp STT + Kokoro TTS over unix sockets.

## Backend selection (addresses the "paid APIs on the landing page" note)

A pane/row on the landing page shows connection options — each as its own
floating tile, user clicks to swap:

```
  [ Local BitNet (default) ]  127.0.0.1:8080
  [ OpenAI ]                   needs OPENAI_API_KEY
  [ Anthropic ]                needs ANTHROPIC_API_KEY
  [ Groq ]                     needs GROQ_API_KEY
  [ DeepSeek ]                 needs DEEPSEEK_API_KEY
  [ Gemini ]                   needs GEMINI_API_KEY
  [ xAI Grok ]                 needs XAI_API_KEY
```

Implementation: a `Backend` struct with `{name, url, auth_header}`. Selection
writes `~/.config/halo-ai/backend.json`. All other panes talk through that
struct via the existing `LLMClient` shape. No special-casing per provider —
OpenAI-compat surface covers every one of those listed (Anthropic requires
a tiny request adapter).

## Shader — frost pass

Single-pass Kawase blur, 4 samples, run twice for ~9×9 effective kernel.
Runs at wallpaper-change time + every `window_resize` event (not per-frame
— cache the blurred texture). ~30 lines of GLSL. Tuned for gfx1151 + Intel
Arc B580 (both handle GLSL 330 cleanly).

```glsl
// frost.frag — kawase blur pass, approximate Gaussian with 4 taps
#version 330 core
in  vec2  v_uv;
out vec4  frag;
uniform sampler2D u_tex;
uniform vec2      u_pixel;
uniform float     u_offset;
void main() {
    vec4 s = texture(u_tex, v_uv + u_pixel * vec2( u_offset,  u_offset));
    s    += texture(u_tex, v_uv + u_pixel * vec2( u_offset, -u_offset));
    s    += texture(u_tex, v_uv + u_pixel * vec2(-u_offset,  u_offset));
    s    += texture(u_tex, v_uv + u_pixel * vec2(-u_offset, -u_offset));
    frag  = s * 0.25;
}
```

## Milestones

Each lands as a separate PR. Claim one in an issue before starting.

| M | Piece | Effort |
|---|---|---|
| 1 | Skeleton: SDL2 + ImGui + blank fullscreen window, quits cleanly | small |
| 2 | Wallpaper layer + frosted-blur shader caching | medium |
| 3 | Floating-tile sim (random walk + collision avoidance), specialist list sourced from an agent-cpp `/agents` HTTP endpoint | medium |
| 4 | Tile focus → detail pane, backend status via `/health` | small |
| 5 | Chat page (F1) — feature parity with `bitnet_tui` | medium |
| 6 | Backend selection UI + `~/.config/halo-ai/backend.json` persistence | small |
| 7 | Man Cave page (F2) — Muse orb with shader visualization, voice loop over unix sockets | large |

## File tree

```
halo-ai-desktop/              new repo OR tools/halo_gui/ inside rocm-cpp
  main.cpp                    SDL2 + ImGui loop
  wallpaper.cpp               textured quad, daily rotation
  frost.cpp / frost.frag      kawase blur pass
  agents_sim.cpp              floating tile physics
  detail_window.cpp           tile focus → service detail
  chat_page.cpp               F1 chat view
  mancave_page.cpp            F2 voice view
  backend_picker.cpp          provider selection
  CMakeLists.txt              FetchContent ImGui + SDL2
```

Directory decision: start as `rocm-cpp/tools/halo_gui/` for proximity to
the HTTP server code; promote to its own repo once it grows past ~2k LOC.
