---
phase: solutioning
owner: cartograph
renamed_from: halo-gaia
rename_date: 2026-04-20
---

# Crate: halo-helm

> Renamed from `halo-gaia` on 2026-04-20 to avoid clash with AMD GAIA
> (amd-gaia.ai). See `docs/wiki/AMD-GAIA-Integration.md`. The old name is
> retained as an install alias in `packages.toml` (`[component.gaia]` →
> `helm`) but no source tree reference to `halo-gaia` / `halo_gaia` should
> remain outside the AMD-GAIA comparison doc.

## Problem

Operator + end-user desktop client for halo-ai. Today the paths are (a) browser at `/studio/voice/` (Apple watch to Android phone to laptop), (b) `halo` CLI (terse, power-user). Nothing pulls everything into one pane: live metrics + voice chat + skill + memory inspection + remote mesh peers. halo-helm is that pane.

Name: a ship's helm is the single surface where the pilot reads instruments and applies control. Fits the role better than the original "halo-gaia" once AMD shipped their own GAIA.

## Invariants

1. **Single binary, no runtime deps.** Native Rust GUI on egui/eframe (glow backend). Ships as one ELF, runs on Linux first, macOS later. No Electron, no web-view shell. No Python, no node. Rule A applies all the way to the user's desktop.
2. **Works offline.** Must render + let the user manage local skills / memory / models even when halo-server is down.
3. **Mesh-aware.** The same binary running on any mesh peer (laptop, second mini-PC, Pi) should see the same halo-ai state by pointing at the mesh IP. No per-device state drift.
4. **Bearer secrets never land in files.** Tokens live in the system keyring (secret-service / kwallet / libsecret). No plaintext token on disk.
5. **One keyboard trip equals one user intention.** No popup dialogs for routine ops. CLI power-user can drive every pane with keyboard alone.

## Non-goals

- Not a web-browser shell (use `/studio/voice/` in Firefox if you want a browser — this is the native client lane).
- Not a model trainer — halo-helm never spawns training jobs. Training happens on Battlemage or a rented box; halo-helm consumes the resulting weights via `halo install`.
- Not a chat-logging product — conversation history lives in the server-side FTS5 sessions DB (halo-agents::sessions), NOT in halo-helm. halo-helm is a view.
- Not a web-facing product. Distribution is via `halo install helm` (aliased from legacy `halo install gaia`) or `cargo install halo-helm` — no signed installers for Windows / macOS ahead of Linux.
- Not a multi-tenant app — each user runs their own mesh peer.

## Interface

### Binary

```text
halo-helm                # opens the window
# env:
HALO_HELM_URL    # default http://127.0.0.1:8180
HALO_HELM_MODEL  # default halo-1bit-2b
HALO_HELM_TOKEN  # optional bearer
# Legacy HALO_GAIA_* env vars are honored as fallbacks.
```

### Public API (crate)

```rust
pub struct HelmApp { /* ...config, panes, transport... */ }
impl HelmApp {
    pub fn new(cfg: SessionConfig) -> Self;                       // runtime-free; test-safe
    pub fn from_cc(cc: &eframe::CreationContext, cfg: SessionConfig) -> Self;
    pub fn refresh_skills(&mut self);                             // reads ~/.halo/skills
    pub fn refresh_memory(&mut self);                             // reads ~/.halo/memories
}

#[derive(Serialize, Deserialize, Copy, Clone, Eq, PartialEq)]
pub enum Pane { Status, Chat, Skills, Memory, Models }
```

`HelmApp` implements `eframe::App`. Panes:

| Pane    | Source of truth                                  | Write path         |
| ------- | ------------------------------------------------ | ------------------ |
| Status  | `GET /_health` + `/v1/models` + probe metrics    | read-only          |
| Chat    | `POST /v1/chat/completions`, `stream: true`      | HelmClient stream  |
| Skills  | `halo_agents::SkillStore::list`                  | read-only for now  |
| Memory  | `halo_agents::MemoryStore::list(Memory \| User)` | read-only for now  |
| Models  | `GET /v1/models`                                 | read-only          |

Top bar: 5-button pane switcher. Left side panel: the same 5 panes as a radio list, plus a small config summary (default model). Bottom panel: status strip with reachability + tok/s OR last error in red.

### Persistence

eframe's `persistence` feature stores state in the platform config dir (`~/.config/halo-helm/` on Linux, via eframe's default). Today we persist the last-open pane under key `helm.current_pane`; window size is handled by eframe's own viewport save. Schema is `serde_json`-compatible so we can change the backing ron later without a migration.

## Test matrix

| Invariant                                            | Test                                             |
| ---------------------------------------------------- | ------------------------------------------------ |
| App constructs without window / tokio / FS access    | `app::tests::new_constructs_without_panic_...`   |
| Status-first default (invariant 5 keyboard-trip)     | `app::tests::default_pane_is_status`             |
| Pane-switcher coverage (tripwire on new variants)    | `app::tests::pane_all_covers_five_panes...`      |
| Stable pane labels (top-bar text contract)           | `app::tests::pane_labels_are_stable_strings`     |
| Persistence round-trip (serde stays wired)           | `app::tests::pane_round_trips_through_serde`     |

Transport modules (`client`, `conversation`, `session`, `stream`) keep their pre-rename test coverage.

## Phase: solutioning

Promoted from `analysis` on 2026-04-20 alongside the rename. The interface sketch above and the first real `HelmApp` + `Pane` types in `crates/halo-helm/src/app.rs` give magistrate + cartograph something concrete to review.

Remaining open questions for the jump to `implementation`:

- Async plumbing for network panes. Today `refresh_models` / chat streaming are marked `not wired yet`. Path of least resistance: a `tokio` runtime handle stashed on `HelmApp`, a `mpsc::UnboundedReceiver<PaneMsg>` drained each frame. Keep the runtime out of `HelmApp::new` so tests stay runtime-free.
- Keyring backend selection per-OS. libsecret on Linux; keychain on macOS.
- Does it own its own halo-server lifecycle (start/stop via systemd --user) or just observe?
- egui theming: go with `egui::Visuals::dark` default or read the system prefers-colour-scheme hint?

## Runtime system deps

eframe's `glow` backend is OpenGL over `winit`. On a stock CachyOS / Arch box that means the existing `mesa` + `libxkbcommon` + `wayland-client` + `libxcb` packages are sufficient — nothing new to add to `packages.toml`. Noting here so a future PKGBUILD author doesn't need to re-trace:

- `libxkbcommon` (winit keyboard)
- `libwayland-client` / `libwayland-egl` (Wayland path)
- `libxcb`, `libx11` (X11 fallback)
- `libgl` (mesa / swrast)
- `fontconfig` (default-fonts feature still bundles its own fonts, but fontconfig is required to render into them on Linux)

None of these are Python; all ship by default on the supported distros. We do NOT add them to PKGBUILD runtime deps until the first user on a slim Wayland-only machine hits a missing library.

## Cross-refs

- CLAUDE.md — Rule A (no Python) applies.
- `docs/wiki/SDD-Workflow.md` — phase gate + review specialist mapping.
- `docs/wiki/VPN-Only-API.md` — mesh-awareness requirement (invariant 3).
- `docs/wiki/AMD-GAIA-Integration.md` — rename rationale + AMD GAIA integration story.
- `crates/halo-helm/` — implementation.
