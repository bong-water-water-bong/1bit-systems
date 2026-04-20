---
phase: analysis
owner: cartograph
---

# Crate: halo-gaia

## Problem

Operator + end-user desktop client for halo-ai. Today the paths are (a) browser at `/studio/voice/` (Apple watch to Android phone to laptop), (b) `halo` CLI (terse, power-user). Nothing pulls everything into one pane: live metrics + voice chat + skill + memory inspection + remote mesh peers. halo-gaia is that pane.

Named after the Greek primordial of Earth — the surface where all the other halo crates (rocm-cpp kernels underneath, halo-server + halo-voice + halo-echo + halo-agents above) finally meet the person using the box.

## Invariants

1. **Single binary, no runtime deps.** Native Rust GUI (iced / egui / ratatui-first then maybe a native-window wrapper). Ships as one ELF, runs on Linux first, macOS later. No Electron, no web-view shell. No Python, no node. Rule A applies all the way to the user's desktop.
2. **Works offline.** Must render + let the user manage local skills / memory / models even when halo-server is down.
3. **Mesh-aware.** The same binary running on any mesh peer (laptop, second mini-PC, Pi) should see the same halo-ai state by pointing at the mesh IP. No per-device state drift.
4. **Bearer secrets never land in files.** Tokens live in the system keyring (secret-service / kwallet / libsecret). No plaintext token on disk.
5. **One keyboard trip equals one user intention.** No popup dialogs for routine ops. CLI power-user can drive every pane with keyboard alone.

## Non-goals

- Not a web-browser shell (use `/studio/voice/` in Firefox if you want a browser — this is the native client lane).
- Not a model trainer — halo-gaia never spawns training jobs. Training happens on Battlemage or a rented box; halo-gaia consumes the resulting weights via `halo install`.
- Not a chat-logging product — conversation history lives in the server-side FTS5 sessions DB (halo-agents::sessions), NOT in halo-gaia. halo-gaia is a view.
- Not a web-facing product. Distribution is via `halo install gaia` or `cargo install halo-gaia` — no signed installers for Windows / macOS ahead of Linux.
- Not a multi-tenant app — each user runs their own mesh peer.

## Phase: analysis

Spec is at `analysis` until invariants + non-goals are reviewed by `magistrate` and the interface shape is approved by `cartograph`. Promote the commit message: `spec(halo-gaia): promote from analysis → solutioning — ...` once both reviewers sign off.

Open questions for solutioning:

- `iced` or `egui` or a split (ratatui on terminal + egui on desktop)?
- Which panes go in the v0 minimum? Proposal: (1) live metrics + service status, (2) voice chat (wraps `/voice/` but native-mic), (3) skill browser (halo-agents::SkillStore), (4) memory browser (MEMORY.md + USER.md), (5) model catalog + install from `halo install`.
- Keyring backend selection per-OS. libsecret on Linux; keychain on macOS.
- Does it own its own halo-server lifecycle (start/stop via systemd --user) or just observe?

## Cross-refs

- CLAUDE.md — Rule A (no Python) applies.
- `docs/wiki/SDD-Workflow.md` — phase gate + review specialist mapping.
- `docs/wiki/VPN-Only-API.md` — mesh-awareness requirement (invariant 3).
- `crates/halo-gaia/` — current scaffold.
