# CLAUDE.md — conventions for halo-ai-rs

Conventions for any Claude (or human) agent working inside this repo.
Keep terse; when in doubt, follow `ARCHITECTURE.md`.

## Hard rules

- **Rule A — no Python at runtime.** Python is fine for scripts that run
  once on a dev box (requantizers, analysis notebooks), never inside a
  systemd unit or a path that serves HTTP. If a component needs orchestration,
  write it in Rust.
- **Rule B — C++20 only for kernels.** All HIP kernels stay in
  `bong-water-water-bong/rocm-cpp`. Do NOT reimplement kernels in Rust.
  FFI through `halo-bitnet-hip`. (Old `stampby/rocm-cpp` is archived,
  `stampby` handle retired.)
- **Rule C — hipBLAS is banned** in the runtime path. Native Tensile kernels
  only. If you find yourself reaching for hipBLAS, step back and port the
  kernel to `rocm-cpp` instead.
- **Rule D — Rust 1.86, edition 2024.** `rust-version` is pinned in the
  workspace `Cargo.toml`. Don't bump without a reason.

## Layout

```
crates/
  halo-cli           unified operator CLI (halo status/logs/doctor/...)
  halo-core          model + tokenizer parsers (pure, no I/O beyond mmap)
  halo-router        backend dispatcher + forward pass driver
  halo-server        axum HTTP, OpenAI-compat, /ppl, /metrics
  halo-agents        17-specialist async registry (one file, one source of truth)
  halo-mcp           tokio stdio JSON-RPC bridge → halo-agents registry
  halo-landing       marketing page on :8190, live /metrics probe
  halo-lemonade      OpenAI-compat model gateway (/v1/models on :8200)
  halo-gaia          desktop client scaffold (no UI yet)
  halo-bitnet-hip    FFI into rocm-cpp
  halo-bitnet-mlx    Apple Silicon backend (feature-gated)
strixhalo/           dotfiles (systemd, caddy, bin, fish) — see strixhalo/README.md
packages.toml        pkg manifest consumed by `halo install`
install.sh           fresh-box bootstrap
```

## Testing

- **Every new crate gets ≥3 in-crate tests** before it can be merged.
- **`cargo test --workspace --release` must stay green on main.** Current
  baseline: ~90 tests across 11 crates.
- **Integration tests that need real GPU / model weights** go behind
  `#[ignore]` + `--features real-backend` and are run by hand on the
  strixhalo box. CI on GitHub Actions runs only the default feature set.
- **Parity vs gen-1** is the ultimate cutover gate. Two harnesses:
  - `benchmarks/ppl-gen2.sh` — PPL on wikitext-103. gen-1 baseline 9.1607.
    gen-2 currently 9.1805 (delta +0.02, within ±0.05 tolerance). PASS.
  - `benchmarks/shadow-burnin.sh` — continuous /v1 vs /v2 argmax compare.
    Current rate: ~90% byte-exact after special-token fix. Logs to
    `~/claude output/shadow-burnin.jsonl`, state in `~/.local/share/halo-ai/`.

## Commit conventions

- Conventional Commits prefixes: `feat / fix / perf / docs / refactor / build / ci / chore / test`.
- **One logical change per commit.** "fix: X" PLUS "feat: Y" should be
  two commits, not one.
- **Messages have a "why" line.** Not "add tokenizer" — "add tokenizer
  special-token handling; gen-1 expects 128009 on EOT boundary and argmax
  diverges without it."
- **Always push to `bong` remote** (`git@github-bong:...`).
  `bong-water-water-bong` is the canonical handle. `stampby` is retired.

## What NOT to do

- **Don't commit tokens, session cookies, or bearer secrets.** Caddyfile's
  bearer lives in `/etc/caddy/Caddyfile` (root-only). The `strixhalo/caddy/Caddyfile`
  tracked copy has `sk-halo-REPLACE_ME` placeholders; never replace in git.
- **Don't add Python deps.** If you think you need Python, talk to the user.
- **Don't touch `ternary_gemv_halo.hip`** in rocm-cpp without a rocprof
  trace showing the improvement. Current kernel is at 92% of LPDDR5 peak.
- **Don't skip the KV-cache reset** at the start of each generation. The
  2026-04-19 SEGV was a `pos` accumulator bug that only showed up under
  sustained load (~200 completions in).
- **Don't add new warnings.** CI doesn't gate on `-D warnings` yet, but that
  gate is coming.

## Deploy flow

After changes that need a live restart:

```bash
cargo build --release --workspace                      # or -p <crate>
# halo-server: binary is held open by the running unit → stop first
systemctl --user stop strix-server
cp target/release/halo-server ~/.local/bin/halo-server-real
systemctl --user start strix-server
# other binaries install via cargo install --path crates/<name>
```

Use `halo install <component>` from `packages.toml` when in doubt — it
does the stop/copy/start cycle correctly.

## Agent / subagent ground rules

When delegating to a background agent:

- Give it **exact file paths + line numbers** where possible.
- Tell it what to **not** touch (other crates, config files, READMEs).
- Ask for a **≤200-word report**, not a full transcript.
- **Trust but verify.** The agent's summary describes intent, not outcome.
  Always re-run tests + inspect the actual diff before claiming success.

## Memory

Per-project persistent memory lives outside this repo at
`~/.claude/projects/-home-bcloud/memory/`. Don't duplicate that here.
Load-bearing memory entries for this repo: `project_strix_ai_rs.md`,
`project_bitnet_live_bench.md`, `project_ppl_harness.md`,
`project_1bit_paper_techniques.md`.
