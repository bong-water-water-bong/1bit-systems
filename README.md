<div align="center">

# 1bit systems

**Local, ternary-weight LLM inference on a mini-PC. All Rust above the kernels, all HIP below, zero Python at runtime.**

[![CI](https://github.com/bong-water-water-bong/1bit-systems/actions/workflows/ci.yml/badge.svg)](https://github.com/bong-water-water-bong/1bit-systems/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![Rust 1.86](https://img.shields.io/badge/rust-1.86-orange.svg)](./Cargo.toml)
[![Platform: Strix Halo gfx1151](https://img.shields.io/badge/platform-Strix%20Halo%20gfx1151-red.svg)](./docs/wiki/Why-Strix-Halo.md)
[![Website: 1bit.systems](https://img.shields.io/badge/web-1bit.systems-7c3aed.svg)](https://1bit.systems)

_Runs a 2B ternary BitNet at **83 tok/s on one AMD APU**, with the whole stack — router, HTTP, agents, MCP bridge, desktop client, package manager — fitting in a single Cargo workspace._

</div>

---

## What this is

`1bit-systems` is a single cargo workspace that runs a 1.58-bit ternary Large Language Model on consumer AMD hardware and exposes it as an OpenAI-compatible HTTP service. The kernels are hand-written HIP targeting **gfx1151** (Strix Halo iGPU). Everything above them — the router, the HTTP server, the agent bus, the MCP bridge, the desktop client, the package manager — is **Rust**.

No containers. No Python at runtime. One mini-PC, one install, one binary per service, the whole thing supervised by `systemd`. The goal is a silent, closed-door, always-on household AI: Bring Your Own APU.

```
  ┌─────────────────────────────────────────────────────────────┐
  │                   any OpenAI-compatible client                │
  │   (curl · DSPy · Open WebUI · LibreChat · Claude Code MCP)    │
  └───────────────────────────────┬─────────────────────────────┘
                                  │ HTTPS (Caddy, :443)
                                  ▼
           ╔══════════════════════╧═════════════════════╗
           ║  1bit-server (axum, Rust)  ──  :8180 /v2/*  ║
           ║  1bit-router  →  1bit-core  →  1bit-hip     ║
           ╚══════════════════════╤═════════════════════╝
                                  ▼
           ┌───────────────────── ▼ ─────────────────────┐
           │  rocm-cpp (HIP, gfx1151, zero hipBLAS)      │
           │  ternary GEMV · RMSNorm · RoPE · FD attn    │
           └─────────────────────────────────────────────┘
```

## Why it's worth looking at

- **Memory-bandwidth-bound, not compute-bound.** The ternary GEMV is sitting at **~92% of measured LPDDR5 peak**. The next frontier is bytes-read reduction (Sherry 1.25-bit).
- **Native kernels, no black boxes.** The inference path goes `Rust → extern "C" → HIP`. No hipBLAS, no rocBLAS, no ONNX Runtime, no Python. Inspect one codebase top-to-bottom.
- **One box.** 128 GB unified memory on the Strix Halo platform is enough for a 2B ternary model + KV cache + Whisper STT + Kokoro TTS + SDXL image generation, all concurrent, all local.
- **OpenAI-compat first.** Point any `openai` SDK, DSPy, Open WebUI, LibreChat, Claude Code, or custom client at `http://localhost:8180/v1` and it just works.

## Benchmarks

Measured on the Strix Halo reference box (Radeon 8060S / gfx1151, 128 GB LPDDR5).

| metric            | value                   | notes                                      |
| ----------------- | ----------------------- | ------------------------------------------ |
| tok/s @ 64 prompt | **83**                  | BitNet b1.58 2B 4T, `real-backend`         |
| tok/s @ 1024      | **68**                  | same model, longer context                 |
| PPL (wikitext-103, 1024 tok) | **9.18**     | gen-1 baseline 9.1607, delta +0.02         |
| shadow-burnin byte-exact     | **96.66%**   | gen-1 ↔ gen-2 over 1,500+ rounds           |
| LPDDR5 bandwidth utilization | **~92%**     | ternary GEMV, measured via rocprof         |
| kernels in Python at runtime | **0**        | Rule A: no Python on the hot path          |

[Full benchmark methodology →](./docs/wiki/Benchmarks.md) · [Peak projection →](./docs/wiki/Peak-Performance-Projection.md)

## Quickstart

```sh
# build the whole workspace
cargo build --release --workspace

# install + supervise the services (reads packages.toml, drives systemd)
halo install core
halo status

# talk to the model over OpenAI-compat HTTP
curl -s http://127.0.0.1:8180/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "halo-1bit-2b",
    "messages": [{"role": "user", "content": "why ternary?"}]
  }'
```

To exercise the real HIP path explicitly (skipping `halo install`):

```sh
cargo run --release -p 1bit-server --features real-backend
```

Full install + first-boot walkthrough: [**Installation Guide (wiki)**](./docs/wiki/Repo-Layout.md) · [**3-minute demo script**](./DEMO.md)

## Table of contents

- [Architecture](./ARCHITECTURE.md) — data flow, crate map, feature gates, systemd layout, cutover plan
- [Contributing](./CONTRIBUTING.md) — how to help, code style, testing
- [Code of conduct](./CODE_OF_CONDUCT.md)
- [Security policy](./SECURITY.md) — responsible disclosure
- [Changelog](./CHANGELOG.md)
- [Conventions for AI agents](./CLAUDE.md) — the hard rules (A/B/C/D/E)
- [Demo script](./DEMO.md) — 3-minute cold-open walkthrough
- [Cutover runbook](./CUTOVER.md) — gen-1 → gen-2 traffic flip criteria
- [**Wiki**](./docs/wiki/Home.md) — one page per architectural decision, plus FAQ and integration guides

## Crates in this workspace

| crate             | role                                                                         | status     |
| ----------------- | ---------------------------------------------------------------------------- | ---------- |
| `1bit-cli`        | unified ops CLI: `status / logs / restart / doctor / update / install / chat / bench / ppl / say / version` | shipped |
| `1bit-core`       | `.h1b` + `.htok` parsers, GGUF loader (IQ2_S), sampler, chat template        | shipped    |
| `1bit-router`     | backend dispatcher (`HipBackend` / `MlxBackend`), format sniffing            | shipped    |
| `1bit-server`     | axum HTTP, OpenAI-compat + `/ppl` + `/metrics`                               | shipped    |
| `1bit-agents`     | 17-specialist async registry, `TypedSpecialist` + JsonSchema                 | shipped    |
| `1bit-mcp`        | MCP bridge, stdio JSON-RPC                                                   | shipped    |
| `1bit-landing`    | marketing page + live `/metrics` probe on `:8190`                            | shipped    |
| `1bit-lemonade`   | OpenAI + Lemonade-SDK compat gateway on `:8200`                              | shipped    |
| `1bit-helm`       | egui/eframe desktop client (formerly halo-gaia)                              | shipped    |
| `1bit-hip`        | FFI → `rocm-cpp` ternary GEMV + Flash-Decoding attention                     | shipped    |
| `1bit-mlx`        | FFI → `bitnet-mlx-rs` (Apple Silicon, feature-gated)                         | shipped    |

All eleven compile under default features with **zero** ROCm deps; `link-rocm` / `real-backend` / `mlx-apple` are the opt-in feature gates. CI builds all three variants.

## The four pillars

1bit systems is four repos working together. This workspace is pillar 1.

1. **Rust orchestration** — this monorepo. Everything above the kernels. Rule A safe (no Python at runtime).
2. **AMD HIP kernels** — [`bong-water-water-bong/rocm-cpp`](https://github.com/bong-water-water-bong/rocm-cpp). Ternary GEMV, RMSNorm, RoPE, split-KV Flash-Decoding attention, rotorquant PlanarQuant-3 KV compression. Folded into this repo's `rocm-cpp/` subtree.
3. **Apple Silicon MLX backend** — [`bong-water-water-bong/bitnet-mlx-rs`](https://github.com/bong-water-water-bong/bitnet-mlx-rs) (fork of `leizerowicz/bitnet-mlx.rs`). Feature-gated behind `--features mlx-apple`.
4. **Lemonade reference** — [`bong-water-water-bong/lemonade-sdk`](https://github.com/bong-water-water-bong/lemonade-sdk) (mirror of `lemonade-sdk/lemonade`, Python). **Not run at runtime.** Kept as the reference for OpenAI-compat surface area and Gaia desktop-app features being ported to Rust.

## Clients

`1bit-server` speaks plain OpenAI chat-completions on `:8180`, so off-the-shelf clients work out of the box. Point them at `http://strixhalo.local:8180/v1` — or, through Caddy, `https://strixhalo.local/v2/...` with the halo bearer token.

### DSPy (Stanford)

```python
import dspy

lm = dspy.LM(
    "openai/halo-bitnet-1.58",
    api_base="http://strixhalo.local:8180/v1",
    api_key="halo-local",
    model_type="chat",
    cache=False,
)
dspy.configure(lm=lm)

qa = dspy.ChainOfThought("question -> answer")
print(qa(question="Why is ternary BitNet memory-bound on Strix Halo?").answer)
```

`1bit-mcp` tools are directly consumable via `dspy.Tool.from_mcp_tool(...)` — no shim needed. Rule A is untouched: Python runs on the caller, `1bit-server` stays Rust.

### Open WebUI / LibreChat

Any OpenAI-compat chat UI works. In Open WebUI:

```
Settings → Connections → Add OpenAI API
  Base URL: https://strixhalo.local/v2
  API Key:  sk-halo-<your-token>
  Model:    halo-1bit-2b
```

Full RAG, multi-conversation, document chat, and MCP tools — Linux/macOS/Windows, zero server-side shim. LibreChat works identically through its `librechat.yaml` `endpoints.custom` block. Both are the blessed desktop clients until the native `1bit-helm` hits feature parity.

## Roadmap

- **Near-term** — Sherry 1.25-bit weight packing (bytes-read reduction), BitNet v2 Hadamard activation quant (W1.58A4), Medusa speculative decoding heads. See [BitNet v2 Hadamard plan](./docs/wiki/BitNet-v2-Hadamard-Plan.md), [Medusa integration plan](./docs/wiki/Medusa-Integration-Plan.md), [Sherry default decision](./docs/wiki/Sherry-Default-Decision.md).
- **Medium** — Voice loop end-to-end (whisper.cpp STT streaming + Kokoro TTS), full SD.cpp image-gen wiring, `halo-helm` as the first-class desktop client.
- **Longer** — XDNA 2 NPU lane via ORT C++ + VitisAI EP for prefill. See [Why no NPU yet](./docs/wiki/Why-No-NPU-Yet.md) and [NPU Kernel Design](./docs/wiki/NPU-Kernel-Design.md) for the current state of the Linux NPU stack.

## How to help

Contributions welcome from anyone running a Strix Halo box (or any AMD APU).

- **File an issue** with a reproducible case and `halo doctor` output.
- **Send a patch** — one logical change per commit, Conventional Commits.
- **Run the benchmark** on your hardware. `halo bench` output against a clean install is gold for the perf table.
- **Test client compatibility** — if you wire Sorana, Aicono, TabNeuron, Hermes Agent, or anything else against `1bit-server`, document the config delta in an issue.

See [CONTRIBUTING.md](./CONTRIBUTING.md) and [CLAUDE.md](./CLAUDE.md) before sending code.

## Acknowledgements

- **[Microsoft Research](https://github.com/microsoft/BitNet)** — the b1.58-2B-4T ternary weights we run and the [paper](https://arxiv.org/abs/2402.17764) that started this.
- **[Light Heart Labs](https://lightheartlabs.io/)** — their [DreamServer](https://github.com/Light-Heart-Labs/DreamServer) is a reference for local-AI-first architecture, and **[@Lightheartdevs](https://github.com/Lightheartdevs)** is a collaborator on this repo.
- **Upstream projects** — AMD ROCm, `stable-diffusion.cpp`, `whisper.cpp`, `kokoro.cpp`, axum, tokio, MLX, the Rust ecosystem. None of this ships without them.

## License

MIT. See [LICENSE](./LICENSE). Ternary model weights follow their own upstream licenses (Microsoft MIT for BitNet b1.58-2B-4T).

---

<div align="center">

**Website:** [**1bit.systems**](https://1bit.systems) · **Status:** pre-public launch, private until the XDNA 2 NPU lane lands · **Handle:** [@bong-water-water-bong](https://github.com/bong-water-water-bong)

</div>
