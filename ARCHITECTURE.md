# 1bit systems gen 2 architecture

This document is the long-form companion to the [README](./README.md). It describes how the four pillars fit together at runtime, what each crate in this workspace is responsible for, how features gate the backends, how the service is laid out on the box, and what the cutover from gen 1 to gen 2 looks like.

## The four pillars

| # | pillar | repo | language | role |
|---|---|---|---|---|
| 1 | Rust orchestration | this workspace (`halo-workspace`) | Rust | CLI, HTTP, router, MCP, agents |
| 2 | AMD HIP kernels | `bong-water-water-bong/rocm-cpp` (private) / `bong-water-water-bong/rocm-cpp` (public gen 1) | C++ / HIP | ternary GEMV, RMSNorm, RoPE, split-KV FD attention |
| 3 | Apple Silicon MLX | `bong-water-water-bong/bitnet-mlx-rs` (fork of `leizerowicz/bitnet-mlx.rs`) | Rust + MLX | Apple-Silicon backend, feature-gated |
| 4 | Lemonade reference | `bong-water-water-bong/lemonade-sdk` (mirror of `lemonade-sdk/lemonade`) | Python | **not runtime** — reference for OpenAI-compat + Gaia UI we're porting |

Rule A: pillars 1 and 3 are the only things that run at inference time. Pillar 4 is reference material. Pillar 2 is a compiled shared object linked into pillar 1.

## Data flow

```
                   HTTP client (OpenAI-compat)
                              |
                              v
                   +----------+----------+
                   |    Caddy (TLS)      |
                   |  /v1/*  ->  :8080   |   <- gen 1 (bitnet_decode, C++)
                   |  /v2/*  ->  :8180   |   <- gen 2 (1bit-server, Rust)
                   +----------+----------+
                              |
                              v  (gen 2 path)
                   +----------+----------+
                   |   1bit-server       |   axum, tokio, OpenAI-compat
                   +----------+----------+
                              |
                              v
                   +----------+----------+
                   |   1bit-router       |   dispatches per Backend trait
                   +---+--------------+--+
                       |              |
             feature:  |              |  feature:
             real-     |              |  mlx-apple
             backend   |              |
                       v              v
          +------------+---+      +---+--------------+
          |  HipBackend    |      |   MlxBackend     |
          |  (halo-bitnet- |      |  (halo-bitnet-   |
          |     hip)       |      |     mlx)         |
          +-------+--------+      +--------+---------+
                  |                        |
                  v                        v
          +-------+--------+      +--------+---------+
          |   rocm-cpp     |      |  bitnet-mlx-rs   |
          |  (HIP kernels, |      | (Apple MLX via   |
          |   extern "C")  |      |  Rust crate)     |
          +----------------+      +------------------+

1bit-core: tokenizer, sampler, chat template, loader (shared by both backends)
1bit-agents + 1bit-mcp: sit alongside 1bit-server, not in the hot path
1bit-cli: operates on systemd units, reads packages.toml
```

## Crate responsibilities

- **`1bit-cli`** — `halo` binary. `status / logs / restart / doctor / update / install / version`. Reads `packages.toml`, drives systemd units, queries the local HTTP server for liveness.
- **`1bit-core`** — GGUF loader, BPE tokenizer, sampler, chat-template renderer. Backend-agnostic. Shared by both `HipBackend` and `MlxBackend`.
- **`1bit-router`** — defines the `Backend` trait and picks an implementation at build time via cargo features. This is the lemonade "sommelier" replaced in Rust.
- **`1bit-server`** — axum app. OpenAI-compat `/v1/chat/completions`, `/v1/completions`, `/v1/models`, plus `/v2/*` mirrors during burn-in. Streams SSE.
- **`1bit-agents`** — tokio-based agent bus for the specialist pool. Sits next to the server, not in the inference hot path.
- **`1bit-mcp`** — stdio JSON-RPC MCP bridge. Lets external Claude/MCP clients drive the local model.
- **`1bit-hip`** — `extern "C"` bindings to `rocm-cpp`. Builds against a system `librocm_cpp.so`. This is where pillar 1 meets pillar 2.
- **`1bit-mlx`** — Rust wrapper around the `bitnet-mlx-rs` fork. Only compiled on `aarch64-apple-darwin` targets with `--features mlx-apple`.

## Feature-gate matrix

| feature | target | backend | intended use |
|---|---|---|---|
| default (no extra features) | any | `NullBackend` (echoes request) | unit tests, CI on machines without a GPU |
| `real-backend` | x86_64-linux, gfx1151 | `HipBackend` → `rocm-cpp` | Strix Halo production |
| `mlx-apple` | aarch64-darwin (M-series) | `MlxBackend` → `bitnet-mlx-rs` | dev on Apple Silicon laptops |
| `real-backend` + `mlx-apple` | — | rejected at compile time | only one backend at a time |

The router picks the backend at build time, not runtime, so a given `1bit-server` binary only carries one inference path. CI builds all three variants.

## Systemd layout

All units live in `/etc/systemd/system/` and are installed by `1bit install <component>`.

| unit | binary | port | notes |
|---|---|---|---|
| `halo-bitnet-decode.service` | gen-1 `bitnet_decode` | 8080 | existing, unchanged |
| `1bit-server.service` | gen-2 `1bit-server` | 8180 | `--features real-backend` |
| `1bit-agents.service` | `1bit-agents` | internal | tokio bus, no HTTP |
| `1bit-mcp.service` | `1bit-mcp` | stdio | socket-activated |
| `caddy.service` | Caddy | 443 | TLS front door, splits `/v1/*` and `/v2/*` |

`1bit status` polls all of these and reports a single rollup. `1bit logs <unit>` wraps `journalctl -u`.

## Caddy `/v1/*` vs `/v2/*` burn-in plan

During burn-in, `1bit-server` exposes a full `/v2/*` surface that mirrors the `/v1/*` surface of `bitnet_decode`. Caddy sends all production traffic to `/v1/*` and mirrors a copy into `/v2/*` (shadow traffic). The gen-2 response is compared offline; the client only ever sees the gen-1 response.

Caddyfile sketch:

```
halo.local {
  handle /v1/* { reverse_proxy localhost:8080 }
  handle /v2/* { reverse_proxy localhost:8180 }
}
```

Shadow mirroring is handled by a small tap service that replays `/v1/*` requests into `/v2/*` and diffs logits / text out of band.

## Chat template (gen-1 parity)

gen-2 matches gen-1 exactly on the wire. One turn is rendered as:

```
User: <msg><|eot_id|>Assistant:
```

Multi-turn concatenates rendered turns with no extra separators. `1bit-core::chat_template` is the single source of truth; both backends call into it. There is a parity test in `1bit-core` that renders a known conversation and byte-compares against a fixture captured from gen-1.

## Cutover criteria

gen 2 takes over `/v1/*` (and gen 1 is demoted to `/legacy/*`) when **both** of the following are true:

1. **PPL parity on wikitext.** `1bit-server --features real-backend` must hit PPL within +/- 0.05 of the current gen-1 number (9.16) on the full wikitext eval.
2. **Shadow-traffic burn-in green for 72 hours.** No diffs above the noise floor between `/v1/*` (gen 1) and `/v2/*` (gen 2) for a continuous 72 h window under real traffic, with no service restarts required and no memory growth on gen-2.

Until both conditions hold, gen 1 stays the production path and gen 2 runs silently on `:8180`.
