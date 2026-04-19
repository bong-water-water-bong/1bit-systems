# halo-workspace — strix-ai-rs gen 2

Rust rewrite of the halo-ai stack. One cargo workspace for everything above the kernels: CLI, HTTP server, router, MCP bridge, agent bus. No Python at runtime. The C++/HIP kernels that already win on gfx1151 stay where they are and are linked in via FFI.

Today the workspace compiles eight crates, all 54 tests green. `halo-server --features real-backend` drives real inference on gfx1151 through `halo-router → halo-bitnet-hip → rocm-cpp`. Gen-1 (`bitnet_decode`) on `:8080` and gen-2 on `:8180` run side-by-side behind Caddy, routed by `/v1/*` and `/v2/*`, so the rewrite burns in under shadow traffic without taking the existing service down.

See [ARCHITECTURE.md](./ARCHITECTURE.md) for the full data-flow, feature-gate matrix, systemd layout, and cutover criteria.

## Crates

| crate | role | status |
|---|---|---|
| `halo-cli` | unified ops CLI (`status / logs / restart / doctor / update / install / version`) | shipped |
| `halo-core` | model loader, tokenizer, sampler, chat template | shipped |
| `halo-router` | backend dispatcher (`HipBackend` / `MlxBackend`) | shipped |
| `halo-server` | axum HTTP, OpenAI-compat `/v1/*` and `/v2/*` | shipped |
| `halo-agents` | specialist bus on tokio | scaffolded |
| `halo-mcp` | MCP bridge, stdio JSON-RPC | scaffolded |
| `halo-bitnet-hip` | FFI → `rocm-cpp` ternary GEMV + attention kernels | shipped |
| `halo-bitnet-mlx` | FFI → `bitnet-mlx-rs` (Apple Silicon, feature-gated) | shipped |

All eight currently compile; "scaffolded" means the crate's public API is in place but the feature set is still small.

## Quickstart

```sh
# build the whole workspace
cargo build --release --workspace

# run the full stack (CLI installs, starts, and supervises services)
halo install core
halo status
```

`packages.toml` is the manifest; `halo install <component>` is the package manager. `halo` itself ships with `status / logs / restart / doctor / update / install / version`.

To exercise the real HIP path:

```sh
cargo run --release -p halo-server --features real-backend
```

## Benchmarks

Measured on the Strix Halo mini-PC (gfx1151, LPDDR5):

- 83 tok/s @ 64-token prompt
- 68 tok/s @ 1024-token prompt
- PPL 9.16 on wikitext

Ternary GEMV is sitting at 92% of LPDDR5 peak bandwidth. Next frontier is bytes-read reduction (Sherry 1.25-bit, see workspace memory).

## The four pillars

halo-ai gen 2 is four repos working together. This workspace is pillar 1.

1. **Rust orchestration** — this monorepo. Everything above the kernels, all Rust, Rule A safe (no Python at runtime).
2. **AMD HIP kernels** — [bong-water-water-bong/rocm-cpp](https://github.com/bong-water-water-bong/rocm-cpp) (private mirror) and [stampby/rocm-cpp](https://github.com/stampby/rocm-cpp) (public gen 1). Ternary GEMV, RMSNorm, RoPE, split-KV Flash-Decoding attention. Linked through `halo-bitnet-hip`.
3. **Apple Silicon MLX backend** — [bong-water-water-bong/bitnet-mlx-rs](https://github.com/bong-water-water-bong/bitnet-mlx-rs) (fork of `leizerowicz/bitnet-mlx.rs`). Feature-gated behind `--features mlx-apple`, keeps the workspace cross-platform on a single codebase.
4. **Lemonade reference** — [bong-water-water-bong/lemonade-sdk](https://github.com/bong-water-water-bong/lemonade-sdk) (mirror of `lemonade-sdk/lemonade`, Python). **Not run at runtime.** Kept as the reference for OpenAI-compat surface area and Gaia desktop-app features we're porting to Rust.

Private until launch.
