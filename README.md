# halo-workspace — strix-ai-rs gen 2

Rust rewrite of the halo-ai stack. Single cargo workspace, per-crate responsibilities, FFI into existing C++/HIP kernels where they're already winning on gfx1151.

## Crates

| crate | role | status |
|---|---|---|
| `halo-cli` | unified ops CLI (`halo status / logs / restart / update / doctor / version`) | scaffolded |
| `halo-core` | model loader, tokenizer, sampler | stub |
| `halo-router` | backend dispatcher (lemonade sommelier equivalent) | stub |
| `halo-server` | axum HTTP, OpenAI-compat | stub |
| `halo-agents` | 17 specialists on tokio bus | stub |
| `halo-mcp` | MCP bridge, stdio JSON-RPC | stub |
| `halo-bitnet-hip` | FFI → rocm-cpp ternary GEMV kernels | stub |

## Design split

- **Rust (this workspace):** orchestration, I/O, HTTP, agents, CLI, MCP, router — everywhere lemonade-sdk / agent-cpp had Python or C++ glue.
- **C++/HIP (external, stays in `stampby/rocm-cpp`):** ternary GEMV, Flash-Decoding attention, RoPE — linked via extern "C" from `halo-bitnet-hip`. Kernels stay in stampby/rocm-cpp; Rust is the caller.
- **Apple Silicon (external):** `strix-ai-rs/bitnet-mlx-rs` (fork of leizerowicz/bitnet-mlx.rs) provides MLX backend; loaded behind a cargo feature.

## References

- Python origin: [lemonade-sdk/lemonade](https://github.com/lemonade-sdk/lemonade) → `strix-ai-rs/lemonade-upstream` (ref fork)
- Apple reference: [leizerowicz/bitnet-mlx.rs](https://github.com/leizerowicz/bitnet-mlx.rs) → `strix-ai-rs/bitnet-mlx-rs` (ref fork)
- Kernels: [stampby/rocm-cpp](https://github.com/stampby/rocm-cpp) (public, gen 1, HIP/C++) — **unchanged, iterated separately**

Private until launch.
