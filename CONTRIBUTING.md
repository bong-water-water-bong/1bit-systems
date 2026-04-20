# Contributing to halo-ai-rs

Private monorepo for the gen-2 Rust halo-ai stack. Public clients and downstream projects are welcome regardless of who holds the workspace checkout.

## How to help

- **File an issue** in this repo with reproducible steps and `halo doctor` output.
- **Send a patch** via pull request against the `bong-water-water-bong/halo-ai-rs` mirror, or via a git bundle if you don't have collaborator access. The team-lead reviews and lands.
- **Run the benchmark** on your Strix Halo box — `halo bench` output against a clean install is gold for our perf table.
- **Test client compatibility** — if you wire Open WebUI, DSPy, LibreChat, Sorana, Aicono, TabNeuron, or anything else against halo-server and hit a snag, document the config delta in an issue.
- **Translate** the README — our current translations are machine-generated. Native speakers welcome.

## Code style

See [`CLAUDE.md`](./CLAUDE.md). Short version:

- **Rule A**: no Python at runtime (scripts are fine, services are Rust).
- **Rule B**: C++20 only for HIP kernels, FFI'd through `halo-bitnet-hip`.
- **Rule C**: hipBLAS banned at runtime.
- **Rule D**: Rust 1.86, edition 2024.

Use Conventional Commits (`feat:`, `fix:`, `docs:`, etc.). Tests must stay green on `cargo test --workspace --release` (default features).

## Acknowledgements

### Light Heart Labs

Huge gratitude to **[Light Heart Labs](https://lightheartlabs.io/)** — their open-source **DreamServer** project ([`Light-Heart-Labs/DreamServer`](https://github.com/Light-Heart-Labs/DreamServer)) has been a reference point for local-AI-first architecture and a supportive neighbor in the ecosystem. Their maintainer **[@Lightheartdevs](https://github.com/Lightheartdevs)** is a pull-access collaborator on this repo.

### Upstream projects halo-ai stands on

- **Microsoft BitNet** — [`microsoft/BitNet`](https://github.com/microsoft/BitNet), Wang & Ma et al. The reference C++ kernels and the `bitnet-b1.58-2B-4T` weights we run today. arXiv: [`2402.17764`](https://arxiv.org/abs/2402.17764) (BitNet b1.58), [`2504.18415`](https://arxiv.org/abs/2504.18415) (BitNet v2).
- **llama.cpp** — [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp). GGUF format, IQ2_S quantization scheme, and the attention-kernel idioms our FD attention grew out of.
- **ROCm / HIP** — the AMD compute runtime that our `rocm-cpp` kernels compile against.
- **axum / tokio / serde / clap / reqwest** — the Rust crate foundations of halo-workspace.
- **ratatui / crossterm** — halo-gaia's TUI layer.
- **puppeteer-core + Bun** — halo-browser's attach-mode CDP driver.

### Research we read (that shipped)

- **Sherry 1.25-bit** (anonymous, vLLM issue [`#33142`](https://github.com/vllm-project/vllm/issues/33142)) — 3:4 sparsity packing we adopted in our requantizer.
- **Split-KV Flash-Decoding** — Dao et al., the attention-kernel strategy we ported to gfx1151.
- **rotorquant / PlanarQuant** ([`scrya-com/rotorquant`](https://github.com/scrya-com/rotorquant)) — KV-cache compression; we're porting to HIP as `--kv-rotor`.
- **MedusaBitNet** ([`parrishcorcoran/MedusaBitNet-2B-4T`](https://huggingface.co/parrishcorcoran/MedusaBitNet-2B-4T)) — speculative decoding heads on BitNet, trained on Ryzen AI MAX+ 395.
- **MemPalace** ([`MemPalace/mempalace`](https://github.com/MemPalace/mempalace)) — memory-tiering ideas we're porting into our auto-memory frontmatter.
- **Sparse-BitNet** (arXiv [`2603.05168`](https://arxiv.org/abs/2603.05168)) — validates our 2-bit + N:M sparsity direction.

### Research we read (that we chose to skip)

- **Clifford-rotor KV compression** — paper's eponymous scheme lost to PlanarQuant in its own Table 5. Shipped via rotorquant's PlanarQuant path instead.
- **bitnet-cuda** ([`Wavegoodvybe2929/bitnet-rust`](https://github.com/Wavegoodvybe2929/bitnet-rust) subcrate) — 1.2k-LOC stub, no working kernels.

### Co-travelers on the AMD ternary road

- **AMD Lemonade** — their official local-LLM server; we ship a Lemonade-SDK-compatible shim so their clients run against halo-server with zero config change.
- **Chi Hoang (Tetramatrix)** — 168-repo AMD-Lemonade ecosystem. His clients (Sorana, Aicono, TabNeuron, diffron, lemonade-python-sdk) are first-class citizens of halo-ai via the `/api/v1/health` + `/api/v0/models` compat paths in `halo-lemonade`.
- **FlyGoat (RyzenAdj)** — the Linux Ryzen power-tuning CLI that will back our future `halo power` subcommand.

## How to apply this file

When you send a patch that pulls in a new external repo / paper / project as a dependency or inspiration, add it to the matching section above. Write the contributor's name and cite the URL. No anonymous ports.
