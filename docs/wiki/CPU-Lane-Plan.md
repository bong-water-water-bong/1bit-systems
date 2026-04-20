# CPU Lane Plan

The 7th surface of 1bit systems's APU stack: the 16 Zen5 cores on Strix Halo,
running the sampler + tokenizer + dispatcher in parallel with the iGPU's
next-token grind.

This page covers what we scaffolded today (2026-04-20), why the CPU is
a peer lane (not a fallback), and the three concrete steps that turn the
scaffold into a measured win.

## What the lane is

The CPU lane is **host-side orchestration in parallel with GPU compute**.
On a live decode:

* The iGPU runs `forward_token` — ternary GEMVs + flash-decoding attention
  — and emits logits.
* The CPU has to pick a token id from those logits (argmax or top-k /
  top-p / temperature sampler), detokenize the byte-level BPE delta,
  match stop strings, and hand the next id back to the GPU.

Gen-1 and current gen-2 run all of that on a single tokio task. Every
microsecond the sampler spends on the critical path is a microsecond
the iGPU sits idle waiting to be fed the next token id. The lane gives
that work its own rayon-backed thread pool:

* Pool size = `available_parallelism() - 2` by default (leave 2 logical
  cores for the tokio reactor + HIP stream callback thread).
* `HALO_CPU_THREADS` env override for operators who want to pin it.
* Each rayon thread named `halo-cpu-<idx>` so `rocprof` / `perf top`
  attribute it cleanly.

Surface entry point: [`onebit_router::cpu_lane::CpuLane`] in
`crates/1bit-router/src/cpu_lane.rs`.

## Why CPU is a peer lane, not a fallback

`project_apu_thesis.md` is the longer form; the short version:

* Strix Halo is a **unified-memory APU**. LPDDR5-8000 is shared by iGPU,
  NPU, and CPU at 256 GB/s peak. A CPU lane doing sampler work doesn't
  compete with the iGPU for bandwidth — the sampler's read pattern is a
  single 32k-element logit scan per token (~128 KB), fits in L2.
* Every lane that idles during decode is a lane that could have been
  running something in parallel. iGPU grinds matmuls, NPU grinds prefill
  GEMMs, CPU grinds the sampler + tokenizer. Amdahl wins when nothing
  idles.
* Peak-Performance-Projection.md tracks this as surface #7. The first six
  are HIP matmul, HIP attention, HIP sampler (today's path), XDNA
  prefill, XDNA decode, shared LPDDR5 DMA. The CPU lane replaces "HIP
  sampler" on the critical path so the iGPU can start the next token's
  matmul while the CPU is still finishing the previous token's sampler.

## What we scaffolded today

2026-04-20:

* `crates/1bit-router/src/cpu_lane.rs` — new module. `CpuLane` struct
  holds a `rayon::ThreadPool`; `parallel_sample(&self, logits, top_k,
  top_p, temp)` demonstrates the rayon parallel-reduction pattern
  (argmax at `temp=0`, chunked top-k fallback at `temp>0`).
* `Backend::Cpu` in `1bit-router`'s routing guard now returns
  `BackendError::CpuLaneStub("CPU sampler lane scaffolded, not yet on
  critical path; see docs/wiki/CPU-Lane-Plan.md")` instead of
  `unimplemented!()`. Distinct error kind from `NotYetWired` so ops
  tooling can count them separately.
* `rayon = "1"` added to `crates/1bit-router/Cargo.toml`. Rayon 1.12 is
  already in the workspace Cargo.lock transitively (via tokenizers);
  adding it directly doesn't grow the build graph.
* Three unit tests in `cpu_lane::tests` — constructor doesn't panic,
  argmax at `temp=0` is deterministic on a 32k-element vector,
  `HALO_CPU_THREADS=4` is respected. Env-var test takes a module-local
  mutex so it can't race with the `HALO_BACKEND` test in
  `backend_config_tests`.

What we did **not** do today, on purpose:

* Wire `CpuLane` into 1bit-server's SSE handler. The hotspot analysis
  comes first — see step (a) below.
* Replace the existing `onebit_core::sampler::Sampler` (full top-k + top-p
  + temperature + repetition penalty). `parallel_sample` today is
  scaffolding — it proves the rayon pattern, not a drop-in replacement.
* Add any C++ deps. Rule A + the 2026-04-20 "bare-metal first" lock-in
  keeps orchestration in Rust; kernels stay in rocm-cpp.

## What's next (three concrete steps)

### (a) Wire `CpuLane::parallel_sample` into 1bit-server's SSE handler

`1bit-server` runs the sampler inline on the main axum task today. The
wire-up:

1. Construct a process-global `CpuLane` in `1bit-server`'s `main` via
   `onebit_router::cpu_lane::global_lane()`.
2. In the `/v1/chat/completions` streaming handler, replace
   `sampler.sample(logits, history)` with a dispatch that routes to
   `lane.parallel_sample(logits, top_k, top_p, temp)` when the CPU lane
   is selected. Keep the single-threaded path behind a
   `HALO_CPU_LANE=off` escape hatch.
3. Ensure the SSE delta emission stays on the tokio task — rayon runs
   the scan, tokio owns the socket.

### (b) Microbenchmark single-threaded vs rayon-parallel sampler

Not wired = no measurement. Target: `criterion` bench in
`crates/1bit-router/benches/cpu_lane.rs`:

* Input: a synthetic 32k-element logit vector (BitNet's vocab size,
  128256 rounded down for easier chunking).
* Harness: compare `onebit_core::sampler::Sampler::sample` (single-threaded)
  against `CpuLane::parallel_sample` at pool sizes 1, 4, 8, 14.
* Pass criterion: rayon-parallel faster at ≥ 4 threads by enough that
  the iGPU overlap (~200 µs per token measured on a clean burn) wouldn't
  be dwarfed by rayon's per-task overhead.

If rayon loses: step (c) becomes rank #1 — a flat `for` loop on the
tokio task beats rayon for a single 128 KB scan, and we'd need BLAS-level
machinery to make the CPU lane pull its weight.

### (c) Decide whether ZenDNN FFI is worth the dep weight

AMD-AI-ML-Tools-Scan.md flagged ZenDNN 5.2 as having an official
llama.cpp backend. It's AMD's Zen-tuned math library — MLP / GEMM /
reduction kernels AVX-512-VNNI-tuned for Zen4+. Two questions to answer
before we pull it in:

1. Is a parallel sampler + tokenizer even the bottleneck, or is the
   iGPU forward pass eating all the time? (Step b's bench answers this.)
2. Does ZenDNN offer anything for our sampler specifically? The library
   is matmul-first; our sampler is a reduction. If the win is only on
   matmuls, the CPU lane is still Rust-native — we only FFI when the
   kernel class matches.

Decision output: either add ZenDNN as an optional feature-gated dep
behind `--features zendnn` (same shape as `--features real-xdna`), or
skip and stay pure Rust.

## Cross-references

* `/home/bcloud/.claude/projects/-home-bcloud/memory/project_apu_thesis.md`
  — the "why CPU is a peer lane" thesis.
* `docs/wiki/Peak-Performance-Projection.md` — the 7/7 lane aspirational
  target; the CPU lane closes surface #7.
* `docs/wiki/AMD-AI-ML-Tools-Scan.md` — ZenDNN 5.2 notes for step (c).
* `crates/1bit-router/src/cpu_lane.rs` — the module this page describes.
