# AMD Ternary -> INT8 NPU Watch

**Question**: Has AMD (Lemonade, FastFlowLM, IRON, mlir-aie, xdna-driver) publicly landed a ternary-to-INT8 weight mapping for BitNet-1.58 on XDNA 2?

**Answer (2026-04-20)**: **No.** Zero public commits, issues, PRs, or release notes mention BitNet / ternary / 1.58 / INT8-mapping in the 7 tracked repos over the last 30 days. The POIZONEverything Discord claim ("soon") has no code footprint yet.

**Last checked**: 2026-04-20
**Re-check cadence**: weekly (Mondays). Flip to daily the first week after any landed signal.

---

## Status: Defer (unchanged)

Per `project_npu_path_analysis.md`: XDNA 2 path stays deferred. This scan adds no reason to ship. Current iGPU decode (~83 tok/s) beats our projected XDNA-2 ceiling (~45-77 tok/s, bandwidth-limited) until native ternary kernels or a bitnet.cpp XDNA backend lands.

## Signals scanned (30d window, 2026-03-21 -> 2026-04-20)

| Rank | Date | Source | Summary |
|------|------|--------|---------|
| 5 | 2026-04-20 | [amd/RyzenAI-SW#366](https://github.com/amd/RyzenAI-SW/issues/366) | **Our own filed issue** asking for Strix Halo Linux SKU support. Self-closed same day after @Geramyl (AMD Discord mod) pointed to `amd/IRON` + `Xilinx/aie-rt` bare-metal path. Confirms the SDK-route workaround but not a ternary mapping. |
| 3 | 2026-04-07 | [FastFlowLM#474](https://github.com/FastFlowLM/FastFlowLM/issues/474) | Community request (kamjin3086) for Bonsai-8B 1-bit support on NPU. **Inbound user ask, not an AMD roadmap signal.** Zero maintainer reply in 13 days. |
| 2 | 2026-04-16 | [microsoft/BitNet#545](https://github.com/microsoft/BitNet/pull/545) | Open PR: heterogeneous CUDA/ROCm multi-backend for Linux. **No XDNA mention.** Still unmerged; irrelevant to NPU path but adjacent. |
| 1 | 2026-02-03 (stale) | [microsoft/BitNet#408](https://github.com/microsoft/BitNet/issues/408) | "Intel & AMD NPU support?" 0 comments, 2.5 months cold. Confirms upstream hasn't started the work. |
| 1 | 2026-04-08 | [lemonade v10.2.0](https://github.com/lemonade-sdk/lemonade/releases/tag/v10.2.0) | Release notes: Qwen Image, Embeddable binary, OpenCode, MLX backend. **No BitNet / ternary / INT8 strings.** |
| 1 | 2026-04-15 | [FastFlowLM v0.9.39](https://github.com/FastFlowLM/FastFlowLM/releases/tag/v0.9.39) | Release notes: Gemma4, reasoning_effort, embedding batch fix. **No ternary.** |
| 1 | n/a | [Xilinx/mlir-aie 30d](https://github.com/Xilinx/mlir-aie/commits/main) | 25 commits: trace flows, placement pass, MobileNet v3. Zero ternary / 1.58 references. |
| 1 | n/a | [Xilinx/llvm-aie 30d](https://github.com/Xilinx/llvm-aie/commits/main) | ~100 commits: postpipeliner, AIE2PS legalization, upstream autobumps. Zero ternary references. |
| 1 | n/a | [amd/IRON 30d](https://github.com/amd/IRON/commits/main) | 8 commits: Phoenix CI, swiglu_decode, mlir-aie v1.3.1 bump. Zero ternary references. |
| 1 | n/a | [amd/xdna-driver 30d](https://github.com/amd/xdna-driver/commits/main) | ~60 commits: VE2 tracing, DPM levels, TDR, AIE4 bringup. Zero ternary / INT8 mapping. |
| - | - | AMD dev articles feed | **Blocked** — `amd.com/en/developer/resources/technical-articles.html` times out via WebFetch (60s). Manual re-check weekly. |
| - | - | FastFlowLM Discord | Not publicly scrapable. Noted blocked. |

Relevance rank legend: 1 = no signal, 2 = adjacent, 3 = inbound request, 4 = AMD-side roadmap hint, 5 = code landed.

## Trip-wire: what would flip Defer -> Ship

Any **one** of these lands -> act immediately:

1. `lemonade-sdk/lemonade` commit touching `backend_utils.cpp`, `router.cpp`, or `backend_versions.json` with "bitnet", "ternary", "1.58", or "xdna" in the diff.
2. `FastFlowLM/FastFlowLM` release note naming a BitNet-family model OR shipping `.xclbin` labelled bitnet/ternary.
3. `amd/IRON` or `Xilinx/mlir-aie` commit adding a `ternary` / `int2` / `i1.58` lowering pass.
4. `microsoft/BitNet` merging a backend labelled `xdna` / `ryzen-ai` / `amd_npu`.
5. AMD ROCm blog or developer article naming "BitNet" + "Ryzen AI" in the same post.

## Action plan if a signal lands

Triggered by any trip-wire above:

1. **Hour 0-1**: pull the diff. Identify the endpoint (HTTP port, gRPC, XRT ioctl) Lemonade/FFLM exposes for the ternary path. Copy the weight-layout doc into `docs/wiki/AMD-Ternary-Shim.md`.
2. **Hour 1-4**: in `halo-router`, add `Backend::Xdna` variant pointing at that endpoint (default `http://localhost:13305/xdna` if Lemonade; else whatever FFLM exposes). Gate behind `halo-router --backend xdna` flag; do not change default routing.
3. **Hour 4-24**: run `bitnet_decode --ppl` through the new backend against wikitext-103. Bit-exactness target: delta PPL < 0.02 vs our gfx1151 split-KV baseline (1.04 rep / ~12 wikitext).
4. **Day 2**: bench decode + prefill tok/s. Publish numbers to `/home/bcloud/claude output/xdna-vs-igpu-<date>.json`. Update `Why-No-NPU-Yet.md` -> `Why-NPU-Now.md` if prefill beats iGPU.
5. **Day 3-7**: if wins are real, promote XDNA backend to default for prefill only (iGPU keeps decode — bandwidth math unchanged for decode even with native int8). Keep iGPU behind `--backend rocm` for rollback.
6. **Do not** rewrite the ternary GEMV on XDNA ourselves ahead of an upstream path. Our Sherry 1.25-bit / split-KV work stays iGPU-only until AMD ships the mapping.

## Weekly re-check procedure

```fish
# Dry run, write fresh report in-place.
cd /home/bcloud/repos/halo-workspace
# 1. Re-run the gh api commits-since query for each repo (30d window).
# 2. Re-run the gh issue/PR search for "ternary OR bitnet OR 1.58".
# 3. WebFetch https://www.amd.com/en/developer/resources/technical-articles.html (timeout tolerated; retry once).
# 4. Update "Last checked" date + any new row in the signals table.
# 5. If Rank-4 or Rank-5 row: execute "Action plan" above.
```

## Sources

| Repo | URL | Last commit seen |
|------|-----|------------------|
| lemonade-sdk/lemonade | <https://github.com/lemonade-sdk/lemonade> | 2026-04-20 (214a939) |
| FastFlowLM/FastFlowLM | <https://github.com/FastFlowLM/FastFlowLM> | 2026-04-18 (52f837c) |
| amd/IRON | <https://github.com/amd/IRON> | 2026-04-17 (6b7326b) |
| Xilinx/mlir-aie | <https://github.com/Xilinx/mlir-aie> | 2026-04-16 (0d9d245) |
| Xilinx/llvm-aie | <https://github.com/Xilinx/llvm-aie> | 2026-04-16 (835da12) |
| amd/xdna-driver | <https://github.com/amd/xdna-driver> | 2026-04-17 (60f7ae4) |
| microsoft/BitNet | <https://github.com/microsoft/BitNet> | 2026-03-10 (01eb415) **stale 41d** |
| AMD dev articles | <https://www.amd.com/en/developer/resources/technical-articles.html> | blocked (WebFetch timeout) |
| FastFlowLM Discord | <https://discord.gg/fastflowlm> | not publicly scrapable |
