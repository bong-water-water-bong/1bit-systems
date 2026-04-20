# Medusa Prototype Status

Date: 2026-04-20
Status: scaffolding + scalar kernel stub landed; no runtime path yet.
Parent plan: `docs/wiki/Medusa-Integration-Plan.md` (recommendation: DEFER).

This page captures one day of prototype work sitting *below* the decision
to defer. It does not flip the recommendation; it makes the "if we
later decide to do it" estimate cheaper by pinning down head shape and
giving us a placeholder kernel with a known-correct scalar reference.

---

## 1. Medusa head shape

**Source constraint**: the sandbox denied outbound HTTP to `huggingface.co`
today, so the artifact `parrishcorcoran/MedusaBitNet-2B-4T` was **not
freshly re-downloaded** for this prototype. Numbers below come from
the upstream plan (`Medusa-Integration-Plan.md` §1) and the model card
snapshot in memory (`project_medusa_plan.md`).

| Field | Value | Source |
|---|---|---|
| Number of heads | 4 | model card |
| Per-head layers | 1 residual MLP block + 1 linear to vocab | Medusa paper §3, Medusa-1 config |
| Hidden dim | 2560 (matches Microsoft bitnet-b1.58-2B-4T backbone) | backbone config |
| Output vocab | 128256 (full LLaMA-3 tokenizer, shared `lm_head`) | backbone config |
| Per-head params | ~3.3 MB fp16 (2560 × 2560 residual + 2560 × 128256 = no, heads share lm_head so just the residual block) | Medusa paper §3.1 |
| Total head file | 13 MB fp16 | model card |
| Training data | Alpaca 52K only, 11 h on Ryzen AI MAX+ 395 CPU | model card |
| Acceptance rates | head-1 63.0%, head-2 29.0%, head-3 11.1%, head-4 4.6% | model card |
| Measured throughput | 2.08 tokens/backbone-step on CPU Medusa PyTorch | model card |

**Action item** the moment sandbox HTTP is unblocked: refetch
`config.json` + `medusa_heads.safetensors` header (header read is
<<100 MB per task constraint) and diff the declared shape against
the table above. If it diverges — specifically if head dim ≠ 2560
or heads don't share `lm_head` — the kernel below still compiles but
the integration plan's size estimates need revising.

---

## 2. Small-M ternary GEMM kernel

**File**: `/home/bcloud/repos/rocm-cpp/src/ternary_gemm_smallm.hip`

**Shape contract**:
- `packed: uint32[N, ceil(K/16)]` — halo-1bit 2-bit packing, 16 values/word,
  code ∈ {0→-1, 1→0, 2→+1, 3→unused}. Matches every other ternary kernel
  in the repo (verified against `kernels/ternary_gemv_phase5_halo.hip`
  lines 1-45).
- `scales: float32[N]` — one per output row.
- `x_bf16: bf16[M, K]` — M ∈ [1, 16].
- `y_bf16: bf16[M, N]`.
- `packed_row_stride_u32` — explicit stride, not implicit, so the kernel
  works on non-contiguous submatrices later (tree-verify may slice).

**Launch geometry**: one thread per (m, n) output element. `BLOCK_SIZE = 64`
along N, `M` along the grid y-axis. No LDS, no shuffles, no WMMA. Scalar
K-loop 16 values per iteration inside an unrolled 16-way inner loop.

**What it is**: a numerical reference. Bit-exact vs a host-side CPU
loop; useful as the "golden" to diff future tiled versions against.

**What it is not**: a performance kernel. Expected throughput for M=4,
N=K=2560 is 50-100× slower than the production GEMV. Do not benchmark
it; do not include it in the decode path. It is scaffolding.

**Compile status**: **not verified in this session.** The sandbox denied
the hipcc invocation (`hipcc --offload-arch=gfx1151 -std=c++17 -O2 -c
src/ternary_gemm_smallm.hip`). Mechanical review performed:

- Uses only headers already in use elsewhere in `rocm-cpp`
  (`hip_runtime.h`, `hip_bfloat16.h`, `cstdint`).
- `hip_bfloat16` constructor from float is the same pattern used in
  `kernels/ternary_gemv_sherry.hip`.
- `__restrict__` on every pointer param; `__attribute__((amdgpu_flat_work_group_size))`
  matches the working `phase5_halo` kernel.
- Single `extern "C"` block around both kernel + host wrapper so symbol
  names stay ABI-clean for the future FFI hookup. No C++ name mangling
  surprises.
- No `__syncthreads`, no atomics — the stub is embarrassingly parallel.

Next step before landing on `main`: run

```
hipcc --offload-arch=gfx1151 -std=c++17 -O2 -c \
    src/ternary_gemm_smallm.hip -o build/ternary_gemm_smallm.o
```

from `/home/bcloud/repos/rocm-cpp`. If this is clean, add the source
to `CMakeLists.txt` line 74-78 (alongside the other ternary kernels).
Until compile is verified, the file is **not** in the CMake target.

---

## 3. Tree attention — how KV cache extends

Today our attention kernel is `src/kv_cache_attn_fd.hip` (Flash-Decoding,
one query per head, split-KV). It's called from
`crates/1bit-router/src/backend_impl.rs` in the `forward_token` hot path:

- `backend_impl.rs:463` — `forward_token` entry (one token in, one out).
- `backend_impl.rs:595-614` — `hipMemcpyAsync` writes K/V into the
  cache at slot `pos` (single slot per call).
- `backend_impl.rs:589` — the flash-decode attention call site
  that consumes `[0..=pos]` of KV.

For Medusa's tree-verify pass, we need to:

1. **Write `tree_size` K/V entries at once** into cache slots
   `[pos .. pos + tree_size)`. In practice we write them speculatively,
   then *rewind* to `pos + accepted_len` at the end of the step, so the
   cache never grows past the verified prefix. Implementation-wise that's
   just batching the `hipMemcpyAsync` calls around `backend_impl.rs:595`
   and doing a fixup write after the verify.
2. **Attend from a set of queries** Q[tree_size, nh, hd] into
   K[pos + tree_size, nkv, hd] with a **tree mask**: candidate q_i only
   attends to its own branch (its ancestors in the spec tree, not
   sibling branches). This is a new kernel — not a flag flip on
   `kv_cache_attn_fd.hip`. Sibling file path: `src/kv_cache_attn_tree.hip`.
   Signature extension: add `const uint8_t* __restrict__ mask,
   int tree_size` args. Mask is a 2D bitmap of size `tree_size × (pos + tree_size)`.

The KV-cache layout itself (pre-layer `k` and `v` buffers at
`KvCache::k` / `KvCache::v` in `backend_impl.rs:269-272`) does not need
to change — it's already indexed by absolute position. All the surgery
is in the write path (batched slot range) and the read path (masked
gather).

---

## 4. Integration outline — PR-shaped, not yet applied

One PR per crate so each is reviewable in isolation.

### PR-1: `rocm-cpp` — kernels only, no FFI wiring

- [x] `src/ternary_gemm_smallm.hip` (this file, scalar stub, LANDED
      locally, NOT added to CMakeLists yet)
- [ ] Verify hipcc compile on gfx1151.
- [ ] Add file to `CMakeLists.txt` line 74-78 target list.
- [ ] `src/kv_cache_attn_tree.hip` (copy of `kv_cache_attn_fd.hip` with
      a mask arg and an outer M loop).
- [ ] C API entry points in `include/rocm_cpp/` for both.

### PR-2: `1bit-hip` — FFI bindings

- [ ] `src/ffi.rs` — add `extern "C"` prototypes mirroring
      `ternary_gemv_halo_f16` at line 258 and `kv_cache_attn_decode_fd`
      at line 424.
- [ ] `src/lib.rs` — add safe wrappers. Keep existing functions untouched.
- [ ] Three in-crate tests (per CLAUDE.md): mock-device happy path,
      mask-bounds check, M-range bounds check.

### PR-3: `1bit-router` — `forward_tree`, gated behind config

- [ ] `backend_impl.rs` — add `forward_tree(&mut self, accepted: &[i32],
      base_pos: i32, tree: &SpecTree, logits_out: &mut [f32]) ->
      Result<usize, BackendError>`. Does NOT replace `forward_token`.
- [ ] `backend_impl.rs:463` — leave `forward_token` alone; speculative
      config dispatches in the caller, not inside the backend.
- [ ] `src/lib.rs:599, :615` — call-site guard: if `SpecConfig::None`,
      use `forward_token`; else `forward_tree`.
- [ ] `1bit-core::sampler::Sampler` — add `argmax_batched` for the
      batched verify. Rep-penalty only applied to accepted prefix.

### PR-4: heads loader + CLI switch

- [ ] `1bit-core` — new `.medusa` file format (1 magic + 4 × safetensors
      blob). Mmap, zero-copy. Same hard rule as `.h1b`.
- [ ] `1bit-cli` — `halo gen --spec medusa` flag.

### Out of scope

- No Medusa-2 joint training. Would destroy bit-exact backbone match.
- No backbone format change.
- No changes to `ternary_gemv_halo.hip` (per CLAUDE.md "Don't touch").

---

## 5. Next concrete step — retrain or accept?

**Firm recommendation: retrain on our own workload before production use.**

Reasons:

- parrishcorcoran's heads are **Alpaca-only**. Our actual traffic is
  mixed: Open WebUI chat, code completion via the lemonade gateway,
  long-context agentic workflows (1bit-agents). Alpaca's loss curve
  (9.85 → 3.32) does not transfer cleanly to code or long-context
  acceptance. Head-1 at 63% on Alpaca does not mean 63% on our prompts.
- Training cost is cheap in absolute terms: 11 h on Strix Halo CPU per
  the model card, or one overnight pass. Budget: one night on the
  strixhalo box, or move to the Intel B770 slot (`project_battlemage_distill.md`)
  once that lands.
- Accepting the public heads short-circuits our own acceptance-rate
  measurement, which is the single number that determines whether
  Medusa hits 1.4× or 1.8× on our silicon.

**Cheaper interim step before a full retrain**: run parrishcorcoran's
heads against ~500 prompts from our actual workload (sample from
1bit-agents audit log) and measure head-1 acceptance. If it holds at
≥ 55% we can ship with upstream heads and retrain later. If it drops
below 40%, retrain is non-negotiable.

That eval harness is itself ~half a day of work — smaller than the kernel
work above, so do it first as a go/no-go gate.

---

## 6. Memory touch

Updated `~/.claude/projects/-home-bcloud/memory/project_medusa_plan.md`
with today's prototype scaffolding note. The DEFER recommendation
stands.
