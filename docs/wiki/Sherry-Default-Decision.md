# Sherry 1.25-bit default-on/off decision — 2026-04-20

**Decision: default-OFF.** Keep `halo-1bit-2b.h1b` (v2 halo, 2.0 bpw) as the
shipped model. Leave the Sherry kernel + v3 loader + `format_version==3`
dispatch path in-tree, unused by default.

## Gates vs observed

| gate | target | observed | verdict |
|---|---|---|---|
| Decode tok/s @ 2B shape | ≥ **1.5×** baseline | **1.03×–1.10×** | FAIL |
| Δ PPL vs v2 baseline on wikitext-103 | ≤ **+0.05** | **+8.66 × 10⁸** | FAIL |

Detailed bench in `~/claude output/sherry-bench-2026-04-20.md`.

## Why the tok/s gate fails

Sherry delivers 37.5% fewer **ternary** bytes per GEMV, but on BitNet-2B
the ternary weights are only ~40% of the total bytes read per decode
step (fp16 KV cache, fp16 norms, fp16 LM head, activations). And at
realistic context lengths the KV read dominates — it's the whole point
of the Flash-Decoding kernel we shipped in the same commit. Net effect:
Sherry's wins decay from ~10% at L=64 to ~3% at L=1024, nowhere near
the 1.5× we set as the promotion threshold.

The 1.5× gate was set with the microbench number in mind
(synthetic GEMV speedup of 1.44×–1.66× that Sherry shows L2-hot). That
number does not survive contact with the full decode loop.

## Why the PPL gate fails

The Sherry paper's "zero accuracy loss" claim assumes from-scratch
training with STE-aware 3:4 constraints. BitNet-b1.58-2B was trained
without that constraint. Three post-hoc requantizers have been tried:

1. **Stub** (heuristic: pick smallest-|ternary| position as zero) —
   2026-04-18. Repetitive garbage output.
2. **Proper** (load bf16 HF shadow weights, pick smallest-|bf16|) —
   2026-04-19. Repetitive garbage output.
3. **Fine-tune** (short SFT with 3:4 constraint active) — 2026-04-19.
   PPL 8.66 × 10⁸, still garbage.

The structural problem is 25% of all weights being forced to zero per
4-weight group destroys too much of the trained signal regardless of
selection heuristic. A real fix needs either a full retrain or a
multi-day fine-tune with 3:4-aware STE on a large calibration set;
neither is in scope for this spike.

## What stays in-tree

- `rocm-cpp/kernels/ternary_gemv_sherry.hip` — 1.25-bpw HIP kernel
  (bit-exact vs halo on random-ternary-with-3:4 input)
- `rocm-cpp/src/h1b_loader.cpp` — `.h1b` v3 detection + `read_ternary_sherry`
- `rocm-cpp/include/rocm_cpp/bitnet_model.h` — `format_version` field
- `rocm-cpp/tools/bitnet_decode.cpp:267-271` — version-dispatched GEMV
  selector (v1/v2 → halo, v3 → sherry, v4 → tq1-halo)
- `rocm-cpp/tools/bench_sherry.cpp` — cross-bench harness
- `halo-ai/models/halo-1bit-2b.sherry.h1b`,
  `halo-1bit-2b.sherry-proper.h1b`,
  `halo-1bit-2b.sherry-ft.h1b` — three candidate model files, 1.65 GB each

Nothing to rip out. The kernel is paid-for infrastructure.

## Re-open criteria

Re-open this decision when any of the following lands:

1. A 3:4-sparsity-aware pretrained 2B ternary model ships upstream
   (watch Sparse-BitNet, arXiv 2603.05168, AAzdi/Sparse-BitNet repo —
   they train 1.58-bit + 2:4 jointly; a 3:4 variant is plausible).
2. A budgeted fine-tune job (on the Battlemage box once it arrives, per
   `project_battlemage_distill.md`) brings the Sherry-ft PPL below ~12
   on wikitext-103.
3. The ternary GEMV becomes decode-bottleneck-dominant again (it isn't,
   after the Flash-Decoding KV kernel landed). Measure with `rocprofv3`
   kernel-time breakdown; re-open if ternary is >60% of decode time.

## PR-shaped diff for flipping to default-on (NOT to apply)

When the above is met, default-on is a **filename swap on the service
side** — no code change in `rocm-cpp`. The dispatch in
`rocm-cpp/tools/bitnet_decode.cpp:267-271` already routes by
`m.format_version`, so any v3 `.h1b` loaded from the canonical path
gets the Sherry kernel automatically.

```diff
 diff --git a/strixhalo/systemd/halo-bitnet.service b/strixhalo/systemd/halo-bitnet.service
 --- a/strixhalo/systemd/halo-bitnet.service
 +++ b/strixhalo/systemd/halo-bitnet.service
 @@ -7,1 +7,1 @@
-ExecStart=/usr/local/bin/bitnet_decode __HOME__/halo-ai/models/halo-1bit-2b.h1b --server 8080
+ExecStart=/usr/local/bin/bitnet_decode __HOME__/halo-ai/models/halo-1bit-2b.sherry-ft.h1b --server 8080

 diff --git a/strixhalo/systemd/strix-server.service b/strixhalo/systemd/strix-server.service
 --- a/strixhalo/systemd/strix-server.service
 +++ b/strixhalo/systemd/strix-server.service
 @@ -8,1 +8,1 @@
-ExecStart=__HOME__/.local/bin/halo-server-real --bind 127.0.0.1:8180 --model __HOME__/halo-ai/models/halo-1bit-2b.h1b
+ExecStart=__HOME__/.local/bin/halo-server-real --bind 127.0.0.1:8180 --model __HOME__/halo-ai/models/halo-1bit-2b.sherry-ft.h1b

 diff --git a/strixhalo/bin/halo-anvil.sh b/strixhalo/bin/halo-anvil.sh
 --- a/strixhalo/bin/halo-anvil.sh
 +++ b/strixhalo/bin/halo-anvil.sh
@@ -60,1 +60,1 @@
-MODEL="$HOME/halo-ai/models/halo-1bit-2b.h1b"
+MODEL="$HOME/halo-ai/models/halo-1bit-2b.sherry-ft.h1b"

 diff --git a/crates/halo-server/src/routes.rs b/crates/halo-server/src/routes.rs
 --- a/crates/halo-server/src/routes.rs
 +++ b/crates/halo-server/src/routes.rs
@@ -909,1 +909,1 @@
-            .unwrap_or_else(|_| PathBuf::from(format!("{home}/halo-ai/models/halo-1bit-2b.h1b")));
+            .unwrap_or_else(|_| PathBuf::from(format!("{home}/halo-ai/models/halo-1bit-2b.sherry-ft.h1b")));
```

Alternative (cleaner): publish the accepted Sherry file as the canonical
`halo-1bit-2b.h1b` release artifact and leave all paths alone. The
loader auto-detects v3 by magic + version field. This is the
recommended form of promotion when the gates eventually pass.

## Linked

- Bench report: `~/claude output/sherry-bench-2026-04-20.md`
- Memory: `~/.claude/projects/-home-bcloud/memory/project_sherry_spike.md`
- Rocprof plan: `~/.claude/projects/-home-bcloud/memory/project_bitnet_rocprof_plan.md`
- Commit that landed Sherry: `rocm-cpp 8db7057` (2026-04-19)
