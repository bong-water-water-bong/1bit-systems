# Rotorquant default: OFF on 2B, opt-in via `--kv-quant rotor-3`

PlanarQuant-3 (2D-Givens rotation + 3-bit Lloyd-Max) landed as an optional
KV-cache compression path in `rocm-cpp`:

- `src/kv_cache_attn_fd_rotor.hip` — split-KV Flash-Decoding with inline
  dequant + inverse Givens.
- `kernels/rotorquant_pack.hip` — fp16 → 3-bit packed requantizer.

5.33× KV compression (fp16 16 B/elem → packed 3 B/elem, exactly). Numerically
clean: `mean_abs_diff ~ 0.02–0.05`, `max_abs_diff` tracks `ref_max_abs`.
**Perf is the problem.**

## 1. Current bench numbers (BitNet-2B-4T shape: NH=20, NKV=5, HD=128)

Source: `/home/bcloud/claude output/rotor_bench-20260420T133827Z.txt`
(post wave-shfl V-loop fix). `us_pq3_total = pq3_attn + pq3_requant` (the
requantize cost must be amortised because a freshly-written KV row
must be packed before any subsequent read).

| seq_len | fp16 attn µs | pq3 attn µs | pq3 requant µs | pq3 total µs | fp16 tok/s | pq3 tok/s | ratio |
|--------:|-------------:|------------:|---------------:|-------------:|-----------:|----------:|------:|
|     512 |         68.1 |        81.0 |            5.1 |         86.1 |     14 675 |    11 617 | 0.79× |
|    1024 |         88.8 |        92.3 |            5.0 |         97.4 |     11 263 |    10 272 | 0.91× |
|    2048 |         95.1 |       106.4 |            5.0 |        111.4 |     10 516 |     8 977 | 0.85× |
|    4096 |        184.7 |       208.0 |            5.0 |        213.0 |      5 413 |     4 696 | 0.87× |

**We do not yet have 8k or 16k numbers.** The bench harness caps at 4096
today. An 8k / 16k sweep is explicitly a prerequisite for flipping the
default (see §4).

Pre-fix baseline for reference (`rotor_bench-20260420T121449Z.txt`, before
the V-loop wave-shfl): 4096 was 2271 tok/s, i.e. the shfl fix doubled it.
PQ3 is now consistently ~0.8–0.9× fp16 at the sizes we can measure. It
does not overtake. The slope of fp16 attn µs/tok flattens with seq_len
(the grid scales with `num_kv_tiles`), so the gap does not obviously
close at 8k+ either.

## 2. Why it is slower despite 5.33× fewer bytes read

At HD=128, NKV=5, the pass-1 block does `TILE=128` positions per tile.
Per position the fp16 path streams `5 × 128 × 2 = 1280 B` of K plus 1280 B
of V from the KV cache. PQ3 streams `5 × 128 × 3/8 = 240 B` of K plus 240 B
of V. Net savings: roughly **2080 B/pos read-bandwidth**.

Against that, PQ3 pays per-position, per-pair (`pairs = head_dim / 2 = 64`):

1. **3-byte triple load + unpack** — byte-wise loads (not coalesced as
   cleanly as the fp16 `ld_global_u16x2`), shift assemble into `u32`,
   then two 3-bit extract-and-mask operations.
2. **LUT dequant** — one LDS read per scalar into a scaled 8-entry LUT.
   Bank-conflict-free (8 entries replicated), but an extra LDS round-trip
   per scalar vs the fp16 path which reads K/V directly.
3. **Inverse Givens rotation** — 2 fmas + 2 muls per pair, per position,
   on both K and V. Plus an LDS read of `s_pair_cs[p][{0,1}]` per pair.
4. **V-side wave shfl** — the recent fix; saves vgpr on the V accumulate
   but still leaves the dequant+unrotate on the critical path.

At gfx1151's LPDDR5 bandwidth (~256 GB/s, ~77–80 GB/s realized to iGPU),
the fp16 attn decode kernel is **not memory-bound** at these shapes — we
sit well under the roofline because the per-position compute is tiny and
the launch/grid/online-softmax overhead dominates below ~2k. So trading
bandwidth for compute *loses* at HD=128/NKV=5. The 5.33× compression
doesn't buy us anything we need; the inverse-rotation is pure overhead.

This would flip at large NKV × HD × seq_len where the KV read genuinely
saturates LPDDR5 — i.e. 70B-class or 32k-context regimes.

## 3. Recommendation

**Default `rotor-3` OFF on BitNet-2B-4T and any ≤4B model.** Keep the
path fully built and wired, expose it behind `--kv-quant rotor-3` on the
CLI (and a matching field in the router config). Flip the default ON
**only** when we ship either:

- A 70B-class model where fp16 KV exceeds working-set headroom in
  128 GB LPDDR5 under realistic multi-session load, or
- A 32k (or longer) context variant of *any* model where fp16 KV
  crosses the point at which the attn kernel becomes bandwidth-bound
  (i.e. `us_attn` scales linearly with `seq_len` and hits roofline).

This matches how the stack is used today (2B short-context on a single
box) while keeping the compression path alive for the two futures that
actually need it. Numeric parity is already good enough for opt-in; we
are not shipping a correctness risk, only a perf trade.

## 4. What to measure before flipping the default

Hard gates, all on gfx1151, `bench.sh`-style harness, 3×20-iter rounds:

1. **8k and 16k sweeps** — extend `rotor_bench` to `seq_len ∈ {8192, 16384}`.
   PQ3 needs to beat fp16 by ≥1.1× at *some* seq_len to justify a default
   flip for long-context.
2. **70B shape microbench** — re-run the microbench at (roughly) NH=64,
   NKV=8, HD=128 to mimic a 70B LLaMA-family layer. Confirm the
   bandwidth crossover appears in that shape.
3. **Service-level bench** — end-to-end tok/s via `bitnet_decode` (not
   just the kernel microbench) at 4k / 8k / 16k prompt + 1024 gen. The
   microbench excludes the K/V write-path requantize called on every
   new token; we need to confirm the amortisation assumption holds.
4. **PPL on wikitext-103 at 1024 ctx** — fp16 baseline 9.1607. PQ3 must
   stay within ≤0.25 absolute PPL to be default-eligible (current
   microbench numeric error suggests this is comfortable, but it has not
   been measured end-to-end).
5. **rocprof counters** — confirm `SQ_WAVES` / `VALU_UTIL` for PQ3 is
   the bottleneck (compute-bound today) and that it inverts at the
   target shape (becomes bandwidth-bound).
6. **Working-set headroom** — on the target model, verify that fp16 KV
   + activations + weights actually spill out of the iGPU's 96 GB
   partition under realistic concurrency. If we comfortably fit, the
   compression isn't forced and we should lean on capacity, not
   compression, for now.

Only when **(1) + (2) + (3)** all show PQ3 > fp16 on service tok/s and
**(4)** confirms PPL parity do we flip the default. Until then: opt-in,
behind the flag.

---

Contact on questions: `project_bitnet_live_bench.md`,
`project_bitnet_rocprof_plan.md`.
