# CK Integration Path — BitNet Prefill on gfx1151

Measured: CK `DeviceGemm_Wmma_CShuffleV3<F16, pk_i4, F16, ...>` runs BitNet ternary
weights at **29.2–29.9 TFlops** on FFN shapes (2560×6912×2560, 2560×2560×6912) —
**0.96× rocBLAS FP16** throughput at **half the B memory**, **4–5× over our v2.4
hand-rolled WMMA INT8** kernel.

Source of the measurement: `ck-prefill/gemm_wmma_fp16_ternary_as_pk_i4.cpp`,
`init_method=99` clamps pk_i4 nibbles to {-1, 0, +1}.

## Dispatcher

```
batch · seq_len    kernel                          why
─────────────────  ──────────────────────────────  ─────────────────────────────
       1           v1 ternary GEMV (rocm-cpp)      memory-bound; CK GEMM underused
       2 … 7       v1 ternary GEMV                 still memory-bound
       8 … ∞       CK FP16 × pk_i4 WMMA            compute-bound; WMMA wins
```

Threshold is empirical; measure at integration time. GEMV stays our v1 kernel
(already shipped, already competitive vs rocBLAS FP16 GEMV at batch=1 where
rocBLAS is 0.2–0.5 TFlops).

## Weight layout (offline, one-time at model load)

BitNet `.h1b` ternary → `pk_i4` with WMMA-compatible permutation:

```
  for each weight W[K, N]:
      1. clamp ternary {-1,0,+1} → int4 {0xF, 0x0, 0x1}  (sign-extended 4-bit)
      2. block-reshape [K, N] → [K0, N, K1]              K1 = KPerBlock = 32
      3. pk_i4x4 lane permute within each 8-value group  01234567 → 20643175
      4. pack 2 int4 per byte                             hi nibble | lo nibble
```

Steps 2–3 are copied verbatim from CK's `gemm_wmma_fp16_pk_i4_v3.cpp` reference
(`PermuteB = true` branch, lines 139–215). Step 1 is our ternary-specific clamp.

Storage: `bytes = K * N / 2` (half of FP16). For BitNet-2B FFN up projection
(6912 × 2560), that's **8.85 MB per matrix** vs 35.4 MB FP16. Across all FFN +
attention projections in 30 layers, we save ~800 MB vs FP16.

## C API (rocm-cpp → consumers)

Expose a thin shim so halo-1bit, lemond, and external users don't pull CK
templates into their translation units.

```c
// include/rocm-cpp/ck_gemm.h
typedef enum { RCPP_LAYOUT_ROW, RCPP_LAYOUT_COL } rcpp_layout_t;

typedef struct rcpp_ck_gemm_handle rcpp_ck_gemm_handle_t;

/* Create a reusable handle — selects instance based on shape. */
rcpp_ck_gemm_handle_t* rcpp_ck_gemm_create(int M, int N, int K,
                                           rcpp_layout_t a_layout,
                                           rcpp_layout_t b_layout);

/* A: FP16 [M,K] row-major. B: pk_i4 [K,N] col-major pre-permuted. C: FP16 [M,N] row-major. */
int rcpp_ck_gemm_run(rcpp_ck_gemm_handle_t* h,
                     const _Float16* A_dev,
                     const int8_t*   B_dev_pk_i4_permuted,
                     _Float16*       C_dev,
                     hipStream_t     stream);

void rcpp_ck_gemm_destroy(rcpp_ck_gemm_handle_t* h);

/* Offline weight packer: ternary int8 [K,N] → pk_i4 [K,N/2] WMMA-permuted. */
void rcpp_ternary_pack_pk_i4(const int8_t* ternary_host,  /* values in {-1,0,+1} */
                             int8_t*       packed_host,   /* output, size K*N/2 */
                             int K, int N);
```

Implementation: `src/ck_gemm.cpp` instantiates the same
`DeviceGemm_Wmma_CShuffleV3` template used in `ck-prefill/` and hides the HIP
include surface behind the C header. Link: `libutility.a` (CK), `hip::device`.

## Integration into halo-1bit/mlx-engine

Today the engine has `ternary_gpu.cpp` (dequantize-matmul, 1.22 tok/s on 2.4B).
Adding CK prefill:

1. At `.h1b` load in `src/mlx_loader.cpp`: after reading ternary weights, call
   `rcpp_ternary_pack_pk_i4` once per weight tensor. Store the packed buffer
   alongside the existing FP16 mirrors (which stay for decode).

2. In the forward pass (`src/mlx_model.cpp`): branch on `batch * seq_len`.
   - ≥ 8: `rcpp_ck_gemm_run` with packed weight
   - < 8: existing GEMV path

3. KV cache path is unchanged — CK only touches weight × activation GEMMs.

## Open items

- Verify the host reference `check_err` path in `ck-prefill/` with a real
  end-to-end ternary tensor (currently verified only on the clamp-to-ternary
  synthetic case). Plug in a known BitNet-2B FFN tensor.

- Measure the batch-threshold crossover empirically on Bonsai-1.7B and
  BitNet-2B. The GEMV → GEMM breakeven is probably in the 4–16 range on
  gfx1151 but needs a sweep.

- Revisit `pk_i1_t` if 8× weight compression beats 4× on decode bandwidth.
  Requires CK dtype extension (days of work). Deferred; pk_i4 hits 0.96× rocBLAS
  already.
