# bitnet-to-tq2

Repack the master bf16 weights of `microsoft/bitnet-b1.58-2B-4T-bf16` into a
`.h1b` **v4** container using PrismML's `TQ2_0_g128` block layout, so they
can be consumed by the new Bonsai TQ2 kernel path on `gfx1151`.

```
hf download microsoft/bitnet-b1.58-2B-4T-bf16 \
    --local-dir /home/bcloud/halo-ai/models/bitnet-bf16/

cargo build --release -p bitnet-to-tq2

./target/release/bitnet-to-tq2 \
    --in  /home/bcloud/halo-ai/models/bitnet-bf16/ \
    --out /home/bcloud/halo-ai/models/halo-bitnet-2b-tq2.h1b
```

## What the output file carries

Header:

```
magic  "H1B\0"          4  B
version   i32 = 4       4  B
cfg[0..9] i32           36 B   (hidden, ff, n_layers, n_heads, n_kv_heads,
                                vocab, max_seq_len, tie_embed, reserved=0x8)
rope_theta   f32        4  B
rms_norm_eps f32        4  B
```

Payload (in order):

1. Token embedding — raw fp32, `vocab × hidden`.
2. Final norm — raw fp32, `hidden`.
3. For each of `num_hidden_layers`:
   - Norm block, matching `onebit_core::h1b::serialize`'s exact layout so
     a downstream dispatcher can reuse the existing offset math:
     `[input_norm][post_attn_norm][attn_sub_norm × 4][trunc_ffn × 2][ffn_sub_norm]`.
   - Seven ternary tensors in canonical order: `q`, `k`, `v`, `o`, `gate`,
     `up`, `down`. Each is `rows × (cols/128) × 34 B` — **no trailing
     per-row scale array**, because TQ2 stores its fp16 scale inline
     (after the codes) in every 34 B block.

## Quantization rule — `absmax`, not `absmean`

We match oxibonsai's canonical PrismML quantizer
(`crates/oxibonsai-core/src/quant_ternary.rs:104-156`):

- `d     = max(|w_i|)` across each 128-weight group.
- `t     = 0.5 * d` (threshold).
- Codes: `0b10` if `w ≥ t`, `0b00` if `w ≤ -t`, `0b01` otherwise.

MS BitNet b1.58-2B-4T's bf16 master weights are already `{-1, 0, +1} × α`
per-tensor (pre-multiplied by the quantization scale). For these weights
`absmax` is lossless: `d = α`, every non-zero element sits on exactly
`±d`, and the round-trip is sign-exact.

`absmean` would under-estimate the non-zero level because ~35-50% of
ternary weights are zero in a typical BitNet layer; the threshold then
snaps samples that should have been `±d` down to zero and flips signs
right at the decision boundary. Don't use absmean on already-ternary
master weights.

## ARCHITECTURAL MISMATCH — must be resolved before decode

**The `H1B_FLAG_BONSAI_TQ2` bit only signals "weights are in TQ2_0_g128
format".** It does NOT tell the loader which transformer architecture to
run. The Bonsai TQ2 format was designed for PrismML's **Qwen3** family,
and the current `bitnet_decode` Bonsai-dispatch branch
(`rocm-cpp/tools/bitnet_decode.cpp`, authored by the HIP agent) assumes
Qwen3: RMSNorm-only, SwiGLU feed-forward, per-head `attn_q_norm` +
`attn_k_norm`.

MS BitNet b1.58-2B-4T, despite sharing the ternary quantization scheme,
is a **different architecture**:

- Squared-ReLU-GLU feed-forward (`hidden_act = "relu2"`), not SwiGLU.
- `attn_sub_norm` (per-layer RMSNorm on the post-attention residual
  stream), `ffn_sub_norm` (per-layer RMSNorm on the MLP gate/up output)
  — absent from Qwen3.
- No per-head `attn_q_norm` / `attn_k_norm`.
- LLaMA-3 vocab (128 256 tokens), tied embeddings, 30 layers, 20 heads,
  5 KV heads — distinct from the Bonsai-1.7B or Bonsai-8B Qwen3 shape.

**Consequence:** a `.h1b` produced by this tool, fed to today's Bonsai
dispatch branch, will produce garbage. The fix is a separate lane:
`bitnet_decode` must split its Bonsai-weight path into

- **"BonsaiTQ2 weights + Qwen3 arch"** (existing) — keeps today's
  PrismML Bonsai checkpoints.
- **"BonsaiTQ2 weights + BitNet arch"** (new) — uses the halo-family
  attention/MLP shape with the TQ2 GEMV kernel.

That split requires **either** adding a new flag bit (e.g.
`H1B_FLAG_BITNET_ARCH = 0x10`) **or** a new reserved-word enum variant.
Flag-bit assignments are not this tool's layer to invent; leaving the
dispatch-side split to a follow-up keeps the format committee ergonomic.

**Until that lane lands: do NOT run `bitnet_decode` with the `.h1b`
produced here.** The file is byte-well-formed (every kernel unit test
passes on its blocks), but the loader will route it to the wrong forward
pass.

## Non-ternary tensors — passed through as fp32

- **Embedding** (`model.embed_tokens.weight`): bf16 `[vocab, hidden]`.
  Decoded to raw fp32 and written into the `.h1b`'s embedding slot.
- **Final norm** (`model.norm.weight`): bf16 `[hidden]`. Raw fp32.
- **Per-layer norms** (`input_layernorm`, `post_attention_layernorm`,
  `self_attn.attn_sub_norm`, `mlp.ffn_sub_norm`): bf16 → raw fp32. The
  BitNet `attn_sub_norm` is written four times to fill the BitNet
  `attn_sub_norm × 4` slot convention in the existing `.h1b` layout
  (matches `onebit_core::h1b::serialize`).

These pass-throughs are already the correct layer shape for BitNet
inference; they're what a BitNet-arch dispatch branch would consume
verbatim.

## Tests

```
cargo test -p bitnet-to-tq2
```

Three in-crate tests:

- `quantize_group_absmean_roundtrip` — synth 128 ternary weights × 0.3,
  repack, unpack, assert ≤ 1% max-abs error.
- `header_flag_composition` — build a tiny 1-layer safetensors blob,
  round-trip through `convert()`, assert `reserved = 0x8`, `version = 4`,
  no other reserved bits set.
- `cli_arg_parsing` — `clap`-level arg parsing.
