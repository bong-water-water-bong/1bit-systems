#!/usr/bin/env python3
# DEV-ONLY one-shot converter. Not a runtime component.
#
# See `CLAUDE.md` Rule A: Python is fine for scripts that run once on a
# dev box (requantizers, analysis notebooks), never inside a systemd
# unit or a path that serves HTTP. This script turns the upstream
# `parrishcorcoran/MedusaBitNet-2B-4T` `medusa_heads_step2000.pt`
# checkpoint into the native `.h1b-medusa` v2 binary consumed by
# `crates/1bit-router/src/medusa/loader.rs`.
#
# Inputs  (from `torch.load`):
#   heads.w_in:  bf16[1, 4, 2560, 2560]
#   heads.w_out: bf16[1, 4, 2560, 2560]
#
# Output (little-endian throughout):
#
#   offset  size   field
#   ------  ----   -----
#     0      8     MAGIC            b"MEDUSA1\0"
#     8      4     version          u32 == 2
#    12      4     num_heads        u32 == 4
#    16      4     hidden_dim       u32 == 2560
#    20      4     residual_layers  u32 == 1
#    24      4     dtype            u32 == 1 (fp16)
#    28     40     reserved         zeros
#    68    ...     per-head payload, in head-index order:
#                    w_in   fp16[2560 * 2560]
#                    w_out  fp16[2560 * 2560]
#
# Per-head = 2 * 2560 * 2560 * 2 = 26_214_400 bytes = 25 MiB.
# Total   = 68 + 4 * 26_214_400  = 104_857_668 bytes ≈ 100 MiB.
#
# Run once, commit the output path to systemd env, never run again.

import argparse
import struct
import sys
from pathlib import Path

import torch

MEDUSA_MAGIC = b"MEDUSA1\0"
MEDUSA_FORMAT_VERSION = 2
EXPECTED_NUM_HEADS = 4
EXPECTED_HIDDEN_DIM = 2560
EXPECTED_RESIDUAL_LAYERS = 1
MEDUSA_DTYPE_FP16 = 1
MEDUSA_HEADER_BYTES = 68


def bf16_to_fp16_clamped(t: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Cast a bf16 tensor to fp16, clamping to the fp16 representable
    range before the cast. Returns (fp16_tensor, clamp_event_count).

    fp16 max finite is 65504. bf16 can represent up to ~3.4e38 so values
    outside fp16 range are possible in principle — in practice the
    Medusa heads are trained with `weight_decay = 0.0` and `grad_clip =
    1.0` so outliers are rare but possible. We clamp elementwise and
    count how many elements got clipped so the caller can log it.
    """
    assert t.dtype == torch.bfloat16, f"expected bf16, got {t.dtype}"
    # Upcast to fp32 for the clamp so we don't lose precision on values
    # near the fp16 edge.
    f32 = t.to(torch.float32)
    fp16_max = 65504.0
    over = (f32.abs() > fp16_max).sum().item()
    f32 = f32.clamp(min=-fp16_max, max=fp16_max)
    return f32.to(torch.float16), int(over)


def main() -> int:
    p = argparse.ArgumentParser(description="Convert Medusa .pt → .h1b-medusa v2")
    p.add_argument("--in", dest="input", required=True, help="path to medusa_heads_step2000.pt")
    p.add_argument("--out", dest="output", required=True, help="path to halo-medusa-heads.h1b-medusa")
    args = p.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()
    if not in_path.exists():
        print(f"error: input {in_path} does not exist", file=sys.stderr)
        return 2

    print(f"loading {in_path} ...", file=sys.stderr)
    obj = torch.load(str(in_path), map_location="cpu", weights_only=False)
    if not isinstance(obj, dict) or "heads" not in obj:
        print(f"error: checkpoint has no 'heads' key (got {type(obj).__name__})", file=sys.stderr)
        return 3

    heads = obj["heads"]
    if "w_in" not in heads or "w_out" not in heads:
        print(f"error: heads missing 'w_in' / 'w_out' (got keys {list(heads.keys())})", file=sys.stderr)
        return 3

    w_in = heads["w_in"]   # expected bf16 [1, 4, 2560, 2560]
    w_out = heads["w_out"] # expected bf16 [1, 4, 2560, 2560]

    # Shape + dtype sanity.
    for name, t in (("w_in", w_in), ("w_out", w_out)):
        if t.dtype != torch.bfloat16:
            print(f"error: {name} dtype {t.dtype}, expected bfloat16", file=sys.stderr)
            return 3
        if tuple(t.shape) != (1, EXPECTED_NUM_HEADS, EXPECTED_HIDDEN_DIM, EXPECTED_HIDDEN_DIM):
            print(
                f"error: {name} shape {tuple(t.shape)}, expected "
                f"(1, {EXPECTED_NUM_HEADS}, {EXPECTED_HIDDEN_DIM}, {EXPECTED_HIDDEN_DIM})",
                file=sys.stderr,
            )
            return 3

    # Drop the leading singleton so indexing lines up with head index.
    w_in = w_in[0]   # [4, 2560, 2560]
    w_out = w_out[0] # [4, 2560, 2560]

    # Build the header — exactly 68 bytes, little-endian throughout.
    header = bytearray()
    header += MEDUSA_MAGIC
    header += struct.pack(
        "<IIIII",
        MEDUSA_FORMAT_VERSION,
        EXPECTED_NUM_HEADS,
        EXPECTED_HIDDEN_DIM,
        EXPECTED_RESIDUAL_LAYERS,
        MEDUSA_DTYPE_FP16,
    )
    header += b"\x00" * 40  # reserved: 10 × u32 of zero
    assert len(header) == MEDUSA_HEADER_BYTES, len(header)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_clamp = 0
    per_head_bytes = 2 * EXPECTED_HIDDEN_DIM * EXPECTED_HIDDEN_DIM * 2
    expected_total = MEDUSA_HEADER_BYTES + EXPECTED_NUM_HEADS * per_head_bytes

    print(
        f"writing {out_path} (expected {expected_total} bytes = "
        f"header {MEDUSA_HEADER_BYTES} + {EXPECTED_NUM_HEADS}×{per_head_bytes})",
        file=sys.stderr,
    )
    with open(out_path, "wb") as f:
        f.write(bytes(header))
        for i in range(EXPECTED_NUM_HEADS):
            # w_in then w_out for head i.
            w_in_i, n_over_in = bf16_to_fp16_clamped(w_in[i].contiguous())
            w_out_i, n_over_out = bf16_to_fp16_clamped(w_out[i].contiguous())
            total_clamp += n_over_in + n_over_out

            # fp16 tensors → little-endian u16 bytes. Torch is natively LE
            # on x86_64; `tobytes` on a contiguous tensor of dtype fp16
            # emits exactly that.
            f.write(w_in_i.numpy().tobytes())
            f.write(w_out_i.numpy().tobytes())
            print(
                f"  head {i}: w_in clamp {n_over_in}, w_out clamp {n_over_out}",
                file=sys.stderr,
            )

    actual = out_path.stat().st_size
    print(f"wrote {actual} bytes", file=sys.stderr)
    if actual != expected_total:
        print(f"error: file size mismatch ({actual} != {expected_total})", file=sys.stderr)
        return 4

    print(f"clamp events (|x| > 65504): {total_clamp}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
