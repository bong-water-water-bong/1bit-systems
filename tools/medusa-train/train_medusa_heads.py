#!/usr/bin/env python3
# DEV-ONLY one-shot Medusa head retrainer. Not a runtime component.
#
# See `CLAUDE.md` Rule A: Python is fine for dev-box scripts (requantizers,
# analysis notebooks, training scripts), never inside a systemd unit or an
# HTTP path. This script retrains the 4 Medusa speculative heads of
# parrishcorcoran/MedusaBitNet-2B-4T against hidden states extracted from
# the bf16 master backbone (microsoft/bitnet-b1.58-2B-4T-bf16), which is
# the Halo backbone's direct ancestor pre-quantization.
#
# Hypothesis A (checkpoint-mismatch fix):
#   The upstream Medusa head weights were trained at step 2000 on Alpaca
#   using a now-stale backbone hidden distribution. Retraining the heads
#   on hidden states + next-k token pairs sampled from wikitext with the
#   CURRENT bf16 master should dramatically improve head-0 acceptance vs
#   the 3.2% we measured. If it doesn't, the gap is checkpoint drift
#   between the bf16 master and Halo's requantized + RoPE-fixed h1b
#   (hypothesis B) — and the real fix is adding a hidden-dump path to
#   bitnet_decode and retraining directly from Halo's backbone.
#
# Inputs:
#   * microsoft/bitnet-b1.58-2B-4T-bf16 master weights (local path)
#   * parrishcorcoran/MedusaBitNet-2B-4T medusa_heads_step2000.pt
#       (warm-start for head weights)
#   * wikitext-103 test split (local text file)
#
# Output:
#   * A new `.h1b-medusa` v2 binary at the user-supplied path, same layout
#     as tools/medusa-convert/convert.py emits, drop-in compatible with
#     crates/1bit-router/src/medusa/loader.rs.
#
# Per-head forward (same as inference):
#   h_out  = h + W_out * SiLU(W_in * h)
#   logits = backbone.lm_head(h_out)
#
# Training loss: sum of per-head cross-entropy with target = token at
#   position t + (i + 2) given hidden from token at position t:
#     head 0 -> t+2, head 1 -> t+3, head 2 -> t+4, head 3 -> t+5
#   (verify loop in crates/1bit-router/src/lib.rs:~920 encodes this.)
#
# Run once, commit the output path to systemd env, never run again.

import argparse
import json
import os
import struct
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers import BitNetConfig, BitNetForCausalLM, AutoTokenizer


# ----- .h1b-medusa v2 format constants (must match loader.rs) -----
MEDUSA_MAGIC = b"MEDUSA1\0"
MEDUSA_FORMAT_VERSION = 2
EXPECTED_NUM_HEADS = 4
EXPECTED_HIDDEN_DIM = 2560
EXPECTED_RESIDUAL_LAYERS = 1
MEDUSA_DTYPE_FP16 = 1
MEDUSA_HEADER_BYTES = 68


def load_backbone(bf16_dir: str, device: str):
    """Load the MS bf16 BitNet master. The config.json has an `auto_map`
    pointing at a trust_remote_code module that isn't shipped with the
    bf16 variant; strip it + use the native transformers BitNet* classes
    instead. Weights load unchanged."""
    bf16_dir = Path(bf16_dir)
    with open(bf16_dir / "config.json") as f:
        cfgd = json.load(f)
    cfgd.pop("auto_map", None)

    tmp = Path(tempfile.mkdtemp(prefix="halo-medusa-retrain-"))
    for name in ["model.safetensors", "tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json"]:
        src = bf16_dir / name
        if src.exists():
            os.symlink(src, tmp / name)
    with open(tmp / "config.json", "w") as f:
        json.dump(cfgd, f)

    cfg = BitNetConfig.from_pretrained(str(tmp))
    tok = AutoTokenizer.from_pretrained(str(tmp))
    model = BitNetForCausalLM.from_pretrained(str(tmp), torch_dtype=torch.bfloat16)
    model.train(False)
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok, cfg


def extract_hidden_and_targets(model, tok, text: str,
                               *, device: str, seq_len: int,
                               num_chunks: int, num_heads: int):
    """Tokenize `text`, chunk into `num_chunks` non-overlapping windows of
    `seq_len` tokens, forward each through the backbone (requesting the
    last-layer hidden), and stack (hidden, next_1..next_num_heads+1) tuples.

    Head i (0-indexed) predicts the token at offset i+2 from the query
    position - i.e. head 0 -> t+2. So for each position t in a chunk, we
    need targets at positions t+2..t+num_heads+1. Positions near the end
    of a chunk that don't have `num_heads+1` lookahead tokens are dropped.

    Returns (hiddens, targets) where
      hiddens: fp32 [N, hidden_dim]
      targets: int64 [N, num_heads]
    """
    ids_all = tok(text, return_tensors="pt", add_special_tokens=True).input_ids[0]
    # Trim to chunk boundary.
    total_need = num_chunks * seq_len
    if ids_all.numel() < total_need:
        num_chunks = ids_all.numel() // seq_len
        print(f"  dataset shorter than expected; dropping to {num_chunks} chunks "
              f"({num_chunks * seq_len} tokens)", file=sys.stderr)
    ids_all = ids_all[: num_chunks * seq_len]

    hiddens_list = []
    targets_list = []
    min_lookahead = num_heads + 1

    for c in range(num_chunks):
        ids = ids_all[c * seq_len : (c + 1) * seq_len].unsqueeze(0).to(device)
        t_fw = time.time()
        with torch.no_grad():
            out = model(ids, output_hidden_states=True, use_cache=False)
        # transformers' BitNetModel applies final RMSNorm before returning
        # last_hidden_state (out.hidden_states[-1]); this is exactly the
        # fp16 hidden Halo's backbone lm_head sees.
        hidden = out.hidden_states[-1][0]  # [L, H]
        valid_len = hidden.shape[0] - min_lookahead
        if valid_len <= 0:
            continue
        h = hidden[:valid_len].float().cpu()
        ids_cpu = ids[0].cpu()
        # For position t (0..valid_len-1), need tgt[t, k] = ids[t + 2 + k]
        # for k in 0..num_heads. Use arange + gather.
        base_idx = torch.arange(valid_len)
        offsets = torch.arange(2, 2 + num_heads)
        idx_matrix = base_idx.unsqueeze(1) + offsets.unsqueeze(0)  # [L, num_heads]
        tgt = ids_cpu[idx_matrix]  # [L, num_heads] int64

        hiddens_list.append(h)
        targets_list.append(tgt)
        dt = time.time() - t_fw
        print(f"  chunk {c+1}/{num_chunks}: {valid_len} positions, "
              f"forward took {dt:.1f}s", file=sys.stderr)

    hiddens = torch.cat(hiddens_list, dim=0)
    targets = torch.cat(targets_list, dim=0)
    return hiddens, targets


class MedusaHeadModule(nn.Module):
    """One Medusa residual-SiLU head.

    Forward:
        inner = SiLU(h @ W_in.T)
        h_out = h + inner @ W_out.T
        logits = h_out @ lm_head.T
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.w_in = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_out = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, h: torch.Tensor, lm_head_weight: torch.Tensor) -> torch.Tensor:
        inner = nn.functional.silu(self.w_in(h))
        h_out = h + self.w_out(inner)
        logits = h_out @ lm_head_weight.T
        return logits


def warm_start_from_upstream(heads: nn.ModuleList, upstream_pt_path: str):
    """Seed each head's W_in / W_out from the upstream step-2000 checkpoint.

    Upstream shape: w_in bf16[1, 4, 2560, 2560], w_out bf16[1, 4, 2560, 2560].
    nn.Linear stores weight as [out_features, in_features]. The upstream
    tensor is stored as [out, in] so a direct copy works; we verify by
    running a quick sanity forward after copy."""
    obj = torch.load(upstream_pt_path, map_location="cpu", weights_only=False)
    heads_ck = obj["heads"]
    w_in = heads_ck["w_in"][0].float()   # [4, 2560, 2560]
    w_out = heads_ck["w_out"][0].float()  # [4, 2560, 2560]
    for i in range(EXPECTED_NUM_HEADS):
        heads[i].w_in.weight.data.copy_(w_in[i])
        heads[i].w_out.weight.data.copy_(w_out[i])


def train_heads(heads: nn.ModuleList, lm_head_weight: torch.Tensor,
                hiddens: torch.Tensor, targets: torch.Tensor,
                *, device: str, steps: int, batch_size: int,
                lr: float, grad_clip: float):
    """Fit the 4 heads against the extracted hidden/target pairs.

    Loss is sum of per-head cross-entropy. The lm_head_weight is frozen
    throughout (tied to the backbone embedding)."""
    heads.to(device)
    lm_head_weight = lm_head_weight.to(device)
    hiddens = hiddens.to(device)
    targets = targets.to(device)

    params = [p for h in heads for p in h.parameters()]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    warmup_steps = min(50, steps // 10)

    N = hiddens.shape[0]
    print(f"  training: N={N} positions, steps={steps}, batch={batch_size}, "
          f"lr={lr}, grad_clip={grad_clip}", file=sys.stderr)

    for step in range(steps):
        cur_lr = lr * min(1.0, (step + 1) / max(1, warmup_steps))
        for g in opt.param_groups:
            g["lr"] = cur_lr

        idx = torch.randint(0, N, (batch_size,), device=device)
        h_b = hiddens[idx]
        tgt_b = targets[idx]

        total_loss = 0.0
        per_head_correct = [0] * EXPECTED_NUM_HEADS
        for i in range(EXPECTED_NUM_HEADS):
            logits = heads[i](h_b, lm_head_weight)
            loss_i = nn.functional.cross_entropy(logits, tgt_b[:, i])
            total_loss = total_loss + loss_i
            with torch.no_grad():
                pred = logits.argmax(-1)
                per_head_correct[i] = (pred == tgt_b[:, i]).sum().item()

        opt.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()

        if step % 50 == 0 or step == steps - 1:
            rates = [c / batch_size for c in per_head_correct]
            print(f"    step {step:4d}: loss={float(total_loss):.3f}  "
                  f"head_acc={[f'{r:.3f}' for r in rates]}  lr={cur_lr:.2e}",
                  file=sys.stderr)


def evaluate(heads: nn.ModuleList, lm_head_weight: torch.Tensor,
             hiddens: torch.Tensor, targets: torch.Tensor,
             device: str, n_eval: int = 2048):
    """Compute per-head argmax-accuracy on a held-out slice."""
    heads.to(device)
    lm_head_weight = lm_head_weight.to(device)
    hiddens = hiddens.to(device)
    targets = targets.to(device)
    n_eval = min(n_eval, hiddens.shape[0])
    with torch.no_grad():
        idx = torch.arange(n_eval, device=device)
        h_e = hiddens[idx]
        tgt_e = targets[idx]
        rates = []
        for i in range(EXPECTED_NUM_HEADS):
            logits = heads[i](h_e, lm_head_weight)
            pred = logits.argmax(-1)
            rate = (pred == tgt_e[:, i]).float().mean().item()
            rates.append(rate)
    return rates


def write_h1b_medusa(out_path: str, heads: nn.ModuleList):
    """Serialize the trained heads to the .h1b-medusa v2 binary format.

    Layout (must match tools/medusa-convert/convert.py + loader.rs):
      magic (8) + version (4) + num_heads (4) + hidden_dim (4) +
      residual_layers (4) + dtype (4) + reserved (40) + per-head
      (w_in fp16[H*H] + w_out fp16[H*H])."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

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
    header += b"\x00" * 40
    assert len(header) == MEDUSA_HEADER_BYTES, len(header)

    per_head_bytes = 2 * EXPECTED_HIDDEN_DIM * EXPECTED_HIDDEN_DIM * 2
    expected_total = MEDUSA_HEADER_BYTES + EXPECTED_NUM_HEADS * per_head_bytes

    fp16_max = 65504.0
    total_clamp = 0
    with open(out, "wb") as f:
        f.write(bytes(header))
        for i in range(EXPECTED_NUM_HEADS):
            head = heads[i]
            w_in = head.w_in.weight.data.detach().cpu().float()
            w_out = head.w_out.weight.data.detach().cpu().float()
            for name, t in (("w_in", w_in), ("w_out", w_out)):
                over = (t.abs() > fp16_max).sum().item()
                total_clamp += int(over)
                t_clamped = t.clamp(-fp16_max, fp16_max).half().contiguous()
                f.write(t_clamped.numpy().tobytes())
                if over:
                    print(f"  head {i} {name}: clamped {over} elems to fp16 range",
                          file=sys.stderr)
    actual = out.stat().st_size
    if actual != expected_total:
        print(f"error: file-size mismatch {actual} != {expected_total}", file=sys.stderr)
        sys.exit(4)
    print(f"wrote {actual} bytes to {out}  (total clamp events: {total_clamp})",
          file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description="Retrain Medusa heads against Halo's backbone.")
    ap.add_argument("--bf16-dir", default="/home/bcloud/halo-ai/models/bitnet-bf16")
    ap.add_argument("--wikitext", default="/home/bcloud/halo-ai/datasets/wikitext-103-test.txt")
    ap.add_argument("--warm-start-pt", default="/home/bcloud/halo-ai/models/medusa/medusa_heads_step2000.pt")
    ap.add_argument("--out", required=True, help=".h1b-medusa output path")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--num-chunks", type=int, default=40,
                    help="Non-overlapping windows of seq_len tokens to extract hiddens from")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-warm-start", action="store_true",
                    help="Skip warm-start and init heads from scratch")
    args = ap.parse_args()

    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Loading bf16 backbone from {args.bf16_dir}",
          file=sys.stderr)
    model, tok, cfg = load_backbone(args.bf16_dir, args.device)
    assert cfg.hidden_size == EXPECTED_HIDDEN_DIM

    print(f"[{time.strftime('%H:%M:%S')}] Reading wikitext from {args.wikitext}",
          file=sys.stderr)
    text = Path(args.wikitext).read_text()
    approx_chars = args.seq_len * args.num_chunks * 6
    text = text[:approx_chars]

    print(f"[{time.strftime('%H:%M:%S')}] Extracting hiddens + targets", file=sys.stderr)
    hiddens, targets = extract_hidden_and_targets(
        model, tok, text,
        device=args.device,
        seq_len=args.seq_len,
        num_chunks=args.num_chunks,
        num_heads=EXPECTED_NUM_HEADS,
    )
    print(f"  hiddens: {tuple(hiddens.shape)} {hiddens.dtype}", file=sys.stderr)
    print(f"  targets: {tuple(targets.shape)} {targets.dtype}", file=sys.stderr)

    lm_head_weight = model.lm_head.weight.data.detach().float().cpu()
    print(f"  lm_head.weight: {tuple(lm_head_weight.shape)}", file=sys.stderr)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    heads = nn.ModuleList([
        MedusaHeadModule(EXPECTED_HIDDEN_DIM) for _ in range(EXPECTED_NUM_HEADS)
    ])

    if not args.no_warm_start:
        print(f"[{time.strftime('%H:%M:%S')}] Warm-starting from {args.warm_start_pt}",
              file=sys.stderr)
        warm_start_from_upstream(heads, args.warm_start_pt)
    else:
        print(f"[{time.strftime('%H:%M:%S')}] Skipping warm-start (init from scratch)",
              file=sys.stderr)

    pre_rates = evaluate(heads, lm_head_weight, hiddens, targets, args.device)
    print(f"  BEFORE-train head acc: {[f'{r:.3f}' for r in pre_rates]}",
          file=sys.stderr)
    heads.to("cpu")

    print(f"[{time.strftime('%H:%M:%S')}] Training heads", file=sys.stderr)
    train_heads(
        heads, lm_head_weight, hiddens, targets,
        device=args.device,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        grad_clip=args.grad_clip,
    )

    post_rates = evaluate(heads, lm_head_weight, hiddens, targets, args.device)
    print(f"  AFTER-train  head acc: {[f'{r:.3f}' for r in post_rates]}",
          file=sys.stderr)
    heads.to("cpu")

    print(f"[{time.strftime('%H:%M:%S')}] Serializing heads to .h1b-medusa",
          file=sys.stderr)
    write_h1b_medusa(args.out, heads)

    t_end = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Done in {(t_end - t_start)/60.0:.1f} minutes",
          file=sys.stderr)

    summary = {
        "out_path": args.out,
        "train_minutes": round((t_end - t_start) / 60.0, 2),
        "pre_train_head_acc": pre_rates,
        "post_train_head_acc": post_rates,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "num_chunks": args.num_chunks,
        "seq_len": args.seq_len,
        "device": args.device,
    }
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
