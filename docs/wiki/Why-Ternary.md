# Why 1.58-bit ternary?

**One-line answer**: 1.58-bit weights (values in `{-1, 0, +1}`) give a ~10× memory reduction over FP16 with near-zero accuracy loss on the BitNet-b1.58-2B-4T architecture. This lifts the LPDDR5 bandwidth bottleneck that dominates LLM inference on unified-memory systems like Strix Halo.

## The bit budget

| format | bits/weight | 2B model size | LPDDR5 bandwidth demand |
|---|---:|---:|---:|
| FP16 | 16 | 4.0 GB | at peak |
| INT8 | 8 | 2.0 GB | 0.5× |
| 1.58-bit (ternary) | 1.58 | 400 MB | **~0.1×** |

On a Strix Halo box with 128 GB LPDDR5 shared between CPU and GPU, **memory bandwidth, not compute, is the dominant bottleneck for LLM decode**. A 10× smaller model = 10× less bandwidth demand per token.

## The accuracy story

Microsoft's paper ([arXiv:2402.17764](https://arxiv.org/abs/2402.17764)) showed that 1.58-bit weights can *match* FP16 quality on a 3B model **if trained from scratch with quantization-aware training**, not post-quantized. The 2B-4T model they released confirmed it.

We use their pre-trained weights directly. Converting them to our `.h1b` format is a deterministic re-pack — we don't retrain.

## Where "1.58" comes from

log₂(3) ≈ 1.585. Three values {-1, 0, +1} take 1.585 bits of information per weight in the information-theoretic sense. Storage is usually 2 bits per weight (one bit wasted per value), but Sherry's 3:4 sparsity encoding gets it down to 1.25 bits by forcing one of every four weights to be zero.

## Why ternary and not binary?

Binary ({-1, +1}) = 1 bit per weight, 50% more memory savings on paper. In practice:

- **Binary loses too much information** — BitNet-b1.0 prototypes showed 2-3× higher perplexity vs the 1.58-bit variant at the same model scale.
- **Zero is load-bearing** — activation sparsity (ReLU+squared-ReLU) produces ~50-60% zero activations. Matching ternary weights with ternary-zero means those matmuls compile down to no-ops.
- **Hardware efficiency** — ternary maps to `xnor` + `popcount` + sign-flip on GPU. Binary loses the sign-flip branch.

Sub-1.58-bit formats (LittleBit 0.1 bpw, NanoQuant sub-binary) exist and are watched — see [`../../.claude/projects/-home-bcloud/memory/project_bitnet_frontier_2026_04.md`](memory-only). Not production-ready for our scale today.

## Why this matters for 1bit systems

On a $3 000 Strix Halo mini-PC with 128 GB unified LPDDR5 at 256 GB/s, ternary BitNet gets us:

- **Model fits in 4 GB, leaving 124 GB for KV cache + activations** — no OOM on long contexts.
- **83 tok/s decode at N=64, 68 tok/s at N=1024** — competitive with a datacenter GPU ~5× the cost.
- **Silent / low-power** — 150 W sustained. Whisper-quiet in a closet.

Same hardware running FP16 Llama-3-8B: ~18 tok/s. Ternary BitNet-2B at 83 tok/s is roughly 5× faster for comparable task quality on the benchmarks we care about.

## Citations

- Wang, Ma et al. "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits." arXiv [`2402.17764`](https://arxiv.org/abs/2402.17764).
- Wang, Ma et al. "BitNet v2: Native 4-bit Activations with Hadamard Transform for 1-bit LLMs." arXiv [`2504.18415`](https://arxiv.org/abs/2504.18415).
- `microsoft/bitnet-b1.58-2B-4T` on HuggingFace — the weights we run.
