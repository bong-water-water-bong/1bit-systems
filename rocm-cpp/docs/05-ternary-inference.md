# Ternary (1-bit) Inference on ROCm

## What Is Ternary

BitNet b1.58 uses ternary weights: each weight is -1, 0, or +1. That's 1.58 bits per weight (log2(3)). This eliminates all floating-point multiplications in the linear layers — matrix-vector products become additions and subtractions only.

A 2.4B parameter model that would be 4.8GB in bf16 becomes 1.755GB in packed ternary format (.h1b).

## Why It's Fast (In Theory)

- No multiplications — just add/subtract based on ternary value
- 16x less memory bandwidth than fp16
- Compute-bound becomes memory-bound becomes trivially parallel
- On unified memory (Strix Halo), the reduced data movement is even more impactful

## Why It's Slow Right Now

Nobody has built the kernel. Current path:

1. Load packed ternary weights
2. **Dequantize to fp16** (expand -1/0/+1 back to floating point)
3. Use standard fp16 GEMM (rocBLAS)
4. Result: 1.22 tok/s

This is absurd. You're paying the full cost of fp16 GEMM after throwing away the entire advantage of ternary encoding.

## The Right Approach

Fused ternary kernel — never dequantize:

1. Load packed ternary weights (16 values per 32-bit word)
2. Decode ternary values in-register using bit manipulation
3. For each weight: if +1, add activation; if -1, subtract; if 0, skip
4. Accumulate in fp32 for numerical stability
5. Wave32 reduction for final output

No GEMM library involved. No intermediate buffers. No dequantization.

## Packing Format

### .h1b Format (Current)

Our export format from `scripts/export_base_h1b.py`:

- Header: model dimensions, layer count, vocab size
- Weights: packed ternary (2 bits per value, 16 values per uint32)
- Encoding: 0b00 = 0, 0b01 = +1, 0b10 = -1

### Q1_0_G128 (carlosfundora)

- Groups of 128 elements
- Each group: 128 ternary values packed into 32 bytes (2 bits each)
- Plus 1 scale factor (fp16)
- Total: 34 bytes per 128 elements

## Kernel Design (Phase 2)

```
Input: packed ternary weight matrix W [M x K], activation vector x [K]
Output: result vector y [M]

For each output element y[m]:
    accumulator = 0.0f
    for k in range(0, K, 128):
        // Load 128 packed ternary values (8 uint32s)
        // Decode in-register
        // Conditional add/subtract from x[k:k+128]
        // Accumulate
    y[m] = accumulator
```

### Thread Mapping

- 1 thread block per output row (or tile of rows)
- 128 threads = 4 Wave32 warps
- Each thread processes K/128 groups
- Wave32 reduction to sum partial results

### Expected Performance

carlosfundora achieved 209 tok/s on RX 6700 XT (1.7B model) with this approach.
Strix Halo has higher memory bandwidth (256 GB/s LPDDR5X vs 192 GB/s GDDR6).
Target: 250+ tok/s on BitNet-2B-4T.

## Model Architecture — BitNet b1.58-2B-4T

- Parameters: 2.4B
- hidden_size: 2560
- num_hidden_layers: 30
- num_attention_heads: 20
- num_key_value_heads: 5 (GQA)
- intermediate_size: 6912
- vocab_size: 128256 (Llama 3 tokenizer)
- rope_theta: 500000.0
- activation: relu2 (NOT SiLU — this matters for the kernel)
- Packed .h1b size: 1.755 GB

## What Exists Already

In `/home/bcloud/halo-1bit/`:
- `mlx-engine/` — C++ server with OpenAI API (Phase 1, dequantize path)
- `scripts/export_base_h1b.py` — safetensors to .h1b converter
- `scripts/export_tokenizer.py` — tokenizer to .htok converter
- `models/halo-1bit-2b.h1b` — exported model (1.755 GB)
- `models/halo-1bit-2b.htok` — exported tokenizer

## What Needs to Be Built

1. Fused Wave32 ternary GEMV kernel (the core)
2. Attention kernel with RoPE (can reuse existing HIP implementations)
3. ReLU2 activation (simple — relu(x)^2)
4. RMSNorm kernel
5. KV cache management for autoregressive generation
6. Integration with the existing server
