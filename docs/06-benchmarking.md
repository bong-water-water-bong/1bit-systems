# Benchmarking

## Baseline — Vulkan Path (llama.cpp)

Qwen3-Coder-Next-GGUF on CachyOS kernel 7.0.0-1-cachyos:

| Test            | Prompt t/s | Gen t/s | TTFT ms | Total ms |
|-----------------|-----------|---------|---------|----------|
| Short Burst     | 146.5     | 84.2    | 137     | 160      |
| Medium Response | 141.2     | 47.8    | 142     | 2889     |
| Long Generation | 196.4     | 47.4    | 168     | 10961    |
| Sustained 2K    | 212.6     | 47.1    | 221     | 43742    |
| Code Gen        | 194.2     | 47.3    | 170     | 21813    |
| Reasoning       | 201.6     | 47.7    | 179     | 3919     |
| Long Context    | 279.5     | 47.4    | 257     | 11056    |

Key observations:
- Generation speed is flat at 47.4 t/s regardless of output length — no degradation
- Prompt processing scales UP with input length (146 → 279 t/s)
- TTFT under 260ms even for long context

## Previous Benchmarks — MLX ROCm

| Path              | Speed     | Notes                        |
|-------------------|-----------|------------------------------|
| MLX ROCm C++      | 151 t/s   | lemon-mlx-engine             |
| MLX ROCm Python   | 52 t/s    | goniz fork                   |
| halo-1bit Phase 1 | 1.22 t/s  | Dequantize path (worst case) |

The 29-85% MLX advantage over vLLM was on 4-bit quantized models using warp-cooperative GEMV.

## What We're Targeting

| Path                  | Target    | How                              |
|-----------------------|-----------|----------------------------------|
| ROCm 4-bit (TheRock)  | 200+ t/s  | Native Tensile GEMM for gfx1151  |
| ROCm ternary (custom) | 250+ t/s  | Wave32 fused ternary kernel      |
| Vulkan (portable)     | 47+ t/s   | Already achieved, baseline       |

## Running the Bench

The lemonade Vulkan backend includes a benchmark suite:

```bash
# Run the standard bench
lemonade bench --backend vulkan --model <model_path>

# Specific tests
lemonade bench --backend vulkan --model <model_path> --test "Short Burst,Long Generation"
```

## Custom Benchmark (for ROCm kernels)

When we build the custom kernels, benchmark with:

```bash
# Kernel-level timing
./tools/bench_kernel --kernel ternary_gemv --m 2560 --k 2560 --iterations 1000

# End-to-end inference timing
./tools/bench_inference --model models/halo-1bit-2b.h1b --prompt "Hello" --max-tokens 512
```

## Metrics That Matter

1. **Generation tok/s** — sustained, not peak. Measure at 512+ tokens.
2. **Prompt tok/s** — how fast prefill processes. Scales with input length.
3. **TTFT (Time to First Token)** — latency. Under 200ms is good.
4. **Memory usage** — GPU memory consumed during inference.
5. **Power draw** — Strix Halo is 120W TDP. Perf/watt matters for APU.

## Comparing Against Others

| Hardware              | Model            | Gen t/s | Notes                    |
|-----------------------|------------------|---------|--------------------------|
| RX 6700 XT            | Bonsai 1.7B Q1   | 209     | carlosfundora Wave32     |
| Strix Halo (Vulkan)   | Qwen3-Coder GGUF | 47.4    | Our baseline             |
| Strix Halo (MLX C++)  | Various 4-bit    | 151     | lemon-mlx-engine         |
| Strix Halo (MLX Py)   | Various 4-bit    | 52      | Python overhead           |
| Strix Halo (halo-1bit)| BitNet-2B        | 1.22    | Phase 1 (dequant path)   |
