# Full 1-bit burn against TheRock 7.13 native gfx1151
Date: Thu Apr 16 08:51:09 PM ADT 2026
Stack: PrismML prism branch + TheRock 7.13
rocBLAS lib: /home/bcloud/therock/build/dist/rocm/lib/librocblas.so.5.4
Tensile libpath: /home/bcloud/therock/build/dist/rocm/lib/rocblas/library

=== Bonsai-1.7B Q1_0 ===
Model: /home/bcloud/models/bonsai/Bonsai-1.7B.gguf

ggml_cuda_init: found 1 ROCm devices (Total VRAM: 63967 MiB):
  Device 0: Radeon 8060S Graphics, gfx1151 (0x1151), VMM: no, Wave Size: 32, VRAM: 63967 MiB
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen3 1.7B Q1_0_g128           | 231.13 MiB |     1.72 B | ROCm       |  99 |           pp512 |      3096.70 ± 28.80 |
| qwen3 1.7B Q1_0_g128           | 231.13 MiB |     1.72 B | ROCm       |  99 |           tg128 |        232.38 ± 0.84 |

build: 520d93d8a (8656)

=== Bonsai-4B Q1_0 ===
Model: /home/bcloud/models/bonsai/Bonsai-4B.gguf

ggml_cuda_init: found 1 ROCm devices (Total VRAM: 63967 MiB):
  Device 0: Radeon 8060S Graphics, gfx1151 (0x1151), VMM: no, Wave Size: 32, VRAM: 63967 MiB
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen3 4B Q1_0_g128             | 540.09 MiB |     4.02 B | ROCm       |  99 |           pp512 |       1644.03 ± 6.25 |
| qwen3 4B Q1_0_g128             | 540.09 MiB |     4.02 B | ROCm       |  99 |           tg128 |        114.89 ± 9.64 |

build: 520d93d8a (8656)

=== Bonsai-8B Q1_0 ===
Model: /home/bcloud/models/bonsai/Bonsai-8B.gguf

ggml_cuda_init: found 1 ROCm devices (Total VRAM: 63967 MiB):
  Device 0: Radeon 8060S Graphics, gfx1151 (0x1151), VMM: no, Wave Size: 32, VRAM: 63967 MiB
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen3 8B Q1_0_g128             |   1.07 GiB |     8.19 B | ROCm       |  99 |           pp512 |        695.57 ± 3.53 |
| qwen3 8B Q1_0_g128             |   1.07 GiB |     8.19 B | ROCm       |  99 |           tg128 |         95.45 ± 0.97 |

build: 520d93d8a (8656)

=== BitNet-2B-4T Q1_0 ===
Model: /home/bcloud/models/bitnet-2b-q1_0.gguf

ggml_cuda_init: found 1 ROCm devices (Total VRAM: 63967 MiB):
  Device 0: Radeon 8060S Graphics, gfx1151 (0x1151), VMM: no, Wave Size: 32, VRAM: 63967 MiB
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| bitnet ?B Q1_0_g128            | 538.03 MiB |     2.41 B | ROCm       |  99 |           pp512 |      3188.95 ± 12.79 |
| bitnet ?B Q1_0_g128            | 538.03 MiB |     2.41 B | ROCm       |  99 |           tg128 |        118.86 ± 0.74 |

build: 520d93d8a (8656)

=== BitNet-2B-4T TQ1_0 ===
Model: /home/bcloud/models/bitnet-2b-tq1_0.gguf

ggml_cuda_init: found 1 ROCm devices (Total VRAM: 63967 MiB):
  Device 0: Radeon 8060S Graphics, gfx1151 (0x1151), VMM: no, Wave Size: 32, VRAM: 63967 MiB
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| bitnet ?B TQ1_0 - 1.69 bpw ternary |   1.02 GiB |     2.41 B | ROCm       |  99 |           pp512 |        283.52 ± 2.21 |
| bitnet ?B TQ1_0 - 1.69 bpw ternary |   1.02 GiB |     2.41 B | ROCm       |  99 |           tg128 |         49.75 ± 0.21 |

build: 520d93d8a (8656)

=== Qwen3-Coder-Next 80B TQ1_0 ===
Model: /home/bcloud/models/Qwen3-Coder-Next-UD-TQ1_0.gguf

ggml_cuda_init: found 1 ROCm devices (Total VRAM: 63967 MiB):
  Device 0: Radeon 8060S Graphics, gfx1151 (0x1151), VMM: no, Wave Size: 32, VRAM: 63967 MiB
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen3next 80B.A3B IQ1_S - 1.5625 bpw |  17.64 GiB |    79.67 B | ROCm       |  99 |           pp512 |        692.66 ± 3.64 |
| qwen3next 80B.A3B IQ1_S - 1.5625 bpw |  17.64 GiB |    79.67 B | ROCm       |  99 |           tg128 |         51.28 ± 0.04 |

build: 520d93d8a (8656)

=== Llama-4-Scout 108B TQ1_0 ===
Model: /home/bcloud/models/Llama-4-Scout-17B-16E-Instruct-UD-TQ1_0.gguf

ggml_cuda_init: found 1 ROCm devices (Total VRAM: 63967 MiB):
  Device 0: Radeon 8060S Graphics, gfx1151 (0x1151), VMM: no, Wave Size: 32, VRAM: 63967 MiB
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| llama4 17Bx16E (Scout) IQ1_S - 1.5625 bpw |  27.24 GiB |   107.77 B | ROCm       |  99 |           pp512 |        322.10 ± 0.61 |
| llama4 17Bx16E (Scout) IQ1_S - 1.5625 bpw |  27.24 GiB |   107.77 B | ROCm       |  99 |           tg128 |         20.95 ± 0.00 |

build: 520d93d8a (8656)


=== BURN COMPLETE === Thu Apr 16 08:52:43 PM ADT 2026
