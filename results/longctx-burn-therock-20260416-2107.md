# Long-context burn against TheRock 7.13 — pp2048 + tg512
Date: Thu Apr 16 09:07:30 PM ADT 2026
Binary: PrismML-Eng prism e2d6742, GGML_HIP=ON

=== Bonsai-1.7B Q1_0 ===
ggml_cuda_init: found 1 ROCm devices (Total VRAM: 63967 MiB):
  Device 0: Radeon 8060S Graphics, gfx1151 (0x1151), VMM: no, Wave Size: 32, VRAM: 63967 MiB
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen3 1.7B Q1_0                | 231.13 MiB |     1.72 B | ROCm       |  99 |           pp512 |      4147.18 ± 35.86 |
| qwen3 1.7B Q1_0                | 231.13 MiB |     1.72 B | ROCm       |  99 |          pp2048 |       3061.90 ± 1.70 |
| qwen3 1.7B Q1_0                | 231.13 MiB |     1.72 B | ROCm       |  99 |           tg128 |        231.97 ± 0.55 |
| qwen3 1.7B Q1_0                | 231.13 MiB |     1.72 B | ROCm       |  99 |           tg512 |        202.62 ± 1.09 |

build: e2d6742 (1)

=== Bonsai-4B Q1_0 ===
ggml_cuda_init: found 1 ROCm devices (Total VRAM: 63967 MiB):
  Device 0: Radeon 8060S Graphics, gfx1151 (0x1151), VMM: no, Wave Size: 32, VRAM: 63967 MiB
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen3 4B Q1_0                  | 540.09 MiB |     4.02 B | ROCm       |  99 |           pp512 |      2012.20 ± 10.88 |
| qwen3 4B Q1_0                  | 540.09 MiB |     4.02 B | ROCm       |  99 |          pp2048 |       1415.65 ± 9.91 |
| qwen3 4B Q1_0                  | 540.09 MiB |     4.02 B | ROCm       |  99 |           tg128 |        123.37 ± 0.25 |
| qwen3 4B Q1_0                  | 540.09 MiB |     4.02 B | ROCm       |  99 |           tg512 |        112.41 ± 0.03 |

build: e2d6742 (1)

=== Bonsai-8B Q1_0 ===
ggml_cuda_init: found 1 ROCm devices (Total VRAM: 63967 MiB):
  Device 0: Radeon 8060S Graphics, gfx1151 (0x1151), VMM: no, Wave Size: 32, VRAM: 63967 MiB
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen3 8B Q1_0                  |   1.07 GiB |     8.19 B | ROCm       |  99 |           pp512 |       1272.52 ± 4.69 |
| qwen3 8B Q1_0                  |   1.07 GiB |     8.19 B | ROCm       |  99 |          pp2048 |       1008.03 ± 3.53 |
| qwen3 8B Q1_0                  |   1.07 GiB |     8.19 B | ROCm       |  99 |           tg128 |         93.92 ± 0.08 |
| qwen3 8B Q1_0                  |   1.07 GiB |     8.19 B | ROCm       |  99 |           tg512 |         87.75 ± 0.02 |

build: e2d6742 (1)

=== BitNet-2B-4T Q1_0 ===
ggml_cuda_init: found 1 ROCm devices (Total VRAM: 63967 MiB):
  Device 0: Radeon 8060S Graphics, gfx1151 (0x1151), VMM: no, Wave Size: 32, VRAM: 63967 MiB
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| bitnet ?B Q1_0                 | 538.03 MiB |     2.41 B | ROCm       |  99 |           pp512 |      3036.56 ± 17.19 |
| bitnet ?B Q1_0                 | 538.03 MiB |     2.41 B | ROCm       |  99 |          pp2048 |      2685.55 ± 13.55 |
| bitnet ?B Q1_0                 | 538.03 MiB |     2.41 B | ROCm       |  99 |           tg128 |        116.78 ± 6.48 |
| bitnet ?B Q1_0                 | 538.03 MiB |     2.41 B | ROCm       |  99 |           tg512 |        108.54 ± 0.32 |

build: e2d6742 (1)

=== BURN COMPLETE === Thu Apr 16 09:09:03 PM ADT 2026
