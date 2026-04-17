# Full 1-bit burn against TheRock 7.13 native gfx1151
Date: Thu Apr 16 08:31:21 PM ADT 2026
Stack: PrismML prism branch + TheRock 7.13
rocBLAS lib: /home/bcloud/therock/build/dist/rocm/lib/librocblas.so.5.4
Tensile libpath: /home/bcloud/therock/build/dist/rocm/lib/rocblas/library

=== Bonsai-1.7B Q1_0 ===
Model: /home/bcloud/models/bonsai/Bonsai-1.7B.gguf

| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen3 1.7B Q1_0_g128           | 231.13 MiB |     1.72 B | CPU        |      16 |           pp512 |         18.03 ± 0.07 |
| qwen3 1.7B Q1_0_g128           | 231.13 MiB |     1.72 B | CPU        |      16 |           tg128 |         13.68 ± 0.03 |

build: 520d93d8a (8656)

=== Bonsai-4B Q1_0 ===
Model: /home/bcloud/models/bonsai/Bonsai-4B.gguf

| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen3 4B Q1_0_g128             | 540.09 MiB |     4.02 B | CPU        |      16 |           pp512 |          6.83 ± 0.03 |
| qwen3 4B Q1_0_g128             | 540.09 MiB |     4.02 B | CPU        |      16 |           tg128 |          5.70 ± 0.07 |

build: 520d93d8a (8656)

=== Bonsai-8B Q1_0 ===
Model: /home/bcloud/models/bonsai/Bonsai-8B.gguf

