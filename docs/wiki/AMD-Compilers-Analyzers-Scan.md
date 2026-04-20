# AMD Compilers, Analyzers, Debuggers — 2026-04-20 scan

Lens: what improves halo-ai's AIE kernel authoring, HIP perf, or ROCm profiling beyond our current stack (gcc 14, clang 20, hipcc, Peano/llvm-aie, rocprof, rocgdb, bindgen). Landing page: https://www.amd.com/en/developer/browse-by-resource-type/software-tools.html (fetch blocked; table from GPUOpen + ROCm docs).

## Catalog

| Tool | What it does | Linux today? | License | Fits our stack? |
|---|---|---|---|---|
| AOCC | Zen-tuned Clang/Flang (LLVM 17), no GPU offload | Yes | AMD EULA | Overlaps clang 20; Zen5 wins marginal vs CachyOS `znver4`. **Skip.** |
| AOMP | LLVM+OpenMP GPU offload staging | Yes | Apache 2.0 | hipcc covers us. **Defer.** |
| hipcc / ROCm LLVM | HIP + amdclang++ | Yes | NCSA/MIT | **Using.** |
| ROCgdb | GDB with AMDGPU wavefront debug | Yes | GPL | **Using.** |
| rocprof / rocprofv2 | HW counters, kernel traces (EOL 2026-Q2) | Yes | MIT | **Using.** Migrate Q2. |
| rocprofiler-sdk / `rocprofv3` | Successor C++ API | Yes | MIT | **Adopt on next bump.** |
| ROCm Compute Profiler (`rocprof-compute`, ex-Omniperf) | Roofline, L1/L2/LDS counters, Grafana UI | Yes (ROCm ≥6.2) | MIT | **Adopt.** Serves Gerganov L1 lead + Sherry bytes-read validation. |
| ROCm Systems Profiler (ex-Omnitrace) | Whole-app timeline, GPU+CPU sampling | Yes | MIT | **Adopt for voice lane.** |
| Radeon GPU Analyzer (RGA) | Offline compiler + ISA inspector (Vulkan/DX/GL/**OpenCL**), VGPR/SGPR/occupancy | Yes (CLI + VS Code ext) | MIT | **gfx1151 supported** (2.12+ adds gfx1150/1151/1152/1201). No native HIP input, but OpenCL+GCN ISA dump fills a real gap vs rocprof. **Adopt.** |
| Radeon GPU Profiler (RGP) | Vulkan/DX12 frame profiler | Yes | MIT | Graphics only. **Skip.** |
| Radeon Raytracing Analyzer (RRA) | BVH inspector | Yes | MIT | **Skip.** |
| Radeon Memory Visualizer (RMV) | VRAM residency timeline | Yes (AMDVLK only) | MIT | 25.20 RADV-default breaks capture. **Skip for now.** |
| AMD uProf | Zen CPU profiler: IBS, branch mispredict, LLC miss, AVX-512 retire; also MI metrics | **Yes, AUR `amduprof`** | AMD EULA | **Adopt for CPU offload lane.** Only tool with clean Zen5 PMC coverage on Linux. |
| GPU PerfStudio / RCP | Deprecated | — | — | **Skip.** |
| llvm-aie `llvm-mc`, `llvm-objdump` | **AIE2/AIE2P assembler + disassembler** in Peano fork | Yes, at `/opt/peano/bin` | Apache-2.0 | Already installed; just not invoking. `llvm-objdump -d --triple=aie2p-none-unknown-elf` disassembles AIE2P ELFs. **Use for AIE debug.** |
| mlir-aie / IRON | MLIR AIE dialect + Python IRON front-end | Yes | Apache-2.0 | Authoring-time Python OK (Rule A covers runtime). **Defer until first AIE2P kernel.** |

## Special-focus answers

- **AIE assembler/disassembler?** YES. `/opt/peano/bin/llvm-mc` + `/opt/peano/bin/llvm-objdump`, triple `aie2p-none-unknown-elf`. https://github.com/Xilinx/llvm-aie (issue #352 confirms AIE2PInstPrinter). No separate AMD-branded assembler exists.
- **Zen5 profiler on Arch/CachyOS?** YES. AMD uProf via AUR `amduprof`; closed tarball, user-mode driver loads on Arch kernels. IBS + branch-mispredict + LLC + AVX-512 retire counters.
- **RGA on gfx1151?** YES. RGA 2.12+ adds gfx1150/1151/1152/1201 across all compile modes. OpenCL offline path works without a runtime launch. Caveat: no direct `.hip` ingest — feed extracted GCN asm (`hipcc --save-temps`) or OpenCL.

## Adopt order (hot-path impact)

1. **RGA** — offline gfx1151 ISA / occupancy / register-pressure on the ternary GEMV (rocprof doesn't show this).
2. **rocprof-compute** — roofline validates "92% of LPDDR5 peak" and Sherry's projected bytes-read drop.
3. **AMD uProf** — CPU offload lane, only clean Zen5 PMC coverage on Linux.
4. **llvm-objdump --triple=aie2p** — free, already on disk.
5. **rocprofiler-sdk** — migrate before rocprof EOL 2026-Q2.

Skips: AOCC, AOMP (overlap); RGP/RRA/RMV (wrong workload); GPU PerfStudio, RCP (deprecated).

## URLs

- Landing: https://www.amd.com/en/developer/browse-by-resource-type/software-tools.html
- AOCC: https://www.amd.com/en/developer/aocc.html
- uProf: https://www.amd.com/en/developer/uprof.html
- RGA: https://gpuopen.com/rga/ · releases: https://github.com/GPUOpen-Tools/radeon_gpu_analyzer/releases
- RGP/RRA/RMV: https://gpuopen.com/rgp/ · https://gpuopen.com/radeon-raytracing-analyzer/ · https://github.com/GPUOpen-Tools/radeon_memory_visualizer
- rocprof-compute: https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/
- rocprofiler-sdk: https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/
- llvm-aie (Peano): https://github.com/Xilinx/llvm-aie
- mlir-aie: https://github.com/Xilinx/mlir-aie
