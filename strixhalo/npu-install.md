# strixhalo/npu-install.md — Peano (llvm-aie) toolchain

Standalone AIE compiler for the Strix Halo XDNA2 NPU. Pure C++ toolchain:
no IRON, no Python runtime, no MLIR-AIE host stack. Produces AIE2P ELF
objects from straight C++ via `--target=aie2p-none-unknown-elf`.

Upstream: https://github.com/Xilinx/llvm-aie (195 stars, active 2026-04).

## Prereqs

Arch/CachyOS:

```bash
sudo pacman -S --needed cmake ninja git clang lld llvm python
```

- `clang`/`llvm` host-side — bootstraps the cross compiler.
- `python` is **configure-time only** (CMake invokes it for a few LLVM
  helper scripts during the configure step). No Python ends up in the
  installed `/opt/peano/` tree; Rule A still holds.
- `ccache` is optional; cache preset sets `LLVM_CCACHE_BUILD=ON` but we
  disable it via `-DLLVM_CCACHE_BUILD=OFF` below to keep the dep list
  small.

XRT (already present on strixhalo, but noted for greenfield boxes): `xrt`
+ `amdxdna` driver from AUR. Peano itself has no XRT dep — you only need
XRT to eventually *run* the ELF, not to build it.

## Clone

Monorepo, no submodules:

```bash
git clone https://github.com/Xilinx/llvm-aie.git /home/bcloud/repos/llvm-aie
```

## Configure

The canonical cache file lives at
`clang/cmake/caches/Peano-AIE.cmake` and is the single source of truth
for what a Peano distribution bundles — `llc`, `clang`, `lld`,
`aie2p-resource-headers`, and cross-compiled runtimes for each AIE target
(`aie-none-unknown-elf`, `aie2-`, `aie2p-`, `aie2ps-`). We inherit it
rather than re-deriving the flags.

```bash
cmake -S /home/bcloud/repos/llvm-aie/llvm \
      -B /home/bcloud/repos/llvm-aie/build \
      -G Ninja \
      -C /home/bcloud/repos/llvm-aie/clang/cmake/caches/Peano-AIE.cmake \
      -DCMAKE_INSTALL_PREFIX=/opt/peano \
      -DLLVM_CCACHE_BUILD=OFF
```

Key bits the cache file sets for us (do NOT override these):

- `LLVM_EXPERIMENTAL_TARGETS_TO_BUILD=AIE` — AIE is an experimental
  backend, so it lives under the experimental knob, not
  `LLVM_TARGETS_TO_BUILD`.
- `LLVM_TARGETS_TO_BUILD=host` — we still need the host target for the
  driver and for TableGen.
- `LLVM_ENABLE_PROJECTS=clang;clang-tools-extra;lld`.
- `LLVM_ENABLE_RUNTIMES=compiler-rt;libc;libcxx;libcxxabi` cross-built
  for every AIE triple (four of them).
- `LLVM_INSTALL_TOOLCHAIN_ONLY=ON` — keeps the install tree lean.

## Build

```bash
ninja -C /home/bcloud/repos/llvm-aie/build -j$(nproc)
```

Expected wall time on strixhalo (32 cores, 128 GB RAM, NVMe): **30-60
min** first build. Peak RAM ~15 GB during parallel link; nowhere near
OOM. Disk footprint: `build/` ≈ 25 GB, install ≈ 1.5 GB.

## Install

```bash
sudo ninja -C /home/bcloud/repos/llvm-aie/build install-distribution
```

Use the `install-distribution` target (not plain `install`) — the cache
file pins a `LLVM_DISTRIBUTION_COMPONENTS` list to avoid dragging in
host-side debug tools we don't need.

## Verify

```bash
/opt/peano/bin/clang --version
/opt/peano/bin/llc --version        # 'AIE' listed under Registered Targets
/opt/peano/bin/llc -mtriple=aie2p-none-unknown-elf -mattr=help 2>&1 | head
```

Hello-tile smoke test (source at
`halo-workspace/rocm-cpp/examples/aie_hello.cpp`):

```bash
/opt/peano/bin/clang --target=aie2p-none-unknown-elf \
    -O2 -c /home/bcloud/repos/halo-workspace/rocm-cpp/examples/aie_hello.cpp \
    -o /tmp/aie_hello.o
/opt/peano/bin/llvm-readelf -h /tmp/aie_hello.o | grep Machine
# expect: Machine: Xilinx AI Engine
```

## Next step (NOT in this runbook)

To actually schedule the ELF onto a tile and run it you need either
IRON/MLIR-AIE or a hand-rolled xrt_coreutil host wrapper. Both are
out-of-scope for this runbook — Peano alone only gets you to the object
file. See `project_npu_path_analysis.md` in memory: NPU is deferred
until a STX-H Linux stack lands, Q3 2026 at earliest.

## Uninstall / rebuild

```bash
sudo rm -rf /opt/peano
rm -rf /home/bcloud/repos/llvm-aie/build
```
