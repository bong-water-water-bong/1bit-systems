#!/usr/bin/env bash
# 1bit systems XRT cmake patch — replaces CachyOS's broken xrt-targets.cmake
# with a minimal version declaring only the 3 shared libs that actually
# ship. Runs automatically as a pacman PostTransaction hook whenever
# /extra/xrt or /cachyos-extra-znver4/xrt is installed or upgraded.
#
# Upstream bug: CachyOS xrt PKGBUILD ships a cmake config that declares
# XRT::*_static, XRT::xilinxopencl, XRT::pyxrt targets whose libraries
# aren't in the package. cmake's find_package(XRT) then hard-errors
# because the imported targets reference nonexistent files. Downstream
# projects (mlir-aie, hsa-runtime, our 1bit-xdna when used with
# find_package) break. This patch is the temporary fix until the
# CachyOS package is corrected upstream.

set -euo pipefail

TARGET=/usr/share/cmake/XRT/xrt-targets.cmake
TARGET_NONE=/usr/share/cmake/XRT/xrt-targets-none.cmake

if [[ ! -f "$TARGET" ]]; then
    echo "[halo-xrt-cmake-patch] $TARGET missing — xrt package not installed?" >&2
    exit 0
fi

cat > "$TARGET" <<'CMAKE'
# 1bit systems patched xrt-targets.cmake — CachyOS xrt ships a broken file
# referencing static + opencl + pyxrt targets whose libraries aren't in
# the package. Replaced by /usr/local/sbin/1bit-xrt-cmake-patch.sh
# (pacman PostTransaction hook 99-halo-xrt-cmake.hook). See memory
# project_strix_halo_hardware.md for the upstream bug description.

cmake_policy(PUSH)
cmake_policy(VERSION 3.10...4.1)

set(_IMPORT_PREFIX "/usr")

if(NOT TARGET XRT::xrt_coreutil)
    add_library(XRT::xrt_coreutil SHARED IMPORTED)
    set_target_properties(XRT::xrt_coreutil PROPERTIES
        IMPORTED_LOCATION "${_IMPORT_PREFIX}/lib/libxrt_coreutil.so.2"
        INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include/xrt"
        INTERFACE_LINK_LIBRARIES "uuid")
endif()

if(NOT TARGET XRT::xrt_core)
    add_library(XRT::xrt_core SHARED IMPORTED)
    set_target_properties(XRT::xrt_core PROPERTIES
        IMPORTED_LOCATION "${_IMPORT_PREFIX}/lib/libxrt_core.so.2"
        INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include/xrt"
        INTERFACE_LINK_LIBRARIES "XRT::xrt_coreutil;uuid;dl;rt;pthread")
endif()

if(NOT TARGET XRT::xrt++)
    add_library(XRT::xrt++ SHARED IMPORTED)
    set_target_properties(XRT::xrt++ PROPERTIES
        IMPORTED_LOCATION "${_IMPORT_PREFIX}/lib/libxrt++.so.2"
        INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include/xrt"
        INTERFACE_LINK_LIBRARIES "XRT::xrt_core;XRT::xrt_coreutil")
endif()

cmake_policy(POP)
CMAKE

cat > "$TARGET_NONE" <<'CMAKE'
# 1bit systems patched — no per-configuration overrides for shared-only set.
CMAKE

echo "[halo-xrt-cmake-patch] repatched $TARGET"
