cmake_minimum_required(VERSION 3.21)

# Pin HIP compiler to TheRock's clang for native gfx1151.
set(CMAKE_HIP_COMPILER "$ENV{HOME}/therock/build/dist/rocm/lib/llvm/bin/clang++"
    CACHE FILEPATH "HIP compiler")
set(CMAKE_C_COMPILER   "$ENV{HOME}/therock/build/dist/rocm/lib/llvm/bin/clang"
    CACHE FILEPATH "C compiler")
set(CMAKE_CXX_COMPILER "$ENV{HOME}/therock/build/dist/rocm/lib/llvm/bin/clang++"
    CACHE FILEPATH "C++ compiler")

project(rocm_cpp LANGUAGES C CXX HIP)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# CK's host-tensor reference code uses 1D access into 2D tensors — only valid
# when assertions are disabled. Match upstream Release build.
add_compile_definitions(NDEBUG)

# TheRock ROCm dist (CK, HIP, rocBLAS)
set(ROCM_ROOT "$ENV{HOME}/therock/build/dist/rocm")
list(PREPEND CMAKE_PREFIX_PATH "${ROCM_ROOT}" "${ROCM_ROOT}/lib/cmake")

find_package(hip REQUIRED CONFIG PATHS "${ROCM_ROOT}")

# CK is still needed for the device-side GEMM template (header-only). Option B
# phases 1-4 will replace that; once they land, this find_package goes away too.
find_package(composable_kernel REQUIRED CONFIG PATHS "${ROCM_ROOT}")

set(CMAKE_HIP_ARCHITECTURES "gfx1151")

# -- librocm_cpp : shared library exposing the C API ----------------------------
# Host packer is pure C++ (no CK). Device kernel uses CK headers only — no
# libutility.a link. Once we vendor the device kernel (Phase 1+), this target
# has no CK dependency at all.
add_library(rocm_cpp SHARED src/ck_gemm.cpp src/prefill_standalone.hip src/wmma_peak_probe.hip)
set_source_files_properties(src/ck_gemm.cpp src/prefill_standalone.hip src/wmma_peak_probe.hip PROPERTIES LANGUAGE HIP)
target_include_directories(rocm_cpp PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_include_directories(rocm_cpp SYSTEM PRIVATE "${ROCM_ROOT}/include")
target_compile_options(rocm_cpp PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3 -ffast-math -munsafe-fp-atomics
        -Wno-unused-result -Wno-unused-parameter -Wno-unused-variable>)
target_link_libraries(rocm_cpp PRIVATE hip::host hip::device)
set_target_properties(rocm_cpp PROPERTIES HIP_ARCHITECTURES "gfx1151")

# -- tests/test_ck_gemm : end-to-end functional proof --------------------------
add_executable(test_ck_gemm tests/test_ck_gemm.cpp)
set_source_files_properties(tests/test_ck_gemm.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(test_ck_gemm PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(test_ck_gemm PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(test_ck_gemm PROPERTIES HIP_ARCHITECTURES "gfx1151")

# -- tests/test_standalone : correctness diff CK vs standalone kernel ----------
add_executable(test_standalone tests/test_standalone.cpp)
set_source_files_properties(tests/test_standalone.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(test_standalone PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(test_standalone PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(test_standalone PROPERTIES HIP_ARCHITECTURES "gfx1151")

# -- ck-prefill : research binaries (kept building for tuning) ------------------
add_subdirectory(ck-prefill)
