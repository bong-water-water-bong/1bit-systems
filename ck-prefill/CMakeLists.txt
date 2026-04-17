cmake_minimum_required(VERSION 3.21)

# Use TheRock's HIP compiler for native gfx1151
set(CMAKE_HIP_COMPILER "$ENV{HOME}/therock/build/dist/rocm/lib/llvm/bin/clang++"
    CACHE FILEPATH "HIP compiler")
set(CMAKE_C_COMPILER "$ENV{HOME}/therock/build/dist/rocm/lib/llvm/bin/clang"
    CACHE FILEPATH "C compiler")
set(CMAKE_CXX_COMPILER "$ENV{HOME}/therock/build/dist/rocm/lib/llvm/bin/clang++"
    CACHE FILEPATH "C++ compiler")

project(ck_prefill LANGUAGES CXX HIP)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Upstream CK examples ship reference code that assumes NDEBUG (1D access into
# 2D host tensor via GetOffsetFromMultiIndex with missing index args works when
# assertions are disabled). Match that.
add_compile_definitions(NDEBUG)

# Point at TheRock's ROCm + CK
set(ROCM_ROOT "$ENV{HOME}/therock/build/dist/rocm")
list(PREPEND CMAKE_PREFIX_PATH "${ROCM_ROOT}" "${ROCM_ROOT}/lib/cmake")

find_package(hip REQUIRED CONFIG PATHS "${ROCM_ROOT}")
find_package(composable_kernel REQUIRED CONFIG COMPONENTS utility PATHS "${ROCM_ROOT}")

set(CMAKE_HIP_ARCHITECTURES "gfx1151")

function(add_ck_example name src)
    add_executable(${name} ${src})
    set_source_files_properties(${src} PROPERTIES LANGUAGE HIP)
    target_compile_options(${name} PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:
            -O3
            -ffast-math
            -munsafe-fp-atomics
            -Wno-unused-result
            -Wno-unused-parameter
            -Wno-unused-variable
        >)
    target_link_libraries(${name} PRIVATE
        composable_kernel::utility
        hip::host
        hip::device)
    set_target_properties(${name} PROPERTIES HIP_ARCHITECTURES "gfx1151")
endfunction()

add_ck_example(gemm_wmma_fp16_v3 gemm_wmma_fp16_v3.cpp)
add_ck_example(gemm_wmma_fp16_pk_i4_v3 gemm_wmma_fp16_pk_i4_v3.cpp)
