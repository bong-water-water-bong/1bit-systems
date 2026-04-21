cmake_minimum_required(VERSION 3.21)

# Standalone CPU lane — build separately from the HIP parent, since the
# parent CMakeLists pins TheRock's clang which has no host CPU libraries
# pre-configured for us. Run from this directory:
#     cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#     cmake --build build -j
#
# If later folded into the parent build, wrap the add_library + add_executable
# below behind an `if(HALO_BUILD_CPU_LANE)` guard and skip the compiler pins.
project(halo_cpu_avx2 LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenMP REQUIRED)

# AVX2 + FMA + F16C. No AVX-512 (widen user base; Zen5 has it but we hold that
# back for a separate lane). -mtune=znver5 picks up Zen5 scheduling hints on
# modern GCC/clang; if tune is unavailable, clang and GCC both no-op silently.
set(HALO_CPU_AVX2_FLAGS
    -O3
    -mavx2
    -mfma
    -mf16c
    -mbmi2
    -mtune=znver5
    -fno-math-errno
    -ffp-contract=fast
    -Wall
    -Wextra
    -Wshadow
    -Wconversion
    -Wno-sign-conversion
    -Wno-unused-function)

add_library(cpu_avx2_ternary STATIC
    src/ternary_gemv_tq2_avx2.cpp
    src/ternary_gemv_scalar_ref.cpp)
target_include_directories(cpu_avx2_ternary PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_options(cpu_avx2_ternary PRIVATE ${HALO_CPU_AVX2_FLAGS})
target_link_libraries(cpu_avx2_ternary PUBLIC OpenMP::OpenMP_CXX)

# Differential test + benchmark harness. Single binary, `--bench` toggles
# microbench mode. Exit 0 on success, 1 on parity failure.
add_executable(test_cpu_ternary_gemv tests/test_cpu_ternary_gemv.cpp)
target_compile_options(test_cpu_ternary_gemv PRIVATE ${HALO_CPU_AVX2_FLAGS})
target_link_libraries(test_cpu_ternary_gemv PRIVATE
    cpu_avx2_ternary
    OpenMP::OpenMP_CXX)
