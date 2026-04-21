# cpp/CMakeLists.txt — developer-side build for the shim's C++ unit tests.
#
# The crate's primary build is cargo + build.rs (`cc` crate compiles
# shim.cpp, links libwhisper). CMake is NOT on the main build path — it
# exists here only to produce the segment-ring unit-test binary, which
# has no libwhisper dependency and can run on any host.
#
# To keep the default `cargo build` green on boxes without CMake, nothing
# in this file is invoked during the crate build. Invoke manually:
#
#     cd crates/1bit-whisper/cpp
#     cmake -S . -B build -DONEBIT_WHISPER_BUILD_SHIM_TESTS=ON
#     cmake --build build
#     ./build/test_shim_streaming

cmake_minimum_required(VERSION 3.20)
project(onebit_whisper_shim_tests CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ONEBIT_WHISPER_BUILD_SHIM_TESTS
       "Build the cpp/ segment-ring unit tests" OFF)

if(ONEBIT_WHISPER_BUILD_SHIM_TESTS)
    add_executable(test_shim_streaming tests/test_shim_streaming.cpp)
    target_compile_options(test_shim_streaming PRIVATE
        -Wall -Wextra -Wpedantic -Werror
    )
    target_include_directories(test_shim_streaming PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    # pthread for std::thread/std::mutex on GCC + libstdc++.
    find_package(Threads REQUIRED)
    target_link_libraries(test_shim_streaming PRIVATE Threads::Threads)

    enable_testing()
    add_test(NAME shim_streaming COMMAND test_shim_streaming)
endif()
