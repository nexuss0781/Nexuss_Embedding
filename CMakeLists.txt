# =============================================================================
# CMakeLists.txt — HFAQE Component 1.2 Token Embedding
# =============================================================================
# Targets
#   hfaqe_main   — orchestrator: steps 1-5 (Core/Input/Train/Metrics/Output)
#   hfaqe_test   — full §5 validation test suite
#
# Options
#   -DHFAQE_AVX512=ON    enable AVX-512BW+F microkernels (default: auto-detect)
#   -DHFAQE_ASAN=ON      enable AddressSanitizer (debug builds)
#
# Usage (Linux)
#   cmake -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j$(nproc)
#   ./build/hfaqe_main
#   ./build/hfaqe_test
# =============================================================================

cmake_minimum_required(VERSION 3.16)
project(HFAQE VERSION 1.0.0 LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Language standard
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS        OFF)

# ---------------------------------------------------------------------------
# Default build type
# ---------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler   : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(HFAQE_AVX512 "Enable AVX-512 BW+F microkernels" OFF)
option(HFAQE_ASAN   "Enable AddressSanitizer"           OFF)

# Auto-detect AVX-512 support on the build host if not explicitly set
if(NOT HFAQE_AVX512)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512F)
    check_cxx_compiler_flag("-mavx512bw" COMPILER_SUPPORTS_AVX512BW)
    if(COMPILER_SUPPORTS_AVX512F AND COMPILER_SUPPORTS_AVX512BW)
        # Only auto-enable if running on an AVX-512 capable CPU
        if(EXISTS "/proc/cpuinfo")
            execute_process(
                COMMAND grep -c avx512f /proc/cpuinfo
                OUTPUT_VARIABLE _AVX512_COUNT
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET)
            if(_AVX512_COUNT GREATER 0)
                set(HFAQE_AVX512 ON)
                message(STATUS "Auto-detected AVX-512 capable CPU — enabling SIMD kernels")
            endif()
        endif()
    endif()
endif()

# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------
set(HFAQE_WARN_FLAGS
    -Wall -Wextra -Wpedantic
    -Wno-unused-variable
    -Wno-unused-parameter
    -Wno-missing-field-initializers
)

set(HFAQE_OPT_FLAGS_RELEASE
    -O3
    -funroll-loops
    -ffast-math
    -march=native          # use native ISA (fallback when AVX-512 not explicit)
)

set(HFAQE_OPT_FLAGS_DEBUG
    -O0 -g3
)

set(HFAQE_SIMD_FLAGS "")
if(HFAQE_AVX512)
    list(APPEND HFAQE_SIMD_FLAGS -mavx512f -mavx512bw)
    message(STATUS "SIMD       : AVX-512 BW+F ENABLED")
else()
    message(STATUS "SIMD       : Scalar fallback (AVX-512 not enabled)")
endif()

set(HFAQE_ASAN_FLAGS "")
if(HFAQE_ASAN)
    list(APPEND HFAQE_ASAN_FLAGS -fsanitize=address -fno-omit-frame-pointer)
    message(STATUS "ASan       : ENABLED")
endif()

# ---------------------------------------------------------------------------
# Compile definitions
# ---------------------------------------------------------------------------
set(HFAQE_DEFS "")
if(HFAQE_AVX512)
    # Force HFAQE_AVX512=1 even when __AVX512F__/__AVX512BW__ might not be
    # detected by the preprocessor (e.g., cross-compile scenarios)
    list(APPEND HFAQE_DEFS HFAQE_FORCE_AVX512=1)
endif()

# ---------------------------------------------------------------------------
# Helper: apply common settings to a target
# ---------------------------------------------------------------------------
function(hfaqe_configure_target target)
    target_compile_options(${target} PRIVATE
        ${HFAQE_WARN_FLAGS}
        ${HFAQE_SIMD_FLAGS}
        ${HFAQE_ASAN_FLAGS}
        $<$<CONFIG:Release>:${HFAQE_OPT_FLAGS_RELEASE}>
        $<$<CONFIG:Debug>:${HFAQE_OPT_FLAGS_DEBUG}>
    )
    target_compile_definitions(${target} PRIVATE ${HFAQE_DEFS})
    if(HFAQE_ASAN)
        target_link_options(${target} PRIVATE ${HFAQE_ASAN_FLAGS})
    endif()
    # Link math library (required for std::sqrt etc. on some Linux configs)
    target_link_libraries(${target} PRIVATE m)
endfunction()

# ---------------------------------------------------------------------------
# Source directory (all .cpp files live alongside this CMakeLists.txt)
# ---------------------------------------------------------------------------
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# ---------------------------------------------------------------------------
# Target 1: hfaqe_main
# Orchestrator — runs Core → Input → Train → Metrics → Output in order.
# Single TU: main.cpp includes all other .cpp files via include-guards.
# ---------------------------------------------------------------------------
add_executable(hfaqe_main ${SRC_DIR}/main.cpp)
hfaqe_configure_target(hfaqe_main)
# Include current source dir so #include "Core.cpp" etc. resolve
target_include_directories(hfaqe_main PRIVATE ${SRC_DIR})

# ---------------------------------------------------------------------------
# Target 2: hfaqe_test
# Full §5 validation suite.
# Single TU: Test.cpp includes Core.cpp via include-guard.
# ---------------------------------------------------------------------------
add_executable(hfaqe_test ${SRC_DIR}/Test.cpp)
hfaqe_configure_target(hfaqe_test)
target_include_directories(hfaqe_test PRIVATE ${SRC_DIR})

# ---------------------------------------------------------------------------
# CTest integration — register both targets as tests
# ---------------------------------------------------------------------------
enable_testing()

add_test(
    NAME  HFAQE_Main_Steps
    COMMAND hfaqe_main
)
set_tests_properties(HFAQE_Main_Steps PROPERTIES
    TIMEOUT 300
    PASS_REGULAR_EXPRESSION "ALL STEPS PASSED"
    FAIL_REGULAR_EXPRESSION  "STEP.*FAIL"
)

add_test(
    NAME  HFAQE_Validation_Suite
    COMMAND hfaqe_test
)
set_tests_properties(HFAQE_Validation_Suite PROPERTIES
    TIMEOUT 600
    PASS_REGULAR_EXPRESSION "0 FAILED"
    FAIL_REGULAR_EXPRESSION  "[1-9][0-9]* FAILED"
)

# ---------------------------------------------------------------------------
# Install targets (optional — for downstream CMake projects)
# ---------------------------------------------------------------------------
install(TARGETS hfaqe_main hfaqe_test
    RUNTIME DESTINATION bin
)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
message(STATUS "----------------------------------------------------")
message(STATUS "Targets    : hfaqe_main  hfaqe_test")
message(STATUS "Install to : ${CMAKE_INSTALL_PREFIX}/bin")
message(STATUS "Run tests  : ctest --test-dir build --output-on-failure")
message(STATUS "----------------------------------------------------")
