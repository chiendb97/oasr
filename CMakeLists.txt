cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(oasr LANGUAGES CXX CUDA)

# =============================================================================
# Options
# =============================================================================
option(BUILD_PYTHON "Build Python bindings" ON)
option(BUILD_TESTS "Build unit tests" OFF)  # Disabled for now - enable when ready
option(BUILD_EXAMPLES "Build examples" OFF)
option(USE_FLASH_ATTENTION "Use FlashAttention kernels" OFF)
option(ENABLE_FP16 "Enable FP16 support" ON)
option(ENABLE_BF16 "Enable BF16 support" ON)

# =============================================================================
# Global settings
# =============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# =============================================================================
# CUDA configuration
# =============================================================================
find_package(CUDAToolkit REQUIRED)

# Put -I flags in compile command (not in .rsp) so compile_commands.json and IntelliSense see them
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)

# Set CUDA architectures if not specified
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Support common architectures: Volta, Turing, Ampere, Ada, Hopper
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
endif()

# CUDA compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fPIC")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g -O0")
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas=-v")
endif()

# =============================================================================
# Dependencies
# =============================================================================

# cuBLAS and cuDNN
find_library(CUBLAS_LIB cublas HINTS ${CUDAToolkit_LIBRARY_DIR})
find_library(CUBLASLT_LIB cublasLt HINTS ${CUDAToolkit_LIBRARY_DIR})
find_library(CUDNN_LIB cudnn HINTS ${CUDAToolkit_LIBRARY_DIR} /usr/lib/x86_64-linux-gnu)

if(NOT CUDNN_LIB)
    message(WARNING "cuDNN not found. Some features may be disabled.")
endif()

# =============================================================================
# CUTLASS
# =============================================================================
set(CUTLASS_DIR "${CMAKE_SOURCE_DIR}/third_party/cutlass" CACHE PATH "CUTLASS directory (header-only)")
if(EXISTS ${CUTLASS_DIR}/include/cutlass/cutlass.h)
    message(STATUS "Using CUTLASS (header-only) from: ${CUTLASS_DIR}")
    add_compile_definitions(OASR_USE_CUTLASS)
else()
    message(STATUS "CUTLASS not found at ${CUTLASS_DIR}. Fetching headers from GitHub...")
    include(FetchContent)
    FetchContent_Declare(
        cutlass
        GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
        GIT_TAG v3.4.0
    )
    # Populate only: do not call FetchContent_MakeAvailable so we do not build CUTLASS library
    FetchContent_Populate(cutlass)
    set(CUTLASS_DIR ${cutlass_SOURCE_DIR} CACHE PATH "CUTLASS directory (header-only)" FORCE)
    add_compile_definitions(OASR_USE_CUTLASS)
endif()

# pybind11 for Python bindings
if(BUILD_PYTHON)
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    
    # Try to find pybind11 via pip or system
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE PYBIND11_FIND_RESULT
    )
    
    if(PYBIND11_FIND_RESULT EQUAL 0)
        list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_CMAKE_DIR})
        find_package(pybind11 CONFIG REQUIRED)
        message(STATUS "Found pybind11: ${pybind11_VERSION}")
    else()
        message(STATUS "pybind11 not found. Fetching from GitHub...")
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v2.11.1
        )
        FetchContent_MakeAvailable(pybind11)
    endif()
endif()

# =============================================================================
# Source files
# =============================================================================
set(OASR_SOURCES
    # Common (minimal for now)
    csrc/common/cuda_utils.cpp
    
    # Kernels
    csrc/kernels/normalization/norm_kernels.cu
    csrc/kernels/convolution/conv_kernels.cu
    
    # GEMM Kernels (CUTLASS-based)
    csrc/kernels/gemm/gemm_kernels.cu
    csrc/kernels/gemm/bmm_kernels.cu
    csrc/kernels/gemm/group_gemm_kernels.cu
    
    # TODO: Add these when ready
    # csrc/common/tensor.cpp
    # csrc/common/allocator.cpp
    # csrc/kernels/attention/attention_kernels.cu
    # csrc/layers/base_layer.cpp
    # csrc/layers/attention_layer.cpp
    # csrc/layers/conv_layer.cpp
    # csrc/layers/ffn_layer.cpp
    # csrc/layers/encoder_layer.cpp
)

set(OASR_HEADERS
    csrc/common/types.h
    csrc/common/cuda_utils.h
    csrc/kernels/normalization/norm_kernels.h
    csrc/kernels/convolution/conv_params.h
    csrc/kernels/convolution/conv_kernels.h
    
    # GEMM Headers
    csrc/kernels/gemm/gemm_configs.h
    csrc/kernels/gemm/gemm_utils.h
    csrc/kernels/gemm/gemm_params.h
    csrc/kernels/gemm/gemm_kernels.h
    csrc/kernels/gemm/bmm_params.h
    csrc/kernels/gemm/bmm_kernels.h
    csrc/kernels/gemm/group_gemm_params.h
    csrc/kernels/gemm/group_gemm_kernels.h
    
    # TODO: Add these when ready
    # csrc/common/tensor.h
    # csrc/common/allocator.h
    # csrc/kernels/attention/attention_params.h
    # csrc/kernels/attention/attention_kernels.h
    # csrc/layers/base_layer.h
    # csrc/layers/attention_layer.h
    # csrc/layers/conv_layer.h
    # csrc/layers/ffn_layer.h
    # csrc/layers/encoder_layer.h
)

# =============================================================================
# Library target
# =============================================================================
add_library(oasr_core STATIC ${OASR_SOURCES} ${OASR_HEADERS})

target_include_directories(oasr_core PUBLIC
    ${CMAKE_SOURCE_DIR}/csrc
    ${CUDAToolkit_INCLUDE_DIRS}
)

# CUTLASS
target_include_directories(oasr_core PUBLIC
    ${CUTLASS_DIR}/include
    ${CUTLASS_DIR}/tools/util/include
    )

target_link_libraries(oasr_core PUBLIC
    CUDA::cudart
    ${CUBLAS_LIB}
    ${CUBLASLT_LIB}
)

if(CUDNN_LIB)
    target_link_libraries(oasr_core PUBLIC ${CUDNN_LIB})
    target_compile_definitions(oasr_core PUBLIC OASR_USE_CUDNN)
endif()

# Compile definitions
if(ENABLE_FP16)
    target_compile_definitions(oasr_core PUBLIC OASR_ENABLE_FP16)
endif()
if(ENABLE_BF16)
    target_compile_definitions(oasr_core PUBLIC OASR_ENABLE_BF16)
endif()

# Set CUDA separable compilation for better link times
set_target_properties(oasr_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# =============================================================================
# Python module
# =============================================================================
if(BUILD_PYTHON)
    set(PYBIND_SOURCES
        csrc/pybind/pybind_main.cpp
    )
    
    pybind11_add_module(_C ${PYBIND_SOURCES})

    target_include_directories(_C PRIVATE
        ${CMAKE_SOURCE_DIR}/csrc
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    
    target_link_libraries(_C PRIVATE oasr_core)

    # Set output directory for the Python module (never write into source tree).
    # - When built via setuptools (setup.py), CMAKE_LIBRARY_OUTPUT_DIRECTORY is
    #   set and the .so is placed under build/.
    # - For standalone CMake builds, use the build tree so python/oasr stays clean.
    if(CMAKE_LIBRARY_OUTPUT_DIRECTORY)
        set(PYTHON_MODULE_OUTPUT_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
    else()
        set(PYTHON_MODULE_OUTPUT_DIR ${CMAKE_BINARY_DIR}/oasr)
    endif()

    set_target_properties(_C PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${PYTHON_MODULE_OUTPUT_DIR}
    )
    
    message(STATUS "Python module will be built to: ${PYTHON_MODULE_OUTPUT_DIR}/_C${CMAKE_SHARED_LIBRARY_SUFFIX}")
endif()

# =============================================================================
# Tests
# =============================================================================
if(BUILD_TESTS)
    enable_testing()
    
    # Find or fetch GoogleTest
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.14.0
        )
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()
    
    add_subdirectory(tests)
endif()

# =============================================================================
# Examples
# =============================================================================
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# =============================================================================
# Install rules
# =============================================================================
install(TARGETS oasr_core
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(DIRECTORY csrc/
    DESTINATION include/oasr
    FILES_MATCHING PATTERN "*.h"
)

# =============================================================================
# Summary
# =============================================================================
message(STATUS "")
message(STATUS "=== OASR Build Configuration ===")
message(STATUS "Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Python bindings:   ${BUILD_PYTHON}")
message(STATUS "Build tests:       ${BUILD_TESTS}")
message(STATUS "Use CUTLASS:       ${USE_CUTLASS}")
message(STATUS "Use FlashAttention: ${USE_FLASH_ATTENTION}")
message(STATUS "Enable FP16:       ${ENABLE_FP16}")
message(STATUS "Enable BF16:       ${ENABLE_BF16}")
message(STATUS "================================")
message(STATUS "")
