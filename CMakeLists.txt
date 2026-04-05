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
option(OASR_USE_K2 "Enable K2 WFST beam search decoder (requires K2)" OFF)

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
    # Support common architectures: Volta, Turing, Ampere, Ada, Hopper, Blackwell
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90 100 120)
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
set(CUTLASS_DIR "${CMAKE_SOURCE_DIR}/3rdparty/cutlass" CACHE PATH "CUTLASS directory (header-only)")


# Python and dependencies for bindings
if(BUILD_PYTHON)
    find_package(Python COMPONENTS Interpreter Development REQUIRED)

    # TVM-FFI is only needed at runtime for JIT compilation (not at build time).

    # pybind11 (still needed for decoder bindings)
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

    # PyTorch (libtorch) — still needed for nn.Module layer wrappers and decoder.
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE TORCH_FIND_RESULT
    )
    if(TORCH_FIND_RESULT EQUAL 0)
        list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")
    endif()
    find_package(Torch REQUIRED)
    find_library(TORCH_PYTHON_LIB torch_python
        PATHS "${TORCH_INSTALL_PREFIX}/lib"
        REQUIRED)
    message(STATUS "Found PyTorch: ${Torch_DIR}")
    message(STATUS "Found torch_python: ${TORCH_PYTHON_LIB}")

    # JIT CUDA launchers are not built by this CMake graph; emit a stub .cu so
    # compile_commands.json contains -I<tvm_ffi>/include for IDE/clangd.
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import tvm_ffi.libinfo as L; print(L.find_include_path())"
        OUTPUT_VARIABLE TVM_FFI_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE TVM_FFI_INCLUDE_FIND_RESULT
    )
    if(TVM_FFI_INCLUDE_FIND_RESULT EQUAL 0 AND EXISTS "${TVM_FFI_INCLUDE_DIR}/tvm/ffi/c_api.h")
        message(STATUS "Found TVM-FFI headers (IDE compile_commands): ${TVM_FFI_INCLUDE_DIR}")
        set(OASR_TVM_FFI_INTELLISENSE_STUB "${CMAKE_BINARY_DIR}/oasr_tvm_ffi_intellisense_stub.cu")
        file(WRITE ${OASR_TVM_FFI_INTELLISENSE_STUB}
            "// Generated by CMake — not linked; IDE/clangd compile_commands only.\n"
            "#include \"tvm_ffi_utils.h\"\n"
        )
        add_library(oasr_tvm_ffi_intellisense OBJECT ${OASR_TVM_FFI_INTELLISENSE_STUB})
        target_include_directories(oasr_tvm_ffi_intellisense PRIVATE
            ${CMAKE_SOURCE_DIR}/csrc
            ${TVM_FFI_INCLUDE_DIR}
            ${CUDAToolkit_INCLUDE_DIRS}
        )
    else()
        message(STATUS "TVM-FFI headers not found (optional). pip install apache-tvm-ffi for JIT/IDE.")
    endif()
endif()

# =============================================================================
# Source files (FlashInfer-style flat csrc/ layout)
# =============================================================================
# ---------------------------------------------------------------------------
# oasr_core: only the decoder (CPU-side C++) is statically compiled.
# Kernel launchers + JIT bindings are compiled at runtime by the JIT system.
# ---------------------------------------------------------------------------
set(OASR_SOURCES
    csrc/decoder/common/utils.cc
    csrc/decoder/context_graph.cc
    csrc/decoder/ctc_prefix_beam_search.cc
    csrc/decoder/ctc_greedy_search.cc
)

set(OASR_HEADERS
    # Common headers (used by JIT-compiled kernels at runtime)
    include/oasr/common/types.h
    include/oasr/common/vec_dtypes.h
    include/oasr/common/math.h
    include/oasr/common/arch_dispatch.h
    include/oasr/common/epilogue_functors.h

    # Decoder headers
    csrc/decoder/common/utils.h
    csrc/decoder/context_graph.h
    csrc/decoder/ctc_prefix_beam_search.h
    csrc/decoder/ctc_greedy_search.h
    csrc/decoder/ctc_wfst_beam_search.h
)

# =============================================================================
# Optional: K2 WFST decoder
# =============================================================================
if(OASR_USE_K2)
    # k2_DIR must be set by the caller (setup.py resolves it from the installed
    # k2 Python package and passes -Dk2_DIR=... on the cmake command line).
    # Manual builds: cmake -DOASR_USE_K2=ON -Dk2_DIR=$(python -c
    #   "import k2,os; print(os.path.join(os.path.dirname(k2.__file__),'share','cmake','k2'))")
    find_package(k2 REQUIRED)
    list(APPEND OASR_SOURCES csrc/decoder/ctc_wfst_beam_search.cc)
    message(STATUS "K2 WFST decoder enabled (k2 version: ${k2_VERSION})")
endif()

# =============================================================================
# Library target (decoder only — kernel launchers are JIT-compiled at runtime)
# =============================================================================
add_library(oasr_core STATIC ${OASR_SOURCES} ${OASR_HEADERS})

target_include_directories(oasr_core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/csrc
)

if(OASR_USE_K2)
    target_compile_definitions(oasr_core PUBLIC OASR_USE_K2)
    # k2/torch_api.h includes torch/script.h, so oasr_core needs Torch headers.
    target_include_directories(oasr_core PUBLIC ${TORCH_INCLUDE_DIRS})
    target_link_libraries(oasr_core PUBLIC k2_torch_api ${TORCH_LIBRARIES})
    message(STATUS "Linking oasr_core with K2")
endif()

# =============================================================================
# Python module
# =============================================================================
if(BUILD_PYTHON)
    set(PYBIND_SOURCES
        csrc/pybind/pybind_main.cpp
    )
    
    pybind11_add_module(_C ${PYBIND_SOURCES})

    target_include_directories(_C PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/csrc
        ${CUDAToolkit_INCLUDE_DIRS}
        ${TORCH_INCLUDE_DIRS}
    )
    
    target_link_libraries(_C PRIVATE oasr_core ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIB})
    target_compile_options(_C PRIVATE ${TORCH_CXX_FLAGS})

    # Set output directory for the Python module (never write into source tree).
    # - When built via setuptools (setup.py), CMAKE_LIBRARY_OUTPUT_DIRECTORY is
    #   set and the .so is placed under build/.
    # - For standalone CMake builds, use the build tree so oasr stays clean.
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

install(DIRECTORY include/oasr/
    DESTINATION include/oasr
    FILES_MATCHING PATTERN "*.h" PATTERN "*.cuh"
)

# =============================================================================
# Summary
# =============================================================================
message(STATUS "")
message(STATUS "=== OASR Build Configuration ===")
message(STATUS "Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Python bindings:   ${BUILD_PYTHON}")
message(STATUS "Use CUTLASS:       ${USE_CUTLASS}")
message(STATUS "Use FlashAttention: ${USE_FLASH_ATTENTION}")
message(STATUS "Enable FP16:       ${ENABLE_FP16}")
message(STATUS "Enable BF16:       ${ENABLE_BF16}")
message(STATUS "================================")
message(STATUS "")
