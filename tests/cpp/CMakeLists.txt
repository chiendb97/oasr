# Tests CMake configuration
# Structure aligned with tests/python/unit/ (test_gemm.py, test_conv_kernels.py, test_norm_kernels.py)

# Test sources (all .cu for consistent extension)
set(TEST_SOURCES
    test_norm.cu
    test_gemm.cu
    test_conv.cu
)

# Add test executable
add_executable(oasr_tests ${TEST_SOURCES})

target_include_directories(oasr_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/csrc
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(oasr_tests PRIVATE
    oasr_core
    GTest::gtest
    GTest::gtest_main
    CUDA::cudart
)

# Add tests to CTest
include(GoogleTest)
gtest_discover_tests(oasr_tests)
