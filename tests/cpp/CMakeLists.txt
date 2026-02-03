# Tests CMake configuration

# Test sources - only norm_kernels for now
set(TEST_SOURCES
    test_norm_kernels.cu
    # TODO: Add these when ready
    # test_tensor.cpp
    # test_attention.cpp
    # test_conv.cpp
    # test_ffn.cpp
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
