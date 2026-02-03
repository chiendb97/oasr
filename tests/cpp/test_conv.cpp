// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "kernels/convolution/conv_kernels.h"
#include "kernels/convolution/conv_params.h"
#include "common/tensor.h"

using namespace oasr;
using namespace oasr::kernels;

class ConvolutionTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count;
        cudaGetDeviceCount(&device_count);
        ASSERT_GT(device_count, 0) << "No CUDA devices available";
    }
};

TEST_F(ConvolutionTest, Conv1DParamsDefaults) {
    Conv1DParams params;
    
    EXPECT_EQ(params.batch_size, 0);
    EXPECT_EQ(params.seq_len, 0);
    EXPECT_EQ(params.in_channels, 0);
    EXPECT_EQ(params.out_channels, 0);
    EXPECT_EQ(params.kernel_size, 0);
    EXPECT_EQ(params.stride, 1);
    EXPECT_EQ(params.padding, 0);
    EXPECT_EQ(params.dilation, 1);
    EXPECT_EQ(params.groups, 1);
    EXPECT_EQ(params.conv_type, ConvType::STANDARD);
    EXPECT_EQ(params.dtype, DataType::FP16);
    EXPECT_TRUE(params.channels_last);
    EXPECT_FALSE(params.is_causal);
}

TEST_F(ConvolutionTest, ConformerConvParamsDefaults) {
    ConformerConvParams params;
    
    EXPECT_EQ(params.batch_size, 0);
    EXPECT_EQ(params.seq_len, 0);
    EXPECT_EQ(params.d_model, 0);
    EXPECT_EQ(params.kernel_size, 31);  // Default Conformer kernel size
    EXPECT_EQ(params.dtype, DataType::FP16);
    EXPECT_FLOAT_EQ(params.batch_norm_eps, 1e-5f);
    EXPECT_FALSE(params.is_causal);
}

TEST_F(ConvolutionTest, ConvStateDefaults) {
    ConvState state;
    
    EXPECT_EQ(state.buffer, nullptr);
    EXPECT_EQ(state.buffer_size, 0);
    EXPECT_EQ(state.channels, 0);
    EXPECT_EQ(state.dtype, DataType::FP16);
}

// TODO: Add more tests when kernel implementations are complete
// - Test depthwise convolution
// - Test pointwise convolution
// - Test Conformer conv module
// - Test causal convolution with state
// - Test GLU activation
// - Test Swish activation
