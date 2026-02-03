// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "kernels/feedforward/ffn_kernels.h"
#include "kernels/feedforward/ffn_params.h"
#include "common/tensor.h"

using namespace oasr;
using namespace oasr::kernels;

class FFNTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count;
        cudaGetDeviceCount(&device_count);
        ASSERT_GT(device_count, 0) << "No CUDA devices available";
    }
};

TEST_F(FFNTest, FFNParamsDefaults) {
    FFNParams params;
    
    EXPECT_EQ(params.batch_size, 0);
    EXPECT_EQ(params.seq_len, 0);
    EXPECT_EQ(params.d_model, 0);
    EXPECT_EQ(params.d_ff, 0);
    EXPECT_EQ(params.activation, ActivationType::SWISH);
    EXPECT_EQ(params.dtype, DataType::FP16);
    EXPECT_FLOAT_EQ(params.dropout_prob, 0.0f);
    EXPECT_FALSE(params.use_gated);
    EXPECT_FALSE(params.fused_gated_weights);
}

TEST_F(FFNTest, ConformerFFNParamsDefaults) {
    ConformerFFNParams params;
    
    // Inherited defaults
    EXPECT_EQ(params.activation, ActivationType::SWISH);
    
    // Conformer-specific defaults
    EXPECT_FLOAT_EQ(params.residual_scale, 0.5f);
    EXPECT_TRUE(params.fuse_residual);
    EXPECT_TRUE(params.fuse_layernorm);
}

// TODO: Add more tests when kernel implementations are complete
// - Test standard FFN forward pass
// - Test gated FFN (SwiGLU)
// - Test Conformer FFN with half-step residual
// - Test fused operations
// - Test different activations (ReLU, GELU, Swish)
