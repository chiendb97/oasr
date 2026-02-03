// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "kernels/attention/attention_kernels.h"
#include "kernels/attention/attention_params.h"
#include "common/tensor.h"
#include "common/cuda_utils.h"

using namespace oasr;
using namespace oasr::kernels;

class AttentionTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count;
        cudaGetDeviceCount(&device_count);
        ASSERT_GT(device_count, 0) << "No CUDA devices available";
    }
};

TEST_F(AttentionTest, AttentionParamsDefaults) {
    AttentionParams params;
    
    EXPECT_EQ(params.batch_size, 0);
    EXPECT_EQ(params.seq_len, 0);
    EXPECT_EQ(params.num_heads, 0);
    EXPECT_EQ(params.head_dim, 0);
    EXPECT_FLOAT_EQ(params.scale, 1.0f);
    EXPECT_EQ(params.attention_type, AttentionType::MULTI_HEAD);
    EXPECT_EQ(params.dtype, DataType::FP16);
    EXPECT_FALSE(params.is_causal);
    EXPECT_TRUE(params.use_flash_attention);
}

TEST_F(AttentionTest, RelativePositionAttentionParamsDefaults) {
    RelativePositionAttentionParams params;
    
    EXPECT_EQ(params.attention_type, AttentionType::RELATIVE_POSITION);
    EXPECT_EQ(params.max_relative_position, 0);
    EXPECT_TRUE(params.clamp_positions);
}

TEST_F(AttentionTest, KVCacheDefaults) {
    KVCache cache;
    
    EXPECT_EQ(cache.k_cache, nullptr);
    EXPECT_EQ(cache.v_cache, nullptr);
    EXPECT_EQ(cache.max_seq_len, 0);
    EXPECT_EQ(cache.current_len, 0);
    EXPECT_EQ(cache.dtype, DataType::FP16);
}

// TODO: Add more tests when kernel implementations are complete
// - Test multi-head attention forward pass
// - Test relative position attention
// - Test KV cache updates
// - Test causal masking
// - Test different data types (FP16, BF16)
