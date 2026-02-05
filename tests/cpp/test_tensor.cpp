// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "common/tensor.h"
#include "common/cuda_utils.h"

using namespace oasr;

class TensorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Ensure CUDA is available
        int device_count;
        cudaGetDeviceCount(&device_count);
        ASSERT_GT(device_count, 0) << "No CUDA devices available";
    }
};

TEST_F(TensorTest, CreateEmpty) {
    Tensor t({2, 3, 4}, DataType::FP32, MemoryType::GPU, 0);
    
    EXPECT_EQ(t.ndim(), 3);
    EXPECT_EQ(t.shape(0), 2);
    EXPECT_EQ(t.shape(1), 3);
    EXPECT_EQ(t.shape(2), 4);
    EXPECT_EQ(t.numel(), 24);
    EXPECT_EQ(t.nbytes(), 24 * 4);  // float32 = 4 bytes
    EXPECT_EQ(t.dtype(), DataType::FP32);
    EXPECT_EQ(t.memoryType(), MemoryType::GPU);
}

TEST_F(TensorTest, CreateZeros) {
    auto t = Tensor::zeros({4, 4}, DataType::FP16, MemoryType::GPU, 0);
    
    EXPECT_EQ(t.numel(), 16);
    EXPECT_EQ(t.nbytes(), 32);  // float16 = 2 bytes
    EXPECT_EQ(t.dtype(), DataType::FP16);
}

TEST_F(TensorTest, CreateOnes) {
    auto t = Tensor::ones({2, 2}, DataType::FP32, MemoryType::GPU, 0);
    
    // Move to CPU to check values
    auto cpu_t = t.toCPU();
    const float* data = cpu_t.data<float>();
    
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(data[i], 1.0f);
    }
}

TEST_F(TensorTest, Clone) {
    auto t1 = Tensor::ones({3, 3}, DataType::FP32, MemoryType::GPU, 0);
    auto t2 = t1.clone();
    
    // Verify shapes match
    EXPECT_EQ(t1.shape(), t2.shape());
    EXPECT_EQ(t1.dtype(), t2.dtype());
    
    // Verify data is copied (not shared)
    EXPECT_NE(t1.data(), t2.data());
}

TEST_F(TensorTest, Reshape) {
    Tensor t({2, 3, 4}, DataType::FP32, MemoryType::GPU, 0);
    
    auto reshaped = t.reshape({6, 4});
    
    EXPECT_EQ(reshaped.ndim(), 2);
    EXPECT_EQ(reshaped.shape(0), 6);
    EXPECT_EQ(reshaped.shape(1), 4);
    EXPECT_EQ(reshaped.numel(), 24);  // Same number of elements
}

TEST_F(TensorTest, Transpose) {
    Tensor t({2, 3}, DataType::FP32, MemoryType::GPU, 0);
    
    auto transposed = t.transpose(0, 1);
    
    EXPECT_EQ(transposed.shape(0), 3);
    EXPECT_EQ(transposed.shape(1), 2);
}

TEST_F(TensorTest, ToDevice) {
    auto gpu_t = Tensor::ones({4, 4}, DataType::FP32, MemoryType::GPU, 0);
    EXPECT_EQ(gpu_t.memoryType(), MemoryType::GPU);
    
    auto cpu_t = gpu_t.toCPU();
    EXPECT_EQ(cpu_t.memoryType(), MemoryType::CPU);
    
    auto gpu_t2 = cpu_t.toGPU();
    EXPECT_EQ(gpu_t2.memoryType(), MemoryType::GPU);
}

TEST_F(TensorTest, ToDataType) {
    auto fp32_t = Tensor::ones({4, 4}, DataType::FP32, MemoryType::GPU, 0);
    EXPECT_EQ(fp32_t.dtype(), DataType::FP32);
    
    auto fp16_t = fp32_t.toFP16();
    EXPECT_EQ(fp16_t.dtype(), DataType::FP16);
    EXPECT_EQ(fp16_t.nbytes(), fp32_t.numel() * 2);  // Half the bytes
}

TEST_F(TensorTest, Fill) {
    auto t = Tensor::empty({3, 3}, DataType::FP32, MemoryType::GPU, 0);
    t.fill(3.14f);
    
    auto cpu_t = t.toCPU();
    const float* data = cpu_t.data<float>();
    
    for (int i = 0; i < 9; ++i) {
        EXPECT_FLOAT_EQ(data[i], 3.14f);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
