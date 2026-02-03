// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <vector>
#include <random>

#include "kernels/normalization/norm_kernels.h"
#include "common/cuda_utils.h"

namespace oasr {
namespace kernels {
namespace test {

// Helper function to generate random data
template <typename T>
std::vector<T> generateRandomData(size_t size, float min_val = -1.0f, float max_val = 1.0f) {
    std::vector<T> data(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min_val, max_val);
    
    for (size_t i = 0; i < size; ++i) {
        data[i] = static_cast<T>(dist(gen));
    }
    return data;
}

// Helper to compute LayerNorm on CPU for verification
template <typename T>
void cpuLayerNorm(const T* input, T* output,
                  const T* gamma, const T* beta,
                  int batch_size, int seq_len, int hidden_size,
                  float eps) {
    for (int b = 0; b < batch_size * seq_len; ++b) {
        const T* row = input + b * hidden_size;
        T* out_row = output + b * hidden_size;
        
        // Compute mean
        float mean = 0.0f;
        for (int i = 0; i < hidden_size; ++i) {
            mean += static_cast<float>(row[i]);
        }
        mean /= static_cast<float>(hidden_size);
        
        // Compute variance
        float var = 0.0f;
        for (int i = 0; i < hidden_size; ++i) {
            float diff = static_cast<float>(row[i]) - mean;
            var += diff * diff;
        }
        var /= static_cast<float>(hidden_size);
        
        // Normalize
        float inv_std = 1.0f / std::sqrt(var + eps);
        for (int i = 0; i < hidden_size; ++i) {
            float normalized = (static_cast<float>(row[i]) - mean) * inv_std;
            float scaled = normalized * static_cast<float>(gamma[i]);
            if (beta != nullptr) {
                scaled += static_cast<float>(beta[i]);
            }
            out_row[i] = static_cast<T>(scaled);
        }
    }
}

// Helper to compute RMSNorm on CPU for verification
template <typename T>
void cpuRMSNorm(const T* input, T* output,
                const T* gamma,
                int batch_size, int seq_len, int hidden_size,
                float eps) {
    for (int b = 0; b < batch_size * seq_len; ++b) {
        const T* row = input + b * hidden_size;
        T* out_row = output + b * hidden_size;
        
        // Compute mean of squares
        float mean_sq = 0.0f;
        for (int i = 0; i < hidden_size; ++i) {
            float val = static_cast<float>(row[i]);
            mean_sq += val * val;
        }
        mean_sq /= static_cast<float>(hidden_size);
        
        // Normalize
        float inv_rms = 1.0f / std::sqrt(mean_sq + eps);
        for (int i = 0; i < hidden_size; ++i) {
            float normalized = static_cast<float>(row[i]) * inv_rms;
            out_row[i] = static_cast<T>(normalized * static_cast<float>(gamma[i]));
        }
    }
}

class NormKernelsTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaGetDevice(&device_id_);
    }
    
    void TearDown() override {
        cudaDeviceSynchronize();
    }
    
    int device_id_ = 0;
};

// Test LayerNorm with FP32
TEST_F(NormKernelsTest, LayerNormFP32) {
    const int batch_size = 2;
    const int seq_len = 128;
    const int hidden_size = 256;
    const float eps = 1e-5f;
    
    // Generate random input
    auto input_host = generateRandomData<float>(batch_size * seq_len * hidden_size);
    auto gamma_host = generateRandomData<float>(hidden_size, 0.5f, 1.5f);
    auto beta_host = generateRandomData<float>(hidden_size, -0.5f, 0.5f);
    
    // Allocate device memory
    float *d_input, *d_output, *d_gamma, *d_beta;
    size_t input_size = batch_size * seq_len * hidden_size * sizeof(float);
    size_t param_size = hidden_size * sizeof(float);
    
    OASR_CUDA_CHECK(cudaMalloc(&d_input, input_size));
    OASR_CUDA_CHECK(cudaMalloc(&d_output, input_size));
    OASR_CUDA_CHECK(cudaMalloc(&d_gamma, param_size));
    OASR_CUDA_CHECK(cudaMalloc(&d_beta, param_size));
    
    // Copy to device
    OASR_CUDA_CHECK(cudaMemcpy(d_input, input_host.data(), input_size, cudaMemcpyHostToDevice));
    OASR_CUDA_CHECK(cudaMemcpy(d_gamma, gamma_host.data(), param_size, cudaMemcpyHostToDevice));
    OASR_CUDA_CHECK(cudaMemcpy(d_beta, beta_host.data(), param_size, cudaMemcpyHostToDevice));
    
    // Run GPU kernel
    invokeLayerNorm(d_input, d_output, d_gamma, d_beta,
                    batch_size, seq_len, hidden_size,
                    eps, DataType::FP32, nullptr);
    
    // Copy result back
    std::vector<float> output_gpu(batch_size * seq_len * hidden_size);
    OASR_CUDA_CHECK(cudaMemcpy(output_gpu.data(), d_output, input_size, cudaMemcpyDeviceToHost));
    
    // Compute CPU reference
    std::vector<float> output_cpu(batch_size * seq_len * hidden_size);
    cpuLayerNorm(input_host.data(), output_cpu.data(),
                 gamma_host.data(), beta_host.data(),
                 batch_size, seq_len, hidden_size, eps);
    
    // Compare results
    float max_diff = 0.0f;
    for (size_t i = 0; i < output_cpu.size(); ++i) {
        float diff = std::abs(output_gpu[i] - output_cpu[i]);
        max_diff = std::max(max_diff, diff);
    }
    
    EXPECT_LT(max_diff, 1e-4f) << "Max difference: " << max_diff;
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_gamma);
    cudaFree(d_beta);
}

// Test LayerNorm with FP16
TEST_F(NormKernelsTest, LayerNormFP16) {
    const int batch_size = 2;
    const int seq_len = 128;
    const int hidden_size = 256;
    const float eps = 1e-5f;
    
    // Generate random input as float, then convert
    auto input_float = generateRandomData<float>(batch_size * seq_len * hidden_size);
    auto gamma_float = generateRandomData<float>(hidden_size, 0.5f, 1.5f);
    auto beta_float = generateRandomData<float>(hidden_size, -0.5f, 0.5f);
    
    // Convert to half
    std::vector<half> input_host(input_float.size());
    std::vector<half> gamma_host(gamma_float.size());
    std::vector<half> beta_host(beta_float.size());
    
    for (size_t i = 0; i < input_float.size(); ++i) {
        input_host[i] = __float2half(input_float[i]);
    }
    for (size_t i = 0; i < gamma_float.size(); ++i) {
        gamma_host[i] = __float2half(gamma_float[i]);
        beta_host[i] = __float2half(beta_float[i]);
    }
    
    // Allocate device memory
    half *d_input, *d_output, *d_gamma, *d_beta;
    size_t input_size = batch_size * seq_len * hidden_size * sizeof(half);
    size_t param_size = hidden_size * sizeof(half);
    
    OASR_CUDA_CHECK(cudaMalloc(&d_input, input_size));
    OASR_CUDA_CHECK(cudaMalloc(&d_output, input_size));
    OASR_CUDA_CHECK(cudaMalloc(&d_gamma, param_size));
    OASR_CUDA_CHECK(cudaMalloc(&d_beta, param_size));
    
    // Copy to device
    OASR_CUDA_CHECK(cudaMemcpy(d_input, input_host.data(), input_size, cudaMemcpyHostToDevice));
    OASR_CUDA_CHECK(cudaMemcpy(d_gamma, gamma_host.data(), param_size, cudaMemcpyHostToDevice));
    OASR_CUDA_CHECK(cudaMemcpy(d_beta, beta_host.data(), param_size, cudaMemcpyHostToDevice));
    
    // Run GPU kernel
    invokeLayerNorm(d_input, d_output, d_gamma, d_beta,
                    batch_size, seq_len, hidden_size,
                    eps, DataType::FP16, nullptr);
    
    // Copy result back
    std::vector<half> output_gpu(batch_size * seq_len * hidden_size);
    OASR_CUDA_CHECK(cudaMemcpy(output_gpu.data(), d_output, input_size, cudaMemcpyDeviceToHost));
    
    // Compute CPU reference (in float for accuracy)
    std::vector<float> output_cpu(batch_size * seq_len * hidden_size);
    cpuLayerNorm(input_float.data(), output_cpu.data(),
                 gamma_float.data(), beta_float.data(),
                 batch_size, seq_len, hidden_size, eps);
    
    // Compare results
    float max_diff = 0.0f;
    for (size_t i = 0; i < output_cpu.size(); ++i) {
        float gpu_val = __half2float(output_gpu[i]);
        float diff = std::abs(gpu_val - output_cpu[i]);
        max_diff = std::max(max_diff, diff);
    }
    
    // FP16 has lower precision
    EXPECT_LT(max_diff, 1e-2f) << "Max difference: " << max_diff;
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_gamma);
    cudaFree(d_beta);
}

// Test RMSNorm with FP32
TEST_F(NormKernelsTest, RMSNormFP32) {
    const int batch_size = 2;
    const int seq_len = 128;
    const int hidden_size = 256;
    const float eps = 1e-5f;
    
    auto input_host = generateRandomData<float>(batch_size * seq_len * hidden_size);
    auto gamma_host = generateRandomData<float>(hidden_size, 0.5f, 1.5f);
    
    float *d_input, *d_output, *d_gamma;
    size_t input_size = batch_size * seq_len * hidden_size * sizeof(float);
    size_t param_size = hidden_size * sizeof(float);
    
    OASR_CUDA_CHECK(cudaMalloc(&d_input, input_size));
    OASR_CUDA_CHECK(cudaMalloc(&d_output, input_size));
    OASR_CUDA_CHECK(cudaMalloc(&d_gamma, param_size));
    
    OASR_CUDA_CHECK(cudaMemcpy(d_input, input_host.data(), input_size, cudaMemcpyHostToDevice));
    OASR_CUDA_CHECK(cudaMemcpy(d_gamma, gamma_host.data(), param_size, cudaMemcpyHostToDevice));
    
    invokeRMSNorm(d_input, d_output, d_gamma,
                  batch_size, seq_len, hidden_size,
                  eps, DataType::FP32, nullptr);
    
    std::vector<float> output_gpu(batch_size * seq_len * hidden_size);
    OASR_CUDA_CHECK(cudaMemcpy(output_gpu.data(), d_output, input_size, cudaMemcpyDeviceToHost));
    
    std::vector<float> output_cpu(batch_size * seq_len * hidden_size);
    cpuRMSNorm(input_host.data(), output_cpu.data(),
               gamma_host.data(),
               batch_size, seq_len, hidden_size, eps);
    
    float max_diff = 0.0f;
    for (size_t i = 0; i < output_cpu.size(); ++i) {
        float diff = std::abs(output_gpu[i] - output_cpu[i]);
        max_diff = std::max(max_diff, diff);
    }
    
    EXPECT_LT(max_diff, 1e-4f) << "Max difference: " << max_diff;
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_gamma);
}

// Test with different hidden sizes
TEST_F(NormKernelsTest, LayerNormDifferentSizes) {
    const std::vector<int> hidden_sizes = {64, 128, 256, 512, 768, 1024};
    const int batch_size = 2;
    const int seq_len = 32;
    const float eps = 1e-5f;
    
    for (int hidden_size : hidden_sizes) {
        auto input_host = generateRandomData<float>(batch_size * seq_len * hidden_size);
        auto gamma_host = generateRandomData<float>(hidden_size, 0.5f, 1.5f);
        auto beta_host = generateRandomData<float>(hidden_size, -0.5f, 0.5f);
        
        float *d_input, *d_output, *d_gamma, *d_beta;
        size_t input_size = batch_size * seq_len * hidden_size * sizeof(float);
        size_t param_size = hidden_size * sizeof(float);
        
        OASR_CUDA_CHECK(cudaMalloc(&d_input, input_size));
        OASR_CUDA_CHECK(cudaMalloc(&d_output, input_size));
        OASR_CUDA_CHECK(cudaMalloc(&d_gamma, param_size));
        OASR_CUDA_CHECK(cudaMalloc(&d_beta, param_size));
        
        OASR_CUDA_CHECK(cudaMemcpy(d_input, input_host.data(), input_size, cudaMemcpyHostToDevice));
        OASR_CUDA_CHECK(cudaMemcpy(d_gamma, gamma_host.data(), param_size, cudaMemcpyHostToDevice));
        OASR_CUDA_CHECK(cudaMemcpy(d_beta, beta_host.data(), param_size, cudaMemcpyHostToDevice));
        
        invokeLayerNorm(d_input, d_output, d_gamma, d_beta,
                        batch_size, seq_len, hidden_size,
                        eps, DataType::FP32, nullptr);
        
        std::vector<float> output_gpu(batch_size * seq_len * hidden_size);
        OASR_CUDA_CHECK(cudaMemcpy(output_gpu.data(), d_output, input_size, cudaMemcpyDeviceToHost));
        
        std::vector<float> output_cpu(batch_size * seq_len * hidden_size);
        cpuLayerNorm(input_host.data(), output_cpu.data(),
                     gamma_host.data(), beta_host.data(),
                     batch_size, seq_len, hidden_size, eps);
        
        float max_diff = 0.0f;
        for (size_t i = 0; i < output_cpu.size(); ++i) {
            float diff = std::abs(output_gpu[i] - output_cpu[i]);
            max_diff = std::max(max_diff, diff);
        }
        
        EXPECT_LT(max_diff, 1e-4f) << "Failed for hidden_size=" << hidden_size 
                                   << ", max_diff=" << max_diff;
        
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_gamma);
        cudaFree(d_beta);
    }
}

} // namespace test
} // namespace kernels
} // namespace oasr
