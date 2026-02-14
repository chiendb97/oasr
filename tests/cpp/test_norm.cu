// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for normalization layers.

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <vector>
#include <random>

#include "kernels/norm/norm_kernels.h"
#include "common/cuda_utils.h"

using namespace oasr;
using namespace oasr::kernels;

namespace oasr {
namespace kernels {
namespace test {

template <typename T>
std::vector<T> generateRandomData(size_t size, float min_val = -1.0f, float max_val = 1.0f) {
    std::vector<T> data(size);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(min_val, max_val);
    for (size_t i = 0; i < size; ++i) {
        data[i] = static_cast<T>(dist(gen));
    }
    return data;
}

template <typename T>
void cpuLayerNorm(const T* input, T* output,
                  const T* gamma, const T* beta,
                  int batch_size, int seq_len, int hidden_size,
                  float eps) {
    for (int b = 0; b < batch_size * seq_len; ++b) {
        const T* row = input + b * hidden_size;
        T* out_row = output + b * hidden_size;
        float mean = 0.0f;
        for (int i = 0; i < hidden_size; ++i) mean += static_cast<float>(row[i]);
        mean /= static_cast<float>(hidden_size);
        float var = 0.0f;
        for (int i = 0; i < hidden_size; ++i) {
            float diff = static_cast<float>(row[i]) - mean;
            var += diff * diff;
        }
        var /= static_cast<float>(hidden_size);
        float inv_std = 1.0f / std::sqrt(var + eps);
        for (int i = 0; i < hidden_size; ++i) {
            float normalized = (static_cast<float>(row[i]) - mean) * inv_std;
            float scaled = normalized * static_cast<float>(gamma[i]);
            if (beta) scaled += static_cast<float>(beta[i]);
            out_row[i] = static_cast<T>(scaled);
        }
    }
}

template <typename T>
void cpuRMSNorm(const T* input, T* output, const T* gamma,
                int batch_size, int seq_len, int hidden_size, float eps) {
    for (int b = 0; b < batch_size * seq_len; ++b) {
        const T* row = input + b * hidden_size;
        T* out_row = output + b * hidden_size;
        float mean_sq = 0.0f;
        for (int i = 0; i < hidden_size; ++i) {
            float val = static_cast<float>(row[i]);
            mean_sq += val * val;
        }
        mean_sq /= static_cast<float>(hidden_size);
        float inv_rms = 1.0f / std::sqrt(mean_sq + eps);
        for (int i = 0; i < hidden_size; ++i)
            out_row[i] = static_cast<T>(static_cast<float>(row[i]) * inv_rms * static_cast<float>(gamma[i]));
    }
}

}  // namespace test
}  // namespace kernels
}  // namespace oasr

//==============================================================================
// TestLayerNorm
//==============================================================================

class TestLayerNorm : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        ASSERT_GT(device_count, 0) << "No CUDA devices available";
    }
    void TearDown() override { cudaDeviceSynchronize(); }
    void RunLayerNorm(int batch_size, int seq_len, int hidden_size, DataType dtype,
                      float rtol, float atol, const float* beta_host = nullptr);
};

void TestLayerNorm::RunLayerNorm(int batch_size, int seq_len, int hidden_size, DataType dtype,
                                 float rtol, float atol, const float* beta_host) {
    const float eps = 1e-5f;
    auto input_f = oasr::kernels::test::generateRandomData<float>(batch_size * seq_len * hidden_size);
    auto gamma_f = oasr::kernels::test::generateRandomData<float>(hidden_size, 0.5f, 1.5f);
    std::vector<float> beta_f(hidden_size);
    if (beta_host) {
        for (int i = 0; i < hidden_size; ++i) beta_f[i] = beta_host[i];
    } else {
        beta_f = oasr::kernels::test::generateRandomData<float>(hidden_size, -0.5f, 0.5f);
    }

    size_t n = static_cast<size_t>(batch_size) * seq_len * hidden_size;
    size_t p = static_cast<size_t>(hidden_size);

    if (dtype == DataType::FP32) {
        float *d_in = nullptr, *d_out = nullptr, *d_gamma = nullptr, *d_beta = nullptr;
        OASR_CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
        OASR_CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
        OASR_CUDA_CHECK(cudaMalloc(&d_gamma, p * sizeof(float)));
        OASR_CUDA_CHECK(cudaMalloc(&d_beta, p * sizeof(float)));
        OASR_CUDA_CHECK(cudaMemcpy(d_in, input_f.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        OASR_CUDA_CHECK(cudaMemcpy(d_gamma, gamma_f.data(), p * sizeof(float), cudaMemcpyHostToDevice));
        OASR_CUDA_CHECK(cudaMemcpy(d_beta, beta_f.data(), p * sizeof(float), cudaMemcpyHostToDevice));

        invokeLayerNorm(d_in, d_out, d_gamma, d_beta,
                        batch_size, seq_len, hidden_size, eps, DataType::FP32, nullptr);

        std::vector<float> out_gpu(n);
        OASR_CUDA_CHECK(cudaMemcpy(out_gpu.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));

        std::vector<float> out_cpu(n);
        oasr::kernels::test::cpuLayerNorm(input_f.data(), out_cpu.data(),
                                         gamma_f.data(), beta_f.data(),
                                         batch_size, seq_len, hidden_size, eps);

        for (size_t i = 0; i < n; ++i)
            EXPECT_LE(std::abs(out_gpu[i] - out_cpu[i]), atol + rtol * std::abs(out_cpu[i]))
                << "i=" << i << " gpu=" << out_gpu[i] << " ref=" << out_cpu[i];

        cudaFree(d_in); cudaFree(d_out); cudaFree(d_gamma); cudaFree(d_beta);
        return;
    }

    // FP16
    std::vector<half> input_h(n), gamma_h(p), beta_h(p);
    for (size_t i = 0; i < n; ++i) input_h[i] = __float2half(input_f[i]);
    for (size_t i = 0; i < p; ++i) { gamma_h[i] = __float2half(gamma_f[i]); beta_h[i] = __float2half(beta_f[i]); }

    half *d_in = nullptr, *d_out = nullptr, *d_gamma = nullptr, *d_beta = nullptr;
    OASR_CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(half)));
    OASR_CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(half)));
    OASR_CUDA_CHECK(cudaMalloc(&d_gamma, p * sizeof(half)));
    OASR_CUDA_CHECK(cudaMalloc(&d_beta, p * sizeof(half)));
    OASR_CUDA_CHECK(cudaMemcpy(d_in, input_h.data(), n * sizeof(half), cudaMemcpyHostToDevice));
    OASR_CUDA_CHECK(cudaMemcpy(d_gamma, gamma_h.data(), p * sizeof(half), cudaMemcpyHostToDevice));
    OASR_CUDA_CHECK(cudaMemcpy(d_beta, beta_h.data(), p * sizeof(half), cudaMemcpyHostToDevice));

    invokeLayerNorm(d_in, d_out, d_gamma, d_beta,
                    batch_size, seq_len, hidden_size, eps, DataType::FP16, nullptr);

    std::vector<half> out_h(n);
    OASR_CUDA_CHECK(cudaMemcpy(out_h.data(), d_out, n * sizeof(half), cudaMemcpyDeviceToHost));

    std::vector<float> out_cpu(n);
    oasr::kernels::test::cpuLayerNorm(input_f.data(), out_cpu.data(),
                                     gamma_f.data(), beta_f.data(),
                                     batch_size, seq_len, hidden_size, eps);

    for (size_t i = 0; i < n; ++i) {
        float gpu_val = __half2float(out_h[i]);
        EXPECT_LE(std::abs(gpu_val - out_cpu[i]), atol + rtol * std::abs(out_cpu[i]))
            << "i=" << i << " gpu=" << gpu_val << " ref=" << out_cpu[i];
    }
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_gamma); cudaFree(d_beta);
}

TEST_F(TestLayerNorm, LayerNorm_1_64_128_FP32) {
    RunLayerNorm(1, 64, 128, DataType::FP32, 1e-4f, 1e-4f);
}
TEST_F(TestLayerNorm, LayerNorm_2_128_256_FP32) {
    RunLayerNorm(2, 128, 256, DataType::FP32, 1e-4f, 1e-4f);
}
TEST_F(TestLayerNorm, LayerNorm_4_256_512_FP32) {
    RunLayerNorm(4, 256, 512, DataType::FP32, 1e-4f, 1e-4f);
}
TEST_F(TestLayerNorm, LayerNorm_8_512_768_FP32) {
    RunLayerNorm(8, 512, 768, DataType::FP32, 1e-4f, 1e-4f);
}
TEST_F(TestLayerNorm, LayerNorm_2_128_1024_FP32) {
    RunLayerNorm(2, 128, 1024, DataType::FP32, 1e-4f, 1e-4f);
}
TEST_F(TestLayerNorm, LayerNorm_1_64_128_FP16) {
    RunLayerNorm(1, 64, 128, DataType::FP16, 1e-2f, 1e-2f);
}
TEST_F(TestLayerNorm, LayerNorm_2_128_256_FP16) {
    RunLayerNorm(2, 128, 256, DataType::FP16, 1e-2f, 1e-2f);
}
TEST_F(TestLayerNorm, LayerNormNoBeta) {
    const int batch_size = 2, seq_len = 128, hidden_size = 256;
    const float eps = 1e-5f;
    auto input_f = oasr::kernels::test::generateRandomData<float>(batch_size * seq_len * hidden_size);
    auto gamma_f = oasr::kernels::test::generateRandomData<float>(hidden_size, 0.5f, 1.5f);
    size_t n = static_cast<size_t>(batch_size) * seq_len * hidden_size;
    size_t p = static_cast<size_t>(hidden_size);

    float *d_in = nullptr, *d_out = nullptr, *d_gamma = nullptr;
    OASR_CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
    OASR_CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    OASR_CUDA_CHECK(cudaMalloc(&d_gamma, p * sizeof(float)));
    OASR_CUDA_CHECK(cudaMemcpy(d_in, input_f.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    OASR_CUDA_CHECK(cudaMemcpy(d_gamma, gamma_f.data(), p * sizeof(float), cudaMemcpyHostToDevice));

    invokeLayerNorm(d_in, d_out, d_gamma, nullptr,
                    batch_size, seq_len, hidden_size, eps, DataType::FP32, nullptr);

    std::vector<float> out_gpu(n);
    OASR_CUDA_CHECK(cudaMemcpy(out_gpu.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> out_cpu(n);
    oasr::kernels::test::cpuLayerNorm(input_f.data(), out_cpu.data(),
                                     gamma_f.data(), static_cast<const float*>(nullptr),
                                     batch_size, seq_len, hidden_size, eps);

    for (size_t i = 0; i < n; ++i)
        EXPECT_LE(std::abs(out_gpu[i] - out_cpu[i]), 1e-4f) << "i=" << i;
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_gamma);
}

//==============================================================================
// TestRMSNorm
//==============================================================================

class TestRMSNorm : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        ASSERT_GT(device_count, 0) << "No CUDA devices available";
    }
    void TearDown() override { cudaDeviceSynchronize(); }
    void RunRMSNorm(int batch_size, int seq_len, int hidden_size, DataType dtype, float rtol, float atol);
};

void TestRMSNorm::RunRMSNorm(int batch_size, int seq_len, int hidden_size, DataType dtype, float rtol, float atol) {
    const float eps = 1e-5f;
    size_t n = static_cast<size_t>(batch_size) * seq_len * hidden_size;
    size_t p = static_cast<size_t>(hidden_size);
    auto input_f = oasr::kernels::test::generateRandomData<float>(n);
    auto gamma_f = oasr::kernels::test::generateRandomData<float>(p, 0.5f, 1.5f);

    if (dtype == DataType::FP32) {
        float *d_in = nullptr, *d_out = nullptr, *d_gamma = nullptr;
        OASR_CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
        OASR_CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
        OASR_CUDA_CHECK(cudaMalloc(&d_gamma, p * sizeof(float)));
        OASR_CUDA_CHECK(cudaMemcpy(d_in, input_f.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        OASR_CUDA_CHECK(cudaMemcpy(d_gamma, gamma_f.data(), p * sizeof(float), cudaMemcpyHostToDevice));

        invokeRMSNorm(d_in, d_out, d_gamma, batch_size, seq_len, hidden_size, eps, DataType::FP32, nullptr);

        std::vector<float> out_gpu(n), out_cpu(n);
        OASR_CUDA_CHECK(cudaMemcpy(out_gpu.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
        oasr::kernels::test::cpuRMSNorm(input_f.data(), out_cpu.data(), gamma_f.data(),
                                        batch_size, seq_len, hidden_size, eps);
        for (size_t i = 0; i < n; ++i)
            EXPECT_LE(std::abs(out_gpu[i] - out_cpu[i]), atol + rtol * std::abs(out_cpu[i])) << "i=" << i;
        cudaFree(d_in); cudaFree(d_out); cudaFree(d_gamma);
        return;
    }

    std::vector<half> input_h(n), gamma_h(p);
    for (size_t i = 0; i < n; ++i) input_h[i] = __float2half(input_f[i]);
    for (size_t i = 0; i < p; ++i) gamma_h[i] = __float2half(gamma_f[i]);
    half *d_in = nullptr, *d_out = nullptr, *d_gamma = nullptr;
    OASR_CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(half)));
    OASR_CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(half)));
    OASR_CUDA_CHECK(cudaMalloc(&d_gamma, p * sizeof(half)));
    OASR_CUDA_CHECK(cudaMemcpy(d_in, input_h.data(), n * sizeof(half), cudaMemcpyHostToDevice));
    OASR_CUDA_CHECK(cudaMemcpy(d_gamma, gamma_h.data(), p * sizeof(half), cudaMemcpyHostToDevice));

    invokeRMSNorm(d_in, d_out, d_gamma, batch_size, seq_len, hidden_size, eps, DataType::FP16, nullptr);

    std::vector<half> out_h(n);
    std::vector<float> out_cpu(n);
    OASR_CUDA_CHECK(cudaMemcpy(out_h.data(), d_out, n * sizeof(half), cudaMemcpyDeviceToHost));
    oasr::kernels::test::cpuRMSNorm(input_f.data(), out_cpu.data(), gamma_f.data(),
                                    batch_size, seq_len, hidden_size, eps);
    for (size_t i = 0; i < n; ++i) {
        float gpu_val = __half2float(out_h[i]);
        EXPECT_LE(std::abs(gpu_val - out_cpu[i]), atol + rtol * std::abs(out_cpu[i])) << "i=" << i;
    }
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_gamma);
}

TEST_F(TestRMSNorm, RMSNorm_1_64_128_FP32) { RunRMSNorm(1, 64, 128, DataType::FP32, 1e-4f, 1e-4f); }
TEST_F(TestRMSNorm, RMSNorm_2_128_256_FP32) { RunRMSNorm(2, 128, 256, DataType::FP32, 1e-4f, 1e-4f); }
TEST_F(TestRMSNorm, RMSNorm_4_256_512_FP32) { RunRMSNorm(4, 256, 512, DataType::FP32, 1e-4f, 1e-4f); }
TEST_F(TestRMSNorm, RMSNorm_1_64_128_FP16) { RunRMSNorm(1, 64, 128, DataType::FP16, 1e-2f, 1e-2f); }
TEST_F(TestRMSNorm, RMSNorm_2_128_256_FP16) { RunRMSNorm(2, 128, 256, DataType::FP16, 1e-2f, 1e-2f); }

//==============================================================================
// TestAddLayerNorm
//==============================================================================

class TestAddLayerNorm : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        ASSERT_GT(device_count, 0) << "No CUDA devices available";
    }
    void TearDown() override { cudaDeviceSynchronize(); }
};

TEST_F(TestAddLayerNorm, AddLayerNorm_2_128_256) {
    const int batch_size = 2, seq_len = 128, hidden_size = 256;
    const float eps = 1e-5f;
    size_t n = static_cast<size_t>(batch_size) * seq_len * hidden_size;
    size_t p = static_cast<size_t>(hidden_size);
    auto x_f = oasr::kernels::test::generateRandomData<float>(n);
    auto res_f = oasr::kernels::test::generateRandomData<float>(n);
    auto gamma_f = oasr::kernels::test::generateRandomData<float>(p, 0.5f, 1.5f);
    auto beta_f = oasr::kernels::test::generateRandomData<float>(p, -0.5f, 0.5f);

    float *d_x = nullptr, *d_res = nullptr, *d_out = nullptr, *d_gamma = nullptr, *d_beta = nullptr;
    OASR_CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    OASR_CUDA_CHECK(cudaMalloc(&d_res, n * sizeof(float)));
    OASR_CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    OASR_CUDA_CHECK(cudaMalloc(&d_gamma, p * sizeof(float)));
    OASR_CUDA_CHECK(cudaMalloc(&d_beta, p * sizeof(float)));
    OASR_CUDA_CHECK(cudaMemcpy(d_x, x_f.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    OASR_CUDA_CHECK(cudaMemcpy(d_res, res_f.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    OASR_CUDA_CHECK(cudaMemcpy(d_gamma, gamma_f.data(), p * sizeof(float), cudaMemcpyHostToDevice));
    OASR_CUDA_CHECK(cudaMemcpy(d_beta, beta_f.data(), p * sizeof(float), cudaMemcpyHostToDevice));

    invokeAddLayerNorm(d_x, d_res, d_out, d_gamma, d_beta,
                       batch_size, seq_len, hidden_size, eps, DataType::FP32, nullptr);

    std::vector<float> out_gpu(n);
    OASR_CUDA_CHECK(cudaMemcpy(out_gpu.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> sum(n);
    for (size_t i = 0; i < n; ++i) sum[i] = x_f[i] + res_f[i];
    std::vector<float> out_cpu(n);
    oasr::kernels::test::cpuLayerNorm(sum.data(), out_cpu.data(),
                                      gamma_f.data(), beta_f.data(),
                                      batch_size, seq_len, hidden_size, eps);

    for (size_t i = 0; i < n; ++i)
        EXPECT_LE(std::abs(out_gpu[i] - out_cpu[i]), 1e-4f) << "i=" << i;
    cudaFree(d_x); cudaFree(d_res); cudaFree(d_out); cudaFree(d_gamma); cudaFree(d_beta);
}

TEST_F(TestAddLayerNorm, AddLayerNorm_4_256_512) {
    const int batch_size = 4, seq_len = 256, hidden_size = 512;
    const float eps = 1e-5f;
    size_t n = static_cast<size_t>(batch_size) * seq_len * hidden_size;
    size_t p = static_cast<size_t>(hidden_size);
    auto x_f = oasr::kernels::test::generateRandomData<float>(n);
    auto res_f = oasr::kernels::test::generateRandomData<float>(n);
    auto gamma_f = oasr::kernels::test::generateRandomData<float>(p, 0.5f, 1.5f);
    auto beta_f = oasr::kernels::test::generateRandomData<float>(p, -0.5f, 0.5f);

    float *d_x = nullptr, *d_res = nullptr, *d_out = nullptr, *d_gamma = nullptr, *d_beta = nullptr;
    OASR_CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    OASR_CUDA_CHECK(cudaMalloc(&d_res, n * sizeof(float)));
    OASR_CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    OASR_CUDA_CHECK(cudaMalloc(&d_gamma, p * sizeof(float)));
    OASR_CUDA_CHECK(cudaMalloc(&d_beta, p * sizeof(float)));
    OASR_CUDA_CHECK(cudaMemcpy(d_x, x_f.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    OASR_CUDA_CHECK(cudaMemcpy(d_res, res_f.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    OASR_CUDA_CHECK(cudaMemcpy(d_gamma, gamma_f.data(), p * sizeof(float), cudaMemcpyHostToDevice));
    OASR_CUDA_CHECK(cudaMemcpy(d_beta, beta_f.data(), p * sizeof(float), cudaMemcpyHostToDevice));

    invokeAddLayerNorm(d_x, d_res, d_out, d_gamma, d_beta,
                       batch_size, seq_len, hidden_size, eps, DataType::FP32, nullptr);

    std::vector<float> out_gpu(n);
    OASR_CUDA_CHECK(cudaMemcpy(out_gpu.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
    std::vector<float> sum(n);
    for (size_t i = 0; i < n; ++i) sum[i] = x_f[i] + res_f[i];
    std::vector<float> out_cpu(n);
    oasr::kernels::test::cpuLayerNorm(sum.data(), out_cpu.data(),
                                      gamma_f.data(), beta_f.data(),
                                      batch_size, seq_len, hidden_size, eps);
    for (size_t i = 0; i < n; ++i)
        EXPECT_LE(std::abs(out_gpu[i] - out_cpu[i]), 1e-4f) << "i=" << i;
    cudaFree(d_x); cudaFree(d_res); cudaFree(d_out); cudaFree(d_gamma); cudaFree(d_beta);
}

//==============================================================================
// TestBatchNorm1D
//==============================================================================

class TestBatchNorm1D : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        ASSERT_GT(device_count, 0) << "No CUDA devices available";
    }
    void TearDown() override { cudaDeviceSynchronize(); }
};

TEST_F(TestBatchNorm1D, BatchNorm1D_2_128_256) {
    const int batch_size = 2, seq_len = 128, channels = 256;
    const float eps = 1e-5f;
    size_t n = static_cast<size_t>(batch_size) * seq_len * channels;
    size_t p = static_cast<size_t>(channels);
    auto x_f = oasr::kernels::test::generateRandomData<float>(n);
    auto gamma_f = oasr::kernels::test::generateRandomData<float>(p, 0.5f, 1.5f);
    auto beta_f = oasr::kernels::test::generateRandomData<float>(p, -0.5f, 0.5f);
    auto mean_f = oasr::kernels::test::generateRandomData<float>(p);
    auto var_f = oasr::kernels::test::generateRandomData<float>(p, 0.1f, 2.0f);

    float *d_x = nullptr, *d_out = nullptr, *d_gamma = nullptr, *d_beta = nullptr;
    float *d_mean = nullptr, *d_var = nullptr;
    OASR_CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    OASR_CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    OASR_CUDA_CHECK(cudaMalloc(&d_gamma, p * sizeof(float)));
    OASR_CUDA_CHECK(cudaMalloc(&d_beta, p * sizeof(float)));
    OASR_CUDA_CHECK(cudaMalloc(&d_mean, p * sizeof(float)));
    OASR_CUDA_CHECK(cudaMalloc(&d_var, p * sizeof(float)));
    OASR_CUDA_CHECK(cudaMemcpy(d_x, x_f.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    OASR_CUDA_CHECK(cudaMemcpy(d_gamma, gamma_f.data(), p * sizeof(float), cudaMemcpyHostToDevice));
    OASR_CUDA_CHECK(cudaMemcpy(d_beta, beta_f.data(), p * sizeof(float), cudaMemcpyHostToDevice));
    OASR_CUDA_CHECK(cudaMemcpy(d_mean, mean_f.data(), p * sizeof(float), cudaMemcpyHostToDevice));
    OASR_CUDA_CHECK(cudaMemcpy(d_var, var_f.data(), p * sizeof(float), cudaMemcpyHostToDevice));

    invokeBatchNorm1D(d_x, d_out, d_gamma, d_beta, d_mean, d_var,
                      batch_size, seq_len, channels, eps, DataType::FP32, nullptr);

    std::vector<float> out_gpu(n);
    OASR_CUDA_CHECK(cudaMemcpy(out_gpu.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < n; ++i) {
        int c = i % channels;
        float expected = (x_f[i] - mean_f[c]) / std::sqrt(var_f[c] + eps) * gamma_f[c] + beta_f[c];
        EXPECT_LE(std::abs(out_gpu[i] - expected), 1e-4f) << "i=" << i;
    }
    cudaFree(d_x); cudaFree(d_out); cudaFree(d_gamma); cudaFree(d_beta); cudaFree(d_mean); cudaFree(d_var);
}

TEST_F(TestBatchNorm1D, BatchNorm1D_4_64_512) {
    const int batch_size = 4, seq_len = 64, channels = 512;
    const float eps = 1e-5f;
    size_t n = static_cast<size_t>(batch_size) * seq_len * channels;
    size_t p = static_cast<size_t>(channels);
    auto x_f = oasr::kernels::test::generateRandomData<float>(n);
    auto gamma_f = oasr::kernels::test::generateRandomData<float>(p, 0.5f, 1.5f);
    auto beta_f = oasr::kernels::test::generateRandomData<float>(p, -0.5f, 0.5f);
    auto mean_f = oasr::kernels::test::generateRandomData<float>(p);
    auto var_f = oasr::kernels::test::generateRandomData<float>(p, 0.1f, 2.0f);

    float *d_x = nullptr, *d_out = nullptr, *d_gamma = nullptr, *d_beta = nullptr;
    float *d_mean = nullptr, *d_var = nullptr;
    OASR_CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    OASR_CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    OASR_CUDA_CHECK(cudaMalloc(&d_gamma, p * sizeof(float)));
    OASR_CUDA_CHECK(cudaMalloc(&d_beta, p * sizeof(float)));
    OASR_CUDA_CHECK(cudaMalloc(&d_mean, p * sizeof(float)));
    OASR_CUDA_CHECK(cudaMalloc(&d_var, p * sizeof(float)));
    OASR_CUDA_CHECK(cudaMemcpy(d_x, x_f.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    OASR_CUDA_CHECK(cudaMemcpy(d_gamma, gamma_f.data(), p * sizeof(float), cudaMemcpyHostToDevice));
    OASR_CUDA_CHECK(cudaMemcpy(d_beta, beta_f.data(), p * sizeof(float), cudaMemcpyHostToDevice));
    OASR_CUDA_CHECK(cudaMemcpy(d_mean, mean_f.data(), p * sizeof(float), cudaMemcpyHostToDevice));
    OASR_CUDA_CHECK(cudaMemcpy(d_var, var_f.data(), p * sizeof(float), cudaMemcpyHostToDevice));

    invokeBatchNorm1D(d_x, d_out, d_gamma, d_beta, d_mean, d_var,
                      batch_size, seq_len, channels, eps, DataType::FP32, nullptr);

    std::vector<float> out_gpu(n);
    OASR_CUDA_CHECK(cudaMemcpy(out_gpu.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n; ++i) {
        int c = i % channels;
        float expected = (x_f[i] - mean_f[c]) / std::sqrt(var_f[c] + eps) * gamma_f[c] + beta_f[c];
        EXPECT_LE(std::abs(out_gpu[i] - expected), 1e-4f) << "i=" << i;
    }
    cudaFree(d_x); cudaFree(d_out); cudaFree(d_gamma); cudaFree(d_beta); cudaFree(d_mean); cudaFree(d_var);
}

//==============================================================================
// TestGroupNorm
//==============================================================================

class TestGroupNorm : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        ASSERT_GT(device_count, 0) << "No CUDA devices available";
    }
    void TearDown() override { cudaDeviceSynchronize(); }
};

// TODO: Re-enable when GroupNorm kernel numerics match CPU ref (vectorized path?)
TEST_F(TestGroupNorm, DISABLED_GroupNorm_2_64_256_32) {
    const int batch_size = 2, seq_len = 64, channels = 256, num_groups = 32;
    const float eps = 1e-5f;
    size_t n = static_cast<size_t>(batch_size) * seq_len * channels;
    size_t p = static_cast<size_t>(channels);
    auto x_f = oasr::kernels::test::generateRandomData<float>(n);
    auto gamma_f = oasr::kernels::test::generateRandomData<float>(p, 0.5f, 1.5f);
    auto beta_f = oasr::kernels::test::generateRandomData<float>(p, -0.5f, 0.5f);

    float *d_x = nullptr, *d_out = nullptr, *d_gamma = nullptr, *d_beta = nullptr;
    OASR_CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    OASR_CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    OASR_CUDA_CHECK(cudaMalloc(&d_gamma, p * sizeof(float)));
    OASR_CUDA_CHECK(cudaMalloc(&d_beta, p * sizeof(float)));
    OASR_CUDA_CHECK(cudaMemcpy(d_x, x_f.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    OASR_CUDA_CHECK(cudaMemcpy(d_gamma, gamma_f.data(), p * sizeof(float), cudaMemcpyHostToDevice));
    OASR_CUDA_CHECK(cudaMemcpy(d_beta, beta_f.data(), p * sizeof(float), cudaMemcpyHostToDevice));

    invokeGroupNorm(d_x, d_out, d_gamma, d_beta,
                    batch_size, seq_len, channels, num_groups, eps, DataType::FP32, nullptr);

    std::vector<float> out_gpu(n);
    OASR_CUDA_CHECK(cudaMemcpy(out_gpu.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));

    int cpq = channels / num_groups;
    for (int b = 0; b < batch_size; ++b) {
        for (int t = 0; t < seq_len; ++t) {
            for (int g = 0; g < num_groups; ++g) {
                float mean = 0.0f, var = 0.0f;
                for (int c = 0; c < cpq; ++c) {
                    float v = x_f[b * seq_len * channels + t * channels + g * cpq + c];
                    mean += v;
                }
                mean /= cpq;
                for (int c = 0; c < cpq; ++c) {
                    float v = x_f[b * seq_len * channels + t * channels + g * cpq + c];
                    var += (v - mean) * (v - mean);
                }
                var /= cpq;
                float inv_std = 1.0f / std::sqrt(var + eps);
                for (int c = 0; c < cpq; ++c) {
                    int idx = b * seq_len * channels + t * channels + g * cpq + c;
                    float expected = (x_f[idx] - mean) * inv_std * gamma_f[idx] + beta_f[idx];
                    EXPECT_LE(std::abs(out_gpu[idx] - expected), 1e-3f) << "idx=" << idx;
                }
            }
        }
    }
    cudaFree(d_x); cudaFree(d_out); cudaFree(d_gamma); cudaFree(d_beta);
}

TEST_F(TestGroupNorm, GroupNorm_4_128_512_64) {
    const int batch_size = 4, seq_len = 128, channels = 512, num_groups = 64;
    const float eps = 1e-5f;
    size_t n = static_cast<size_t>(batch_size) * seq_len * channels;
    size_t p = static_cast<size_t>(channels);
    auto x_f = oasr::kernels::test::generateRandomData<float>(n);
    auto gamma_f = oasr::kernels::test::generateRandomData<float>(p, 0.5f, 1.5f);
    auto beta_f = oasr::kernels::test::generateRandomData<float>(p, -0.5f, 0.5f);

    float *d_x = nullptr, *d_out = nullptr, *d_gamma = nullptr, *d_beta = nullptr;
    OASR_CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    OASR_CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    OASR_CUDA_CHECK(cudaMalloc(&d_gamma, p * sizeof(float)));
    OASR_CUDA_CHECK(cudaMalloc(&d_beta, p * sizeof(float)));
    OASR_CUDA_CHECK(cudaMemcpy(d_x, x_f.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    OASR_CUDA_CHECK(cudaMemcpy(d_gamma, gamma_f.data(), p * sizeof(float), cudaMemcpyHostToDevice));
    OASR_CUDA_CHECK(cudaMemcpy(d_beta, beta_f.data(), p * sizeof(float), cudaMemcpyHostToDevice));

    invokeGroupNorm(d_x, d_out, d_gamma, d_beta,
                    batch_size, seq_len, channels, num_groups, eps, DataType::FP32, nullptr);

    std::vector<float> out_gpu(n);
    OASR_CUDA_CHECK(cudaMemcpy(out_gpu.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
    int cpq = channels / num_groups;
    for (size_t idx = 0; idx < n; ++idx) {
        int b = idx / (seq_len * channels);
        int rest = idx % (seq_len * channels);
        int t = rest / channels;
        int c = rest % channels;
        int g = c / cpq;
        size_t base = static_cast<size_t>(b) * seq_len * channels + t * channels + g * cpq;
        float mean = 0.0f, var = 0.0f;
        for (int i = 0; i < cpq; ++i) mean += x_f[base + i];
        mean /= cpq;
        for (int i = 0; i < cpq; ++i) { float v = x_f[base + i]; var += (v - mean) * (v - mean); }
        var /= cpq;
        float expected = (x_f[idx] - mean) / std::sqrt(var + eps) * gamma_f[c] + beta_f[c];
        EXPECT_LE(std::abs(out_gpu[idx] - expected), 1e-3f) << "idx=" << idx;
    }
    cudaFree(d_x); cudaFree(d_out); cudaFree(d_gamma); cudaFree(d_beta);
}

// TODO: Re-enable when GroupNorm kernel numerics match CPU ref
TEST_F(TestGroupNorm, DISABLED_GroupNorm_2_64_128_8) {
    const int batch_size = 2, seq_len = 64, channels = 128, num_groups = 8;
    const float eps = 1e-5f;
    size_t n = static_cast<size_t>(batch_size) * seq_len * channels;
    size_t p = static_cast<size_t>(channels);
    auto x_f = oasr::kernels::test::generateRandomData<float>(n);
    auto gamma_f = oasr::kernels::test::generateRandomData<float>(p, 0.5f, 1.5f);
    auto beta_f = oasr::kernels::test::generateRandomData<float>(p, -0.5f, 0.5f);

    float *d_x = nullptr, *d_out = nullptr, *d_gamma = nullptr, *d_beta = nullptr;
    OASR_CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    OASR_CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    OASR_CUDA_CHECK(cudaMalloc(&d_gamma, p * sizeof(float)));
    OASR_CUDA_CHECK(cudaMalloc(&d_beta, p * sizeof(float)));
    OASR_CUDA_CHECK(cudaMemcpy(d_x, x_f.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    OASR_CUDA_CHECK(cudaMemcpy(d_gamma, gamma_f.data(), p * sizeof(float), cudaMemcpyHostToDevice));
    OASR_CUDA_CHECK(cudaMemcpy(d_beta, beta_f.data(), p * sizeof(float), cudaMemcpyHostToDevice));

    invokeGroupNorm(d_x, d_out, d_gamma, d_beta,
                    batch_size, seq_len, channels, num_groups, eps, DataType::FP32, nullptr);

    std::vector<float> out_gpu(n);
    OASR_CUDA_CHECK(cudaMemcpy(out_gpu.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
    int cpq = channels / num_groups;
    for (int b = 0; b < batch_size; ++b) {
        for (int t = 0; t < seq_len; ++t) {
            for (int g = 0; g < num_groups; ++g) {
                float mean = 0.0f, var = 0.0f;
                for (int c = 0; c < cpq; ++c) {
                    float v = x_f[b * seq_len * channels + t * channels + g * cpq + c];
                    mean += v;
                }
                mean /= cpq;
                for (int c = 0; c < cpq; ++c) {
                    float v = x_f[b * seq_len * channels + t * channels + g * cpq + c];
                    var += (v - mean) * (v - mean);
                }
                var /= cpq;
                float inv_std = 1.0f / std::sqrt(var + eps);
                for (int c = 0; c < cpq; ++c) {
                    int idx = b * seq_len * channels + t * channels + g * cpq + c;
                    float expected = (x_f[idx] - mean) * inv_std * gamma_f[idx] + beta_f[idx];
                    EXPECT_LE(std::abs(out_gpu[idx] - expected), 1e-3f) << "idx=" << idx;
                }
            }
        }
    }
    cudaFree(d_x); cudaFree(d_out); cudaFree(d_gamma); cudaFree(d_beta);
}
