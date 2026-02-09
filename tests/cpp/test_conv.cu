// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for convolution kernels.
// Structure and behavior aligned with tests/python/unit/test_conv_kernels.py

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <random>

#include "kernels/convolution/conv_kernels.h"
#include "kernels/convolution/conv_params.h"
#include "common/cuda_utils.h"

using namespace oasr;
using namespace oasr::kernels;

//-----------------------------------------------------------------------------
// Fixture and CPU reference helpers
//-----------------------------------------------------------------------------

class ConvTestFixture : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        ASSERT_GT(device_count, 0) << "No CUDA devices available";
    }
    void TearDown() override {
        cudaDeviceSynchronize();
    }
};

// CPU reference: depthwise conv1d [B,T,C], weight [C,K], bias [C] -> out [B,T,C], padding
static void cpuDepthwiseConv1D(const float* x, const float* w, const float* bias,
                               float* out, int B, int T, int C, int K, int padding) {
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            for (int c = 0; c < C; ++c) {
                float sum = bias[c];
                for (int k = 0; k < K; ++k) {
                    int idx = t - padding + k;
                    if (idx >= 0 && idx < T)
                        sum += x[b * T * C + idx * C + c] * w[c * K + k];
                }
                out[b * T * C + t * C + c] = sum;
            }
        }
    }
}

// CPU reference: pointwise = linear per position: out[b,t,:] = x[b,t,:] @ W.T + bias
static void cpuPointwiseConv1D(const float* x, const float* w, const float* bias,
                               float* out, int B, int T, int in_ch, int out_ch) {
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            for (int j = 0; j < out_ch; ++j) {
                float sum = bias[j];
                for (int i = 0; i < in_ch; ++i)
                    sum += x[(b * T + t) * in_ch + i] * w[j * in_ch + i];
                out[(b * T + t) * out_ch + j] = sum;
            }
        }
    }
}

// CPU reference: GLU - out[b,t,c] = x[b,t,c] * sigmoid(x[b,t,C+c])
// Layout [B, T, 2*C] last dim contiguous; output [B, T, C]
static void cpuGLU(const float* x, float* out, int B, int T, int C) {
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            for (int c = 0; c < C; ++c) {
                int value_idx = (b * T + t) * (2 * C) + c;
                int gate_idx = (b * T + t) * (2 * C) + C + c;
                float val = x[value_idx];
                float gate = x[gate_idx];
                out[(b * T + t) * C + c] = val * (1.0f / (1.0f + std::exp(-gate)));
            }
        }
    }
}

// CPU reference: Swish = x * sigmoid(x)
static void cpuSwish(const float* x, float* out, int N) {
    for (int i = 0; i < N; ++i)
        out[i] = x[i] * (1.0f / (1.0f + std::exp(-x[i])));
}

//-----------------------------------------------------------------------------
// TestDepthwiseConv1D
//-----------------------------------------------------------------------------

class TestDepthwiseConv1D : public ConvTestFixture {};

TEST_F(TestDepthwiseConv1D, DepthwiseConv1D_1_64_128_3) {
    const int batch_size = 1, seq_len = 64, channels = 128, kernel_size = 3;
    const int padding = kernel_size / 2;
    const float eps = 1e-4f;
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    size_t n = static_cast<size_t>(batch_size) * seq_len * channels;
    size_t w_size = static_cast<size_t>(channels) * kernel_size;
    std::vector<float> x_host(n), w_host(w_size), b_host(channels), out_host(n);
    for (float& v : x_host) v = dist(gen);
    for (float& v : w_host) v = dist(gen);
    for (float& v : b_host) v = dist(gen);

    std::vector<float> ref(n);
    cpuDepthwiseConv1D(x_host.data(), w_host.data(), b_host.data(), ref.data(),
                       batch_size, seq_len, channels, kernel_size, padding);

    float *d_x = nullptr, *d_w = nullptr, *d_b = nullptr, *d_out = nullptr;
    OASR_CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    OASR_CUDA_CHECK(cudaMalloc(&d_w, w_size * sizeof(float)));
    OASR_CUDA_CHECK(cudaMalloc(&d_b, channels * sizeof(float)));
    OASR_CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    OASR_CUDA_CHECK(cudaMemcpy(d_x, x_host.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    OASR_CUDA_CHECK(cudaMemcpy(d_w, w_host.data(), w_size * sizeof(float), cudaMemcpyHostToDevice));
    OASR_CUDA_CHECK(cudaMemcpy(d_b, b_host.data(), channels * sizeof(float), cudaMemcpyHostToDevice));

    invokeDepthwiseConv1D(d_x, d_w, d_b, d_out,
                          batch_size, seq_len, channels, kernel_size, padding,
                          false, DataType::FP32, nullptr);

    OASR_CUDA_CHECK(cudaMemcpy(out_host.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n; ++i)
        EXPECT_NEAR(out_host[i], ref[i], eps) << "i=" << i;
    cudaFree(d_x); cudaFree(d_w); cudaFree(d_b); cudaFree(d_out);
}

TEST_F(TestDepthwiseConv1D, Conv1DParamsDefaults) {
    Conv1DParams params{};
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
    EXPECT_TRUE(params.channels_last);
    EXPECT_FALSE(params.is_causal);
}

//-----------------------------------------------------------------------------
// TestPointwiseConv1D
//-----------------------------------------------------------------------------

class TestPointwiseConv1D : public ConvTestFixture {};

TEST_F(TestPointwiseConv1D, PointwiseConv1D_2_128_256_512) {
    const int batch_size = 2, seq_len = 128, in_ch = 256, out_ch = 512;
    const float eps = 1e-4f;
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    size_t n_in = static_cast<size_t>(batch_size) * seq_len * in_ch;
    size_t n_out = static_cast<size_t>(batch_size) * seq_len * out_ch;
    std::vector<float> x_host(n_in), w_host(out_ch * in_ch), b_host(out_ch), out_host(n_out);
    for (float& v : x_host) v = dist(gen);
    for (float& v : w_host) v = dist(gen);
    for (float& v : b_host) v = dist(gen);

    std::vector<float> ref(n_out);
    cpuPointwiseConv1D(x_host.data(), w_host.data(), b_host.data(), ref.data(),
                       batch_size, seq_len, in_ch, out_ch);

    float *d_x = nullptr, *d_w = nullptr, *d_b = nullptr, *d_out = nullptr;
    OASR_CUDA_CHECK(cudaMalloc(&d_x, n_in * sizeof(float)));
    OASR_CUDA_CHECK(cudaMalloc(&d_w, w_host.size() * sizeof(float)));
    OASR_CUDA_CHECK(cudaMalloc(&d_b, out_ch * sizeof(float)));
    OASR_CUDA_CHECK(cudaMalloc(&d_out, n_out * sizeof(float)));
    OASR_CUDA_CHECK(cudaMemcpy(d_x, x_host.data(), n_in * sizeof(float), cudaMemcpyHostToDevice));
    OASR_CUDA_CHECK(cudaMemcpy(d_w, w_host.data(), w_host.size() * sizeof(float), cudaMemcpyHostToDevice));
    OASR_CUDA_CHECK(cudaMemcpy(d_b, b_host.data(), out_ch * sizeof(float), cudaMemcpyHostToDevice));

    invokePointwiseConv1D(d_x, d_w, d_b, d_out,
                          batch_size, seq_len, in_ch, out_ch,
                          PointwiseConvBackend::NATIVE,
                          ActivationType::SWISH, false, DataType::FP32, nullptr);

    OASR_CUDA_CHECK(cudaMemcpy(out_host.data(), d_out, n_out * sizeof(float), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n_out; ++i)
        EXPECT_NEAR(out_host[i], ref[i], eps) << "i=" << i;
    cudaFree(d_x); cudaFree(d_w); cudaFree(d_b); cudaFree(d_out);
}

//-----------------------------------------------------------------------------
// TestGLU
//-----------------------------------------------------------------------------

class TestGLU : public ConvTestFixture {};

TEST_F(TestGLU, Glu_2_128_256) {
    const int batch_size = 2, seq_len = 128, channels = 256;
    const float eps = 1e-5f;
    size_t n_in = static_cast<size_t>(batch_size) * seq_len * (2 * channels);
    size_t n_out = static_cast<size_t>(batch_size) * seq_len * channels;
    std::vector<float> x_host(n_in), out_host(n_out);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& v : x_host) v = dist(gen);

    std::vector<float> ref(n_out);
    cpuGLU(x_host.data(), ref.data(), batch_size, seq_len, channels);

    float *d_x = nullptr, *d_out = nullptr;
    OASR_CUDA_CHECK(cudaMalloc(&d_x, n_in * sizeof(float)));
    OASR_CUDA_CHECK(cudaMalloc(&d_out, n_out * sizeof(float)));
    OASR_CUDA_CHECK(cudaMemcpy(d_x, x_host.data(), n_in * sizeof(float), cudaMemcpyHostToDevice));

    invokeGLU(d_x, d_out, batch_size, seq_len, channels, DataType::FP32, nullptr);

    OASR_CUDA_CHECK(cudaMemcpy(out_host.data(), d_out, n_out * sizeof(float), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n_out; ++i)
        EXPECT_NEAR(out_host[i], ref[i], eps) << "i=" << i;
    cudaFree(d_x); cudaFree(d_out);
}

//-----------------------------------------------------------------------------
// TestSwish
//-----------------------------------------------------------------------------

class TestSwish : public ConvTestFixture {};

TEST_F(TestSwish, Swish_2_128_256) {
    const int batch_size = 2, seq_len = 128, channels = 256;
    const float eps = 1e-5f;
    size_t n = static_cast<size_t>(batch_size) * seq_len * channels;
    std::vector<float> x_host(n), out_host(n);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& v : x_host) v = dist(gen);

    std::vector<float> ref(n);
    cpuSwish(x_host.data(), ref.data(), static_cast<int>(n));

    float *d_x = nullptr, *d_out = nullptr;
    OASR_CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    OASR_CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    OASR_CUDA_CHECK(cudaMemcpy(d_x, x_host.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    invokeSwish(d_x, d_out, batch_size, seq_len, channels, DataType::FP32, nullptr);

    OASR_CUDA_CHECK(cudaMemcpy(out_host.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n; ++i)
        EXPECT_NEAR(out_host[i], ref[i], eps) << "i=" << i;
    cudaFree(d_x); cudaFree(d_out);
}

//-----------------------------------------------------------------------------
// TestBatchNormSwish
//-----------------------------------------------------------------------------

class TestBatchNormSwish : public ConvTestFixture {};

TEST_F(TestBatchNormSwish, BatchNormSwish_2_128_256) {
    const int batch_size = 2, seq_len = 128, channels = 256;
    const float eps = 1e-5f;
    size_t n = static_cast<size_t>(batch_size) * seq_len * channels;
    std::vector<float> x_host(n), gamma(channels), beta(channels), mean(channels), var(channels), out_host(n);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& v : x_host) v = dist(gen);
    for (float& v : gamma) v = dist(gen);
    for (float& v : beta) v = dist(gen);
    for (float& v : mean) v = dist(gen);
    for (float& v : var) v = std::abs(dist(gen)) + 0.1f;

    float *d_x = nullptr, *d_out = nullptr, *d_g = nullptr, *d_b = nullptr, *d_m = nullptr, *d_v = nullptr;
    OASR_CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    OASR_CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    OASR_CUDA_CHECK(cudaMalloc(&d_g, channels * sizeof(float)));
    OASR_CUDA_CHECK(cudaMalloc(&d_b, channels * sizeof(float)));
    OASR_CUDA_CHECK(cudaMalloc(&d_m, channels * sizeof(float)));
    OASR_CUDA_CHECK(cudaMalloc(&d_v, channels * sizeof(float)));
    OASR_CUDA_CHECK(cudaMemcpy(d_x, x_host.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    OASR_CUDA_CHECK(cudaMemcpy(d_g, gamma.data(), channels * sizeof(float), cudaMemcpyHostToDevice));
    OASR_CUDA_CHECK(cudaMemcpy(d_b, beta.data(), channels * sizeof(float), cudaMemcpyHostToDevice));
    OASR_CUDA_CHECK(cudaMemcpy(d_m, mean.data(), channels * sizeof(float), cudaMemcpyHostToDevice));
    OASR_CUDA_CHECK(cudaMemcpy(d_v, var.data(), channels * sizeof(float), cudaMemcpyHostToDevice));

    invokeBatchNormSwish(d_x, d_out, d_g, d_b, d_m, d_v,
                         batch_size, seq_len, channels, eps, DataType::FP32, nullptr);

    OASR_CUDA_CHECK(cudaMemcpy(out_host.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n; ++i) {
        int c = i % channels;
        float bn = (x_host[i] - mean[c]) / std::sqrt(var[c] + eps) * gamma[c] + beta[c];
        float expected = bn * (1.0f / (1.0f + std::exp(-bn)));
        EXPECT_NEAR(out_host[i], expected, 1e-4f) << "i=" << i;
    }
    cudaFree(d_x); cudaFree(d_out); cudaFree(d_g); cudaFree(d_b); cudaFree(d_m); cudaFree(d_v);
}

//-----------------------------------------------------------------------------
// TestConv1D (standard conv)
//-----------------------------------------------------------------------------

class TestConv1D : public ConvTestFixture {};

TEST_F(TestConv1D, Conv1DParamsDefaults) {
    Conv1DParams params{};
    EXPECT_EQ(params.batch_size, 0);
    EXPECT_EQ(params.kernel_size, 0);
    EXPECT_EQ(params.stride, 1);
    EXPECT_EQ(params.padding, 0);
    EXPECT_EQ(params.conv_type, ConvType::STANDARD);
}

//-----------------------------------------------------------------------------
// TestConformerConvPattern
//-----------------------------------------------------------------------------

class TestConformerConvPattern : public ConvTestFixture {};

TEST_F(TestConformerConvPattern, ConformerConvParamsDefaults) {
    ConformerConvParams params{};
    EXPECT_EQ(params.batch_size, 0);
    EXPECT_EQ(params.seq_len, 0);
    EXPECT_EQ(params.d_model, 0);
    EXPECT_EQ(params.kernel_size, 31);
    EXPECT_EQ(params.dtype, DataType::FP16);
    EXPECT_FLOAT_EQ(params.batch_norm_eps, 1e-5f);
    EXPECT_FALSE(params.is_causal);
}

TEST_F(TestConformerConvPattern, ConvStateDefaults) {
    ConvState state{};
    EXPECT_EQ(state.buffer, nullptr);
    EXPECT_EQ(state.buffer_size, 0);
    EXPECT_EQ(state.channels, 0);
    EXPECT_EQ(state.dtype, DataType::FP16);
}
