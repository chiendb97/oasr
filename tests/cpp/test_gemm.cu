// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <vector>
#include <random>

#include "kernels/gemm/gemm_params.h"
#include "kernels/gemm/gemm_kernels.h"
#include "kernels/gemm/bmm_params.h"
#include "kernels/gemm/bmm_kernels.h"
#include "kernels/gemm/group_gemm_params.h"
#include "common/cuda_utils.h"

using namespace oasr;
using namespace oasr::kernels::gemm;

namespace oasr {
namespace kernels {
namespace test {

//-----------------------------------------------------------------------------
// CPU reference GEMM (float, row-major)
//-----------------------------------------------------------------------------
void cpuGemm(const float* A, const float* B, float* D,
             int M, int N, int K,
             int lda, int ldb, int ldd,
             float alpha, float beta) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            D[i * ldd + j] = alpha * sum + (beta != 0.0f ? beta * D[i * ldd + j] : 0.0f);
        }
    }
}

// Convert half to float for comparison
inline float halfToFloat(half h) {
    return __half2float(h);
}

}  // namespace test
}  // namespace kernels
}  // namespace oasr

//==============================================================================
// Parameter default tests (follow test_conv.cpp style)
//==============================================================================

class GemmParamsTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        ASSERT_GT(device_count, 0) << "No CUDA devices available";
    }
};

TEST_F(GemmParamsTest, GemmParamsDefaults) {
    GemmParams params;

    EXPECT_EQ(params.A, nullptr);
    EXPECT_EQ(params.B, nullptr);
    EXPECT_EQ(params.C, nullptr);
    EXPECT_EQ(params.D, nullptr);
    EXPECT_EQ(params.M, 0);
    EXPECT_EQ(params.N, 0);
    EXPECT_EQ(params.K, 0);
    EXPECT_EQ(params.lda, 0);
    EXPECT_EQ(params.ldb, 0);
    EXPECT_EQ(params.ldc, 0);
    EXPECT_EQ(params.ldd, 0);
    EXPECT_FLOAT_EQ(params.alpha, 1.0f);
    EXPECT_FLOAT_EQ(params.beta, 0.0f);
    EXPECT_EQ(params.trans_a, TransposeOp::NoTranspose);
    EXPECT_EQ(params.trans_b, TransposeOp::NoTranspose);
    EXPECT_EQ(params.dtype_a, DataType::FP16);
    EXPECT_EQ(params.dtype_b, DataType::FP16);
    EXPECT_EQ(params.dtype_d, DataType::FP16);
    EXPECT_EQ(params.epilogue_fusion, EpilogueFusion::NONE);
    EXPECT_EQ(params.stream, nullptr);
}

TEST_F(GemmParamsTest, BmmParamsDefaults) {
    BmmParams params;

    EXPECT_EQ(params.A, nullptr);
    EXPECT_EQ(params.B, nullptr);
    EXPECT_EQ(params.D, nullptr);
    EXPECT_EQ(params.batch_size, 0);
    EXPECT_EQ(params.M, 0);
    EXPECT_EQ(params.N, 0);
    EXPECT_EQ(params.K, 0);
    EXPECT_FALSE(params.use_pointer_array);
    EXPECT_EQ(params.stream, nullptr);
}

TEST_F(GemmParamsTest, BmmParamsStridedFactory) {
    // Strided factory sets correct strides
    void* dummy = nullptr;
    auto params = BmmParams::Strided(dummy, dummy, dummy, 8, 64, 128, 256,
                                      DataType::FP16, nullptr);

    EXPECT_EQ(params.batch_size, 8);
    EXPECT_EQ(params.M, 64);
    EXPECT_EQ(params.N, 128);
    EXPECT_EQ(params.K, 256);
    EXPECT_EQ(params.lda, 256);
    EXPECT_EQ(params.ldb, 128);
    EXPECT_EQ(params.ldd, 128);
    EXPECT_EQ(params.stride_a, static_cast<int64_t>(64) * 256);
    EXPECT_EQ(params.stride_b, static_cast<int64_t>(256) * 128);
    EXPECT_EQ(params.stride_d, static_cast<int64_t>(64) * 128);
    EXPECT_FALSE(params.use_pointer_array);
}

TEST_F(GemmParamsTest, GemmProblemDescDefaults) {
    GemmProblemDesc desc;

    EXPECT_EQ(desc.M, 0);
    EXPECT_EQ(desc.N, 0);
    EXPECT_EQ(desc.K, 0);
}

TEST_F(GemmParamsTest, GemmProblemDescConstructor) {
    GemmProblemDesc desc(10, 20, 30);

    EXPECT_EQ(desc.M, 10);
    EXPECT_EQ(desc.N, 20);
    EXPECT_EQ(desc.K, 30);
}

TEST_F(GemmParamsTest, GroupGemmParamsDefaults) {
    GroupGemmParams params;

    EXPECT_EQ(params.problems, nullptr);
    EXPECT_EQ(params.num_problems, 0);
    EXPECT_EQ(params.A_array, nullptr);
    EXPECT_EQ(params.B_array, nullptr);
    EXPECT_EQ(params.D_array, nullptr);
    EXPECT_EQ(params.lda_array, nullptr);
    EXPECT_EQ(params.ldb_array, nullptr);
    EXPECT_EQ(params.ldd_array, nullptr);
    EXPECT_FLOAT_EQ(params.alpha, 1.0f);
    EXPECT_FLOAT_EQ(params.beta, 0.0f);
    EXPECT_EQ(params.dtype_a, DataType::FP16);
    EXPECT_EQ(params.stream, nullptr);
}

//==============================================================================
// GEMM kernel correctness (FP16, small size)
//==============================================================================

class GemmKernelsTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaGetDevice(&device_id_);
    }

    void TearDown() override {
        cudaDeviceSynchronize();
    }

    int device_id_ = 0;
};

TEST_F(GemmKernelsTest, GEMM_FP16_SmallCorrectness) {
    const int M = 64, N = 128, K = 256;
    const int lda = K, ldb = N, ldd = N;

    // Host data (float for CPU reference)
    std::vector<float> A_host(M * K), B_host(K * N), D_ref(M * N, 0.0f);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (int i = 0; i < M * K; ++i) A_host[i] = dist(gen);
    for (int i = 0; i < K * N; ++i) B_host[i] = dist(gen);

    // Convert to half for device
    std::vector<half> A_half(M * K), B_half(K * N), D_half(M * N, __float2half(0.0f));
    for (int i = 0; i < M * K; ++i) A_half[i] = __float2half(A_host[i]);
    for (int i = 0; i < K * N; ++i) B_half[i] = __float2half(B_host[i]);

    // CPU reference
    oasr::kernels::test::cpuGemm(A_host.data(), B_host.data(), D_ref.data(),
                                 M, N, K, lda, ldb, ldd, 1.0f, 0.0f);

    // Device allocation
    half *d_A = nullptr, *d_B = nullptr, *d_D = nullptr;
    OASR_CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(half)));
    OASR_CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(half)));
    OASR_CUDA_CHECK(cudaMalloc(&d_D, M * N * sizeof(half)));

    OASR_CUDA_CHECK(cudaMemcpy(d_A, A_half.data(), M * K * sizeof(half), cudaMemcpyHostToDevice));
    OASR_CUDA_CHECK(cudaMemcpy(d_B, B_half.data(), K * N * sizeof(half), cudaMemcpyHostToDevice));
    OASR_CUDA_CHECK(cudaMemset(d_D, 0, M * N * sizeof(half)));

    GemmParams params;
    params.A = d_A;
    params.B = d_B;
    params.C = nullptr;
    params.D = d_D;
    params.M = M;
    params.N = N;
    params.K = K;
    params.lda = lda;
    params.ldb = ldb;
    params.ldd = ldd;
    params.alpha = 1.0f;
    params.beta = 0.0f;
    params.dtype_a = DataType::FP16;
    params.dtype_b = DataType::FP16;
    params.dtype_d = DataType::FP16;
    params.stream = nullptr;

    GemmStatus status = invokeGemm(params);
    ASSERT_EQ(status, GemmStatus::SUCCESS) << "invokeGemm failed: " << getGemmStatusString(status);

    OASR_CUDA_CHECK(cudaMemcpy(D_half.data(), d_D, M * N * sizeof(half), cudaMemcpyDeviceToHost));

    float max_diff = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float gpu_val = oasr::kernels::test::halfToFloat(D_half[i]);
        float diff = std::abs(gpu_val - D_ref[i]);
        max_diff = std::max(max_diff, diff);
    }
    EXPECT_LT(max_diff, 5e-2f) << "Max difference: " << max_diff;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_D);
}

TEST_F(GemmKernelsTest, BMM_FP16_StridedSmallCorrectness) {
    const int batch = 4, M = 64, N = 64, K = 128;
    const int64_t stride_a = M * K, stride_b = K * N, stride_d = M * N;

    std::vector<float> A_host(batch * M * K), B_host(batch * K * N), D_ref(batch * M * N, 0.0f);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (size_t i = 0; i < A_host.size(); ++i) A_host[i] = dist(gen);
    for (size_t i = 0; i < B_host.size(); ++i) B_host[i] = dist(gen);

    std::vector<half> A_half(batch * M * K), B_half(batch * K * N), D_half(batch * M * N, __float2half(0.0f));
    for (size_t i = 0; i < A_half.size(); ++i) A_half[i] = __float2half(A_host[i]);
    for (size_t i = 0; i < B_half.size(); ++i) B_half[i] = __float2half(B_host[i]);

    for (int b = 0; b < batch; ++b) {
        oasr::kernels::test::cpuGemm(
            A_host.data() + b * M * K, B_host.data() + b * K * N,
            D_ref.data() + b * M * N,
            M, N, K, K, N, N, 1.0f, 0.0f);
    }

    half *d_A = nullptr, *d_B = nullptr, *d_D = nullptr;
    OASR_CUDA_CHECK(cudaMalloc(&d_A, batch * M * K * sizeof(half)));
    OASR_CUDA_CHECK(cudaMalloc(&d_B, batch * K * N * sizeof(half)));
    OASR_CUDA_CHECK(cudaMalloc(&d_D, batch * M * N * sizeof(half)));

    OASR_CUDA_CHECK(cudaMemcpy(d_A, A_half.data(), A_half.size() * sizeof(half), cudaMemcpyHostToDevice));
    OASR_CUDA_CHECK(cudaMemcpy(d_B, B_half.data(), B_half.size() * sizeof(half), cudaMemcpyHostToDevice));
    OASR_CUDA_CHECK(cudaMemset(d_D, 0, batch * M * N * sizeof(half)));

    BmmParams params = BmmParams::Strided(d_A, d_B, d_D, batch, M, N, K, DataType::FP16, nullptr);

    GemmStatus status = invokeBmm(params);
    ASSERT_EQ(status, GemmStatus::SUCCESS) << "invokeBmm failed: " << getGemmStatusString(status);

    OASR_CUDA_CHECK(cudaMemcpy(D_half.data(), d_D, D_half.size() * sizeof(half), cudaMemcpyDeviceToHost));

    float max_diff = 0.0f;
    for (size_t i = 0; i < D_ref.size(); ++i) {
        float gpu_val = oasr::kernels::test::halfToFloat(D_half[i]);
        float diff = std::abs(gpu_val - D_ref[i]);
        max_diff = std::max(max_diff, diff);
    }
    EXPECT_LT(max_diff, 5e-2f) << "Max difference: " << max_diff;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_D);
}
