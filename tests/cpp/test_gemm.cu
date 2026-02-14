// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for GEMM, Batched GEMM (BMM), and Grouped GEMM layers.

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <vector>
#include <random>
#include <string>

#include "kernels/gemm/gemm_kernels.h"
#include "kernels/gemm/gemm_utils.h"
#include "kernels/gemm/bmm_kernels.h"
#include "kernels/gemm/group_gemm_kernels.h"
#include "common/cuda_utils.h"

using namespace oasr;
using namespace oasr::kernels::gemm;

namespace oasr {
namespace kernels {
namespace test {

// CPU reference GEMM (float, row-major): D = alpha * A @ B + beta * C
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

inline float halfToFloat(half h) {
    return __half2float(h);
}

}  // namespace test
}  // namespace kernels
}  // namespace oasr

//==============================================================================
// Smoke tests (module and enums exist)
//==============================================================================

class TestGemmModule : public ::testing::Test {};

TEST_F(TestGemmModule, GemmStatusEnum) {
    EXPECT_EQ(static_cast<int>(GemmStatus::SUCCESS), 0);
    ASSERT_NE(getGemmStatusString(GemmStatus::SUCCESS), nullptr);
}

TEST_F(TestGemmModule, GetGemmStatusString) {
    const char* s = getGemmStatusString(GemmStatus::SUCCESS);
    ASSERT_NE(s, nullptr);
    std::string str(s);
    EXPECT_GT(str.size(), 0u);
}

//==============================================================================
// Single GEMM: D = alpha * A @ B + beta * C
//==============================================================================

class TestGemm : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        ASSERT_GT(device_count, 0) << "No CUDA devices available";
    }
    void TearDown() override {
        cudaDeviceSynchronize();
    }
    void RunGemmFp16(int M, int N, int K, int lda, int ldb, int ldd, float alpha, float beta);
};

TEST_F(TestGemm, GemmFp16_64_128_256) {
    const int M = 64, N = 128, K = 256;
    const int lda = K, ldb = N, ldd = N;
    RunGemmFp16(M, N, K, lda, ldb, ldd, 1.0f, 0.0f);
}

TEST_F(TestGemm, GemmFp16_32_32_32) {
    const int M = 32, N = 32, K = 32;
    const int lda = K, ldb = N, ldd = N;
    RunGemmFp16(M, N, K, lda, ldb, ldd, 1.0f, 0.0f);
}

TEST_F(TestGemm, GemmFp16_256_32_128) {
    const int M = 256, N = 32, K = 128;
    const int lda = K, ldb = N, ldd = N;
    RunGemmFp16(M, N, K, lda, ldb, ldd, 1.0f, 0.0f);
}

TEST_F(TestGemm, GemmAlphaBeta) {
    const int M = 32, N = 64, K = 48;
    const int lda = K, ldb = N, ldd = N;
    RunGemmFp16(M, N, K, lda, ldb, ldd, 2.0f, 0.0f);
}

void TestGemm::RunGemmFp16(int M, int N, int K, int lda, int ldb, int ldd,
                           float alpha, float beta) {
    std::vector<float> A_host(M * K), B_host(K * N), D_ref(M * N, 0.0f);
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (int i = 0; i < M * K; ++i) A_host[i] = dist(gen);
    for (int i = 0; i < K * N; ++i) B_host[i] = dist(gen);

    oasr::kernels::test::cpuGemm(A_host.data(), B_host.data(), D_ref.data(),
                                 M, N, K, lda, ldb, ldd, alpha, beta);

    std::vector<half> A_half(M * K), B_half(K * N), D_half(M * N, __float2half(0.0f));
    for (int i = 0; i < M * K; ++i) A_half[i] = __float2half(A_host[i]);
    for (int i = 0; i < K * N; ++i) B_half[i] = __float2half(B_host[i]);

    half *d_A = nullptr, *d_B = nullptr, *d_D = nullptr;
    OASR_CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(half)));
    OASR_CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(half)));
    OASR_CUDA_CHECK(cudaMalloc(&d_D, M * N * sizeof(half)));

    OASR_CUDA_CHECK(cudaMemcpy(d_A, A_half.data(), M * K * sizeof(half), cudaMemcpyHostToDevice));
    OASR_CUDA_CHECK(cudaMemcpy(d_B, B_half.data(), K * N * sizeof(half), cudaMemcpyHostToDevice));
    OASR_CUDA_CHECK(cudaMemset(d_D, 0, M * N * sizeof(half)));

    GemmStatus status = invokeGemm(
        d_A, d_B, d_D, M, N, K, lda, ldb, ldd,
        alpha, beta,
        TransposeOp::NoTranspose, TransposeOp::NoTranspose,
        DataType::FP16, nullptr);
    ASSERT_EQ(status, GemmStatus::SUCCESS)
        << "invokeGemm failed: " << getGemmStatusString(status);

    OASR_CUDA_CHECK(cudaMemcpy(D_half.data(), d_D, M * N * sizeof(half), cudaMemcpyDeviceToHost));

    const float rtol = 1e-2f;
    const float atol = 1e-2f;
    for (int i = 0; i < M * N; ++i) {
        float gpu_val = oasr::kernels::test::halfToFloat(D_half[i]);
        float diff = std::abs(gpu_val - D_ref[i]);
        EXPECT_LE(diff, atol + rtol * std::abs(D_ref[i]))
            << "at index " << i << " gpu=" << gpu_val << " ref=" << D_ref[i];
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_D);
}

//==============================================================================
// Batched GEMM (strided): D[b] = A[b] @ B[b]
//==============================================================================

class TestBmm : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        ASSERT_GT(device_count, 0) << "No CUDA devices available";
    }
    void TearDown() override {
        cudaDeviceSynchronize();
    }
    void RunBmmFp16(int batch_size, int M, int N, int K);
};

TEST_F(TestBmm, BmmFp16_4_64_64_128) {
    RunBmmFp16(4, 64, 64, 128);
}

TEST_F(TestBmm, BmmFp16_2_32_32_32) {
    RunBmmFp16(2, 32, 32, 32);
}

void TestBmm::RunBmmFp16(int batch_size, int M, int N, int K) {
    std::vector<float> A_host(batch_size * M * K), B_host(batch_size * K * N),
        D_ref(batch_size * M * N, 0.0f);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (size_t i = 0; i < A_host.size(); ++i) A_host[i] = dist(gen);
    for (size_t i = 0; i < B_host.size(); ++i) B_host[i] = dist(gen);

    for (int b = 0; b < batch_size; ++b) {
        oasr::kernels::test::cpuGemm(
            A_host.data() + b * M * K, B_host.data() + b * K * N,
            D_ref.data() + b * M * N,
            M, N, K, K, N, N, 1.0f, 0.0f);
    }

    std::vector<half> A_half(A_host.size()), B_half(B_host.size()),
        D_half(D_ref.size(), __float2half(0.0f));
    for (size_t i = 0; i < A_host.size(); ++i) A_half[i] = __float2half(A_host[i]);
    for (size_t i = 0; i < B_host.size(); ++i) B_half[i] = __float2half(B_host[i]);

    half *d_A = nullptr, *d_B = nullptr, *d_D = nullptr;
    OASR_CUDA_CHECK(cudaMalloc(&d_A, A_half.size() * sizeof(half)));
    OASR_CUDA_CHECK(cudaMalloc(&d_B, B_half.size() * sizeof(half)));
    OASR_CUDA_CHECK(cudaMalloc(&d_D, D_half.size() * sizeof(half)));

    OASR_CUDA_CHECK(cudaMemcpy(d_A, A_half.data(), A_half.size() * sizeof(half), cudaMemcpyHostToDevice));
    OASR_CUDA_CHECK(cudaMemcpy(d_B, B_half.data(), B_half.size() * sizeof(half), cudaMemcpyHostToDevice));
    OASR_CUDA_CHECK(cudaMemset(d_D, 0, D_half.size() * sizeof(half)));

    int64_t stride_a = static_cast<int64_t>(M) * K;
    int64_t stride_b = static_cast<int64_t>(K) * N;
    int64_t stride_d = static_cast<int64_t>(M) * N;

    GemmStatus status = invokeBmm(
        d_A, d_B, d_D, batch_size, M, N, K,
        K, N, N, stride_a, stride_b, stride_d,
        1.0f, 0.0f,
        TransposeOp::NoTranspose, TransposeOp::NoTranspose,
        DataType::FP16, nullptr);
    ASSERT_EQ(status, GemmStatus::SUCCESS)
        << "invokeBmm failed: " << getGemmStatusString(status);

    OASR_CUDA_CHECK(cudaMemcpy(D_half.data(), d_D, D_half.size() * sizeof(half), cudaMemcpyDeviceToHost));

    const float rtol = 1e-2f;
    const float atol = 1e-2f;
    for (size_t i = 0; i < D_ref.size(); ++i) {
        float gpu_val = oasr::kernels::test::halfToFloat(D_half[i]);
        float diff = std::abs(gpu_val - D_ref[i]);
        EXPECT_LE(diff, atol + rtol * std::abs(D_ref[i]))
            << "at index " << i << " gpu=" << gpu_val << " ref=" << D_ref[i];
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_D);
}

//==============================================================================
// Grouped GEMM (variable-sized problems)
//==============================================================================

class TestGroupGemm : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        ASSERT_GT(device_count, 0) << "No CUDA devices available";
    }
    void TearDown() override {
        cudaDeviceSynchronize();
    }
    void RunGroupGemmFp16(const std::vector<std::tuple<int, int, int>>& problem_sizes);
};

void TestGroupGemm::RunGroupGemmFp16(const std::vector<std::tuple<int, int, int>>& problem_sizes) {
    const int num_problems = static_cast<int>(problem_sizes.size());
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    std::vector<GemmProblemDesc> problems_host(num_problems);
    std::vector<int64_t> lda_host(num_problems), ldb_host(num_problems), ldd_host(num_problems);
    std::vector<const half*> A_ptrs_host(num_problems);
    std::vector<const half*> B_ptrs_host(num_problems);
    std::vector<half*> D_ptrs_host(num_problems);

    std::vector<half> A_half, B_half, D_half;
    std::vector<float> D_ref_flat;

    for (int i = 0; i < num_problems; ++i) {
        int M = std::get<0>(problem_sizes[i]);
        int N = std::get<1>(problem_sizes[i]);
        int K = std::get<2>(problem_sizes[i]);
        problems_host[i] = GemmProblemDesc(M, N, K);
        lda_host[i] = K;
        ldb_host[i] = N;
        ldd_host[i] = N;

        size_t a_off = A_half.size();
        size_t b_off = B_half.size();
        size_t d_off = D_half.size();
        A_half.resize(a_off + M * K);
        B_half.resize(b_off + K * N);
        D_half.resize(d_off + M * N);
        for (int j = 0; j < M * K; ++j) A_half[a_off + j] = __float2half(dist(gen));
        for (int j = 0; j < K * N; ++j) B_half[b_off + j] = __float2half(dist(gen));
        for (int j = 0; j < M * N; ++j) D_half[d_off + j] = __float2half(0.0f);

        std::vector<float> A_f(M * K), B_f(K * N), D_f(M * N, 0.0f);
        for (int j = 0; j < M * K; ++j) A_f[j] = __half2float(A_half[a_off + j]);
        for (int j = 0; j < K * N; ++j) B_f[j] = __half2float(B_half[b_off + j]);
        oasr::kernels::test::cpuGemm(A_f.data(), B_f.data(), D_f.data(), M, N, K, K, N, N, 1.0f, 0.0f);
        for (float v : D_f) D_ref_flat.push_back(v);
    }

    std::vector<void*> d_A_ptrs(num_problems), d_B_ptrs(num_problems), d_D_ptrs(num_problems);
    size_t a_offset = 0, b_offset = 0, d_offset = 0;
    for (int i = 0; i < num_problems; ++i) {
        int M = std::get<0>(problem_sizes[i]);
        int N = std::get<1>(problem_sizes[i]);
        int K = std::get<2>(problem_sizes[i]);
        half *d_A = nullptr, *d_B = nullptr, *d_D = nullptr;
        OASR_CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(half)));
        OASR_CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(half)));
        OASR_CUDA_CHECK(cudaMalloc(&d_D, M * N * sizeof(half)));
        OASR_CUDA_CHECK(cudaMemcpy(d_A, &A_half[a_offset], M * K * sizeof(half), cudaMemcpyHostToDevice));
        OASR_CUDA_CHECK(cudaMemcpy(d_B, &B_half[b_offset], K * N * sizeof(half), cudaMemcpyHostToDevice));
        OASR_CUDA_CHECK(cudaMemset(d_D, 0, M * N * sizeof(half)));
        d_A_ptrs[i] = d_A;
        d_B_ptrs[i] = d_B;
        d_D_ptrs[i] = d_D;
        a_offset += M * K;
        b_offset += K * N;
        d_offset += M * N;
    }

    GemmProblemDesc* d_problems = nullptr;
    int64_t* d_lda = nullptr, *d_ldb = nullptr, *d_ldd = nullptr;
    void** d_A_array = nullptr, ** d_B_array = nullptr, ** d_D_array = nullptr;
    OASR_CUDA_CHECK(cudaMalloc(&d_problems, num_problems * sizeof(GemmProblemDesc)));
    OASR_CUDA_CHECK(cudaMalloc(&d_lda, num_problems * sizeof(int64_t)));
    OASR_CUDA_CHECK(cudaMalloc(&d_ldb, num_problems * sizeof(int64_t)));
    OASR_CUDA_CHECK(cudaMalloc(&d_ldd, num_problems * sizeof(int64_t)));
    OASR_CUDA_CHECK(cudaMalloc(&d_A_array, num_problems * sizeof(void*)));
    OASR_CUDA_CHECK(cudaMalloc(&d_B_array, num_problems * sizeof(void*)));
    OASR_CUDA_CHECK(cudaMalloc(&d_D_array, num_problems * sizeof(void*)));

    OASR_CUDA_CHECK(cudaMemcpy(d_problems, problems_host.data(), num_problems * sizeof(GemmProblemDesc), cudaMemcpyHostToDevice));
    OASR_CUDA_CHECK(cudaMemcpy(d_lda, lda_host.data(), num_problems * sizeof(int64_t), cudaMemcpyHostToDevice));
    OASR_CUDA_CHECK(cudaMemcpy(d_ldb, ldb_host.data(), num_problems * sizeof(int64_t), cudaMemcpyHostToDevice));
    OASR_CUDA_CHECK(cudaMemcpy(d_ldd, ldd_host.data(), num_problems * sizeof(int64_t), cudaMemcpyHostToDevice));
    OASR_CUDA_CHECK(cudaMemcpy(d_A_array, d_A_ptrs.data(), num_problems * sizeof(void*), cudaMemcpyHostToDevice));
    OASR_CUDA_CHECK(cudaMemcpy(d_B_array, d_B_ptrs.data(), num_problems * sizeof(void*), cudaMemcpyHostToDevice));
    OASR_CUDA_CHECK(cudaMemcpy(d_D_array, d_D_ptrs.data(), num_problems * sizeof(void*), cudaMemcpyHostToDevice));

    size_t float_ws = 0, int_ws = 0;
    queryGroupGemmWorkspaceSize(num_problems, nullptr, DataType::FP16, float_ws, int_ws);
    void* workspace_float = nullptr;
    void* workspace_int = nullptr;
    if (float_ws > 0) OASR_CUDA_CHECK(cudaMalloc(&workspace_float, float_ws));
    if (int_ws > 0) OASR_CUDA_CHECK(cudaMalloc(&workspace_int, int_ws));

    GemmStatus status = invokeGroupGemm(
        d_problems, num_problems,
        reinterpret_cast<const void* const*>(d_A_array),
        reinterpret_cast<const void* const*>(d_B_array),
        reinterpret_cast<void* const*>(d_D_array),
        d_lda, d_ldb, d_ldd,
        DataType::FP16,
        workspace_float, float_ws,
        nullptr);
    ASSERT_EQ(status, GemmStatus::SUCCESS)
        << "invokeGroupGemm failed: " << getGemmStatusString(status);

    const float rtol = 1e-2f;
    const float atol = 1e-2f;
    size_t idx = 0;
    for (int i = 0; i < num_problems; ++i) {
        int M = std::get<0>(problem_sizes[i]);
        int N = std::get<1>(problem_sizes[i]);
        std::vector<half> out(M * N);
        OASR_CUDA_CHECK(cudaMemcpy(out.data(), d_D_ptrs[i], M * N * sizeof(half), cudaMemcpyDeviceToHost));
        for (int j = 0; j < M * N; ++j) {
            float gpu_val = oasr::kernels::test::halfToFloat(out[j]);
            float ref_val = D_ref_flat[idx++];
            float diff = std::abs(gpu_val - ref_val);
            EXPECT_LE(diff, atol + rtol * std::abs(ref_val))
                << "problem " << i << " index " << j;
        }
    }

    for (int i = 0; i < num_problems; ++i) {
        cudaFree(d_A_ptrs[i]);
        cudaFree(d_B_ptrs[i]);
        cudaFree(d_D_ptrs[i]);
    }
    cudaFree(d_problems);
    cudaFree(d_lda);
    cudaFree(d_ldb);
    cudaFree(d_ldd);
    cudaFree(d_A_array);
    cudaFree(d_B_array);
    cudaFree(d_D_array);
    if (workspace_float) cudaFree(workspace_float);
    if (workspace_int) cudaFree(workspace_int);
}

TEST_F(TestGroupGemm, GroupGemmSingleProblem) {
    RunGroupGemmFp16({{64, 128, 256}});
}

TEST_F(TestGroupGemm, GroupGemmMultipleSameSize) {
    RunGroupGemmFp16({{32, 64, 64}, {32, 64, 64}, {32, 64, 64}, {32, 64, 64}});
}

TEST_F(TestGroupGemm, GroupGemmVariableSizes) {
    RunGroupGemmFp16({{32, 64, 48}, {64, 32, 96}, {16, 128, 64}});
}

//==============================================================================
// Helper APIs (workspace, status, config)
//==============================================================================

class TestGemmHelpers : public ::testing::Test {};

TEST_F(TestGemmHelpers, QueryGemmWorkspaceSize) {
    size_t size = queryGemmWorkspaceSize(128, 256, 512, DataType::FP16);
    EXPECT_GE(size, 0u);
}

TEST_F(TestGemmHelpers, QueryBmmWorkspaceSize) {
    size_t size = queryBmmWorkspaceSize(8, 64, 64, 128, DataType::FP16);
    EXPECT_GE(size, 0u);
}

TEST_F(TestGemmHelpers, QueryGroupGemmWorkspaceSize) {
    size_t float_ws = 0, int_ws = 0;
    queryGroupGemmWorkspaceSize(4, nullptr, DataType::FP16, float_ws, int_ws);
    EXPECT_GE(float_ws, 0u);
    EXPECT_GE(int_ws, 0u);
}

TEST_F(TestGemmHelpers, GetGemmStatusStringContainsSuccess) {
    const char* s = getGemmStatusString(GemmStatus::SUCCESS);
    ASSERT_NE(s, nullptr);
    EXPECT_TRUE(std::string(s).find("SUCCESS") != std::string::npos);
}

TEST_F(TestGemmHelpers, GetSmVersion) {
    int sm = getSMVersion(-1);
    EXPECT_GE(sm, 0);
}

//==============================================================================
// API consistency: invoke functions accept direct parameters
//==============================================================================

class GemmApiTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        ASSERT_GT(device_count, 0) << "No CUDA devices available";
    }
};

TEST_F(GemmApiTest, InvokeGemmAcceptsDirectParams) {
    // Smoke test: invoke_gemm accepts all parameters directly
    void* dummy = nullptr;
    GemmStatus s = invokeGemm(
        dummy, dummy, dummy, 1, 1, 1,
        1, 1, 1, 1.0f, 0.0f,
        TransposeOp::NoTranspose, TransposeOp::NoTranspose,
        DataType::FP16, nullptr);
    EXPECT_TRUE(s == GemmStatus::INVALID_ARGUMENT || s == GemmStatus::SUCCESS);
}

TEST_F(GemmApiTest, InvokeBmmAcceptsDirectParams) {
    void* dummy = nullptr;
    GemmStatus s = invokeBmm(
        dummy, dummy, dummy, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1,
        1.0f, 0.0f,
        TransposeOp::NoTranspose, TransposeOp::NoTranspose,
        DataType::FP16, nullptr);
    EXPECT_TRUE(s == GemmStatus::INVALID_ARGUMENT || s == GemmStatus::SUCCESS);
}
