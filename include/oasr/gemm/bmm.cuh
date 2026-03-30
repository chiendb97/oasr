// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Batched Matrix Multiplication (BMM) kernel -- pure CUDA + CUTLASS.
// Uses GemmConfig + SmMMATraits dispatch (FlashInfer-style).

#pragma once

#include <cstdint>

#ifdef __GNUC__
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wstrict-aliasing"
    #pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/device_memory.h>


#ifdef __GNUC__
    #pragma GCC diagnostic pop
#endif

#include <oasr/common/arch_dispatch.h>
#include <oasr/common/utils.h>
#include <oasr/gemm/cutlass_gemm_configs.h>
#include <oasr/gemm/gemm_cutlass_template.h>

namespace oasr {
namespace gemm {


//==============================================================================
// Dispatch helpers
//==============================================================================

namespace detail {

template <int SM_VERSION, typename ElementA, typename ElementB,
          typename ElementCD>
static GemmStatus dispatchBmmWithSmVersion(const ElementA* A_ptr, const ElementB* B_ptr,
                                         ElementCD* D_ptr, int batch_size, int M, int N, int K,
                                         uint64_t lda, uint64_t ldb, uint64_t ldd,
                                         int64_t stride_a, int64_t stride_b, int64_t stride_d,
                                         float alpha, float beta, cudaStream_t stream) {

    if constexpr (SM_VERSION == 75) {
        using Config = CutlassGemmConfig<16, 128, 64, 16, 32, 64, 3, 75>;
        return CutlassBmmKernel<Config, ElementA, ElementB, ElementCD>::run(A_ptr, B_ptr, D_ptr, batch_size, M, N, K, lda, ldb, ldd, stride_a, stride_b, stride_d, alpha, beta, stream);
    } else if constexpr (SM_VERSION == 80) {
        using Config = CutlassGemmConfig<16, 128, 64, 16, 32, 64, 3, 80>;
        return CutlassBmmKernel<Config, ElementA, ElementB, ElementCD>::run(A_ptr, B_ptr, D_ptr, batch_size, M, N, K, lda, ldb, ldd, stride_a, stride_b, stride_d, alpha, beta, stream);
    } else if constexpr (SM_VERSION == 86) {
        using Config = CutlassGemmConfig<16, 128, 64, 16, 32, 64, 3, 86>;
        return CutlassBmmKernel<Config, ElementA, ElementB, ElementCD>::run(A_ptr, B_ptr, D_ptr, batch_size, M, N, K, lda, ldb, ldd, stride_a, stride_b, stride_d, alpha, beta, stream);
    } else if constexpr (SM_VERSION == 90) {
        using Config = CutlassGemmConfigSm90<64, 16, 128, 1, 1, 1, 1, 3, 90>;
        return CutlassBmmKernel<Config, ElementA, ElementB, ElementCD>::run(A_ptr, B_ptr, D_ptr, batch_size, M, N, K, lda, ldb, ldd, stride_a, stride_b, stride_d, alpha, beta, stream);
    } else if constexpr (SM_VERSION == 100) {
        using Config = CutlassGemmConfigSm90<64, 16, 128, 1, 1, 1, 1, 3, 100>;
        return CutlassBmmKernel<Config, ElementA, ElementB, ElementCD>::run(A_ptr, B_ptr, D_ptr, batch_size, M, N, K, lda, ldb, ldd, stride_a, stride_b, stride_d, alpha, beta, stream);
    } else if constexpr (SM_VERSION == 120) {
        using Config = CutlassGemmConfigSm90<64, 16, 128, 1, 1, 1, 1, 3, 120>;
        return CutlassBmmKernel<Config, ElementA, ElementB, ElementCD>::run(A_ptr, B_ptr, D_ptr, batch_size, M, N, K, lda, ldb, ldd, stride_a, stride_b, stride_d, alpha, beta, stream);
    } else {
        return GemmStatus::INVALID_ARGUMENT;
    }
}

}  // namespace detail

//==============================================================================
// Typed Launcher: Bmm
//==============================================================================

/**
 * @brief Execute strided batched GEMM: D[b] = alpha * A[b] @ B[b] + beta * D[b]
 *
 * Layout: A [batch, M, K] row-major, B [batch, N, K] col-major, D [batch, M, N] row-major.
 */
template <typename ElementA, typename ElementB, typename ElementCD>
cudaError_t Bmm(const ElementA* A, const ElementB* B, ElementCD* D, int batch_size, int M, int N,
                int K, cudaStream_t stream) {
    if (A == nullptr || B == nullptr || D == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (batch_size <= 0 || M <= 0 || N <= 0 || K <= 0) {
        return cudaErrorInvalidValue;
    }

    float alpha = 1.0f;
    float beta = 0.0f;

    uint64_t lda = K;
    uint64_t ldb = K;
    uint64_t ldd = N;

    int64_t stride_a = static_cast<int64_t>(M) * K;
    int64_t stride_b = static_cast<int64_t>(N) * K;
    int64_t stride_d = static_cast<int64_t>(M) * N;

    return Bmm(A, B, D, batch_size, M, N, K, lda, ldb, ldd, stride_a, stride_b, stride_d, alpha, beta, stream);
}

/**
 * @brief Execute strided batched GEMM with full control over strides and scaling.
 */
template <typename ElementA, typename ElementB, typename ElementCD>
cudaError_t Bmm(const ElementA* A, const ElementB* B, ElementCD* D, int batch_size, int M, int N,
                int K, int64_t lda, int64_t ldb, int64_t ldd, int64_t stride_a, int64_t stride_b,
                int64_t stride_d, float alpha, float beta, cudaStream_t stream) {
    if (A == nullptr || B == nullptr || D == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (batch_size <= 0 || M <= 0 || N <= 0 || K <= 0) {
        return cudaErrorInvalidValue;
    }

    GemmStatus status = GemmStatus::SUCCESS;

#ifdef OASR_TARGET_SM
    {
        constexpr int SM_VERSION = OASR_TARGET_SM;
        status = detail::dispatchBmmWithSmVersion<SM_VERSION, ElementA, ElementB, ElementCD>(A, B, D, batch_size, M, N, K, lda, ldb, ldd, stride_a, stride_b, stride_d, alpha, beta, stream);
    }
#else
    int sm = oasr::getDeviceSmVersion();
    OASR_DISPATCH_SM(sm, SM_VERSION, {
        status = detail::dispatchBmmWithSmVersion<SM_VERSION, ElementA, ElementB, ElementCD>(A, B, D, batch_size, M, N, K, lda, ldb, ldd, stride_a, stride_b, stride_d, alpha, beta, stream);
    });
#endif

    if (status != GemmStatus::SUCCESS) {
        return cudaErrorUnknown;
    }
    return cudaSuccess;
}

}  // namespace gemm
}  // namespace oasr
