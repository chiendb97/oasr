// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// GEMM dispatch interface — public API for GEMM and GemmActivation.
//
// Supports two dispatch modes:
//   - JIT mode: OASR_TARGET_SM defined, single SM instantiation.
//               Optional OASR_GEMM_TILE_M etc. for custom tile config.
//   - AOT mode: Full runtime dispatch over compiled SM versions.

#pragma once

#include <cstdint>

#include "oasr/common/arch_dispatch.h"
#include "oasr/common/types.h"
#include "oasr/gemm/cutlass_gemm_configs.h"
#include "oasr/gemm/gemm_cutlass_template.h"
#include "oasr/gemm/gemm_cutlass_template_sm90.h"

namespace oasr {
namespace gemm {

//==============================================================================
// Internal dispatch helpers
//==============================================================================

namespace detail {

template <int SM_VERSION, typename ElementA, typename ElementB,
          typename ElementCD, ActivationType activation_type>
static GemmStatus dispatchGemmWithSmVersion(const ElementA* A_ptr, const ElementB* B_ptr,
                                          const ElementCD* C_ptr, ElementCD* D_ptr, int M, int N,
                                          int K, uint64_t lda, uint64_t ldb, uint64_t ldc,
                                          float alpha, cudaStream_t stream, int split_k_slices = 1) {
    if constexpr (SM_VERSION == 75) {
        using Config = CutlassGemmConfig<16, 128, 64, 16, 32, 64, 3, 75>;
        return CutlassGemmKernel<Config, ElementA, ElementB, ElementCD, activation_type>::run(A_ptr, B_ptr, C_ptr, D_ptr, M, N, K, lda, ldb, ldc, alpha, stream, split_k_slices);
    } else if constexpr (SM_VERSION == 80) {
        using Config = CutlassGemmConfig<16, 128, 64, 16, 32, 64, 3, 80>;
        return CutlassGemmKernel<Config, ElementA, ElementB, ElementCD, activation_type>::run(A_ptr, B_ptr, C_ptr, D_ptr, M, N, K, lda, ldb, ldc, alpha, stream, split_k_slices);
    } else if constexpr (SM_VERSION == 86) {
        using Config = CutlassGemmConfig<16, 128, 64, 16, 32, 64, 3, 86>;
        return CutlassGemmKernel<Config, ElementA, ElementB, ElementCD, activation_type>::run(A_ptr, B_ptr, C_ptr, D_ptr, M, N, K, lda, ldb, ldc, alpha, stream, split_k_slices);
    } else if constexpr (SM_VERSION == 89) {
        using Config = CutlassGemmConfig<16, 128, 64, 16, 32, 64, 3, 89>;
        return CutlassGemmKernel<Config, ElementA, ElementB, ElementCD, activation_type>::run(A_ptr, B_ptr, C_ptr, D_ptr, M, N, K, lda, ldb, ldc, alpha, stream, split_k_slices);
    } else if constexpr (SM_VERSION == 90) {
        using Config = CutlassGemmConfigSm90<64, 16, 128, 1, 1, 1, 1, 3, 90>;
        return CutlassGemmKernelSm90<Config, ElementA, ElementB, ElementCD, activation_type>::run(A_ptr, B_ptr, C_ptr, D_ptr, M, N, K, lda, ldb, ldc, alpha, stream, split_k_slices);
    } else if constexpr (SM_VERSION == 100) {
        using Config = CutlassGemmConfigSm90<64, 16, 128, 1, 1, 1, 1, 3, 100>;
        return CutlassGemmKernelSm90<Config, ElementA, ElementB, ElementCD, activation_type>::run(A_ptr, B_ptr, C_ptr, D_ptr, M, N, K, lda, ldb, ldc, alpha, stream, split_k_slices);
    } else if constexpr (SM_VERSION == 120) {
        using Config = CutlassGemmConfigSm90<64, 16, 128, 1, 1, 1, 1, 3, 120>;
        return CutlassGemmKernelSm90<Config, ElementA, ElementB, ElementCD, activation_type>::run(A_ptr, B_ptr, C_ptr, D_ptr, M, N, K, lda, ldb, ldc, alpha, stream, split_k_slices);
    } else {
        return GemmStatus::INVALID_ARGUMENT;
    }
}

}  // namespace detail

//==============================================================================
// Gemm — public API
//==============================================================================

/**
 * @brief Execute GEMM operation: D = alpha * A @ B + beta * C
 *
 * When C is nullptr, computes D = alpha * A @ B.
 * Layout: A is [M, K] row-major, B is [N, K] column-major, C/D are [M, N] row-major.
 */
template <typename ElementA, typename ElementB, typename ElementCD>
cudaError_t Gemm(const ElementA* A, const ElementB* B, const ElementCD* C, ElementCD* D, int M,
                 int N, int K, cudaStream_t stream, int split_k_slices = 1) {
    if (A == nullptr || B == nullptr || D == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (M <= 0 || N <= 0 || K <= 0) {
        return cudaErrorInvalidValue;
    }

    float alpha = 1.0f;
    uint64_t lda = K;
    uint64_t ldb = K;
    uint64_t ldc = N;

    GemmStatus status = GemmStatus::SUCCESS;

#ifdef OASR_TARGET_SM
    {
        constexpr int SM_VERSION = OASR_TARGET_SM;
        status = detail::dispatchGemmWithSmVersion<SM_VERSION, ElementA, ElementB, ElementCD,
                                                    ActivationType::IDENTITY>(
            A, B, C, D, M, N, K, lda, ldb, ldc, alpha, stream, split_k_slices);
    }
#else
    int sm = oasr::getDeviceSmVersion();
    OASR_DISPATCH_SM(sm, SM_VERSION, {
        status = detail::dispatchGemmWithSmVersion<SM_VERSION, ElementA, ElementB, ElementCD,
                                                    ActivationType::IDENTITY>(
            A, B, C, D, M, N, K, lda, ldb, ldc, alpha, stream, split_k_slices);
    });
#endif

    if (status != GemmStatus::SUCCESS) {
        return cudaErrorUnknown;
    }
    return cudaSuccess;
}

//==============================================================================
// GemmActivation — public API
//==============================================================================

/**
 * @brief Execute GEMM with fused activation: D = activation(alpha * A @ B + beta * C)
 */
template <typename ElementA, typename ElementB, typename ElementCD>
cudaError_t GemmActivation(const ElementA* A, const ElementB* B, const ElementCD* C, ElementCD* D,
                           int M, int N, int K, ActivationType activation, cudaStream_t stream,
                           int split_k_slices = 1) {
    if (A == nullptr || B == nullptr || D == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (M <= 0 || N <= 0 || K <= 0) {
        return cudaErrorInvalidValue;
    }

    float alpha = 1.0f;
    uint64_t lda = K;
    uint64_t ldb = K;
    uint64_t ldc = N;

    GemmStatus status = GemmStatus::SUCCESS;

#ifdef OASR_TARGET_SM
    {
        constexpr int SM_VERSION = OASR_TARGET_SM;
        status = detail::dispatchGemmWithSmVersion<SM_VERSION, ElementA, ElementB, ElementCD,
                                                    activation>(
                A, B, C, D, M, N, K, lda, ldb, ldc, alpha, stream, split_k_slices);
        }
    }
#else
    int sm = oasr::getDeviceSmVersion();
    OASR_DISPATCH_SM(sm, SM_VERSION, {
        status = detail::dispatchGemmWithSmVersion<SM_VERSION, ElementA, ElementB, ElementCD,
                                                    activation>(
                A, B, C, D, M, N, K, lda, ldb, ldc, alpha, stream, split_k_slices);
    });
#endif

    if (status != GemmStatus::SUCCESS) {
        return cudaErrorUnknown;
    }
    return cudaSuccess;
}

}  // namespace gemm
}  // namespace oasr
