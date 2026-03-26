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

#include <oasr/common/arch_dispatch.h>
#include <oasr/common/epilogue_functors.h>
#include <oasr/common/types.h>
#include <oasr/gemm/cutlass_gemm_configs.h>
#include <oasr/gemm/gemm_cutlass_template.h>
#include <oasr/gemm/gemm_utils.h>

namespace oasr {
namespace gemm {

//==============================================================================
// Internal dispatch helpers
//==============================================================================

namespace detail {

template <typename Config, typename MMATraits, typename ElementA, typename ElementB,
          typename ElementCD, template <int, typename, typename> class EpilogueFn>
static GemmStatus dispatchGemmWithConfig(const ElementA* A_ptr, const ElementB* B_ptr,
                                          const ElementCD* C_ptr, ElementCD* D_ptr, int M, int N,
                                          int K, uint64_t lda, uint64_t ldb, uint64_t ldc,
                                          float alpha, cudaStream_t stream,
                                          int split_k_slices = 1) {
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;

    return CutlassGemmKernel<Config, MMATraits, ElementA, ElementB, ElementCD, LayoutA, LayoutB,
                              LayoutC, EpilogueFn>::run(A_ptr, B_ptr, C_ptr, D_ptr, M, N, K, lda,
                                                        ldb, ldc, alpha, stream, split_k_slices);
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
#ifdef OASR_GEMM_TILE_M
        using Config = JitGemmConfig;
#else
        using Config = typename DefaultGemmConfig<SM_VERSION>::type;
#endif
        using MMA = SmMMATraits<SM_VERSION>;
        status = detail::dispatchGemmWithConfig<Config, MMA, ElementA, ElementB, ElementCD,
                                                 oasr::EpilogueIdentity>(
            A, B, C, D, M, N, K, lda, ldb, ldc, alpha, stream, split_k_slices);
    }
#else
    int sm = oasr::getDeviceSmVersion();
    OASR_DISPATCH_SM(sm, SM_VERSION, {
        using Config = typename DefaultGemmConfig<SM_VERSION>::type;
        using MMA = SmMMATraits<SM_VERSION>;
        status = detail::dispatchGemmWithConfig<Config, MMA, ElementA, ElementB, ElementCD,
                                                 oasr::EpilogueIdentity>(
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
#ifdef OASR_GEMM_TILE_M
        using Config = JitGemmConfig;
#else
        using Config = typename DefaultGemmConfig<SM_VERSION>::type;
#endif
        using MMA = SmMMATraits<SM_VERSION>;
        if (activation == ActivationType::RELU) {
            status = detail::dispatchGemmWithConfig<Config, MMA, ElementA, ElementB, ElementCD,
                                                     oasr::EpilogueRelu>(
                A, B, C, D, M, N, K, lda, ldb, ldc, alpha, stream, split_k_slices);
        } else if (activation == ActivationType::GELU) {
            status = detail::dispatchGemmWithConfig<Config, MMA, ElementA, ElementB, ElementCD,
                                                     oasr::EpilogueGelu>(
                A, B, C, D, M, N, K, lda, ldb, ldc, alpha, stream, split_k_slices);
        } else if (activation == ActivationType::SWISH) {
            status = detail::dispatchGemmWithConfig<Config, MMA, ElementA, ElementB, ElementCD,
                                                     oasr::EpilogueSwish>(
                A, B, C, D, M, N, K, lda, ldb, ldc, alpha, stream, split_k_slices);
        } else {
            status = GemmStatus::INVALID_ARGUMENT;
        }
    }
#else
    int sm = oasr::getDeviceSmVersion();
    OASR_DISPATCH_SM(sm, SM_VERSION, {
        using Config = typename DefaultGemmConfig<SM_VERSION>::type;
        using MMA = SmMMATraits<SM_VERSION>;
        if (activation == ActivationType::RELU) {
            status = detail::dispatchGemmWithConfig<Config, MMA, ElementA, ElementB, ElementCD,
                                                     oasr::EpilogueRelu>(
                A, B, C, D, M, N, K, lda, ldb, ldc, alpha, stream, split_k_slices);
        } else if (activation == ActivationType::GELU) {
            status = detail::dispatchGemmWithConfig<Config, MMA, ElementA, ElementB, ElementCD,
                                                     oasr::EpilogueGelu>(
                A, B, C, D, M, N, K, lda, ldb, ldc, alpha, stream, split_k_slices);
        } else if (activation == ActivationType::SWISH) {
            status = detail::dispatchGemmWithConfig<Config, MMA, ElementA, ElementB, ElementCD,
                                                     oasr::EpilogueSwish>(
                A, B, C, D, M, N, K, lda, ldb, ldc, alpha, stream, split_k_slices);
        } else {
            status = GemmStatus::INVALID_ARGUMENT;
        }
    });
#endif

    if (status != GemmStatus::SUCCESS) {
        return cudaErrorUnknown;
    }
    return cudaSuccess;
}

}  // namespace gemm
}  // namespace oasr
