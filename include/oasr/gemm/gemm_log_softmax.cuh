// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Fused GEMM + log_softmax dispatch interface.
//
// Computes  D = log_softmax(A @ B + bias, dim=N)  where N is the row-stride
// dimension of D ([M, N] row-major).  Reuses the existing CUTLASS 2.x/3.x GEMM
// kernel with the IDENTITY epilogue (bias supplied via the GEMM's broadcast-C
// path), then runs an online log_softmax pass over each output row.
//
// A single-pass CUTLASS epilogue cannot compute log_softmax for typical CTC
// vocab sizes (5k-30k) because the row-max and row-sum reductions span more
// columns than a single threadblock tile.  This file therefore follows the
// CUTLASS example 35 (gemm_softmax) pattern: GEMM kernel + normalization
// kernel chained on the same stream.

#pragma once

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "oasr/common/arch_dispatch.h"
#include "oasr/common/types.h"
#include "oasr/gemm/cutlass_gemm_configs.h"
#include "oasr/gemm/gemm_cutlass_template.h"
#include "oasr/softmax.cuh"
#if !defined(OASR_TARGET_SM) || OASR_TARGET_SM == 90 || OASR_TARGET_SM == 100
#include "oasr/gemm/gemm_cutlass_template_sm90.h"
#endif

namespace oasr {
namespace gemm {

namespace detail {

// CUTLASS storage type <-> native CUDA type. The two are bit-compatible so a
// reinterpret_cast suffices for the post-GEMM log_softmax pass.
template <typename T>
struct NativeFor {
    using type = T;
};
template <>
struct NativeFor<cutlass::half_t> {
    using type = __half;
};
template <>
struct NativeFor<cutlass::bfloat16_t> {
    using type = __nv_bfloat16;
};

template <int SM_VERSION, typename ElementA, typename ElementB, typename ElementCD>
static GemmStatus dispatchGemmLogSoftmax(const ElementA* A_ptr, const ElementB* B_ptr,
                                         const ElementCD* C_ptr, ElementCD* D_ptr, int M, int N,
                                         int K, uint64_t lda, uint64_t ldb, uint64_t ldc,
                                         float alpha, cudaStream_t stream, int split_k_slices) {
    GemmStatus status;
    if constexpr (SM_VERSION == 75) {
        using Config = CutlassGemmConfig<16, 128, 64, 16, 32, 64, 3, 75>;
        status = CutlassGemmKernel<Config, ElementA, ElementB, ElementCD,
                                   ActivationType::IDENTITY>::run(A_ptr, B_ptr, C_ptr, D_ptr, M, N,
                                                                  K, lda, ldb, ldc, alpha, stream,
                                                                  split_k_slices);
    } else if constexpr (SM_VERSION == 80) {
        using Config = CutlassGemmConfig<16, 128, 64, 16, 32, 64, 3, 80>;
        status = CutlassGemmKernel<Config, ElementA, ElementB, ElementCD,
                                   ActivationType::IDENTITY>::run(A_ptr, B_ptr, C_ptr, D_ptr, M, N,
                                                                  K, lda, ldb, ldc, alpha, stream,
                                                                  split_k_slices);
    } else if constexpr (SM_VERSION == 86) {
        using Config = CutlassGemmConfig<16, 128, 64, 16, 32, 64, 3, 86>;
        status = CutlassGemmKernel<Config, ElementA, ElementB, ElementCD,
                                   ActivationType::IDENTITY>::run(A_ptr, B_ptr, C_ptr, D_ptr, M, N,
                                                                  K, lda, ldb, ldc, alpha, stream,
                                                                  split_k_slices);
    } else if constexpr (SM_VERSION == 89) {
        using Config = CutlassGemmConfig<16, 128, 64, 16, 32, 64, 3, 89>;
        status = CutlassGemmKernel<Config, ElementA, ElementB, ElementCD,
                                   ActivationType::IDENTITY>::run(A_ptr, B_ptr, C_ptr, D_ptr, M, N,
                                                                  K, lda, ldb, ldc, alpha, stream,
                                                                  split_k_slices);
#if !defined(OASR_TARGET_SM) || OASR_TARGET_SM == 90 || OASR_TARGET_SM == 100
    } else if constexpr (SM_VERSION == 90) {
        using Config = CutlassGemmConfigSm90<64, 16, 128, 1, 1, 1, 1, 3, 90>;
        status = CutlassGemmKernelSm90<Config, ElementA, ElementB, ElementCD,
                                       ActivationType::IDENTITY>::run(A_ptr, B_ptr, C_ptr, D_ptr, M,
                                                                      N, K, lda, ldb, ldc, alpha,
                                                                      stream, split_k_slices);
    } else if constexpr (SM_VERSION == 100) {
        using Config = CutlassGemmConfigSm90<64, 16, 128, 1, 1, 1, 1, 3, 100>;
        status = CutlassGemmKernelSm90<Config, ElementA, ElementB, ElementCD,
                                       ActivationType::IDENTITY>::run(A_ptr, B_ptr, C_ptr, D_ptr, M,
                                                                      N, K, lda, ldb, ldc, alpha,
                                                                      stream, split_k_slices);
#endif
    } else if constexpr (SM_VERSION == 120) {
        using Config = CutlassGemmConfig<16, 128, 64, 16, 32, 64, 3, 120>;
        status = CutlassGemmKernel<Config, ElementA, ElementB, ElementCD,
                                   ActivationType::IDENTITY>::run(A_ptr, B_ptr, C_ptr, D_ptr, M, N,
                                                                  K, lda, ldb, ldc, alpha, stream,
                                                                  split_k_slices);
    } else {
        return GemmStatus::INVALID_ARGUMENT;
    }
    return status;
}

}  // namespace detail

//==============================================================================
// GemmLogSoftmax — public API
//==============================================================================

/**
 * @brief Execute GEMM with fused log_softmax along the N (output column)
 *        dimension: D[i, :] = log_softmax(alpha * (A @ B)[i, :] + bias[:]).
 *
 * Layout: A is [M, K] row-major, B is [N, K] column-major (i.e. weight stored
 * as [N, K]), D is [M, N] row-major.  When @p bias is non-null it must be of
 * shape [N] and is broadcast along the M dimension via CUTLASS's stride-0 C
 * tensor.
 */
template <typename ElementA, typename ElementB, typename ElementCD>
cudaError_t GemmLogSoftmax(const ElementA* A, const ElementB* B, const ElementCD* bias,
                           ElementCD* D, int M, int N, int K, cudaStream_t stream,
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
        status = detail::dispatchGemmLogSoftmax<SM_VERSION, ElementA, ElementB, ElementCD>(
            A, B, bias, D, M, N, K, lda, ldb, ldc, alpha, stream, split_k_slices);
    }
#else
    int sm = oasr::getDeviceSmVersion();
    OASR_DISPATCH_SM(sm, SM_VERSION, {
        status = detail::dispatchGemmLogSoftmax<SM_VERSION, ElementA, ElementB, ElementCD>(
            A, B, bias, D, M, N, K, lda, ldb, ldc, alpha, stream, split_k_slices);
    });
#endif

    if (status != GemmStatus::SUCCESS) {
        return cudaErrorUnknown;
    }

    // Row-wise log_softmax over D in-place. M rows × N columns, row-major.
    // CUTLASS half_t / bfloat16_t are bit-compatible with __half / __nv_bfloat16;
    // reinterpret so the softmax kernel's Vec<T> specialisations apply.
    using NativeCD = typename detail::NativeFor<ElementCD>::type;
    NativeCD* D_native = reinterpret_cast<NativeCD*>(D);
    return oasr::softmax::LogSoftmax<NativeCD>(D_native, D_native, static_cast<unsigned int>(M),
                                                static_cast<unsigned int>(N), stream);
}

}  // namespace gemm
}  // namespace oasr
