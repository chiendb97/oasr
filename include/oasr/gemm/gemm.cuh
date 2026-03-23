// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Standard GEMM kernel — pure CUDA + CUTLASS, no framework dependencies.
// Supports BF16 and FP16 precision with architecture-aware dispatch.
//
// Merges gemm_impl.h (CUTLASS template) and gemm_kernels.cu (dispatch logic).

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

#ifdef __GNUC__
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wstrict-aliasing"
    #pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/device_memory.h>

#ifdef __GNUC__
    #pragma GCC diagnostic pop
#endif

#include <oasr/common/arch_dispatch.h>
#include <oasr/common/arch_traits.h>
#include <oasr/common/epilogue_functors.h>
#include <oasr/common/tuneable_traits.h>
#include <oasr/common/types.h>
#include <oasr/gemm/gemm_utils.h>

namespace oasr {
namespace gemm {

//==============================================================================
// CUTLASS GEMM Template (from gemm_impl.h)
//==============================================================================

template <typename Traits, typename ElementA, typename ElementB, typename ElementCD, typename LayoutA,
          typename LayoutB, typename LayoutCD,
          template <int, typename, typename> class EpilogueFunctor>
struct CutlassGemm {
    using ElementAccumulator = float;
    using ElementComputeEpilogue = ElementAccumulator;

    static constexpr int EpilogueAlignment = 128 / cutlass::sizeof_bits<ElementCD>::value;

    using MMAOp = typename Traits::MMAOp;
    using SmArch = typename Traits::SmArch;

    using ShapeMMAThreadBlock = typename Traits::Gemm::ThreadBlock;
    using ShapeMMAWarps = typename Traits::Gemm::Warps;
    using ShapeMMAOp = typename Traits::Gemm::MMAShape;

    using SwizzleThreadblock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

    using EpilogueOp =
        typename EpilogueFunctor<EpilogueAlignment, ElementCD, ElementComputeEpilogue>::Op;

    static constexpr int NumStages = Traits::Gemm::NumStages;

    using Gemm = cutlass::gemm::device::Gemm<ElementA, LayoutA, ElementB, LayoutB, ElementCD,
                                             LayoutCD, ElementAccumulator, MMAOp, SmArch,
                                             ShapeMMAThreadBlock, ShapeMMAWarps, ShapeMMAOp,
                                             EpilogueOp, SwizzleThreadblock, NumStages>;

    static GemmStatus run(const ElementA* A, const ElementB* B, const ElementCD* C, ElementCD* D,
                          int M, int N, int K, int64_t lda, int64_t ldb, int64_t ldc,
                          ElementComputeEpilogue alpha, cudaStream_t stream,
                          int split_k_slices = 1) {
        float beta = (C == nullptr) ? 0.0f : 1.0f;
        typename Gemm::Arguments args({M, N, K}, {A, lda}, {B, ldb}, {C, 0}, {D, ldc},
                                      {alpha, beta}, split_k_slices);

        Gemm gemm_op;
        cutlass::Status status = gemm_op.can_implement(args);
        if (status != cutlass::Status::kSuccess) {
            return GemmStatus::NOT_SUPPORTED;
        }

        size_t workspace_size = Gemm::get_workspace_size(args);
        cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

        status = gemm_op.initialize(args, workspace.get(), stream);
        if (status != cutlass::Status::kSuccess) {
            return GemmStatus::INTERNAL_ERROR;
        }

        status = gemm_op(stream);
        if (status != cutlass::Status::kSuccess) {
            return GemmStatus::CUTLASS_ERROR;
        }

        return GemmStatus::SUCCESS;
    }
};

//==============================================================================
// Architecture + dtype dispatch helpers
//==============================================================================

namespace detail {

template <int SmVersion, typename ElementA, typename ElementB, typename ElementCD,
          template <int, typename, typename> class EpilogueFn>
static GemmStatus dispatchGemmImpl(const ElementA* A_ptr, const ElementB* B_ptr,
                                   const ElementCD* C_ptr, ElementCD* D_ptr, int M, int N, int K,
                                   uint64_t lda, uint64_t ldb, uint64_t ldc, float alpha,
                                   cudaStream_t stream, int split_k_slices = 1) {
    using Traits = oasr::TuneableTraits<SmVersion>;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;

    return CutlassGemm<Traits, ElementA, ElementB, ElementCD, LayoutA, LayoutB, LayoutC,
                        EpilogueFn>::run(A_ptr, B_ptr, C_ptr, D_ptr, M, N, K, lda, ldb, ldc,
                                         alpha, stream, split_k_slices);
}

}  // namespace detail

//==============================================================================
// Typed Launcher: Gemm
//==============================================================================

/**
 * @brief Execute GEMM operation: D = alpha * A @ B + beta * C
 *
 * When C is nullptr, computes D = alpha * A @ B.
 * Layout: A is [M, K] row-major, B is [N, K] column-major, C/D are [M, N] row-major.
 *
 * @tparam ElementA  Element type of matrix A (cutlass::half_t or cutlass::bfloat16_t)
 * @tparam ElementB  Element type of matrix B
 * @tparam ElementCD Element type of matrices C and D
 * @return cudaError_t  cudaSuccess on success
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

    int sm = oasr::getDeviceSmVersion();
    OASR_DISPATCH_ARCH(sm, SM_VERSION, {
        status = detail::dispatchGemmImpl<SM_VERSION, ElementA, ElementB, ElementCD,
                                          oasr::EpilogueIdentity>(A, B, C, D, M, N, K, lda, ldb,
                                                                   ldc, alpha, stream,
                                                                   split_k_slices);
    });

    if (status != GemmStatus::SUCCESS) {
        return cudaErrorUnknown;
    }
    return cudaSuccess;
}

//==============================================================================
// Typed Launcher: GemmActivation
//==============================================================================

/**
 * @brief Execute GEMM with fused activation: D = activation(alpha * A @ B + beta * C)
 *
 * @tparam ElementA  Element type of matrix A
 * @tparam ElementB  Element type of matrix B
 * @tparam ElementCD Element type of matrices C and D
 * @return cudaError_t  cudaSuccess on success
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

    int sm = oasr::getDeviceSmVersion();
    OASR_DISPATCH_ARCH(sm, SM_VERSION, {
        if (activation == ActivationType::RELU) {
            status = detail::dispatchGemmImpl<SM_VERSION, ElementA, ElementB, ElementCD,
                                              oasr::EpilogueRelu>(A, B, C, D, M, N, K, lda, ldb,
                                                                   ldc, alpha, stream,
                                                                   split_k_slices);
        } else if (activation == ActivationType::GELU) {
            status = detail::dispatchGemmImpl<SM_VERSION, ElementA, ElementB, ElementCD,
                                              oasr::EpilogueGelu>(A, B, C, D, M, N, K, lda, ldb,
                                                                   ldc, alpha, stream,
                                                                   split_k_slices);
        } else if (activation == ActivationType::SWISH) {
            status = detail::dispatchGemmImpl<SM_VERSION, ElementA, ElementB, ElementCD,
                                              oasr::EpilogueSwish>(A, B, C, D, M, N, K, lda, ldb,
                                                                    ldc, alpha, stream,
                                                                    split_k_slices);
        } else {
            status = GemmStatus::INVALID_ARGUMENT;
        }
    });

    if (status != GemmStatus::SUCCESS) {
        return cudaErrorUnknown;
    }
    return cudaSuccess;
}

}  // namespace gemm
}  // namespace oasr
