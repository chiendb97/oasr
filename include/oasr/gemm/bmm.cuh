// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Batched Matrix Multiplication (BMM) kernel — pure CUDA + CUTLASS, no framework dependencies.
// Supports BF16 and FP16 precision with architecture-aware dispatch.
//
// Merges bmm_impl.h (CUTLASS template) and bmm_kernels.cu (dispatch logic).

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
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/device_memory.h>

#include <type_traits>

#ifdef __GNUC__
    #pragma GCC diagnostic pop
#endif

#include <oasr/common/arch_dispatch.h>
#include <oasr/common/arch_traits.h>
#include <oasr/common/tuneable_traits.h>
#include <oasr/gemm/gemm_utils.h>

namespace oasr {
namespace gemm {

//==============================================================================
// CUTLASS Batched GEMM Template (from bmm_impl.h)
//==============================================================================

template <typename Traits, typename ElementA, typename ElementB, typename ElementCD, typename LayoutA,
          typename LayoutB, typename LayoutCD>
struct CutlassBmm {
    using ElementAccumulator = float;
    using ElementComputeEpilogue = ElementAccumulator;

    // Float32 requires SIMT; half/bf16 use TensorOp from ArchTraits
    static constexpr bool kUseSIMT = std::is_same_v<ElementA, float>;

    using MMAOp = std::conditional_t<kUseSIMT, cutlass::arch::OpClassSimt, typename Traits::MMAOp>;
    using SmArch = typename Traits::SmArch;

    using ShapeMMAThreadBlock = std::conditional_t<kUseSIMT,
        cutlass::gemm::GemmShape<128, 128, 8>, typename Traits::Gemm::ThreadBlock>;
    using ShapeMMAWarps = std::conditional_t<kUseSIMT,
        cutlass::gemm::GemmShape<32, 64, 8>, typename Traits::Gemm::Warps>;
    using ShapeMMAOp = std::conditional_t<kUseSIMT,
        cutlass::gemm::GemmShape<1, 1, 1>, typename Traits::Gemm::MMAShape>;

    using SwizzleThreadblock = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;

    // SIMT epilogue requires scalar access (alignment=1)
    static constexpr int EpilogueAlignment = kUseSIMT ? 1 : 128 / cutlass::sizeof_bits<ElementCD>::value;

    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementCD, EpilogueAlignment, ElementComputeEpilogue,
        ElementComputeEpilogue, cutlass::epilogue::thread::ScaleType::Default>;

    static constexpr int NumStages = kUseSIMT ? 2 : Traits::Gemm::NumStages;

    using Gemm = cutlass::gemm::device::GemmBatched<ElementA, LayoutA, ElementB, LayoutB, ElementCD,
                                                    LayoutCD, ElementAccumulator, MMAOp, SmArch,
                                                    ShapeMMAThreadBlock, ShapeMMAWarps, ShapeMMAOp,
                                                    EpilogueOp, SwizzleThreadblock, NumStages>;

    static GemmStatus run(const ElementA* A, const ElementB* B, ElementCD* D, int batch_size, int M,
                          int N, int K, int64_t lda, int64_t ldb, int64_t ldd, int64_t stride_a,
                          int64_t stride_b, int64_t stride_d, float alpha, float beta,
                          cudaStream_t stream) {
        typename Gemm::Arguments args({M, N, K}, {A, lda}, stride_a, {B, ldb}, stride_b, {D, ldd},
                                      stride_d, {D, ldd}, stride_d, {alpha, beta}, batch_size);

        Gemm gemm_op;
        if (gemm_op.can_implement(args) != cutlass::Status::kSuccess)
            return GemmStatus::NOT_SUPPORTED;

        size_t ws_size = Gemm::get_workspace_size(args);
        cutlass::device_memory::allocation<uint8_t> ws(ws_size);

        if (gemm_op.initialize(args, ws.get(), stream) != cutlass::Status::kSuccess)
            return GemmStatus::INTERNAL_ERROR;

        return (gemm_op(stream) == cutlass::Status::kSuccess) ? GemmStatus::SUCCESS
                                                              : GemmStatus::CUTLASS_ERROR;
    }
};

//==============================================================================
// Architecture + dtype dispatch helper
//==============================================================================

namespace detail {

template <int SmVersion, typename ElementA, typename ElementB, typename ElementCD>
static GemmStatus dispatchBmmImpl(const ElementA* A_ptr, const ElementB* B_ptr, ElementCD* D_ptr,
                                  int batch_size, int M, int N, int K, uint64_t lda, uint64_t ldb,
                                  uint64_t ldd, int64_t stride_a, int64_t stride_b,
                                  int64_t stride_d, float alpha, float beta, cudaStream_t stream) {
    using Traits = oasr::TuneableTraits<SmVersion>;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutCD = cutlass::layout::RowMajor;

    return CutlassBmm<Traits, ElementA, ElementB, ElementCD, LayoutA, LayoutB, LayoutCD>::run(
        A_ptr, B_ptr, D_ptr, batch_size, M, N, K, lda, ldb, ldd, stride_a, stride_b, stride_d,
        alpha, beta, stream);
}

}  // namespace detail

//==============================================================================
// Typed Launcher: Bmm
//==============================================================================

/**
 * @brief Execute strided batched GEMM operation.
 *
 * Computes: D[b] = alpha * A[b] @ B[b] + beta * D[b] for b in [0, batch_size)
 *
 * Layout: A is [batch, M, K] row-major, B is [batch, N, K] column-major,
 *         D is [batch, M, N] row-major.
 *
 * @tparam ElementA  Element type of matrix A (cutlass::half_t or cutlass::bfloat16_t)
 * @tparam ElementB  Element type of matrix B
 * @tparam ElementCD Element type of output matrix D
 * @return cudaError_t  cudaSuccess on success
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

    GemmStatus status = GemmStatus::SUCCESS;

    int sm = oasr::getDeviceSmVersion();
    OASR_DISPATCH_ARCH(sm, SM_VERSION, {
        status = detail::dispatchBmmImpl<SM_VERSION>(A, B, D, batch_size, M, N, K, lda, ldb, ldd,
                                                     stride_a, stride_b, stride_d, alpha, beta,
                                                     stream);
    });

    if (status != GemmStatus::SUCCESS) {
        return cudaErrorUnknown;
    }
    return cudaSuccess;
}

/**
 * @brief Execute strided batched GEMM with full control over strides and scaling.
 *
 * @tparam ElementA  Element type of matrix A
 * @tparam ElementB  Element type of matrix B
 * @tparam ElementCD Element type of output matrix D
 * @return cudaError_t  cudaSuccess on success
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

    int sm = oasr::getDeviceSmVersion();
    OASR_DISPATCH_ARCH(sm, SM_VERSION, {
        status = detail::dispatchBmmImpl<SM_VERSION>(A, B, D, batch_size, M, N, K, lda, ldb, ldd,
                                                     stride_a, stride_b, stride_d, alpha, beta,
                                                     stream);
    });

    if (status != GemmStatus::SUCCESS) {
        return cudaErrorUnknown;
    }
    return cudaSuccess;
}

}  // namespace gemm
}  // namespace oasr
