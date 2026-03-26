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

#include <type_traits>

#ifdef __GNUC__
    #pragma GCC diagnostic pop
#endif

#include <oasr/common/arch_dispatch.h>
#include <oasr/gemm/cutlass_gemm_configs.h>
#include <oasr/gemm/gemm_utils.h>

namespace oasr {
namespace gemm {

//==============================================================================
// CUTLASS Batched GEMM Template
//==============================================================================

template <typename Config, typename MMATraits, typename ElementA, typename ElementB,
          typename ElementCD, typename LayoutA, typename LayoutB, typename LayoutCD>
struct CutlassBmmKernel {
    using ElementAccumulator = float;
    using ElementComputeEpilogue = ElementAccumulator;

    // Float32 requires SIMT; half/bf16 use TensorOp from MMATraits
    static constexpr bool kUseSIMT = std::is_same_v<ElementA, float>;

    using MMAOp =
        std::conditional_t<kUseSIMT, cutlass::arch::OpClassSimt, typename MMATraits::MMAOp>;
    using SmArch = typename MMATraits::SmArch;

    using ShapeMMAThreadBlock = std::conditional_t<kUseSIMT,
        cutlass::gemm::GemmShape<128, 128, 8>, typename Config::ThreadBlock>;
    using ShapeMMAWarps = std::conditional_t<kUseSIMT,
        cutlass::gemm::GemmShape<32, 64, 8>, typename Config::Warps>;
    using ShapeMMAOp = std::conditional_t<kUseSIMT,
        cutlass::gemm::GemmShape<1, 1, 1>, typename MMATraits::MMAShape>;

    using SwizzleThreadblock = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;

    // SIMT epilogue requires scalar access (alignment=1)
    static constexpr int EpilogueAlignment =
        kUseSIMT ? 1 : 128 / cutlass::sizeof_bits<ElementCD>::value;

    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementCD, EpilogueAlignment, ElementComputeEpilogue, ElementComputeEpilogue,
        cutlass::epilogue::thread::ScaleType::Default>;

    static constexpr int NumStages = kUseSIMT ? 2 : Config::NumStages;

    using Gemm = cutlass::gemm::device::GemmBatched<
        ElementA, LayoutA, ElementB, LayoutB, ElementCD, LayoutCD, ElementAccumulator, MMAOp,
        SmArch, ShapeMMAThreadBlock, ShapeMMAWarps, ShapeMMAOp, EpilogueOp, SwizzleThreadblock,
        NumStages>;

    static GemmStatus run(const ElementA* A, const ElementB* B, ElementCD* D, int batch_size,
                          int M, int N, int K, int64_t lda, int64_t ldb, int64_t ldd,
                          int64_t stride_a, int64_t stride_b, int64_t stride_d, float alpha,
                          float beta, cudaStream_t stream) {
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
// Dispatch helpers
//==============================================================================

namespace detail {

template <typename Config, typename MMATraits, typename ElementA, typename ElementB,
          typename ElementCD>
static GemmStatus dispatchBmmWithConfig(const ElementA* A_ptr, const ElementB* B_ptr,
                                         ElementCD* D_ptr, int batch_size, int M, int N, int K,
                                         uint64_t lda, uint64_t ldb, uint64_t ldd,
                                         int64_t stride_a, int64_t stride_b, int64_t stride_d,
                                         float alpha, float beta, cudaStream_t stream) {
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutCD = cutlass::layout::RowMajor;

    return CutlassBmmKernel<Config, MMATraits, ElementA, ElementB, ElementCD, LayoutA, LayoutB,
                             LayoutCD>::run(A_ptr, B_ptr, D_ptr, batch_size, M, N, K, lda, ldb,
                                            ldd, stride_a, stride_b, stride_d, alpha, beta,
                                            stream);
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
        status = detail::dispatchBmmWithConfig<Config, MMA>(A, B, D, batch_size, M, N, K, lda, ldb,
                                                             ldd, stride_a, stride_b, stride_d,
                                                             alpha, beta, stream);
    }
#else
    int sm = oasr::getDeviceSmVersion();
    OASR_DISPATCH_SM(sm, SM_VERSION, {
        using Config = typename DefaultGemmConfig<SM_VERSION>::type;
        using MMA = SmMMATraits<SM_VERSION>;
        status = detail::dispatchBmmWithConfig<Config, MMA>(A, B, D, batch_size, M, N, K, lda, ldb,
                                                             ldd, stride_a, stride_b, stride_d,
                                                             alpha, beta, stream);
    });
#endif

    if (status != GemmStatus::SUCCESS) {
        return cudaErrorUnknown;
    }
    return cudaSuccess;
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
#ifdef OASR_GEMM_TILE_M
        using Config = JitGemmConfig;
#else
        using Config = typename DefaultGemmConfig<SM_VERSION>::type;
#endif
        using MMA = SmMMATraits<SM_VERSION>;
        status = detail::dispatchBmmWithConfig<Config, MMA>(A, B, D, batch_size, M, N, K, lda, ldb,
                                                             ldd, stride_a, stride_b, stride_d,
                                                             alpha, beta, stream);
    }
#else
    int sm = oasr::getDeviceSmVersion();
    OASR_DISPATCH_SM(sm, SM_VERSION, {
        using Config = typename DefaultGemmConfig<SM_VERSION>::type;
        using MMA = SmMMATraits<SM_VERSION>;
        status = detail::dispatchBmmWithConfig<Config, MMA>(A, B, D, batch_size, M, N, K, lda, ldb,
                                                             ldd, stride_a, stride_b, stride_d,
                                                             alpha, beta, stream);
    });
#endif

    if (status != GemmStatus::SUCCESS) {
        return cudaErrorUnknown;
    }
    return cudaSuccess;
}

}  // namespace gemm
}  // namespace oasr
