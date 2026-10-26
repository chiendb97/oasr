// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// The native scaled GEMM: CUTLASS 3.x, ``D[M,N] = epilogue(A @ Bᵀ)`` on sm90
// (TMA + wgmma, ping-pong or cooperative warpgroups) and sm100 (TMA + tcgen05,
// accumulating in tensor memory), with the dequant epilogue of
// ``scaled_mm_epilogue_sm90.h``.  The CUTLASS 2.x ``mma.sync`` kernel
// (``scaled_mm_cutlass_template.h``) is the same contract on every other SM, and
// the A/B arm here (``OASR_SCALED_MM_BACKEND=mma_sync``).
//
// A is ``[M, K]`` row-major 8-bit (quantized activations), B ``[N, K]``
// row-major 8-bit (the ``nn.Linear`` weight layout, i.e. CUTLASS's column-major
// B): both K-major, which 8-bit wgmma requires.  The config names the element and
// accumulator, the tile, the cluster and the schedule
// (``cutlass_scaled_mm_configs_sm90.h``).
//
// The kernel is persistent and data-parallel: each output tile is one K pass,
// so the result does not depend on how tiles are scheduled.  No split-K and no
// Stream-K -- for fp8 a split would change the summation order with the tile.

#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __GNUC__
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wstrict-aliasing"
    #pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/numeric_types.h>
// Declares cutlass::make_cute_packed_stride (qualified lookup at parse time).
#include <cutlass/util/packed_stride.hpp>

#include <cute/tensor.hpp>

#ifdef __GNUC__
    #pragma GCC diagnostic pop
#endif

#include <oasr/common/graph_safe_workspace.h>
#include <oasr/common/workspace_cache.h>
#include <oasr/scaled_mm/cutlass_scaled_mm_configs_sm90.h>
#include <oasr/scaled_mm/scaled_mm_epilogue_sm90.h>
#include <oasr/scaled_mm/scaled_mm_status.h>

namespace oasr {
namespace scaled_mm {

template <typename Config, typename ElementD, typename EVT>
struct CutlassScaledMmGemmNative {
    using ElementAB = typename Config::ElementAB;
    using ElementAccumulator = typename Config::ElementAccumulator;
    using ElementCompute = float;
    using ArchTag = typename Config::ArchTag;
    using OperatorClass = cutlass::arch::OpClassTensorOp;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutD = cutlass::layout::RowMajor;

    // 16-byte TMA rows: K % 16 for the 8-bit operands, N % (16 / sizeof(D)) for
    // the output (8 for fp16/bf16, 16 for an 8-bit D).
    static constexpr int kAlignmentAB = 128 / cutlass::sizeof_bits<ElementAB>::value;
    static constexpr int kAlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

    // ElementC is void: there is no source matrix (the bias is a broadcast leaf
    // of the tree), so no C tile is loaded or allocated.
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass, typename Config::TileShape, typename Config::ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute, void,
        LayoutD, kAlignmentD, ElementD, LayoutD, kAlignmentD, typename Config::EpilogueSchedule,
        EVT>::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass, ElementAB, LayoutA, kAlignmentAB, ElementAB, LayoutB, kAlignmentAB,
        ElementAccumulator, typename Config::TileShape, typename Config::ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        typename Config::MainloopSchedule>::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>,
                                                            CollectiveMainloop, CollectiveEpilogue>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    using StrideA = typename GemmKernel::StrideA;
    using StrideB = typename GemmKernel::StrideB;
    using StrideC = typename GemmKernel::StrideC;
    using StrideD = typename GemmKernel::StrideD;

    static ScaledMmStatus run(ElementAB const* A, ElementAB const* B,
                              typename EVT::Arguments const& epilogue_args, ElementD* D, int M,
                              int N, int K, int64_t lda, int64_t ldb, int64_t ldd,
                              cudaStream_t stream) {
        // Packed strides give the static modes their types; then the caller's
        // leading dimensions.  Mode 0 is the dynamic one for A (M,K) row-major,
        // B (N,K) column-major and D (M,N) row-major alike.
        StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
        StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
        StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));
        cute::get<0>(stride_a) = lda;
        cute::get<0>(stride_b) = ldb;
        cute::get<0>(stride_d) = ldd;

        // The fp8 sm90 mainloop's ``mma_promotion_interval`` keeps its default
        // (4 MMAs = 128 K elements): one summation order for every tile.
        typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                                      {M, N, K, 1},
                                      {A, stride_a, B, stride_b},
                                      {epilogue_args, nullptr, StrideC{}, D, stride_d}};

        Gemm gemm_op;
        if (gemm_op.can_implement(args) != cutlass::Status::kSuccess) {
            return ScaledMmStatus::CANNOT_IMPLEMENT;
        }
        // The data-parallel persistent schedulers need little or no workspace;
        // prefer the persistent scratch buffer when they do, so a captured launch
        // carries no allocation.
        const size_t workspace_size = Gemm::get_workspace_size(args);
        if (workspace_size == 0) {
            return launch(gemm_op, args, nullptr, stream);
        }
        void* cached = getCachedWorkspace(workspace_size, stream, WorkspacePool::kScratch);
        if (cached != nullptr) {
            return launch(gemm_op, args, cached, stream);
        }
        GraphSafeWorkspace workspace(workspace_size, stream);
        return launch(gemm_op, args, workspace.get(), stream);
    }

private:
    static ScaledMmStatus launch(Gemm& gemm_op, typename Gemm::Arguments const& args,
                                 void* workspace, cudaStream_t stream) {
        if (gemm_op.initialize(args, workspace, stream) != cutlass::Status::kSuccess) {
            return ScaledMmStatus::INITIALIZE_FAILED;
        }
        if (gemm_op.run(stream) != cutlass::Status::kSuccess) {
            return ScaledMmStatus::RUN_FAILED;
        }
        return ScaledMmStatus::SUCCESS;
    }
};

// fp16/bf16 output.  ``a_per_token``: ``a_scale`` holds M floats, else one.
template <typename Config, typename ElementD>
ScaledMmStatus runScaledMmNative(int activation, typename Config::ElementAB const* A,
                                 typename Config::ElementAB const* B, float const* a_scale,
                                 bool a_per_token, float const* b_scale, ElementD const* bias,
                                 ElementD* D, int M, int N, int K, int64_t lda, int64_t ldb,
                                 int64_t ldd, cudaStream_t stream) {
    if (!isScaledMmActivation(activation)) {
        return ScaledMmStatus::CANNOT_IMPLEMENT;
    }
    using Epilogue = native::NativeScaledMmEpilogue<typename Config::CtaTileShape, ElementD>;
    return CutlassScaledMmGemmNative<Config, ElementD, typename Epilogue::EVT>::run(
        A, B, Epilogue::make_args(activation, a_scale, a_per_token, b_scale, bias), D, M, N, K,
        lda, ldb, ldd, stream);
}

// 8-bit output (the operands' format) for a consumer with a static input scale
// (``out_mul`` is its per-channel quantize multiplier); ``ElementT`` is the dtype
// the fp output would have had, which the activation output is rounded to first.
// The 8-bit TMA store needs N % 16 == 0; the Python side computes the unfused
// pair otherwise (bit-identical), so this refuses rather than serves it.
template <typename Config, typename ElementT>
ScaledMmStatus runScaledMmQuantNative(int activation, typename Config::ElementAB const* A,
                                      typename Config::ElementAB const* B, float const* a_scale,
                                      bool a_per_token, float const* b_scale, ElementT const* bias,
                                      float const* out_mul, typename Config::ElementAB* Q, int M,
                                      int N, int K, int64_t lda, int64_t ldb, int64_t ldq,
                                      cudaStream_t stream) {
    using ElementQ = typename Config::ElementAB;
    if (!isScaledMmActivation(activation)) {
        return ScaledMmStatus::CANNOT_IMPLEMENT;
    }
    using Epilogue =
        native::NativeScaledMmQuantEpilogue<typename Config::CtaTileShape, ElementT, ElementQ>;
    return CutlassScaledMmGemmNative<Config, ElementQ, typename Epilogue::EVT>::run(
        A, B, Epilogue::make_args(activation, a_scale, a_per_token, b_scale, bias, out_mul), Q, M,
        N, K, lda, ldb, ldq, stream);
}

}  // namespace scaled_mm
}  // namespace oasr
