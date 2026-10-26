// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// CUTLASS 2.x scaled GEMM: ``D[M,N] = epilogue(A @ Bᵀ)`` with A ``[M, K]``
// row-major 8-bit (quantized activations), B ``[N, K]`` row-major 8-bit (the
// ``nn.Linear`` weight layout, i.e. CUTLASS's column-major B -- no transpose, no
// repack).  The config names the element and accumulator (int8 -> int32, fp8
// e4m3 -> fp32; ``cutlass_scaled_mm_configs.h``); the epilogue tree decides the
// rest (``scaled_mm_epilogue.h``): an fp16/bf16 output, or 8-bit for a consumer
// GEMM.
//
// Two K decompositions:
//
// * data-parallel -- one K pass per output tile, no workspace;
// * Stream-K (int8 only) -- K iterations spread over every SM, partial tiles
//   reduced through a workspace.  The partials are the int32 accumulator, so the
//   reduction is an integer sum: any order, any split, the same bits as the
//   data-parallel kernel.  That is what makes Stream-K safe for batch invariance
//   here, where fp32 partials (the fp16 GEMM's, or fp8's) are not.
//
// No serial or parallel split-K: the Sm80 EVT kernel rejects them only by
// ``assert`` (gemm_universal_with_visitor.h), which NDEBUG compiles out, so a
// split launch would run and be wrong.  Stream-K is the supported way to split K.

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle_streamk.h>
#include <cutlass/numeric_types.h>

#include <type_traits>

#include <oasr/common/graph_safe_workspace.h>
#include <oasr/common/workspace_cache.h>
#include <oasr/scaled_mm/cutlass_scaled_mm_configs.h>
#include <oasr/scaled_mm/scaled_mm_epilogue.h>
#include <oasr/scaled_mm/scaled_mm_status.h>

namespace oasr {
namespace scaled_mm {

// Fp32 compute fragments, four per access: the thread map every epilogue
// visitor partitions the tile with (vLLM's sm80 W8A8 choice).
template <typename Config>
struct ScaledMmTileMap {
    static constexpr int kElementsPerAccess = 4;
    using type = cutlass::epilogue::threadblock::OutputTileThreadLayout<
        typename Config::ThreadblockShape, typename Config::WarpShape, float, kElementsPerAccess,
        /*EpilogueStages=*/1>;
};

template <typename Config, typename EVTD, bool kStreamK>
struct CutlassScaledMmGemm {
    using ElementAB = typename Config::ElementAB;
    using ElementAccumulator = typename Config::ElementAccumulator;
    using ElementCompute = float;
    static_assert(!kStreamK || std::is_same_v<ElementAccumulator, int32_t>,
                  "Stream-K is exact only over an integer accumulator");

    static constexpr int kAlignmentAB = 128 / cutlass::sizeof_bits<ElementAB>::value;  // 16

    // ``DefaultGemmWithVisitor`` selects the Stream-K kernel
    // (``GemmWithEpilogueVisitorStreamk``) from the swizzle alone.
    using Swizzle =
        std::conditional_t<kStreamK, cutlass::gemm::threadblock::ThreadblockSwizzleStreamK,
                           cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>>;

    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
        ElementAB, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, kAlignmentAB,
        ElementAB, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, kAlignmentAB,
        ElementCompute, cutlass::layout::RowMajor, ScaledMmTileMap<Config>::kElementsPerAccess,
        ElementAccumulator, ElementCompute, cutlass::arch::OpClassTensorOp,
        typename Config::ArchTag, typename Config::ThreadblockShape, typename Config::WarpShape,
        typename Config::InstructionShape, EVTD, Swizzle, Config::kStages,
        typename Config::Operator, /*EpilogueStages=*/1>::GemmKernel;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    static ScaledMmStatus run(ElementAB const* A, ElementAB const* B,
                              typename EVTD::Arguments const& epilogue_args, int M, int N, int K,
                              int64_t lda, int64_t ldb, int64_t ldd, cudaStream_t stream) {
        typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                                      {M, N, K},
                                      /*batch_count=*/1,
                                      epilogue_args,
                                      A,
                                      B,
                                      /*ptr_C=*/nullptr,
                                      /*ptr_D=*/nullptr,
                                      /*batch_stride_A=*/0,
                                      /*batch_stride_B=*/0,
                                      /*batch_stride_C=*/0,
                                      /*batch_stride_D=*/0,
                                      lda,
                                      ldb,
                                      ldd,
                                      ldd};

        Gemm gemm_op;
        if (gemm_op.can_implement(args) != cutlass::Status::kSuccess) {
            return ScaledMmStatus::CANNOT_IMPLEMENT;
        }
        if constexpr (kStreamK) {
            // The barrier flags are re-zeroed by ``initialize`` every launch (a
            // memset node under capture); the allocation itself is reusable, so
            // prefer the persistent scratch buffer over a per-call alloc/free.
            const size_t workspace_size = Gemm::get_workspace_size(args);
            void* cached = getCachedWorkspace(workspace_size, stream, WorkspacePool::kScratch);
            if (cached != nullptr) {
                return launch(gemm_op, args, cached, stream);
            }
            GraphSafeWorkspace workspace(workspace_size, stream);
            return launch(gemm_op, args, workspace.get(), stream);
        } else {
            // One K pass needs no workspace, which also keeps the launch
            // capturable without an allocation.
            return launch(gemm_op, args, nullptr, stream);
        }
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

}  // namespace scaled_mm
}  // namespace oasr
