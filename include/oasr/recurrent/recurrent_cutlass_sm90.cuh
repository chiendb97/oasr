// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// SM90+ recurrent step GEMM -- CUTLASS 3.x TMA warp-specialized.
//
// The CUTLASS 2.x compositions in recurrent_cutlass.cuh reach Hopper and
// Blackwell through the forward-compatible Sm80 tensor op, which works but
// leaves TMA and wgmma/tcgen05 on the table.  This header adds the 3.x
// collective path for the targets whose builders support FP16/BF16:
// SM90 (Hopper, wgmma) and SM100 (Blackwell datacenter, tcgen05).
//
// SM120 is deliberately absent: its 3.x OpClassTensorOp builder is restricted
// to F8/F6/F4 MMA, so GeForce Blackwell stays on the 2.x path -- the same
// reason `oasr/gemm/cutlass_gemm_configs.h` aliases CutlassArch<120> to Sm80.

#pragma once

// Only the two targets whose CUTLASS 3.x OpClassTensorOp builders accept
// FP16/BF16.  The includer gates on the same condition; tripping this means the
// header was pulled in for a target that cannot use it.
#if !defined(OASR_TARGET_SM) || (OASR_TARGET_SM != 90 && OASR_TARGET_SM != 100)
    #error "recurrent_cutlass_sm90.cuh requires a JIT build targeting SM90 or SM100"
#endif

#include <cstdint>

#ifdef __GNUC__
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wstrict-aliasing"
    #pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/gemm.h>

#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/fusion/operations.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/util/packed_stride.hpp>

#include <cute/tensor.hpp>

#ifdef __GNUC__
    #pragma GCC diagnostic pop
#endif

#include <oasr/common/epilogue_functors.h>
#include <oasr/common/graph_safe_workspace.h>
#include <oasr/common/types.h>
#include <oasr/gemm/cutlass_gemm_configs.h>
#include <oasr/gemm/gemm_cutlass_template.h>
#include <oasr/recurrent/recurrent.cuh>

namespace oasr {
namespace recurrent {

// One recurrent timestep as a CUTLASS 3.x collective GEMM:
//
//     D = activation(previous_h @ weight_hh^T + input_gates)
//
// Two things separate this from the general GEMM waist's 3.x template.  C is a
// *matrix* rather than a broadcast bias row -- it is the sequence-wide input
// projection, sliced at this timestep -- and both C and D carry the caller's
// own row stride rather than a packed one, because the projection is a slice of
// a [T, B, gates*hidden] tensor and D is a reusable tile.  Everything else is
// the ordinary CollectiveBuilder pipeline.
template <typename Config, typename Element, ActivationType kActivation>
struct RecurrentStepGemmSm90 {
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutCD = cutlass::layout::RowMajor;

    using ElementAccumulator = float;
    using ElementCompute = float;
    using ArchTag = typename Config::SmArch;
    using OperatorClass = cutlass::arch::OpClassTensorOp;

    using TileShape = typename Config::TileShape;
    using ClusterShape = typename Config::ClusterShape;

    static constexpr int kAlignmentA = 128 / cutlass::sizeof_bits<Element>::value;
    static constexpr int kAlignmentB = 128 / cutlass::sizeof_bits<Element>::value;
    static constexpr int kAlignmentEpilogue = 128 / cutlass::sizeof_bits<Element>::value;

    using EpilogueSchedule = typename Config::EpilogueSchedule;
    using MainloopSchedule = typename Config::MainloopSchedule;
    using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;

    // alpha * accumulator + beta * C, then the recurrent nonlinearity.  TANH is
    // why FusionEpilogueOpSm90 needed a tanh specialisation.
    using FusionOp =
        typename FusionEpilogueOpSm90<kActivation, Element, ElementCompute, Element>::type;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass, TileShape, ClusterShape, EpilogueTileType, ElementAccumulator,
        ElementCompute, Element, LayoutCD, kAlignmentEpilogue, Element, LayoutCD,
        kAlignmentEpilogue, EpilogueSchedule, FusionOp>::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass, Element, LayoutA, kAlignmentA, Element, LayoutB, kAlignmentB,
        ElementAccumulator, TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        MainloopSchedule>::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>,
                                                            CollectiveMainloop, CollectiveEpilogue>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;

    /// \param output      D, row-major, row stride \p ldd
    /// \param previous_h  A, (M, K) row-major and packed
    /// \param weight_hh   B, (N, K) column-major and packed
    /// \param input_gates C, row-major, row stride \p ldc
    static gemm::GemmStatus run(Element* output, const Element* previous_h,
                                const Element* weight_hh, const Element* input_gates,
                                int batch_size, int gate_columns, int hidden_size, int64_t ldc,
                                int64_t ldd, cudaStream_t stream) {
        const int M = batch_size;
        const int N = gate_columns;
        const int K = hidden_size;

        auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
        auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
        // Start from the packed strides so the static and batch components come
        // out with the right types, then substitute the caller's row strides.
        StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
        StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));
        cute::get<0>(stride_C) = ldc;
        cute::get<0>(stride_D) = ldd;

        typename Gemm::Arguments arguments{cutlass::gemm::GemmUniversalMode::kGemm,
                                           {M, N, K, 1},
                                           {previous_h, stride_A, weight_hh, stride_B},
                                           {{}, input_gates, stride_C, output, stride_D}};
        // beta = 1: C is the input projection this timestep accumulates onto,
        // not a bias to be scaled away.
        arguments.epilogue.thread.alpha = 1.0f;
        arguments.epilogue.thread.beta = 1.0f;

        Gemm gemm;
        if (gemm.can_implement(arguments) != cutlass::Status::kSuccess)
            return gemm::GemmStatus::NOT_SUPPORTED;

        const size_t workspace_size = gemm.get_workspace_size(arguments);
        oasr::GraphSafeWorkspace workspace(workspace_size, stream);
        if (gemm.initialize(arguments, workspace.get(), stream) != cutlass::Status::kSuccess)
            return gemm::GemmStatus::INTERNAL_ERROR;
        if (gemm.run(stream) != cutlass::Status::kSuccess)
            return gemm::GemmStatus::CUTLASS_ERROR;
        return gemm::GemmStatus::SUCCESS;
    }
};

// LSTM on the 3.x path is the *decomposed* tactic: the collective GEMM
// materializes one gate-interleaved tile and the existing finalizer applies the
// state transition.  The 2.x fused variant cannot come along -- its custom
// epilogue reconstructs logical coordinates from a PredicatedTileIterator thread
// map, and 3.x replaced that machinery with cute layouts and an epilogue visitor
// tree, where a four-gates-to-one-cell column reduction is not an elementwise
// node.  Decomposing costs one extra launch and reuses a finalizer that is
// already covered against cuDNN.
template <typename Config, typename Element, typename CudaType>
cudaError_t LstmStateGemmSm90(CudaType* output, CudaType* cells, CudaType* final_h,
                              CudaType* final_c, const Element* previous_h,
                              const Element* weight_hh, const Element* input_gates,
                              const CudaType* previous_c, CudaType* workspace, int batch_size,
                              int hidden_size, int64_t input_gate_batch_stride,
                              int64_t previous_cell_batch_stride, cudaStream_t stream) {
    const gemm::GemmStatus status =
        RecurrentStepGemmSm90<Config, Element, ActivationType::IDENTITY>::run(
            reinterpret_cast<Element*>(workspace), previous_h, weight_hh, input_gates, batch_size,
            4 * hidden_size, hidden_size, input_gate_batch_stride, 4 * hidden_size, stream);
    if (status != gemm::GemmStatus::SUCCESS)
        return cudaErrorUnknown;
    return LstmInterleavedGateStep<CudaType>(output, cells, final_h, final_c, workspace, previous_c,
                                             batch_size, hidden_size, previous_cell_batch_stride,
                                             stream);
}

// The vanilla RNN has one gate, so its nonlinearity *is* an elementwise
// epilogue and stays fused in the collective.
template <typename Config, typename Element, RnnActivation Activation>
gemm::GemmStatus RnnStateGemmSm90(Element* output, const Element* previous_h,
                                  const Element* weight_hh, const Element* input_gates,
                                  int batch_size, int hidden_size, int64_t input_gate_batch_stride,
                                  cudaStream_t stream) {
    constexpr ActivationType kActivation =
        Activation == RnnActivation::RELU ? ActivationType::RELU : ActivationType::TANH;
    return RecurrentStepGemmSm90<Config, Element, kActivation>::run(
        output, previous_h, weight_hh, input_gates, batch_size, hidden_size, hidden_size,
        input_gate_batch_stride, hidden_size, stream);
}

// SM90's cooperative schedule static_asserts on an M tile below 128 rows, and a
// recurrent cohort is nowhere near 128 rows, so the thin tile takes the pingpong
// schedule instead.  SM100 picks its schedule from kSMs and ignores the flag, so
// one pair of configs covers both.
constexpr int kRecurrentSm90Arch = OASR_TARGET_SM;

using RecurrentSm90Config64 =
    gemm::CutlassGemmConfigSm90<64, 128, 64, 1, 1, 1, 3, kRecurrentSm90Arch, true>;
using RecurrentSm90Config128 =
    gemm::CutlassGemmConfigSm90<128, 128, 64, 1, 1, 1, 3, kRecurrentSm90Arch, false>;

}  // namespace recurrent
}  // namespace oasr
