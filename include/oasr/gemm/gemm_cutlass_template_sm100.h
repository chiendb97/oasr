// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// SM100 (Blackwell) GEMM template -- CUTLASS 3.x TMA warp-specialized.
//
// Uses CUTLASS 3.x CollectiveBuilder pattern with TMA (Tensor Memory Accelerator)
// and wgmma instructions. Supports 1SM and 2SM cooperative scheduling modes.
//
// Reference: FlashInfer bf16_gemm_template_sm100.h

#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __GNUC__
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wstrict-aliasing"
    #pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include <cute/tensor.hpp>
#include <cutlass/arch/arch.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/util/device_memory.h>

// CUTLASS 3.x fusion operations
#include <cutlass/epilogue/fusion/operations.hpp>

#ifdef __GNUC__
    #pragma GCC diagnostic pop
#endif

#include <oasr/common/epilogue_functors.h>
#include <oasr/gemm/cutlass_gemm_configs.h>

namespace oasr {
namespace gemm {

//==============================================================================
// Sm100FusionOp -- maps OASR EpilogueFunctor to CUTLASS 3.x fusion operation
//==============================================================================

template <template <int, typename, typename> class EpilogueFn,
          typename ElementD, typename ElementCompute, typename ElementC = ElementD>
struct Sm100FusionOp {
    // Default: identity (linear combination)
    using type = cutlass::epilogue::fusion::LinearCombination<
        ElementD, ElementCompute, ElementC, ElementCompute>;
};

template <typename ElementD, typename ElementCompute, typename ElementC>
struct Sm100FusionOp<oasr::EpilogueIdentity, ElementD, ElementCompute, ElementC> {
    using type = cutlass::epilogue::fusion::LinearCombination<
        ElementD, ElementCompute, ElementC, ElementCompute>;
};

template <typename ElementD, typename ElementCompute, typename ElementC>
struct Sm100FusionOp<oasr::EpilogueRelu, ElementD, ElementCompute, ElementC> {
    using type = cutlass::epilogue::fusion::LinCombEltAct<
        cutlass::epilogue::thread::ReLu,
        ElementD, ElementCompute, ElementC, ElementCompute>;
};

template <typename ElementD, typename ElementCompute, typename ElementC>
struct Sm100FusionOp<oasr::EpilogueGelu, ElementD, ElementCompute, ElementC> {
    using type = cutlass::epilogue::fusion::LinCombEltAct<
        cutlass::epilogue::thread::GELU,
        ElementD, ElementCompute, ElementC, ElementCompute>;
};

template <typename ElementD, typename ElementCompute, typename ElementC>
struct Sm100FusionOp<oasr::EpilogueSwish, ElementD, ElementCompute, ElementC> {
    using type = cutlass::epilogue::fusion::LinCombEltAct<
        cutlass::epilogue::thread::SiLu,
        ElementD, ElementCompute, ElementC, ElementCompute>;
};

//==============================================================================
// SM scheduling type adapter (1SM / 2SM cooperative modes)
//==============================================================================

struct _1SM;
struct _2SM;

template <typename>
struct SMTypeAdapter;

template <>
struct SMTypeAdapter<_1SM> {
    static constexpr int Scale = 1;
    using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized1Sm;
    using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized1SmSm100;
};

template <>
struct SMTypeAdapter<_2SM> {
    static constexpr int Scale = 2;
    using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized2Sm;
    using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized2SmSm100;
};

//==============================================================================
// CutlassGemmKernelSm100 -- CUTLASS 3.x GEMM template for Blackwell
//==============================================================================

template <int32_t CTA_M_, int32_t CTA_N_, int32_t CTA_K_,
          typename ClusterShape_,
          typename ElementA, typename ElementB, typename ElementCD,
          typename LayoutA, typename LayoutB, typename LayoutCD,
          template <int, typename, typename> class EpilogueFunctor,
          typename SmSchedule = _1SM>
struct CutlassGemmKernelSm100 {
    using ElementAccumulator = float;
    using ElementCompute = float;
    using ArchTag = cutlass::arch::Sm100;
    using OperatorClass = cutlass::arch::OpClassTensorOp;

    // Tile shape: M dimension is scaled by SM count (1SM or 2SM)
    static constexpr int ScaledM = CTA_M_ * SMTypeAdapter<SmSchedule>::Scale;
    using TileShape = cute::Shape<cute::Int<ScaledM>, cute::Int<CTA_N_>, cute::Int<CTA_K_>>;
    using ClusterShape = ClusterShape_;

    // Alignment (128-bit = 16 bytes)
    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
    static constexpr int AlignmentCD = 128 / cutlass::sizeof_bits<ElementCD>::value;

    // Schedule types from SMTypeAdapter
    using EpilogueSchedule = typename SMTypeAdapter<SmSchedule>::EpilogueSchedule;
    using MainloopSchedule = typename SMTypeAdapter<SmSchedule>::MainloopSchedule;
    using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;

    // CUTLASS 3.x fusion operation for epilogue (identity, relu, gelu, swish)
    using FusionOp = typename Sm100FusionOp<EpilogueFunctor, ElementCD, ElementCompute, ElementCD>::type;

    // Build epilogue collective via CUTLASS 3.x builder
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass, TileShape, ClusterShape, EpilogueTileType,
        ElementAccumulator, ElementCompute,
        ElementCD, LayoutCD, AlignmentCD,
        ElementCD, LayoutCD, AlignmentCD,
        EpilogueSchedule,
        FusionOp>::CollectiveOp;

    // Build mainloop collective via CUTLASS 3.x builder
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ElementA, LayoutA, AlignmentA,
        ElementB, LayoutB, AlignmentB,
        ElementAccumulator,
        TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        MainloopSchedule>::CollectiveOp;

    // Assemble kernel and device adapter
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int, int, int, int>,
        CollectiveMainloop, CollectiveEpilogue>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    // Stride types from kernel
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;

    static GemmStatus run(const ElementA* A, const ElementB* B,
                          const ElementCD* C, ElementCD* D,
                          int M, int N, int K,
                          ElementCompute alpha, ElementCompute beta,
                          cudaStream_t stream) {
        // Compute strides (batch dimension = 1 for standard GEMM)
        auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
        auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
        auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
        auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

        typename Gemm::Arguments arguments{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K, 1},  // problem shape (M, N, K, batch=1)
            {A, stride_A, B, stride_B},
            {{}, C, stride_C, D, stride_D}};

        // Set epilogue scaling factors
        arguments.epilogue.thread.alpha = alpha;
        arguments.epilogue.thread.beta = beta;

        Gemm gemm;

        // Query workspace size and allocate
        size_t workspace_size = gemm.get_workspace_size(arguments);
        cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

        cutlass::Status status = gemm.can_implement(arguments);
        if (status != cutlass::Status::kSuccess) {
            return GemmStatus::NOT_SUPPORTED;
        }

        status = gemm.initialize(arguments, workspace.get(), stream);
        if (status != cutlass::Status::kSuccess) {
            return GemmStatus::INTERNAL_ERROR;
        }

        status = gemm.run(stream);
        if (status != cutlass::Status::kSuccess) {
            return GemmStatus::CUTLASS_ERROR;
        }

        return GemmStatus::SUCCESS;
    }
};

}  // namespace gemm
}  // namespace oasr
