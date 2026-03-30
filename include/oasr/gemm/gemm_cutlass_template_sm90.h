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

#include <cutlass/arch/arch.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/util/device_memory.h>

#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/fusion/operations.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>

#include <cute/tensor.hpp>

#ifdef __GNUC__
    #pragma GCC diagnostic pop
#endif

#include <oasr/common/epilogue_functors.h>
#include <oasr/common/utils.h>
#include <oasr/gemm/cutlass_gemm_configs.h>

namespace oasr {
namespace gemm {

//==============================================================================
// CutlassGemmKernelSm100 -- CUTLASS 3.x GEMM template for Blackwell
//==============================================================================

template <typename CutlassGemmConfig, typename ElementA, typename ElementB, typename ElementCD,
          ActivationType activation_type>
struct CutlassGemmKernelSm90 {
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutCD = cutlass::layout::RowMajor;

    using ElementAccumulator = float;
    using ElementCompute = float;
    using ArchTag = typename CutlassGemmConfig::SmArch;

    using OperatorClass = cutlass::arch::OpClassTensorOp;

    using TileShape = typename CutlassGemmConfig::TileShape;
    using ClusterShape = typename CutlassGemmConfig::ClusterShape;

    // Alignment (128-bit = 16 bytes)
    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
    static constexpr int AlignmentCD = 128 / cutlass::sizeof_bits<ElementCD>::value;

    // Schedule types from SMTypeAdapter
    using EpilogueSchedule = typename CutlassGemmConfig::EpilogueSchedule;
    using MainloopSchedule = typename CutlassGemmConfig::MainloopSchedule;
    using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;

    // CUTLASS 3.x fusion operation for epilogue (identity, relu, gelu, swish)
    using FusionOp =
        typename FusionEpilogueOpSm90<activation_type, ElementCD, ElementCompute, ElementCD>::type;

    // Build epilogue collective via CUTLASS 3.x builder
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass, TileShape, ClusterShape, EpilogueTileType, ElementAccumulator,
        ElementCompute, ElementCD, LayoutCD, AlignmentCD, ElementCD, LayoutCD, AlignmentCD,
        EpilogueSchedule, FusionOp>::CollectiveOp;

    // Build mainloop collective via CUTLASS 3.x builder
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass, ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
        ElementAccumulator, TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        MainloopSchedule>::CollectiveOp;

    // Assemble kernel and device adapter
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>,
                                                            CollectiveMainloop, CollectiveEpilogue>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    // Stride types from kernel
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;

    static GemmStatus run(const ElementA* A, const ElementB* B, const ElementCD* C, ElementCD* D,
                          int M, int N, int K, ElementCompute alpha, ElementCompute beta,
                          cudaStream_t stream) {
        // Compute strides (batch dimension = 1 for standard GEMM)
        auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
        auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
        auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
        auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

        typename Gemm::Arguments arguments{cutlass::gemm::GemmUniversalMode::kGemm,
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

//==============================================================================
// CutlassBmmKernelSm90 -- CUTLASS 3.x BMM template for SM90+
//==============================================================================

template <typename CutlassGemmConfig, typename ElementA, typename ElementB, typename ElementCD,
          ActivationType activation_type>
struct CutlassBmmKernelSm90 {
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutCD = cutlass::layout::RowMajor;

    using ElementAccumulator = float;
    using ElementCompute = float;
    using ArchTag = typename CutlassGemmConfig::SmArch;

    using OperatorClass = cutlass::arch::OpClassTensorOp;

    using TileShape = typename CutlassGemmConfig::TileShape;
    using ClusterShape = typename CutlassGemmConfig::ClusterShape;

    // Alignment (128-bit = 16 bytes)
    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
    static constexpr int AlignmentCD = 128 / cutlass::sizeof_bits<ElementCD>::value;

    // Schedule types from SMTypeAdapter
    using EpilogueSchedule = typename CutlassGemmConfig::EpilogueSchedule;
    using MainloopSchedule = typename CutlassGemmConfig::MainloopSchedule;
    using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;

    // CUTLASS 3.x fusion operation for epilogue (identity, relu, gelu, swish)
    using FusionOp =
        typename FusionEpilogueOpSm90<activation_type, ElementCD, ElementCompute, ElementCD>::type;

    // Build epilogue collective via CUTLASS 3.x builder
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass, TileShape, ClusterShape, EpilogueTileType, ElementAccumulator,
        ElementCompute, ElementCD, LayoutCD, AlignmentCD, ElementCD, LayoutCD, AlignmentCD,
        EpilogueSchedule, FusionOp>::CollectiveOp;

    // Build mainloop collective via CUTLASS 3.x builder
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass, ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
        ElementAccumulator, TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        MainloopSchedule>::CollectiveOp;

    // Assemble kernel and device adapter
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>,
                                                            CollectiveMainloop, CollectiveEpilogue>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    // Stride types from kernel
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;

    static GemmStatus run(const ElementA* A, const ElementB* B, const ElementCD* C, ElementCD* D,
                          int M, int N, int K, ElementCompute alpha, ElementCompute beta,
                          cudaStream_t stream) {
        // Compute strides (batch dimension = 1 for standard GEMM)
        auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
        auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
        auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
        auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

        typename Gemm::Arguments arguments{cutlass::gemm::GemmUniversalMode::kGemm,
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

//==============================================================================
// CutlassGroupGemmKernelSm90 -- CUTLASS 3.x Group GEMM template for SM90+
//==============================================================================

template <typename CutlassGemmConfig, typename ElementA, typename ElementB, typename ElementCD,
          ActivationType activation_type>
struct CutlassGroupGemmKernelSm90 {
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutCD = cutlass::layout::RowMajor;

    using ElementAccumulator = float;
    using ElementCompute = float;
    using ArchTag = typename CutlassGemmConfig::SmArch;

    using OperatorClass = cutlass::arch::OpClassTensorOp;

    using TileShape = typename CutlassGemmConfig::TileShape;
    using ClusterShape = typename CutlassGemmConfig::ClusterShape;

    // Alignment (128-bit = 16 bytes)
    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
    static constexpr int AlignmentCD = 128 / cutlass::sizeof_bits<ElementCD>::value;

    // Schedule types from SMTypeAdapter
    using EpilogueSchedule = typename CutlassGemmConfig::EpilogueSchedule;
    using MainloopSchedule = typename CutlassGemmConfig::MainloopSchedule;
    using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;

    // CUTLASS 3.x fusion operation for epilogue (identity, relu, gelu, swish)
    using FusionOp =
        typename FusionEpilogueOpSm90<activation_type, ElementCD, ElementCompute, ElementCD>::type;

    // Build epilogue collective via CUTLASS 3.x builder
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass, TileShape, ClusterShape, EpilogueTileType, ElementAccumulator,
        ElementCompute, ElementCD, LayoutCD, AlignmentCD, ElementCD, LayoutCD, AlignmentCD,
        EpilogueSchedule, FusionOp>::CollectiveOp;

    // Build mainloop collective via CUTLASS 3.x builder
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass, ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
        ElementAccumulator, TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        MainloopSchedule>::CollectiveOp;

    // Assemble kernel and device adapter
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>,
                                                            CollectiveMainloop, CollectiveEpilogue>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    // Stride types from kernel
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;

    static GemmStatus run(const ElementA* A, const ElementB* B, const ElementCD* C, ElementCD* D,
                          int M, int N, int K, ElementCompute alpha, ElementCompute beta,
                          cudaStream_t stream) {
        // Compute strides (batch dimension = 1 for standard GEMM)
        auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
        auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
        auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
        auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

        typename Gemm::Arguments arguments{cutlass::gemm::GemmUniversalMode::kGemm,
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
