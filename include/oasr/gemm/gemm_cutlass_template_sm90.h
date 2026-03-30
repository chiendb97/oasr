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
#include <oasr/gemm/gemm_cutlass_template.h>

#include <cutlass/gemm/group_array_problem_shape.hpp>

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

template <typename CutlassGemmConfig, typename ElementA, typename ElementB, typename ElementCD>
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

    // Linear combination epilogue (no activation for BMM)
    using FusionOp = cutlass::epilogue::fusion::LinearCombination<ElementCD, ElementCompute,
                                                                   ElementCD, ElementCompute>;

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

    static GemmStatus run(const ElementA* A, const ElementB* B, ElementCD* D, int batch_size,
                          int M, int N, int K, int64_t lda, int64_t ldb, int64_t ldd,
                          int64_t stride_a, int64_t stride_b, int64_t stride_d, float alpha,
                          float beta, cudaStream_t stream) {
        // Compute strides with batch dimension
        auto cute_stride_A =
            cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, batch_size));
        auto cute_stride_B =
            cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, batch_size));
        auto cute_stride_D =
            cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, batch_size));

        typename Gemm::Arguments arguments{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K, batch_size},
            {A, cute_stride_A, B, cute_stride_B},
            {{}, D, cute_stride_D, D, cute_stride_D}};

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
//
// Uses ptr-array TMA warp-specialized cooperative scheduling with
// GroupProblemShape for variable-size grouped GEMM.
//==============================================================================

template <typename CutlassGemmConfig, typename ElementA, typename ElementB, typename ElementCD>
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

    // Ptr-array schedule types for grouped GEMM (different from regular GEMM schedules)
    using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
    using MainloopSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
    using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;

    // Linear combination epilogue (no activation for group GEMM)
    using FusionOp = cutlass::epilogue::fusion::LinearCombination<ElementCD, ElementCompute,
                                                                   ElementCD, ElementCompute>;

    // Group problem shape for variable-size grouped GEMM
    using ProblemShape = cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>;

    // Build epilogue collective via CUTLASS 3.x builder (ptr-array: LayoutCD *)
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass, TileShape, ClusterShape, EpilogueTileType, ElementAccumulator,
        ElementCompute, ElementCD, LayoutCD*, AlignmentCD, ElementCD, LayoutCD*, AlignmentCD,
        EpilogueSchedule, FusionOp>::CollectiveOp;

    // Build mainloop collective via CUTLASS 3.x builder (ptr-array: LayoutA *, LayoutB *)
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass, ElementA, LayoutA*, AlignmentA, ElementB, LayoutB*, AlignmentB,
        ElementAccumulator, TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        MainloopSchedule>::CollectiveOp;

    // Assemble kernel and device adapter with GroupProblemShape
    using GemmKernel =
        cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    // Stride types from kernel (these are per-group stride types)
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;

    static GemmStatus run(GroupedGemmProblemDesc<ElementA, ElementB, ElementCD>& problem_desc,
                          int problem_count, cudaStream_t stream) {
        // Build host-side arrays of problem shapes and strides
        std::vector<typename ProblemShape::UnderlyingProblemShape> problem_shapes_host(problem_count);
        std::vector<StrideA> strides_A_host(problem_count);
        std::vector<StrideB> strides_B_host(problem_count);
        std::vector<StrideD> strides_D_host(problem_count);

        for (int i = 0; i < problem_count; ++i) {
            auto& ps = problem_desc.problem_sizes[i];
            int M = ps.m(), N = ps.n(), K = ps.k();
            problem_shapes_host[i] = cute::make_shape(M, N, K);
            strides_A_host[i] =
                cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
            strides_B_host[i] =
                cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
            strides_D_host[i] =
                cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));
        }

        // Allocate and copy to device
        cutlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape>
            problem_shapes_device(problem_count);
        cutlass::DeviceAllocation<StrideA> strides_A_device(problem_count);
        cutlass::DeviceAllocation<StrideB> strides_B_device(problem_count);
        cutlass::DeviceAllocation<StrideD> strides_D_device(problem_count);

        problem_shapes_device.copy_from_host(problem_shapes_host.data());
        strides_A_device.copy_from_host(strides_A_host.data());
        strides_B_device.copy_from_host(strides_B_host.data());
        strides_D_device.copy_from_host(strides_D_host.data());

        // Build arguments for grouped GEMM
        typename Gemm::Arguments arguments{
            cutlass::gemm::GemmUniversalMode::kGrouped,
            {problem_count, problem_shapes_device.get(), problem_shapes_host.data()},
            {reinterpret_cast<const ElementA**>(problem_desc.ptr_A_device.get()),
             strides_A_device.get(),
             reinterpret_cast<const ElementB**>(problem_desc.ptr_B_device.get()),
             strides_B_device.get()},
            {{1.0f, 0.0f},
             nullptr, strides_D_device.get(),
             reinterpret_cast<ElementCD**>(problem_desc.ptr_D_device.get()),
             strides_D_device.get()}};

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
