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
// Declares cutlass::make_cute_packed_stride, used below.  Qualified lookup
// happens at parse time even inside a template, so without this every
// translation unit that merely includes this header failed to build.
#include <cutlass/util/packed_stride.hpp>

#include <cute/tensor.hpp>

#ifdef __GNUC__
    #pragma GCC diagnostic pop
#endif

#include <cutlass/gemm/group_array_problem_shape.hpp>

#include <oasr/common/epilogue_functors.h>
#include <oasr/common/graph_safe_workspace.h>
#include <oasr/common/utils.h>
#include <oasr/gemm/cutlass_gemm_configs.h>
#include <oasr/gemm/gemm_cutlass_template.h>

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
    static constexpr int AlignmentEpilogue = 128 / cutlass::sizeof_bits<ElementCD>::value;

    // Schedule types from SMTypeAdapter
    using EpilogueSchedule = typename CutlassGemmConfig::EpilogueSchedule;
    using MainloopSchedule = typename CutlassGemmConfig::MainloopSchedule;
    using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;

    // D = activation(alpha * (A @ B) + bias[n]).  The bias arrives as a fusion
    // input rather than the C operand, because the 2.x path's C is a length-N
    // row with a zero M-stride and TMA has no zero-stride mode.  ElementC is
    // therefore `void`: there is no source matrix to load, which also means no
    // TMA descriptor is built over a possibly-null bias pointer.
    using FusionOp = typename FusionEpilogueOpSm90PerColBias<activation_type, ElementCD,
                                                            ElementCompute, ElementCD>::type;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass, TileShape, ClusterShape, EpilogueTileType, ElementAccumulator,
        ElementCompute, void, LayoutCD, AlignmentEpilogue, ElementCD, LayoutCD, AlignmentEpilogue,
        EpilogueSchedule, FusionOp>::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass, ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
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

    /// Parameter list mirrors the CUTLASS 2.x `CutlassGemmKernel::run`, so the
    /// two are interchangeable at a dispatch site.  \p C is the length-N bias
    /// row (or null), \p ldc its unused-but-kept leading dimension.
    static GemmStatus run(const ElementA* A, const ElementB* B, const ElementCD* C, ElementCD* D,
                          int M, int N, int K, int64_t lda, int64_t ldb, int64_t ldc,
                          ElementCompute alpha, cudaStream_t stream, int split_k_slices = 1,
                          bool broadcast_c = true, int64_t ldd = -1) {
        // A matrix-valued C is different arithmetic from a broadcast bias row,
        // so refuse it rather than quietly computing the other thing.  (The
        // recurrent family needs exactly that and builds its own 3.x epilogue
        // for it: oasr/recurrent/recurrent_cutlass_sm90.cuh.)
        if (!broadcast_c) {
            return GemmStatus::NOT_SUPPORTED;
        }
        // split_k_slices is accepted and unused on purpose: the collective
        // mainloop pipelines K itself, so there is no serial or parallel split
        // to arrange, and the result is the same either way.  Refusing would
        // break callers for whom this is only ever a performance hint.
        (void)split_k_slices;

        // Build packed strides so the static and batch modes get the right
        // types, then substitute the caller's leading dimensions.  Mode 0 is the
        // dynamic one for A(M,K,L) row-major, B(N,K,L) column-major and
        // D(M,N,L) row-major alike -- see cutlass/detail/layout.hpp.
        StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
        StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
        StrideD stride_D =
            cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));
        cute::get<0>(stride_A) = lda;
        cute::get<0>(stride_B) = ldb;
        cute::get<0>(stride_D) = (ldd < 0) ? ldc : ldd;

        typename Gemm::Arguments arguments{cutlass::gemm::GemmUniversalMode::kGemm,
                                           {M, N, K, 1},  // problem shape (M, N, K, batch=1)
                                           {A, stride_A, B, stride_B},
                                           {{}, nullptr, StrideC{}, D, stride_D}};

        arguments.epilogue.thread.alpha = alpha;
        // beta multiplies the (absent) source matrix; the bias rides its own
        // leaf, where a null pointer contributes a literal zero.
        arguments.epilogue.thread.beta = 0.0f;
        arguments.epilogue.thread.bias_ptr = C;

        Gemm gemm;

        size_t workspace_size = gemm.get_workspace_size(arguments);
        oasr::GraphSafeWorkspace workspace(workspace_size, stream);

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
    static constexpr int AlignmentEpilogue = 128 / cutlass::sizeof_bits<ElementCD>::value;

    // Schedule types from SMTypeAdapter
    using EpilogueSchedule = typename CutlassGemmConfig::EpilogueSchedule;
    using MainloopSchedule = typename CutlassGemmConfig::MainloopSchedule;
    using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;

    // Linear combination epilogue (no activation for BMM)
    using FusionOp = cutlass::epilogue::fusion::LinearCombination<ElementCD, ElementCompute,
                                                                  ElementCD, ElementCompute>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass, TileShape, ClusterShape, EpilogueTileType, ElementAccumulator,
        ElementCompute, ElementCD, LayoutCD, AlignmentEpilogue, ElementCD, LayoutCD,
        AlignmentEpilogue, EpilogueSchedule, FusionOp>::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass, ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
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

    static GemmStatus run(const ElementA* A, const ElementB* B, ElementCD* D, int batch_size, int M,
                          int N, int K, int64_t lda, int64_t ldb, int64_t ldd, int64_t stride_a,
                          int64_t stride_b, int64_t stride_d, float alpha, float beta,
                          cudaStream_t stream) {
        auto cute_stride_A =
            cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, batch_size));
        auto cute_stride_B =
            cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, batch_size));
        auto cute_stride_D =
            cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, batch_size));

        typename Gemm::Arguments arguments{cutlass::gemm::GemmUniversalMode::kGemm,
                                           {M, N, K, batch_size},
                                           {A, cute_stride_A, B, cute_stride_B},
                                           {{}, D, cute_stride_D, D, cute_stride_D}};

        arguments.epilogue.thread.alpha = alpha;
        arguments.epilogue.thread.beta = beta;

        Gemm gemm;

        size_t workspace_size = gemm.get_workspace_size(arguments);
        oasr::GraphSafeWorkspace workspace(workspace_size, stream);

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
    static constexpr int AlignmentEpilogue = 128 / cutlass::sizeof_bits<ElementCD>::value;

    // Ptr-array schedule types for grouped GEMM (different from regular GEMM schedules)
    using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
    using MainloopSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
    using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;

    // Linear combination epilogue (no activation for group GEMM)
    using FusionOp = cutlass::epilogue::fusion::LinearCombination<ElementCD, ElementCompute,
                                                                  ElementCD, ElementCompute>;

    // Group problem shape for variable-size grouped GEMM
    using ProblemShape = cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>;

    // No C operand: GroupedGemmProblemDesc carries A, B and D only.  void
    // ElementC also keeps the epilogue from building a TMA descriptor over a C
    // pointer array that does not exist.
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass, TileShape, ClusterShape, EpilogueTileType, ElementAccumulator,
        ElementCompute, void, LayoutCD*, AlignmentEpilogue, ElementCD, LayoutCD*,
        AlignmentEpilogue, EpilogueSchedule, FusionOp>::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass, ElementA, LayoutA*, AlignmentA, ElementB, LayoutB*, AlignmentB,
        ElementAccumulator, TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        MainloopSchedule>::CollectiveOp;

    using GemmKernel =
        cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    // A grouped kernel's StrideA/B/C/D are *pointer* types -- one stride per
    // group -- so make_cute_packed_stride cannot be handed them.  The per-group
    // stride types are the Internal* aliases; the pointer types are what the
    // Arguments then take, which is exactly `strides_*_device.get()` below.
    using StrideA = typename Gemm::GemmKernel::InternalStrideA;
    using StrideB = typename Gemm::GemmKernel::InternalStrideB;
    using StrideC = typename Gemm::GemmKernel::InternalStrideC;
    using StrideD = typename Gemm::GemmKernel::InternalStrideD;

    static GemmStatus run(GroupedGemmProblemDesc<ElementA, ElementB, ElementCD>& problem_desc,
                          int problem_count, cudaStream_t stream) {
        std::vector<typename ProblemShape::UnderlyingProblemShape> problem_shapes_host(
            problem_count);
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

        cutlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape>
            problem_shapes_device(problem_count);
        cutlass::DeviceAllocation<StrideA> strides_A_device(problem_count);
        cutlass::DeviceAllocation<StrideB> strides_B_device(problem_count);
        cutlass::DeviceAllocation<StrideD> strides_D_device(problem_count);

        problem_shapes_device.copy_from_host(problem_shapes_host.data());
        strides_A_device.copy_from_host(strides_A_host.data());
        strides_B_device.copy_from_host(strides_B_host.data());
        strides_D_device.copy_from_host(strides_D_host.data());

        typename Gemm::Arguments arguments{
            cutlass::gemm::GemmUniversalMode::kGrouped,
            {problem_count, problem_shapes_device.get(), problem_shapes_host.data()},
            {const_cast<const ElementA**>(problem_desc.ptr_A_device.get()),
             strides_A_device.get(),
             const_cast<const ElementB**>(problem_desc.ptr_B_device.get()),
             strides_B_device.get()},
            // beta = 0 and no C: the epilogue is source-less (void ElementC), so
            // its C pointer array and stride array are both absent.
            {{1.0f, 0.0f},
             nullptr,
             nullptr,
             problem_desc.ptr_D_device.get(),
             strides_D_device.get()}};

        Gemm gemm;

        // Query workspace size and allocate
        size_t workspace_size = gemm.get_workspace_size(arguments);
        oasr::GraphSafeWorkspace workspace(workspace_size, stream);

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
