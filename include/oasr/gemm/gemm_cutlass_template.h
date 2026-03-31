// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// CUTLASS GEMM kernel template — parameterized by GemmConfig + SmMMATraits.
//
// This is the core CUTLASS 2.x GEMM implementation. Config provides tile
// dimensions and pipeline stages; MMATraits provides hardware-specific MMA
// shape, op class, and SM architecture tag.

#pragma once

#include <cstdint>

#ifdef __GNUC__
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wstrict-aliasing"
    #pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/device_memory.h>

#ifdef __GNUC__
    #pragma GCC diagnostic pop
#endif

#include <oasr/common/epilogue_functors.h>
#include <oasr/common/utils.h>
#include <oasr/gemm/cutlass_gemm_configs.h>

namespace oasr {
namespace gemm {

//==============================================================================
// CutlassGemmKernel — CUTLASS 2.x GEMM template
//==============================================================================

template <typename CutlassGemmConfig, typename ElementA, typename ElementB, typename ElementCD,
          ActivationType activation_type>
struct CutlassGemmKernel {
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutCD = cutlass::layout::RowMajor;
    using ElementAccumulator = float;
    using ElementComputeEpilogue = ElementAccumulator;

    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
    static constexpr int AlignmentEpilogue = 128 / cutlass::sizeof_bits<ElementCD>::value;

    using MMAOp = cutlass::arch::OpClassTensorOp;
    using SmArch = typename CutlassGemmConfig::SmArch;

    using ThreadblockShape = typename CutlassGemmConfig::ThreadblockShape;
    using WarpShape = typename CutlassGemmConfig::WarpShape;
    using InstructionShape = typename CutlassGemmConfig::InstructionShape;

    using SwizzleThreadblock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

    using EpilogueOp = typename FusionEpilogueOp<activation_type, AlignmentEpilogue, ElementCD,
                                                 ElementComputeEpilogue, ElementCD>::type;

    static constexpr int Stages = CutlassGemmConfig::Stages;

    using Gemm =
        cutlass::gemm::device::Gemm<ElementA, LayoutA, ElementB, LayoutB, ElementCD, LayoutCD,
                                    ElementAccumulator, MMAOp, SmArch, ThreadblockShape, WarpShape,
                                    InstructionShape, EpilogueOp, SwizzleThreadblock, Stages,
                                    AlignmentA, AlignmentB>;

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

template <typename CutlassGemmConfig, typename ElementA, typename ElementB, typename ElementCD>
struct CutlassBmmKernel {
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutCD = cutlass::layout::RowMajor;
    using ElementAccumulator = float;
    using ElementComputeEpilogue = ElementAccumulator;

    using MMAOp = cutlass::arch::OpClassTensorOp;
    using SmArch = typename CutlassGemmConfig::SmArch;

    using ThreadblockShape = typename CutlassGemmConfig::ThreadblockShape;
    using WarpShape = typename CutlassGemmConfig::WarpShape;
    using InstructionShape = typename CutlassGemmConfig::InstructionShape;

    using SwizzleThreadblock = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;

    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
    static constexpr int AlignmentEpilogue = 128 / cutlass::sizeof_bits<ElementCD>::value;

    using EpilogueOp =
        cutlass::epilogue::thread::LinearCombination<ElementCD, AlignmentEpilogue,
                                                     ElementComputeEpilogue, ElementComputeEpilogue,
                                                     cutlass::epilogue::thread::ScaleType::Default>;

    static constexpr int NumStages = CutlassGemmConfig::NumStages;

    using Gemm = cutlass::gemm::device::GemmBatched<
        ElementA, LayoutA, ElementB, LayoutB, ElementCD, LayoutCD, ElementAccumulator, MMAOp,
        SmArch, ThreadblockShape, WarpShape, InstructionShape, EpilogueOp, SwizzleThreadblock,
        NumStages, AlignmentA, AlignmentB>;

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
// CUTLASS Grouped GEMM Template
//==============================================================================

template <typename ElementA, typename ElementB, typename ElementCD>
struct GroupedGemmProblemDesc {
    std::vector<cutlass::gemm::GemmCoord> problem_sizes;
    cutlass::DeviceAllocation<cutlass::gemm::GemmCoord> problems_sizes_device;

    cutlass::DeviceAllocation<ElementA*> ptr_A_device;
    cutlass::DeviceAllocation<ElementB*> ptr_B_device;
    cutlass::DeviceAllocation<ElementCD*> ptr_D_device;

    cutlass::DeviceAllocation<int64_t> lda_device;
    cutlass::DeviceAllocation<int64_t> ldb_device;
    cutlass::DeviceAllocation<int64_t> ldd_device;

    GroupedGemmProblemDesc(int problem_count, int K, int N, const ElementA* A_ptr,
                           const ElementB* B_ptr, ElementCD* D_ptr, const int* offsets_host)
        : problem_sizes(problem_count),
          problems_sizes_device(problem_count),
          ptr_A_device(problem_count),
          ptr_B_device(problem_count),
          ptr_D_device(problem_count),
          lda_device(problem_count),
          ldb_device(problem_count),
          ldd_device(problem_count) {
        std::vector<ElementA*> ptr_A(problem_count);
        std::vector<ElementB*> ptr_B(problem_count);
        std::vector<ElementCD*> ptr_D(problem_count);
        std::vector<int64_t> lda(problem_count);
        std::vector<int64_t> ldb(problem_count);
        std::vector<int64_t> ldd(problem_count);

        int offset_M = 0;
        for (int i = 0; i < problem_count; ++i) {
            int next_offset_M = offsets_host[i];
            int M = next_offset_M - offset_M;
            problem_sizes[i] = cutlass::gemm::GemmCoord(M, N, K);
            lda[i] = K;
            ldb[i] = K;
            ldd[i] = N;

            ptr_A[i] = const_cast<ElementA*>(A_ptr) + static_cast<int64_t>(offset_M) * K;
            ptr_B[i] = const_cast<ElementB*>(B_ptr) + static_cast<int64_t>(i) * N * K;
            ptr_D[i] = D_ptr + static_cast<int64_t>(offset_M) * N;

            offset_M = next_offset_M;
        }

        problems_sizes_device.copy_from_host(problem_sizes.data());
        ptr_A_device.copy_from_host(ptr_A.data());
        ptr_B_device.copy_from_host(ptr_B.data());
        ptr_D_device.copy_from_host(ptr_D.data());
        lda_device.copy_from_host(lda.data());
        ldb_device.copy_from_host(ldb.data());
        ldd_device.copy_from_host(ldd.data());
    }
};

template <typename CutlassGemmConfig, typename ElementA, typename ElementB, typename ElementCD>
struct CutlassGroupGemmKernel {
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutCD = cutlass::layout::RowMajor;
    using ElementAccumulator = float;
    using ElementComputeEpilogue = ElementAccumulator;

    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
    static constexpr int AlignmentEpilogue = 128 / cutlass::sizeof_bits<ElementCD>::value;

    using MMAOp = cutlass::arch::OpClassTensorOp;
    using SmArch = typename CutlassGemmConfig::SmArch;

    using ThreadblockShape = typename CutlassGemmConfig::ThreadblockShape;
    using WarpShape = typename CutlassGemmConfig::WarpShape;
    using InstructionShape = typename CutlassGemmConfig::InstructionShape;

    using SwizzleThreadblock = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;

    using EpilogueOp =
        cutlass::epilogue::thread::LinearCombination<ElementCD, AlignmentEpilogue,
                                                     ElementAccumulator, ElementAccumulator>;

    static constexpr int NumStages = CutlassGemmConfig::NumStages;

    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
        ElementA, LayoutA, cutlass::ComplexTransform::kNone, AlignmentA, ElementB, LayoutB,
        cutlass::ComplexTransform::kNone, AlignmentB, ElementCD, LayoutCD, ElementAccumulator,
        MMAOp, SmArch, ThreadblockShape, WarpShape, InstructionShape, EpilogueOp,
        SwizzleThreadblock, NumStages>::GemmKernel;

    using Gemm = cutlass::gemm::device::GemmGrouped<GemmKernel>;

    static GemmStatus run(GroupedGemmProblemDesc<ElementA, ElementB, ElementCD>& problem_desc,
                          int problem_count, cudaStream_t stream) {
        typename EpilogueOp::Params epilogue_params(1.0f, 0.0f);

        int threadblock_count = Gemm::sufficient(problem_desc.problem_sizes.data(), problem_count);
        typename Gemm::Arguments args(
            problem_desc.problems_sizes_device.get(), problem_count, threadblock_count,
            epilogue_params, problem_desc.ptr_A_device.get(), problem_desc.ptr_B_device.get(),
            problem_desc.ptr_D_device.get(), problem_desc.ptr_D_device.get(),
            problem_desc.lda_device.get(), problem_desc.ldb_device.get(),
            problem_desc.ldd_device.get(), problem_desc.ldd_device.get());

        Gemm gemm_op;
        cutlass::Status status = gemm_op.can_implement(args);
        if (status != cutlass::Status::kSuccess) {
            return GemmStatus::NOT_SUPPORTED;
        }

        size_t workspace_size = gemm_op.get_workspace_size(args);
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

}  // namespace gemm
}  // namespace oasr
