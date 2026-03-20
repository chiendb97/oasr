// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Traits-parameterized Grouped GEMM template using CUTLASS 2.x API

#pragma once

#ifdef __GNUC__
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wstrict-aliasing"
    #pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/device_memory.h>

#ifdef __GNUC__
    #pragma GCC diagnostic pop
#endif

#include "gemm_utils.h"
#include "group_gemm_kernels.h"

namespace oasr {
namespace kernels {
namespace gemm {

template <typename Traits, typename ElementA, typename ElementB, typename ElementCD, typename LayoutA,
          typename LayoutB, typename LayoutCD>
struct CutlassGroupGemm {
    using ElementAccumulator = float;
    using ElementComputeEpilogue = ElementAccumulator;

    static constexpr int kAlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static constexpr int kAlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
    static constexpr int kAlignmentCD = 128 / cutlass::sizeof_bits<ElementCD>::value;

    using MMAOp = typename Traits::MMAOp;
    using SmArch = typename Traits::SmArch;

    using ShapeMMAThreadBlock = typename Traits::Gemm::ThreadBlock;
    using ShapeMMAWarps = typename Traits::Gemm::Warps;
    using ShapeMMAOp = typename Traits::Gemm::MMAShape;

    using SwizzleThreadblock = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;

    using EpilogueOp =
        cutlass::epilogue::thread::LinearCombination<ElementCD, kAlignmentCD, ElementAccumulator,
                                                     ElementAccumulator>;

    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
        ElementA, LayoutA, cutlass::ComplexTransform::kNone, kAlignmentA, ElementB, LayoutB,
        cutlass::ComplexTransform::kNone, kAlignmentB, ElementCD, LayoutCD, ElementAccumulator,
        MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarps, ShapeMMAOp, EpilogueOp,
        SwizzleThreadblock, 4>::GemmKernel;

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
}  // namespace kernels
}  // namespace oasr
