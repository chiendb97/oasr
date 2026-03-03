// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Grouped GEMM kernel implementations using CUTLASS
// Supports variable-sized problems with BF16 and FP16 precision

#include "gemm_utils.h"
#include "group_gemm_kernels.h"

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

#include <stdexcept>

namespace oasr {
namespace kernels {
namespace gemm {

//==============================================================================
// Grouped GEMM Implementation (SM80)
//==============================================================================

template <typename ElementA, typename ElementB, typename ElementCD, typename LayoutA,
          typename LayoutB, typename LayoutCD>
struct CutlassGroupGemmSM80 {
    using ElementAccumulator = float;
    using ElementComputeEpilogue = ElementAccumulator;

    constexpr static int kAlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    constexpr static int kAlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
    constexpr static int kAlignmentCD = 128 / cutlass::sizeof_bits<ElementCD>::value;

    using MMAOp = cutlass::arch::OpClassTensorOp;

    using SmArch = cutlass::arch::Sm80;

    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;
    using ShapeMMAWarps = cutlass::gemm::GemmShape<64, 64, 32>;
    using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;

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
            problem_desc.ldd_device.get(), problem_desc.ldd_device.get()
        );

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

//==============================================================================
// Public API Implementations
//==============================================================================

torch::Tensor invokeGroupGemm(const torch::Tensor& A, const torch::Tensor& B,
                              const torch::Tensor& offset, cudaStream_t stream) {
    int problem_count = offset.size(0);
    auto L = A.size(0);
    auto N = B.size(1);
    auto K = A.size(1);

    auto D = torch::empty({L, N}, A.options());

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutCD = cutlass::layout::RowMajor;

    GemmStatus status = GemmStatus::SUCCESS;

    if (A.dtype() == torch::kFloat16) {
        GroupedGemmProblemDesc<cutlass::half_t, cutlass::half_t, cutlass::half_t> problem_desc(
            problem_count, L, K, N, A, B, offset, D);
        status = CutlassGroupGemmSM80<cutlass::half_t, cutlass::half_t, cutlass::half_t, LayoutA,
                                      LayoutB, LayoutCD>::run(problem_desc, problem_count, stream);
    } else if (A.dtype() == torch::kBFloat16) {
        GroupedGemmProblemDesc<cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t>
            problem_desc(problem_count, L, K, N, A, B, offset, D);
        status = CutlassGroupGemmSM80<cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t,
                                      LayoutA, LayoutB, LayoutCD>::run(problem_desc, problem_count,
                                                                       stream);
    } else {
        throw std::invalid_argument("Invalid input tensor type");
    }
    if (status != GemmStatus::SUCCESS) {
        throw std::runtime_error("Grouped GEMM failed");
    }

    return D;
}

}  // namespace gemm
}  // namespace kernels
}  // namespace oasr
