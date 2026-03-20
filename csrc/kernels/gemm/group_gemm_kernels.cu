// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Grouped GEMM kernel implementations using CUTLASS
// Supports variable-sized problems with BF16 and FP16 precision

#include <stdexcept>

#include "gemm_utils.h"
#include "group_gemm_impl.h"
#include "group_gemm_kernels.h"
#include "kernels/common/arch_dispatch.h"
#include "kernels/common/arch_traits.h"

namespace oasr {
namespace kernels {
namespace gemm {

template <int SmVersion>
static GemmStatus dispatchGroupGemmDtype(const torch::Tensor& A, const torch::Tensor& B,
                                         const torch::Tensor& offset, torch::Tensor& D,
                                         int problem_count, int64_t L, int64_t K, int64_t N,
                                         cudaStream_t stream) {
    if constexpr (SmVersion < 80) {
        throw std::runtime_error("Grouped GEMM requires SM80 or newer (current: SM" +
                                 std::to_string(SmVersion) + ")");
    } else {
        using Traits = ArchTraits<SmVersion>;
        using LayoutA = cutlass::layout::RowMajor;
        using LayoutB = cutlass::layout::ColumnMajor;
        using LayoutCD = cutlass::layout::RowMajor;

        if (A.dtype() == torch::kFloat16) {
            GroupedGemmProblemDesc<cutlass::half_t, cutlass::half_t, cutlass::half_t> problem_desc(
                problem_count, L, K, N, A, B, offset, D);
            return CutlassGroupGemm<Traits, cutlass::half_t, cutlass::half_t, cutlass::half_t,
                                    LayoutA, LayoutB, LayoutCD>::run(problem_desc, problem_count,
                                                                     stream);
        } else if (A.dtype() == torch::kBFloat16) {
            GroupedGemmProblemDesc<cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t>
                problem_desc(problem_count, L, K, N, A, B, offset, D);
            return CutlassGroupGemm<Traits, cutlass::bfloat16_t, cutlass::bfloat16_t,
                                    cutlass::bfloat16_t, LayoutA, LayoutB, LayoutCD>::run(
                problem_desc, problem_count, stream);
        }
        throw std::invalid_argument("Invalid input tensor type");
    }
}

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

    GemmStatus status = GemmStatus::SUCCESS;

    int sm = getDeviceSmVersion();
    OASR_DISPATCH_ARCH(sm, SM_VERSION, {
        status = dispatchGroupGemmDtype<SM_VERSION>(A, B, offset, D, problem_count, L, K, N, stream);
    });

    if (status != GemmStatus::SUCCESS) {
        throw std::runtime_error("Grouped GEMM failed");
    }

    return D;
}

}  // namespace gemm
}  // namespace kernels
}  // namespace oasr
