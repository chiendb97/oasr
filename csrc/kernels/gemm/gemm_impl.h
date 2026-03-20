// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Traits-parameterized GEMM template using CUTLASS 2.x API

#pragma once

#ifdef __GNUC__
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wstrict-aliasing"
    #pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/device_memory.h>

#ifdef __GNUC__
    #pragma GCC diagnostic pop
#endif

#include "gemm_utils.h"

namespace oasr {
namespace kernels {
namespace gemm {

template <typename Traits, typename ElementA, typename ElementB, typename ElementCD, typename LayoutA,
          typename LayoutB, typename LayoutCD,
          template <int, typename, typename> class EpilogueFunctor>
struct CutlassGemm {
    using ElementAccumulator = float;
    using ElementComputeEpilogue = ElementAccumulator;

    static constexpr int EpilogueAlignment = 128 / cutlass::sizeof_bits<ElementCD>::value;

    using MMAOp = typename Traits::MMAOp;
    using SmArch = typename Traits::SmArch;

    using ShapeMMAThreadBlock = typename Traits::Gemm::ThreadBlock;
    using ShapeMMAWarps = typename Traits::Gemm::Warps;
    using ShapeMMAOp = typename Traits::Gemm::MMAShape;

    using SwizzleThreadblock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

    using EpilogueOp =
        typename EpilogueFunctor<EpilogueAlignment, ElementCD, ElementComputeEpilogue>::Op;

    static constexpr int NumStages = Traits::Gemm::NumStages;

    using Gemm = cutlass::gemm::device::Gemm<ElementA, LayoutA, ElementB, LayoutB, ElementCD,
                                             LayoutCD, ElementAccumulator, MMAOp, SmArch,
                                             ShapeMMAThreadBlock, ShapeMMAWarps, ShapeMMAOp,
                                             EpilogueOp, SwizzleThreadblock, NumStages>;

    static GemmStatus run(const ElementA* A, const ElementB* B, const ElementCD* C, ElementCD* D,
                          int M, int N, int K, int64_t lda, int64_t ldb, int64_t ldc,
                          ElementComputeEpilogue alpha, cudaStream_t stream) {
        int split_k_slices = 1;
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

}  // namespace gemm
}  // namespace kernels
}  // namespace oasr
