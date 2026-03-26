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
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/device_memory.h>

#ifdef __GNUC__
    #pragma GCC diagnostic pop
#endif

#include <oasr/gemm/cutlass_gemm_configs.h>

namespace oasr {
namespace gemm {

//==============================================================================
// CutlassGemmKernel — CUTLASS 2.x GEMM template
//==============================================================================

template <typename Config, typename MMATraits, typename ElementA, typename ElementB,
          typename ElementCD, typename LayoutA, typename LayoutB, typename LayoutCD,
          template <int, typename, typename> class EpilogueFunctor>
struct CutlassGemmKernel {
    using ElementAccumulator = float;
    using ElementComputeEpilogue = ElementAccumulator;

    static constexpr int EpilogueAlignment = 128 / cutlass::sizeof_bits<ElementCD>::value;

    using MMAOp = typename MMATraits::MMAOp;
    using SmArch = typename MMATraits::SmArch;

    using ShapeMMAThreadBlock = typename Config::ThreadBlock;
    using ShapeMMAWarps = typename Config::Warps;
    using ShapeMMAOp = typename MMATraits::MMAShape;

    using SwizzleThreadblock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

    using EpilogueOp =
        typename EpilogueFunctor<EpilogueAlignment, ElementCD, ElementComputeEpilogue>::Op;

    static constexpr int NumStages = Config::NumStages;

    using Gemm = cutlass::gemm::device::Gemm<ElementA, LayoutA, ElementB, LayoutB, ElementCD,
                                              LayoutCD, ElementAccumulator, MMAOp, SmArch,
                                              ShapeMMAThreadBlock, ShapeMMAWarps, ShapeMMAOp,
                                              EpilogueOp, SwizzleThreadblock, NumStages>;

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

}  // namespace gemm
}  // namespace oasr
