// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Traits-parameterized Batched GEMM (BMM) template using CUTLASS 2.x API

#pragma once

#ifdef __GNUC__
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wstrict-aliasing"
    #pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm_batched.h>
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
          typename LayoutB, typename LayoutCD>
struct CutlassBmm {
    using ElementAccumulator = float;
    using ElementComputeEpilogue = ElementAccumulator;

    using MMAOp = typename Traits::MMAOp;
    using SmArch = typename Traits::SmArch;

    using ShapeMMAThreadBlock = typename Traits::Gemm::ThreadBlock;
    using ShapeMMAWarps = typename Traits::Gemm::Warps;
    using ShapeMMAOp = typename Traits::Gemm::MMAShape;

    using SwizzleThreadblock = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;

    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementCD, 128 / cutlass::sizeof_bits<ElementCD>::value, ElementComputeEpilogue,
        ElementComputeEpilogue, cutlass::epilogue::thread::ScaleType::Default>;

    static constexpr int NumStages = Traits::Gemm::NumStages;

    using Gemm = cutlass::gemm::device::GemmBatched<ElementA, LayoutA, ElementB, LayoutB, ElementCD,
                                                    LayoutCD, ElementAccumulator, MMAOp, SmArch,
                                                    ShapeMMAThreadBlock, ShapeMMAWarps, ShapeMMAOp,
                                                    EpilogueOp, SwizzleThreadblock, NumStages>;

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

}  // namespace gemm
}  // namespace kernels
}  // namespace oasr
