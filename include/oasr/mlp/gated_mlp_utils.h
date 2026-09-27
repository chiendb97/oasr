// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// The one CuTe helper that belongs to the fused gated MLP alone: the dual-B
// gemm that is the whole point of the fusion.
//
// Everything else this family's collectives use -- the swizzled shared-memory
// atom, the accumulator re-view, the packed down-conversion, the zero-filling
// and the skipping tiled copies -- is the Ampere-class toolkit shared with the
// attention and recurrent-step families (`include/oasr/common/cute_sm80.h`),
// and is spelled `cute_sm80::` at every call site so its provenance is
// visible.

#pragma once

#include <oasr/common/cute_sm80.h>

#include "cutlass_gated_mlp_configs.h"

namespace oasr {
namespace mlp {

using namespace cute;

// ---------------------------------------------------------------------------
// The dual-B gemm
// ---------------------------------------------------------------------------

/*! \brief `acc0 += A @ B0` and `acc1 += A @ B1` over one K tile, A read once.
 *
 * This is CUTLASS's `examples/45_dual_gemm` mainloop in the FlashAttention
 * idiom, and it is the reason a gated MLP is one kernel rather than two: the
 * gate and the up projection differ only in B, so A's `ldmatrix` is issued
 * once and fed to both `mma.sync` chains.
 *
 * The `ldmatrix` for k-step `i+1` is issued before step `i`'s MMAs, so the
 * shared-memory read overlaps the tensor-core work.  Two independent
 * accumulator chains also give the scheduler real work to fill the pipe with,
 * which is latency the single-B version has to hide with the prefetch alone.
 *
 * \param fn fires once, after the first k-step's `ldmatrix` has been issued.
 *   The mainloop uses it to launch the next stage's cp.async so that copy
 *   overlaps the MMA chain instead of sitting behind it.  At the shipped tiles
 *   that chain is only two to four k-steps long and the placement measures
 *   within noise of issuing the copy after the gemm; it is kept because it is
 *   the structure a longer chain wants and it costs nothing.
 */
template <typename Acc0, typename Acc1, typename FrgA, typename FrgB, typename SmemA,
          typename SmemB, typename TiledMma, typename TiledCopyA, typename TiledCopyB,
          typename ThrCopyA, typename ThrCopyB, typename Hook>
CUTLASS_DEVICE void gemm_dual_sm80(Acc0& acc0, Acc1& acc1, FrgA& tCrA, FrgB& tCrB0,
                                   FrgB& tCrB1, SmemA const& tCsA, SmemB const& tCsB0,
                                   SmemB const& tCsB1, TiledMma tiled_mma,
                                   TiledCopyA const& smem_tiled_copy_A,
                                   TiledCopyB const& smem_tiled_copy_B,
                                   ThrCopyA const& smem_thr_copy_A,
                                   ThrCopyB const& smem_thr_copy_B, Hook fn) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc0));
    CUTE_STATIC_ASSERT_V(size<1>(tCrB0) == size<2>(acc0));
    CUTE_STATIC_ASSERT_V(size<1>(tCrB1) == size<2>(acc1));
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB0));
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB1));
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    Tensor tCrB0_copy_view = smem_thr_copy_B.retile_D(tCrB0);
    Tensor tCrB1_copy_view = smem_thr_copy_B.retile_D(tCrB1);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));
    CUTE_STATIC_ASSERT_V(size<1>(tCsB0) == size<1>(tCrB0_copy_view));
    cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{}));
    cute::copy(smem_tiled_copy_B, tCsB0(_, _, _0{}), tCrB0_copy_view(_, _, _0{}));
    cute::copy(smem_tiled_copy_B, tCsB1(_, _, _0{}), tCrB1_copy_view(_, _, _0{}));
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1));
            cute::copy(smem_tiled_copy_B, tCsB0(_, _, i + 1), tCrB0_copy_view(_, _, i + 1));
            cute::copy(smem_tiled_copy_B, tCsB1(_, _, i + 1), tCrB1_copy_view(_, _, i + 1));
        }
        if (i == 0) {
            fn();
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB0(_, _, i), acc0);
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB1(_, _, i), acc1);
    }
}

}  // namespace mlp
}  // namespace oasr
