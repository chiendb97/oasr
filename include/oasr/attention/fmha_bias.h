// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// The additive attention bias: a gmem-direct read straight into the MMA-C
// partition, and one FFMA into the score accumulator.
//
// This has no FlashAttention counterpart -- FA carries softcap and ALiBi, not a
// dense `(B, H, T_q, T_k)` additive bias -- but it occupies the same slot in
// the mainloop that FA's softcap does, and for the same reason: a score
// modification must land **before** the mask, because that is the only order in
// which a `-inf` stays `-inf`.
//
// It is the reason the fused kernel exists at all for OASR's encoders.  Every
// production caller passes one (Conformer's and Nemotron's relative-position
// `matrix_bd`, Paraformer's SANM), and it is what an SDPA fallback would have
// to materialise a `(B, H, T_q, T_k)` tensor to express.
//
// ---------------------------------------------------------------------------
// Two paths, and why both must exist
// ---------------------------------------------------------------------------
//
// The bias is partitioned through the *same* `partition_C` as the score
// accumulator, so no re-layout is needed -- element `i` of the fragment is
// element `i` of `acc_s`.  What differs is how it is read:
//
//  * **interior + vectorisable**: an unpredicated vector copy.  Adjacent
//    columns are adjacent in memory and every row start is 4-byte aligned, so
//    the compiler emits 32-bit pair loads.
//
//  * **boundary, or not vectorisable**: a per-element predicated copy into a
//    zero-filled fragment.  Slower -- the predicate breaks the vectorisation --
//    but it never *forms* an out-of-bounds address.
//
// The second path is not defensive programming.  The unpredicated read covers
// the tile's full `kBlockM x kBlockN` footprint, so on a boundary tile it
// addresses up to `kBlockM - 1` rows past the last real one; at a short packed
// segment (8x8 = 64 elements) a 64x64 tile reads ~500 elements past the block.
// The out-of-bounds *values* were always harmless -- the mask overwrites those
// score slots immediately after -- but the addresses are not optional, and this
// faulted for real whenever the allocator left the next page unmapped.
//
// `vectorizable` is a CTA-uniform runtime bool (`FmhaParams::
// bias_vectorizable`) rather than a template parameter.  Specialising on it
// would not remove the predicated path -- boundary tiles need it regardless --
// so it would only delete the *fast* path from half the compiled variants while
// doubling their count.

#pragma once

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>

#include "fmha_utils.h"

namespace oasr {
namespace attention {

using namespace cute;

/*! \brief `acc_s += bias * inv_softmax_scale`, for one Q-tile x K-tile block.
 *
 * \param mBias     this (batch, head)'s `(T_q, T_k)` bias plane
 * \param inv_scale `1 / softmax_scale`
 *
 * The scale division is what makes the bias a *post*-scale logit, matching
 * SDPA: the softmax computes `exp2((acc_s + bias/scale) * scale * log2e)`,
 * which collapses to `exp(scale * acc_s + bias)`.
 *
 * Predication is against the **bias tensor's own extents**, not against the
 * sequence lengths.  They can differ -- a paged caller's bias spans the whole
 * logical KV extent while `seqlen_k` is this stream's -- and it is the
 * allocation, not the sequence, that faults.
 */
template <int kBlockM, int kBlockN, class TiledMma, class EngineS, class LayoutS,
          class TensorB>
CUTLASS_DEVICE void add_bias_tile(Tensor<EngineS, LayoutS>& acc_s, TensorB const& mBias,
                                  TiledMma tiled_mma, int const thread_idx, int const m_block,
                                  int const n_block, bool const vectorizable,
                                  float const inv_scale) {
    using ElementBias = typename TensorB::value_type;
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);

    Tensor gBias = local_tile(mBias, Shape<Int<kBlockM>, Int<kBlockN>>{},
                              make_coord(m_block, n_block));
    Tensor tBias = thr_mma.partition_C(gBias);
    Tensor rBias = make_tensor<ElementBias>(shape(tBias));

    int const rows = int(size<0>(mBias));
    int const cols = int(size<1>(mBias));
    bool const interior =
        (m_block + 1) * kBlockM <= rows && (n_block + 1) * kBlockN <= cols;

    if (vectorizable && interior) {
        cute::copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<32>, ElementBias>{},
                   tBias, rBias);
    } else {
        cute::clear(rBias);
        Tensor cS = cute::make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
        Tensor tScS = thr_mma.partition_C(cS);
        Tensor tScS_rowcol =
            make_tensor(tScS.data(), convert_layout_acc_rowcol(tScS.layout()));
        Tensor tBias_rowcol =
            make_tensor(tBias.data(), convert_layout_acc_rowcol(tBias.layout()));
        Tensor rBias_rowcol =
            make_tensor(rBias.data(), convert_layout_acc_rowcol(rBias.layout()));
        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < size<0>(tBias_rowcol); ++m) {
            int const row = int(get<0>(tScS_rowcol(m, _0{}))) + m_block * kBlockM;
            if (row >= rows) {
                continue;
            }
            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < size<1>(tBias_rowcol); ++n) {
                int const col = int(get<1>(tScS_rowcol(m, n))) + n_block * kBlockN;
                if (col < cols) {
                    rBias_rowcol(m, n) = tBias_rowcol(m, n);
                }
            }
        }
    }

    // Pure compute.  On the vectorised path an out-of-range entry holds stale
    // gmem, which the mask overwrites with -inf a few instructions later; on
    // the predicated path it is zero and the add is a no-op.
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(acc_s); ++i) {
        acc_s(i) += float(rBias(i)) * inv_scale;
    }
}

}  // namespace attention
}  // namespace oasr
