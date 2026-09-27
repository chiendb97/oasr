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
// Read, then add -- and why the read is not a tile ahead
// ---------------------------------------------------------------------------
//
// The read (`load_bias_tile`) and the add (`apply_bias_tile`) are separate so
// the read can be scheduled independently of the add, and the obvious schedule
// -- issue tile `n - 1`'s read as soon as tile `n`'s is consumed, carrying the
// fragment across the loop -- was built and measured.  It is not what ships:
// against issuing the read right after the QK gemm, it bought 3-4% on the
// streaming chunks and on the longest offline rows, and *cost* 3-6% on
// interior Q tiles that walk two to four K tiles -- including a realistic
// Conformer batch, `B16 H4 T256`, which read 0.983x of the kernel before any
// of this work.  Two placements of the in-loop prefetch and a read-only
// (`ld.global.nc`) spelling of the loads all showed the same split.  With the
// loads at their point of use the same production-shaped sweep reads 1.20x
// with every row at or above 1.036x, so the simpler schedule is also the one
// without a regression.
//
// What did pay is the *shape* of the read, below: rows past the plane are
// branched around, and boundary tiles read column pairs.  Measured before it,
// the bias doubled the kernel under ncu on a long sequence (`B1 H8 T1500 D64`:
// 63.6 -> 131.9 us) and more than doubled it on short query tiles
// (`B16 H4 Tq20 Tk400`: 11.2 -> 26.8 us).

// The bias is partitioned through the *same* `partition_C` as the score
// accumulator, so no re-layout is needed -- element `i` of the fragment is
// element `i` of `acc_s`.  What differs is how it is read:
//
//  * **interior + vectorisable**: an unpredicated vector copy.  Adjacent
//    columns are adjacent in memory and every row start is 4-byte aligned, so
//    the compiler emits 32-bit pair loads.
//
//  * **boundary, or not vectorisable**: predicated, but still **per column
//    pair** where the pair is in range and aligned -- one 32-bit load instead
//    of two 16-bit ones -- and with rows past the extent skipped outright.  A
//    Q tile shorter than `kBlockM` rows is a boundary tile on every K tile,
//    and that is every streaming chunk, so this is the common path there
//    rather than the edge case the name suggests.  Elements out of range are
//    zero, so the add is a no-op for them.
//
// The predicated path is not defensive programming.  The unpredicated read
// covers the tile's full `kBlockM x kBlockN` footprint, so on a boundary tile
// it addresses up to `kBlockM - 1` rows past the last real one; at a short
// packed segment (8x8 = 64 elements) a 64x64 tile reads ~500 elements past the
// block.  The out-of-bounds *values* were always harmless -- the mask
// overwrites those score slots immediately after -- but the addresses are not
// optional, and this faulted for real whenever the allocator left the next page
// unmapped.
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

/*! \brief A register fragment for one tile's bias, laid out like the score
 *  accumulator of \p tiled_mma.
 */
template <int kBlockM, int kBlockN, class ElementBias, class TiledMma>
CUTLASS_DEVICE auto make_bias_fragment(TiledMma tiled_mma, int const thread_idx) {
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor cS = cute::make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
    return make_tensor<ElementBias>(shape(thr_mma.partition_C(cS)));
}

/*! \brief Read one Q-tile x K-tile block of the bias into \p rBias.
 *
 * \param mBias  this (batch, head)'s `(T_q, T_k)` bias plane
 *
 * Only issues loads; nothing waits on them until `apply_bias_tile`.
 *
 * Predication is against the **bias tensor's own extents**, not against the
 * sequence lengths.  They can differ -- a paged caller's bias spans the whole
 * logical KV extent while `seqlen_k` is this stream's -- and it is the
 * allocation, not the sequence, that faults.  That is also why the columns
 * stay predicated on every tile, not just the first: only the first tile
 * straddles `seqlen_k`, but nothing on the device can check that the plane is
 * at least that wide for every stream of a paged batch, and a read that never
 * faults on an undersized plane is the contract this had before.
 *
 * Rows past the extent are *branched* around, not predicated off.  These are
 * register loads, so there is no asynchronous copy for a branch to be ordered
 * against, and a Q tile shorter than `kBlockM` leaves whole warps with no row
 * in range (at `T_q = 8`, three of the four): predicating their loads off
 * instead issued 43% more `LDG` instructions for nothing.
 */
template <int kBlockM, int kBlockN, class TiledMma, class TensorB, class FrgB>
CUTLASS_DEVICE void load_bias_tile(FrgB& rBias, TensorB const& mBias, TiledMma tiled_mma,
                                   int const thread_idx, int const m_block, int const n_block,
                                   bool const vectorizable) {
    using ElementBias = typename TensorB::value_type;
    using PairCopy = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<32>, ElementBias>;
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);

    Tensor gBias = local_tile(mBias, Shape<Int<kBlockM>, Int<kBlockN>>{},
                              make_coord(m_block, n_block));
    Tensor tBias = thr_mma.partition_C(gBias);  // ((2 cols, 2 rows), MMA_M, MMA_N)
    static_assert(decltype(rank<0>(tBias))::value == 2,
                  "SM80's MMA-C fragment is ((2, 2), MMA_M, MMA_N)");

    int const rows = int(size<0>(mBias));
    int const cols = int(size<1>(mBias));
    bool const interior =
        (m_block + 1) * kBlockM <= rows && (n_block + 1) * kBlockN <= cols;

    if (vectorizable && interior) {
        cute::copy(PairCopy{}, tBias, rBias);
        return;
    }
    cute::clear(rBias);
    Tensor cS = cute::make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
    Tensor tScS = thr_mma.partition_C(cS);
    int const row0 = m_block * kBlockM;
    int const col0 = n_block * kBlockN;
    CUTLASS_PRAGMA_UNROLL
    for (int mi = 0; mi < size<1>(tBias); ++mi) {
        CUTLASS_PRAGMA_UNROLL
        for (int r = 0; r < 2; ++r) {
            if (int(get<0>(tScS(make_coord(0, r), mi, 0))) + row0 >= rows) {
                continue;
            }
            CUTLASS_PRAGMA_UNROLL
            for (int ni = 0; ni < size<2>(tBias); ++ni) {
                // The pair is two adjacent columns of one row: `(0, r)` and
                // `(1, r)` of the atom's `(col, row)` mode.
                int const col = int(get<1>(tScS(make_coord(0, r), mi, ni))) + col0;
                if (vectorizable && col + 1 < cols) {
                    cute::copy(PairCopy{}, tBias(make_coord(_, r), mi, ni),
                               rBias(make_coord(_, r), mi, ni));
                } else {
                    if (col < cols) {
                        rBias(make_coord(0, r), mi, ni) = tBias(make_coord(0, r), mi, ni);
                    }
                    if (col + 1 < cols) {
                        rBias(make_coord(1, r), mi, ni) = tBias(make_coord(1, r), mi, ni);
                    }
                }
            }
        }
    }
}

/*! \brief `acc_s += bias * inv_softmax_scale`, from a fragment `load_bias_tile` filled.
 *
 * \param inv_scale `1 / softmax_scale`
 *
 * The scale division is what makes the bias a *post*-scale logit, matching
 * SDPA: the softmax computes `exp2((acc_s + bias/scale) * scale * log2e)`,
 * which collapses to `exp(scale * acc_s + bias)`.  Pure compute: an
 * out-of-range entry is zero and the add is a no-op, and the mask overwrites
 * those score slots with `-inf` right after anyway.
 */
template <class EngineS, class LayoutS, class FrgB>
CUTLASS_DEVICE void apply_bias_tile(Tensor<EngineS, LayoutS>& acc_s, FrgB const& rBias,
                                    float const inv_scale) {
    CUTE_STATIC_ASSERT_V(size(acc_s) == size(rBias));
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(acc_s); ++i) {
        acc_s(i) += float(rBias(i)) * inv_scale;
    }
}

}  // namespace attention
}  // namespace oasr
