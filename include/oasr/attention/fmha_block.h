// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// The half-open K-tile range `[n_block_min, n_block_max)` a Q tile must walk,
// and the three interior split points that decide which of those tiles need a
// mask at all.
//
// Structurally modelled on FlashAttention's `hopper/block.h`, with **one
// deliberate difference, in two places**: FA computes every causal and local
// bound relative to `seqlen_k - seqlen_q`, i.e. it aligns the diagonal to the
// bottom-right corner.  OASR is top-left aligned, matching torch's `is_causal`,
// so that term is absent here and in `fmha_mask.h`.  Those are the only two
// sites; keeping them together and naming the convention is what stops one
// being changed without the other.
//
// Bounding the loop at *both* ends is load-bearing, not an optimisation.
// Running to block 0 regardless made a left-padded batch load, MMA and then
// `-inf`-mask every tile below its start: measured 258 / 214 / 183 / 164 us at
// 0 / 25 / 50 / 75 % padding with the bound, against a flat ~257 us without it.

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/fast_math.h>

#include "fmha_seqlen.h"

namespace oasr {
namespace attention {

/*! \brief K-tile bounds and mask split points for one Q tile.
 *
 * \tparam kBlockM   Q rows per CTA
 * \tparam kBlockN   K columns per tile
 * \tparam Is_causal top-left causal
 * \tparam Is_local  sliding window (`Is_causal` and `Is_local` are exclusive;
 *                   causal-with-a-left-window arrives as `Is_local` with
 *                   `window_right == 0`)
 */
template <int kBlockM, int kBlockN, bool Is_causal, bool Is_local>
struct BlockMN {
    /*! \brief `[n_block_min, n_block_max)`; empty when `max <= min`.
     *
     * An empty range means this CTA has no work: the caller writes zeros and
     * returns.  That is strictly better than the CuTeDSL backend's "clamp so at
     * least one block always runs", which existed only to keep the cp.async
     * prologue on one code path, and it is what retires the `seqlen_k >= 1`
     * precondition.
     */
    CUTLASS_DEVICE static cute::tuple<int, int> get_n_block_min_max(
        SeqlenInfoQK const& info, int const m_block, int const window_size_left,
        int const window_size_right) {
        int const seqlen_k = info.seqlen_k;
        int n_block_max = cute::ceil_div(seqlen_k, kBlockN);

        if constexpr (Is_causal || Is_local) {
            // Top-left: the last key row this Q tile can see is its own last
            // query row, plus whatever right window is open.  No
            // `seqlen_k - seqlen_q` term -- see the file header.
            int const wr = Is_causal ? 0 : window_size_right;
            if (wr >= 0) {
                int const row_max = (m_block + 1) * kBlockM + wr;
                n_block_max = std::min(n_block_max, cute::ceil_div(row_max, kBlockN));
            }
        }

        int n_block_min = 0;
        // Keys below this stream's start are padding and can be skipped whole
        // tiles at a time; the straddling tile still needs the predicate.
        if (info.seqstart_k > 0) {
            n_block_min = info.seqstart_k / kBlockN;
        }
        if constexpr (Is_local) {
            if (window_size_left >= 0) {
                int const row_min = m_block * kBlockM - window_size_left;
                n_block_min = std::max(n_block_min, row_min > 0 ? row_min / kBlockN : 0);
            }
        }
        // A Q tile wholly past the end of this stream has nothing to attend to.
        if (m_block * kBlockM >= info.seqlen_q) {
            n_block_max = n_block_min;
        }
        return {n_block_min, n_block_max};
    }

    /*! \brief Lowest tile index that still needs the causal/local *right* mask.
     *
     * Tiles at or above this index straddle the diagonal; tiles below it are
     * entirely under it and need no right predicate.
     */
    CUTLASS_DEVICE static int get_n_block_min_right_masked(int const m_block,
                                                          int const n_block_min,
                                                          int const window_size_right) {
        if constexpr (!Is_causal && !Is_local) {
            return n_block_min;
        }
        int const wr = Is_causal ? 0 : window_size_right;
        if (wr < 0) {
            // Unbounded to the right: no tile has a right edge, so the masked
            // loop covers the whole range and the unmasked interior is empty.
            // Correct rather than minimal -- the left-window predicate still
            // has to run somewhere and this is the loop that carries it.
            // Spelling it out because the arithmetic below would *also* return
            // `n_block_min` here, by way of C++ truncating `-1 / kBlockN`
            // toward zero, which is not a rule to leave load-bearing.
            return n_block_min;
        }
        // The first query row of this tile sees keys up to `row + wr`; any tile
        // whose columns all lie at or below that needs no right mask.
        return std::max(n_block_min, (m_block * kBlockM + wr) / kBlockN);
    }

    /*! \brief Lowest tile index that needs *no* left mask.
     *
     * Below this, a tile straddles either the sliding window's left edge or
     * this stream's `seqstart_k`, and needs the left predicate.
     */
    CUTLASS_DEVICE static int get_n_block_min_left_unmasked(SeqlenInfoQK const& info,
                                                           int const m_block,
                                                           int const n_block_min,
                                                           int const window_size_left) {
        int bound = n_block_min;
        if constexpr (Is_local) {
            if (window_size_left >= 0) {
                // The last query row of this tile sees keys from
                // `row - window_left`; a tile entirely above that is unmasked.
                int const row_min = (m_block + 1) * kBlockM - 1 - window_size_left;
                bound = std::max(bound, cute::ceil_div(row_min > 0 ? row_min : 0, kBlockN));
            }
        }
        // At most one tile straddles `seqstart_k`, and only when the start is
        // not itself tile-aligned.
        if (info.seqstart_k > 0 && (info.seqstart_k % kBlockN) != 0) {
            bound = std::max(bound, info.seqstart_k / kBlockN + 1);
        }
        return bound;
    }
};

}  // namespace attention
}  // namespace oasr
