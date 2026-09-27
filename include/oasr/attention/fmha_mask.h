// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// The element-wise `-inf` mask over one Q-tile x K-tile score block.
//
// Structurally modelled on FlashAttention's `hopper/mask.h`, with two
// differences:
//
//  * **Top-left causal.**  FA's `causal_row_offset` carries
//    `+ seqlen_k - seqlen_q`, aligning the diagonal to the bottom-right corner.
//    OASR aligns top-left to match `torch`'s `is_causal`, which every parity
//    test compares against, so that term is absent.  This and
//    `fmha_block.h::get_n_block_min_max` are the only two sites that encode the
//    convention.
//
//  * **A per-row key start** (`Seqstart_mask`).  Left padding -- an HF-convention
//    batched prompt -- has a per-stream *lower* bound on the key index that a
//    compile-time window cannot express, since the pad amount differs per row
//    of the batch.  It is row-independent *within* a CTA, though, so it costs a
//    column-only predicate rather than the row x column one causal needs.
//
// Each predicate is a separate template bool, and the mainloop instantiates the
// cheapest combination each part of its K loop actually needs -- boundary tiles
// get the length and diagonal predicates, the interior gets none at all.  The
// CuTeDSL backend applies the full predicate set to every tile; at 64x64 that
// is 4096 compares and selects per tile that the interior does not need.
//
// Masking is **assignment** to `-INFINITY`, never `min`/`fmin`.  Shared memory
// past a sequence end can hold a NaN bit pattern (the K/V load is bounded by
// length, not by the tensor extent), `Q dot K_stale` is then NaN, and only an
// assignment intercepts it.  `fmin(NaN, -inf)` does not.

#pragma once

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>

#include "fmha_utils.h"

namespace oasr {
namespace attention {

using namespace cute;

/*! \brief Per-tile score mask.
 *
 * \tparam kBlockM  Q rows per CTA
 * \tparam kBlockN  K columns per tile
 * \tparam TiledMma the MMA whose C partition `tSrS` came from
 */
template <int kBlockM, int kBlockN, typename TiledMma>
struct Mask {
    int const thread_idx;
    int const seqlen_k;
    int const seqstart_k;
    int const window_size_left;
    int const window_size_right;

    CUTLASS_DEVICE Mask(int const thread_idx, int const seqlen_k, int const seqstart_k,
                        int const window_size_left, int const window_size_right)
        : thread_idx(thread_idx),
          seqlen_k(seqlen_k),
          seqstart_k(seqstart_k),
          window_size_left(window_size_left),
          window_size_right(window_size_right) {}

    /*! \brief Mask `tSrS` in place.
     *
     * \tparam Seqlenk_mask  bound columns above by `seqlen_k`
     * \tparam Causal_mask   top-left causal
     * \tparam Local_mask    sliding window (exclusive with `Causal_mask`)
     * \tparam Seqstart_mask bound columns below by `seqstart_k`
     *
     * With all four false this compiles to nothing, which is what the interior
     * of the K loop instantiates.
     */
    template <bool Seqlenk_mask = false, bool Causal_mask = false, bool Local_mask = false,
              bool Seqstart_mask = false, typename Engine, typename Layout>
    CUTLASS_DEVICE void apply(Tensor<Engine, Layout>& tSrS, int const m_block,
                              int const n_block) const {
        static_assert(!(Causal_mask && Local_mask), "causal and local are exclusive");
        static_assert(Layout::rank == 3, "tSrS is an MMA-C fragment");
        if constexpr (!Seqlenk_mask && !Causal_mask && !Local_mask && !Seqstart_mask) {
            return;
        }

        auto thread_mma = TiledMma{}.get_thread_slice(thread_idx);
        auto thread0_mma = TiledMma{}.get_thread_slice(_0{});

        Tensor cS = cute::make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
        Tensor tScS = thread_mma.partition_C(cS);
        Tensor tSrS_rowcol =
            make_tensor(tSrS.data(), cute_sm80::convert_layout_acc_rowcol(tSrS.layout()));
        Tensor tScS_rowcol =
            make_tensor(tScS.data(), cute_sm80::convert_layout_acc_rowcol(tScS.layout()));
        // Thread 0's coordinates are compile-time constants.  Comparing against
        // a limit that has *this* thread's own offset subtracted out therefore
        // keeps the inner compare free of per-thread address arithmetic.
        Tensor t0ScS = thread0_mma.partition_C(cS);
        Tensor t0ScS_rowcol =
            make_tensor(t0ScS.data(), cute_sm80::convert_layout_acc_rowcol(t0ScS.layout()));

        int const thread_col_offset = get<1>(tScS_rowcol(_0{}, _0{}));
        int const seqlenk_col_limit = seqlen_k - n_block * kBlockN - thread_col_offset;
        int const seqstart_col_limit = seqstart_k - n_block * kBlockN - thread_col_offset;

        if constexpr (!Causal_mask && !Local_mask) {
            // Column-only predicates: the whole column is masked or none of it
            // is, so the row loop is inside.
            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                int const col = int(get<1>(t0ScS_rowcol(_0{}, n)));
                bool const oob = (Seqlenk_mask && col >= seqlenk_col_limit) ||
                                 (Seqstart_mask && col < seqstart_col_limit);
                if (oob) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                        tSrS_rowcol(m, n) = -INFINITY;
                    }
                }
            }
        } else {
            // Top-left causal: query row `r` sees keys `[0, r]`, so within this
            // tile the exclusive right limit is `r + 1 - n_block*kBlockN`.  FA
            // has `+ seqlen_k - seqlen_q` here; we do not.  See the header.
            int const causal_row_offset = 1 - n_block * kBlockN - thread_col_offset;
            int const row_offset_right =
                causal_row_offset + (Causal_mask ? 0 : window_size_right);
            int const row_offset_left = causal_row_offset - 1 - window_size_left;
            // A negative window bound means *unbounded on that side* -- the
            // same convention `get_n_block_min_max` uses when it declines to
            // clamp `n_block_max`.  The left side is already guarded inside the
            // loop; the right needs its own, because `window_size_right == -1`
            // otherwise shrinks the limit by one column and masks the diagonal
            // itself.  That failure is quiet: the row still sums to 1 and
            // nothing goes NaN, it just attends to the wrong keys.
            bool const has_right = Causal_mask || (window_size_right >= 0);
            CUTLASS_PRAGMA_UNROLL
            for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                int const row_idx = int(get<0>(tScS_rowcol(m, _0{}))) + m_block * kBlockM;
                // Fold the length bound into the diagonal bound where both
                // apply, so the inner loop keeps one compare per side.
                int const col_limit_right =
                    !has_right
                        ? (Seqlenk_mask ? seqlenk_col_limit : int(kBlockN))
                        : (!Seqlenk_mask
                               ? row_idx + row_offset_right
                               : std::min(row_idx + row_offset_right, seqlenk_col_limit));
                int const col_limit_left = row_idx + row_offset_left;
                CUTLASS_PRAGMA_UNROLL
                for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                    int const col = int(get<1>(t0ScS_rowcol(_0{}, n)));
                    bool oob = col >= col_limit_right;
                    if constexpr (Local_mask) {
                        oob = oob || (window_size_left >= 0 && col < col_limit_left);
                    }
                    if constexpr (Seqstart_mask) {
                        oob = oob || (col < seqstart_col_limit);
                    }
                    if (oob) {
                        tSrS_rowcol(m, n) = -INFINITY;
                    }
                }
            }
        }
    }
};

}  // namespace attention
}  // namespace oasr
