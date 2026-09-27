// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Gathering one K or V tile out of a paged KV pool.
//
// Structurally FlashAttention's `hopper/paged_kv.h`, and it encodes the same
// invariant the CuTeDSL backend documents at length after paying for it:
//
//   **Partition the whole `(kBlockN, kHeadDim)` tile once, then vary the gmem
//   source per partitioned row.**
//
// The tempting alternative -- slice shared memory into `kBlockN / page_size`
// sub-tiles of `(page_size, kHeadDim)` and re-run `partition_D` on each -- is
// wrong, and wrong silently.  The gmem tiled copy's thread layout covers
// `NumThreads * elems_per_load / kBlockKGmem` **rows** per pass, which is a
// property of `head_dim`, not of `page_size`.  When it exceeds `page_size` the
// surplus threads address rows past the end of their sub-tile, i.e. into the
// next page's rows, which that page's own copy also writes:
//
//     head_dim   smem row width   tiler rows   vs page_size 16
//     64         64               16           exact fit, one writer per row
//     32         32               32           16 rows spill -> WAW race
//
// That was a real defect, not a hypothetical: `compute-sanitizer racecheck`
// reported 176 WAW hazards at head_dim 32 and 0 at head_dim 64, and the visible
// symptom was two CUDA-graph captures of the same streaming shape disagreeing
// by ~1e-1 in log-probs.  The canonical full-tile partition cannot express it,
// because every smem element has exactly one writer by construction.
//
// Two things this does that the CuTeDSL loader does not:
//
//  * **The page id is predicated.**  A row past `seqlen_k`, or past the block
//    table's width, never reads the table at all.  That retires the caller-side
//    requirement that the table be widened to a whole number of K tiles -- an
//    unpredicated read of a short table dereferences whatever followed it in
//    memory *as a page id*, which the very next instruction uses as an address.
//  * **`page_size` is a runtime value.**  Walking a partial page costs a
//    compare; making it a compile-time axis would have multiplied the JIT
//    variant space by every page size any caller might choose.

#pragma once

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/fast_math.h>

#include "fmha_utils.h"

namespace oasr {
namespace attention {

using namespace cute;

/*! \brief Gather `(kBlockN, kHeadDim)` of K or V from a paged pool into smem.
 *
 * \param mPool      the whole pool, `(page_size, head_dim, H_kv, num_pages)`
 * \param sDst       destination smem tile for this ring stage, `(N, D)`
 * \param page_table `(batch, max_pages)` int32
 * \param n_block    which K tile
 * \tparam Seqlenk_mask   bound the rows by `seqlen_k` (only the first tile needs it)
 *
 * **Everything skipped arrives as zeros**, through the ZFILL atom's own
 * `src_size` rather than through a branch -- rows past `seqlen_k` or past the
 * block table's width, and the head-dim residue alike.  V needs its rows
 * zeroed (a stale NaN reaches the output through `P @ V`, where no mask can
 * intercept it); K's out-of-range scores are masked, so zeros there are merely
 * harmless.  The residue needs zeroing for both: `head_dim` need only be a
 * multiple of the 128-bit load width while the smem layouts are built on the
 * padded dim, and the QK gemm runs over the *padded* extent, so leaving those
 * columns stale feeds uninitialised shared memory into every score -- which
 * once produced wrong scores at head_dim 16 for one batch of two, the
 * signature of reading shared memory that merely *happened* to be zero.
 *
 * A predicated-off copy's source address is still formed, so it has to be
 * computable: an out-of-range row reads page 0 of the pool, never the table.
 */
template <int kBlockN, int kHeadDim, bool Seqlenk_mask, class TensorPool, class TensorDst,
          class TiledCopy, class ThrCopy, class TensorCoord, class TensorPred>
CUTLASS_DEVICE void paged_gather_tile(TensorPool const& mPool, TensorDst&& sDst,
                                      int32_t const* const page_table,
                                      int64_t const page_table_stride, int const bidb,
                                      int const bidh_kv, int const n_block,
                                      int const seqlen_k, int const max_pages,
                                      cutlass::FastDivmod const& page_size_divmod,
                                      TiledCopy const& tiled_copy, ThrCopy const& thr_copy,
                                      TensorCoord const& tKVcKV, TensorPred const& tKVpKV) {
    using Element = typename TensorPool::value_type;
    constexpr int kElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);

    Tensor tDst = thr_copy.partition_D(sDst);

    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < size<1>(tDst); ++m) {
        int const row = int(get<0>(tKVcKV(_0{}, m, _0{})));
        int const abs_row = n_block * kBlockN + row;
        int page_idx = 0, offset_in_page = 0;
        page_size_divmod(page_idx, offset_in_page, abs_row);
        // The page id itself is read only when the row is real.  A short block
        // table read unpredicated hands back whatever followed it, and the next
        // instruction dereferences that as an address.
        bool const row_ok = (row < kBlockN) && (page_idx < max_pages) &&
                            (!Seqlenk_mask || abs_row < seqlen_k);
        int const phys = row_ok ? page_table[int64_t(bidb) * page_table_stride + page_idx] : 0;

        // This row's `(head_dim,)` slice of its physical page, re-viewed so one
        // copy covers one 128-bit vector.
        Tensor gRow = cute::tiled_divide(mPool(offset_in_page, _, bidh_kv, phys),
                                         Shape<Int<kElemsPerLoad>>{});
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < size<2>(tDst); ++k) {
            int const d0 = int(get<1>(tKVcKV(_0{}, _0{}, k)));
            bool const ok = row_ok && bool(tKVpKV(k));
            cute::copy(tiled_copy.with(ok), gRow(_, d0 / kElemsPerLoad), tDst(_, m, k));
        }
    }
}

}  // namespace attention
}  // namespace oasr
