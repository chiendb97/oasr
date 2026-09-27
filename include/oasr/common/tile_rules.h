// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// The CuTe-free arithmetic every Ampere-class (mma.sync + cp.async) CuTe tile
// obeys: the shared-memory swizzle row, the gmem<->smem pass geometry, the warp
// tiling of the m16n8k16 atom, and the occupancy estimate.
//
// These are facts about the *instructions and the shared-memory banks*, not
// about any one kernel, which is why they live here rather than in each
// family.  Three families -- attention, the gated MLP and the recurrent step
// (`include/oasr/{attention,mlp,recurrent}/`) -- each carried their own copy of
// this arithmetic, and the copies are how a rule drifts: a clause fixed in one
// validator is still missing from the next.  Their tile validators are now
// compositions of the predicates below, plus the clauses that really are
// theirs.
//
// Pure integers, and `constexpr` over arguments -- the same contract as
// `arch_facts.h`, and for the same reason: the families' capability-oracle
// translation units (`csrc/*_jit_binding.cu`) include only this and their
// `*_tiles.h`, so the Python mirrors can be checked against the originals in a
// couple of seconds instead of paying a CUTLASS compile.  The Python mirror of
// this file is `oasr/jit/arch_facts.py`.

#pragma once

#include <oasr/common/arch_facts.h>

namespace oasr {

//! `ceil(a / b)` for non-negative \p a and positive \p b.
constexpr int tileCeilDiv(int a, int b) {
    return (a + b - 1) / b;
}

/*! \brief Bytes one 128-bit gmem<->smem vector moves.
 *
 * `cp.async.cg` and the vectorised epilogue stores both move 16 bytes, so an
 * operand's contiguous extent has to be a whole number of these.
 */
inline constexpr int kTileVectorBytes = 16;

/*! \brief Elements per swizzled shared-memory row for a tile whose contiguous
 *  extent is \p extent elements.
 *
 * One row should cover a whole number of 128-byte lines -- the width of the 32
 * four-byte banks -- where the extent allows, falling back to 64 and then 32
 * bytes.  That makes the swizzle bank-conflict free for the widest `ldmatrix`
 * that fits, and it is wider than the CuTeDSL lanes' `make_smem_swizzle_atom`
 * can express (it tops out at 64 elements).
 */
constexpr int smemSwizzleRowWidth(int extent, int elem_size) {
    int const bytes = extent * elem_size;
    int const b = (bytes % 128 == 0) ? 128 : ((bytes % 64 == 0) ? 64 : 32);
    return b / elem_size;
}

/*! \brief `Swizzle<B, 3, 3>`'s `B` for a row of \p row_width elements.
 *
 * Keyed on the *element* count, exactly as every family computed it; for the
 * 2-byte types in scope that is 3 for a 128-byte row and 2 for a 64-byte one.
 */
constexpr int smemSwizzleBits(int row_width) {
    return row_width == 128 ? 4 : (row_width == 64 ? 3 : 2);
}

/*! \brief Can `(warps_m, warps_n)` warps tile a `(block_m, block_n)` accumulator?
 *
 * The MMA is tiled over a 16x16 permutation of the m16n8k16 atom, so each axis
 * must cover a whole number of those -- the clause `cute::gemm`'s static
 * asserts would otherwise report as an unreadable layout mismatch.
 */
constexpr bool sm80WarpTilingValid(int block_m, int block_n, int threads, int warps_n) {
    if (threads % 32 != 0 || threads < 32 || threads > 1024) {
        return false;
    }
    int const warps = threads / 32;
    if (warps_n < 1 || warps % warps_n != 0) {
        return false;
    }
    int const warps_m = warps / warps_n;
    return block_m % (16 * warps_m) == 0 && block_n % (16 * warps_n) == 0;
}

/*! \brief Rows one pass of a 128-bit tiled copy covers, or 0 if the copy is not
 *  expressible.
 *
 * The copy puts `row_width / vector` threads on one swizzled row and stacks
 * the rest; \p threads has to be a whole number of rows of them.
 */
constexpr int sm80TiledCopyRowsPerPass(int extent, int threads, int elem_size) {
    int const vec = kTileVectorBytes / elem_size;
    int const row = smemSwizzleRowWidth(extent, elem_size);
    int const threads_per_row = row / vec;
    if (threads_per_row <= 0 || threads % threads_per_row != 0 || extent % row != 0) {
        return 0;
    }
    return threads / threads_per_row;
}

/*! \brief Does a 128-bit tiled copy cover a `(rows, extent)` tile in whole passes?
 *
 * Not a performance rule: a pass wider than the tile puts the *partition* out
 * of range, which is an illegal access no predicate can intercept.
 */
constexpr bool sm80TiledCopyFits(int rows, int extent, int threads, int elem_size) {
    int const rows_per_pass = sm80TiledCopyRowsPerPass(extent, threads, elem_size);
    return rows_per_pass > 0 && rows % rows_per_pass == 0;
}

/*! \brief Hardware ceiling on resident blocks per SM, for the occupancy estimate.
 *
 * The tiles in scope never approach it -- shared memory binds at one or two --
 * but leaving it out would let a hypothetical tiny tile claim absurd
 * occupancy.
 */
inline constexpr int kTileMaxBlocksPerSm = 24;

/*! \brief How many CTAs of \p smem_bytes and \p threads are resident per SM.
 *
 * Shared memory and warp slots only.  Registers are deliberately not
 * modelled: every tile these families compile measures well inside what
 * shared memory already allows, so a register bound would never bind, and a
 * tile that changed that would show up as a *measured* regression rather than
 * as a wrong number here.  Never less than one.
 */
constexpr int tileCtasPerSm(int smem_bytes, int threads, int sm) {
    int const budget = smemBudgetForSm(sm);
    int const by_smem = smem_bytes > 0 ? budget / smem_bytes : kTileMaxBlocksPerSm;
    int const threads_per_sm = maxThreadsPerSmForSm(sm);
    int const by_threads = threads > 0 ? threads_per_sm / threads : 0;
    int r = by_smem < by_threads ? by_smem : by_threads;
    if (r > kTileMaxBlocksPerSm) {
        r = kTileMaxBlocksPerSm;
    }
    return r < 1 ? 1 : r;
}

/*! \brief Is \p sm able to host one CTA of \p smem_bytes and \p threads at all? */
constexpr bool tileFitsSm(int smem_bytes, int threads, int sm) {
    int const budget = smemBudgetForSm(sm);
    return budget > 0 && maxThreadsPerSmForSm(sm) >= threads && smem_bytes <= budget;
}

}  // namespace oasr
