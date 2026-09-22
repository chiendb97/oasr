// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// The gated MLP's CTA tiles, and the arithmetic that picks one.
//
// Pure integers: this header pulls in nothing but `oasr/common/arch_facts.h`,
// no CuTe and no CUDA.  That is deliberate.  `csrc/gated_mlp_jit_binding.cu`
// exports these answers so that `oasr/jit/gated_mlp.py`'s mirror of them can
// be checked against the original rather than trusted, and a mirror test is
// worth having only if the oracle is cheap to build -- the same reason the
// answers are `constexpr` over an `sm` **argument** rather than over a device
// query: one built module then holds the capability line for every
// architecture from whatever box CI happens to run on.
//
// Why the tile is a *table* here and a pure resolver in attention
// ---------------------------------------------------------------
// `fmhaResolveTile` derives its tile from the architecture alone, because what
// it is solving for is "what fits".  This kernel's choice is a *wave-count*
// decision -- the last wave has to have enough CTAs in flight to saturate DRAM
// -- so it is a function of the problem (`rows`, `N`) and of the machine (SM
// count, shared-memory budget) as well.  That cannot collapse to one
// compile-time answer, so the tiles are compiled as sibling variants of one
// module and chosen per call.

#pragma once

#include <oasr/common/arch_facts.h>

namespace oasr {
namespace mlp {

/*! \brief One CTA tile of the gated-MLP mainloop.
 *
 * A plain struct rather than a `cute::Shape` so it is usable from host code
 * that does not include CuTe -- `csrc/gated_mlp_jit_binding.cu` exports it and
 * the Python routing mirrors it.
 */
struct GatedMlpTile {
    int block_m;
    int block_n;
    int block_k;
    int stages;
    int threads;
    int warps_n;  //!< how many of `threads / 32` warps tile N rather than M
};

/*! \brief The tiles this lane compiles, grouped by `block_m`, in preference order.
 *
 * Two entries per group, and the difference between them is not the tile but
 * the *ring*: the first is a deep 4-stage ring at 64-wide K that fills shared
 * memory and gets one CTA per SM; the second is a 32-wide K tile whose ring is
 * half the size and therefore fits **two**.  `gatedMlpSelectTile` chooses
 * between them by wave arithmetic, and which one wins is decided by `N`.
 *
 * `block_n` is 64 throughout because at every width in scope the N axis
 * already supplies more CTAs than a machine has SMs, so a wider N tile only
 * deepens the ring and halves the grid.  `block_m` stops at 64 on purpose: one
 * m-tile is what makes every weight element a single DRAM read, which is the
 * bandwidth argument the whole fusion rests on.  With two m-tiles each weight
 * tile is loaded twice and the kernel is an ordinary GEMM competing with cuBLAS
 * on cuBLAS's terms.
 *
 * A warp count is not stored: `warps_m` is `threads / 32 / warps_n`.
 *
 * Provenance: these six are the CuTeDSL lane's measured candidate list
 * (`oasr/jit/mlp.py::_CANDIDATES`), inherited rather than re-derived, so that
 * an A/B between the two lanes compares kernels and not tile choices.  Both
 * lanes pick the same entry for every shape in the sweep, which is asserted.
 *
 * Measured on one card (RTX 5090, sm_120, 170 SMs).  Which six tiles are in
 * this list does not travel; `gatedMlpSelectTile` reads the machine, so the
 * choice among them does.  `oasr/jit/gated_mlp.py::_MEASURED` records the
 * split, and `note_extrapolation` counts it off that card.
 */
inline constexpr GatedMlpTile kGatedMlpTiles[] = {
    {16, 64, 64, 4, 128, 4},  // 0
    {16, 64, 32, 4, 64, 2},   // 1
    {32, 64, 64, 4, 128, 4},  // 2
    {32, 64, 32, 4, 64, 2},   // 3
    {64, 64, 64, 4, 256, 4},  // 4
    {64, 64, 32, 4, 128, 4},  // 5
};

inline constexpr int kGatedMlpTileCount =
    int(sizeof(kGatedMlpTiles) / sizeof(kGatedMlpTiles[0]));

/*! \brief Hardware ceiling on resident blocks per SM.
 *
 * The tiles here never approach it -- shared memory binds at 1 or 2 -- but
 * leaving it out would let a hypothetical tiny tile claim absurd occupancy.
 */
inline constexpr int kGatedMlpMaxBlocksPerSm = 24;

//! `ceil(a / b)` for non-negative `a` and positive `b`.
constexpr int gatedMlpCeilDiv(int a, int b) { return (a + b - 1) / b; }

/*! \brief Elements per shared-memory "row" for a tile of \p extent elements.
 *
 * One row should cover a whole number of 128-byte cache lines where it can, so
 * the swizzle is bank-conflict free for the widest `ldmatrix` that fits.  Same
 * rule as the attention family's `fmhaBlockKGmem`, and it agrees with the
 * CuTeDSL lane's `make_smem_swizzle_atom` at every extent either of them uses.
 */
constexpr int gatedMlpSmemRowWidth(int extent, int elem_size) {
    int const bytes = extent * elem_size;
    int const b = (bytes % 128 == 0) ? 128 : ((bytes % 64 == 0) ? 64 : 32);
    return b / elem_size;
}

/*! \brief The cp.async ring, in bytes: `stages * (M + 2N) * K`.
 *
 * Half again what a plain GEMM's ring costs, because one stage carries A **and
 * both** Bs.  That is the tuning pressure that makes `block_k = 32` a real
 * option here and not merely a small tile.
 */
constexpr int gatedMlpMainloopSmemBytes(GatedMlpTile t, int elem_size) {
    return t.stages * (t.block_m + 2 * t.block_n) * t.block_k * elem_size;
}

/*! \brief The epilogue's staging buffer, in bytes.
 *
 * The MMA-C partition scatters a thread's values across the row, so a direct
 * global store is a handful of narrow transactions; staging through shared
 * memory and re-reading with the gmem tiled copy makes it one 128-bit access
 * per thread.
 */
constexpr int gatedMlpEpilogueSmemBytes(GatedMlpTile t, int elem_size) {
    return t.block_m * t.block_n * elem_size;
}

/*! \brief Shared memory the kernel requests, in bytes.
 *
 * The epilogue's buffer is a **union** with the mainloop's ring, not an
 * addition: the mainloop is finished with every stage by the time the epilogue
 * writes.  So the cost is the larger of the two, and a tile whose output tile
 * exceeds its ring is legal here.  (The CuTeDSL lane aliases rather than
 * unions, so it has to refuse that case -- `GatedMlpCute.can_implement`.)
 */
constexpr int gatedMlpSmemBytes(GatedMlpTile t, int elem_size) {
    int const mainloop = gatedMlpMainloopSmemBytes(t, elem_size);
    int const epilogue = gatedMlpEpilogueSmemBytes(t, elem_size);
    return mainloop > epilogue ? mainloop : epilogue;
}

/*! \brief Would this tile compile, fit, and address only its own tiles?
 *
 * Refuses rather than degrading.  Every clause is a *silent* failure if it is
 * dropped:
 *
 *  * the MMA tiling clauses (`block_m % (16 * warps_m)`) are what
 *    `cute::gemm`'s static asserts would otherwise report as an unreadable
 *    layout mismatch;
 *  * the `rows_per_pass` clauses are an **illegal access**, not a predicated
 *    no-op: a gmem->smem pass wider than the tile puts the *partition* out of
 *    range, so no predicate can intercept it;
 *  * the shared-memory clause is a launch failure with an empty error message.
 */
constexpr bool gatedMlpTileValid(GatedMlpTile t, int sm, int elem_size) {
    if (elem_size != 2) {
        return false;  // fp16 / bf16; the MMA atom and the 128-bit vector both assume it
    }
    if (t.threads % 32 != 0 || t.threads < 32 || t.threads > 1024) {
        return false;
    }
    int const warps = t.threads / 32;
    if (t.warps_n < 1 || warps % t.warps_n != 0) {
        return false;
    }
    int const warps_m = warps / t.warps_n;
    // The MMA is tiled (warps_m, warps_n) over a 16x16 permutation of the
    // m16n8k16 atom, so each axis must cover a whole number of those.
    if (t.block_m % (16 * warps_m) != 0 || t.block_n % (16 * t.warps_n) != 0) {
        return false;
    }
    // 32 is the swizzle atom's narrowest row; 16 would be legal for the MMA
    // and illegal for `tile_to_shape`.
    if (t.block_k % 32 != 0 || t.stages < 2) {
        return false;
    }
    int const vec = 16 / elem_size;  // the 128-bit cp.async / store width
    int const k_row = gatedMlpSmemRowWidth(t.block_k, elem_size);
    int const threads_per_row = k_row / vec;
    if (threads_per_row <= 0 || t.threads % threads_per_row != 0 ||
        t.block_k % k_row != 0) {
        return false;
    }
    int const rows_per_pass = t.threads / threads_per_row;
    if (rows_per_pass <= 0 || t.block_m % rows_per_pass != 0 ||
        t.block_n % rows_per_pass != 0) {
        return false;
    }
    // ...and the same again for the epilogue, whose atom is sized on N.
    int const n_row = gatedMlpSmemRowWidth(t.block_n, elem_size);
    int const o_threads_per_row = n_row / vec;
    if (o_threads_per_row <= 0 || t.threads % o_threads_per_row != 0 ||
        t.block_n % n_row != 0) {
        return false;
    }
    int const o_rows_per_pass = t.threads / o_threads_per_row;
    if (o_rows_per_pass <= 0 || t.block_m % o_rows_per_pass != 0) {
        return false;
    }
    int const budget = smemBudgetForSm(sm);
    if (budget <= 0 || maxThreadsPerSmForSm(sm) < t.threads) {
        return false;
    }
    return gatedMlpSmemBytes(t, elem_size) <= budget;
}

/*! \brief How many of these CTAs are resident at once, by shared memory and warps.
 *
 * Registers are deliberately not modelled: every tile here measures ~64-96
 * registers per thread, which binds at 8 blocks or more -- far above the 1-2
 * that shared memory allows.  A tile that changed that would show up as a
 * *measured* regression, not as a wrong number here.
 */
constexpr int gatedMlpCtasPerSm(GatedMlpTile t, int sm, int elem_size) {
    int const bytes = gatedMlpSmemBytes(t, elem_size);
    int const budget = smemBudgetForSm(sm);
    int const by_smem = bytes > 0 ? budget / bytes : kGatedMlpMaxBlocksPerSm;
    int const threads_per_sm = maxThreadsPerSmForSm(sm);
    int const by_threads = t.threads > 0 ? threads_per_sm / t.threads : 0;
    int r = by_smem < by_threads ? by_smem : by_threads;
    if (r > kGatedMlpMaxBlocksPerSm) {
        r = kGatedMlpMaxBlocksPerSm;
    }
    return r < 1 ? 1 : r;
}

/*! \brief Waves this tile's grid takes on a machine with \p num_sms SMs. */
constexpr int gatedMlpWaves(GatedMlpTile t, int sm, int num_sms, int rows, int n,
                            int elem_size) {
    int const slots = num_sms * gatedMlpCtasPerSm(t, sm, elem_size);
    int const grid = gatedMlpCeilDiv(n, t.block_n) * gatedMlpCeilDiv(rows, t.block_m);
    return gatedMlpCeilDiv(grid, slots);
}

/*! \brief The `block_m` group that owns \p rows: the smallest that covers it. */
constexpr int gatedMlpTileGroupM(int rows) {
    int largest = 0;
    int pick = 0;
    for (int i = 0; i < kGatedMlpTileCount; ++i) {
        int const bm = kGatedMlpTiles[i].block_m;
        if (bm > largest) {
            largest = bm;
        }
        if (bm >= rows && (pick == 0 || bm < pick)) {
            pick = bm;
        }
    }
    return pick != 0 ? pick : largest;
}

/*! \brief Index into `kGatedMlpTiles` for this problem, or -1 if none fits.
 *
 * Fewest waves wins; ties go to the earlier entry, which is the preference
 * order the table is written in.
 *
 * Why `N` is in the decision and not only `M`: the kernel is bandwidth bound,
 * so what decides its time is whether the *last* wave still has enough CTAs in
 * flight to saturate DRAM.  A rows-keyed table tuned at N=18944 picked a
 * one-CTA-per-SM ring that left 172 CTAs on 170 SMs -- one wave plus a tail of
 * two -- and read 0.987x end to end.
 *
 * **Never make this depend on CUDA-graph capture state** (`AGENTS.md` rule 11):
 * two tiles sum the K loop in different orders, so a capture-dependent answer
 * would make a replayed graph produce different numbers than eager.
 * Everything here is a function of the arguments, `num_sms` included.
 */
constexpr int gatedMlpSelectTile(int sm, int num_sms, int rows, int n, int elem_size) {
    if (rows <= 0 || n <= 0 || num_sms <= 0) {
        return -1;
    }
    int const group = gatedMlpTileGroupM(rows);
    int best = -1;
    int best_waves = 0;
    for (int i = 0; i < kGatedMlpTileCount; ++i) {
        if (kGatedMlpTiles[i].block_m != group ||
            !gatedMlpTileValid(kGatedMlpTiles[i], sm, elem_size)) {
            continue;
        }
        int const w = gatedMlpWaves(kGatedMlpTiles[i], sm, num_sms, rows, n, elem_size);
        if (best < 0 || w < best_waves) {
            best = i;
            best_waves = w;
        }
    }
    if (best >= 0) {
        return best;
    }
    // Degenerate: no tile in the preferred group fits this architecture.  Take
    // any valid tile rather than declining -- a smaller `block_m` costs padding,
    // which is strictly better than routing the shape back to two GEMMs.
    for (int i = 0; i < kGatedMlpTileCount; ++i) {
        if (!gatedMlpTileValid(kGatedMlpTiles[i], sm, elem_size)) {
            continue;
        }
        int const w = gatedMlpWaves(kGatedMlpTiles[i], sm, num_sms, rows, n, elem_size);
        if (best < 0 || w < best_waves) {
            best = i;
            best_waves = w;
        }
    }
    return best;
}

/*! \brief The 128-bit vector contract, in elements.
 *
 * The epilogue stores `out` and the mainloop loads `x` and both weights in
 * 128-bit pieces, so both contiguous extents have to be 8-element multiples.
 * Same number, same reason, as `oasr.layers._backend.GEMM_ALIGNMENT`.
 */
inline constexpr int kGatedMlpAlignment = 8;
}  // namespace mlp
}  // namespace oasr
