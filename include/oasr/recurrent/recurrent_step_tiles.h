// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// The fused recurrent step's CTA tiles, and the ladder that picks one.
//
// Pure integers: this header pulls in nothing but `oasr/common/arch_facts.h`,
// no CuTe and no CUDA.  That is deliberate, and it is the same split
// `include/oasr/mlp/gated_mlp_tiles.h` makes for the same reason --
// `csrc/recurrent_step_jit_binding.cu` exports these answers so that
// `oasr/jit/recurrent_step.py`'s mirror of them can be *checked* against the
// original rather than trusted, and a mirror test is worth having only if the
// oracle is cheap to build.  Same reason the answers are `constexpr` over an
// `sm` **argument** rather than over a device query: one built module then
// holds the capability line for every architecture from whatever box CI
// happens to run on.
//
// Why the choice is a table here and wave arithmetic in the gated MLP
// ------------------------------------------------------------------
// `gatedMlpSelectTile` scores candidates by wave count, because that kernel's
// decision is "does the last wave still saturate DRAM".  This one's is not: a
// recurrent step is a *dependent* launch -- step t+1 cannot start until t
// lands -- so there is no next kernel to overlap the tail with, and what the
// measurements actually found was a boundary between two different regimes
// (weight-read bound at small batch, MMA bound at large) rather than a wave
// count.  The ladder below is that boundary, measured.
//
// It is also, deliberately, **the same ladder the CuTeDSL lane uses**
// (`oasr/jit/recurrent_cute.py::_TILES`), inherited rather than re-derived, so
// that an A/B between the two lanes compares kernels and not tile choices.
// `tests/kernels/test_recurrent_cpp.py` pins the two equal entry for entry.

#pragma once

#include <oasr/common/arch_facts.h>

namespace oasr {
namespace recurrent {

/*! \brief One CTA tile of the fused recurrent step.
 *
 * A plain struct rather than a `cute::Shape` so it is usable from host code
 * that does not include CuTe -- `csrc/recurrent_step_jit_binding.cu` exports
 * it and the Python routing mirrors it.
 */
struct RecurrentStepTile {
    int block_m;  //!< rows of the cohort per CTA
    int block_n;  //!< gate-interleaved columns per CTA; `block_n / gates` hidden units
    int block_k;  //!< K tile; K is the hidden width being reduced over
    int stages;   //!< depth of the cp.async ring
    int threads;  //!< CTA size
    int warps_n;  //!< how many of `threads / 32` warps tile N rather than M
};

/*! \brief The tiles this lane compiles, in the order the ladder indexes them.
 *
 * These eight are exactly the distinct entries of the CuTeDSL lane's measured
 * table.  Note how many use 512 threads: at large batch the profile said
 * *occupancy*, not tiling -- 15% achieved, 0.47 waves per SM -- so what helped
 * was more warps at a fixed tile rather than a bigger tile.  A candidate list
 * that stopped at 256 threads left 11-26% on the table at B >= 128.
 *
 * `block_n` is a multiple of 32 throughout, which is what makes a tile hold
 * whole hidden units for both gate counts (4 for an LSTM, 1 for a vanilla
 * RNN) and keeps the state transition inside one CTA tile: a hidden unit's
 * gates are *adjacent columns*, so no cross-tile reduction is ever needed.
 *
 * Measured on one card (RTX 5090, sm_120, 170 SMs).
 * `oasr/jit/recurrent_step.py::_MEASURED` records the provenance and
 * `note_extrapolation` counts consulting it off that card.
 */
inline constexpr RecurrentStepTile kRecurrentStepTiles[] = {
    {16, 64, 64, 4, 128, 4},    // 0
    {16, 64, 64, 5, 128, 4},    // 1
    {32, 32, 64, 4, 128, 2},    // 2
    {32, 64, 64, 3, 256, 4},    // 3
    {64, 64, 64, 3, 512, 4},    // 4
    {64, 64, 64, 4, 512, 4},    // 5
    {128, 64, 64, 3, 512, 2},   // 6
    {128, 128, 64, 3, 512, 4},  // 7
};

inline constexpr int kRecurrentStepTileCount =
    int(sizeof(kRecurrentStepTiles) / sizeof(kRecurrentStepTiles[0]));

//! Stands in for "no upper bound" in the ladder below.
inline constexpr int kRecurrentStepUnbounded = 1 << 30;

/*! \brief One rung: the first rung both of whose bounds fit wins. */
struct RecurrentStepRoute {
    int hidden_max;
    int batch_max;
    int tile;
};

/*! \brief `(hidden, batch)` -> tile index, scanned in order.
 *
 * Mirrors `oasr/jit/recurrent_cute.py::_TILES` rung for rung.  Reproduced
 * rather than re-derived so the two lanes run the *same* tile on every shape
 * and an A/B measures the kernels.
 */
inline constexpr RecurrentStepRoute kRecurrentStepRoutes[] = {
    {256, 128, 2},
    {256, 256, 3},
    {256, kRecurrentStepUnbounded, 5},
    {768, 64, 2},
    {768, 128, 3},
    {768, 256, 5},
    {768, kRecurrentStepUnbounded, 6},
    {1536, 16, 2},
    {1536, 32, 1},
    {1536, 64, 3},
    {1536, 128, 4},
    {1536, kRecurrentStepUnbounded, 6},
    {kRecurrentStepUnbounded, 8, 0},
    {kRecurrentStepUnbounded, 16, 1},
    {kRecurrentStepUnbounded, 32, 3},
    {kRecurrentStepUnbounded, 64, 4},
    {kRecurrentStepUnbounded, 128, 6},
    {kRecurrentStepUnbounded, kRecurrentStepUnbounded, 7},
};

inline constexpr int kRecurrentStepRouteCount =
    int(sizeof(kRecurrentStepRoutes) / sizeof(kRecurrentStepRoutes[0]));

/*! \brief Hardware ceiling on resident blocks per SM.
 *
 * The tiles here never approach it -- shared memory binds at 1 or 2 -- but
 * leaving it out would let a hypothetical tiny tile claim absurd occupancy.
 */
inline constexpr int kRecurrentStepMaxBlocksPerSm = 24;

/*! \brief The 128-bit vector contract, in elements.
 *
 * The mainloop loads `previous_h` and `weight_hh` in 128-bit pieces along K,
 * and the epilogue reads a hidden unit's gates as one piece along N.  Same
 * number, same reason, as `oasr.layers._backend.GEMM_ALIGNMENT`.
 */
inline constexpr int kRecurrentStepAlignment = 8;

//! `ceil(a / b)` for non-negative \p a and positive \p b.
constexpr int recurrentStepCeilDiv(int a, int b) {
    return (a + b - 1) / b;
}

/*! \brief Elements per shared-memory "row" for a tile of \p extent elements.
 *
 * One row covers a whole number of 128-byte cache lines where it can, so the
 * swizzle is bank-conflict free for the widest `ldmatrix` that fits.  Same
 * rule as `gatedMlpSmemRowWidth` and as the attention family's
 * `fmhaBlockKGmem`, and it agrees with the CuTeDSL lane's
 * `make_smem_swizzle_atom` at every extent either of them uses.
 */
constexpr int recurrentStepSmemRowWidth(int extent, int elem_size) {
    int const bytes = extent * elem_size;
    int const b = (bytes % 128 == 0) ? 128 : ((bytes % 64 == 0) ? 64 : 32);
    return b / elem_size;
}

/*! \brief The cp.async ring, in bytes: `stages * (M + N) * K`.
 *
 * A stage carries one `previous_h` tile and one `weight_hh` tile.
 */
constexpr int recurrentStepMainloopSmemBytes(RecurrentStepTile t, int elem_size) {
    return t.stages * (t.block_m + t.block_n) * t.block_k * elem_size;
}

//! Floats of padding per staged accumulator row.  See
//! `recurrent_step_epilogue.h`: 8 is the one pad that is bank-conflict free
//! for the MMA-C store *and* leaves every row 16-byte aligned, so a cell's
//! four gates come back in one `LDS.128`.  The CuTeDSL lane pads by 1, which
//! is conflict-free for the store and misaligns the load.
inline constexpr int kRecurrentStepAccPad = 8;

/*! \brief The epilogue's accumulator staging buffer, in bytes.
 *
 * It exists because the MMA-C thread-value layout gives one thread only *two*
 * of a cell's four gate columns.  Staging makes the gather independent of that
 * layout instead of encoding it -- which is also what keeps this epilogue
 * correct if a future architecture's MMA atom partitions C differently.
 */
constexpr int recurrentStepEpilogueSmemBytes(RecurrentStepTile t) {
    return t.block_m * (t.block_n + kRecurrentStepAccPad) * 4;
}

/*! \brief Shared memory the kernel requests, in bytes.
 *
 * A **union**, not a sum: the mainloop has drained its ring and barriered by
 * the time the epilogue writes.  So a tile whose staging buffer exceeds its
 * ring is legal here -- the CuTeDSL lane *aliases* rather than unions and has
 * to refuse that case (`RecurrentStepCute.can_implement`).
 */
constexpr int recurrentStepSmemBytes(RecurrentStepTile t, int elem_size) {
    int const mainloop = recurrentStepMainloopSmemBytes(t, elem_size);
    int const epilogue = recurrentStepEpilogueSmemBytes(t);
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
 *  * `block_n % (8 * gates)` is what keeps a CTA tile holding whole hidden
 *    units.  Without it a cell's gates straddle two tiles and the state
 *    transition silently reads a neighbour's accumulator;
 *  * the `rows_per_pass` clauses are an **illegal access**, not a predicated
 *    no-op: a gmem->smem pass wider than the tile puts the *partition* out of
 *    range, so no predicate can intercept it;
 *  * the shared-memory clause is a launch failure with an empty error message
 *    -- which is exactly how `num_stages=5` configs got past the CuTeDSL
 *    lane's first `can_implement` and then died at launch.
 */
constexpr bool recurrentStepTileValid(RecurrentStepTile t, int sm, int elem_size, int gates) {
    if (elem_size != 2) {
        return false;  // fp16 / bf16; the MMA atom and the 128-bit vector both assume it
    }
    if (gates != 1 && gates != 4) {
        return false;
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
    // A tile holds whole hidden units, and the MMA atom is 8 wide.
    if (t.block_n % (8 * gates) != 0 || t.block_n % 8 != 0) {
        return false;
    }
    // 32 is the swizzle atom's narrowest row; 16 would be legal for the MMA
    // and illegal for `tile_to_shape`.
    if (t.block_k % 32 != 0 || t.stages < 2) {
        return false;
    }
    int const vec = 16 / elem_size;  // the 128-bit cp.async width
    int const k_row = recurrentStepSmemRowWidth(t.block_k, elem_size);
    int const threads_per_row = k_row / vec;
    if (threads_per_row <= 0 || t.threads % threads_per_row != 0 || t.block_k % k_row != 0) {
        return false;
    }
    int const rows_per_pass = t.threads / threads_per_row;
    if (rows_per_pass <= 0 || t.block_m % rows_per_pass != 0 || t.block_n % rows_per_pass != 0) {
        return false;
    }
    // The epilogue walks `(row, hidden unit)` slots, `t.threads` at a time.
    // Requiring the CTA to divide them exactly is what lets its loop carry no
    // bound check on the slot index -- the alternative is a compare per slot
    // on a path that is already only a few instructions long.
    if ((t.block_m * (t.block_n / gates)) % t.threads != 0) {
        return false;
    }
    int const budget = smemBudgetForSm(sm);
    if (budget <= 0 || maxThreadsPerSmForSm(sm) < t.threads) {
        return false;
    }
    return recurrentStepSmemBytes(t, elem_size) <= budget;
}

/*! \brief How many of these CTAs are resident at once, by shared memory and warps.
 *
 * Registers are deliberately not modelled: these tiles measure well inside
 * what shared memory already allows.  A tile that changed that would show up
 * as a *measured* regression, not as a wrong number here.
 */
constexpr int recurrentStepCtasPerSm(RecurrentStepTile t, int sm, int elem_size) {
    int const bytes = recurrentStepSmemBytes(t, elem_size);
    int const budget = smemBudgetForSm(sm);
    int const by_smem = bytes > 0 ? budget / bytes : kRecurrentStepMaxBlocksPerSm;
    int const threads_per_sm = maxThreadsPerSmForSm(sm);
    int const by_threads = t.threads > 0 ? threads_per_sm / t.threads : 0;
    int r = by_smem < by_threads ? by_smem : by_threads;
    if (r > kRecurrentStepMaxBlocksPerSm) {
        r = kRecurrentStepMaxBlocksPerSm;
    }
    return r < 1 ? 1 : r;
}

/*! \brief Index into `kRecurrentStepTiles` for this shape, or -1 if none fits.
 *
 * The first rung of `kRecurrentStepRoutes` whose two bounds both fit wins; if
 * that rung's tile does not fit this architecture, the scan continues, so a
 * part with a smaller shared-memory budget degrades to a shallower ring rather
 * than declining outright.
 *
 * **Never make this depend on CUDA-graph capture state** (`AGENTS.md` rule
 * 11): two tiles sum the K loop in different orders, so a capture-dependent
 * answer would make a replayed graph produce different numbers than eager, and
 * a one-ulp difference has changed a decoded token in this repo before.
 * Everything here is a function of the arguments.
 */
constexpr int recurrentStepSelectTile(int sm, int hidden, int batch, int elem_size, int gates) {
    if (hidden <= 0 || batch <= 0) {
        return -1;
    }
    for (int i = 0; i < kRecurrentStepRouteCount; ++i) {
        RecurrentStepRoute const r = kRecurrentStepRoutes[i];
        if (hidden <= r.hidden_max && batch <= r.batch_max &&
            recurrentStepTileValid(kRecurrentStepTiles[r.tile], sm, elem_size, gates)) {
            return r.tile;
        }
    }
    // Degenerate: nothing the ladder names fits this architecture.  Take the
    // smallest ring that does rather than declining -- a shallower pipeline is
    // strictly better than routing the shape back to a GEMM plus a finalizer.
    int best = -1;
    int best_bytes = 0;
    for (int i = 0; i < kRecurrentStepTileCount; ++i) {
        if (!recurrentStepTileValid(kRecurrentStepTiles[i], sm, elem_size, gates)) {
            continue;
        }
        int const bytes = recurrentStepSmemBytes(kRecurrentStepTiles[i], elem_size);
        if (best < 0 || bytes < best_bytes) {
            best = i;
            best_bytes = bytes;
        }
    }
    return best;
}

}  // namespace recurrent
}  // namespace oasr
