// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Per-architecture traits and the CTA-tile resolver for the fused multi-head
// attention kernel.  This is the `cutlass_*_configs.h` third of the
// three-header CUTLASS pattern (config / template / dispatch) that the GEMM,
// BMM and Conv2D families already follow -- see `include/oasr/gemm/
// cutlass_gemm_configs.h`.
//
// Two things live here and nowhere else:
//
//   * `FmhaArch<SM>`  -- what an architecture *is*: its shared-memory budget,
//     its MMA and copy atoms, and explicit capability bools.
//   * `fmhaResolveTile` -- what tile that architecture can run for a shape.
//
// Both are `constexpr` over an `sm` **argument**, never over `__CUDA_ARCH__` and
// never over a live device query.  That is deliberate and load-bearing:
//
//   * `csrc/fmha_jit_binding.cu` exports the resolver so one built module can
//     answer for every architecture, which is what lets a single box hold the
//     capability line for four of them (`tests/kernels/test_fmha_cpp.py::
//     TestTileSelectionAgrees`, mirroring the CuTeDSL side's
//     `TestEveryArchBudgetsItsOwnSmem`).
//   * A live `cudaDevAttrMaxSharedMemoryPerBlockOptin` read would make sm_120's
//     budget answer 163 KB on an A30 and delete exactly the coverage that
//     caught sm_86/sm_89 being budgeted with A100's shared memory
//     (`.artifacts/arch_portability_audit.md` § A3).
//
// The device query is not absent -- it is *demoted*.  `fmha.cuh` asserts the
// resolved `smem_bytes` against `oasr::getDeviceMaxSharedMemoryOptin()` at
// launch, so a part that grants less than this table claims fails loudly with
// both numbers rather than at `cudaErrorInvalidValue` inside
// `cudaFuncSetAttribute`.

#pragma once

// `cute/tensor.hpp` is CuTe's entry point and must come before any individual
// `cute/atom/*` header -- the atoms' free functions are declared against
// declarations it pulls in, and including them bare is a parse error.
#include <cute/tensor.hpp>

#include <oasr/common/arch_facts.h>

#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma_sm80.h>
#include <cutlass/numeric_types.h>

#include <cstdint>
#include <type_traits>

namespace oasr {
namespace attention {

/*! \brief Shared memory the driver keeps for itself, in bytes.
 *
 * Budgeting against the architectural maximum is what let the CuTeDSL recurrent
 * step clear `can_implement` at `num_stages=5` and then die at launch with an
 * empty error.  Same constant and same rationale as
 * `oasr/kernels/cute/attention/fmha_sm80.py::_DRIVER_SMEM_RESERVE` and
 * `oasr/kernels/cute/mlp/gated.py`, so the two backends budget alike.
 *
 * A property of the *driver*, not of attention, so it is stated once in
 * `oasr/common/arch_facts.h` and aliased here under this family's name.
 */
inline constexpr int kFmhaDriverSmemReserve = kDriverSmemReserve;

/*! \brief Opt-in shared memory an architecture offers a single block, in bytes.
 *
 * These are the architectural maxima, *not* a device query -- see the file
 * header.  They match what CuTeDSL's `get_smem_capacity_in_bytes("sm_NN")`
 * returns, which is what the CuTeDSL backend budgets against, so the two
 * backends agree on which shapes are implementable.
 *
 * Returns 0 for an architecture this family does not know, which
 * `fmhaResolveTile` turns into "no tile fits" rather than a wrong answer.
 */
constexpr int fmhaSmemCapacity(int sm) { return smemCapacityForSm(sm); }

/*! \brief Shared memory a launch on \p sm will actually be granted, in bytes. */
constexpr int fmhaSmemBudget(int sm) {
    return fmhaSmemCapacity(sm) == 0 ? 0 : fmhaSmemCapacity(sm) - kFmhaDriverSmemReserve;
}

/*! \brief Head dim rounded up to the m16n8k16 MMA k-stride.
 *
 * The smem layouts are built from this, so the budget must be costed with it
 * too.  Costing the raw value under-counts by up to a third (72 pads to 96) and
 * can approve a config that will not fit at launch.
 */
constexpr int fmhaPaddedHeadDim(int head_dim) { return (head_dim + 31) / 32 * 32; }

// ---------------------------------------------------------------------------
// Per-architecture traits
// ---------------------------------------------------------------------------

/*! \brief What one architecture is, for this kernel family.
 *
 * `Tag` selects *instructions*; `kIsSm86Or89` and `kSmemBudgetBytes` select
 * *tuning*.  Keeping those two jobs apart is why sm_86 and sm_89 route through
 * `cutlass::arch::Sm80` -- the tag they share is the tag whose instructions
 * they run -- while still budgeting their own 99 KB.  FlashAttention makes the
 * same split (`hopper/flash_fwd_launch_template.h:36` pairs
 * `cutlass::arch::Sm80` with a separate `Arch == 86 || Arch == 89` bool).
 *
 * \warning Never branch on `Tag::kMinComputeCapability >= 90`.  `Sm120`'s is
 *   120, so that test is *true* on consumer Blackwell -- which has no TMA and
 *   no warp specialization.  FA's epilogue selects its TMA store path exactly
 *   that way (`hopper/epilogue_fwd.hpp:37`), so porting the test along with the
 *   code would silently pick a path this kernel does not implement.  Branch on
 *   `kHasTma` / `kIsWarpSpecialized`, which say what they mean.
 */
template <int SmVersion>
struct FmhaArch;

namespace detail {

/*! \brief The Ampere-class (mma.sync + cp.async) traits every sm_8x/sm_12x shares. */
template <int SmVersion>
struct FmhaArchAmpere {
    using Tag = cutlass::arch::Sm80;

    static constexpr int kSmVersion = SmVersion;
    static constexpr int kSmemCapacityBytes = fmhaSmemCapacity(SmVersion);
    static constexpr int kSmemBudgetBytes = fmhaSmemBudget(SmVersion);

    //! Register file and L2 differ enough on the Ampere/Ada consumer parts to
    //! move the tile choice; they do not change which instructions are legal.
    static constexpr bool kIsSm86Or89 = (SmVersion == 86 || SmVersion == 89);

    static constexpr bool kHasCpAsync = true;
    static constexpr bool kHasTma = false;
    static constexpr bool kIsWarpSpecialized = false;

    //! Deepest cp.async ring worth building. Past this the extra latency hiding
    //! stops paying for the shared memory -- FlashAttention's own ceiling.
    static constexpr int kMaxStages = 3;
    static constexpr int kMaxThreadsPerBlock = 256;

    template <class Element>
    using MmaAtom = std::conditional_t<std::is_same_v<Element, cutlass::half_t>,
                                       cute::MMA_Atom<cute::SM80_16x8x16_F32F16F16F32_TN>,
                                       cute::MMA_Atom<cute::SM80_16x8x16_F32BF16BF16F32_TN>>;

    //! gmem -> smem for Q/K/V.  ZFILL, not the plain cp.async: a predicated-off
    //! copy writes **zeros** rather than leaving stale shared memory, which is
    //! what lets the K/V load be bounded by `cache_seqlens` and retires the
    //! caller-side "V must be finite past the length" precondition.
    template <class Element>
    using GmemCopyAtomKV =
        cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>, Element>;

    template <class Element>
    using GmemCopyAtomO =
        cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>, Element>;

    template <class Element>
    using SmemCopyAtom = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>;

    //! V is consumed transposed by the PV gemm; `ldmatrix.trans` does it for free.
    template <class Element>
    using SmemCopyAtomTransposed = cute::Copy_Atom<cute::SM75_U16x8_LDSM_T, Element>;
};

}  // namespace detail

template <>
struct FmhaArch<80> : detail::FmhaArchAmpere<80> {};
template <>
struct FmhaArch<86> : detail::FmhaArchAmpere<86> {};
template <>
struct FmhaArch<89> : detail::FmhaArchAmpere<89> {};
template <>
struct FmhaArch<120> : detail::FmhaArchAmpere<120> {};

// sm_90 and sm_100 deliberately have no specialization yet.  The Ampere-class
// mainloop would *run* there (mma.sync and cp.async are both available), but
// routing to it would be a performance change with no measurement behind it,
// and the right answer on those parts is a wgmma/TMA mainloop of their own.
// `oasr/jit/fmha.py` declares the omission and counts it rather than hiding it.
// Adding them later is: one `FmhaArch<90>` here, one collective, one
// `conditional_t` arm in `fmha_launch_template.h`.  Nothing else moves.

static_assert(!FmhaArch<120>::kHasTma && !FmhaArch<120>::kIsWarpSpecialized,
              "consumer Blackwell has neither; a kMinComputeCapability >= 90 test would "
              "claim both");

// ---------------------------------------------------------------------------
// Tile resolution
// ---------------------------------------------------------------------------

/*! \brief A resolved CTA tile, or `valid == false` if nothing fits. */
//! A half-open `[lo, hi)` K-tile range.  A plain struct rather than
//! `cute::tuple` so it can be `constexpr` in host code the tests compile
//! without CuTe.
struct cute_fmha_range {
    int lo;
    int hi;
};

struct FmhaTile {
    bool valid;
    int block_m;
    int block_n;
    int num_warps;
    int num_stages;
    bool q_in_regs;
    int smem_bytes;
};

/*! \brief Shared memory one tile needs, in bytes.
 *
 * `sQ + num_stages * (sK + sV)`, unioned with the epilogue's `sO`.  The
 * epilogue writes `sO` over `sV + sK` and never over `sQ`, so the mainloop
 * region only has to be padded when `sO` is the larger of the two.
 *
 * With `q_in_regs == false` and `head_dim_v == head_dim_q` this reduces to
 * `(M*D + N*D*stages*2) * elem_size`, which is bit-identical to the CuTeDSL
 * backend's `FmhaSm80.smem_bytes` -- so both backends approve the same set of
 * shapes for the same reason.
 */
constexpr int fmhaSmemBytes(int block_m, int block_n, int head_dim_q, int head_dim_v,
                            int num_stages, bool q_in_regs, int elem_size) {
    int const q = block_m * head_dim_q * elem_size;
    int const k = block_n * head_dim_q * num_stages * elem_size;
    int const v = block_n * head_dim_v * num_stages * elem_size;
    int const o = block_m * head_dim_v * elem_size;
    // `q_in_regs` aliases sQ onto sV: Q is read into registers in the prologue,
    // then the same bytes carry the V ring.
    int const mainloop = q_in_regs ? ((q > v ? q : v) + k) : (q + k + v);
    int const pad = (o - v - k) > 0 ? (o - v - k) : 0;
    int const with_pad = mainloop + pad;
    return with_pad > o ? with_pad : o;
}

/*! \brief K-tile widths to fall back on, widest first.
 *
 * 16 is the floor, not 8: the m16n8k16 atom is tiled across the warps, so a
 * 64x8 S tile leaves the QK accumulator's N mode not matching the fragment's
 * and `cute::gemm` refuses it.  `block_n` must stay a multiple of 16 for the
 * `Tile<Int<16*warps>, _16, _16>` permutation.
 */
inline constexpr int kFmhaNBlockLadder[] = {64, 32, 16};

/*! \brief Resolve `(block_m, block_n, warps, stages)` for a shape, or `invalid`.
 *
 * \param sm          compute capability as `major*10 + minor`
 * \param head_dim    **padded** head dim (`fmhaPaddedHeadDim`)
 * \param head_dim_v  padded value head dim; equal to \p head_dim today
 * \param elem_size   2 for fp16/bf16
 *
 * The widest K tile is tried first and at the deepest ring that fits, so a
 * shape that already worked keeps its config.  Only when *no* ring depth fits
 * does the K tile narrow: that costs iterations, so it is a last resort rather
 * than a default, and it is what makes very wide heads reachable at all
 * (head_dim 512 needs 192 KB at 64x64 even with one stage, and 96 KB at 64x16).
 *
 * `block_m` is not a search axis in this lane.  `(block_m * 2) % num_threads`
 * must be 0, so at 128 threads it is a multiple of 64 and cannot usefully
 * shrink.  A second lane with `block_m = 128` for long query extents is a
 * separate, measured decision -- OASR's two commonest shapes are `T_q = 8` and
 * `T_q = 1`, where a 128-row M tile is 94% and 99% padding respectively.
 *
 * Unlike the CuTeDSL backend, `num_stages == 1` is reachable here.  Its
 * `MIN_NUM_STAGES = 2` is a CuTeDSL IR artifact -- indexing a size-1 stage mode
 * fails verification there -- and does not apply to `sK(_, _, _0{})` in C++.
 * One stage is what makes a wide head fit a wide K tile.
 */
constexpr FmhaTile fmhaResolveTile(int sm, int head_dim, int head_dim_v, int elem_size) {
    constexpr int kNumWarps = 4;
    constexpr int kBlockM = 64;
    int const budget = fmhaSmemBudget(sm);
    if (budget <= 0 || head_dim <= 0 || elem_size <= 0) {
        return FmhaTile{false, 0, 0, 0, 0, false, 0};
    }
    int const max_stages = FmhaArch<80>::kMaxStages;  // same ceiling on every Ampere-class part
    for (int i = 0; i < int(sizeof(kFmhaNBlockLadder) / sizeof(int)); ++i) {
        int const block_n = kFmhaNBlockLadder[i];
        for (int stages = max_stages; stages >= 1; --stages) {
            int const bytes =
                fmhaSmemBytes(kBlockM, block_n, head_dim, head_dim_v, stages, false, elem_size);
            if (bytes <= budget) {
                return FmhaTile{true, kBlockM, block_n, kNumWarps, stages, false, bytes};
            }
        }
    }
    return FmhaTile{false, 0, 0, 0, 0, false, 0};
}

/*! \brief `fmhaResolveTile` taking a *raw* head dim, padding it first. */
constexpr FmhaTile fmhaResolveTileRaw(int sm, int head_dim_raw, int elem_size) {
    int const d = fmhaPaddedHeadDim(head_dim_raw);
    return fmhaResolveTile(sm, d, d, elem_size);
}

// ---------------------------------------------------------------------------
// Split-KV (flash-decoding)
// ---------------------------------------------------------------------------

//! Most K-range splits a decode step may be cut into.
//!
//! Bounded, not unbounded: the combine kernel re-reads one LSE per split per
//! row, and the partial buffers are `splits * B * H * T_q * D` fp32.  Sixteen
//! is enough to fill 170 SMs from a single-stream decode (`B*H = 32` there)
//! and keeps the workspace under a megabyte at the shapes that use it.
inline constexpr int kFmhaMaxSplits = 16;

//! Fewest K tiles a split must own to be worth launching.
//!
//! A split that walks one K tile pays the whole prologue -- Q load, cp.async
//! ring fill, the fp32 partial store -- for a single MMA pass.
inline constexpr int kFmhaMinBlocksPerSplit = 2;

/*! \brief Fewest K tiles in the whole range before splitting is considered.
 *
 * **Measured, not chosen**, and re-measured after the combine pass was fixed
 * (`fmha_combine.h`).  The previous value of this constant was 32, derived
 * against a combine that tiled by the attention pass's M tile and so ran at
 * 0.02 waves -- 2.6x the cost of the split mainloop it reduced.  That table
 * was measuring the combine, not an intrinsic cost of splitting: with the
 * combine decomposed per row instead, the same shapes that read 0.53x at 8 K
 * tiles now read 1.40x.
 *
 * At four K tiles every measured row still loses (0.83-0.87x eager, and one
 * 0.82x that only appears graph-replayed), because two tiles per split cannot
 * repay a second launch.  At six, every row at or below the utilisation gate
 * wins or ties in *both* regimes.
 */
inline constexpr int kFmhaMinBlocksForSplit = 6;

/*! \brief Utilisation above which the unsplit grid is already good enough.
 *
 * As a percentage of `waves / ceil(waves)`, where `waves = cta_count/num_sms`.
 *
 * Utilisation rather than raw CTA count, because the two come apart exactly
 * where it matters.  On 170 SMs, 128 CTAs is 0.75 waves -- one wave, 75 % full
 * -- and splitting it *loses* (0.59-0.86x).  192 CTAs is 1.13 waves, a full
 * wave plus a 13 %-full second one, so utilisation is 0.565 and splitting
 * *wins* (1.05-1.38x).  A rule on CTA count alone gets one of those two
 * backwards whichever threshold it picks.
 *
 * 60 admits 0.565 and refuses 0.753, which is the measured break.
 */
inline constexpr int kFmhaSplitMaxUtilPct = 60;

/*! \brief K tiles required per unit of grid fill before splitting pays.
 *
 * The gate is `n_blocks * num_sms >= kFmhaSplitDepthPerCta * cta_count`: the
 * fuller the machine already is, the more K depth a split has to divide before
 * it repays the combine.  Measured break-evens on 170 SMs at `head_dim 64`:
 *
 *      CTAs | waves | break-even K tiles | this rule admits from
 *      -----+-------+--------------------+----------------------
 *         8 | 0.05  |         6          |          6
 *        16 | 0.09  |         6          |          6
 *        32 | 0.19  |         6          |          6
 *        48 | 0.28  |        12          |         12
 *        64 | 0.38  |        12          |         12
 *        96 | 0.57  |        24          |         17
 *       192 | 1.13  |        16          |         34
 *
 * 30 is the largest value that still admits `(64, 12)` and the smallest that
 * still refuses `(96, 16)`, which measured 1.13x and 0.85x respectively.  The
 * cost is the last row: `(192, 24)` and `(192, 32)` measured 1.12x and 1.20x
 * and are refused.  Nothing that is a function of shape alone separates them
 * from `(128, 24)` at 0.84x, and the standing rule is that no row regresses.
 */
inline constexpr int kFmhaSplitDepthPerCta = 30;

//! `ceil(a / b)` for non-negative `a` and positive `b`.
constexpr int fmhaCeilDiv(int a, int b) {
    return (a + b - 1) / b;
}

/*! \brief Would an `s`-way split actually shorten the longest chunk?
 *
 * With 16 K tiles, 5 splits hand out `ceil(16/5) = 4` tiles just as 4 splits
 * do -- the fifth CTA set exists but removes no work from the critical path,
 * so it is pure combine cost.  Measured: 16 K tiles at 32 CTAs runs 12.29 us
 * at 4 splits and 14.34 us at 5.  FlashAttention guards the same case.
 */
constexpr bool fmhaSplitEligible(int s, int n_blocks) {
    return s <= 1 || fmhaCeilDiv(n_blocks, s) != fmhaCeilDiv(n_blocks, s - 1);
}

/*! \brief How many ways to cut the K range, from shape and SM count alone.
 *
 * **This must never depend on CUDA-graph capture state** (`AGENTS.md` rule 11).
 * Splitting changes the order the fp32 partials are summed, so a
 * capture-dependent answer makes a replayed graph produce different numbers
 * than eager -- and a one-ulp difference in attention has changed decoded
 * tokens in this repo before.  Everything here is a function of the arguments,
 * `num_sms` is passed in rather than queried, and the result is asserted equal
 * to the Python mirror by `tests/kernels/test_fmha_cpp.py`.
 *
 * **Integer arithmetic throughout, deliberately.**  The efficiency comparison
 * is a ratio, and the obvious spelling is `float`; but the Python mirror would
 * evaluate the same expression in double and the two can land on opposite
 * sides of a tie.  Cross-multiplying keeps both exact, which is what makes the
 * mirror test meaningful rather than approximately true.
 *
 * \param cta_count  `batch * num_heads * m_blocks` -- the CTAs the unsplit
 *                   grid would launch
 * \param n_blocks   K tiles the *longest* stream walks
 * \param num_sms    multiprocessors on the target device
 *
 * Three gates, then a search.  The gates are the measured constants above; the
 * search picks the smallest split count whose grid utilisation is within 15 %
 * of the best available, which biases small because the combine's traffic is
 * linear in the split count while the occupancy it buys saturates.
 */
constexpr int fmhaNumSplits(int cta_count, int n_blocks, int num_sms) {
    if (cta_count <= 0 || n_blocks <= 0 || num_sms <= 0) {
        return 1;
    }
    if (n_blocks < kFmhaMinBlocksForSplit) {
        return 1;  // too little K work to divide -- see the constant
    }
    if (int64_t(100) * cta_count >
        int64_t(kFmhaSplitMaxUtilPct) * num_sms * fmhaCeilDiv(cta_count, num_sms)) {
        return 1;  // the unsplit grid already uses the machine well enough
    }
    if (int64_t(n_blocks) * num_sms < int64_t(kFmhaSplitDepthPerCta) * cta_count) {
        return 1;  // not enough K depth for how full the grid already is
    }
    int hi = kFmhaMaxSplits;
    if (hi > num_sms) {
        hi = num_sms;
    }
    if (hi > n_blocks / kFmhaMinBlocksPerSplit) {
        hi = n_blocks / kFmhaMinBlocksPerSplit;
    }
    if (hi < 2) {
        return 1;
    }
    // Utilisation of an `s`-way grid as the exact rational `a_s / b_s`.
    int64_t am = int64_t(cta_count);
    int64_t bm = int64_t(num_sms) * fmhaCeilDiv(cta_count, num_sms);
    for (int s = 2; s <= hi; ++s) {
        if (!fmhaSplitEligible(s, n_blocks)) {
            continue;
        }
        int64_t const a = int64_t(cta_count) * s;
        int64_t const b = int64_t(num_sms) * fmhaCeilDiv(cta_count * s, num_sms);
        if (a * bm > am * b) {
            am = a;
            bm = b;
        }
    }
    for (int s = 1; s <= hi; ++s) {
        if (!fmhaSplitEligible(s, n_blocks)) {
            continue;
        }
        int64_t const a = int64_t(cta_count) * s;
        int64_t const b = int64_t(num_sms) * fmhaCeilDiv(cta_count * s, num_sms);
        if (int64_t(100) * a * bm >= int64_t(85) * am * b) {
            return s;
        }
    }
    return 1;
}

/*! \brief `[lo, hi)` of the K-tile range `[n_block_min, n_block_max)` for one split.
 *
 * Contiguous chunks, ceil-sized, so the last split may be short or empty.  An
 * empty split still runs -- it writes a `-inf` LSE and zero partial, which the
 * combine weights out -- because a grid whose CTA count depends on the *data*
 * (the per-stream `cache_seqlens`) is not capturable.
 */
constexpr cute_fmha_range fmhaSplitRange(int n_block_min, int n_block_max, int split_idx,
                                         int num_splits) {
    int const total = n_block_max - n_block_min;
    if (total <= 0 || num_splits <= 1) {
        return cute_fmha_range{n_block_min, n_block_max};
    }
    int const per = (total + num_splits - 1) / num_splits;
    int lo = n_block_min + split_idx * per;
    if (lo > n_block_max) {
        lo = n_block_max;
    }
    int hi = lo + per;
    if (hi > n_block_max) {
        hi = n_block_max;
    }
    return cute_fmha_range{lo, hi};
}

}  // namespace attention
}  // namespace oasr
