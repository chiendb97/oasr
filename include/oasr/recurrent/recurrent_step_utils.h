// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// CuTe helpers for the fused recurrent step: the swizzled shared-memory atom,
// the branch-free zero-filling tiled copy, and the gemm that carries the
// mainloop's refill hook.
//
// The attention and gated-MLP families carry their own copies of the first two
// (`include/oasr/attention/fmha_utils.h`, `include/oasr/mlp/gated_mlp_utils.h`).
// They are kept apart on purpose: `include/oasr/<family>/` is this repo's unit
// of kernel independence, and a shared CuTe grab-bag would couple three
// families' inner loops through a header that none of them owns.  The pieces
// that are genuinely facts about the *hardware* rather than about a kernel --
// shared-memory capacity, warp slots -- do live in one place
// (`oasr/common/arch_facts.h`); these are not that.

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include <cute/tensor.hpp>

#include "cutlass_recurrent_step_configs.h"

namespace oasr {
namespace recurrent {

using namespace cute;

// ---------------------------------------------------------------------------
// Shared-memory layout atom
// ---------------------------------------------------------------------------

constexpr int recurrentStepSwizzleBits(int row_width) {
    return row_width == 128 ? 4 : (row_width == 64 ? 3 : 2);
}

/*! \brief The swizzled `(8, row)` atom both mainloop operands tile from.
 *
 * \tparam kExtent the tile's *contiguous* extent -- `block_k`, because K is
 *   the contiguous axis of `previous_h` (row-major) and of `weight_hh`
 *   (row-major over the gate-interleaved N, K fastest).
 *
 * The swizzle is what makes the `ldmatrix` that follows bank-conflict free.
 * One row covers a whole number of 128-byte cache lines where the extent
 * allows, which is a wider atom -- and so fewer, larger `ldmatrix`
 * transactions -- than the CuTeDSL lane's helper can express
 * (`make_smem_swizzle_atom` tops out at 64 elements).
 */
template <class Element, int kExtent>
struct RecurrentStepSmemLayoutAtom {
    static constexpr int kRowWidth = recurrentStepSmemRowWidth(kExtent, int(sizeof(Element)));
    static constexpr int kSwizzle = recurrentStepSwizzleBits(kRowWidth);
    using type =
        decltype(composition(Swizzle<kSwizzle, 3, 3>{},
                             Layout<Shape<_8, Int<kRowWidth>>, Stride<Int<kRowWidth>, _1>>{}));
};

// ---------------------------------------------------------------------------
// Predicated tiled copy
// ---------------------------------------------------------------------------

/*! \brief Branch-free gmem->smem copy that **zero-fills** what it skips.
 *
 * Two predicates -- a precomputed bool per row, and a limit on the contiguous
 * axis -- folded into the ZFILL cp.async's own `src_size` operand instead of
 * into control flow.  A predicated-off copy writes zeros, and a zero is the
 * identity for the dot product being accumulated, so the row residue *and* the
 * K residue are both **correct** rather than merely safe.  That is what lets
 * this lane accept a hidden width that is not a whole number of K tiles, which
 * the CuTeDSL lane cannot.
 *
 * This is worth a helper of its own because the branching form -- which is
 * what the CuTeDSL lane writes -- is expensive here in a way that is invisible
 * in the source.  Written as `if (pred) copy(...)`, ptxas has to order a
 * synchronous store against an asynchronous `LDGSTS` to the same shared
 * address, and it does that by bracketing each copy in `BSSY`/`BSYNC` and
 * padding it with dead `@!PT LDS RZ, [RZ]`.  The gated-MLP family measured the
 * same substitution taking its K loop's load section from ~60 instructions to
 * ~12 and its LSU pipe from 2.34M to 1.2M instructions.  Nothing about the C++
 * says any of that; the SASS does.
 *
 * \tparam Is_even_MN  skip the row predicate entirely
 * \tparam Is_even_K   skip the contiguous-axis predicate entirely
 *
 * \param identity_MN  a **thread-0** identity tensor.  Its entries are
 *   compile-time constants, so comparing them against a limit that has had
 *   this thread's own offset subtracted out keeps the predicate free of
 *   per-thread arithmetic.  That trick is FlashAttention's.
 * \param max_K  contiguous elements still in range, already offset-adjusted.
 */
template <bool Is_even_MN = true, bool Is_even_K = true, class CopyAtom, class TV, class Tiler,
          typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Engine2,
          typename Layout2, typename Engine3, typename Layout3>
CUTLASS_DEVICE void copy_zfill_2d(TiledCopy<CopyAtom, TV, Tiler> const& tiled_copy,
                                  Tensor<Engine0, Layout0> const& S, Tensor<Engine1, Layout1>& D,
                                  Tensor<Engine2, Layout2> const& identity_MN,
                                  Tensor<Engine3, Layout3> const& predicate_MN, int max_K = 0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));
    auto copy_atom = static_cast<CopyAtom const&>(tiled_copy);
    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < size<1>(S); ++m) {
        bool const row_ok = Is_even_MN || predicate_MN(m);
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < size<2>(S); ++k) {
            bool const ok =
                row_ok && (Is_even_K || int(get<1>(identity_MN(_0{}, _0{}, k))) < max_K);
            cute::copy(copy_atom.with(ok), S(_, m, k), D(_, m, k));
        }
    }
}

// ---------------------------------------------------------------------------
// The mainloop gemm
// ---------------------------------------------------------------------------

/*! \brief `acc += A @ B` over one K tile, with a hook for the next refill.
 *
 * The `ldmatrix` for k-step `i + 1` is issued before step `i`'s MMAs, so the
 * shared-memory read overlaps the tensor-core work instead of running strictly
 * serially in front of it.
 *
 * Two differences from the CuTeDSL lane's `gemm_with_smem_prefetch`, both of
 * which cost it work it does not use:
 *
 *  * the prefetch is guarded by `i < size - 1` rather than wrapping with
 *    `(i + 1) % K_tiles`, so the last k-step does not issue a whole extra
 *    round of `ldmatrix` whose result is discarded;
 *  * the refill hook fires from *inside* the chain, after the first k-step's
 *    `ldmatrix`, so the next stage's cp.async overlaps the MMAs rather than
 *    sitting behind them.
 *
 * \param fn fires once, after the first k-step's `ldmatrix` has been issued.
 */
template <typename Acc, typename FrgA, typename FrgB, typename SmemA, typename SmemB,
          typename TiledMma, typename TiledCopyA, typename TiledCopyB, typename ThrCopyA,
          typename ThrCopyB, typename Hook>
CUTLASS_DEVICE void gemm_sm80(Acc& acc, FrgA& tCrA, FrgB& tCrB, SmemA const& tCsA,
                              SmemB const& tCsB, TiledMma tiled_mma,
                              TiledCopyA const& smem_tiled_copy_A,
                              TiledCopyB const& smem_tiled_copy_B, ThrCopyA const& smem_thr_copy_A,
                              ThrCopyB const& smem_thr_copy_B, Hook fn) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));
    cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{}));
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1));
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
        }
        if (i == 0) {
            fn();
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

}  // namespace recurrent
}  // namespace oasr
