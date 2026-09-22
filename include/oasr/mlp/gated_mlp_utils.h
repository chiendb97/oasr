// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// CuTe helpers for the fused gated MLP: the swizzled shared-memory atom, the
// accumulator re-view, a doubly-predicated tiled copy, and the dual-B gemm
// that is the whole point of the fusion.
//
// The attention family carries its own copies of the first three
// (`include/oasr/attention/fmha_utils.h`).  They are kept apart on purpose:
// `include/oasr/<family>/` is this repo's unit of kernel independence, and a
// shared CuTe grab-bag would couple two families' inner loops through a header
// that neither owns.  The pieces that are genuinely facts about the *hardware*
// rather than about a kernel -- shared-memory capacity, warp slots -- do live
// in one place (`oasr/common/arch_facts.h`); these are not that.
//
// `gemm_dual_sm80` has no counterpart over there and is the reason this file
// exists at all.

#pragma once

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "cutlass_gated_mlp_configs.h"

namespace oasr {
namespace mlp {

using namespace cute;

// ---------------------------------------------------------------------------
// Shared-memory layout atom
// ---------------------------------------------------------------------------

constexpr int gatedMlpSwizzleBits(int row_width) {
    return row_width == 128 ? 4 : (row_width == 64 ? 3 : 2);
}

/*! \brief The swizzled `(8, row)` atom every X / W / O layout tiles from.
 *
 * \tparam kExtent the tile's *contiguous* extent -- `block_k` for the mainloop
 *   operands (X and both weights are K-major) and `block_n` for the output.
 *
 * The swizzle is what makes the `ldmatrix` that follows bank-conflict free.
 * One row covers a whole number of 128-byte cache lines where the extent
 * allows, which is a wider atom -- and so fewer, larger `ldmatrix`
 * transactions -- than the CuTeDSL lane's helper can express (it tops out at
 * 64 elements).
 */
template <class Element, int kExtent>
struct GatedMlpSmemLayoutAtom {
    static constexpr int kRowWidth = gatedMlpSmemRowWidth(kExtent, int(sizeof(Element)));
    static constexpr int kSwizzle = gatedMlpSwizzleBits(kRowWidth);
    using type = decltype(composition(
        Swizzle<kSwizzle, 3, 3>{},
        Layout<Shape<_8, Int<kRowWidth>>, Stride<Int<kRowWidth>, _1>>{}));
};

// ---------------------------------------------------------------------------
// Accumulator re-view
// ---------------------------------------------------------------------------

/*! \brief Re-view an MMA-C accumulator as `(row, col)`.
 *
 * `acc` comes out of `partition_fragment_C` shaped `((2, 2), MMA_M, MMA_N)`.
 * The m16n8k16 atom gives each thread a 2x2 block whose inner mode (stride 1)
 * is a *column* pair and whose outer mode is a *row* pair.  The epilogue needs
 * this because the bias is per **column**: in this view a thread's distinct
 * columns are `size<1>`, so the bias is read once per column instead of once
 * per accumulator element.
 */
template <class Layout>
CUTLASS_DEVICE auto convert_layout_acc_rowcol(Layout acc_layout) {
    static_assert(decltype(rank(acc_layout))::value == 3);
    static_assert(decltype(rank<0>(acc_layout))::value == 2,
                  "SM80's MMA-C fragment is ((2, 2), MMA_M, MMA_N)");
    return make_layout(make_layout(get<0, 1>(acc_layout), get<1>(acc_layout)),
                       make_layout(get<0, 0>(acc_layout), get<2>(acc_layout)));
}

/*! \brief Round-to-nearest convert an fp32 fragment down to the served dtype.
 *
 * Returns an **owning** register tensor.  FlashAttention's version of this
 * (and the attention family's copy of it) returns a view over a local
 * `cutlass::Array` that has already gone out of scope; it survives because
 * everything inlines and the array stays in registers, but it is a dangling
 * reference on paper and there is no reason to inherit it.
 *
 * The packed `NumericArrayConverter` rather than a per-element `static_cast`:
 * it emits `cvt.rn.f16x2.f32`, halving the instruction count for the same
 * rounding.
 */
template <typename To_type, typename Engine, typename Layout>
CUTLASS_DEVICE auto convert_type(Tensor<Engine, Layout> const& tensor) {
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    Tensor out = make_tensor<To_type>(tensor.layout());
    cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    *reinterpret_cast<cutlass::Array<To_type, numel>*>(out.data()) =
        convert_op(*reinterpret_cast<cutlass::Array<From_type, numel> const*>(tensor.data()));
    return out;
}

// ---------------------------------------------------------------------------
// Predicated tiled copy
// ---------------------------------------------------------------------------

/*! \brief Tiled copy that **skips** what it cannot write.
 *
 * The epilogue's store, and only that: rows past `M` belong to no output row
 * at all, and columns past `N` belong to the *next row* of a row-major buffer,
 * so writing either would corrupt real output rather than merely waste a
 * store.  There is deliberately no "clear the skipped part" mode -- the one
 * caller that needs zeros is the mainloop, and it gets them from the ZFILL
 * cp.async in :func:`copy_zfill_2d` instead of from an `STS` that ptxas would
 * have to order against the asynchronous copies around it.
 *
 * \tparam Is_even_MN  skip the row predicate entirely
 * \tparam Is_even_K   skip the contiguous-axis predicate entirely
 *
 * \param predicate_MN  a **precomputed bool per row**.
 * \param identity_MN  a **thread-0** identity tensor, for the K axis.  Its
 *   entries are compile-time constants, so comparing them against a limit that
 *   has had this thread's own offset subtracted out keeps the predicate free
 *   of per-thread arithmetic.  That trick is FlashAttention's.
 * \param max_K  contiguous elements still in range, already offset-adjusted.
 */
template <bool Is_even_MN = true, bool Is_even_K = true, class CopyAtom, class TV,
          class Tiler, typename Engine0, typename Layout0, typename Engine1,
          typename Layout1, typename Engine2, typename Layout2, typename Engine3,
          typename Layout3>
CUTLASS_DEVICE void copy_predicated_2d(TiledCopy<CopyAtom, TV, Tiler> const& tiled_copy,
                                       Tensor<Engine0, Layout0> const& S,
                                       Tensor<Engine1, Layout1>& D,
                                       Tensor<Engine2, Layout2> const& identity_MN,
                                       Tensor<Engine3, Layout3> const& predicate_MN,
                                       int max_K = 0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));
    auto copy_atom = static_cast<CopyAtom const&>(tiled_copy);
    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < size<1>(S); ++m) {
        if (Is_even_MN || predicate_MN(m)) {
            CUTLASS_PRAGMA_UNROLL
            for (int k = 0; k < size<2>(S); ++k) {
                if (Is_even_K || int(get<1>(identity_MN(_0{}, _0{}, k))) < max_K) {
                    cute::copy(copy_atom, S(_, m, k), D(_, m, k));
                }
            }
        }
    }
}

/*! \brief Branch-free gmem->smem copy that **zero-fills** what it skips.
 *
 * Same two predicates as :func:`copy_predicated_2d`, folded into the ZFILL
 * cp.async's own `src_size` operand instead of into control flow.  A
 * predicated-off copy writes zeros, and a zero is the identity for the dot
 * product being accumulated, so the row residue and the K residue are both
 * *correct* rather than merely safe.
 *
 * This is worth a helper of its own because the branching form is expensive
 * here in a way that is invisible in the source.  Written as
 * `if (pred) copy(...) else clear(...)`, ptxas has to order a synchronous
 * `STS` against an asynchronous `LDGSTS` to the same shared address, and it
 * does that by bracketing each copy in `BSSY`/`BSYNC` and padding it with
 * three dead `@!PT LDS RZ, [RZ]`.  Measured on the 64x64x32 tile: the load
 * section of the K loop went from ~60 instructions to ~12, the LSU pipe from
 * 2.34M to 1.2M instructions, and the kernel from 0.89x of the CuTeDSL
 * backend to faster than it.  Nothing about the C++ says any of that; the
 * SASS does.
 */
template <bool Is_even_MN = true, bool Is_even_K = true, class CopyAtom, class TV, class Tiler,
          typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
CUTLASS_DEVICE void copy_zfill_2d(TiledCopy<CopyAtom, TV, Tiler> const& tiled_copy,
                                  Tensor<Engine0, Layout0> const& S,
                                  Tensor<Engine1, Layout1>& D,
                                  Tensor<Engine2, Layout2> const& identity_MN,
                                  Tensor<Engine3, Layout3> const& predicate_MN,
                                  int max_K = 0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
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
// The dual-B gemm
// ---------------------------------------------------------------------------

/*! \brief `acc0 += A @ B0` and `acc1 += A @ B1` over one K tile, A read once.
 *
 * This is CUTLASS's `examples/45_dual_gemm` mainloop in the FlashAttention
 * idiom, and it is the reason a gated MLP is one kernel rather than two: the
 * gate and the up projection differ only in B, so A's `ldmatrix` is issued
 * once and fed to both `mma.sync` chains.
 *
 * The `ldmatrix` for k-step `i+1` is issued before step `i`'s MMAs, so the
 * shared-memory read overlaps the tensor-core work.  Two independent
 * accumulator chains also give the scheduler real work to fill the pipe with,
 * which is latency the single-B version has to hide with the prefetch alone.
 *
 * \param fn fires once, after the first k-step's `ldmatrix` has been issued.
 *   The mainloop uses it to launch the next stage's cp.async so that copy
 *   overlaps the MMA chain instead of sitting behind it.  At the shipped tiles
 *   that chain is only two to four k-steps long and the placement measures
 *   within noise of issuing the copy after the gemm; it is kept because it is
 *   the structure a longer chain wants and it costs nothing.
 */
template <typename Acc0, typename Acc1, typename FrgA, typename FrgB, typename SmemA,
          typename SmemB, typename TiledMma, typename TiledCopyA, typename TiledCopyB,
          typename ThrCopyA, typename ThrCopyB, typename Hook>
CUTLASS_DEVICE void gemm_dual_sm80(Acc0& acc0, Acc1& acc1, FrgA& tCrA, FrgB& tCrB0,
                                   FrgB& tCrB1, SmemA const& tCsA, SmemB const& tCsB0,
                                   SmemB const& tCsB1, TiledMma tiled_mma,
                                   TiledCopyA const& smem_tiled_copy_A,
                                   TiledCopyB const& smem_tiled_copy_B,
                                   ThrCopyA const& smem_thr_copy_A,
                                   ThrCopyB const& smem_thr_copy_B, Hook fn) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc0));
    CUTE_STATIC_ASSERT_V(size<1>(tCrB0) == size<2>(acc0));
    CUTE_STATIC_ASSERT_V(size<1>(tCrB1) == size<2>(acc1));
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB0));
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB1));
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    Tensor tCrB0_copy_view = smem_thr_copy_B.retile_D(tCrB0);
    Tensor tCrB1_copy_view = smem_thr_copy_B.retile_D(tCrB1);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));
    CUTE_STATIC_ASSERT_V(size<1>(tCsB0) == size<1>(tCrB0_copy_view));
    cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{}));
    cute::copy(smem_tiled_copy_B, tCsB0(_, _, _0{}), tCrB0_copy_view(_, _, _0{}));
    cute::copy(smem_tiled_copy_B, tCsB1(_, _, _0{}), tCrB1_copy_view(_, _, _0{}));
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1));
            cute::copy(smem_tiled_copy_B, tCsB0(_, _, i + 1), tCrB0_copy_view(_, _, i + 1));
            cute::copy(smem_tiled_copy_B, tCsB1(_, _, i + 1), tCrB1_copy_view(_, _, i + 1));
        }
        if (i == 0) {
            fn();
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB0(_, _, i), acc0);
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB1(_, _, i), acc1);
    }
}

}  // namespace mlp
}  // namespace oasr
