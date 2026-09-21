// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Shared CuTe helpers for the FMHA kernel: accumulator re-views, the
// two gemm shapes the mainloop needs, a predicated tiled copy, and the
// row-quad reduction.
//
// Structurally modelled on FlashAttention's `hopper/utils.h`; the
// implementation is OASR-native.  Where behaviour differs it is called out at
// the site, because "it matches FA" is the reason most of this file is shaped
// the way it is and the exceptions are the interesting part.

#pragma once

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

namespace oasr {
namespace attention {

using namespace cute;

// ---------------------------------------------------------------------------
// Accumulator re-views
// ---------------------------------------------------------------------------

/*! \brief Re-view an MMA-C accumulator as `(row, col)`.
 *
 * `acc` comes out of `partition_fragment_C` shaped `((2, 2, V), MMA_M, MMA_N)`.
 * The m16n8k16 atom gives each thread two rows per MMA_M, so the row axis is
 * `(2, MMA_M)` and everything else is column.  Every per-row quantity in this
 * kernel -- the running max, the running sum, the mask's row index -- indexes
 * through this view.
 */
template <class Layout>
CUTLASS_DEVICE auto convert_layout_acc_rowcol(Layout acc_layout) {
    static_assert(decltype(rank(acc_layout))::value == 3);
    // `((2, 2), MMA_M, MMA_N)`.  m16n8k16 gives each thread a 2x2 block: the
    // inner mode (stride 1) is a *column* pair, the outer (stride 2) a *row*
    // pair.  FA3's version also carries a `V` mode and asserts rank<0> == 3;
    // that is the Hopper fragment, not this one.
    static_assert(decltype(rank<0>(acc_layout))::value == 2,
                  "SM80's MMA-C fragment is ((2, 2), MMA_M, MMA_N)");
    return make_layout(make_layout(get<0, 1>(acc_layout), get<1>(acc_layout)),
                       make_layout(get<0, 0>(acc_layout), get<2>(acc_layout)));
}

/*! \brief Re-view an MMA-C accumulator as an MMA-A fragment.
 *
 * `P` is produced as the QK gemm's output and consumed as the PV gemm's A
 * operand without ever going through shared memory.  The two layouts differ
 * only in how the leaf modes group, so this is a pure re-view.
 */
template <class MMA, class Layout>
CUTLASS_DEVICE auto convert_layout_acc_Aregs(Layout acc_layout) {
    using X = Underscore;
    static_assert(decltype(rank(acc_layout))::value == 3);
    static_assert(decltype(rank<0>(acc_layout))::value == 2);
    // `((2, 2), MMA_M, MMA_N)` -> `(((2, 2), 2), MMA_M, MMA_N / 2)`: the A
    // operand of m16n8k16 is twice as wide in k as one C tile is in n, so two
    // neighbouring n tiles fold into one k tile.
    auto l = logical_divide(acc_layout, Shape<X, X, _2>{});
    return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
}

// ---------------------------------------------------------------------------
// Reductions across the 4-thread row quad
// ---------------------------------------------------------------------------

template <typename T>
struct MaxOp {
    CUTLASS_DEVICE T operator()(T const& a, T const& b) const { return a > b ? a : b; }
};
template <typename T>
struct SumOp {
    CUTLASS_DEVICE T operator()(T const& a, T const& b) const { return a + b; }
};

/*! \brief Butterfly all-reduce across `Width` neighbouring lanes.
 *
 * m16n8k16 puts each accumulator row on a quad of four lanes, so a row-wise max
 * or sum is two `shfl.bfly` steps at offsets 2 and 1.  Keep it exactly that:
 * a wider `__reduce_*` or a tree changes the fp32 association, and the online
 * softmax's carried state is order-sensitive.
 */
template <int Width>
struct Allreduce {
    static_assert(Width == 32 || Width == 16 || Width == 8 || Width == 4 || Width == 2);
    template <typename T, typename Op>
    static CUTLASS_DEVICE T run(T x, Op& op) {
        constexpr int kOffset = Width / 2;
        x = op(x, __shfl_xor_sync(uint32_t(-1), x, kOffset));
        return Allreduce<kOffset>::run(x, op);
    }
};
template <>
struct Allreduce<2> {
    template <typename T, typename Op>
    static CUTLASS_DEVICE T run(T x, Op& op) {
        return op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
    }
};

/*! \brief Per-thread reduction along a row of the `(row, col)` accumulator view. */
template <bool zero_init = true, typename Engine0, typename Layout0, typename Engine1,
          typename Layout1, typename Operator>
CUTLASS_DEVICE void thread_reduce_(Tensor<Engine0, Layout0> const& tensor,
                                   Tensor<Engine1, Layout1>& summary, Operator& op) {
    static_assert(Layout0::rank == 2, "only 2D (row, col) views reduce");
    static_assert(Layout1::rank == 1, "summary is per row");
    CUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
    CUTLASS_PRAGMA_UNROLL
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        summary(mi) = zero_init ? tensor(mi, 0) : op(summary(mi), tensor(mi, 0));
        CUTLASS_PRAGMA_UNROLL
        for (int ni = 1; ni < size<1>(tensor); ++ni) {
            summary(mi) = op(summary(mi), tensor(mi, ni));
        }
    }
}

template <typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Operator>
CUTLASS_DEVICE void quad_allreduce_(Tensor<Engine0, Layout0>& dst,
                                    Tensor<Engine1, Layout1>& src, Operator& op) {
    CUTE_STATIC_ASSERT_V(size(dst) == size(src));
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(dst); ++i) {
        dst(i) = Allreduce<4>::run(src(i), op);
    }
}

template <bool zero_init = true, typename Engine0, typename Layout0, typename Engine1,
          typename Layout1>
CUTLASS_DEVICE void reduce_max(Tensor<Engine0, Layout0> const& tensor,
                               Tensor<Engine1, Layout1>& max) {
    MaxOp<float> max_op;
    thread_reduce_<zero_init>(tensor, max, max_op);
    quad_allreduce_(max, max, max_op);
}

/*! \brief Row sum, thread-local only.
 *
 * The cross-lane reduction is deliberately **not** done here: the running sum
 * is only ever read in `finalize`, so reducing per tile would be `n_block`
 * butterflies for one answer.
 */
template <bool zero_init = true, typename Engine0, typename Layout0, typename Engine1,
          typename Layout1>
CUTLASS_DEVICE void reduce_sum(Tensor<Engine0, Layout0> const& tensor,
                               Tensor<Engine1, Layout1>& sum) {
    SumOp<float> sum_op;
    thread_reduce_<zero_init>(tensor, sum, sum_op);
}

/*! \brief `1/x` via the hardware approximate reciprocal.
 *
 * Spelled as PTX rather than `1.f / x` so it is `rcp.approx.f32` whether or not
 * `--use_fast_math` is on, matching the CuTeDSL backend's
 * `cute.arch.rcp_approx`.  The two backends are compared against each other at
 * tight tolerance, so a divide that changes with a compiler flag is a hazard.
 */
CUTLASS_DEVICE float rcp_approx(float x) {
    float r;
    asm volatile("rcp.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

// ---------------------------------------------------------------------------
// Type conversion
// ---------------------------------------------------------------------------

/*! \brief Round-to-nearest convert an fp32 fragment down to the served dtype. */
template <typename To_type, typename Engine, typename Layout>
CUTLASS_DEVICE auto convert_type(Tensor<Engine, Layout> const& tensor) {
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel>*>(
        tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

// ---------------------------------------------------------------------------
// Shared-memory layout atom
// ---------------------------------------------------------------------------

/*! \brief Elements per smem "row" for a given head dim.
 *
 * One row should cover a whole number of 128-byte cache lines where it can.
 * At head_dim 128 fp16 that is 128 elements and a `Swizzle<4,3,3>` atom, where
 * the CuTeDSL backend's layout helper tops out at 64 -- a wider atom means
 * fewer, larger ldmatrix transactions for the same tile.
 */
constexpr int fmhaBlockKGmem(int head_dim, int elem_size) {
    int const bytes = head_dim * elem_size;
    int const b = (bytes % 128 == 0) ? 128 : ((bytes % 64 == 0) ? 64 : 32);
    return b / elem_size;
}

constexpr int fmhaSwizzleBits(int block_k_gmem) {
    return block_k_gmem == 128 ? 4 : (block_k_gmem == 64 ? 3 : 2);
}

/*! \brief The swizzled `(8, kBlockKGmem)` atom every Q/K/V/O layout tiles from.
 *
 * Shared by the mainloop and the epilogue so that `sO` can alias the mainloop's
 * `sV + sK` region: two different atoms would make the union unsound.
 */
template <class Element, int kHeadDim>
struct FmhaSmemLayoutAtom {
    static constexpr int kBlockKGmem = fmhaBlockKGmem(kHeadDim, int(sizeof(Element)));
    static constexpr int kSwizzle = fmhaSwizzleBits(kBlockKGmem);
    using type = decltype(composition(
        Swizzle<kSwizzle, 3, 3>{},
        Layout<Shape<_8, Int<kBlockKGmem>>, Stride<Int<kBlockKGmem>, _1>>{}));
};

// ---------------------------------------------------------------------------
// Predicated tiled copy
// ---------------------------------------------------------------------------

/*! \brief Tiled copy with independent M/N and K predication.
 *
 * \tparam Is_even_MN   skip the row predicate entirely
 * \tparam Is_even_K    skip the k predicate entirely
 * \tparam Clear_OOB_MN zero the destination rows that are skipped
 * \tparam Clear_OOB_K  zero the destination k-slices that are skipped
 *
 * `identity_MN` is a thread-0 identity tensor: its entries are compile-time
 * constants, so comparing against a limit that has had this thread's own offset
 * subtracted out keeps the predicate free of per-thread arithmetic.  That trick
 * is FA's and it is worth keeping.
 *
 * With the ZFILL cp.async atom, `Clear_OOB_K` costs nothing -- a predicated-off
 * copy already writes zeros -- which is what makes it safe to bound the K/V
 * load by the sequence length instead of by the tensor extent.
 */
template <bool Is_even_MN = true, bool Is_even_K = true, bool Clear_OOB_MN = false,
          bool Clear_OOB_K = true, class CopyAtom, class TV, class Tiler, typename Engine0,
          typename Layout0, typename Engine1, typename Layout1, typename Engine2,
          typename Layout2, typename Engine3, typename Layout3>
CUTLASS_DEVICE void copy_predicated(TiledCopy<CopyAtom, TV, Tiler> const& tiled_copy,
                                    Tensor<Engine0, Layout0> const& S,
                                    Tensor<Engine1, Layout1>& D,
                                    Tensor<Engine2, Layout2> const& identity_MN,
                                    Tensor<Engine3, Layout3> const& predicate_K,
                                    int max_MN = 0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));
    auto copy_atom = static_cast<CopyAtom const&>(tiled_copy);
    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < size<1>(S); ++m) {
        if (Is_even_MN || get<0>(identity_MN(_0{}, m, _0{})) < max_MN) {
            CUTLASS_PRAGMA_UNROLL
            for (int k = 0; k < size<2>(S); ++k) {
                if (Is_even_K || predicate_K(k)) {
                    cute::copy(copy_atom, S(_, m, k), D(_, m, k));
                } else if (Clear_OOB_K) {
                    cute::clear(D(_, m, k));
                }
            }
        } else if (Clear_OOB_MN) {
            cute::clear(D(_, m, _));
        }
    }
}

// ---------------------------------------------------------------------------
// The two gemm shapes the mainloop needs
// ---------------------------------------------------------------------------

/*! \brief `acc += A @ B` with both operands read from shared memory.
 *
 * The QK gemm.  `ldmatrix` for k-tile 0 is issued before the loop and k-tile
 * `i+1` before step `i`'s MMA, so the shared-memory read for the next step
 * overlaps this step's tensor-core work.  `hook` fires once, after the first
 * k-tile's `ldmatrix` has been issued -- the mainloop uses it to launch the
 * next V cp.async so that copy overlaps the whole gemm rather than sitting
 * before it.
 *
 * \tparam A_in_regs  A was read into registers in the prologue; skip its
 *                    per-step `ldmatrix` entirely.
 */
template <bool A_in_regs = false, typename Tensor0, typename Tensor1, typename Tensor2,
          typename Tensor3, typename Tensor4, typename TiledMma, typename TiledCopyA,
          typename TiledCopyB, typename ThrCopyA, typename ThrCopyB, typename Hook>
CUTLASS_DEVICE void gemm_sm80(Tensor0& acc, Tensor1& tCrA, Tensor2& tCrB, Tensor3 const& tCsA,
                              Tensor4 const& tCsB, TiledMma tiled_mma,
                              TiledCopyA const& smem_tiled_copy_A,
                              TiledCopyB const& smem_tiled_copy_B,
                              ThrCopyA const& smem_thr_copy_A,
                              ThrCopyB const& smem_thr_copy_B, Hook fn) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));
    if constexpr (!A_in_regs) {
        cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{}));
    }
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            if constexpr (!A_in_regs) {
                cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1));
            }
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
        }
        if (i == 0) {
            fn();
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

/*! \brief `acc += A @ B` with A already in registers.
 *
 * The PV gemm: `A` is `P`, which the softmax just produced in registers, so
 * only `B` (the transposed V tile) comes from shared memory.
 */
template <typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
          typename TiledMma, typename TiledCopy, typename ThrCopy>
CUTLASS_DEVICE void gemm_rs_sm80(Tensor0& acc, Tensor1 const& tCrA, Tensor2& tCrB,
                                 Tensor3 const& tCsB, TiledMma tiled_mma,
                                 TiledCopy const& smem_tiled_copy_B,
                                 ThrCopy const& smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

// ---------------------------------------------------------------------------
// cp.async fences
// ---------------------------------------------------------------------------

/*! \brief Wait until at most \p N cp.async groups are still in flight. */
template <int N>
CUTLASS_DEVICE void cp_async_wait() {
    cute::cp_async_wait<N>();
}

}  // namespace attention
}  // namespace oasr
