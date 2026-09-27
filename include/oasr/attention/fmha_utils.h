// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// The CuTe helpers that belong to the FMHA kernel alone: the row-quad
// reductions the online softmax is built on, the approximate reciprocal its
// finalize uses, and the first K tile's load.
//
// Structurally modelled on FlashAttention's `hopper/utils.h`; the
// implementation is OASR-native.  Where behaviour differs it is called out at
// the site, because "it matches FA" is the reason most of this file is shaped
// the way it is and the exceptions are the interesting part.
//
// Everything that is not attention's own -- the accumulator re-views, the
// packed down-conversion, the swizzled shared-memory atom, the two warp-level
// gemm shapes, the zero-filling load and skipping store copies -- is the
// Ampere-class toolkit shared
// with the gated-MLP and recurrent-step families
// (`include/oasr/common/cute_sm80.h`), spelled `cute_sm80::` at every call
// site.

#pragma once

#include <oasr/common/cute_sm80.h>

namespace oasr {
namespace attention {

using namespace cute;

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
// The first K tile's load
// ---------------------------------------------------------------------------

/*! \brief Copy the rows a predicate admits; zero everything else with `STS`.
 *
 * The *branching* form of `cute_sm80::copy_zfill`, kept for exactly one load:
 * the first (highest) K tile's K and V, which are bounded by `seqlen_k` and
 * issued once per CTA, in the prologue.  Every other load in this family --
 * Q, every deeper K/V tile, the paged gather -- is branch-free, because in
 * the K loop the branching form costs ptxas a `BSSY`/`BSYNC` pair and three
 * dead `LDS` per copy (144-168 dead instructions per variant, removed for
 * 1.06x across the A/B).
 *
 * Why this one stays branching is scheduling, and it was measured rather than
 * assumed.  With the prologue entirely straight-line, ptxas hoists the address
 * arithmetic of the *later* ring stages above the Q and first-K requests, so
 * the two loads the first QK gemm is waiting for go out ~35 instructions
 * later.  The reconvergence regions here pin them to the head of the
 * prologue.  That is worth nothing on a long K loop and everything on a CTA
 * that walks one or two K tiles -- the streaming chunk and the sliding window:
 * over the 91-row A/B, branching here took the geomean from 1.062x to 1.069x
 * and the worst row (`B4 H8 T256 D64`, window 56) from 0.973x to 0.99x.
 *
 * Zeroing the skipped rows is correct for both operands: V's must be zero (a
 * stale NaN reaches the output through `P @ V`), and K's are masked anyway.
 */
template <class CopyAtom, class TensorS, class TensorD, class RowPred, class ColPred>
CUTLASS_DEVICE void copy_rows_or_clear(CopyAtom const& copy_atom, TensorS const& S,
                                       TensorD&& D, RowPred const& row_ok,
                                       ColPred const& col_ok) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < size<1>(S); ++m) {
        if (row_ok(m)) {
            CUTLASS_PRAGMA_UNROLL
            for (int k = 0; k < size<2>(S); ++k) {
                if (col_ok(k)) {
                    cute::copy(copy_atom, S(_, m, k), D(_, m, k));
                } else {
                    cute::clear(D(_, m, k));
                }
            }
        } else {
            cute::clear(D(_, m, _));
        }
    }
}

}  // namespace attention
}  // namespace oasr
