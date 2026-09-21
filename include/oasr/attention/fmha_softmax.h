// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Online softmax for the OASR FMHA kernel.
//
// Structurally modelled on FlashAttention's `hopper/softmax.h`, and this is the
// one file in the family where that model must be followed *selectively*.
// Seven of the numbered points below are places where copying FA's reference
// verbatim is wrong for OASR, each for a reason that cost a shipped defect to
// learn.  The list is the contract; an implementer should be able to check it
// line by line.
//
// ---------------------------------------------------------------------------
// The contract
// ---------------------------------------------------------------------------
//
//  1. The `exp2` argument is `(s - m_ref) * scale_log2` -- **subtract first**,
//     through a named local.  Never `s * scale - m_ref * scale`.
//
//     FA's `softmax.h:85` writes the second form on purpose ("This allows the
//     compiler to use the ffma instruction").  Measured on this repo's own nvcc
//     flags (`.artifacts/fmha_cpp_validation.md` § (b)) it lowers to
//
//         FMUL  R0, m, scale        // rounded product, hoisted out of the loop
//         FFMA  R0, s, scale, -R0   // s*scale at full precision, minus it
//         MUFU.EX2
//
//     so the subtraction is between a full-precision product and a *rounded*
//     one, and the result can come out positive when `s <= m_ref`.  The
//     subtract-first form lowers to FADD/FMUL/MUFU and cannot.
//
//     The condition that reaches it, measured rather than assumed: a row whose
//     running max **is** a large *finite* mask floor.  `Check_inf` clamps
//     `-inf` but deliberately not a finite max, so such a row computes
//     `exp2(floor*c - round(floor*c))`; at `floor = -1e12` the ulp of that
//     product is large enough that the argument lands in the hundreds and
//     `exp2` overflows.  It is bf16-only in practice, because fp16 flushes a
//     floor that large to `-inf` and takes the clamp instead.  Conformer's
//     `mask_to_bias` writes `-1e10`, so this is a shipped configuration, not a
//     synthetic one.  Reproduced both ways: with the FA form the output goes
//     non-finite at `-1e12`; with this one it stays finite at every floor.
//
//  2. The carried `row_max` is the **true** max, `-inf` included.  The
//     empty-row clamp lives in `row_max_ref` and is local to the tile.
//
//     Storing the clamp is `AGENTS.md` rule 10: the running max becomes
//     `max(0, m)`, every later tile whose real max is negative gets
//     exponentiated about 0 instead of about itself, `P` loses `max(P) == 1`,
//     and the cast to fp16 turns the row subnormal (m ~ -11) or flushes it to
//     zero (m <~ -20) while `row_sum` stays fp32-exact.  Pinned by
//     `tests/kernels/test_fmha.py::TestInfiniteMaskFloorWithALargeBias`.
//
//     The clamp is computed in exactly one place (`max_get_scale`) and stored
//     in `row_max_ref` for `online_softmax` to reuse.  FA recomputes it in two
//     places; having one copy is what stops them drifting apart.
//
//  3. The rescale uses an explicit `prev_empty` select rather than relying on
//     `exp2f(-inf) == 0`.  A row with no unmasked column yet has `row_sum == 0`
//     and `acc_O == 0`, so the correct factor is exactly 0 -- and spelling it
//     out keeps `-inf` out of the arithmetic, where `softmax_scale == 0` (a
//     legal input) would otherwise make `(-inf - 0) * 0` a NaN.
//
//  4. `finalize` maps a zero or NaN row sum to scale **1.0**, not 0.0.  For a
//     fully masked row `acc_O` is exactly zero, so both give zero -- but 1.0
//     also survives a non-finite `acc_O`, where `0 * inf` would be NaN.  A
//     fully masked query row must come back exactly zero: SDPA's math backend
//     gives NaN there, and a NaN pad row is not inert, because the next layer's
//     masked key still contributes `0 * NaN` and poisons the real rows.
//
//  5. Row reductions are two `shfl.bfly` steps at offsets 2 then 1 across the
//     m16n8k16 row quad (`Allreduce<4>`).  Not a wider intrinsic, not a tree:
//     the carried state is order-sensitive.
//
//  6. `Check_inf` is **true on every tile, including the unmasked interior**.
//
//     FA passes `false` for its interior loop (`mainloop_fwd_sm80.hpp:639`).
//     The guard is needed wherever a row may have had **no live key in any tile
//     visited so far** -- not, as one might assume, only where a tile is itself
//     fully masked.  The loop descends, so a row that is masked all the way
//     down still carries `row_max == -inf` when it reaches the interior, and
//     with `Check_inf = false` it computes `(-inf) - (-inf) = NaN`, which
//     propagates through `exp2` into `row_sum` and out to the caller.
//
//     Verified by reverting it: the fully-masked-row case below stops returning
//     zero.  An additive `-inf` bias filling one *interior* tile does **not**
//     reach it -- the earlier tiles have already set a finite max -- so that is
//     not the test to write.  Cost when true: one compare and one select per
//     row per tile.
//
//  7. The row sum reduces this tile's `p` from `0.0f` and adds
//     `row_sum_prev * delta` once, at the end:
//
//         row_sum = (((0 + p0) + p1) + ...) + row_sum_prev * delta
//
//     FA scales `row_sum` first and accumulates each `p` onto it
//     (`softmax.h:120,134`), giving `((row_sum*delta + p0) + p1) + ...`.  The
//     two are algebraically equal and not bit-equal, and the CuTeDSL backend
//     uses the first -- so copying FA here would change every output of a
//     kernel whose whole point is to be compared against that one.
//
//  8. The masked value is assigned `-INFINITY`, never `min`/`fmin` (see
//     `fmha_mask.h`).  Shared memory past a sequence end can hold a NaN bit
//     pattern, `Q dot K_stale` is then NaN, and only an assignment intercepts
//     it.
//
// `softmax_scale` arrives pre-multiplied by `log2(e)` from the host, so the
// kernel can use `exp2` throughout (cheaper than `exp` on Ampere) and the
// factor folds into the same subtraction it would have anyway.

#pragma once

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>

#include "fmha_utils.h"

namespace oasr {
namespace attention {

using namespace cute;

/*! \brief Per-thread online-softmax state for one Q tile.
 *
 * \tparam kNRows rows of the MMA-C accumulator this thread owns
 *   (`2 * (2 * kBlockM / NumThreads)` for m16n8k16).
 *
 * Use:
 * \code
 *   Softmax<kNRows> softmax(scale_log2);
 *   softmax.init();
 *   for (n_block descending) {
 *       auto scale = softmax.max_get_scale<Is_first, Check_inf>(acc_s);
 *       softmax.online_softmax<Is_first>(acc_s, scale);
 *       ... cast acc_s to P ...
 *       if (!Is_first) softmax.rescale_o(acc_o, scale);
 *       ... PV gemm ...
 *   }
 *   softmax.rescale_o(acc_o, softmax.finalize());
 * \endcode
 *
 * The max/exp split exists so the caller can defer the `acc_O` rescale past the
 * `P` conversion, overlapping it with the cast; it is a scheduling split, not a
 * numerical one, and `row_max_ref` is what keeps the two halves agreeing on the
 * reference point.
 */
template <int kNRows>
struct Softmax {
    using TensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));

    TensorT row_max;      //!< the true running max, `-inf` included (contract 2)
    TensorT row_max_ref;  //!< the clamped reference for *this* tile only
    TensorT row_sum;
    float const softmax_scale_log2;

    CUTLASS_DEVICE Softmax(float const scale_log2) : softmax_scale_log2(scale_log2) {}

    CUTLASS_DEVICE void init() {
        cute::fill(row_max, -INFINITY);
        cute::fill(row_sum, 0.f);
    }

    /*! \brief Fold this tile's max into the running one; return the O rescale.
     *
     * Updates `row_max` (true max) and `row_max_ref` (clamped, tile-local), and
     * returns the factor the accumulators built against the *previous*
     * reference must be multiplied by to rebase onto this one.
     */
    template <bool Is_first, bool Check_inf = true, typename Tensor0>
    CUTLASS_DEVICE TensorT max_get_scale(Tensor0& acc_s) {
        Tensor scores = make_tensor(acc_s.data(), convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(decltype(size<0>(scores))::value == kNRows);
        TensorT scores_scale;
        if constexpr (Is_first) {
            reduce_max</*zero_init=*/true>(scores, row_max);
            CUTLASS_PRAGMA_UNROLL
            for (int mi = 0; mi < size(row_max); ++mi) {
                // Contract 2: clamp for the exponent reference only; `row_max`
                // itself keeps the -inf.
                row_max_ref(mi) =
                    (Check_inf && row_max(mi) == -INFINITY) ? 0.0f : row_max(mi);
            }
            cute::fill(scores_scale, 1.f);
        } else {
            TensorT row_max_prev;
            cute::copy(row_max, row_max_prev);
            reduce_max</*zero_init=*/false>(scores, row_max);
            CUTLASS_PRAGMA_UNROLL
            for (int mi = 0; mi < size(row_max); ++mi) {
                row_max_ref(mi) =
                    (Check_inf && row_max(mi) == -INFINITY) ? 0.0f : row_max(mi);
                // Contract 3: an explicit empty-previous select.  Nothing has
                // been accumulated into this row yet, so the factor is exactly
                // zero and no -inf enters the arithmetic.
                bool const prev_empty = (row_max_prev(mi) == -INFINITY);
                // Contract 1: subtract first, then scale.
                float const d = row_max_prev(mi) - row_max_ref(mi);
                scores_scale(mi) = prev_empty ? 0.0f : ::exp2f(d * softmax_scale_log2);
            }
        }
        return scores_scale;
    }

    /*! \brief `acc_s <- exp2((acc_s - ref) * scale)`, and fold into `row_sum`.
     *
     * \param scores_scale the value `max_get_scale` returned for this tile.
     */
    template <bool Is_first, typename Tensor0>
    CUTLASS_DEVICE void online_softmax(Tensor0& acc_s, TensorT const& scores_scale) {
        Tensor scores = make_tensor(acc_s.data(), convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(decltype(size<0>(scores))::value == kNRows);
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < size<0>(scores); ++mi) {
            float const m_ref = row_max_ref(mi);
            CUTLASS_PRAGMA_UNROLL
            for (int ni = 0; ni < size<1>(scores); ++ni) {
                // Contract 1 again, on the hot path: the subtraction is its own
                // rounding step, so the argument is provably <= 0.
                float const d = scores(mi, ni) - m_ref;
                scores(mi, ni) = ::exp2f(d * softmax_scale_log2);
            }
        }
        // Contract 7: this tile reduces from zero into a temporary; the carried
        // sum is rebased and added once, afterwards.
        TensorT tile_sum;
        reduce_sum</*zero_init=*/true>(scores, tile_sum);
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < size(row_sum); ++mi) {
            row_sum(mi) = Is_first ? tile_sum(mi) : tile_sum(mi) + row_sum(mi) * scores_scale(mi);
        }
    }

    /*! \brief Reduce the row sums across the quad and return `1/sum`.
     *
     * Contract 4: a zero or NaN sum yields 1.0, which leaves an already-zero
     * `acc_O` at zero rather than turning a non-finite one into NaN.
     */
    CUTLASS_DEVICE TensorT finalize() {
        SumOp<float> sum_op;
        quad_allreduce_(row_sum, row_sum, sum_op);
        TensorT scores_scale;
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < size(row_sum); ++mi) {
            float const s = row_sum(mi);
            bool const bad = (s == 0.f) || (s != s);
            scores_scale(mi) = bad ? 1.0f : rcp_approx(s);
        }
        return scores_scale;
    }

    /*! \brief Per-row log-sum-exp, **base 2**, for split-KV.
     *
     * Call after :func:`finalize`, which is what reduces `row_sum` across the
     * quad.  The value is `row_max * softmax_scale_log2 + log2(row_sum)`; base
     * 2 rather than FlashAttention's natural log because the only consumer is
     * the combine kernel, which immediately feeds it back to `exp2`, and a
     * `log`/`exp` pair around a `log2`/`exp2` one is two conversions that
     * cancel.
     *
     * A row with no live key returns `-INFINITY` rather than a NaN.  That is
     * load-bearing: the combine weights each split by `exp2(lse - lse_max)`, so
     * `-inf` makes an empty split contribute exactly nothing, whereas the NaN
     * that `-inf * scale + log2(0)` would otherwise produce poisons the row.
     * Same family as contract 4, one layer out.
     */
    CUTLASS_DEVICE TensorT log_sum_exp() const {
        TensorT lse;
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < size(row_sum); ++mi) {
            float const s = row_sum(mi);
            bool const empty = (row_max(mi) == -INFINITY) || (s <= 0.f) || (s != s);
            lse(mi) = empty ? -INFINITY : row_max(mi) * softmax_scale_log2 + ::log2f(s);
        }
        return lse;
    }

    /*! \brief `acc_o(row, :) *= scores_scale(row)`. */
    template <typename Tensor1>
    CUTLASS_DEVICE void rescale_o(Tensor1& acc_o, TensorT const& scores_scale) {
        Tensor acc_o_rowcol =
            make_tensor(acc_o.data(), convert_layout_acc_rowcol(acc_o.layout()));
        static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
            CUTLASS_PRAGMA_UNROLL
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
                acc_o_rowcol(mi, ni) *= scores_scale(mi);
            }
        }
    }
};

}  // namespace attention
}  // namespace oasr
