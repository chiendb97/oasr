// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// The fused recurrent step's arguments, and the compile-time state-transition
// dispatch.

#pragma once

#include <cutlass/cutlass.h>

#include <cstdint>
#include <oasr/common/math.h>

namespace oasr {
namespace recurrent {

/*! \brief Which recurrence one instantiation computes.
 *
 * A single enum rather than the CuTeDSL lane's `(gate_count, activation)`
 * pair, whose two halves have to agree (`(activation == "lstm") ==
 * (gate_count == 4)` is a clause of its `can_implement`).  Here the gate count
 * is *derived* from the kind, so the inconsistent combination cannot be
 * spelled.
 */
enum class RecurrentKind : int {
    LSTM = 0,      //!< four interleaved gates: i, f, g, o -- PyTorch / cuDNN order
    RNN_TANH = 1,  //!< one gate, `tanh`
    RNN_RELU = 2,  //!< one gate, `max(x, 0)`
};

//! Gate columns per hidden unit, for \p kind.
constexpr int recurrentGateCount(RecurrentKind kind) {
    return kind == RecurrentKind::LSTM ? 4 : 1;
}

//! Does \p kind carry a cell state?  When false neither cell pointer is live.
constexpr bool recurrentHasCell(RecurrentKind kind) {
    return kind == RecurrentKind::LSTM;
}

/*! \brief Everything one launch needs.
 *
 * Plain extents and row strides rather than CuTe shape tuples: every tensor
 * here is rank 2 with a contiguous trailing axis, so an `(extent, stride)`
 * pair per operand says all there is to say and keeps the struct readable from
 * the host binding.  The *row* strides are runtime values, which is what lets
 * a caller hand over a row-slice of a wider buffer without a copy -- the
 * CuTeDSL lane has to demand a fully compact tensor
 * (`mark_compact_shape_dynamic`).
 *
 * Shapes, with `G = recurrentGateCount(kind)`:
 *
 *   previous_h    (M, K)      K is the hidden width being reduced over
 *   weight_hh     (G*H, K)    gate-interleaved rows: row n is unit n/G, gate n%G
 *   input_gates   (M, G*H)    the sequence-wide input projection, already summed
 *   previous_c    (M, H)      LSTM only
 *   h             (M, H)      output
 *   c             (M, H)      output, LSTM only
 *
 * `K` and `H` are independent: `K` is the width of the state being read and
 * `H` the width being written.  They are equal for every shipped recurrent
 * layer, and the kernel never assumes it.
 */
template <class Element>
struct RecurrentStepArguments {
    Element* ptr_h = nullptr;
    //! Never dereferenced unless the kernel was instantiated with a cell state.
    Element* ptr_c = nullptr;

    Element const* ptr_prev_h = nullptr;
    Element const* ptr_weight = nullptr;
    Element const* ptr_gates = nullptr;
    //! Never dereferenced unless the kernel was instantiated with a cell state.
    Element const* ptr_prev_c = nullptr;

    int M = 0;  //!< cohort rows
    int H = 0;  //!< hidden units written
    int K = 0;  //!< hidden width reduced over
    int N = 0;  //!< `gates * H` -- the gate-interleaved column count

    int64_t stride_prev_h = 0;  //!< elements between consecutive rows
    int64_t stride_weight = 0;
    int64_t stride_gates = 0;
    int64_t stride_prev_c = 0;
    int64_t stride_h = 0;
    int64_t stride_c = 0;
};

template <class Element>
using RecurrentStepParams = RecurrentStepArguments<Element>;

// ---------------------------------------------------------------------------
// The state transition
// ---------------------------------------------------------------------------

/*! \brief One hidden unit's transition, in FP32, selected at compile time.
 *
 * `apply` takes the `gates` pre-activations this unit owns -- already summed
 * with the input projection -- plus the incoming cell value, and returns the
 * new hidden and cell values.
 *
 * The intrinsics are `fastSigmoid` and `tanhf`, which is what every other
 * kernel in this family uses (`include/oasr/recurrent/recurrent.cuh`, and the
 * CUTLASS 2.x fused epilogue in `recurrent_cutlass.cuh`) and what the CuTeDSL
 * lane's `fastmath=True` compiles to.  Matching them is not cosmetic: the
 * three paths are compared against each other by
 * `tests/kernels/test_recurrent.py`, and a `expf` here would put this lane a
 * few ulp off a reference the others hit exactly.
 *
 * `RELU` is `fmaxf(x, 0)`, which is also what the rest of the family computes
 * (`recurrent.cuh` line 169 and its siblings) and what the CuTeDSL lane spells
 * `cute.math.max`.  PTX `max.f32` returns the non-NaN operand, so this
 * disagrees with `oasr::relu`'s NaN propagation -- and agreeing with the
 * recurrent family is the right side of that disagreement to be on, because
 * that is what the cross-path tests assert.
 */
template <RecurrentKind Kind>
struct RecurrentTransition;

template <>
struct RecurrentTransition<RecurrentKind::LSTM> {
    static constexpr int kGates = 4;
    static constexpr bool kHasCell = true;

    /*! \param g  the four pre-activations, in PyTorch's i, f, g, o order
     *  \param prev_c  this unit's incoming cell value
     *  \param out_h,out_c  the new hidden and cell values
     */
    static CUTLASS_DEVICE void apply(float const (&g)[4], float prev_c, float& out_h,
                                     float& out_c) {
        float const i_gate = oasr::fastSigmoid(g[0]);
        float const f_gate = oasr::fastSigmoid(g[1]);
        float const cell_gate = tanhf(g[2]);
        float const o_gate = oasr::fastSigmoid(g[3]);
        out_c = f_gate * prev_c + i_gate * cell_gate;
        out_h = o_gate * tanhf(out_c);
    }
};

template <>
struct RecurrentTransition<RecurrentKind::RNN_TANH> {
    static constexpr int kGates = 1;
    static constexpr bool kHasCell = false;

    static CUTLASS_DEVICE void apply(float const (&g)[1], float, float& out_h, float&) {
        out_h = tanhf(g[0]);
    }
};

template <>
struct RecurrentTransition<RecurrentKind::RNN_RELU> {
    static constexpr int kGates = 1;
    static constexpr bool kHasCell = false;

    static CUTLASS_DEVICE void apply(float const (&g)[1], float, float& out_h, float&) {
        out_h = fmaxf(g[0], 0.0f);
    }
};

}  // namespace recurrent
}  // namespace oasr
