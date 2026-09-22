// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// The gated MLP's arguments, and the compile-time activation dispatch.

#pragma once

#include <cutlass/cutlass.h>

#include <oasr/common/math.h>
#include <oasr/common/types.h>

#include <cstdint>

namespace oasr {
namespace mlp {

/*! \brief Everything one launch needs.
 *
 * Plain extents and row strides rather than CuTe shape tuples: every tensor
 * here is rank 2 with a contiguous trailing axis, so a `(extent, stride)` pair
 * per operand says all there is to say and keeps the struct readable from the
 * host binding.  The *row* strides are runtime values, which is what lets a
 * caller hand over a row-slice of a wider buffer without a copy -- the CuTeDSL
 * lane has to demand a fully contiguous tensor.
 *
 * `x` is `(M, K)`, both weights are `(N, K)` -- `nn.Linear`'s own `(out, in)`
 * layout, so a checkpoint is read where it lies -- and the output is `(M, N)`.
 */
template <class Element>
struct GatedMlpArguments {
    Element* ptr_o = nullptr;
    Element const* ptr_x = nullptr;
    Element const* ptr_wg = nullptr;
    Element const* ptr_wu = nullptr;
    //! Never dereferenced unless the kernel was instantiated with `Has_bias`.
    Element const* ptr_bg = nullptr;
    Element const* ptr_bu = nullptr;

    int M = 0;
    int N = 0;
    int K = 0;

    int64_t stride_x = 0;   //!< elements between consecutive rows of `x`
    int64_t stride_wg = 0;  //!< ...of `w_gate`
    int64_t stride_wu = 0;  //!< ...of `w_up`
    int64_t stride_o = 0;   //!< ...of `out`
};

template <class Element>
using GatedMlpParams = GatedMlpArguments<Element>;

// ---------------------------------------------------------------------------
// Activation
// ---------------------------------------------------------------------------

/*! \brief The gate activation, in FP32, selected at compile time.
 *
 * Keyed by `oasr::ActivationType` so the fused epilogue and the two-GEMM path
 * it replaces (`oasr.gemm_activation`, whose CUTLASS epilogues wrap the same
 * `oasr/common/math.h` functions) cannot drift apart -- that agreement is what
 * `tests/kernels/test_gated_mlp.py::test_fused_matches_two_gemm_path` asserts.
 *
 * `GELU` is the tanh approximation and `GELU_ERF` the exact-erf form.  They
 * stay separate for the same reason they do in the layer waist: they are
 * numerically different epilogues and a checkpoint means one of them.
 *
 * \note `RELU` here is `oasr::relu`, which **propagates NaN** -- PyTorch's
 *   semantics, and what `oasr.gemm_activation` computes.  The CuTeDSL lane
 *   spells it `max(x, 0)`, and PTX `max.f32` returns the non-NaN operand, so
 *   the two lanes disagree on a NaN input and only there.  Matching the C++
 *   two-GEMM path is the right side of that disagreement to be on: it is what
 *   the fused/unfused layer comparison asserts.
 */
template <int Act>
struct GatedMlpActivation;

template <>
struct GatedMlpActivation<int(ActivationType::IDENTITY)> {
    static CUTLASS_DEVICE float apply(float x) { return x; }
};

template <>
struct GatedMlpActivation<int(ActivationType::RELU)> {
    static CUTLASS_DEVICE float apply(float x) { return oasr::relu(x); }
};

template <>
struct GatedMlpActivation<int(ActivationType::SWISH)> {
    static CUTLASS_DEVICE float apply(float x) { return oasr::swish(x); }
};

template <>
struct GatedMlpActivation<int(ActivationType::GELU)> {
    static CUTLASS_DEVICE float apply(float x) { return oasr::gelu(x); }
};

template <>
struct GatedMlpActivation<int(ActivationType::GELU_ERF)> {
    static CUTLASS_DEVICE float apply(float x) { return oasr::gelu_erf(x); }
};

template <>
struct GatedMlpActivation<int(ActivationType::TANH)> {
    static CUTLASS_DEVICE float apply(float x) { return tanhf(x); }
};

}  // namespace mlp
}  // namespace oasr
