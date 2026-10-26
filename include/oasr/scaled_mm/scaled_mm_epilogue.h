// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Epilogue visitor trees (CUTLASS 2.x Sm80 EVT) for the 8-bit scaled GEMM
// (int8 and fp8 e4m3 operands alike: the trees see only the fp32 accumulator).
//
// ``ScaledMmEpilogue`` -- the fp16/bf16 output of a quantized layer:
//
//     t   = b_scale[n] * f32(acc[m, n])          per output channel (row broadcast)
//     y   = fma(a_scale[m], t, f32(bias[n]))     per token, or one per-tensor scale
//     out = round_to_out(act(y))                 one rounding
//
// ``ScaledMmQuantEpilogue`` -- the same ``y`` and activation, then quantized
// straight to 8 bits for a consumer GEMM with a static per-tensor input scale, so
// the layer's fp output and the consumer's quantize launch both disappear:
//
//     y_T = round_to_T(act(y))                   exactly the value the fp path stores
//     q   = int8: clamp(rint(f32(y_T) * out_mul[n]), -127, 127)
//           fp8:  e4m3_rne_satfinite(f32(y_T) * out_mul[n])
//
// ``out_mul[n] = smooth_inv[n] * (1 / input_scale)`` is the consumer's
// per-channel quantize multiplier (``oasr.functionals.quant.static_multiplier``).
// Rounding to T first is what makes the fused output bit-identical to the
// unfused ``scaled_mm`` -> ``quantize_static`` pair (``quantize.cuh``'s
// ``QuantTraits``: the same IEEE multiply, the same rounding).
//
// The order of the shared part follows vLLM's ``ScaledEpilogueBias``.  Bias
// absent is a null pointer, which the row-broadcast visitor reads as its zero
// default.

#pragma once

#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>
#include <cutlass/functional.h>

#include <oasr/common/types.h>
#include <oasr/scaled_mm/scaled_mm_functors.h>
#include <oasr/scaled_mm/scaled_mm_visitors.h>

namespace oasr {
namespace scaled_mm {

// The dequantized pre-activation shared by both trees: fma(a_scale, b_scale*acc, bias).
template <typename OutputTileThreadMap, typename ElementBias>
struct ScaledMmDequant {
    using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;

    // Activation scale: one float per output row (per-token), or one for all.
    using ScaleA = cutlass::epilogue::threadblock::OasrVisitorColOrScalarBroadcast<
        OutputTileThreadMap, float, cute::Stride<cute::_1, cute::_0, cute::_0>>;
    // Per-channel weight scale and the bias: one value per output column.
    using ScaleB = cutlass::epilogue::threadblock::VisitorRowBroadcast<
        OutputTileThreadMap, float, cute::Stride<cute::_0, cute::_1, cute::_0>>;
    using Bias = cutlass::epilogue::threadblock::VisitorRowBroadcast<
        OutputTileThreadMap, ElementBias, cute::Stride<cute::_0, cute::_1, cute::_0>>;

    using Compute0 =
        cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, float, float,
                                                       cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT0 = cutlass::epilogue::threadblock::Sm80EVT<Compute0, ScaleB, Accum>;

    using Compute1 =
        cutlass::epilogue::threadblock::VisitorCompute<cutlass::homogeneous_multiply_add, float,
                                                       float,
                                                       cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT1 = cutlass::epilogue::threadblock::Sm80EVT<Compute1, ScaleA, EVT0, Bias>;

    static typename EVT1::Arguments make_args(float const* a_scale, bool a_per_token,
                                              float const* b_scale, ElementBias const* bias) {
        typename ScaleA::Arguments a_args{a_scale, a_per_token, {}};
        typename ScaleB::Arguments b_args{b_scale};
        typename Bias::Arguments bias_args{bias};
        typename EVT0::Arguments evt0_args{b_args, {}, {}};
        return typename EVT1::Arguments{a_args, evt0_args, bias_args, {}};
    }
};

template <typename ElementD, typename OutputTileThreadMap, ActivationType act>
struct ScaledMmEpilogue {
    using Dequant = ScaledMmDequant<OutputTileThreadMap, ElementD>;

    using Compute2 = cutlass::epilogue::threadblock::VisitorCompute<
        ScaledMmActivation<act>::template Fn, ElementD, float,
        cutlass::FloatRoundStyle::round_to_nearest>;
    using EVTCompute = cutlass::epilogue::threadblock::Sm80EVT<Compute2, typename Dequant::EVT1>;

    using D = cutlass::epilogue::threadblock::VisitorAuxStore<
        OutputTileThreadMap, ElementD, cutlass::FloatRoundStyle::round_to_nearest,
        cute::Stride<int64_t, cute::_1, cute::_0>>;
    using EVTD = cutlass::epilogue::threadblock::Sm80EVT<D, EVTCompute>;

    static typename EVTD::Arguments make_args(float const* a_scale, bool a_per_token,
                                              float const* b_scale, ElementD const* bias,
                                              ElementD* out, int64_t ldd) {
        typename EVTCompute::Arguments compute_args{
            Dequant::make_args(a_scale, a_per_token, b_scale, bias), {}};
        typename D::Arguments d_args{out, cute::Stride<int64_t, cute::_1, cute::_0>{ldd, {}, {}}};
        return typename EVTD::Arguments{compute_args, d_args};
    }
};

template <typename ElementT, typename OutputTileThreadMap, ActivationType act,
          typename ElementQ = int8_t>
struct ScaledMmQuantEpilogue {
    using Dequant = ScaledMmDequant<OutputTileThreadMap, ElementT>;

    // act(y), rounded to T: the value the fp epilogue would have stored.
    using Compute2 = cutlass::epilogue::threadblock::VisitorCompute<
        ScaledMmActivation<act>::template Fn, ElementT, float,
        cutlass::FloatRoundStyle::round_to_nearest>;
    using EVTAct = cutlass::epilogue::threadblock::Sm80EVT<Compute2, typename Dequant::EVT1>;

    // The consumer's per-channel quantize multiplier.
    using OutMul = cutlass::epilogue::threadblock::VisitorRowBroadcast<
        OutputTileThreadMap, float, cute::Stride<cute::_0, cute::_1, cute::_0>>;
    using Compute3 = cutlass::epilogue::threadblock::VisitorCompute<
        ScaledMmQuantize<ElementQ>::template Fn, ElementQ, float,
        cutlass::FloatRoundStyle::round_to_nearest>;
    using EVTQuant = cutlass::epilogue::threadblock::Sm80EVT<Compute3, EVTAct, OutMul>;

    using D = cutlass::epilogue::threadblock::VisitorAuxStore<
        OutputTileThreadMap, ElementQ, cutlass::FloatRoundStyle::round_to_nearest,
        cute::Stride<int64_t, cute::_1, cute::_0>>;
    using EVTD = cutlass::epilogue::threadblock::Sm80EVT<D, EVTQuant>;

    static typename EVTD::Arguments make_args(float const* a_scale, bool a_per_token,
                                              float const* b_scale, ElementT const* bias,
                                              float const* out_mul, ElementQ* out, int64_t ldq) {
        typename EVTAct::Arguments act_args{
            Dequant::make_args(a_scale, a_per_token, b_scale, bias), {}};
        typename OutMul::Arguments mul_args{out_mul};
        typename EVTQuant::Arguments quant_args{act_args, mul_args, {}};
        typename D::Arguments d_args{out, cute::Stride<int64_t, cute::_1, cute::_0>{ldq, {}, {}}};
        return typename EVTD::Arguments{quant_args, d_args};
    }
};

}  // namespace scaled_mm
}  // namespace oasr
