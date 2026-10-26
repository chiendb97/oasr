// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Epilogue visitor trees (CUTLASS 3.x Sm90 EVT) for the native scaled GEMM on
// sm90 and sm100 -- node for node the trees of the 2.x lane
// (``scaled_mm_epilogue.h``), over the same element functors
// (``scaled_mm_functors.h``):
//
//     t   = b_scale[n] * f32(acc[m, n])          per output channel (row broadcast)
//     y   = fma(a_scale[m], t, f32(bias[n]))     per token, or one per-tensor scale
//     out = round_to_out(act(y))                 one rounding
//
// and, for a GEMM-chain producer, the 8-bit output of a static-scale consumer:
//
//     y_T = round_to_T(act(y))                   exactly the value the fp path stores
//     q   = int8: clamp(rint(f32(y_T) * out_mul[n]), -127, 127)
//           fp8:  e4m3_rne_satfinite(f32(y_T) * out_mul[n])
//
// Two differences from the 2.x trees, neither in the arithmetic:
//
// * The activation-scale leaf is CUTLASS's own ``Sm90ColBroadcast`` with a
//   *dynamic* broadcast (``Stride<bool, _0, _0>``): ``true`` reads one scale per
//   row (per-token), ``false`` one scale for every row (a static per-tensor
//   scale) -- the runtime switch the 2.x lane needs a custom visitor for.  The
//   scale is read from device memory either way, so a captured CUDA graph sees a
//   value written after capture.
// * The activation is a runtime argument (``ScaledMmRuntimeActivation``), not a
//   template parameter, to keep the slow 3.x build to one kernel per tile and
//   output dtype.
//
// The 3.x epilogue stores D itself (TMA), so there is no store node: the tree's
// root converts to the output element.  Bias absent is a null pointer, which the
// row-broadcast leaf reads as its zero default.

#pragma once

#include <cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp>
#include <cutlass/epilogue/fusion/sm90_visitor_compute_tma_warpspecialized.hpp>
#include <cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp>
#include <cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp>
#include <cutlass/functional.h>

#include <cute/tensor.hpp>

#include <oasr/scaled_mm/scaled_mm_functors.h>

namespace oasr {
namespace scaled_mm {
namespace native {

using cutlass::FloatRoundStyle;
using cutlass::epilogue::fusion::Sm90AccFetch;
using cutlass::epilogue::fusion::Sm90ColBroadcast;
using cutlass::epilogue::fusion::Sm90Compute;
using cutlass::epilogue::fusion::Sm90EVT;
using cutlass::epilogue::fusion::Sm90RowBroadcast;

// One value per output row (``true``) or one for all rows (``false``).
using StrideScaleA = cute::Stride<bool, cute::_0, cute::_0>;
// One value per output column.
using StrideRow = cute::Stride<cute::_0, cute::_1, cute::_0>;

// The dequantized pre-activation shared by both trees: fma(a_scale, b_scale*acc, bias).
// ``CtaTileShape`` is the tile one CTA stores (half the MMA tile's M on a 2-SM
// sm100 schedule): the broadcast leaves stage their row of it in shared memory.
template <class CtaTileShape, class ElementBias>
struct NativeDequant {
    using Accum = Sm90AccFetch;
    using ScaleA = Sm90ColBroadcast<0, CtaTileShape, float, float, StrideScaleA>;
    using ScaleB = Sm90RowBroadcast<0, CtaTileShape, float, float, StrideRow>;
    using Bias = Sm90RowBroadcast<0, CtaTileShape, ElementBias, float, StrideRow>;

    using Compute0 =
        Sm90Compute<cutlass::multiplies, float, float, FloatRoundStyle::round_to_nearest>;
    using EVT0 = Sm90EVT<Compute0, ScaleB, Accum>;

    using Compute1 = Sm90Compute<cutlass::homogeneous_multiply_add, float, float,
                                 FloatRoundStyle::round_to_nearest>;
    using EVT1 = Sm90EVT<Compute1, ScaleA, EVT0, Bias>;

    static typename EVT1::Arguments make_args(float const* a_scale, bool a_per_token,
                                              float const* b_scale, ElementBias const* bias) {
        typename ScaleA::Arguments a_args{a_scale, 0.0f, StrideScaleA{a_per_token, {}, {}}};
        typename ScaleB::Arguments b_args{b_scale, 0.0f, StrideRow{}};
        typename Bias::Arguments bias_args{bias, ElementBias(0), StrideRow{}};
        typename EVT0::Arguments evt0_args{b_args, {}, {}};
        return typename EVT1::Arguments{a_args, evt0_args, bias_args, {}};
    }
};

// The fp16/bf16 output of a quantized layer.
template <class CtaTileShape, class ElementD>
struct NativeScaledMmEpilogue {
    using Dequant = NativeDequant<CtaTileShape, ElementD>;
    using Compute2 = Sm90Compute<ScaledMmRuntimeActivation, ElementD, float,
                                 FloatRoundStyle::round_to_nearest>;
    using EVT = Sm90EVT<Compute2, typename Dequant::EVT1>;

    static typename EVT::Arguments make_args(int activation, float const* a_scale,
                                             bool a_per_token, float const* b_scale,
                                             ElementD const* bias) {
        return typename EVT::Arguments{Dequant::make_args(a_scale, a_per_token, b_scale, bias),
                                       ScaledMmActivationArgs{activation}};
    }
};

// The same ``y`` and activation, rounded to ``ElementT`` and quantized to the
// operands' 8-bit format for a consumer GEMM with a static per-tensor input scale.
template <class CtaTileShape, class ElementT, class ElementQ>
struct NativeScaledMmQuantEpilogue {
    using Dequant = NativeDequant<CtaTileShape, ElementT>;

    // act(y), rounded to T: the value the fp epilogue would have stored.
    using Compute2 = Sm90Compute<ScaledMmRuntimeActivation, ElementT, float,
                                 FloatRoundStyle::round_to_nearest>;
    using EVTAct = Sm90EVT<Compute2, typename Dequant::EVT1>;

    // The consumer's per-channel quantize multiplier.
    using OutMul = Sm90RowBroadcast<0, CtaTileShape, float, float, StrideRow>;
    using Compute3 = Sm90Compute<ScaledMmQuantize<ElementQ>::template Fn, ElementQ, float,
                                 FloatRoundStyle::round_to_nearest>;
    using EVT = Sm90EVT<Compute3, EVTAct, OutMul>;

    static typename EVT::Arguments make_args(int activation, float const* a_scale,
                                             bool a_per_token, float const* b_scale,
                                             ElementT const* bias, float const* out_mul) {
        typename EVTAct::Arguments act_args{
            Dequant::make_args(a_scale, a_per_token, b_scale, bias),
            ScaledMmActivationArgs{activation}};
        typename OutMul::Arguments mul_args{out_mul, 0.0f, StrideRow{}};
        return typename EVT::Arguments{act_args, mul_args, {}};
    }
};

}  // namespace native
}  // namespace scaled_mm
}  // namespace oasr
