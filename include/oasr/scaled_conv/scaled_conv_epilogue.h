// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Output functor of the int8 implicit-GEMM convolution (``scaled_conv``):
//
//     y   = fma(scale[k], f32(acc[m, k]), f32(bias[k]))     (or scale[k] * acc without a bias)
//     out = round_to_out(act(y))                             one rounding
//
// ``scale[k] = input_scale * weight_scale[k]`` is folded once, when the layer is
// bound: a convolution's activation scale is one per-tensor constant (a per-token
// scale does not factor out of a receptive field that spans several tokens), so
// the two scales of the GEMM epilogue collapse into one per-channel vector.
//
// CUTLASS 2.x has no per-channel-alpha convolution epilogue, so this is the
// output op of ``DefaultConv2dFpropWithBroadcast`` (``EpilogueWithBroadcast``):
// the broadcast vector ``V`` is the fp32 per-channel scale and the source ``C``
// is the bias, read through a zero-stride tensor so every output pixel sees
// ``bias[k]`` -- the same broadcast the fp16 convolution already uses.

#pragma once

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include <oasr/common/types.h>
#include <oasr/scaled_mm/scaled_mm_epilogue.h>

namespace oasr {
namespace scaled_conv {

template <typename ElementOut, typename ElementAcc, int ElementsPerAccess, ActivationType act>
class ScaledConvOutputOp {
public:
    using ElementOutput = ElementOut;
    using ElementD = ElementOut;
    using ElementC = ElementOut;
    using ElementAccumulator = ElementAcc;
    using ElementCompute = float;
    using ElementScalar = float;
    using ElementZ = ElementOut;
    using ElementT = ElementOut;
    using ElementVector = float;

    static int const kElementsPerAccess = ElementsPerAccess;
    static int const kCount = ElementsPerAccess;
    static bool const kIsSingleSource = true;
    static bool const kStoreZ = true;
    static bool const kStoreT = false;
    static bool const kIsHeavy = act != ActivationType::IDENTITY && act != ActivationType::RELU;
    static cutlass::FloatRoundStyle const kRound = cutlass::FloatRoundStyle::round_to_nearest;

    using FragmentAccumulator = cutlass::Array<ElementAccumulator, kElementsPerAccess>;
    using FragmentCompute = cutlass::Array<ElementCompute, kElementsPerAccess>;
    using FragmentC = cutlass::Array<ElementC, kElementsPerAccess>;
    using FragmentZ = cutlass::Array<ElementZ, kElementsPerAccess>;
    using FragmentT = cutlass::Array<ElementT, kElementsPerAccess>;
    using FragmentSource = FragmentC;
    using FragmentOutput = FragmentZ;

    struct Params {
        bool has_bias = false;

        CUTLASS_HOST_DEVICE
        Params() {}

        CUTLASS_HOST_DEVICE
        explicit Params(bool has_bias_) : has_bias(has_bias_) {}
    };

    CUTLASS_HOST_DEVICE
    explicit ScaledConvOutputOp(Params const& params) : has_bias_(params.has_bias) {}

    /// The bias is the source tensor: read only when there is one.
    CUTLASS_HOST_DEVICE
    bool is_source_needed() const { return has_bias_; }

    /// No split-K: every output tile is finished in one K pass.
    CUTLASS_HOST_DEVICE
    void set_k_partition(int, int) {}

    /// With a bias: ``act(fma(scale, acc, bias))``.
    CUTLASS_HOST_DEVICE
    void operator()(FragmentZ& frag_Z, FragmentT&, FragmentAccumulator const& AB,
                    FragmentC const& frag_C, FragmentCompute const& V) const {
        cutlass::NumericArrayConverter<float, ElementAccumulator, kElementsPerAccess> acc_cvt;
        cutlass::NumericArrayConverter<float, ElementC, kElementsPerAccess> bias_cvt;
        FragmentCompute acc = acc_cvt(AB);
        FragmentCompute bias = bias_cvt(frag_C);
        FragmentCompute y;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kElementsPerAccess; ++i) {
            y[i] = fmaf(V[i], acc[i], bias[i]);
        }
        frag_Z = finish(y);
    }

    /// Without a bias: ``act(scale * acc)``.
    CUTLASS_HOST_DEVICE
    void operator()(FragmentZ& frag_Z, FragmentT&, FragmentAccumulator const& AB,
                    FragmentCompute const& V) const {
        cutlass::NumericArrayConverter<float, ElementAccumulator, kElementsPerAccess> acc_cvt;
        FragmentCompute acc = acc_cvt(AB);
        FragmentCompute y;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kElementsPerAccess; ++i) {
            y[i] = __fmul_rn(V[i], acc[i]);
        }
        frag_Z = finish(y);
    }

private:
    bool has_bias_;

    CUTLASS_HOST_DEVICE
    FragmentZ finish(FragmentCompute const& y) const {
        typename scaled_mm::ScaledMmActivation<act>::template Fn<FragmentCompute> activation;
        cutlass::NumericArrayConverter<ElementZ, float, kElementsPerAccess, kRound> out_cvt;
        return out_cvt(activation(y));
    }
};

}  // namespace scaled_conv
}  // namespace oasr
