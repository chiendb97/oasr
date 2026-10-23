// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Dispatch for the int8 convolution: runtime activation id -> the epilogue
// instantiation.  The tile config and the output dtype are fixed by the caller
// (one rendered translation unit per tile config).

#pragma once

#include <oasr/scaled_conv/scaled_conv_cutlass_template.h>

namespace oasr {
namespace scaled_conv {

#define OASR_SCALED_CONV_ACTIVATIONS(CASE) \
    CASE(IDENTITY)                         \
    CASE(RELU)                             \
    CASE(SWISH)                            \
    CASE(GELU)                             \
    CASE(GELU_ERF)

template <typename Config, typename ElementD>
ScaledConvStatus dispatchScaledConv2d(int activation, typename Config::ElementAB const* x,
                                      typename Config::ElementAB const* w, float const* scale,
                                      ElementD const* bias, ElementD* out, ConvGeometry const& g,
                                      cudaStream_t stream) {
#define OASR_SCALED_CONV_CASE(ACT)                                                          \
    case static_cast<int>(ActivationType::ACT):                                            \
        return CutlassScaledConv2d<Config, ElementD, ActivationType::ACT>::run(x, w, scale, \
                                                                               bias, out, g, \
                                                                               stream);
    switch (activation) {
        OASR_SCALED_CONV_ACTIVATIONS(OASR_SCALED_CONV_CASE)
        default:
            return ScaledConvStatus::CANNOT_IMPLEMENT;
    }
#undef OASR_SCALED_CONV_CASE
}

#undef OASR_SCALED_CONV_ACTIVATIONS

}  // namespace scaled_conv
}  // namespace oasr
