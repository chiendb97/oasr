// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Conv2D dispatch interface -- public API for Conv2D and Conv2DActivation.

#pragma once

#include <oasr/common/arch_dispatch.h>
#include <oasr/common/epilogue_functors.h>
#include <oasr/common/types.h>
#include <oasr/conv/cutlass_conv2d_configs.h>
#include <oasr/conv/conv2d_cutlass_template.h>

namespace oasr {
namespace conv {

//==============================================================================
// Internal dispatch helpers
//==============================================================================

namespace detail {

template <typename Config, typename MMATraits, template <int, typename, typename> class EpilogueFn,
          typename ElementType>
static Conv2dStatus dispatchConv2dWithConfig(const void* input_ptr, const void* filter_ptr,
                                              const void* bias_ptr, void* output_ptr, int N, int H,
                                              int W, int IC, int K, int R, int S, int pad_h,
                                              int pad_w, int stride_h, int stride_w,
                                              int dilation_h, int dilation_w,
                                              cudaStream_t stream) {
    return CutlassConv2dFpropKernel<Config, MMATraits, ElementType, ElementType, ElementType,
                                     EpilogueFn>::run(
        reinterpret_cast<const ElementType*>(input_ptr),
        reinterpret_cast<const ElementType*>(filter_ptr),
        reinterpret_cast<const ElementType*>(bias_ptr),
        reinterpret_cast<ElementType*>(output_ptr), N, H, W, IC, K, R, S, pad_h, pad_w, stride_h,
        stride_w, dilation_h, dilation_w, 1.0f, stream);
}

}  // namespace detail

//==============================================================================
// Conv2D -- public API
//==============================================================================

template <typename T>
cudaError_t Conv2D(const T* input, const T* filter, const T* bias, T* output, int N, int H, int W,
                   int IC, int K, int R, int S, int pad_h, int pad_w, int stride_h, int stride_w,
                   int dilation_h, int dilation_w, cudaStream_t stream) {
    Conv2dStatus status = Conv2dStatus::SUCCESS;

#ifdef OASR_TARGET_SM
    {
        constexpr int SM_VERSION = OASR_TARGET_SM;
        if constexpr (SM_VERSION < 80) {
            return cudaErrorNotSupported;
        } else {
#ifdef OASR_CONV2D_TILE_M
            using Config = JitConv2dConfig;
#else
            using Config = typename DefaultConv2dConfig<SM_VERSION>::type;
#endif
            using MMA = SmConv2dTraits<SM_VERSION>;
            status = detail::dispatchConv2dWithConfig<Config, MMA, oasr::EpilogueIdentity, T>(
                input, filter, bias, output, N, H, W, IC, K, R, S, pad_h, pad_w, stride_h,
                stride_w, dilation_h, dilation_w, stream);
        }
    }
#else
    int sm = oasr::getDeviceSmVersion();
    OASR_DISPATCH_SM(sm, SM_VERSION, {
        if constexpr (SM_VERSION < 80) {
            status = Conv2dStatus::NOT_SUPPORTED;
        } else {
            using Config = typename DefaultConv2dConfig<SM_VERSION>::type;
            using MMA = SmConv2dTraits<SM_VERSION>;
            status = detail::dispatchConv2dWithConfig<Config, MMA, oasr::EpilogueIdentity, T>(
                input, filter, bias, output, N, H, W, IC, K, R, S, pad_h, pad_w, stride_h,
                stride_w, dilation_h, dilation_w, stream);
        }
    });
#endif

    if (status != Conv2dStatus::SUCCESS) {
        return cudaErrorNotSupported;
    }
    return cudaGetLastError();
}

//==============================================================================
// Conv2DActivation -- public API
//==============================================================================

template <typename T>
cudaError_t Conv2DActivation(const T* input, const T* filter, const T* bias, T* output,
                             ActivationType activation, int N, int H, int W, int IC, int K, int R,
                             int S, int pad_h, int pad_w, int stride_h, int stride_w,
                             int dilation_h, int dilation_w, cudaStream_t stream) {
    Conv2dStatus status = Conv2dStatus::SUCCESS;

#ifdef OASR_TARGET_SM
    {
        constexpr int SM_VERSION = OASR_TARGET_SM;
        if constexpr (SM_VERSION < 80) {
            return cudaErrorNotSupported;
        } else {
#ifdef OASR_CONV2D_TILE_M
            using Config = JitConv2dConfig;
#else
            using Config = typename DefaultConv2dConfig<SM_VERSION>::type;
#endif
            using MMA = SmConv2dTraits<SM_VERSION>;
            if (activation == ActivationType::RELU) {
                status = detail::dispatchConv2dWithConfig<Config, MMA, oasr::EpilogueRelu, T>(
                    input, filter, bias, output, N, H, W, IC, K, R, S, pad_h, pad_w, stride_h,
                    stride_w, dilation_h, dilation_w, stream);
            } else if (activation == ActivationType::GELU) {
                status = detail::dispatchConv2dWithConfig<Config, MMA, oasr::EpilogueGelu, T>(
                    input, filter, bias, output, N, H, W, IC, K, R, S, pad_h, pad_w, stride_h,
                    stride_w, dilation_h, dilation_w, stream);
            } else if (activation == ActivationType::SWISH) {
                status = detail::dispatchConv2dWithConfig<Config, MMA, oasr::EpilogueSwish, T>(
                    input, filter, bias, output, N, H, W, IC, K, R, S, pad_h, pad_w, stride_h,
                    stride_w, dilation_h, dilation_w, stream);
            } else {
                status = Conv2dStatus::INVALID_ARGUMENT;
            }
        }
    }
#else
    int sm = oasr::getDeviceSmVersion();
    OASR_DISPATCH_SM(sm, SM_VERSION, {
        if constexpr (SM_VERSION < 80) {
            status = Conv2dStatus::NOT_SUPPORTED;
        } else {
            using Config = typename DefaultConv2dConfig<SM_VERSION>::type;
            using MMA = SmConv2dTraits<SM_VERSION>;
            if (activation == ActivationType::RELU) {
                status = detail::dispatchConv2dWithConfig<Config, MMA, oasr::EpilogueRelu, T>(
                    input, filter, bias, output, N, H, W, IC, K, R, S, pad_h, pad_w, stride_h,
                    stride_w, dilation_h, dilation_w, stream);
            } else if (activation == ActivationType::GELU) {
                status = detail::dispatchConv2dWithConfig<Config, MMA, oasr::EpilogueGelu, T>(
                    input, filter, bias, output, N, H, W, IC, K, R, S, pad_h, pad_w, stride_h,
                    stride_w, dilation_h, dilation_w, stream);
            } else if (activation == ActivationType::SWISH) {
                status = detail::dispatchConv2dWithConfig<Config, MMA, oasr::EpilogueSwish, T>(
                    input, filter, bias, output, N, H, W, IC, K, R, S, pad_h, pad_w, stride_h,
                    stride_w, dilation_h, dilation_w, stream);
            } else {
                status = Conv2dStatus::INVALID_ARGUMENT;
            }
        }
    });
#endif

    if (status != Conv2dStatus::SUCCESS) {
        return cudaErrorNotSupported;
    }
    return cudaGetLastError();
}

}  // namespace conv
}  // namespace oasr
