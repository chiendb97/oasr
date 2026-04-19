// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Conv2D dispatch interface -- public API for Conv2D and Conv2DActivation.

#pragma once

#include <oasr/common/arch_dispatch.h>
#include <oasr/common/epilogue_functors.h>
#include <oasr/common/types.h>
#include <oasr/gemm/cutlass_gemm_configs.h>
#include <oasr/conv/conv2d_cutlass_template.h>
// SM120 falls back to the CUTLASS 2.x Conv2D path (Sm80 tensor ops are
// forward-compatible).  Only include the SM90+ Conv2D template when either
// in AOT mode or JIT-targeting SM90 / SM100.
#if !defined(OASR_TARGET_SM) || OASR_TARGET_SM == 90 || OASR_TARGET_SM == 100
#include <oasr/conv/conv2d_cutlass_template_sm90.h>
#endif

namespace oasr {
namespace conv {

//==============================================================================
// Internal dispatch helpers
//==============================================================================

namespace detail {

template <int SM_VERSION, typename ElementA, typename ElementB, typename ElementCD,
          oasr::ActivationType activation_type,
          typename ElementType>
static gemm::GemmStatus dispatchConv2dWithSmVersion(const void* input_ptr, const void* filter_ptr,
                                              const void* bias_ptr, void* output_ptr, int N, int H,
                                              int W, int IC, int K, int R, int S, int pad_h,
                                              int pad_w, int stride_h, int stride_w,
                                              int dilation_h, int dilation_w,
                                              cudaStream_t stream) {
                                                if constexpr (SM_VERSION == 75) {
                                                    using Config = conv::CutlassConv2dConfig<16, 128, 64, 16, 32, 64, 3, 75>;
                                                    return conv::CutlassConv2dFpropKernel<Config, ElementA, ElementB, ElementCD, activation_type>::run(input_ptr, filter_ptr, bias_ptr, output_ptr, N, H, W, IC, K, R, S, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,  1.0f, stream);
                                                } else if constexpr (SM_VERSION == 80) {
                                                    using Config = conv::CutlassConv2dConfig<16, 128, 64, 16, 32, 64, 3, 80>;
                                                    return conv::CutlassConv2dFpropKernel<Config, ElementA, ElementB, ElementCD, activation_type>::run(input_ptr, filter_ptr, bias_ptr, output_ptr, N, H, W, IC, K, R, S, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, 1.0f, stream);
                                                } else if constexpr (SM_VERSION == 86) {
                                                    using Config = conv::CutlassConv2dConfig<16, 128, 64, 16, 32, 64, 3, 86>;
                                                    return conv::CutlassConv2dFpropKernel<Config, ElementA, ElementB, ElementCD, activation_type>::run(input_ptr, filter_ptr, bias_ptr, output_ptr, N, H, W, IC, K, R, S, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, 1.0f, stream);
                                                } else if constexpr (SM_VERSION == 89) {
                                                    using Config = conv::CutlassConv2dConfig<16, 128, 64, 16, 32, 64, 3, 89>;
                                                    return conv::CutlassConv2dFpropKernel<Config, ElementA, ElementB, ElementCD, activation_type>::run(input_ptr, filter_ptr, bias_ptr, output_ptr, N, H, W, IC, K, R, S, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, 1.0f, stream);
#if !defined(OASR_TARGET_SM) || OASR_TARGET_SM == 90 || OASR_TARGET_SM == 100
                                                } else if constexpr (SM_VERSION == 90) {
                                                    using Config = conv::CutlassConv2dConfigSm90<64, 16, 128, 1, 1, 1, 1, 3, 90>;
                                                    return conv::CutlassConv2dFpropKernelSm90<Config, ElementA, ElementB, ElementCD, activation_type>::run(input_ptr, filter_ptr, bias_ptr, output_ptr, N, H, W, IC, K, R, S, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, 1.0f, stream);
                                                } else if constexpr (SM_VERSION == 100) {
                                                    using Config = conv::CutlassConv2dConfigSm90<64, 16, 128, 1, 1, 1, 1, 3, 100>;
                                                    return conv::CutlassConv2dFpropKernelSm90<Config, ElementA, ElementB, ElementCD, activation_type>::run(input_ptr, filter_ptr, bias_ptr, output_ptr, N, H, W, IC, K, R, S, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, 1.0f, stream);
#endif
                                                } else if constexpr (SM_VERSION == 120) {
                                                    // SM120 TMA warp-specialised builder supports only F8/F6/F4 MMA; fall
                                                    // back to the CUTLASS 2.x path (Sm80 tensor op is forward-compatible).
                                                    using Config = conv::CutlassConv2dConfig<16, 128, 64, 16, 32, 64, 3, 120>;
                                                    return conv::CutlassConv2dFpropKernel<Config, ElementA, ElementB, ElementCD, activation_type>::run(input_ptr, filter_ptr, bias_ptr, output_ptr, N, H, W, IC, K, R, S, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, 1.0f, stream);
                                                } else {
                                                    return gemm::GemmStatus::NOT_SUPPORTED;
                                                }
}

}  // namespace detail

//==============================================================================
// Conv2D -- public API
//==============================================================================

template <typename T>
cudaError_t Conv2D(const T* input, const T* filter, const T* bias, T* output, int N, int H, int W,
                   int IC, int K, int R, int S, int pad_h, int pad_w, int stride_h, int stride_w,
                   int dilation_h, int dilation_w, cudaStream_t stream) {
    gemm::GemmStatus status = gemm::GemmStatus::SUCCESS;

#ifdef OASR_TARGET_SM
    {
        constexpr int SM_VERSION = OASR_TARGET_SM;
        status = detail::dispatchConv2dWithSmVersion<SM_VERSION, T, T, T, oasr::ActivationType::IDENTITY>(input, filter, bias, output, N, H, W, IC, K, R, S, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, stream);
        
    }
#else
    int sm = oasr::getDeviceSmVersion();
    OASR_DISPATCH_SM(sm, SM_VERSION, {
        status = detail::dispatchConv2dWithSmVersion<SM_VERSION, T, T, T, oasr::ActivationType::IDENTITY>(input, filter, bias, output, N, H, W, IC, K, R, S, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, stream);
    });
#endif
}

//==============================================================================
// Conv2DActivation -- public API
//==============================================================================

template <typename T>
cudaError_t Conv2DActivation(const T* input, const T* filter, const T* bias, T* output,
                             ActivationType activation, int N, int H, int W, int IC, int K, int R,
                             int S, int pad_h, int pad_w, int stride_h, int stride_w,
                             int dilation_h, int dilation_w, cudaStream_t stream) {
    gemm::GemmStatus status = gemm::GemmStatus::SUCCESS;

#ifdef OASR_TARGET_SM
    {
        constexpr int SM_VERSION = OASR_TARGET_SM;
        status = detail::dispatchConv2dWithSmVersion<SM_VERSION, T, T, T, activation>(input, filter, bias, output, N, H, W, IC, K, R, S, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, stream);
    }
#else
    int sm = oasr::getDeviceSmVersion();
    OASR_DISPATCH_SM(sm, SM_VERSION, {
        status = detail::dispatchConv2dWithSmVersion<SM_VERSION, T, T, T, activation>(input, filter, bias, output, N, H, W, IC, K, R, S, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, stream);
    });
#endif

    if (status != gemm::GemmStatus::SUCCESS) {
        return cudaErrorNotSupported;
    }
    return cudaGetLastError();
}

}  // namespace conv
}  // namespace oasr
