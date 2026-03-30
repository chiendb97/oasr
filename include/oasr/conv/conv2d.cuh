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
                                                    using Config = gemm::CutlassGemmConfig<16, 128, 64, 16, 32, 64, 3, 75>;
                                                    return conv::CutlassConv2dFpropKernel<Config, ElementA, ElementB, ElementCD, activation_type>(input_ptr, filter_ptr, bias_ptr, output_ptr, N, H, W, IC, K, R, S, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, stream);
                                                } else if constexpr (SM_VERSION == 80) {
                                                    using Config = gemm::CutlassGemmConfig<16, 128, 64, 16, 32, 64, 3, 80>;
                                                    return conv::CutlassConv2dFpropKernel<Config, ElementA, ElementB, ElementCD, activation_type>(input_ptr, filter_ptr, bias_ptr, output_ptr, N, H, W, IC, K, R, S, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, stream);
                                                } else if constexpr (SM_VERSION == 86) {
                                                    using Config = gemm::CutlassGemmConfig<16, 128, 64, 16, 32, 64, 3, 86>;
                                                    return conv::CutlassConv2dFpropKernel<Config, ElementA, ElementB, ElementCD, activation_type>(input_ptr, filter_ptr, bias_ptr, output_ptr, N, H, W, IC, K, R, S, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, stream);
                                                } else if constexpr (SM_VERSION == 89) {
                                                    using Config = gemm::CutlassGemmConfig<16, 128, 64, 16, 32, 64, 3, 89>;
                                                    return conv::CutlassConv2dFpropKernel<Config, ElementA, ElementB, ElementCD, activation_type>(input_ptr, filter_ptr, bias_ptr, output_ptr, N, H, W, IC, K, R, S, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, stream);
                                                } else if constexpr (SM_VERSION == 90) {
                                                    using Config = gemm::CutlassGemmConfig<16, 128, 64, 16, 32, 64, 3, 90>;
                                                    return conv::CutlassConv2dFpropKernel<Config, ElementA, ElementB, ElementCD, activation_type>(input_ptr, filter_ptr, bias_ptr, output_ptr, N, H, W, IC, K, R, S, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, stream);
                                                } else if constexpr (SM_VERSION == 100) {
                                                    using Config = gemm::CutlassGemmConfig<16, 128, 64, 16, 32, 64, 3, 100>;
                                                } else if constexpr (SM_VERSION == 103) {
                                                    using Config = gemm::CutlassGemmConfig<16, 128, 64, 16, 32, 64, 3, 103>;
                                                } else if constexpr (SM_VERSION == 120) {
                                                    using Config = gemm::CutlassGemmConfig<16, 128, 64, 16, 32, 64, 3, 120>;
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
