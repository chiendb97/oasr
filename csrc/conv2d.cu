// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// TVM-FFI launcher layer for conv2d kernels.

#ifdef OASR_GROUPED_CONV2D_ONLY
    #include <oasr/common/types.h>
    #include <oasr/conv/grouped_conv2d.cuh>
#else
    #include <oasr/conv/conv2d.cuh>
#endif

#include "tvm_ffi_utils.h"

using namespace oasr;

// =============================================================================
// Helper: map CUDA types to CUTLASS types for Conv2D
// =============================================================================

namespace {

#ifndef OASR_GROUPED_CONV2D_ONLY
template <typename T>
struct ToCutlassType {
    using type = T;
};

template <>
struct ToCutlassType<half> {
    using type = cutlass::half_t;
};

template <>
struct ToCutlassType<__nv_bfloat16> {
    using type = cutlass::bfloat16_t;
};
#endif

}  // namespace

// =============================================================================
// Grouped / depthwise Conv2D launch helpers
// =============================================================================

namespace {

template <typename T, typename Activation>
cudaError_t launch_grouped_conv2d(const T* input, const T* filter, const T* bias, T* output, int N,
                                  int H, int W, int IC, int K, int R, int S, int pad_h, int pad_w,
                                  int stride_h, int stride_w, int dilation_h, int dilation_w,
                                  int groups, cudaStream_t stream) {
    return conv::GroupedConv2D<T, Activation>(input, filter, bias, output, N, H, W, IC, K, R, S,
                                              pad_h, pad_w, stride_h, stride_w, dilation_h,
                                              dilation_w, groups, stream);
}

template <typename T>
cudaError_t dispatch_grouped_conv2d_activation(const T* input, const T* filter, const T* bias,
                                               T* output, ActivationType activation, int N, int H,
                                               int W, int IC, int K, int R, int S, int pad_h,
                                               int pad_w, int stride_h, int stride_w,
                                               int dilation_h, int dilation_w, int groups,
                                               cudaStream_t stream) {
    switch (activation) {
        case ActivationType::RELU:
            return launch_grouped_conv2d<T, ReluActivation>(
                input, filter, bias, output, N, H, W, IC, K, R, S, pad_h, pad_w, stride_h, stride_w,
                dilation_h, dilation_w, groups, stream);
        case ActivationType::GELU:
            return launch_grouped_conv2d<T, GeluActivation>(
                input, filter, bias, output, N, H, W, IC, K, R, S, pad_h, pad_w, stride_h, stride_w,
                dilation_h, dilation_w, groups, stream);
        case ActivationType::SWISH:
            return launch_grouped_conv2d<T, SwishActivation>(
                input, filter, bias, output, N, H, W, IC, K, R, S, pad_h, pad_w, stride_h, stride_w,
                dilation_h, dilation_w, groups, stream);
        case ActivationType::IDENTITY:
            return launch_grouped_conv2d<T, IdentityActivation>(
                input, filter, bias, output, N, H, W, IC, K, R, S, pad_h, pad_w, stride_h, stride_w,
                dilation_h, dilation_w, groups, stream);
    }
    return cudaErrorInvalidValue;
}

void validate_grouped_conv2d(TensorView output, TensorView input, TensorView filter,
                             Optional bias_opt, int64_t pad_h, int64_t pad_w, int64_t stride_h,
                             int64_t stride_w, int64_t dilation_h, int64_t dilation_w,
                             int64_t groups) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_INPUT(filter);
    CHECK_DIM(4, input);
    CHECK_DIM(4, output);
    CHECK_DIM(4, filter);
    CHECK_CONTIGUOUS_INPUT(input);
    CHECK_CONTIGUOUS_INPUT(output);
    CHECK_CONTIGUOUS_INPUT(filter);
    CHECK_DEVICE(input, output);
    CHECK_DEVICE(input, filter);

    TVM_FFI_ICHECK(groups > 0) << "groups must be positive, got " << groups;
    TVM_FFI_ICHECK(pad_h >= 0 && pad_w >= 0) << "padding must be non-negative";
    TVM_FFI_ICHECK(stride_h > 0 && stride_w > 0) << "stride must be positive";
    TVM_FFI_ICHECK(dilation_h > 0 && dilation_w > 0) << "dilation must be positive";

    int64_t N = input.size(0);
    int64_t H = input.size(1);
    int64_t W = input.size(2);
    int64_t IC = input.size(3);
    int64_t K = filter.size(0);
    int64_t R = filter.size(1);
    int64_t S = filter.size(2);
    TVM_FFI_ICHECK(IC % groups == 0 && K % groups == 0)
        << "groups=" << groups << " must divide input channels=" << IC
        << " and output channels=" << K;
    TVM_FFI_ICHECK(filter.size(3) == IC / groups)
        << "grouped Conv2D filter must be [K,R,S,IC/groups], got trailing dimension "
        << filter.size(3) << " for IC/groups=" << IC / groups;
    TVM_FFI_ICHECK(R > 0 && S > 0) << "filter dimensions must be positive";
    int64_t P = (H + 2 * pad_h - dilation_h * (R - 1) - 1) / stride_h + 1;
    int64_t Q = (W + 2 * pad_w - dilation_w * (S - 1) - 1) / stride_w + 1;
    TVM_FFI_ICHECK(P > 0 && Q > 0) << "grouped Conv2D output dimensions must be positive";
    TVM_FFI_ICHECK(output.size(0) == N && output.size(1) == P && output.size(2) == Q &&
                   output.size(3) == K)
        << "grouped Conv2D output must have shape [" << N << "," << P << "," << Q << "," << K
        << "]";
    TVM_FFI_ICHECK(input.dtype().code == filter.dtype().code &&
                   input.dtype().bits == filter.dtype().bits)
        << "input and filter dtypes must match";
    TVM_FFI_ICHECK(input.dtype().code == output.dtype().code &&
                   input.dtype().bits == output.dtype().bits)
        << "input and output dtypes must match";
    if (bias_opt.has_value()) {
        TensorView bias = bias_opt.value();
        CHECK_INPUT(bias);
        CHECK_DIM(1, bias);
        CHECK_CONTIGUOUS_INPUT(bias);
        CHECK_DEVICE(input, bias);
        TVM_FFI_ICHECK(bias.size(0) == K) << "bias must contain K=" << K << " elements";
        TVM_FFI_ICHECK(input.dtype().code == bias.dtype().code &&
                       input.dtype().bits == bias.dtype().bits)
            << "input and bias dtypes must match";
    }
}

}  // namespace

void grouped_conv2d(TensorView output, TensorView input, TensorView filter, Optional bias_opt,
                    int64_t pad_h, int64_t pad_w, int64_t stride_h, int64_t stride_w,
                    int64_t dilation_h, int64_t dilation_w, int64_t groups) {
    validate_grouped_conv2d(output, input, filter, bias_opt, pad_h, pad_w, stride_h, stride_w,
                            dilation_h, dilation_w, groups);
    cudaStream_t stream = get_stream(input.device());
    DISPATCH_DLPACK_HALF_DTYPE(input.dtype(), c_type, [&] {
        const c_type* bias = bias_opt.has_value()
                                 ? static_cast<const c_type*>(bias_opt.value().data_ptr())
                                 : nullptr;
        cudaError_t status = launch_grouped_conv2d<c_type, IdentityActivation>(
            static_cast<const c_type*>(input.data_ptr()),
            static_cast<const c_type*>(filter.data_ptr()), bias,
            static_cast<c_type*>(output.data_ptr()), input.size(0), input.size(1), input.size(2),
            input.size(3), filter.size(0), filter.size(1), filter.size(2), pad_h, pad_w, stride_h,
            stride_w, dilation_h, dilation_w, groups, stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "GroupedConv2D kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}

void grouped_conv2d_activation(TensorView output, TensorView input, TensorView filter,
                               Optional bias_opt, int64_t activation_type, int64_t pad_h,
                               int64_t pad_w, int64_t stride_h, int64_t stride_w,
                               int64_t dilation_h, int64_t dilation_w, int64_t groups) {
    validate_grouped_conv2d(output, input, filter, bias_opt, pad_h, pad_w, stride_h, stride_w,
                            dilation_h, dilation_w, groups);
    TVM_FFI_ICHECK(activation_type >= static_cast<int64_t>(ActivationType::RELU) &&
                   activation_type <= static_cast<int64_t>(ActivationType::IDENTITY))
        << "invalid activation type " << activation_type;
    cudaStream_t stream = get_stream(input.device());
    DISPATCH_DLPACK_HALF_DTYPE(input.dtype(), c_type, [&] {
        const c_type* bias = bias_opt.has_value()
                                 ? static_cast<const c_type*>(bias_opt.value().data_ptr())
                                 : nullptr;
        cudaError_t status = dispatch_grouped_conv2d_activation<c_type>(
            static_cast<const c_type*>(input.data_ptr()),
            static_cast<const c_type*>(filter.data_ptr()), bias,
            static_cast<c_type*>(output.data_ptr()), static_cast<ActivationType>(activation_type),
            input.size(0), input.size(1), input.size(2), input.size(3), filter.size(0),
            filter.size(1), filter.size(2), pad_h, pad_w, stride_h, stride_w, dilation_h,
            dilation_w, groups, stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "GroupedConv2DActivation kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}

// =============================================================================
// Conv2D launcher
// =============================================================================

#ifndef OASR_GROUPED_CONV2D_ONLY
void conv2d(TensorView output, TensorView input, TensorView filter, Optional bias_opt,
            int64_t pad_h, int64_t pad_w, int64_t stride_h, int64_t stride_w, int64_t dilation_h,
            int64_t dilation_w) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_INPUT(filter);
    CHECK_DIM(4, input);   // [N, H, W, IC]
    CHECK_DIM(4, filter);  // [K, R, S, IC]

    int N = input.size(0);
    int H = input.size(1);
    int W = input.size(2);
    int IC = input.size(3);
    int K = filter.size(0);
    int R = filter.size(1);
    int S = filter.size(2);

    cudaStream_t stream = get_stream(input.device());

    DISPATCH_DLPACK_HALF_DTYPE(input.dtype(), c_type, [&] {
        using CutlassType = typename ToCutlassType<c_type>::type;

        const CutlassType* bias_ptr = nullptr;
        if (bias_opt.has_value()) {
            bias_ptr = reinterpret_cast<const CutlassType*>(bias_opt.value().data_ptr());
        }

        cudaError_t status = conv::Conv2D<CutlassType>(
            reinterpret_cast<const CutlassType*>(input.data_ptr()),
            reinterpret_cast<const CutlassType*>(filter.data_ptr()), bias_ptr,
            reinterpret_cast<CutlassType*>(output.data_ptr()), N, H, W, IC, K, R, S,
            static_cast<int>(pad_h), static_cast<int>(pad_w), static_cast<int>(stride_h),
            static_cast<int>(stride_w), static_cast<int>(dilation_h), static_cast<int>(dilation_w),
            stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "Conv2D kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}

// =============================================================================
// Conv2D + Activation launcher
// =============================================================================

void conv2d_activation(TensorView output, TensorView input, TensorView filter, Optional bias_opt,
                       int64_t activation_type, int64_t pad_h, int64_t pad_w, int64_t stride_h,
                       int64_t stride_w, int64_t dilation_h, int64_t dilation_w) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_INPUT(filter);
    CHECK_DIM(4, input);
    CHECK_DIM(4, filter);

    int N = input.size(0);
    int H = input.size(1);
    int W = input.size(2);
    int IC = input.size(3);
    int K = filter.size(0);
    int R = filter.size(1);
    int S = filter.size(2);
    auto activation = static_cast<ActivationType>(activation_type);

    cudaStream_t stream = get_stream(input.device());

    DISPATCH_DLPACK_HALF_DTYPE(input.dtype(), c_type, [&] {
        using CutlassType = typename ToCutlassType<c_type>::type;

        const CutlassType* bias_ptr = nullptr;
        if (bias_opt.has_value()) {
            bias_ptr = reinterpret_cast<const CutlassType*>(bias_opt.value().data_ptr());
        }

        cudaError_t status = conv::Conv2DActivation<CutlassType>(
            reinterpret_cast<const CutlassType*>(input.data_ptr()),
            reinterpret_cast<const CutlassType*>(filter.data_ptr()), bias_ptr,
            reinterpret_cast<CutlassType*>(output.data_ptr()), activation, N, H, W, IC, K, R, S,
            static_cast<int>(pad_h), static_cast<int>(pad_w), static_cast<int>(stride_h),
            static_cast<int>(stride_w), static_cast<int>(dilation_h), static_cast<int>(dilation_w),
            stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "Conv2DActivation kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}
#endif
