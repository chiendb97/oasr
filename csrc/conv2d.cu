// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// TVM-FFI launcher layer for conv2d kernels.

#include <oasr/conv/conv2d.cuh>

#include "tvm_ffi_utils.h"

using namespace oasr;

// =============================================================================
// Helper: map CUDA types to CUTLASS types for Conv2D
// =============================================================================

namespace {

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

}  // namespace

// =============================================================================
// Conv2D launcher
// =============================================================================

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
            static_cast<int>(stride_w), static_cast<int>(dilation_h),
            static_cast<int>(dilation_w), stream);
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
            static_cast<int>(stride_w), static_cast<int>(dilation_h),
            static_cast<int>(dilation_w), stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "Conv2DActivation kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}
