// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// cuDNN Conv2D kernel launcher — used for small input channel counts (IC < 8)
// where CUTLASS implicit GEMM with scalar alignment is suboptimal.
//
// Follows FlashInfer's cudnn_sdpa_kernel_launcher.cu pattern:
//   - Self-contained launcher with cuDNN API calls
//   - TVM-FFI exports at the bottom of the file
//
// Layout: NHWC for input/output, KRSC (= cuDNN NHWC) for filters.

#include <oasr/common/types.h>

#include "cudnn_conv2d_utils.h"
#include "tvm_ffi_utils.h"

using namespace oasr;
using namespace oasr::conv::cudnn_impl;

// =============================================================================
// Element-wise activation kernels (GELU / Swish not native in cuDNN)
// =============================================================================

namespace {

template <typename T>
__global__ void gelu_inplace_kernel(T* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = static_cast<float>(data[idx]);
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        data[idx] = static_cast<T>(x * cdf);
    }
}

template <typename T>
__global__ void swish_inplace_kernel(T* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = static_cast<float>(data[idx]);
        data[idx] = static_cast<T>(x / (1.0f + expf(-x)));
    }
}

// =============================================================================
// cuDNN Conv2D forward — templated on element type
// =============================================================================

template <typename T>
cudaError_t cudnn_conv2d_fwd(const T* input, const T* filter, const T* bias, T* output, int N,
                              int H, int W, int IC, int K, int R, int S, int pad_h, int pad_w,
                              int stride_h, int stride_w, int dilation_h, int dilation_w,
                              cudaStream_t stream) {
    cudnnHandle_t handle = getCudnnHandle();
    OASR_CUDNN_CHECK(cudnnSetStream(handle, stream));

    constexpr cudnnDataType_t dtype = CudnnDtype<T>::value;

    int P = (H + 2 * pad_h - dilation_h * (R - 1) - 1) / stride_h + 1;
    int Q = (W + 2 * pad_w - dilation_w * (S - 1) - 1) / stride_w + 1;

    // Input [N, H, W, IC] — NHWC strides
    ScopedTensorDesc x_desc;
    OASR_CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(x_desc, dtype, N, IC, H, W,
                                                   H * W * IC, 1, W * IC, IC));

    // Filter [K, R, S, IC] — cuDNN NHWC format
    ScopedFilterDesc w_desc;
    OASR_CUDNN_CHECK(cudnnSetFilter4dDescriptor(w_desc, dtype, CUDNN_TENSOR_NHWC, K, IC, R, S));

    // Output [N, P, Q, K] — NHWC strides
    ScopedTensorDesc y_desc;
    OASR_CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(y_desc, dtype, N, K, P, Q,
                                                   P * Q * K, 1, Q * K, K));

    // Convolution descriptor
    ScopedConvDesc conv_desc;
    OASR_CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc, pad_h, pad_w, stride_h, stride_w,
                                                      dilation_h, dilation_w,
                                                      CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    OASR_CUDNN_CHECK(cudnnSetConvolutionMathType(conv_desc, CUDNN_DEFAULT_MATH));

    // Algorithm selection (heuristic — avoids benchmarking overhead)
    int returned = 0;
    cudnnConvolutionFwdAlgoPerf_t perf;
    OASR_CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(handle, x_desc, w_desc, conv_desc,
                                                             y_desc, 1, &returned, &perf));
    cudnnConvolutionFwdAlgo_t algo = perf.algo;

    // Workspace
    size_t ws_size = 0;
    OASR_CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle, x_desc, w_desc, conv_desc,
                                                              y_desc, algo, &ws_size));
    void* ws = (ws_size > 0) ? getWorkspace(ws_size) : nullptr;

    // Forward convolution
    float alpha = 1.0f, beta = 0.0f;
    OASR_CUDNN_CHECK(cudnnConvolutionForward(handle, &alpha, x_desc, input, w_desc, filter,
                                              conv_desc, algo, ws, ws_size, &beta, y_desc,
                                              output));

    // Bias: broadcast [K] → [N, P, Q, K]
    if (bias != nullptr) {
        ScopedTensorDesc b_desc;
        OASR_CUDNN_CHECK(
            cudnnSetTensor4dDescriptor(b_desc, CUDNN_TENSOR_NCHW, dtype, 1, K, 1, 1));
        float one = 1.0f;
        OASR_CUDNN_CHECK(cudnnAddTensor(handle, &one, b_desc, bias, &one, y_desc, output));
    }

    return cudaGetLastError();
}

// =============================================================================
// cuDNN Conv2D + Activation forward
// =============================================================================

template <typename T>
cudaError_t cudnn_conv2d_activation_fwd(const T* input, const T* filter, const T* bias, T* output,
                                         ActivationType activation, int N, int H, int W, int IC,
                                         int K, int R, int S, int pad_h, int pad_w, int stride_h,
                                         int stride_w, int dilation_h, int dilation_w,
                                         cudaStream_t stream) {
    // Run convolution (+ bias)
    cudaError_t err =
        cudnn_conv2d_fwd(input, filter, bias, output, N, H, W, IC, K, R, S, pad_h, pad_w,
                         stride_h, stride_w, dilation_h, dilation_w, stream);
    if (err != cudaSuccess) return err;

    int P = (H + 2 * pad_h - dilation_h * (R - 1) - 1) / stride_h + 1;
    int Q = (W + 2 * pad_w - dilation_w * (S - 1) - 1) / stride_w + 1;
    int n_elements = N * P * Q * K;

    if (activation == ActivationType::RELU) {
        // cuDNN native ReLU
        cudnnHandle_t handle = getCudnnHandle();
        constexpr cudnnDataType_t dtype = CudnnDtype<T>::value;
        ScopedTensorDesc y_desc;
        OASR_CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(y_desc, dtype, N, K, P, Q,
                                                       P * Q * K, 1, Q * K, K));
        ScopedActivationDesc act;
        OASR_CUDNN_CHECK(
            cudnnSetActivationDescriptor(act, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));
        float alpha = 1.0f, beta = 0.0f;
        OASR_CUDNN_CHECK(
            cudnnActivationForward(handle, act, &alpha, y_desc, output, &beta, y_desc, output));
    } else if (activation == ActivationType::GELU) {
        constexpr int kThreads = 256;
        int blocks = (n_elements + kThreads - 1) / kThreads;
        gelu_inplace_kernel<<<blocks, kThreads, 0, stream>>>(output, n_elements);
    } else if (activation == ActivationType::SWISH) {
        constexpr int kThreads = 256;
        int blocks = (n_elements + kThreads - 1) / kThreads;
        swish_inplace_kernel<<<blocks, kThreads, 0, stream>>>(output, n_elements);
    }

    return cudaGetLastError();
}

}  // anonymous namespace

// =============================================================================
// TVM-FFI launchers
// =============================================================================

void cudnn_conv2d(TensorView output, TensorView input, TensorView filter, Optional bias_opt,
                  int64_t pad_h, int64_t pad_w, int64_t stride_h, int64_t stride_w,
                  int64_t dilation_h, int64_t dilation_w) {
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
        const c_type* bias_ptr = nullptr;
        if (bias_opt.has_value()) {
            bias_ptr = reinterpret_cast<const c_type*>(bias_opt.value().data_ptr());
        }

        cudaError_t status = cudnn_conv2d_fwd<c_type>(
            reinterpret_cast<const c_type*>(input.data_ptr()),
            reinterpret_cast<const c_type*>(filter.data_ptr()), bias_ptr,
            reinterpret_cast<c_type*>(output.data_ptr()), N, H, W, IC, K, R, S,
            static_cast<int>(pad_h), static_cast<int>(pad_w), static_cast<int>(stride_h),
            static_cast<int>(stride_w), static_cast<int>(dilation_h),
            static_cast<int>(dilation_w), stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "cuDNN Conv2D kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}

void cudnn_conv2d_activation(TensorView output, TensorView input, TensorView filter,
                              Optional bias_opt, int64_t activation_type, int64_t pad_h,
                              int64_t pad_w, int64_t stride_h, int64_t stride_w,
                              int64_t dilation_h, int64_t dilation_w) {
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
        const c_type* bias_ptr = nullptr;
        if (bias_opt.has_value()) {
            bias_ptr = reinterpret_cast<const c_type*>(bias_opt.value().data_ptr());
        }

        cudaError_t status = cudnn_conv2d_activation_fwd<c_type>(
            reinterpret_cast<const c_type*>(input.data_ptr()),
            reinterpret_cast<const c_type*>(filter.data_ptr()), bias_ptr,
            reinterpret_cast<c_type*>(output.data_ptr()), activation, N, H, W, IC, K, R, S,
            static_cast<int>(pad_h), static_cast<int>(pad_w), static_cast<int>(stride_h),
            static_cast<int>(stride_w), static_cast<int>(dilation_h),
            static_cast<int>(dilation_w), stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "cuDNN Conv2DActivation kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}

// =============================================================================
// TVM-FFI symbol exports
// =============================================================================

TVM_FFI_DLL_EXPORT_TYPED_FUNC(cudnn_conv2d, cudnn_conv2d);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(cudnn_conv2d_activation, cudnn_conv2d_activation);
