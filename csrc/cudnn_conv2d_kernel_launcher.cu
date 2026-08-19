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

//: ``OASR_CUDNN_CONV_FUSED=0`` restores the unfused conv → addTensor →
//: activation sequence.  Kept as an A/B switch in the repo's usual style: the
//: fused entry point picks its own epilogue, so a numerical difference against
//: the three-pass form has to be attributable to exactly this.
bool cudnn_fusion_enabled() {
    static const bool enabled = [] {
        const char* v = getenv("OASR_CUDNN_CONV_FUSED");
        return !(v && v[0] == '0' && v[1] == '\0');
    }();
    return enabled;
}

// ``bias`` may be null.  Folding it in here is what removes the separate
// ``cudnnAddTensor`` pass: the bias add is a full read-modify-write of the
// output, so on its own it costs about what the activation does.
template <typename T>
__global__ void gelu_bias_inplace_kernel(T* data, const T* bias, int n, int channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = static_cast<float>(data[idx]);
        if (bias)
            x += static_cast<float>(bias[idx % channels]);
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        data[idx] = static_cast<T>(x * cdf);
    }
}

template <typename T>
__global__ void swish_bias_inplace_kernel(T* data, const T* bias, int n, int channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = static_cast<float>(data[idx]);
        if (bias)
            x += static_cast<float>(bias[idx % channels]);
        data[idx] = static_cast<T>(x / (1.0f + expf(-x)));
    }
}

// =============================================================================
// cuDNN Conv2D forward — templated on element type
// =============================================================================

template <typename T>
ConvKey conv_key(int N, int H, int W, int IC, int K, int R, int S, int pad_h, int pad_w,
                 int stride_h, int stride_w, int dilation_h, int dilation_w) {
    ConvKey key;
    key.dtype = static_cast<int>(CudnnDtype<T>::value);
    key.N = N;
    key.H = H;
    key.W = W;
    key.IC = IC;
    key.K = K;
    key.R = R;
    key.S = S;
    key.pad_h = pad_h;
    key.pad_w = pad_w;
    key.stride_h = stride_h;
    key.stride_w = stride_w;
    key.dilation_h = dilation_h;
    key.dilation_w = dilation_w;
    return key;
}

template <typename T>
cudaError_t cudnn_conv2d_fwd(const T* input, const T* filter, const T* bias, T* output, int N,
                             int H, int W, int IC, int K, int R, int S, int pad_h, int pad_w,
                             int stride_h, int stride_w, int dilation_h, int dilation_w,
                             cudaStream_t stream) {
    cudnnHandle_t handle = getCudnnHandle();
    OASR_CUDNN_CHECK(cudnnSetStream(handle, stream));

    ConvPlan& plan = getConvPlan(
        conv_key<T>(N, H, W, IC, K, R, S, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w),
        CudnnDtype<T>::value);
    void* ws = (plan.ws_size > 0) ? getWorkspace(plan.ws_size) : nullptr;

    // Forward convolution
    float alpha = 1.0f, beta = 0.0f;
    OASR_CUDNN_CHECK(cudnnConvolutionForward(handle, &alpha, plan.x_desc, input, plan.w_desc,
                                             filter, plan.conv_desc, plan.algo, ws, plan.ws_size,
                                             &beta, plan.y_desc, output));

    // Bias: broadcast [K] → [N, P, Q, K]
    if (bias != nullptr) {
        float one = 1.0f;
        OASR_CUDNN_CHECK(
            cudnnAddTensor(handle, &one, plan.bias_desc, bias, &one, plan.y_desc, output));
    }

    return cudaGetLastError();
}

// =============================================================================
// cuDNN Conv2D + Activation forward
//
// The bias add and the activation are both full read-modify-write passes over
// the output, so running conv → addTensor → activation reads and writes the
// whole [N, P, Q, K] tensor three times for one convolution.
//
// ReLU takes cuDNN's fused entry point, one pass for all three operations.
// GELU and Swish are not cuDNN activation modes, so they fold the bias into
// their own elementwise kernel instead: two passes rather than three.
// =============================================================================

template <typename T>
cudaError_t cudnn_conv2d_activation_fwd(const T* input, const T* filter, const T* bias, T* output,
                                        ActivationType activation, int N, int H, int W, int IC,
                                        int K, int R, int S, int pad_h, int pad_w, int stride_h,
                                        int stride_w, int dilation_h, int dilation_w,
                                        cudaStream_t stream) {
    cudnnHandle_t handle = getCudnnHandle();
    ConvPlan& plan = getConvPlan(
        conv_key<T>(N, H, W, IC, K, R, S, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w),
        CudnnDtype<T>::value);

    // ----- Fused conv + bias + ReLU -----
    if (activation == ActivationType::RELU && bias != nullptr && plan.fused_relu_ok != 0 &&
        cudnn_fusion_enabled()) {
        OASR_CUDNN_CHECK(cudnnSetStream(handle, stream));
        void* ws = (plan.ws_size > 0) ? getWorkspace(plan.ws_size) : nullptr;
        // y = relu(1.0 * conv(x, w) + 0.0 * z + bias).  ``z`` must be described
        // and non-null; aliasing it to ``y`` with alpha2 = 0 is cuDNN's
        // documented in-place form and reads nothing.
        float alpha1 = 1.0f, alpha2 = 0.0f;
        cudnnStatus_t s = cudnnConvolutionBiasActivationForward(
            handle, &alpha1, plan.x_desc, input, plan.w_desc, filter, plan.conv_desc, plan.algo, ws,
            plan.ws_size, &alpha2, plan.y_desc, output, plan.bias_desc, bias, plan.relu_desc,
            plan.y_desc, output);
        if (s == CUDNN_STATUS_SUCCESS) {
            plan.fused_relu_ok = 1;
            return cudaGetLastError();
        }
        if (s != CUDNN_STATUS_NOT_SUPPORTED && s != CUDNN_STATUS_ARCH_MISMATCH) {
            throw std::runtime_error(std::string("cuDNN error in "
                                                 "cudnnConvolutionBiasActivationForward: ") +
                                     cudnnGetErrorString(s));
        }
        // Remembered per shape so the probe is paid once, not per call.  The
        // fused entry point rejects some (algorithm, shape) pairs the plain
        // forward accepts, and which ones is a cuDNN-version detail.
        plan.fused_relu_ok = 0;
    }

    const bool fold_bias_into_activation =
        bias != nullptr && cudnn_fusion_enabled() &&
        (activation == ActivationType::GELU || activation == ActivationType::SWISH);

    // Run convolution (+ bias, unless the activation kernel below folds it in)
    cudaError_t err = cudnn_conv2d_fwd(input, filter, fold_bias_into_activation ? nullptr : bias,
                                       output, N, H, W, IC, K, R, S, pad_h, pad_w, stride_h,
                                       stride_w, dilation_h, dilation_w, stream);
    if (err != cudaSuccess)
        return err;

    const int P = plan.P;
    const int Q = plan.Q;
    const int n_elements = N * P * Q * K;

    if (activation == ActivationType::RELU) {
        // cuDNN native ReLU (no bias, or the fused path was unsupported)
        float alpha = 1.0f, beta = 0.0f;
        OASR_CUDNN_CHECK(cudnnActivationForward(handle, plan.relu_desc, &alpha, plan.y_desc, output,
                                                &beta, plan.y_desc, output));
    } else if (activation == ActivationType::GELU) {
        constexpr int kThreads = 256;
        int blocks = (n_elements + kThreads - 1) / kThreads;
        gelu_bias_inplace_kernel<<<blocks, kThreads, 0, stream>>>(
            output, fold_bias_into_activation ? bias : nullptr, n_elements, K);
    } else if (activation == ActivationType::SWISH) {
        constexpr int kThreads = 256;
        int blocks = (n_elements + kThreads - 1) / kThreads;
        swish_bias_inplace_kernel<<<blocks, kThreads, 0, stream>>>(
            output, fold_bias_into_activation ? bias : nullptr, n_elements, K);
    } else if (activation != ActivationType::IDENTITY) {
        // Declared, not routed around: this backend has kernels for RELU, GELU
        // (tanh) and SWISH.  Anything else — GELU_ERF today — would otherwise
        // return an un-activated convolution that looks like a correct result.
        return cudaErrorInvalidValue;
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
            static_cast<int>(stride_w), static_cast<int>(dilation_h), static_cast<int>(dilation_w),
            stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "cuDNN Conv2D kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}

void cudnn_conv2d_activation(TensorView output, TensorView input, TensorView filter,
                             Optional bias_opt, int64_t activation_type, int64_t pad_h,
                             int64_t pad_w, int64_t stride_h, int64_t stride_w, int64_t dilation_h,
                             int64_t dilation_w) {
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
            static_cast<int>(stride_w), static_cast<int>(dilation_h), static_cast<int>(dilation_w),
            stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "cuDNN Conv2DActivation kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}

// =============================================================================
// Dense Conv1D launchers
//
// cuDNN's public convolution API is 2-D here, but packed BTC/KSC tensors are
// exactly packed NHWC/KRSC tensors with H=R_h=1.  These exports are both the
// unaligned-channel path and an autotuning baseline for the CUTLASS kernels.
// =============================================================================

void cudnn_conv1d(TensorView output, TensorView input, TensorView filter, Optional bias_opt,
                  int64_t padding, int64_t stride, int64_t dilation) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_INPUT(filter);
    CHECK_DIM(3, input);
    CHECK_DIM(3, filter);
    CHECK_DIM(3, output);
    CHECK_CONTIGUOUS_INPUT(input);
    CHECK_CONTIGUOUS_INPUT(output);
    CHECK_CONTIGUOUS_INPUT(filter);
    CHECK_DEVICE(input, output);
    CHECK_DEVICE(input, filter);
    TVM_FFI_ICHECK(padding >= 0 && stride > 0 && dilation > 0)
        << "padding must be non-negative and stride/dilation must be positive";

    int B = input.size(0);
    int T = input.size(1);
    int IC = input.size(2);
    int K = filter.size(0);
    int R = filter.size(1);
    int Q = (T + 2 * static_cast<int>(padding) - static_cast<int>(dilation) * (R - 1) - 1) /
                static_cast<int>(stride) +
            1;
    TVM_FFI_ICHECK(filter.size(2) == IC) << "Conv1D filter/input channel mismatch";
    TVM_FFI_ICHECK(output.size(0) == B && output.size(1) == Q && output.size(2) == K)
        << "Conv1D output shape mismatch";

    cudaStream_t stream = get_stream(input.device());
    DISPATCH_DLPACK_HALF_DTYPE(input.dtype(), c_type, [&] {
        const c_type* bias_ptr = nullptr;
        if (bias_opt.has_value()) {
            CHECK_INPUT(bias_opt.value());
            CHECK_DIM(1, bias_opt.value());
            CHECK_DEVICE(input, bias_opt.value());
            TVM_FFI_ICHECK(bias_opt.value().size(0) == K) << "Conv1D bias shape mismatch";
            bias_ptr = reinterpret_cast<const c_type*>(bias_opt.value().data_ptr());
        }
        cudaError_t status = cudnn_conv2d_fwd<c_type>(
            reinterpret_cast<const c_type*>(input.data_ptr()),
            reinterpret_cast<const c_type*>(filter.data_ptr()), bias_ptr,
            reinterpret_cast<c_type*>(output.data_ptr()), B, 1, T, IC, K, 1, R, 0,
            static_cast<int>(padding), 1, static_cast<int>(stride), 1, static_cast<int>(dilation),
            stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "cuDNN Conv1D kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}

void cudnn_conv1d_activation(TensorView output, TensorView input, TensorView filter,
                             Optional bias_opt, int64_t activation_type, int64_t padding,
                             int64_t stride, int64_t dilation) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_INPUT(filter);
    CHECK_DIM(3, input);
    CHECK_DIM(3, filter);
    CHECK_DIM(3, output);
    CHECK_CONTIGUOUS_INPUT(input);
    CHECK_CONTIGUOUS_INPUT(output);
    CHECK_CONTIGUOUS_INPUT(filter);
    CHECK_DEVICE(input, output);
    CHECK_DEVICE(input, filter);
    TVM_FFI_ICHECK(padding >= 0 && stride > 0 && dilation > 0)
        << "padding must be non-negative and stride/dilation must be positive";

    int B = input.size(0);
    int T = input.size(1);
    int IC = input.size(2);
    int K = filter.size(0);
    int R = filter.size(1);
    int Q = (T + 2 * static_cast<int>(padding) - static_cast<int>(dilation) * (R - 1) - 1) /
                static_cast<int>(stride) +
            1;
    TVM_FFI_ICHECK(filter.size(2) == IC) << "Conv1D filter/input channel mismatch";
    TVM_FFI_ICHECK(output.size(0) == B && output.size(1) == Q && output.size(2) == K)
        << "Conv1D output shape mismatch";
    auto activation = static_cast<ActivationType>(activation_type);

    cudaStream_t stream = get_stream(input.device());
    DISPATCH_DLPACK_HALF_DTYPE(input.dtype(), c_type, [&] {
        const c_type* bias_ptr = nullptr;
        if (bias_opt.has_value()) {
            CHECK_INPUT(bias_opt.value());
            CHECK_DIM(1, bias_opt.value());
            CHECK_DEVICE(input, bias_opt.value());
            TVM_FFI_ICHECK(bias_opt.value().size(0) == K) << "Conv1D bias shape mismatch";
            bias_ptr = reinterpret_cast<const c_type*>(bias_opt.value().data_ptr());
        }
        cudaError_t status = cudnn_conv2d_activation_fwd<c_type>(
            reinterpret_cast<const c_type*>(input.data_ptr()),
            reinterpret_cast<const c_type*>(filter.data_ptr()), bias_ptr,
            reinterpret_cast<c_type*>(output.data_ptr()), activation, B, 1, T, IC, K, 1, R, 0,
            static_cast<int>(padding), 1, static_cast<int>(stride), 1, static_cast<int>(dilation),
            stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "cuDNN Conv1DActivation kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}

// =============================================================================
// TVM-FFI symbol exports
// =============================================================================

TVM_FFI_DLL_EXPORT_TYPED_FUNC(cudnn_conv2d, cudnn_conv2d);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(cudnn_conv2d_activation, cudnn_conv2d_activation);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(cudnn_conv1d, cudnn_conv1d);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(cudnn_conv1d_activation, cudnn_conv1d_activation);
