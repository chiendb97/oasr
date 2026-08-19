// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// cuDNN utilities for Conv2D kernels — handle management, RAII descriptor
// wrappers, and dtype mapping.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace oasr {
namespace conv {
namespace cudnn_impl {

// =============================================================================
// Error checking
// =============================================================================

#define OASR_CUDNN_CHECK(expr)                                                   \
    do {                                                                         \
        cudnnStatus_t _s = (expr);                                               \
        if (_s != CUDNN_STATUS_SUCCESS) {                                        \
            throw std::runtime_error(std::string("cuDNN error in " #expr ": ") + \
                                     cudnnGetErrorString(_s));                   \
        }                                                                        \
    } while (0)

// =============================================================================
// cuDNN handle (thread-local singleton)
// =============================================================================

inline cudnnHandle_t getCudnnHandle() {
    static thread_local cudnnHandle_t handle = nullptr;
    if (handle == nullptr) {
        OASR_CUDNN_CHECK(cudnnCreate(&handle));
    }
    return handle;
}

// =============================================================================
// cuDNN data-type mapping
// =============================================================================

template <typename T>
struct CudnnDtype;

template <>
struct CudnnDtype<float> {
    static constexpr cudnnDataType_t value = CUDNN_DATA_FLOAT;
};

template <>
struct CudnnDtype<half> {
    static constexpr cudnnDataType_t value = CUDNN_DATA_HALF;
};

template <>
struct CudnnDtype<__nv_bfloat16> {
    static constexpr cudnnDataType_t value = CUDNN_DATA_BFLOAT16;
};

// =============================================================================
// Convolution plan cache (thread-local)
//
// Descriptors and the algorithm heuristic depend only on shape, so rebuilding
// them per call adds avoidable host work to the critical path.
//
// ``fused_relu_ok`` records whether ``cudnnConvolutionBiasActivationForward`` accepts
// this shape + algorithm, so the probe happens once rather than on every call.
// =============================================================================

struct ConvKey {
    int dtype, N, H, W, IC, K, R, S;
    int pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w;

    bool operator==(const ConvKey& o) const {
        return dtype == o.dtype && N == o.N && H == o.H && W == o.W && IC == o.IC && K == o.K &&
               R == o.R && S == o.S && pad_h == o.pad_h && pad_w == o.pad_w &&
               stride_h == o.stride_h && stride_w == o.stride_w && dilation_h == o.dilation_h &&
               dilation_w == o.dilation_w;
    }
};

struct ConvKeyHash {
    size_t operator()(const ConvKey& k) const {
        const int* p = &k.dtype;
        size_t h = 1469598103934665603ull;
        for (size_t i = 0; i < sizeof(ConvKey) / sizeof(int); ++i) {
            h ^= static_cast<size_t>(static_cast<uint32_t>(p[i]));
            h *= 1099511628211ull;
        }
        return h;
    }
};

struct ConvPlan {
    cudnnTensorDescriptor_t x_desc = nullptr;
    cudnnTensorDescriptor_t y_desc = nullptr;
    cudnnTensorDescriptor_t bias_desc = nullptr;
    cudnnFilterDescriptor_t w_desc = nullptr;
    cudnnConvolutionDescriptor_t conv_desc = nullptr;
    cudnnActivationDescriptor_t relu_desc = nullptr;
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    size_t ws_size = 0;
    int P = 0, Q = 0;
    //: -1 = not probed yet, 0 = unsupported, 1 = supported.  Probed on first use
    //: because which algorithms the fused entry point accepts is a cuDNN-version
    //: and shape detail, and a wrong guess here is a hard failure rather than a
    //: slow path.
    int fused_relu_ok = -1;
};

//: Distinct conv shapes kept before the cache is dropped wholesale.  Offline
//: micro-batches vary in N and W, so the key space is open-ended; a plain cap is
//: enough because a rebuild costs one algorithm query.
inline constexpr size_t kConvPlanCacheMax = 512;

inline ConvPlan& getConvPlan(const ConvKey& key, cudnnDataType_t dtype) {
    static thread_local std::unordered_map<ConvKey, ConvPlan, ConvKeyHash> cache;
    auto it = cache.find(key);
    if (it != cache.end())
        return it->second;

    if (cache.size() >= kConvPlanCacheMax) {
        for (auto& kv : cache) {
            cudnnDestroyTensorDescriptor(kv.second.x_desc);
            cudnnDestroyTensorDescriptor(kv.second.y_desc);
            cudnnDestroyTensorDescriptor(kv.second.bias_desc);
            cudnnDestroyFilterDescriptor(kv.second.w_desc);
            cudnnDestroyConvolutionDescriptor(kv.second.conv_desc);
            cudnnDestroyActivationDescriptor(kv.second.relu_desc);
        }
        cache.clear();
    }

    ConvPlan plan;
    plan.P = (key.H + 2 * key.pad_h - key.dilation_h * (key.R - 1) - 1) / key.stride_h + 1;
    plan.Q = (key.W + 2 * key.pad_w - key.dilation_w * (key.S - 1) - 1) / key.stride_w + 1;

    OASR_CUDNN_CHECK(cudnnCreateTensorDescriptor(&plan.x_desc));
    OASR_CUDNN_CHECK(cudnnCreateTensorDescriptor(&plan.y_desc));
    OASR_CUDNN_CHECK(cudnnCreateTensorDescriptor(&plan.bias_desc));
    OASR_CUDNN_CHECK(cudnnCreateFilterDescriptor(&plan.w_desc));
    OASR_CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&plan.conv_desc));
    OASR_CUDNN_CHECK(cudnnCreateActivationDescriptor(&plan.relu_desc));

    // Input [N, H, W, IC] / output [N, P, Q, K] — NHWC strides.
    OASR_CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(plan.x_desc, dtype, key.N, key.IC, key.H, key.W,
                                                  key.H * key.W * key.IC, 1, key.W * key.IC,
                                                  key.IC));
    OASR_CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(plan.y_desc, dtype, key.N, key.K, plan.P, plan.Q,
                                                  plan.P * plan.Q * key.K, 1, plan.Q * key.K,
                                                  key.K));
    // Filter [K, R, S, IC] — cuDNN NHWC format.
    OASR_CUDNN_CHECK(cudnnSetFilter4dDescriptor(plan.w_desc, dtype, CUDNN_TENSOR_NHWC, key.K,
                                                key.IC, key.R, key.S));
    // Bias broadcasts [K] over [N, P, Q, K].
    OASR_CUDNN_CHECK(
        cudnnSetTensor4dDescriptor(plan.bias_desc, CUDNN_TENSOR_NCHW, dtype, 1, key.K, 1, 1));

    OASR_CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        plan.conv_desc, key.pad_h, key.pad_w, key.stride_h, key.stride_w, key.dilation_h,
        key.dilation_w, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    OASR_CUDNN_CHECK(cudnnSetConvolutionMathType(plan.conv_desc, CUDNN_DEFAULT_MATH));

    OASR_CUDNN_CHECK(cudnnSetActivationDescriptor(plan.relu_desc, CUDNN_ACTIVATION_RELU,
                                                  CUDNN_PROPAGATE_NAN, 0.0));

    cudnnHandle_t handle = getCudnnHandle();
    int returned = 0;
    cudnnConvolutionFwdAlgoPerf_t perf;
    OASR_CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
        handle, plan.x_desc, plan.w_desc, plan.conv_desc, plan.y_desc, 1, &returned, &perf));
    plan.algo = perf.algo;
    OASR_CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        handle, plan.x_desc, plan.w_desc, plan.conv_desc, plan.y_desc, plan.algo, &plan.ws_size));

    return cache.emplace(key, plan).first->second;
}

// =============================================================================
// Grow-only workspace cache (thread-local)
// =============================================================================

inline void* getWorkspace(size_t required) {
    static thread_local void* buf = nullptr;
    static thread_local size_t cap = 0;
    if (required > cap) {
        if (buf)
            cudaFree(buf);
        cudaMalloc(&buf, required);
        cap = required;
    }
    return buf;
}

}  // namespace cudnn_impl
}  // namespace conv
}  // namespace oasr
