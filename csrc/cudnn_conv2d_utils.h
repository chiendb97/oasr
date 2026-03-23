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

#include <stdexcept>
#include <string>

namespace oasr {
namespace conv {
namespace cudnn_impl {

// =============================================================================
// Error checking
// =============================================================================

#define OASR_CUDNN_CHECK(expr)                                                             \
    do {                                                                                    \
        cudnnStatus_t _s = (expr);                                                         \
        if (_s != CUDNN_STATUS_SUCCESS) {                                                  \
            throw std::runtime_error(std::string("cuDNN error in " #expr ": ") +           \
                                     cudnnGetErrorString(_s));                              \
        }                                                                                   \
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
// RAII descriptor wrappers
// =============================================================================

struct ScopedTensorDesc {
    cudnnTensorDescriptor_t desc;
    ScopedTensorDesc() { OASR_CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc)); }
    ~ScopedTensorDesc() { cudnnDestroyTensorDescriptor(desc); }
    ScopedTensorDesc(const ScopedTensorDesc&) = delete;
    ScopedTensorDesc& operator=(const ScopedTensorDesc&) = delete;
    operator cudnnTensorDescriptor_t() const { return desc; }
};

struct ScopedFilterDesc {
    cudnnFilterDescriptor_t desc;
    ScopedFilterDesc() { OASR_CUDNN_CHECK(cudnnCreateFilterDescriptor(&desc)); }
    ~ScopedFilterDesc() { cudnnDestroyFilterDescriptor(desc); }
    ScopedFilterDesc(const ScopedFilterDesc&) = delete;
    ScopedFilterDesc& operator=(const ScopedFilterDesc&) = delete;
    operator cudnnFilterDescriptor_t() const { return desc; }
};

struct ScopedConvDesc {
    cudnnConvolutionDescriptor_t desc;
    ScopedConvDesc() { OASR_CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&desc)); }
    ~ScopedConvDesc() { cudnnDestroyConvolutionDescriptor(desc); }
    ScopedConvDesc(const ScopedConvDesc&) = delete;
    ScopedConvDesc& operator=(const ScopedConvDesc&) = delete;
    operator cudnnConvolutionDescriptor_t() const { return desc; }
};

struct ScopedActivationDesc {
    cudnnActivationDescriptor_t desc;
    ScopedActivationDesc() { OASR_CUDNN_CHECK(cudnnCreateActivationDescriptor(&desc)); }
    ~ScopedActivationDesc() { cudnnDestroyActivationDescriptor(desc); }
    ScopedActivationDesc(const ScopedActivationDesc&) = delete;
    ScopedActivationDesc& operator=(const ScopedActivationDesc&) = delete;
    operator cudnnActivationDescriptor_t() const { return desc; }
};

// =============================================================================
// Grow-only workspace cache (thread-local)
// =============================================================================

inline void* getWorkspace(size_t required) {
    static thread_local void* buf = nullptr;
    static thread_local size_t cap = 0;
    if (required > cap) {
        if (buf) cudaFree(buf);
        cudaMalloc(&buf, required);
        cap = required;
    }
    return buf;
}

}  // namespace cudnn_impl
}  // namespace conv
}  // namespace oasr
