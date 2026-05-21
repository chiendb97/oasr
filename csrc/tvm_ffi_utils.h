// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Shared TVM-FFI utilities for OASR launchers.
// Mirrors FlashInfer's tvm_ffi_utils.h pattern.

#pragma once

#include <cuda_runtime.h>

#include <tvm/ffi/c_api.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/optional.h>

using TensorView = tvm::ffi::TensorView;
using Optional = tvm::ffi::Optional<TensorView>;

namespace oasr {

// =============================================================================
// DLPack dtype constants
// =============================================================================

static constexpr DLDataType dl_float16 = {kDLFloat, 16, 1};
static constexpr DLDataType dl_bfloat16 = {kDLBfloat, 16, 1};
static constexpr DLDataType dl_float32 = {kDLFloat, 32, 1};
static constexpr DLDataType dl_int8 = {kDLInt, 8, 1};
static constexpr DLDataType dl_int32 = {kDLInt, 32, 1};

// =============================================================================
// Validation macros
// =============================================================================

#define CHECK_INPUT(x)                                                                    \
    TVM_FFI_ICHECK((x).device().device_type == kDLCUDA) << "Input must be a CUDA tensor"

#define CHECK_DIM(expected, x)                                                            \
    TVM_FFI_ICHECK((x).ndim() == (expected))                                             \
        << "Expected " << (expected) << "D tensor, got " << (x).ndim() << "D"

#define CHECK_DEVICE(x, y)                                                                \
    TVM_FFI_ICHECK((x).device().device_id == (y).device().device_id)                     \
        << "Tensors must be on the same device"

#define CHECK_LAST_DIM_CONTIGUOUS_INPUT(x)                                                \
    TVM_FFI_ICHECK((x).stride((x).ndim() - 1) == 1)                                     \
        << "Tensor must be contiguous along the last dimension"

// =============================================================================
// Dtype dispatch macros
// =============================================================================

// Dispatch for FP16/BF16/FP32 dtypes
#define DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(dl_dtype, c_type, ...)                        \
    [&]() -> bool {                                                                       \
        if ((dl_dtype).code == kDLFloat && (dl_dtype).bits == 32) {                       \
            using c_type = float;                                                         \
            return __VA_ARGS__();                                                         \
        } else if ((dl_dtype).code == kDLFloat && (dl_dtype).bits == 16) {                \
            using c_type = half;                                                          \
            return __VA_ARGS__();                                                         \
        } else if ((dl_dtype).code == kDLBfloat && (dl_dtype).bits == 16) {               \
            using c_type = __nv_bfloat16;                                                 \
            return __VA_ARGS__();                                                         \
        } else {                                                                          \
            TVM_FFI_ICHECK(false) << "Unsupported dtype: code=" << (dl_dtype).code        \
                                  << " bits=" << (dl_dtype).bits;                         \
            return false;                                                                 \
        }                                                                                 \
    }()

// Alias: same as DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16 (dispatches FP32/FP16/BF16)
#define DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP32_FP16 DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16

// Dispatch for FP16/BF16 only (half-precision input types for GEMM/conv)
#define DISPATCH_DLPACK_HALF_DTYPE(dl_dtype, c_type, ...)                                 \
    [&]() -> bool {                                                                       \
        if ((dl_dtype).code == kDLFloat && (dl_dtype).bits == 16) {                       \
            using c_type = half;                                                          \
            return __VA_ARGS__();                                                         \
        } else if ((dl_dtype).code == kDLBfloat && (dl_dtype).bits == 16) {               \
            using c_type = __nv_bfloat16;                                                 \
            return __VA_ARGS__();                                                         \
        } else {                                                                          \
            TVM_FFI_ICHECK(false) << "Unsupported input dtype for GEMM/conv: "            \
                                  << "code=" << (dl_dtype).code                           \
                                  << " bits=" << (dl_dtype).bits                          \
                                  << ". Only FP16 and BF16 are supported.";               \
            return false;                                                                 \
        }                                                                                 \
    }()

// =============================================================================
// CUDA helpers
// =============================================================================

inline cudaStream_t get_stream(DLDevice device) {
    // Use the caller's current CUDA stream (set on the FFI env by the
    // framework, e.g. ``tvm_ffi.use_torch_stream`` or torch's autograd
    // dispatcher). Returning ``nullptr`` (the null/default stream) here
    // breaks CUDA Graph capture: ``torch.cuda.graph`` records kernels
    // launched on the *capture* stream and silently skips any launched on
    // the null stream, so every JIT-compiled OASR op (conv2d, gemm, glu,
    // norm, ...) is left out of the captured graph, the resulting graph is
    // empty (PyTorch warns ``CUDA Graph is empty``), and replays produce
    // wrong / NaN outputs because none of the encoder ops actually run.
    TVMFFIStreamHandle s = TVMFFIEnvGetStream(device.device_type, device.device_id);
    return static_cast<cudaStream_t>(s);
}

inline size_t get_element_size(DLDataType dtype) {
    return (dtype.bits * dtype.lanes + 7) / 8;
}

}  // namespace oasr
