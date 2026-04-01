// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// TVM-FFI launcher layer for conv1d kernels.

#include <oasr/conv/conv1d.cuh>
#include <oasr/gemm/gemm.cuh>

#include "tvm_ffi_utils.h"

using namespace oasr;

// =============================================================================
// Depthwise Conv1D launcher
// =============================================================================

void depthwise_conv1d(TensorView output, TensorView input, TensorView weight, Optional bias_opt,
                      int64_t padding) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_INPUT(weight);
    CHECK_DIM(3, input);

    int batch_size = input.size(0);
    int seq_len = input.size(1);
    int channels = input.size(2);
    int kernel_size = weight.size(0);

    cudaStream_t stream = get_stream(input.device());

    DISPATCH_DLPACK_HALF_DTYPE(input.dtype(), c_type, [&] {
        const c_type* bias_ptr = nullptr;
        if (bias_opt.has_value()) {
            bias_ptr = static_cast<const c_type*>(bias_opt.value().data_ptr());
        }
        cudaError_t status = conv::DepthwiseConv1D<c_type>(
            static_cast<const c_type*>(input.data_ptr()),
            static_cast<const c_type*>(weight.data_ptr()), bias_ptr,
            static_cast<c_type*>(output.data_ptr()), batch_size, seq_len, channels, kernel_size,
            static_cast<int>(padding), stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "DepthwiseConv1D kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}

// =============================================================================
// Depthwise Conv1D + SiLU launcher
// =============================================================================

void depthwise_conv1d_silu(TensorView output, TensorView input, TensorView weight,
                           Optional bias_opt, int64_t padding) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_INPUT(weight);
    CHECK_DIM(3, input);

    int batch_size = input.size(0);
    int seq_len = input.size(1);
    int channels = input.size(2);
    int kernel_size = weight.size(0);

    cudaStream_t stream = get_stream(input.device());

    DISPATCH_DLPACK_HALF_DTYPE(input.dtype(), c_type, [&] {
        const c_type* bias_ptr = nullptr;
        if (bias_opt.has_value()) {
            bias_ptr = static_cast<const c_type*>(bias_opt.value().data_ptr());
        }
        cudaError_t status = conv::DepthwiseConv1DSilu<c_type>(
            static_cast<const c_type*>(input.data_ptr()),
            static_cast<const c_type*>(weight.data_ptr()), bias_ptr,
            static_cast<c_type*>(output.data_ptr()), batch_size, seq_len, channels, kernel_size,
            static_cast<int>(padding), stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "DepthwiseConv1DSilu kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}

// =============================================================================
// Causal Conv1D launcher
// =============================================================================

void causal_conv1d(TensorView output, TensorView input, TensorView state, TensorView weight,
                   Optional bias_opt) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_INPUT(state);
    CHECK_INPUT(weight);
    CHECK_DIM(3, input);

    int batch_size = input.size(0);
    int chunk_len = input.size(1);
    int channels = input.size(2);
    int kernel_size = weight.size(2);

    cudaStream_t stream = get_stream(input.device());

    DISPATCH_DLPACK_HALF_DTYPE(input.dtype(), c_type, [&] {
        const c_type* bias_ptr = nullptr;
        if (bias_opt.has_value()) {
            bias_ptr = static_cast<const c_type*>(bias_opt.value().data_ptr());
        }
        cudaError_t status = conv::CausalConv1D<c_type>(
            static_cast<const c_type*>(input.data_ptr()),
            static_cast<c_type*>(state.data_ptr()),
            static_cast<const c_type*>(weight.data_ptr()), bias_ptr,
            static_cast<c_type*>(output.data_ptr()), batch_size, chunk_len, channels, kernel_size,
            stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "CausalConv1D kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}
