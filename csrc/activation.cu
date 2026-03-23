// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// TVM-FFI launcher layer for activation kernels.

#include <oasr/activation.cuh>

#include "tvm_ffi_utils.h"

using namespace oasr;

// =============================================================================
// GLU launcher
// =============================================================================

void glu(TensorView output, TensorView input) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(input);
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(output);

    unsigned int batch_size = input.size(0);
    unsigned int seq_len = input.size(1);
    unsigned int channels = input.size(2) / 2;

    cudaStream_t stream = get_stream(input.device());

    DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(input.dtype(), c_type, [&] {
        cudaError_t status = activation::GLU<c_type>(
            static_cast<const c_type*>(input.data_ptr()),
            static_cast<c_type*>(output.data_ptr()),
            batch_size, seq_len, channels, stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "GLU kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}

// =============================================================================
// Swish launcher
// =============================================================================

void swish(TensorView output, TensorView input) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(input);
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(output);

    unsigned int batch_size = input.size(0);
    unsigned int seq_len = input.size(1);
    unsigned int channels = input.size(2);

    cudaStream_t stream = get_stream(input.device());

    DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(input.dtype(), c_type, [&] {
        cudaError_t status = activation::Swish<c_type>(
            static_cast<const c_type*>(input.data_ptr()),
            static_cast<c_type*>(output.data_ptr()),
            batch_size, seq_len, channels, stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "Swish kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}
