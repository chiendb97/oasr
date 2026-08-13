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
        cudaError_t status = activation::GLU<c_type>(static_cast<const c_type*>(input.data_ptr()),
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
        cudaError_t status = activation::Swish<c_type>(static_cast<const c_type*>(input.data_ptr()),
                                                       static_cast<c_type*>(output.data_ptr()),
                                                       batch_size, seq_len, channels, stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "Swish kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}

// =============================================================================
// Unary activation launchers (elementwise; input/output must be contiguous)
// =============================================================================

template <typename Activation>
void elementwise_activation(TensorView output, TensorView input, Activation activation_op,
                            const char* op_name) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_CONTIGUOUS_INPUT(output);
    CHECK_DEVICE(input, output);
    const bool regular_rows = input.ndim() == 0 || [&] {
        if (input.stride(input.ndim() - 1) != 1)
            return false;
        for (int i = 0; i < input.ndim() - 2; ++i) {
            if (input.stride(i) != input.size(i + 1) * input.stride(i + 1))
                return false;
        }
        return true;
    }();
    TVM_FFI_ICHECK(regular_rows)
        << op_name << " input must be contiguous or have regularly strided contiguous rows";
    TVM_FFI_ICHECK(input.ndim() == output.ndim() &&
                   [&] {
                       for (int i = 0; i < input.ndim(); ++i) {
                           if (input.size(i) != output.size(i))
                               return false;
                       }
                       return true;
                   }())
        << op_name << " input and output must have the same shape";
    TVM_FFI_ICHECK(input.dtype().code == output.dtype().code &&
                   input.dtype().bits == output.dtype().bits &&
                   input.dtype().lanes == output.dtype().lanes)
        << op_name << " input and output must have the same dtype";

    int64_t n = 1;
    for (int i = 0; i < input.ndim(); ++i) {
        n *= input.size(i);
    }

    cudaStream_t stream = get_stream(input.device());
    const int64_t columns = input.ndim() == 0 ? 1 : input.size(input.ndim() - 1);
    const int64_t rows = columns == 0 ? 0 : n / columns;
    const int64_t input_row_stride = input.ndim() < 2 ? columns : input.stride(input.ndim() - 2);

    DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(input.dtype(), c_type, [&] {
        cudaError_t status;
        if (input.IsContiguous()) {
            status = activation::Elementwise<c_type>(static_cast<const c_type*>(input.data_ptr()),
                                                     static_cast<c_type*>(output.data_ptr()), n,
                                                     activation_op, stream);
        } else {
            status = activation::ElementwiseStridedRows<c_type>(
                static_cast<const c_type*>(input.data_ptr()),
                static_cast<c_type*>(output.data_ptr()), rows, columns, input_row_stride,
                activation_op, stream);
        }
        TVM_FFI_ICHECK(status == cudaSuccess)
            << op_name << " kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}

void gelu_erf(TensorView output, TensorView input) {
    elementwise_activation(output, input, GeluErfActivation{}, "GeluErf");
}

void sigmoid(TensorView output, TensorView input) {
    elementwise_activation(output, input, SigmoidActivation{}, "Sigmoid");
}

void tanh_activation(TensorView output, TensorView input) {
    elementwise_activation(output, input, TanhActivation{}, "Tanh");
}

void relu(TensorView output, TensorView input) {
    elementwise_activation(output, input, ReluActivation{}, "Relu");
}

// =============================================================================
// Swoosh-L launcher (elementwise; input/output must be contiguous)
// =============================================================================

void swoosh_l(TensorView output, TensorView input) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);

    int64_t n = 1;
    for (int i = 0; i < input.ndim(); ++i) {
        n *= input.size(i);
    }

    cudaStream_t stream = get_stream(input.device());

    DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(input.dtype(), c_type, [&] {
        cudaError_t status = activation::SwooshL<c_type>(
            static_cast<const c_type*>(input.data_ptr()), static_cast<c_type*>(output.data_ptr()),
            static_cast<int>(n), stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "SwooshL kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}

// =============================================================================
// Swoosh-R launcher (elementwise; input/output must be contiguous)
// =============================================================================

void swoosh_r(TensorView output, TensorView input) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);

    int64_t n = 1;
    for (int i = 0; i < input.ndim(); ++i) {
        n *= input.size(i);
    }

    cudaStream_t stream = get_stream(input.device());

    DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(input.dtype(), c_type, [&] {
        cudaError_t status = activation::SwooshR<c_type>(
            static_cast<const c_type*>(input.data_ptr()), static_cast<c_type*>(output.data_ptr()),
            static_cast<int>(n), stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "SwooshR kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}
