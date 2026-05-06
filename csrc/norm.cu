// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// TVM-FFI launcher layer for normalization kernels.

#include <oasr/norm.cuh>

#include "tvm_ffi_utils.h"

using namespace oasr;

// =============================================================================
// LayerNorm launcher
// =============================================================================

void layernorm(TensorView output, TensorView input, TensorView weight, Optional bias_opt,
               double eps) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_INPUT(weight);
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(input);
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(output);

    unsigned int num_rows = 1;
    for (int i = 0; i < input.ndim() - 1; ++i) {
        num_rows *= input.size(i);
    }
    unsigned int hidden_size = input.size(input.ndim() - 1);

    cudaStream_t stream = get_stream(input.device());

    DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(input.dtype(), c_type, [&] {
        const c_type* bias_ptr = nullptr;
        if (bias_opt.has_value()) {
            bias_ptr = static_cast<const c_type*>(bias_opt.value().data_ptr());
        }
        cudaError_t status = norm::LayerNorm<c_type>(
            static_cast<const c_type*>(input.data_ptr()),
            static_cast<const c_type*>(weight.data_ptr()), bias_ptr,
            static_cast<c_type*>(output.data_ptr()), num_rows, hidden_size,
            static_cast<float>(eps), stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "LayerNorm kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}

// =============================================================================
// RMSNorm launcher
// =============================================================================

void rmsnorm(TensorView output, TensorView input, TensorView weight, Optional bias_opt,
             double eps) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_INPUT(weight);
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(input);
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(output);

    unsigned int num_rows = 1;
    for (int i = 0; i < input.ndim() - 1; ++i) {
        num_rows *= input.size(i);
    }
    unsigned int hidden_size = input.size(input.ndim() - 1);

    cudaStream_t stream = get_stream(input.device());

    DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(input.dtype(), c_type, [&] {
        const c_type* bias_ptr = nullptr;
        if (bias_opt.has_value()) {
            bias_ptr = static_cast<const c_type*>(bias_opt.value().data_ptr());
        }
        cudaError_t status = norm::RMSNorm<c_type>(
            static_cast<const c_type*>(input.data_ptr()),
            static_cast<const c_type*>(weight.data_ptr()), bias_ptr,
            static_cast<c_type*>(output.data_ptr()), num_rows, hidden_size,
            static_cast<float>(eps), stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "RMSNorm kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}

// =============================================================================
// BatchNorm1D launcher
// =============================================================================

void batchnorm1d(TensorView output, TensorView input, TensorView weight, TensorView bias,
                 TensorView running_mean, TensorView running_var, double eps) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_DIM(3, input);

    unsigned int batch_size = input.size(0);
    unsigned int seq_len = input.size(1);
    unsigned int channels = input.size(2);

    cudaStream_t stream = get_stream(input.device());

    DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(input.dtype(), c_type, [&] {
        cudaError_t status = norm::BatchNorm1D<c_type>(
            static_cast<const c_type*>(input.data_ptr()),
            static_cast<const c_type*>(weight.data_ptr()),
            static_cast<const c_type*>(bias.data_ptr()),
            static_cast<const c_type*>(running_mean.data_ptr()),
            static_cast<const c_type*>(running_var.data_ptr()),
            static_cast<c_type*>(output.data_ptr()), batch_size, seq_len, channels,
            static_cast<float>(eps), stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "BatchNorm1D kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}

// =============================================================================
// GroupNorm launcher
// =============================================================================

void groupnorm(TensorView output, TensorView input, TensorView weight, TensorView bias,
               int64_t num_groups, double eps) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_DIM(3, input);

    unsigned int batch_size = input.size(0);
    unsigned int seq_len = input.size(1);
    unsigned int channels = input.size(2);

    cudaStream_t stream = get_stream(input.device());

    DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(input.dtype(), c_type, [&] {
        cudaError_t status = norm::GroupNorm<c_type>(
            static_cast<const c_type*>(input.data_ptr()),
            static_cast<const c_type*>(weight.data_ptr()),
            static_cast<const c_type*>(bias.data_ptr()),
            static_cast<c_type*>(output.data_ptr()), batch_size, seq_len, channels,
            static_cast<unsigned int>(num_groups), static_cast<float>(eps), stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "GroupNorm kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}

// =============================================================================
// AddLayerNorm launcher
// =============================================================================

void addlayernorm(TensorView output, TensorView input, TensorView residual, TensorView weight,
                  Optional bias_opt, double eps) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_INPUT(residual);
    CHECK_INPUT(weight);
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(input);
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(output);

    unsigned int num_rows = 1;
    for (int i = 0; i < input.ndim() - 1; ++i) {
        num_rows *= input.size(i);
    }
    unsigned int hidden_size = input.size(input.ndim() - 1);

    cudaStream_t stream = get_stream(input.device());

    DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(input.dtype(), c_type, [&] {
        const c_type* bias_ptr = nullptr;
        if (bias_opt.has_value()) {
            bias_ptr = static_cast<const c_type*>(bias_opt.value().data_ptr());
        }
        cudaError_t status = norm::AddLayerNorm<c_type>(
            static_cast<const c_type*>(input.data_ptr()),
            static_cast<const c_type*>(residual.data_ptr()),
            static_cast<const c_type*>(weight.data_ptr()), bias_ptr,
            static_cast<c_type*>(output.data_ptr()), num_rows, hidden_size,
            static_cast<float>(eps), stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "AddLayerNorm kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}

// =============================================================================
// CMVN launcher
// =============================================================================

void cmvn(TensorView output, TensorView input, TensorView mean, TensorView istd) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_INPUT(mean);
    CHECK_INPUT(istd);

    unsigned int num_rows = 1;
    for (int i = 0; i < input.ndim() - 1; ++i) {
        num_rows *= input.size(i);
    }
    unsigned int num_cols = input.size(input.ndim() - 1);

    cudaStream_t stream = get_stream(input.device());

    DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(input.dtype(), c_type, [&] {
        cudaError_t status = norm::CMVN<c_type>(
            static_cast<const c_type*>(input.data_ptr()),
            static_cast<const c_type*>(mean.data_ptr()),
            static_cast<const c_type*>(istd.data_ptr()),
            static_cast<c_type*>(output.data_ptr()), num_rows, num_cols, stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "CMVN kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}

// =============================================================================
// Fused norm+activation launchers
// =============================================================================

void layernorm_activation(TensorView output, TensorView input, TensorView weight,
                          Optional bias_opt, double eps, int64_t activation_type) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_INPUT(weight);
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(input);
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(output);

    unsigned int num_rows = 1;
    for (int i = 0; i < input.ndim() - 1; ++i) {
        num_rows *= input.size(i);
    }
    unsigned int hidden_size = input.size(input.ndim() - 1);
    auto activation = static_cast<ActivationType>(activation_type);

    cudaStream_t stream = get_stream(input.device());

    DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(input.dtype(), c_type, [&] {
        const c_type* bias_ptr = nullptr;
        if (bias_opt.has_value()) {
            bias_ptr = static_cast<const c_type*>(bias_opt.value().data_ptr());
        }
        cudaError_t status = norm::LayerNormActivation<c_type>(
            static_cast<const c_type*>(input.data_ptr()),
            static_cast<const c_type*>(weight.data_ptr()), bias_ptr,
            static_cast<c_type*>(output.data_ptr()), num_rows, hidden_size,
            static_cast<float>(eps), activation, stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "LayerNormActivation kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}

void rmsnorm_activation(TensorView output, TensorView input, TensorView weight, Optional bias_opt,
                        double eps, int64_t activation_type) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_INPUT(weight);
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(input);
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(output);

    unsigned int num_rows = 1;
    for (int i = 0; i < input.ndim() - 1; ++i) {
        num_rows *= input.size(i);
    }
    unsigned int hidden_size = input.size(input.ndim() - 1);
    auto activation = static_cast<ActivationType>(activation_type);

    cudaStream_t stream = get_stream(input.device());

    DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(input.dtype(), c_type, [&] {
        const c_type* bias_ptr = nullptr;
        if (bias_opt.has_value()) {
            bias_ptr = static_cast<const c_type*>(bias_opt.value().data_ptr());
        }
        cudaError_t status = norm::RMSNormActivation<c_type>(
            static_cast<const c_type*>(input.data_ptr()),
            static_cast<const c_type*>(weight.data_ptr()), bias_ptr,
            static_cast<c_type*>(output.data_ptr()), num_rows, hidden_size,
            static_cast<float>(eps), activation, stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "RMSNormActivation kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}

void batchnorm_activation(TensorView output, TensorView input, TensorView weight, TensorView bias,
                          TensorView running_mean, TensorView running_var, double eps,
                          int64_t activation_type) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_DIM(3, input);

    unsigned int batch_size = input.size(0);
    unsigned int seq_len = input.size(1);
    unsigned int channels = input.size(2);
    auto activation = static_cast<ActivationType>(activation_type);

    cudaStream_t stream = get_stream(input.device());

    DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(input.dtype(), c_type, [&] {
        cudaError_t status = norm::BatchNormActivation<c_type>(
            static_cast<const c_type*>(input.data_ptr()),
            static_cast<const c_type*>(weight.data_ptr()),
            static_cast<const c_type*>(bias.data_ptr()),
            static_cast<const c_type*>(running_mean.data_ptr()),
            static_cast<const c_type*>(running_var.data_ptr()),
            static_cast<c_type*>(output.data_ptr()), batch_size, seq_len, channels,
            static_cast<float>(eps), activation, stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "BatchNormActivation kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}

void batchnorm_swish(TensorView output, TensorView input, TensorView weight, TensorView bias,
                     TensorView running_mean, TensorView running_var, double eps) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_DIM(3, input);

    unsigned int batch_size = input.size(0);
    unsigned int seq_len = input.size(1);
    unsigned int channels = input.size(2);

    cudaStream_t stream = get_stream(input.device());

    DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(input.dtype(), c_type, [&] {
        cudaError_t status = norm::BatchNormSwish<c_type>(
            static_cast<const c_type*>(input.data_ptr()),
            static_cast<const c_type*>(weight.data_ptr()),
            static_cast<const c_type*>(bias.data_ptr()),
            static_cast<const c_type*>(running_mean.data_ptr()),
            static_cast<const c_type*>(running_var.data_ptr()),
            static_cast<c_type*>(output.data_ptr()), batch_size, seq_len, channels,
            static_cast<float>(eps), stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "BatchNormSwish kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}
