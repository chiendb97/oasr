// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// TVM-FFI launcher layer for normalization kernels.

#include <oasr/norm.cuh>

#include "tvm_ffi_utils.h"

using namespace oasr;

namespace {

bool same_dtype(const TensorView& lhs, const TensorView& rhs) {
    return lhs.dtype().code == rhs.dtype().code && lhs.dtype().bits == rhs.dtype().bits &&
           lhs.dtype().lanes == rhs.dtype().lanes;
}

void check_add_norm_inputs(TensorView output, TensorView input, TensorView residual,
                           TensorView weight, Optional bias_opt) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_INPUT(residual);
    CHECK_INPUT(weight);
    CHECK_DEVICE(input, output);
    CHECK_DEVICE(input, residual);
    CHECK_DEVICE(input, weight);
    TVM_FFI_ICHECK(input.ndim() >= 1) << "Norm input must have at least one dimension";
    CHECK_ROW_DENSE_INPUT(input);
    CHECK_SAME_LAYOUT(input, output);
    CHECK_SAME_LAYOUT(input, residual);
    TVM_FFI_ICHECK(same_dtype(input, output) && same_dtype(input, residual) &&
                   same_dtype(input, weight))
        << "Input, residual, weight, and output must have the same dtype";

    int64_t hidden_size = input.size(input.ndim() - 1);
    TVM_FFI_ICHECK(hidden_size > 0) << "Norm hidden dimension must be positive";
    TVM_FFI_ICHECK(weight.ndim() == 1 && weight.size(0) == hidden_size)
        << "Norm weight must be 1-D with length equal to the input's last dimension";
    CHECK_CONTIGUOUS_INPUT(weight);

    if (bias_opt.has_value()) {
        TensorView bias = bias_opt.value();
        CHECK_INPUT(bias);
        CHECK_DEVICE(input, bias);
        CHECK_CONTIGUOUS_INPUT(bias);
        TVM_FFI_ICHECK(same_dtype(input, bias)) << "Norm bias must have the input dtype";
        TVM_FFI_ICHECK(bias.ndim() == 1 && bias.size(0) == hidden_size)
            << "Norm bias must be 1-D with length equal to the input's last dimension";
    }
}

// The quant-out norms write an 8-bit tensor (int8, or float8_e4m3fn) and a
// per-row scale in place of a T output.  Everything is required contiguous (not
// merely row-dense): the 8-bit rows are walked in the input's memory order, so a
// permuted input would land its rows at the wrong logical positions of a
// contiguous ``out_q``.  Returns the number of rows.
int64_t check_norm_quant_args(const char* name, TensorView out_q, TensorView scale,
                              TensorView input, TensorView weight, Optional bias_opt) {
    CHECK_INPUT(input);
    CHECK_INPUT(out_q);
    CHECK_INPUT(scale);
    CHECK_INPUT(weight);
    CHECK_DEVICE(input, out_q);
    CHECK_DEVICE(input, scale);
    CHECK_DEVICE(input, weight);
    TVM_FFI_ICHECK(input.ndim() >= 1) << name << ": input must have at least one dimension";
    CHECK_CONTIGUOUS_INPUT(input);
    CHECK_CONTIGUOUS_INPUT(out_q);
    CHECK_CONTIGUOUS_INPUT(scale);
    const DLDataType dt = input.dtype();
    TVM_FFI_ICHECK((dt.code == kDLFloat || dt.code == kDLBfloat) && dt.bits == 16 && dt.lanes == 1)
        << name << ": input must be float16 or bfloat16";
    TVM_FFI_ICHECK((out_q.dtype().code == kDLInt || out_q.dtype().code == kDLFloat8_e4m3fn) &&
                   out_q.dtype().bits == 8 && out_q.dtype().lanes == 1)
        << name << ": out must be int8 or float8_e4m3fn";
    TVM_FFI_ICHECK(scale.dtype().code == kDLFloat && scale.dtype().bits == 32)
        << name << ": scale must be float32";

    const int64_t hidden_size = input.size(input.ndim() - 1);
    TVM_FFI_ICHECK(hidden_size > 0) << name << ": the hidden dimension must be positive";
    TVM_FFI_ICHECK(out_q.numel() == input.numel() && out_q.size(out_q.ndim() - 1) == hidden_size)
        << name << ": out must be shaped like input (one byte per input element)";
    const int64_t num_rows = FLATTENED_ROWS(input);
    TVM_FFI_ICHECK(scale.numel() == num_rows)
        << name << ": scale needs one float per row (" << num_rows << "), got " << scale.numel();

    TVM_FFI_ICHECK(same_dtype(input, weight)) << name << ": weight must have the input dtype";
    TVM_FFI_ICHECK(weight.ndim() == 1 && weight.size(0) == hidden_size)
        << name << ": weight must be 1-D with length equal to the input's last dimension";
    CHECK_CONTIGUOUS_INPUT(weight);
    if (bias_opt.has_value()) {
        TensorView bias = bias_opt.value();
        CHECK_INPUT(bias);
        CHECK_DEVICE(input, bias);
        CHECK_CONTIGUOUS_INPUT(bias);
        TVM_FFI_ICHECK(same_dtype(input, bias)) << name << ": bias must have the input dtype";
        TVM_FFI_ICHECK(bias.ndim() == 1 && bias.size(0) == hidden_size)
            << name << ": bias must be 1-D with length equal to the input's last dimension";
    }
    return num_rows;
}

}  // namespace

// =============================================================================
// LayerNorm launcher
// =============================================================================

void layernorm(TensorView output, TensorView input, TensorView weight, Optional bias_opt,
               double eps) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_INPUT(weight);
    CHECK_ROW_DENSE_INPUT(input);
    CHECK_SAME_LAYOUT(input, output);

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
        cudaError_t status =
            norm::LayerNorm<c_type>(static_cast<const c_type*>(input.data_ptr()),
                                    static_cast<const c_type*>(weight.data_ptr()), bias_ptr,
                                    static_cast<c_type*>(output.data_ptr()), num_rows, hidden_size,
                                    static_cast<float>(eps), stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "LayerNorm kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}

// =============================================================================
// BiasNorm launcher (Zipformer)
// =============================================================================

void bias_norm(TensorView output, TensorView input, TensorView bias, TensorView log_scale) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_INPUT(bias);
    CHECK_ROW_DENSE_INPUT(input);
    CHECK_SAME_LAYOUT(input, output);

    unsigned int num_rows = 1;
    for (int i = 0; i < input.ndim() - 1; ++i) {
        num_rows *= input.size(i);
    }
    unsigned int hidden_size = input.size(input.ndim() - 1);

    cudaStream_t stream = get_stream(input.device());

    DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(input.dtype(), c_type, [&] {
        cudaError_t status = norm::BiasNorm<c_type>(
            static_cast<const c_type*>(input.data_ptr()),
            static_cast<const c_type*>(bias.data_ptr()),
            static_cast<const c_type*>(log_scale.data_ptr()),
            static_cast<c_type*>(output.data_ptr()), num_rows, hidden_size, stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "BiasNorm kernel failed: " << cudaGetErrorString(status);
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
    CHECK_ROW_DENSE_INPUT(input);
    CHECK_SAME_LAYOUT(input, output);

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
        cudaError_t status =
            norm::RMSNorm<c_type>(static_cast<const c_type*>(input.data_ptr()),
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
        cudaError_t status =
            norm::BatchNorm1D<c_type>(static_cast<const c_type*>(input.data_ptr()),
                                      static_cast<const c_type*>(weight.data_ptr()),
                                      static_cast<const c_type*>(bias.data_ptr()),
                                      static_cast<const c_type*>(running_mean.data_ptr()),
                                      static_cast<const c_type*>(running_var.data_ptr()),
                                      static_cast<c_type*>(output.data_ptr()), batch_size, seq_len,
                                      channels, static_cast<float>(eps), stream);
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
            static_cast<const c_type*>(bias.data_ptr()), static_cast<c_type*>(output.data_ptr()),
            batch_size, seq_len, channels, static_cast<unsigned int>(num_groups),
            static_cast<float>(eps), stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "GroupNorm kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}

// =============================================================================
// AddLayerNorm launcher
// =============================================================================

void addlayernorm(TensorView output, TensorView input, TensorView residual, TensorView weight,
                  Optional bias_opt, double eps, double alpha) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_INPUT(residual);
    CHECK_INPUT(weight);
    CHECK_ROW_DENSE_INPUT(input);
    CHECK_SAME_LAYOUT(input, output);

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
            static_cast<c_type*>(output.data_ptr()), num_rows, hidden_size, static_cast<float>(eps),
            static_cast<float>(alpha), stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "AddLayerNorm kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}

// =============================================================================
// AddLayerNormResidual launcher
// =============================================================================

void addlayernorm_residual(TensorView output, TensorView residual_out, TensorView input,
                           TensorView residual, TensorView weight, Optional bias_opt, double eps,
                           double alpha) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_INPUT(residual);
    CHECK_INPUT(residual_out);
    CHECK_INPUT(weight);
    CHECK_ROW_DENSE_INPUT(input);
    CHECK_SAME_LAYOUT(input, output);
    // This kernel reads `residual` and writes `residual_out` row-wise too, so
    // both share the input's layout.
    CHECK_SAME_LAYOUT(input, residual);
    CHECK_SAME_LAYOUT(input, residual_out);

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
        cudaError_t status = norm::AddLayerNormResidual<c_type>(
            static_cast<const c_type*>(input.data_ptr()),
            static_cast<const c_type*>(residual.data_ptr()),
            static_cast<const c_type*>(weight.data_ptr()), bias_ptr,
            static_cast<c_type*>(output.data_ptr()), static_cast<c_type*>(residual_out.data_ptr()),
            num_rows, hidden_size, static_cast<float>(eps), static_cast<float>(alpha), stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "AddLayerNormResidual kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}

// =============================================================================
// LayerNorm + per-token 8-bit launchers (the output tensor's dtype is the format)
// =============================================================================

void layernorm_quant(TensorView out_q, TensorView scale, TensorView input, TensorView weight,
                     Optional bias_opt, double eps, double qeps) {
    const int64_t num_rows =
        check_norm_quant_args("layer_norm_quant", out_q, scale, input, weight, bias_opt);
    const unsigned int hidden_size = static_cast<unsigned int>(input.size(input.ndim() - 1));
    cudaStream_t stream = get_stream(input.device());

    DISPATCH_DLPACK_HALF_DTYPE(input.dtype(), c_type, [&] {
        cudaError_t status = norm::LayerNormQuant<c_type>(
            static_cast<const c_type*>(input.data_ptr()),
            static_cast<const c_type*>(weight.data_ptr()), OptionalDataPtr<const c_type>(bias_opt),
            out_q.data_ptr(), static_cast<float*>(scale.data_ptr()),
            static_cast<unsigned int>(num_rows), hidden_size, static_cast<float>(eps),
            static_cast<float>(qeps), out_q.dtype().code == kDLFloat8_e4m3fn, stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "LayerNormQuant kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}

void addlayernorm_residual_quant(TensorView out_q, TensorView scale, TensorView residual_out,
                                 TensorView input, TensorView residual, TensorView weight,
                                 Optional bias_opt, double eps, double alpha, double qeps) {
    const char* name = "add_layer_norm_residual_quant";
    const int64_t num_rows = check_norm_quant_args(name, out_q, scale, input, weight, bias_opt);
    CHECK_INPUT(residual);
    CHECK_INPUT(residual_out);
    CHECK_DEVICE(input, residual);
    CHECK_DEVICE(input, residual_out);
    CHECK_CONTIGUOUS_INPUT(residual);
    CHECK_CONTIGUOUS_INPUT(residual_out);
    CHECK_SAME_LAYOUT(input, residual);
    CHECK_SAME_LAYOUT(input, residual_out);
    TVM_FFI_ICHECK(same_dtype(input, residual) && same_dtype(input, residual_out))
        << name << ": residual and residual_out must have the input dtype";
    const unsigned int hidden_size = static_cast<unsigned int>(input.size(input.ndim() - 1));
    cudaStream_t stream = get_stream(input.device());

    DISPATCH_DLPACK_HALF_DTYPE(input.dtype(), c_type, [&] {
        cudaError_t status = norm::AddLayerNormResidualQuant<c_type>(
            static_cast<const c_type*>(input.data_ptr()),
            static_cast<const c_type*>(residual.data_ptr()),
            static_cast<const c_type*>(weight.data_ptr()), OptionalDataPtr<const c_type>(bias_opt),
            out_q.data_ptr(), static_cast<float*>(scale.data_ptr()),
            static_cast<c_type*>(residual_out.data_ptr()), static_cast<unsigned int>(num_rows),
            hidden_size, static_cast<float>(eps), static_cast<float>(alpha),
            static_cast<float>(qeps), out_q.dtype().code == kDLFloat8_e4m3fn, stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "AddLayerNormResidualQuant kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}

// =============================================================================
// AddRMSNorm launchers
// =============================================================================

void addrmsnorm(TensorView output, TensorView input, TensorView residual, TensorView weight,
                Optional bias_opt, double eps, double alpha) {
    check_add_norm_inputs(output, input, residual, weight, bias_opt);

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
        cudaError_t status =
            norm::AddRMSNorm<c_type>(static_cast<const c_type*>(input.data_ptr()),
                                     static_cast<const c_type*>(residual.data_ptr()),
                                     static_cast<const c_type*>(weight.data_ptr()), bias_ptr,
                                     static_cast<c_type*>(output.data_ptr()), num_rows, hidden_size,
                                     static_cast<float>(eps), static_cast<float>(alpha), stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "AddRMSNorm kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}

void addrmsnorm_residual(TensorView output, TensorView residual_out, TensorView input,
                         TensorView residual, TensorView weight, Optional bias_opt, double eps,
                         double alpha) {
    check_add_norm_inputs(output, input, residual, weight, bias_opt);
    CHECK_INPUT(residual_out);
    CHECK_DEVICE(input, residual_out);
    CHECK_SAME_LAYOUT(input, residual_out);
    TVM_FFI_ICHECK(same_dtype(input, residual_out)) << "Residual output must have the input dtype";

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
        cudaError_t status = norm::AddRMSNormResidual<c_type>(
            static_cast<const c_type*>(input.data_ptr()),
            static_cast<const c_type*>(residual.data_ptr()),
            static_cast<const c_type*>(weight.data_ptr()), bias_ptr,
            static_cast<c_type*>(output.data_ptr()), static_cast<c_type*>(residual_out.data_ptr()),
            num_rows, hidden_size, static_cast<float>(eps), static_cast<float>(alpha), stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "AddRMSNormResidual kernel failed: " << cudaGetErrorString(status);
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
        cudaError_t status =
            norm::CMVN<c_type>(static_cast<const c_type*>(input.data_ptr()),
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

void layernorm_activation(TensorView output, TensorView input, TensorView weight, Optional bias_opt,
                          double eps, int64_t activation_type) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_INPUT(weight);
    CHECK_ROW_DENSE_INPUT(input);
    CHECK_SAME_LAYOUT(input, output);

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
            static_cast<c_type*>(output.data_ptr()), num_rows, hidden_size, static_cast<float>(eps),
            activation, stream);
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
    CHECK_ROW_DENSE_INPUT(input);
    CHECK_SAME_LAYOUT(input, output);

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
            static_cast<c_type*>(output.data_ptr()), num_rows, hidden_size, static_cast<float>(eps),
            activation, stream);
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
        cudaError_t status =
            norm::BatchNormSwish<c_type>(static_cast<const c_type*>(input.data_ptr()),
                                         static_cast<const c_type*>(weight.data_ptr()),
                                         static_cast<const c_type*>(bias.data_ptr()),
                                         static_cast<const c_type*>(running_mean.data_ptr()),
                                         static_cast<const c_type*>(running_var.data_ptr()),
                                         static_cast<c_type*>(output.data_ptr()), batch_size,
                                         seq_len, channels, static_cast<float>(eps), stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "BatchNormSwish kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}
