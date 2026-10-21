// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// TVM-FFI launchers for activation quantization.

#include <oasr/quant/quantize.cuh>

#include "tvm_ffi_utils.h"

using namespace oasr;

// =============================================================================
// Per-token int8 quantization
// =============================================================================

void quantize_per_token(TensorView output, TensorView scale, TensorView input, Optional smooth_inv,
                        double eps) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_INPUT(scale);
    CHECK_CONTIGUOUS_INPUT(input);
    CHECK_CONTIGUOUS_INPUT(output);
    CHECK_CONTIGUOUS_INPUT(scale);
    TVM_FFI_ICHECK(output.dtype().code == kDLInt && output.dtype().bits == 8)
        << "quantize_per_token: output must be int8";
    TVM_FFI_ICHECK(scale.dtype().code == kDLFloat && scale.dtype().bits == 32)
        << "quantize_per_token: scale must be float32";
    TVM_FFI_ICHECK(output.numel() == input.numel())
        << "quantize_per_token: output must hold one int8 per input element";

    const int hidden_size = static_cast<int>(input.size(input.ndim() - 1));
    const int64_t num_rows = FLATTENED_ROWS(input);
    TVM_FFI_ICHECK(scale.numel() == num_rows)
        << "quantize_per_token: scale needs one float per row (" << num_rows << "), got "
        << scale.numel();

    const float* smooth_ptr = nullptr;
    if (smooth_inv.has_value()) {
        TensorView s = smooth_inv.value();
        CHECK_INPUT(s);
        CHECK_CONTIGUOUS_INPUT(s);
        TVM_FFI_ICHECK(s.dtype().code == kDLFloat && s.dtype().bits == 32)
            << "quantize_per_token: smooth_inv must be float32";
        TVM_FFI_ICHECK(s.numel() == hidden_size)
            << "quantize_per_token: smooth_inv needs one float per channel (" << hidden_size
            << "), got " << s.numel();
        smooth_ptr = static_cast<const float*>(s.data_ptr());
    }

    cudaStream_t stream = get_stream(input.device());

    DISPATCH_DLPACK_HALF_DTYPE(input.dtype(), c_type, [&] {
        cudaError_t status = quant::QuantizePerToken<c_type>(
            static_cast<const c_type*>(input.data_ptr()), smooth_ptr,
            static_cast<int8_t*>(output.data_ptr()), static_cast<float*>(scale.data_ptr()),
            num_rows, hidden_size, static_cast<float>(eps), stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "QuantizePerToken kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}

// =============================================================================
// Static int8 quantization (calibrated per-tensor scale, per-channel multiplier)
// =============================================================================

void quantize_static(TensorView output, TensorView input, TensorView mul) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_INPUT(mul);
    CHECK_CONTIGUOUS_INPUT(input);
    CHECK_CONTIGUOUS_INPUT(output);
    CHECK_CONTIGUOUS_INPUT(mul);
    TVM_FFI_ICHECK(output.dtype().code == kDLInt && output.dtype().bits == 8)
        << "quantize_static: output must be int8";
    TVM_FFI_ICHECK(mul.dtype().code == kDLFloat && mul.dtype().bits == 32)
        << "quantize_static: mul must be float32";
    TVM_FFI_ICHECK(output.numel() == input.numel())
        << "quantize_static: output must hold one int8 per input element";

    const int hidden_size = static_cast<int>(input.size(input.ndim() - 1));
    const int64_t num_rows = FLATTENED_ROWS(input);
    TVM_FFI_ICHECK(mul.numel() == hidden_size)
        << "quantize_static: mul needs one float per channel (" << hidden_size << "), got "
        << mul.numel();

    cudaStream_t stream = get_stream(input.device());

    DISPATCH_DLPACK_HALF_DTYPE(input.dtype(), c_type, [&] {
        cudaError_t status = quant::QuantizeStatic<c_type>(
            static_cast<const c_type*>(input.data_ptr()), static_cast<const float*>(mul.data_ptr()),
            static_cast<int8_t*>(output.data_ptr()), num_rows, hidden_size, stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "QuantizeStatic kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}
