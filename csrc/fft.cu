// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// TVM-FFI launcher layer for the FFT kernel family.
//
// Two entry points:
//   rfft(output, input, n_fft)        -- output is float32 (..., n_fft/2+1, 2)
//                                        viewed as complex by the Python wrapper.
//   rfft_power(output, input, n_fft)  -- output is float32 (..., n_fft/2+1) holding
//                                        the power spectrum |X[k]|^2.

#include <oasr/fft.cuh>

#include "tvm_ffi_utils.h"

using namespace oasr;

namespace {

inline int compute_batch_size(const TensorView& input) {
    int batch = 1;
    for (int i = 0; i < input.ndim() - 1; ++i) {
        batch *= input.size(i);
    }
    return batch;
}

inline void check_input_dtype_fp32(const TensorView& input) {
    TVM_FFI_ICHECK(input.dtype().code == kDLFloat && input.dtype().bits == 32)
        << "FFT input must be float32";
}

inline void check_n_fft(int n_fft) {
    TVM_FFI_ICHECK(n_fft >= 8 && n_fft <= 2048 && (n_fft & (n_fft - 1)) == 0)
        << "n_fft must be a power of two in [8, 2048], got " << n_fft;
}

}  // namespace

void rfft(TensorView output, TensorView input, int64_t n_fft) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_DEVICE(input, output);
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(input);
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(output);
    check_input_dtype_fp32(input);
    TVM_FFI_ICHECK(output.dtype().code == kDLFloat && output.dtype().bits == 32)
        << "FFT complex output must be float32 (interleaved real/imag, last dim = 2)";

    const int n = static_cast<int>(n_fft);
    check_n_fft(n);

    TVM_FFI_ICHECK(input.size(input.ndim() - 1) == n)
        << "Input last dim (" << input.size(input.ndim() - 1) << ") must equal n_fft (" << n
        << ")";
    TVM_FFI_ICHECK(output.ndim() == input.ndim() + 1)
        << "Output ndim must be input ndim + 1 (real/imag last axis), got " << output.ndim();
    TVM_FFI_ICHECK(output.size(output.ndim() - 1) == 2)
        << "Output last dim must be 2 (interleaved real/imag), got "
        << output.size(output.ndim() - 1);
    TVM_FFI_ICHECK(output.size(output.ndim() - 2) == n / 2 + 1)
        << "Output frequency-bin dim must be n_fft/2+1 = " << (n / 2 + 1) << ", got "
        << output.size(output.ndim() - 2);

    const int batch = compute_batch_size(input);
    const int input_stride = n;
    const int output_stride = n / 2 + 1;  // in float2 units

    cudaStream_t stream = get_stream(input.device());

    cudaError_t status = fft::Rfft(static_cast<const float*>(input.data_ptr()),
                                   static_cast<float2*>(output.data_ptr()), n, batch,
                                   input_stride, output_stride, stream);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "rfft kernel failed: " << cudaGetErrorString(status);
}

void rfft_power(TensorView output, TensorView input, int64_t n_fft) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_DEVICE(input, output);
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(input);
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(output);
    check_input_dtype_fp32(input);
    TVM_FFI_ICHECK(output.dtype().code == kDLFloat && output.dtype().bits == 32)
        << "FFT power output must be float32";

    const int n = static_cast<int>(n_fft);
    check_n_fft(n);

    TVM_FFI_ICHECK(input.size(input.ndim() - 1) == n)
        << "Input last dim (" << input.size(input.ndim() - 1) << ") must equal n_fft (" << n
        << ")";
    TVM_FFI_ICHECK(output.ndim() == input.ndim())
        << "Power output ndim must equal input ndim, got " << output.ndim();
    TVM_FFI_ICHECK(output.size(output.ndim() - 1) == n / 2 + 1)
        << "Power output last dim must be n_fft/2+1 = " << (n / 2 + 1) << ", got "
        << output.size(output.ndim() - 1);

    const int batch = compute_batch_size(input);
    const int input_stride = n;
    const int output_stride = n / 2 + 1;

    cudaStream_t stream = get_stream(input.device());

    cudaError_t status = fft::RfftPower(static_cast<const float*>(input.data_ptr()),
                                        static_cast<float*>(output.data_ptr()), n, batch,
                                        input_stride, output_stride, stream);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "rfft_power kernel failed: " << cudaGetErrorString(status);
}
