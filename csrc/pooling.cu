// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// TVM-FFI launcher layer for pooling kernels.

#include <cstdint>
#include <limits>
#include <oasr/pooling.cuh>

#include "tvm_ffi_utils.h"

using namespace oasr;

namespace {

int64_t floor_div(int64_t numerator, int64_t denominator) {
    if (numerator >= 0)
        return numerator / denominator;
    return -((-numerator + denominator - 1) / denominator);
}

int64_t pool_output_length(int64_t input_length, int64_t kernel_size, int64_t stride,
                           int64_t padding, bool ceil_mode) {
    int64_t numerator = input_length + 2 * padding - kernel_size;
    if (ceil_mode)
        numerator += stride - 1;
    int64_t output_length = floor_div(numerator, stride) + 1;
    if (ceil_mode && output_length > 0 && (output_length - 1) * stride >= input_length + padding) {
        --output_length;
    }
    return output_length;
}

bool same_dtype(const TensorView& a, const TensorView& b) {
    return a.dtype().code == b.dtype().code && a.dtype().bits == b.dtype().bits &&
           a.dtype().lanes == b.dtype().lanes;
}

}  // namespace

void avg_pool1d(TensorView output, TensorView input, int64_t kernel_size, int64_t stride,
                int64_t padding, bool ceil_mode, bool count_include_pad) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_DEVICE(input, output);
    CHECK_CONTIGUOUS_INPUT(input);
    CHECK_CONTIGUOUS_INPUT(output);

    TVM_FFI_ICHECK(input.ndim() == 2 || input.ndim() == 3)
        << "AvgPool1D expects TC or BTC input, got " << input.ndim() << " dimensions";
    TVM_FFI_ICHECK(output.ndim() == input.ndim())
        << "AvgPool1D output must have the same rank as input";
    TVM_FFI_ICHECK(kernel_size > 0) << "kernel_size must be positive, got " << kernel_size;
    TVM_FFI_ICHECK(stride > 0) << "stride must be positive, got " << stride;
    TVM_FFI_ICHECK(padding >= 0 && padding <= kernel_size / 2)
        << "padding must be non-negative and at most half of kernel_size, got padding=" << padding
        << " kernel_size=" << kernel_size;
    TVM_FFI_ICHECK(kernel_size <= std::numeric_limits<int>::max() &&
                   stride <= std::numeric_limits<int>::max() &&
                   padding <= std::numeric_limits<int>::max())
        << "AvgPool1D parameters exceed the CUDA kernel's int32 indexing range";
    TVM_FFI_ICHECK(same_dtype(input, output)) << "AvgPool1D input and output dtypes must match";

    const int time_dim = input.ndim() - 2;
    const int channel_dim = input.ndim() - 1;
    const int64_t batch_size = input.ndim() == 3 ? input.size(0) : 1;
    const int64_t input_length = input.size(time_dim);
    const int64_t channels = input.size(channel_dim);
    TVM_FFI_ICHECK(input_length > 0 && channels > 0)
        << "AvgPool1D time and channel dimensions must be positive, got T=" << input_length
        << " C=" << channels;

    const int64_t output_length =
        pool_output_length(input_length, kernel_size, stride, padding, ceil_mode);
    TVM_FFI_ICHECK(output_length > 0)
        << "AvgPool1D produces an invalid output length " << output_length
        << " from T=" << input_length << " kernel_size=" << kernel_size << " stride=" << stride
        << " padding=" << padding << " ceil_mode=" << ceil_mode;
    TVM_FFI_ICHECK(output.size(time_dim) == output_length && output.size(channel_dim) == channels)
        << "AvgPool1D output must have T=" << output_length << " C=" << channels;
    if (input.ndim() == 3) {
        TVM_FFI_ICHECK(output.size(0) == batch_size)
            << "AvgPool1D output batch must equal input batch " << batch_size;
    }

    TVM_FFI_ICHECK(batch_size <= 65535 && output_length <= 65535 &&
                   input_length <= std::numeric_limits<int>::max() &&
                   channels <= std::numeric_limits<int>::max())
        << "AvgPool1D shape exceeds CUDA grid/index limits: B=" << batch_size
        << " T=" << input_length << " T_out=" << output_length << " C=" << channels;

    cudaStream_t stream = get_stream(input.device());
    DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(input.dtype(), c_type, [&] {
        cudaError_t status = pooling::AvgPool1d<c_type>(
            static_cast<const c_type*>(input.data_ptr()), static_cast<c_type*>(output.data_ptr()),
            static_cast<int>(batch_size), static_cast<int>(input_length),
            static_cast<int>(output_length), static_cast<int>(channels),
            static_cast<int>(kernel_size), static_cast<int>(stride), static_cast<int>(padding),
            count_include_pad, stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "AvgPool1D kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}

void max_pool1d(TensorView output, TensorView input, int64_t kernel_size, int64_t stride,
                int64_t padding, bool ceil_mode) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_DEVICE(input, output);
    CHECK_CONTIGUOUS_INPUT(input);
    CHECK_CONTIGUOUS_INPUT(output);

    TVM_FFI_ICHECK(input.ndim() == 2 || input.ndim() == 3)
        << "MaxPool1D expects TC or BTC input, got " << input.ndim() << " dimensions";
    TVM_FFI_ICHECK(output.ndim() == input.ndim())
        << "MaxPool1D output must have the same rank as input";
    TVM_FFI_ICHECK(kernel_size > 0) << "kernel_size must be positive, got " << kernel_size;
    TVM_FFI_ICHECK(stride > 0) << "stride must be positive, got " << stride;
    // Same bound as AvgPool1D, and here it is load-bearing rather than merely
    // matching PyTorch: it is what guarantees every window overlaps at least
    // one real element, so the max reduction is never over pure -inf padding.
    TVM_FFI_ICHECK(padding >= 0 && padding <= kernel_size / 2)
        << "padding must be non-negative and at most half of kernel_size, got padding=" << padding
        << " kernel_size=" << kernel_size;
    TVM_FFI_ICHECK(kernel_size <= std::numeric_limits<int>::max() &&
                   stride <= std::numeric_limits<int>::max() &&
                   padding <= std::numeric_limits<int>::max())
        << "MaxPool1D parameters exceed the CUDA kernel's int32 indexing range";
    TVM_FFI_ICHECK(same_dtype(input, output)) << "MaxPool1D input and output dtypes must match";

    const int time_dim = input.ndim() - 2;
    const int channel_dim = input.ndim() - 1;
    const int64_t batch_size = input.ndim() == 3 ? input.size(0) : 1;
    const int64_t input_length = input.size(time_dim);
    const int64_t channels = input.size(channel_dim);
    TVM_FFI_ICHECK(input_length > 0 && channels > 0)
        << "MaxPool1D time and channel dimensions must be positive, got T=" << input_length
        << " C=" << channels;

    const int64_t output_length =
        pool_output_length(input_length, kernel_size, stride, padding, ceil_mode);
    TVM_FFI_ICHECK(output_length > 0)
        << "MaxPool1D produces an invalid output length " << output_length
        << " from T=" << input_length << " kernel_size=" << kernel_size << " stride=" << stride
        << " padding=" << padding << " ceil_mode=" << ceil_mode;
    TVM_FFI_ICHECK(output.size(time_dim) == output_length && output.size(channel_dim) == channels)
        << "MaxPool1D output must have T=" << output_length << " C=" << channels;
    if (input.ndim() == 3) {
        TVM_FFI_ICHECK(output.size(0) == batch_size)
            << "MaxPool1D output batch must equal input batch " << batch_size;
    }

    TVM_FFI_ICHECK(batch_size <= 65535 && output_length <= 65535 &&
                   input_length <= std::numeric_limits<int>::max() &&
                   channels <= std::numeric_limits<int>::max())
        << "MaxPool1D shape exceeds CUDA grid/index limits: B=" << batch_size
        << " T=" << input_length << " T_out=" << output_length << " C=" << channels;

    cudaStream_t stream = get_stream(input.device());
    DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(input.dtype(), c_type, [&] {
        cudaError_t status = pooling::MaxPool1d<c_type>(
            static_cast<const c_type*>(input.data_ptr()), static_cast<c_type*>(output.data_ptr()),
            static_cast<int>(batch_size), static_cast<int>(input_length),
            static_cast<int>(output_length), static_cast<int>(channels),
            static_cast<int>(kernel_size), static_cast<int>(stride), static_cast<int>(padding),
            stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "MaxPool1D kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}
