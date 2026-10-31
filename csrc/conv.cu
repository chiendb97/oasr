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

namespace {

bool same_dtype(const TensorView& a, const TensorView& b) {
    return a.dtype().code == b.dtype().code && a.dtype().bits == b.dtype().bits &&
           a.dtype().lanes == b.dtype().lanes;
}

void check_depthwise_conv1d(TensorView output, TensorView input, TensorView weight,
                            Optional bias_opt, int64_t padding_left, int64_t padding_right) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_INPUT(weight);
    CHECK_DIM(3, input);
    CHECK_DIM(3, output);
    CHECK_CONTIGUOUS_INPUT(input);
    CHECK_CONTIGUOUS_INPUT(output);
    CHECK_CONTIGUOUS_INPUT(weight);
    CHECK_DEVICE(input, output);
    CHECK_DEVICE(input, weight);

    TVM_FFI_ICHECK(weight.ndim() == 2 || weight.ndim() == 3)
        << "DepthwiseConv1D weight must be [K, C] or [K, 1, C]";

    int64_t batch_size = input.size(0);
    int64_t seq_len = input.size(1);
    int64_t channels = input.size(2);
    int64_t kernel_size = weight.size(0);
    int64_t out_len = seq_len + padding_left + padding_right - kernel_size + 1;

    TVM_FFI_ICHECK(padding_left >= 0 && padding_right >= 0)
        << "DepthwiseConv1D padding must be non-negative, got (" << padding_left << ", "
        << padding_right << ")";
    TVM_FFI_ICHECK(kernel_size > 0 && out_len > 0)
        << "DepthwiseConv1D has invalid output length " << out_len;
    TVM_FFI_ICHECK(weight.numel() == kernel_size * channels)
        << "DepthwiseConv1D weight/channel mismatch";
    if (weight.ndim() == 3) {
        TVM_FFI_ICHECK(weight.size(1) == 1 && weight.size(2) == channels)
            << "DepthwiseConv1D weight must be [K, 1, C]";
    } else {
        TVM_FFI_ICHECK(weight.size(1) == channels) << "DepthwiseConv1D weight must be [K, C]";
    }
    TVM_FFI_ICHECK(output.size(0) == batch_size && output.size(1) == out_len &&
                   output.size(2) == channels)
        << "DepthwiseConv1D output shape mismatch";
    // The kernels offset whole rows in 64 bits and positions inside one in 32.
    TVM_FFI_ICHECK(seq_len * channels <= 0x7fffffff && out_len * channels <= 0x7fffffff)
        << "DepthwiseConv1D row of " << seq_len * channels << " input / " << out_len * channels
        << " output elements is past the kernels' 32-bit in-row offsets";
    TVM_FFI_ICHECK(same_dtype(input, output) && same_dtype(input, weight))
        << "DepthwiseConv1D input, weight, and output dtypes must match";

    if (bias_opt.has_value()) {
        TensorView bias = bias_opt.value();
        CHECK_INPUT(bias);
        CHECK_DIM(1, bias);
        CHECK_CONTIGUOUS_INPUT(bias);
        CHECK_DEVICE(input, bias);
        TVM_FFI_ICHECK(bias.size(0) == channels) << "DepthwiseConv1D bias shape mismatch";
        TVM_FFI_ICHECK(same_dtype(input, bias)) << "DepthwiseConv1D bias dtype mismatch";
    }
}

}  // namespace

void depthwise_conv1d(TensorView output, TensorView input, TensorView weight, Optional bias_opt,
                      int64_t padding_left, int64_t padding_right, Optional mask_opt,
                      bool add_input) {
    check_depthwise_conv1d(output, input, weight, bias_opt, padding_left, padding_right);

    int batch_size = input.size(0);
    int seq_len = input.size(1);
    int channels = input.size(2);
    int kernel_size = weight.size(0);
    int out_len = output.size(1);

    bool mask_is_bool = false;
    if (mask_opt.has_value()) {
        TensorView mask = mask_opt.value();
        CHECK_INPUT(mask);
        CHECK_DIM(3, mask);
        CHECK_CONTIGUOUS_INPUT(mask);
        CHECK_DEVICE(input, mask);
        TVM_FFI_ICHECK(mask.size(0) == batch_size && mask.size(1) == seq_len && mask.size(2) == 1)
            << "DepthwiseConv1D mask must be [B, T, 1]";
        TVM_FFI_ICHECK(out_len == seq_len)
            << "DepthwiseConv1D masking requires a length-preserving convolution";
        mask_is_bool = mask.dtype().code == kDLBool && mask.dtype().bits == 8;
        TVM_FFI_ICHECK(mask_is_bool || same_dtype(input, mask))
            << "DepthwiseConv1D mask must be bool or have the input dtype";
    }
    TVM_FFI_ICHECK(!add_input || out_len == seq_len)
        << "DepthwiseConv1D add_input requires a length-preserving convolution";

    cudaStream_t stream = get_stream(input.device());

    DISPATCH_DLPACK_HALF_DTYPE(input.dtype(), c_type, [&] {
        const c_type* bias_ptr = nullptr;
        if (bias_opt.has_value()) {
            bias_ptr = static_cast<const c_type*>(bias_opt.value().data_ptr());
        }
        cudaError_t status;
        if (mask_is_bool) {
            status = conv::DepthwiseConv1D<c_type, bool>(
                static_cast<const c_type*>(input.data_ptr()),
                static_cast<const c_type*>(weight.data_ptr()), bias_ptr,
                static_cast<const bool*>(mask_opt.value().data_ptr()),
                static_cast<c_type*>(output.data_ptr()), batch_size, seq_len, channels, kernel_size,
                static_cast<int>(padding_left), static_cast<int>(padding_right), add_input, stream);
        } else {
            const c_type* mask_ptr = nullptr;
            if (mask_opt.has_value()) {
                mask_ptr = static_cast<const c_type*>(mask_opt.value().data_ptr());
            }
            status = conv::DepthwiseConv1D<c_type, c_type>(
                static_cast<const c_type*>(input.data_ptr()),
                static_cast<const c_type*>(weight.data_ptr()), bias_ptr, mask_ptr,
                static_cast<c_type*>(output.data_ptr()), batch_size, seq_len, channels, kernel_size,
                static_cast<int>(padding_left), static_cast<int>(padding_right), add_input, stream);
        }
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "DepthwiseConv1D kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}

// =============================================================================
// Depthwise Conv1D + SiLU launcher
// =============================================================================

void depthwise_conv1d_silu(TensorView output, TensorView input, TensorView weight,
                           Optional bias_opt, int64_t padding_left, int64_t padding_right) {
    check_depthwise_conv1d(output, input, weight, bias_opt, padding_left, padding_right);

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
            static_cast<int>(padding_left), static_cast<int>(padding_right), stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "DepthwiseConv1DSilu kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}

// =============================================================================
// Causal Conv1D launcher
// =============================================================================

namespace {

// Whether two tensors' storage overlaps, by byte range.  Enough for the dense
// tensors the causal convolution takes.
bool overlaps(const TensorView& a, const TensorView& b) {
    const auto begin_a = reinterpret_cast<uintptr_t>(a.data_ptr());
    const auto begin_b = reinterpret_cast<uintptr_t>(b.data_ptr());
    const auto end_a = begin_a + static_cast<uintptr_t>(a.numel()) * get_element_size(a.dtype());
    const auto end_b = begin_b + static_cast<uintptr_t>(b.numel()) * get_element_size(b.dtype());
    return a.numel() > 0 && b.numel() > 0 && begin_a < end_b && begin_b < end_a;
}

}  // namespace

// Streaming causal convolution over one chunk: `output` from [state | input],
// then `state` <- its last kernel_size - 1 positions, in place.
void causal_conv1d(TensorView output, TensorView input, TensorView state, TensorView weight,
                   Optional bias_opt) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_INPUT(state);
    CHECK_INPUT(weight);
    CHECK_DEVICE(input, output);
    CHECK_DEVICE(input, state);
    CHECK_DEVICE(input, weight);
    CHECK_DIM(3, input);
    CHECK_DIM(3, output);
    CHECK_DIM(3, state);
    CHECK_DIM(3, weight);
    // Every operand is indexed as a dense row-major array.  An empty one (a
    // kernel_size-1 state, an empty chunk) is never read, and TVM does not call
    // a tensor with a zero extent contiguous.
    for (const TensorView* operand : {&input, &output, &state, &weight}) {
        if (operand->numel() > 0) {
            CHECK_CONTIGUOUS_INPUT(*operand);
        }
    }

    const int64_t batch_size = input.size(0);
    const int64_t chunk_len = input.size(1);
    const int64_t channels = input.size(2);
    const int64_t kernel_size = weight.size(2);

    TVM_FFI_ICHECK(kernel_size >= 1) << "CausalConv1D kernel_size must be at least 1";
    TVM_FFI_ICHECK(weight.size(0) == channels && weight.size(1) == 1)
        << "CausalConv1D weight must be [C, 1, K] with C=" << channels << ", got ["
        << weight.size(0) << ", " << weight.size(1) << ", " << kernel_size << "]";
    TVM_FFI_ICHECK(state.size(0) == batch_size && state.size(1) == kernel_size - 1 &&
                   state.size(2) == channels)
        << "CausalConv1D state must be [B, K - 1, C] = [" << batch_size << ", " << kernel_size - 1
        << ", " << channels << "], got [" << state.size(0) << ", " << state.size(1) << ", "
        << state.size(2) << "]";
    TVM_FFI_ICHECK(output.size(0) == batch_size && output.size(1) == chunk_len &&
                   output.size(2) == channels)
        << "CausalConv1D output must be shaped like the input";
    TVM_FFI_ICHECK(same_dtype(input, output) && same_dtype(input, state) &&
                   same_dtype(input, weight))
        << "CausalConv1D input, state, weight, and output dtypes must match";
    // The convolution reads the state and the input while it writes the output,
    // and the state update then reads the input again.
    TVM_FFI_ICHECK(!overlaps(output, input) && !overlaps(output, state) && !overlaps(state, input))
        << "CausalConv1D output, input and state must not share memory";
    // Extents travel as int, and the weight is indexed in 32 bits.
    TVM_FFI_ICHECK(batch_size <= 0x7fffffff && chunk_len <= 0x7fffffff &&
                   channels * kernel_size <= 0x7fffffff)
        << "CausalConv1D extents are past the kernels' 32-bit sizes";

    if (bias_opt.has_value()) {
        TensorView bias = bias_opt.value();
        CHECK_INPUT(bias);
        CHECK_DIM(1, bias);
        CHECK_CONTIGUOUS_INPUT(bias);
        CHECK_DEVICE(input, bias);
        TVM_FFI_ICHECK(bias.size(0) == channels) << "CausalConv1D bias shape mismatch";
        TVM_FFI_ICHECK(same_dtype(input, bias)) << "CausalConv1D bias dtype mismatch";
    }

    cudaStream_t stream = get_stream(input.device());

    DISPATCH_DLPACK_HALF_DTYPE(input.dtype(), c_type, [&] {
        const c_type* bias_ptr = nullptr;
        if (bias_opt.has_value()) {
            bias_ptr = static_cast<const c_type*>(bias_opt.value().data_ptr());
        }
        cudaError_t status = conv::CausalConv1D<c_type>(
            static_cast<const c_type*>(input.data_ptr()), static_cast<c_type*>(state.data_ptr()),
            static_cast<const c_type*>(weight.data_ptr()), bias_ptr,
            static_cast<c_type*>(output.data_ptr()), static_cast<int>(batch_size),
            static_cast<int>(chunk_len), static_cast<int>(channels), static_cast<int>(kernel_size),
            stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "CausalConv1D kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}
