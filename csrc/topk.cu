// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// TVM-FFI launcher layer for top-k kernel.

#include <oasr/topk.cuh>

#include "tvm_ffi_utils.h"

using namespace oasr;

/*!
 * \brief Select k largest values along the last dimension of input.
 *
 * \param values   Output values [*batch_dims, k], same dtype as input.
 * \param indices  Output indices [*batch_dims, k], dtype int32.
 * \param input    Input tensor [*batch_dims, num_cols].
 * \param k        Number of top elements to select per row.
 */
void topk(TensorView values, TensorView indices, TensorView input, int64_t k) {
    CHECK_INPUT(input);
    CHECK_INPUT(values);
    CHECK_INPUT(indices);
    CHECK_DEVICE(input, values);
    CHECK_DEVICE(input, indices);
    // Rows are read at `input + row * num_cols` and written at `+ row * k`, so
    // all three must be plain row-major: a padded or permuted view would pair an
    // input row with another row's outputs.
    CHECK_CONTIGUOUS_INPUT(input);
    CHECK_CONTIGUOUS_INPUT(values);
    CHECK_CONTIGUOUS_INPUT(indices);
    TVM_FFI_ICHECK(input.ndim() >= 1) << "topk input must have at least one dimension";

    TVM_FFI_ICHECK(k > 0) << "k must be positive, got " << k;

    const int64_t num_cols = input.size(input.ndim() - 1);
    TVM_FFI_ICHECK(num_cols >= k) << "k (" << k << ") must be <= num_cols (" << num_cols << ")";

    // Register-resident row data caps num_cols at 16384 (1024 threads * PER_THREAD=16).
    TVM_FFI_ICHECK(num_cols <= 16384)
        << "num_cols=" << num_cols << " exceeds the supported limit of 16384.";

    // The outputs are the input's shape with the last extent replaced by k.
    const auto shaped_like_the_selection = [&](const TensorView& out) {
        if (out.ndim() != input.ndim() || out.size(out.ndim() - 1) != k) {
            return false;
        }
        for (int i = 0; i < input.ndim() - 1; ++i) {
            if (out.size(i) != input.size(i)) {
                return false;
            }
        }
        return true;
    };
    TVM_FFI_ICHECK(shaped_like_the_selection(values))
        << "topk values must be shaped [..., k] like the input's leading dims, with k=" << k;
    TVM_FFI_ICHECK(shaped_like_the_selection(indices))
        << "topk indices must be shaped [..., k] like the input's leading dims, with k=" << k;
    TVM_FFI_ICHECK(values.dtype().code == input.dtype().code &&
                   values.dtype().bits == input.dtype().bits &&
                   values.dtype().lanes == input.dtype().lanes)
        << "topk values must have the input's dtype";
    TVM_FFI_ICHECK(indices.dtype().code == kDLInt && indices.dtype().bits == 32 &&
                   indices.dtype().lanes == 1)
        << "indices tensor must have dtype int32";

    // One block per row, on the grid's x axis.
    const int64_t num_rows = input.numel() / num_cols;
    TVM_FFI_ICHECK(num_rows <= 0x7fffffff)
        << "topk launches one block per row and cannot address " << num_rows << " rows";

    cudaStream_t stream = get_stream(input.device());

    DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(input.dtype(), c_type, [&] {
        cudaError_t status = topk::TopK<c_type>(
            static_cast<const c_type*>(input.data_ptr()), static_cast<c_type*>(values.data_ptr()),
            static_cast<int32_t*>(indices.data_ptr()), static_cast<unsigned int>(num_rows),
            static_cast<unsigned int>(num_cols), static_cast<int>(k), stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "TopK kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}
