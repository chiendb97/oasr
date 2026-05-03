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
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(input);
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(values);
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(indices);

    TVM_FFI_ICHECK(k > 0) << "k must be positive, got " << k;

    unsigned int num_cols = input.size(input.ndim() - 1);
    TVM_FFI_ICHECK(static_cast<int64_t>(num_cols) >= k)
        << "k (" << k << ") must be <= num_cols (" << num_cols << ")";

    // Register-resident row data caps num_cols at 16384 (1024 threads * PER_THREAD=16).
    TVM_FFI_ICHECK(num_cols <= 16384u)
        << "num_cols=" << num_cols << " exceeds the supported limit of 16384.";

    unsigned int num_rows = 1;
    for (int i = 0; i < input.ndim() - 1; ++i) {
        num_rows *= static_cast<unsigned int>(input.size(i));
    }

    TVM_FFI_ICHECK(indices.dtype().code == kDLInt && indices.dtype().bits == 32)
        << "indices tensor must have dtype int32";

    cudaStream_t stream = get_stream(input.device());

    DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(input.dtype(), c_type, [&] {
        cudaError_t status = topk::TopK<c_type>(
            static_cast<const c_type*>(input.data_ptr()),
            static_cast<c_type*>(values.data_ptr()),
            static_cast<int32_t*>(indices.data_ptr()),
            num_rows, num_cols, static_cast<int>(k), stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "TopK kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}
