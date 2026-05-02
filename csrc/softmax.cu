// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// TVM-FFI launcher layer for softmax kernel.

#include <oasr/softmax.cuh>

#include "tvm_ffi_utils.h"

using namespace oasr;

void softmax(TensorView output, TensorView input) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(input);
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(output);

    unsigned int num_rows = 1;
    for (int i = 0; i < input.ndim() - 1; ++i) {
        num_rows *= input.size(i);
    }
    unsigned int num_cols = input.size(input.ndim() - 1);

    cudaStream_t stream = get_stream(input.device());

    DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(input.dtype(), c_type, [&] {
        cudaError_t status = softmax::Softmax<c_type>(
            static_cast<const c_type*>(input.data_ptr()),
            static_cast<c_type*>(output.data_ptr()), num_rows, num_cols, stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "Softmax kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}
