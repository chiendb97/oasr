// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// TVM-FFI JIT binding exports for pooling kernels.

#include "tvm_ffi_utils.h"

void avg_pool1d(TensorView output, TensorView input, int64_t kernel_size, int64_t stride,
                int64_t padding, bool ceil_mode, bool count_include_pad);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(avg_pool1d, avg_pool1d);
