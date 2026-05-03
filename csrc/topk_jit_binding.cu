// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#include "tvm_ffi_utils.h"

void topk(TensorView values, TensorView indices, TensorView input, int64_t k);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(topk, topk);
