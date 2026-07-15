// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// TVM-FFI JIT binding exports for softmax kernel.

#include "tvm_ffi_utils.h"

void softmax(TensorView output, TensorView input);
void log_softmax(TensorView output, TensorView input);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(softmax, softmax);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(log_softmax, log_softmax);
