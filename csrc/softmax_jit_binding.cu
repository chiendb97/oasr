// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// TVM-FFI JIT binding exports for softmax kernel.

#include "tvm_ffi_utils.h"

void softmax(TensorView output, TensorView input);
void log_softmax(TensorView output, TensorView input);
void masked_softmax(TensorView output, TensorView input, Optional bias_opt, Optional mask_opt,
                    Optional mask2_opt, double mask_value);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(softmax, softmax);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(log_softmax, log_softmax);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(masked_softmax, masked_softmax);
