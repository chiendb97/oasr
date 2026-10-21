// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// TVM-FFI JIT binding exports for activation quantization kernels.

#include "tvm_ffi_utils.h"

// Forward declarations of launcher functions
void quantize_per_token(TensorView output, TensorView scale, TensorView input, Optional smooth_inv,
                        double eps);
void quantize_static(TensorView output, TensorView input, TensorView mul);

// TVM-FFI symbol exports
TVM_FFI_DLL_EXPORT_TYPED_FUNC(quantize_per_token, quantize_per_token);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(quantize_static, quantize_static);
