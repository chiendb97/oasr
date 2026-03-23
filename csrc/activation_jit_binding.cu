// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// TVM-FFI JIT binding exports for activation kernels.

#include "tvm_ffi_utils.h"

// Forward declarations of launcher functions
void glu(TensorView output, TensorView input);
void swish(TensorView output, TensorView input);

// TVM-FFI symbol exports
TVM_FFI_DLL_EXPORT_TYPED_FUNC(glu, glu);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(swish, swish);
