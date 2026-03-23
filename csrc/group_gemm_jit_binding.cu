// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// TVM-FFI JIT binding exports for grouped GEMM kernel.

#include "tvm_ffi_utils.h"

// Forward declaration — GroupGemm launcher
void group_gemm(TensorView output, TensorView A, TensorView B, TensorView offsets);

// TVM-FFI symbol export
TVM_FFI_DLL_EXPORT_TYPED_FUNC(group_gemm, group_gemm);
