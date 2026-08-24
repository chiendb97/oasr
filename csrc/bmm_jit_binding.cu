// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// TVM-FFI JIT binding exports for batched GEMM (BMM) kernel.

#include "tvm_ffi_utils.h"

// Forward declaration — the general BMM launcher (csrc/bmm.cu).  The 23
// rendered alignment-8 tile variants export their own `bmm_<config>` symbols.
void bmm(TensorView output, TensorView A, TensorView B);

// TVM-FFI symbol export
TVM_FFI_DLL_EXPORT_TYPED_FUNC(bmm, bmm);
