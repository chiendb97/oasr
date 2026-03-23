// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// TVM-FFI JIT binding exports for GEMM kernels.

#include "tvm_ffi_utils.h"

// Forward declarations — GEMM launchers
void gemm(TensorView output, TensorView A, TensorView B, Optional C_opt,
          int64_t split_k_slices);
void gemm_activation(TensorView output, TensorView A, TensorView B, Optional C_opt,
                     int64_t activation_type, int64_t split_k_slices);

// TVM-FFI symbol exports
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemm, gemm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemm_activation, gemm_activation);
