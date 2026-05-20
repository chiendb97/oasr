// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// TVM-FFI JIT binding for fused GEMM + log_softmax.

#include "tvm_ffi_utils.h"

void gemm_log_softmax(TensorView output, TensorView A, TensorView B, Optional bias_opt,
                      int64_t split_k_slices);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemm_log_softmax, gemm_log_softmax);
