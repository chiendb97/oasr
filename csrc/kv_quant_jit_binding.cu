// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#include "tvm_ffi_utils.h"

void kv_quant_store(TensorView k_out, TensorView k_scale, Optional v_out, Optional v_scale,
                    TensorView k, Optional v, Optional pos);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(kv_quant_store, kv_quant_store);
