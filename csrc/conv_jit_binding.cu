// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// TVM-FFI JIT binding exports for conv1d kernels.

#include "tvm_ffi_utils.h"

// Forward declarations — conv1d launchers
void depthwise_conv1d(TensorView output, TensorView input, TensorView weight, Optional bias_opt,
                      int64_t padding);
void depthwise_conv1d_silu(TensorView output, TensorView input, TensorView weight,
                           Optional bias_opt, int64_t padding);
void pointwise_conv1d(TensorView output, TensorView input, TensorView weight, Optional bias_opt);
void pointwise_conv1d_activation(TensorView output, TensorView input, TensorView weight,
                                 Optional bias_opt, int64_t activation_type);
void causal_conv1d(TensorView output, TensorView input, TensorView state, TensorView weight,
                   Optional bias_opt);

// TVM-FFI symbol exports
TVM_FFI_DLL_EXPORT_TYPED_FUNC(depthwise_conv1d, depthwise_conv1d);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(depthwise_conv1d_silu, depthwise_conv1d_silu);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(pointwise_conv1d, pointwise_conv1d);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(pointwise_conv1d_activation, pointwise_conv1d_activation);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(causal_conv1d, causal_conv1d);
