// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// TVM-FFI JIT binding exports for conv2d kernels.

#include "tvm_ffi_utils.h"

// Forward declarations — conv2d launchers
void conv2d(TensorView output, TensorView input, TensorView filter, Optional bias_opt,
            int64_t pad_h, int64_t pad_w, int64_t stride_h, int64_t stride_w, int64_t dilation_h,
            int64_t dilation_w);
void conv2d_activation(TensorView output, TensorView input, TensorView filter, Optional bias_opt,
                       int64_t activation_type, int64_t pad_h, int64_t pad_w, int64_t stride_h,
                       int64_t stride_w, int64_t dilation_h, int64_t dilation_w);

// TVM-FFI symbol exports
TVM_FFI_DLL_EXPORT_TYPED_FUNC(conv2d, conv2d);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(conv2d_activation, conv2d_activation);
