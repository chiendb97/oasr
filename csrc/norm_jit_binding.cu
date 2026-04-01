// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// TVM-FFI JIT binding exports for normalization kernels.

#include "tvm_ffi_utils.h"

// Forward declarations of launcher functions
void layernorm(TensorView output, TensorView input, TensorView weight, Optional bias_opt,
               double eps);
void rmsnorm(TensorView output, TensorView input, TensorView weight, Optional bias_opt,
             double eps);
void batchnorm1d(TensorView output, TensorView input, TensorView weight, TensorView bias,
                 TensorView running_mean, TensorView running_var, double eps);
void groupnorm(TensorView output, TensorView input, TensorView weight, TensorView bias,
               int64_t num_groups, double eps);
void addlayernorm(TensorView output, TensorView input, TensorView residual, TensorView weight,
                  Optional bias_opt, double eps);
void layernorm_activation(TensorView output, TensorView input, TensorView weight,
                          Optional bias_opt, double eps, int64_t activation_type);
void rmsnorm_activation(TensorView output, TensorView input, TensorView weight, Optional bias_opt,
                        double eps, int64_t activation_type);
void batchnorm_activation(TensorView output, TensorView input, TensorView weight, TensorView bias,
                          TensorView running_mean, TensorView running_var, double eps,
                          int64_t activation_type);
void batchnorm_swish(TensorView output, TensorView input, TensorView weight, TensorView bias,
                     TensorView running_mean, TensorView running_var, double eps);
void cmvn(TensorView output, TensorView input, TensorView mean, TensorView istd);

// TVM-FFI symbol exports
TVM_FFI_DLL_EXPORT_TYPED_FUNC(layernorm, layernorm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(rmsnorm, rmsnorm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(batchnorm1d, batchnorm1d);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(groupnorm, groupnorm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(addlayernorm, addlayernorm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(layernorm_activation, layernorm_activation);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(rmsnorm_activation, rmsnorm_activation);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(batchnorm_activation, batchnorm_activation);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(batchnorm_swish, batchnorm_swish);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(cmvn, cmvn);
