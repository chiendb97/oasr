// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// TVM-FFI JIT binding exports for the FFT kernel family.

#include "tvm_ffi_utils.h"

void rfft(TensorView output, TensorView input, int64_t n_fft);
void rfft_power(TensorView output, TensorView input, int64_t n_fft);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(rfft, rfft);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(rfft_power, rfft_power);
