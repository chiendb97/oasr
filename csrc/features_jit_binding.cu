// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// TVM-FFI JIT binding exports for the FBANK / MFCC kernels.

#include "tvm_ffi_utils.h"

void fbank_preprocess(TensorView output, TensorView frames, TensorView window,
                      double preemph_coef, bool remove_dc_offset, bool apply_preemph);
void mel_log(TensorView output, TensorView power, TensorView mel_mat, double log_floor);
void dct_lifter(TensorView output, TensorView log_mel, TensorView dct_mat,
                Optional lifter_opt, Optional energy_opt,
                bool replace_c0_with_energy);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fbank_preprocess, fbank_preprocess);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mel_log, mel_log);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(dct_lifter, dct_lifter);
