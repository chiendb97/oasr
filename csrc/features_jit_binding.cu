// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// TVM-FFI JIT binding exports for the FBANK / MFCC kernels.

#include "tvm_ffi_utils.h"

void stft_frame(TensorView output, TensorView waveform, TensorView lengths, TensorView window,
                int64_t hop_length, int64_t center_offset, int64_t win_offset, double preemph_coef,
                bool preemph_replicate, bool remove_dc_offset, bool reflect_pad,
                int64_t signal_length);
void fbank_preprocess(TensorView output, TensorView frames, TensorView window, double preemph_coef,
                      bool remove_dc_offset, bool apply_preemph);
void mel_log(TensorView output, TensorView power, TensorView mel_mat, double log_floor,
             double log_offset, Optional frame_lengths_opt);
void dct_lifter(TensorView output, TensorView log_mel, TensorView dct_mat, Optional lifter_opt,
                Optional energy_opt, bool replace_c0_with_energy);
void whisper_logmel(TensorView output, TensorView power, TensorView mel_mat, double log_floor,
                    double max_floor, double offset, double scale);
void lfr_gather(TensorView output, TensorView input, TensorView lengths, int64_t lfr_m,
                int64_t lfr_n);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(stft_frame, stft_frame);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fbank_preprocess, fbank_preprocess);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mel_log, mel_log);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(dct_lifter, dct_lifter);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(whisper_logmel, whisper_logmel);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(lfr_gather, lfr_gather);
