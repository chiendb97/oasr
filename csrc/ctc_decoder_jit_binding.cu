// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// TVM-FFI JIT binding exports for GPU CTC prefix beam search decoder.

#include "tvm_ffi_utils.h"

// Forward declarations of launcher functions
int64_t ctc_decoder_workspace_size(int64_t batch, int64_t beam,
                                   int64_t vocab_size, int64_t max_seq_len);
int64_t ctc_decoder_state_size(int64_t batch, int64_t beam,
                               int64_t vocab_size, int64_t max_seq_len);
void ctc_beam_search_decode(TensorView out_tokens, TensorView out_lengths,
                            TensorView out_scores, TensorView log_prob,
                            TensorView seq_lengths, TensorView workspace,
                            int64_t beam, int64_t blank_id,
                            double blank_threshold);
void ctc_beam_search_init_state(TensorView state_buffer,
                                int64_t batch, int64_t beam,
                                int64_t vocab_size, int64_t max_seq_len,
                                int64_t blank_id);
void ctc_beam_search_step(TensorView state_buffer, TensorView log_prob_frame,
                          int64_t beam, int64_t blank_id,
                          int64_t step, double blank_threshold,
                          int64_t actual_frame_index,
                          int64_t batch, int64_t vocab_size,
                          int64_t max_seq_len, int64_t use_paged_memory,
                          int64_t page_size);
void ctc_beam_search_read_state(TensorView out_tokens, TensorView out_lengths,
                                TensorView out_scores, TensorView state_buffer,
                                int64_t step, int64_t batch, int64_t beam,
                                int64_t vocab_size, int64_t max_seq_len,
                                int64_t use_paged_memory, int64_t page_size);

// Paged variants
int64_t ctc_decoder_paged_workspace_size(int64_t batch, int64_t beam,
                                         int64_t vocab_size, int64_t max_seq_len,
                                         int64_t page_size);
int64_t ctc_decoder_paged_state_size(int64_t batch, int64_t beam,
                                     int64_t vocab_size, int64_t max_seq_len,
                                     int64_t page_size);
void ctc_beam_search_decode_paged(TensorView out_tokens, TensorView out_lengths,
                                  TensorView out_scores, TensorView log_prob,
                                  TensorView seq_lengths, TensorView workspace,
                                  int64_t beam, int64_t blank_id,
                                  double blank_threshold, int64_t page_size);
void ctc_beam_search_init_state_paged(TensorView state_buffer,
                                      int64_t batch, int64_t beam,
                                      int64_t vocab_size, int64_t max_seq_len,
                                      int64_t blank_id, int64_t page_size);

// TVM-FFI symbol exports
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ctc_decoder_workspace_size, ctc_decoder_workspace_size);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ctc_decoder_state_size, ctc_decoder_state_size);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ctc_beam_search_decode, ctc_beam_search_decode);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ctc_beam_search_init_state, ctc_beam_search_init_state);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ctc_beam_search_step, ctc_beam_search_step);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ctc_beam_search_read_state, ctc_beam_search_read_state);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ctc_decoder_paged_workspace_size, ctc_decoder_paged_workspace_size);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ctc_decoder_paged_state_size, ctc_decoder_paged_state_size);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ctc_beam_search_decode_paged, ctc_beam_search_decode_paged);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ctc_beam_search_init_state_paged, ctc_beam_search_init_state_paged);
