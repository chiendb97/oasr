// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// TVM-FFI JIT binding exports for the transducer decode kernels.

#include "tvm_ffi_utils.h"

void stateless_greedy_decode(TensorView tokens, TensorView frames, Optional probs,
                             TensorView counts, TensorView window_out, TensorView dec_proj_out,
                             TensorView enc_proj, TensorView lengths, TensorView window_in,
                             TensorView dec_proj_in, TensorView w_out_t, Optional b_out,
                             TensorView emb, Optional conv_w, TensorView w_dp_t, Optional b_dp,
                             int64_t vocab, int64_t group, int64_t max_sym, int64_t blank,
                             int64_t activation, int64_t rows_per_cta, int64_t cluster);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(stateless_greedy_decode, stateless_greedy_decode);

void transducer_beam_topk(TensorView context_out, TensorView scores_out, TensorView parent_out,
                          TensorView label_out, TensorView logits, TensorView scores,
                          TensorView context, TensorView active, int64_t blank);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(transducer_beam_topk, transducer_beam_topk);

void stateless_beam_decode(TensorView context_out, TensorView scores_out, TensorView parents,
                           TensorView labels, Optional walk, TensorView enc_proj,
                           TensorView lengths, TensorView context_in, TensorView scores_in,
                           TensorView w_out_t, Optional b_out, TensorView emb, Optional conv_w,
                           TensorView w_dp_t, Optional b_dp, int64_t vocab, int64_t group,
                           int64_t blank, int64_t activation, int64_t cluster);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(stateless_beam_decode, stateless_beam_decode);

bool stateless_beam_fits(int64_t beam, int64_t J, int64_t D, int64_t vocab);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(stateless_beam_fits, stateless_beam_fits);

bool stateless_greedy_fits(int64_t J, int64_t D);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(stateless_greedy_fits, stateless_greedy_fits);

void lstm_greedy_decode(TensorView tokens, TensorView frames, Optional probs, TensorView counts,
                        TensorView h_out, TensorView c_out, TensorView dec_proj_out,
                        TensorView workspace, TensorView enc_proj, TensorView lengths,
                        TensorView h_in, TensorView c_in, TensorView dec_proj_in, TensorView emb,
                        TensorView w_ih, TensorView w_hh, Optional bias, TensorView w_dp,
                        Optional b_dp, TensorView w_out, Optional b_out, int64_t vocab,
                        int64_t max_sym, int64_t blank, int64_t grid);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(lstm_greedy_decode, lstm_greedy_decode);

int64_t lstm_greedy_workspace_bytes(int64_t batch, int64_t hidden, int64_t layers, int64_t joiner,
                                    int64_t vocab, int64_t grid);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(lstm_greedy_workspace_bytes, lstm_greedy_workspace_bytes);
