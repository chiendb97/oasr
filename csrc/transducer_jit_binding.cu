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
                             int64_t activation, int64_t rows_per_cta);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(stateless_greedy_decode, stateless_greedy_decode);
