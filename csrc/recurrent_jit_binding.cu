// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#include "tvm_ffi_utils.h"

void lstm_layer(TensorView output, TensorView final_h, TensorView final_c, TensorView cells,
                TensorView input, TensorView initial_h, TensorView initial_c, TensorView weight_ih,
                TensorView weight_hh, Optional bias_ih, Optional bias_hh, bool batch_first);
void rnn_layer(TensorView output, TensorView final_h, TensorView input, TensorView initial_h,
               TensorView weight_ih, TensorView weight_hh, Optional bias_ih, Optional bias_hh,
               int64_t activation, bool batch_first);
void lstm_gemm_layer(TensorView output, TensorView final_h, TensorView final_c, TensorView cells,
                     TensorView workspace, TensorView input_gates, TensorView initial_h,
                     TensorView initial_c, TensorView weight_hh, Optional bias_hh,
                     bool input_batch_first, int64_t tactic, int64_t split_k_slices);
void rnn_gemm_layer(TensorView output, TensorView final_h, TensorView workspace,
                    TensorView input_gates, TensorView initial_h, TensorView weight_hh,
                    Optional bias_hh, int64_t activation, bool input_batch_first, int64_t tactic,
                    int64_t split_k_slices);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(lstm_layer, lstm_layer);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(rnn_layer, rnn_layer);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(lstm_gemm_layer, lstm_gemm_layer);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(rnn_gemm_layer, rnn_gemm_layer);
