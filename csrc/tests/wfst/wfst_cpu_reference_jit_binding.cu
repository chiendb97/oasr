// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// TVM-FFI JIT binding exports for the test-only WFST CPU reference oracle.

#include <tvm/ffi/string.h>

#include "tvm_ffi_utils.h"

// Forward declaration of the launcher (see wfst_cpu_reference.cu).
void wfst_cpu_decode(tvm::ffi::String graph_path, TensorView log_probs, double search_beam,
                     double output_beam, int64_t min_active, int64_t max_active,
                     int64_t allow_partial, int64_t online, int64_t eps_iterations,
                     TensorView out_words, TensorView out_word_len, TensorView out_score,
                     TensorView out_meta);

// TVM-FFI symbol export
TVM_FFI_DLL_EXPORT_TYPED_FUNC(wfst_cpu_decode, wfst_cpu_decode);
