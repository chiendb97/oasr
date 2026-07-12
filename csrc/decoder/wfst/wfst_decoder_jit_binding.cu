// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// TVM-FFI JIT binding exports for the in-tree GPU WFST beam-search decoder.

#include <tvm/ffi/string.h>

#include "tvm_ffi_utils.h"

// Forward declarations of launcher functions (see wfst_decoder.cu).
int64_t wfst_load_graph(tvm::ffi::String path);
void wfst_free_graph(int64_t graph_handle);
void wfst_graph_info(int64_t graph_handle, TensorView out_info);
int64_t wfst_create_decoder(int64_t graph_handle, double search_beam, double output_beam,
                            int64_t min_active, int64_t max_active, int64_t allow_partial,
                            int64_t max_lanes, int64_t max_frames, int64_t device,
                            int64_t main_q_factor, int64_t cand_factor, int64_t use_cuda_graphs,
                            int64_t lattice, int64_t fp16_logprobs, int64_t streaming,
                            int64_t lat_prune_interval, int64_t eps_iterations);
void wfst_free_decoder(int64_t handle);
void wfst_decode_batch(int64_t handle, TensorView log_probs, TensorView lengths,
                       TensorView out_words, TensorView out_word_lens, TensorView out_scores,
                       TensorView out_meta);
int64_t wfst_create_stream(int64_t handle);
void wfst_release_stream(int64_t handle, int64_t channel);
void wfst_advance_chunk(int64_t handle, TensorView channels, TensorView log_probs,
                        TensorView lengths, int64_t want_partial, TensorView out_words,
                        TensorView out_word_lens, TensorView out_channels,
                        TensorView out_overflow);
void wfst_finalize_stream(int64_t handle, int64_t channel, TensorView out_words,
                          TensorView out_word_len, TensorView out_score, TensorView out_meta);
void wfst_cpu_decode(int64_t graph_handle, TensorView log_probs, double search_beam,
                     double output_beam, int64_t min_active, int64_t max_active,
                     int64_t allow_partial, int64_t online, int64_t eps_iterations,
                     TensorView out_words, TensorView out_word_len, TensorView out_score,
                     TensorView out_meta);

// TVM-FFI symbol exports
TVM_FFI_DLL_EXPORT_TYPED_FUNC(wfst_load_graph, wfst_load_graph);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(wfst_free_graph, wfst_free_graph);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(wfst_graph_info, wfst_graph_info);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(wfst_create_decoder, wfst_create_decoder);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(wfst_free_decoder, wfst_free_decoder);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(wfst_decode_batch, wfst_decode_batch);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(wfst_create_stream, wfst_create_stream);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(wfst_release_stream, wfst_release_stream);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(wfst_advance_chunk, wfst_advance_chunk);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(wfst_finalize_stream, wfst_finalize_stream);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(wfst_cpu_decode, wfst_cpu_decode);
