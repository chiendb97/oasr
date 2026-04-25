// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// TVM-FFI launcher layer for GPU CTC prefix beam search decoder.

#include <oasr/ctc_decoder.cuh>

#include "tvm_ffi_utils.h"

using namespace oasr;

// =============================================================================
// Workspace size query
// =============================================================================

int64_t ctc_decoder_workspace_size(int64_t batch, int64_t beam,
                                   int64_t vocab_size, int64_t max_seq_len) {
  return static_cast<int64_t>(
      ctc_decoder::calculate_workspace_size(batch, beam, vocab_size, max_seq_len));
}

// =============================================================================
// Streaming state buffer size query
// =============================================================================

int64_t ctc_decoder_state_size(int64_t batch, int64_t beam,
                               int64_t vocab_size, int64_t max_seq_len) {
  return static_cast<int64_t>(
      ctc_decoder::calculate_state_buffer_size(batch, beam, vocab_size, max_seq_len));
}

// =============================================================================
// Full offline decode
// =============================================================================

void ctc_beam_search_decode(TensorView out_tokens, TensorView out_lengths,
                            TensorView out_scores, TensorView log_prob,
                            TensorView seq_lengths, TensorView workspace,
                            int64_t beam, int64_t blank_id,
                            double blank_threshold) {
  CHECK_INPUT(log_prob);
  CHECK_INPUT(seq_lengths);
  CHECK_INPUT(workspace);
  CHECK_INPUT(out_tokens);
  CHECK_INPUT(out_lengths);
  CHECK_INPUT(out_scores);
  CHECK_DIM(3, log_prob);
  CHECK_DIM(1, seq_lengths);
  CHECK_DIM(3, out_tokens);
  CHECK_DIM(2, out_lengths);
  CHECK_DIM(2, out_scores);

  TVM_FFI_ICHECK(log_prob.dtype().code == kDLFloat && log_prob.dtype().bits == 32)
      << "log_prob must be float32";
  TVM_FFI_ICHECK(seq_lengths.dtype().code == kDLInt && seq_lengths.dtype().bits == 32)
      << "seq_lengths must be int32";

  int batch = log_prob.size(0);
  int max_seq_len = log_prob.size(1);
  int vocab_size = log_prob.size(2);
  int max_out_len = out_tokens.size(2);

  int batch_stride = log_prob.stride(0);
  int seq_stride = log_prob.stride(1);
  int vocab_stride = log_prob.stride(2);

  cudaStream_t stream = get_stream(log_prob.device());

  cudaError_t status = ctc_decoder::ctc_beam_search_decode_batch(
      static_cast<const float*>(log_prob.data_ptr()),
      batch_stride, seq_stride, vocab_stride,
      static_cast<const int*>(seq_lengths.data_ptr()),
      static_cast<int*>(out_tokens.data_ptr()),
      static_cast<int*>(out_lengths.data_ptr()),
      static_cast<float*>(out_scores.data_ptr()),
      workspace.data_ptr(),
      batch, static_cast<int>(beam), vocab_size,
      max_seq_len, max_out_len,
      static_cast<int>(blank_id), -1, // space_id = -1 (unused)
      static_cast<float>(blank_threshold),
      stream);

  TVM_FFI_ICHECK(status == cudaSuccess)
      << "CTC beam search decode failed: " << cudaGetErrorString(status);
}

// =============================================================================
// Streaming: initialize state
// =============================================================================

void ctc_beam_search_init_state(TensorView state_buffer,
                                int64_t batch, int64_t beam,
                                int64_t vocab_size, int64_t max_seq_len,
                                int64_t blank_id) {
  CHECK_INPUT(state_buffer);

  cudaStream_t stream = get_stream(state_buffer.device());

  ctc_decoder::init_streaming_state(
      state_buffer.data_ptr(),
      static_cast<int>(batch), static_cast<int>(beam),
      static_cast<int>(vocab_size), static_cast<int>(max_seq_len),
      static_cast<int>(blank_id), stream);
}

// =============================================================================
// Streaming: single step
// =============================================================================

void ctc_beam_search_step(TensorView state_buffer, TensorView log_prob_frame,
                          int64_t beam, int64_t blank_id,
                          int64_t step, double blank_threshold,
                          int64_t actual_frame_index,
                          int64_t batch, int64_t vocab_size,
                          int64_t max_seq_len, int64_t use_paged_memory,
                          int64_t page_size) {
  CHECK_INPUT(state_buffer);
  CHECK_INPUT(log_prob_frame);
  CHECK_DIM(2, log_prob_frame);

  TVM_FFI_ICHECK(log_prob_frame.dtype().code == kDLFloat &&
                  log_prob_frame.dtype().bits == 32)
      << "log_prob_frame must be float32";

  int batch_stride = log_prob_frame.stride(0);
  int vocab_stride = log_prob_frame.stride(1);

  cudaStream_t stream = get_stream(state_buffer.device());

  cudaError_t status = ctc_decoder::streaming_step(
      state_buffer.data_ptr(),
      static_cast<const float*>(log_prob_frame.data_ptr()),
      batch_stride, vocab_stride,
      static_cast<int>(step),
      static_cast<int>(blank_id), -1, // space_id = -1
      static_cast<int>(actual_frame_index),
      static_cast<int>(batch), static_cast<int>(beam),
      static_cast<int>(vocab_size), static_cast<int>(max_seq_len),
      static_cast<int>(use_paged_memory), static_cast<int>(page_size),
      0,  // num_pages=0 → auto
      stream);

  TVM_FFI_ICHECK(status == cudaSuccess)
      << "CTC beam search step failed: " << cudaGetErrorString(status);
}

// =============================================================================
// Paged: workspace and state size queries
// =============================================================================

int64_t ctc_decoder_paged_workspace_size(int64_t batch, int64_t beam,
                                         int64_t vocab_size, int64_t max_seq_len,
                                         int64_t page_size) {
  return static_cast<int64_t>(ctc_decoder::calculate_paged_workspace_size(
      static_cast<int>(batch), static_cast<int>(beam),
      static_cast<int>(vocab_size), static_cast<int>(max_seq_len),
      static_cast<int>(page_size)));
}

int64_t ctc_decoder_paged_state_size(int64_t batch, int64_t beam,
                                     int64_t vocab_size, int64_t max_seq_len,
                                     int64_t page_size) {
  return static_cast<int64_t>(ctc_decoder::calculate_paged_state_buffer_size(
      static_cast<int>(batch), static_cast<int>(beam),
      static_cast<int>(vocab_size), static_cast<int>(max_seq_len),
      static_cast<int>(page_size)));
}

// =============================================================================
// Paged: full offline decode
// =============================================================================

void ctc_beam_search_decode_paged(TensorView out_tokens, TensorView out_lengths,
                                  TensorView out_scores, TensorView log_prob,
                                  TensorView seq_lengths, TensorView workspace,
                                  int64_t beam, int64_t blank_id,
                                  double blank_threshold, int64_t page_size) {
  CHECK_INPUT(log_prob);
  CHECK_INPUT(seq_lengths);
  CHECK_INPUT(workspace);
  CHECK_INPUT(out_tokens);
  CHECK_INPUT(out_lengths);
  CHECK_INPUT(out_scores);
  CHECK_DIM(3, log_prob);
  CHECK_DIM(1, seq_lengths);
  CHECK_DIM(3, out_tokens);
  CHECK_DIM(2, out_lengths);
  CHECK_DIM(2, out_scores);

  TVM_FFI_ICHECK(log_prob.dtype().code == kDLFloat && log_prob.dtype().bits == 32)
      << "log_prob must be float32";
  TVM_FFI_ICHECK(seq_lengths.dtype().code == kDLInt && seq_lengths.dtype().bits == 32)
      << "seq_lengths must be int32";

  int batch = log_prob.size(0);
  int max_seq_len = log_prob.size(1);
  int vocab_size = log_prob.size(2);
  int max_out_len = out_tokens.size(2);

  int batch_stride = log_prob.stride(0);
  int seq_stride = log_prob.stride(1);
  int vocab_stride = log_prob.stride(2);

  cudaStream_t stream = get_stream(log_prob.device());

  // Compute num_pages from workspace size to pass 0 (auto-compute inside)
  cudaError_t status = ctc_decoder::ctc_beam_search_decode_batch_paged(
      static_cast<const float*>(log_prob.data_ptr()),
      batch_stride, seq_stride, vocab_stride,
      static_cast<const int*>(seq_lengths.data_ptr()),
      static_cast<int*>(out_tokens.data_ptr()),
      static_cast<int*>(out_lengths.data_ptr()),
      static_cast<float*>(out_scores.data_ptr()),
      workspace.data_ptr(),
      batch, static_cast<int>(beam), vocab_size,
      max_seq_len, max_out_len,
      static_cast<int>(blank_id), -1,
      static_cast<float>(blank_threshold),
      static_cast<int>(page_size), 0,  // num_pages=0 → auto
      stream);

  TVM_FFI_ICHECK(status == cudaSuccess)
      << "CTC paged beam search decode failed: " << cudaGetErrorString(status);
}

// =============================================================================
// Paged: streaming init
// =============================================================================

void ctc_beam_search_init_state_paged(TensorView state_buffer,
                                      int64_t batch, int64_t beam,
                                      int64_t vocab_size, int64_t max_seq_len,
                                      int64_t blank_id, int64_t page_size) {
  CHECK_INPUT(state_buffer);

  cudaStream_t stream = get_stream(state_buffer.device());

  ctc_decoder::init_streaming_state_paged(
      state_buffer.data_ptr(),
      static_cast<int>(batch), static_cast<int>(beam),
      static_cast<int>(vocab_size), static_cast<int>(max_seq_len),
      static_cast<int>(blank_id),
      static_cast<int>(page_size), 0,  // num_pages=0 → auto
      stream);
}

// =============================================================================
// Streaming: process a whole chunk of frames in one call
// =============================================================================
//
// Replaces the per-frame Python loop in ``GpuStreamingDecoder.decode_chunk``
// with a single C++ launcher that iterates ``chunk_T`` frames and calls
// ``streaming_step`` once per active frame.  Eliminates ~10 μs of Python
// overhead per frame (Python→C++ trip + tensor-slice bookkeeping) which was
// dominating wall time for batched streaming workloads.
//
// ``is_speech_mask`` is an optional CPU uint8 / bool tensor of length
// ``chunk_T``: when present, frames where the mask is 0 are skipped (the
// caller pre-computed the blank-threshold mask on GPU and copied it once).
// When empty, every frame is decoded.
//
// Returns the new ``step`` (= start_step + #active frames decoded, capped at
// ``max_seq_len``).  ``frame_idx`` after the chunk is always
// ``start_frame_idx + chunk_T`` (or earlier if ``max_seq_len`` was reached);
// the caller can compute it without a return value.
int64_t ctc_beam_search_chunk(TensorView state_buffer, TensorView log_prob_chunk,
                              TensorView is_speech_mask,
                              int64_t beam, int64_t blank_id,
                              int64_t start_step, double blank_threshold,
                              int64_t start_frame_idx,
                              int64_t batch, int64_t vocab_size,
                              int64_t max_seq_len, int64_t use_paged_memory,
                              int64_t page_size) {
  CHECK_INPUT(state_buffer);
  CHECK_INPUT(log_prob_chunk);
  CHECK_DIM(3, log_prob_chunk);

  TVM_FFI_ICHECK(log_prob_chunk.dtype().code == kDLFloat &&
                  log_prob_chunk.dtype().bits == 32)
      << "log_prob_chunk must be float32";

  int chunk_t = log_prob_chunk.size(1);
  int batch_stride = log_prob_chunk.stride(0);
  int seq_stride = log_prob_chunk.stride(1);
  int vocab_stride = log_prob_chunk.stride(2);
  const float* lp_data = static_cast<const float*>(log_prob_chunk.data_ptr());

  const uint8_t* mask_data = nullptr;
  if (is_speech_mask.numel() > 0) {
    TVM_FFI_ICHECK(is_speech_mask.dtype().code == kDLUInt &&
                    is_speech_mask.dtype().bits == 8)
        << "is_speech_mask must be uint8";
    TVM_FFI_ICHECK(is_speech_mask.size(0) == chunk_t)
        << "is_speech_mask length (" << is_speech_mask.size(0)
        << ") must equal chunk_T (" << chunk_t << ")";
    mask_data = static_cast<const uint8_t*>(is_speech_mask.data_ptr());
  }

  cudaStream_t stream = get_stream(state_buffer.device());

  int step = static_cast<int>(start_step);
  int frame_idx = static_cast<int>(start_frame_idx);
  for (int t = 0; t < chunk_t; ++t) {
    if (step >= max_seq_len) {
      break;
    }
    if (mask_data && !mask_data[t]) {
      ++frame_idx;
      continue;
    }
    const float* lp_frame = lp_data + static_cast<size_t>(t) * seq_stride;
    cudaError_t status = ctc_decoder::streaming_step(
        state_buffer.data_ptr(), lp_frame,
        batch_stride, vocab_stride,
        step,
        static_cast<int>(blank_id), -1,
        frame_idx,
        static_cast<int>(batch), static_cast<int>(beam),
        static_cast<int>(vocab_size), static_cast<int>(max_seq_len),
        static_cast<int>(use_paged_memory), static_cast<int>(page_size),
        0,  // num_pages=0 → auto
        stream);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "CTC chunk step failed: " << cudaGetErrorString(status);
    ++step;
    ++frame_idx;
  }
  return static_cast<int64_t>(step);
}

// =============================================================================
// Streaming: read results
// =============================================================================

void ctc_beam_search_read_state(TensorView out_tokens, TensorView out_lengths,
                                TensorView out_scores, TensorView state_buffer,
                                int64_t step, int64_t batch, int64_t beam,
                                int64_t vocab_size, int64_t max_seq_len,
                                int64_t use_paged_memory, int64_t page_size) {
  CHECK_INPUT(out_tokens);
  CHECK_INPUT(out_lengths);
  CHECK_INPUT(out_scores);
  CHECK_INPUT(state_buffer);
  CHECK_DIM(3, out_tokens);
  CHECK_DIM(2, out_lengths);
  CHECK_DIM(2, out_scores);

  int max_out_len = out_tokens.size(2);
  cudaStream_t stream = get_stream(state_buffer.device());

  cudaError_t status = ctc_decoder::read_streaming_results(
      state_buffer.data_ptr(),
      static_cast<int*>(out_tokens.data_ptr()),
      static_cast<int*>(out_lengths.data_ptr()),
      static_cast<float*>(out_scores.data_ptr()),
      max_out_len, static_cast<int>(step),
      static_cast<int>(batch), static_cast<int>(beam),
      static_cast<int>(vocab_size), static_cast<int>(max_seq_len),
      static_cast<int>(use_paged_memory), static_cast<int>(page_size),
      0,  // num_pages=0 → auto
      stream);

  TVM_FFI_ICHECK(status == cudaSuccess)
      << "CTC beam search read state failed: " << cudaGetErrorString(status);
}
