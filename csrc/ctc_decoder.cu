// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// TVM-FFI launcher layer for GPU CTC prefix beam search decoder.

#include <oasr/ctc_decoder.cuh>

#include <functional>
#include <mutex>
#include <unordered_map>

#include "tvm_ffi_utils.h"

using namespace oasr;

// =============================================================================
// Step 4: per-state captured CTC graphs
// -----------------------------------------------------------------------------
// Each ``StreamState`` lazily captures four CUDA Graphs on its first call
// with ``use_cuda_graphs=true``:
//
//   * graph_first — captures ``streaming_step_persistent(step_for_parity=0)``
//     plus ``advance_counters``.  Used for the *step==0* frame, which goes
//     through the ``first_step_kernel`` and writes to ``clen[0]/clist[0]``.
//   * graph_odd   — captures ``streaming_step_persistent(step_for_parity=1)``
//     plus ``advance_counters``.  Used for steps 1, 3, 5, …: ``src_parity=0``,
//     ``dst_parity=1``.
//   * graph_even  — captures ``streaming_step_persistent(step_for_parity=2)``
//     plus ``advance_counters``.  Used for steps 2, 4, 6, …: ``src_parity=1``,
//     ``dst_parity=0``.
//   * graph_blank — captures only ``advance_frame_idx_kernel``.  Used for
//     blank frames (where the host-precomputed mask says skip).
//
// Per-frame the host loop becomes ``cudaMemcpyAsync(d_lp_frame_buf, ...) +
// cudaGraphLaunch(picked_graph)`` — two launches replace the six kernels of
// the eager path.  All step-aware kernels read ``*d_step`` at block entry
// (Step 3 refactor) so the same captured graph is valid for every step at
// the matching parity.
// =============================================================================

namespace {

struct CtcStreamGraphCache {
    int batch = 0;
    int beam = 0;
    int vocab_size = 0;
    int max_seq_len = 0;
    int use_paged_memory = 0;
    int page_size = 0;
    int blank_id = 0;
    bool captured = false;
    cudaGraphExec_t graph_first = nullptr;
    cudaGraphExec_t graph_even = nullptr;
    cudaGraphExec_t graph_odd = nullptr;
    // One log-prob frame as bytes for the per-replay D2D copy.
    size_t lp_frame_bytes = 0;
    // d_lp_frame_buf address (snapshot of data.d_lp_frame_buf after
    // setup_internal_data_pointers).  The same address is baked into the
    // captured kernel launches, so the per-frame memcpy must target it.
    void* d_lp_frame_buf = nullptr;
    // Pinned host int2: ``[step, frame_idx]``.  Captured graphs start with
    // an ``cudaMemcpyAsync(d_step_ptr, pinned_counters_host, 8 bytes, H2D)``
    // node that reads from this stable address.  The chunk launcher writes
    // the current host counters here before each non-blank replay; blank
    // frames just bump a host-side counter without launching anything.
    // Removes the per-blank-frame ``advance_frame_idx`` launch entirely.
    int* pinned_counters_host = nullptr;
};

std::mutex g_ctc_graph_mutex;
std::unordered_map<void*, CtcStreamGraphCache> g_ctc_graph_cache;

// Capture one graph by running ``work`` inside a stream-capture region.
// The capture stream must not be the default stream.
cudaError_t capture_one_graph(cudaGraphExec_t* out, cudaStream_t capture_stream,
                              const std::function<cudaError_t()>& work) {
    cudaError_t err = cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeThreadLocal);
    if (err != cudaSuccess) return err;
    cudaError_t werr = work();
    cudaGraph_t graph = nullptr;
    cudaError_t end_err = cudaStreamEndCapture(capture_stream, &graph);
    if (werr != cudaSuccess) {
        if (graph) cudaGraphDestroy(graph);
        return werr;
    }
    if (end_err != cudaSuccess) return end_err;
    err = cudaGraphInstantiate(out, graph, nullptr, nullptr, 0);
    cudaGraphDestroy(graph);
    return err;
}

cudaError_t ensure_graphs_captured(CtcStreamGraphCache* cache, void* state_ptr) {
    if (cache->captured) return cudaSuccess;

    // Pinned host int2 holds the (step, frame_idx) values the captured H2D
    // node reads at every replay.  Stable address for the cache's lifetime.
    if (cache->pinned_counters_host == nullptr) {
        cudaError_t herr = cudaMallocHost(
            reinterpret_cast<void**>(&cache->pinned_counters_host),
            sizeof(int) * 2);
        if (herr != cudaSuccess) return herr;
    }

    cudaStream_t capture_stream = nullptr;
    cudaError_t err = cudaStreamCreate(&capture_stream);
    if (err != cudaSuccess) return err;

    void* workspace =
        reinterpret_cast<char*>(state_ptr) + ctc_decoder::STATE_HEADER_SIZE;
    ctc_decoder::InternalData data;
    if (cache->use_paged_memory) {
        ctc_decoder::setup_internal_data_paged_pointers(
            &data, workspace, cache->batch, cache->beam, cache->vocab_size,
            cache->max_seq_len, cache->page_size, 0);
    } else {
        ctc_decoder::setup_internal_data_pointers(
            &data, workspace, cache->batch, cache->beam, cache->vocab_size,
            cache->max_seq_len);
    }

    cache->d_lp_frame_buf = data.d_lp_frame_buf;
    cache->lp_frame_bytes =
        sizeof(float) * static_cast<size_t>(cache->batch) * cache->vocab_size;

    const int batch_stride = cache->vocab_size;  // d_lp_frame_buf is (batch, vocab_size)
    const int vocab_stride = 1;

    // The captured graph's first op is a tiny H2D copy that refreshes the
    // device-resident counters from the pinned host buffer.  At replay
    // time it reads whatever the chunk-launcher wrote into
    // ``pinned_counters_host`` just before ``cudaGraphLaunch``.  The
    // streaming kernels then read those values via ``__ldg(d_step)``.
    // Because the captured launch arguments (source pointer = pinned host
    // address, destination pointer = state-relative device counter) are
    // stable, this is replay-safe across all frames and parities.
    auto launch_step = [&](int step_for_parity) -> cudaError_t {
        cudaError_t e = cudaMemcpyAsync(
            ctc_decoder::device_step_ptr(state_ptr),
            cache->pinned_counters_host, sizeof(int) * 2,
            cudaMemcpyHostToDevice, capture_stream);
        if (e != cudaSuccess) return e;
        return ctc_decoder::streaming_step_persistent(
            state_ptr, data.d_lp_frame_buf, batch_stride, vocab_stride,
            cache->blank_id, -1, cache->batch, cache->beam, cache->vocab_size,
            cache->max_seq_len, cache->use_paged_memory, cache->page_size, 0,
            step_for_parity, capture_stream);
    };

    err = capture_one_graph(&cache->graph_first, capture_stream,
                            [&]() { return launch_step(0); });
    if (err != cudaSuccess) goto cleanup;

    err = capture_one_graph(&cache->graph_odd, capture_stream,
                            [&]() { return launch_step(1); });
    if (err != cudaSuccess) goto cleanup;

    err = capture_one_graph(&cache->graph_even, capture_stream,
                            [&]() { return launch_step(2); });
    if (err != cudaSuccess) goto cleanup;

    cache->captured = true;

cleanup:
    cudaStreamDestroy(capture_stream);
    return err;
}

void destroy_graph_cache_entry(CtcStreamGraphCache* cache) {
    if (cache->graph_first) cudaGraphExecDestroy(cache->graph_first);
    if (cache->graph_even) cudaGraphExecDestroy(cache->graph_even);
    if (cache->graph_odd) cudaGraphExecDestroy(cache->graph_odd);
    cache->graph_first = cache->graph_even = cache->graph_odd = nullptr;
    if (cache->pinned_counters_host) {
        cudaFreeHost(cache->pinned_counters_host);
        cache->pinned_counters_host = nullptr;
    }
    cache->captured = false;
}

}  // namespace

// Public: release the captured graphs for a given state buffer pointer.
// Called from Python when a ``StreamState`` is reset (shape change) or
// destroyed (request finalised) so cudaGraphExec_t handles don't leak.
void ctc_decoder_release_graphs(TensorView state_buffer) {
    std::lock_guard<std::mutex> lock(g_ctc_graph_mutex);
    auto it = g_ctc_graph_cache.find(state_buffer.data_ptr());
    if (it == g_ctc_graph_cache.end()) return;
    destroy_graph_cache_entry(&it->second);
    g_ctc_graph_cache.erase(it);
}

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
                              int64_t page_size, int64_t use_cuda_graphs) {
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

  // Host-side counters drive the loop.  Blank frames simply increment
  // ``frame_idx`` (no kernel launch).  Non-blank frames either (a) push the
  // current ``(step, frame_idx)`` into the captured graph's pinned host
  // buffer + ``cudaGraphLaunch`` (graph fast path), or (b) write to the
  // device counters via ``set_stream_counters`` + run ``streaming_step_persistent``
  // (eager fallback).  Pre-Step-4 launched one kernel per blank frame; for
  // 80%-blank streams that was the dominant launch-overhead source.
  int step = static_cast<int>(start_step);
  int frame_idx = static_cast<int>(start_frame_idx);

  // ----- Graph-captured fast path -----
  // When the caller opted in via ``use_cuda_graphs=1``, lazily capture the
  // three per-state non-blank graphs (first / odd / even).  Each captured
  // graph begins with an H2D ``cudaMemcpyAsync`` that refreshes the device
  // counters from the per-state pinned host buffer; per non-blank frame we
  // do one D2D for the log-prob slice + one ``cudaGraphLaunch``.  Falls
  // through to the eager path on capture failure (graph cache not populated)
  // so behaviour is safe by default.
  CtcStreamGraphCache* cache = nullptr;
  if (use_cuda_graphs) {
    std::lock_guard<std::mutex> lock(g_ctc_graph_mutex);
    auto& entry = g_ctc_graph_cache[state_buffer.data_ptr()];
    // Stale-entry guard: if the same data_ptr was previously captured for a
    // different shape (torch's caching allocator can hand out a freed
    // address to a new state) drop the old graphs and recapture.  Python
    // also issues an explicit release on reset_state when the buffer is
    // reallocated; this is the defence-in-depth path.
    if (entry.captured &&
        (entry.batch != static_cast<int>(batch) ||
         entry.beam != static_cast<int>(beam) ||
         entry.vocab_size != static_cast<int>(vocab_size) ||
         entry.max_seq_len != static_cast<int>(max_seq_len) ||
         entry.use_paged_memory != static_cast<int>(use_paged_memory) ||
         entry.page_size != static_cast<int>(page_size) ||
         entry.blank_id != static_cast<int>(blank_id))) {
      destroy_graph_cache_entry(&entry);
    }
    if (!entry.captured) {
      entry.batch = static_cast<int>(batch);
      entry.beam = static_cast<int>(beam);
      entry.vocab_size = static_cast<int>(vocab_size);
      entry.max_seq_len = static_cast<int>(max_seq_len);
      entry.use_paged_memory = static_cast<int>(use_paged_memory);
      entry.page_size = static_cast<int>(page_size);
      entry.blank_id = static_cast<int>(blank_id);
      cudaError_t cerr = ensure_graphs_captured(&entry, state_buffer.data_ptr());
      if (cerr != cudaSuccess) {
        // Capture failed (e.g. CUDA Graph not supported in this context).
        // Drop the entry and fall through to the eager loop so the caller
        // still gets a correct result.
        destroy_graph_cache_entry(&entry);
        g_ctc_graph_cache.erase(state_buffer.data_ptr());
        cache = nullptr;
      } else {
        cache = &entry;
      }
    } else {
      cache = &entry;
    }
  }

  if (cache != nullptr) {
    const size_t lp_row_bytes = sizeof(float) * static_cast<size_t>(vocab_size);
    for (int t = 0; t < chunk_t; ++t) {
      if (step >= max_seq_len) break;
      if (mask_data && !mask_data[t]) {
        // Blank: host-only frame_idx increment.  The next non-blank's
        // captured H2D will push the updated value to the device.
        ++frame_idx;
        continue;
      }
      // Update the pinned host counters that the captured graph's H2D
      // reads at replay time.  This is a 2×int CPU write, no kernel launch.
      cache->pinned_counters_host[0] = step;
      cache->pinned_counters_host[1] = frame_idx;
      // Refresh ``d_lp_frame_buf`` from this frame's log-prob slice.
      const float* lp_frame = lp_data + static_cast<size_t>(t) * seq_stride;
      cudaError_t cerr = cudaMemcpy2DAsync(
          cache->d_lp_frame_buf, lp_row_bytes,
          lp_frame, static_cast<size_t>(batch_stride) * sizeof(float),
          lp_row_bytes, static_cast<size_t>(batch),
          cudaMemcpyDeviceToDevice, stream);
      TVM_FFI_ICHECK(cerr == cudaSuccess)
          << "CTC chunk D2D copy failed: " << cudaGetErrorString(cerr);
      cudaGraphExec_t g;
      if (step == 0) {
        g = cache->graph_first;
      } else if (step & 1) {
        g = cache->graph_odd;
      } else {
        g = cache->graph_even;
      }
      cudaError_t lerr = cudaGraphLaunch(g, stream);
      TVM_FFI_ICHECK(lerr == cudaSuccess)
          << "CTC chunk graph launch failed: " << cudaGetErrorString(lerr);
      ++step;
      ++frame_idx;
    }
    return static_cast<int64_t>(step);
  }

  // ----- Eager fallback -----
  // Blanks: host-only ``++frame_idx`` (no kernel launch).  Non-blanks: one
  // ``set_stream_counters`` to refresh the device counters that the
  // step-aware kernels read via ``__ldg(d_step)``, then
  // ``streaming_step_persistent`` (no internal counter advance).  Trades
  // pre-Step-3's per-frame host scalar args for one extra counter launch per
  // non-blank, but recovers the per-blank no-op behaviour that Step 3
  // accidentally regressed.
  for (int t = 0; t < chunk_t; ++t) {
    if (step >= max_seq_len) {
      break;
    }
    if (mask_data && !mask_data[t]) {
      ++frame_idx;
      continue;
    }
    ctc_decoder::set_stream_counters(state_buffer.data_ptr(), step, frame_idx,
                                     stream);
    const float* lp_frame = lp_data + static_cast<size_t>(t) * seq_stride;
    cudaError_t status = ctc_decoder::streaming_step_persistent(
        state_buffer.data_ptr(), lp_frame,
        batch_stride, vocab_stride,
        static_cast<int>(blank_id), -1,
        static_cast<int>(batch), static_cast<int>(beam),
        static_cast<int>(vocab_size), static_cast<int>(max_seq_len),
        static_cast<int>(use_paged_memory), static_cast<int>(page_size),
        0,  // num_pages=0 → auto
        step,  // host parity / step==0 selector
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
