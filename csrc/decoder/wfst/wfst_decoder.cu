// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// TVM-FFI launcher for the in-tree GPU WFST beam-search decoder.
//
// The decoder is a stateful C++ object (GpuDecoder: persistent device arena +
// uploaded graph + CUDA-graph capture caches + streaming lanes).  It is exposed to
// Python over TVM-FFI's typed-function ABI via opaque int64 handles (create/use/free),
// mirroring the GPU CTC decoder's JIT packaging while keeping the decoder core intact.
//
// Result marshalling: decode results are host-side after backtrack, so the caller
// pre-allocates CPU output tensors (words [.,cap] i32, word_lens i32, scores f64,
// meta [.,3] i32 = {ok, reached_final, overflow}) and the launcher fills them.

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <tvm/ffi/string.h>

#include "decoder/wfst/config.h"
#include "decoder/wfst/decoder.h"
#include "decoder/wfst/graph.h"
#include "tvm_ffi_utils.h"

namespace {

using oasr::wfst::DecodeResult;
using oasr::wfst::DecoderConfig;
using oasr::wfst::GpuDecoder;
using oasr::wfst::GraphImage;
using oasr::wfst::LoadGraphImage;

// Handle payloads.  A decoder co-owns the graph image (shared_ptr) so the graph
// outlives the decoder even if Python frees the graph handle first — the decoder
// keeps a borrowed pointer into it for aux (word) lookups during backtrack.
struct GraphHandle {
  std::shared_ptr<GraphImage> graph;
};
struct DecoderHandle {
  std::shared_ptr<GraphImage> graph;
  std::unique_ptr<GpuDecoder> dec;
  bool fp16 = false;  // the log-prob dtype the decoder reads
  int device = 0;
};

constexpr DLDataType kFloat64 = {kDLFloat, 64, 1};

inline GraphHandle* graph_of(int64_t handle) {
  auto* gh = reinterpret_cast<GraphHandle*>(handle);
  TVM_FFI_ICHECK(gh != nullptr) << "null graph handle";
  return gh;
}

inline DecoderHandle* decoder_of(int64_t handle) {
  auto* dh = reinterpret_cast<DecoderHandle*>(handle);
  TVM_FFI_ICHECK(dh != nullptr) << "null decoder handle";
  return dh;
}

inline void check_cpu(TensorView t, const char* what) {
  TVM_FFI_ICHECK(t.device().device_type == kDLCPU) << what << " must be a CPU tensor";
}

// The launchers read and write host buffers by raw pointer, so each must be a dense CPU
// tensor of the expected dtype and rank, with at least `rows` rows: a short or strided
// one was read or written out of bounds.
inline void check_host(TensorView t, DLDataType dtype, int ndim, int64_t rows,
                       const char* what) {
  check_cpu(t, what);
  TVM_FFI_ICHECK(t.dtype() == dtype) << what << " has the wrong dtype";
  TVM_FFI_ICHECK(t.ndim() == ndim) << what << " must be " << ndim << "-D";
  TVM_FFI_ICHECK(t.IsContiguous()) << what << " must be contiguous";
  TVM_FFI_ICHECK(t.size(0) >= rows) << what << " has fewer than " << rows << " rows";
}

// The device log-probs: packed [rows, T, V] on the decoder's device, in its dtype, wide
// enough for every label the graph reads.
inline void check_log_probs(const DecoderHandle& dh, TensorView lp) {
  CHECK_INPUT(lp);
  TVM_FFI_ICHECK(lp.ndim() == 3) << "log_probs must be [B, T, V]";
  TVM_FFI_ICHECK(lp.device().device_id == dh.device)
      << "log_probs are on cuda:" << lp.device().device_id << ", the decoder on cuda:"
      << dh.device;
  TVM_FFI_ICHECK(lp.dtype() == (dh.fp16 ? oasr::dl_float16 : oasr::dl_float32))
      << "log_probs must be " << (dh.fp16 ? "float16" : "float32") << " for this decoder";
  // The decode kernels read the tensor in place (device-resident descriptor).
  TVM_FFI_ICHECK(lp.IsContiguous()) << "log_probs must be a packed [B, T, V] tensor";
  TVM_FFI_ICHECK(lp.size(2) >= dh.graph->vocab_size)
      << "log_probs have " << lp.size(2) << " columns; the graph's labels need "
      << dh.graph->vocab_size;
}

// Read a CPU int32 1-D tensor of exactly n entries into a host vector.
inline std::vector<int32_t> read_i32(TensorView t, int64_t n, const char* what) {
  check_host(t, oasr::dl_int32, 1, n, what);
  TVM_FFI_ICHECK(t.size(0) == n) << what << " must have " << n << " entries";
  const int32_t* p = static_cast<const int32_t*>(t.data_ptr());
  return std::vector<int32_t>(p, p + n);
}

// Copy up to `cap` words into `dst`; write the TRUE word count into `*len_dst`.
// Reporting the true (unclamped) length lets the caller detect truncation
// (len_dst > cap) and re-issue with a larger buffer.
inline void write_words(int32_t* dst, int64_t cap, int32_t* len_dst,
                        const std::vector<int32_t>& words) {
  int64_t n = std::min<int64_t>(static_cast<int64_t>(words.size()), cap);
  for (int64_t i = 0; i < n; ++i) dst[i] = words[i];
  *len_dst = static_cast<int32_t>(words.size());
}

}  // namespace

// ---------------------------------------------------------------------------
// Graph lifecycle
// ---------------------------------------------------------------------------

int64_t wfst_load_graph(tvm::ffi::String path) {
  std::shared_ptr<GraphImage> g = LoadGraphImage(std::string(path.data(), path.size()));
  return reinterpret_cast<int64_t>(new GraphHandle{std::move(g)});
}

void wfst_free_graph(int64_t graph_handle) {
  delete reinterpret_cast<GraphHandle*>(graph_handle);
}

// out_info: CPU int64 [5] = {num_states, num_arcs, vocab_size, start_state, finals_at_end}
void wfst_graph_info(int64_t graph_handle, TensorView out_info) {
  GraphHandle* gh = graph_of(graph_handle);
  check_host(out_info, {kDLInt, 64, 1}, 1, 5, "out_info");
  int64_t* p = static_cast<int64_t*>(out_info.data_ptr());
  p[0] = gh->graph->num_states;
  p[1] = gh->graph->num_arcs;
  p[2] = gh->graph->vocab_size;
  p[3] = gh->graph->start_state;
  p[4] = gh->graph->finals_at_end ? 1 : 0;
}

// ---------------------------------------------------------------------------
// Decoder lifecycle
// ---------------------------------------------------------------------------

int64_t wfst_create_decoder(int64_t graph_handle, double search_beam, double output_beam,
                            int64_t min_active, int64_t max_active, int64_t allow_partial,
                            int64_t max_lanes, int64_t max_frames, int64_t device,
                            int64_t main_q_factor, int64_t cand_factor, int64_t use_cuda_graphs,
                            int64_t lattice, int64_t fp16_logprobs, int64_t streaming,
                            int64_t lat_prune_interval, int64_t eps_iterations,
                            int64_t arena_budget_entries, int64_t stream_log_entries,
                            int64_t gc_interval) {
  GraphHandle* gh = graph_of(graph_handle);

  GpuDecoder::Options opts;
  opts.cfg.search_beam = static_cast<float>(search_beam);
  opts.cfg.output_beam = static_cast<float>(output_beam);
  opts.cfg.min_active_states = static_cast<int32_t>(min_active);
  opts.cfg.max_active_states = static_cast<int32_t>(max_active);
  opts.cfg.allow_partial = allow_partial != 0;
  opts.cfg.max_lanes = static_cast<int32_t>(max_lanes);
  opts.cfg.max_frames = static_cast<int32_t>(max_frames);
  opts.cfg.main_q_factor = static_cast<int32_t>(main_q_factor);
  opts.cfg.cand_factor = static_cast<int32_t>(cand_factor);
  opts.cfg.lattice = lattice != 0;
  opts.cfg.fp16_logprobs = fp16_logprobs != 0;
  opts.cfg.streaming = streaming != 0;
  opts.cfg.lat_prune_interval = static_cast<int32_t>(lat_prune_interval);
  opts.cfg.eps_iterations = static_cast<int32_t>(eps_iterations);
  opts.cfg.arena_budget_entries = arena_budget_entries;
  opts.cfg.stream_log_entries = stream_log_entries;
  opts.cfg.gc_interval = static_cast<int32_t>(gc_interval);
  opts.device = static_cast<int>(device);
  opts.debug_snapshots = false;
  opts.use_cuda_graphs = use_cuda_graphs != 0;

  auto* dh = new DecoderHandle{gh->graph, std::make_unique<GpuDecoder>(*gh->graph, opts),
                               opts.cfg.fp16_logprobs, opts.device};
  return reinterpret_cast<int64_t>(dh);
}

void wfst_free_decoder(int64_t handle) { delete reinterpret_cast<DecoderHandle*>(handle); }

// out_stats: CPU int64 [4] = {reserved_bytes, committed_bytes, fixed_bytes,
// arena_high_water_entries} for the decoder's device memory.
void wfst_decoder_mem_stats(int64_t handle, TensorView out_stats) {
  DecoderHandle* dh = decoder_of(handle);
  check_host(out_stats, {kDLInt, 64, 1}, 1, 4, "out_stats");
  const GpuDecoder::MemStats s = dh->dec->GetMemStats();
  int64_t* p = static_cast<int64_t*>(out_stats.data_ptr());
  p[0] = s.reserved_bytes;
  p[1] = s.committed_bytes;
  p[2] = s.fixed_bytes;
  p[3] = s.arena_high_water;
}

// ---------------------------------------------------------------------------
// Offline batched decode
// ---------------------------------------------------------------------------
// log_probs: CUDA [B, T, V] (fp32, or fp16 if the decoder was built fp16).
// lengths:   CPU int32 [B].
// out_words: CPU int32 [B, cap]; out_word_lens: CPU int32 [B];
// out_scores: CPU f64 [B]; out_meta: CPU int32 [B, 3] = {ok, reached_final, overflow}.
void wfst_decode_batch(int64_t handle, TensorView log_probs, TensorView lengths,
                       TensorView out_words, TensorView out_word_lens, TensorView out_scores,
                       TensorView out_meta) {
  DecoderHandle* dh = decoder_of(handle);
  check_log_probs(*dh, log_probs);
  const int64_t batch = log_probs.size(0);
  const int64_t max_frames = log_probs.size(1);
  const int64_t vocab = log_probs.size(2);
  std::vector<int32_t> frames = read_i32(lengths, batch, "lengths");
  check_host(out_words, oasr::dl_int32, 2, batch, "out_words");
  check_host(out_word_lens, oasr::dl_int32, 1, batch, "out_word_lens");
  check_host(out_scores, kFloat64, 1, batch, "out_scores");
  check_host(out_meta, oasr::dl_int32, 2, batch, "out_meta");
  TVM_FFI_ICHECK(out_meta.size(1) == 3) << "out_meta must be [B, 3]";
  const int64_t cap = out_words.size(1);

  std::vector<DecodeResult> res =
      dh->dec->DecodeBatch(log_probs.data_ptr(), batch, max_frames, vocab, frames);

  int32_t* wptr = static_cast<int32_t*>(out_words.data_ptr());
  int32_t* wlen = static_cast<int32_t*>(out_word_lens.data_ptr());
  double* sc = static_cast<double*>(out_scores.data_ptr());
  int32_t* meta = static_cast<int32_t*>(out_meta.data_ptr());
  for (int64_t b = 0; b < batch; ++b) {
    const DecodeResult& r = res[b];
    write_words(wptr + b * cap, cap, wlen + b, r.words);
    sc[b] = r.score;
    meta[b * 3 + 0] = r.ok ? 1 : 0;
    meta[b * 3 + 1] = r.reached_final ? 1 : 0;
    meta[b * 3 + 2] = static_cast<int32_t>(r.overflow);
  }
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

int64_t wfst_create_stream(int64_t handle) { return decoder_of(handle)->dec->CreateStream(); }

void wfst_release_stream(int64_t handle, int64_t channel) {
  decoder_of(handle)->dec->ReleaseStream(static_cast<int32_t>(channel));
}

// channels: CPU int32 [C]; log_probs: CUDA [C, Tc, V]; lengths: CPU int32 [C].
// out_words: CPU int32 [C, cap]; out_word_lens: CPU int32 [C];
// out_channels: CPU int32 [C]; out_overflow: CPU int32 [C].
void wfst_advance_chunk(int64_t handle, TensorView channels, TensorView log_probs,
                        TensorView lengths, int64_t want_partial, TensorView out_words,
                        TensorView out_word_lens, TensorView out_channels,
                        TensorView out_overflow) {
  DecoderHandle* dh = decoder_of(handle);
  check_log_probs(*dh, log_probs);
  const int64_t num = log_probs.size(0);
  std::vector<int32_t> chans = read_i32(channels, num, "channels");
  std::vector<int32_t> lens = read_i32(lengths, num, "lengths");
  check_host(out_words, oasr::dl_int32, 2, num, "out_words");
  check_host(out_word_lens, oasr::dl_int32, 1, num, "out_word_lens");
  check_host(out_channels, oasr::dl_int32, 1, num, "out_channels");
  check_host(out_overflow, oasr::dl_int32, 1, num, "out_overflow");
  const int64_t chunk_frames = log_probs.size(1);
  const int64_t vocab = log_probs.size(2);
  const int64_t cap = out_words.size(1);

  std::vector<GpuDecoder::StreamPartial> parts = dh->dec->AdvanceChunk(
      chans, log_probs.data_ptr(), chunk_frames, vocab, lens, want_partial != 0);

  int32_t* wptr = static_cast<int32_t*>(out_words.data_ptr());
  int32_t* wlen = static_cast<int32_t*>(out_word_lens.data_ptr());
  int32_t* outc = static_cast<int32_t*>(out_channels.data_ptr());
  int32_t* ovf = static_cast<int32_t*>(out_overflow.data_ptr());
  for (size_t i = 0; i < parts.size(); ++i) {
    write_words(wptr + static_cast<int64_t>(i) * cap, cap, wlen + i, parts[i].words);
    outc[i] = parts[i].channel;
    ovf[i] = static_cast<int32_t>(parts[i].overflow);
  }
}

// out_words: CPU int32 [cap]; out_word_len: CPU int32 [1]; out_score: CPU f64 [1];
// out_meta: CPU int32 [3].
void wfst_finalize_stream(int64_t handle, int64_t channel, TensorView out_words,
                          TensorView out_word_len, TensorView out_score, TensorView out_meta) {
  DecoderHandle* dh = decoder_of(handle);
  check_host(out_words, oasr::dl_int32, 1, 0, "out_words");
  check_host(out_word_len, oasr::dl_int32, 1, 1, "out_word_len");
  check_host(out_score, kFloat64, 1, 1, "out_score");
  check_host(out_meta, oasr::dl_int32, 1, 3, "out_meta");
  DecodeResult r = dh->dec->FinalizeStream(static_cast<int32_t>(channel));
  const int64_t cap = out_words.size(0);
  write_words(static_cast<int32_t*>(out_words.data_ptr()), cap,
              static_cast<int32_t*>(out_word_len.data_ptr()), r.words);
  static_cast<double*>(out_score.data_ptr())[0] = r.score;
  int32_t* meta = static_cast<int32_t*>(out_meta.data_ptr());
  meta[0] = r.ok ? 1 : 0;
  meta[1] = r.reached_final ? 1 : 0;
  meta[2] = static_cast<int32_t>(r.overflow);
}
