#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "decoder/wfst/config.h"
#include "decoder/wfst/graph.h"

namespace oasr::wfst {

// Per-utterance decode outcome (host-side, after backtrack).
struct DecodeResult {
  bool ok = false;
  bool reached_final = false;
  double score = 0.0;
  std::vector<int32_t> arc_path;  // graph arc ids, time order
  std::vector<int32_t> words;     // aux_pool values > 0 along the path
  uint32_t overflow = 0;          // bit0 cand, bit1 hash, bit2 winners-log, bit3 frontier
  // Debug (cfg.debug_snapshots): per-frame frontier after recombination, sorted by state.
  // snapshots[0] = start frontier; snapshots[t+1] = after frame t; last = final tokens.
  std::vector<std::vector<std::pair<int32_t, float>>> snapshots;
};

// Batched GPU decoder. Allocates its workspace once at construction (no allocations in
// DecodeBatch). Not thread-safe; one instance per stream/GPU.
class GpuDecoder {
 public:
  struct Options {
    DecoderConfig cfg;
    int device = 0;
    bool debug_snapshots = false;   // copy per-frame frontiers back (tests only)
    bool use_cuda_graphs = true;    // whole-batch graph replay (auto-off with snapshots)
  };

  GpuDecoder(const GraphImage& graph, const Options& opts);
  ~GpuDecoder();
  GpuDecoder(const GpuDecoder&) = delete;
  GpuDecoder& operator=(const GpuDecoder&) = delete;

  // log_probs: device pointer, fp32 (or fp16 when cfg.fp16_logprobs), [batch,
  // max_frames, vocab_stride] contiguous. frames[b] = real frame count of lane b
  // (<= max_frames). batch <= cfg.max_lanes. Synchronizes at the end (single D2H);
  // returns per-lane results.
  std::vector<DecodeResult> DecodeBatch(const void* d_log_probs, int64_t batch,
                                        int64_t max_frames, int64_t vocab_stride,
                                        const std::vector<int32_t>& frames);

  // ---- Streaming API (cfg.streaming; lattice unsupported in streaming v1) ----
  // A channel occupies one lane for its lifetime (max cfg.max_lanes concurrent).
  struct StreamPartial {
    int32_t channel = -1;
    std::vector<int32_t> words;  // best-path words so far (empty when not requested)
    uint32_t overflow = 0;
  };
  // Returns a channel id, or -1 if all lanes are busy.
  int32_t CreateStream();
  void ReleaseStream(int32_t channel);
  // log_probs: device, [channels.size(), chunk_frames, vocab_stride] contiguous, dtype
  // per cfg.fp16_logprobs; lens[i] <= chunk_frames <= cfg.max_frames.
  std::vector<StreamPartial> AdvanceChunk(const std::vector<int32_t>& channels,
                                          const void* d_log_probs, int64_t chunk_frames,
                                          int64_t vocab_stride,
                                          const std::vector<int32_t>& lens,
                                          bool want_partial);
  // Runs the k2 final-frame step (beam 1e10, allow_partial semantics) and backtracks.
  // The channel stays allocated until ReleaseStream.
  DecodeResult FinalizeStream(int32_t channel);

  // Lattice mode (cfg.lattice): flat records of the last batch's output-beam-pruned
  // lattice arcs, 8 x i32 each: {src_tok, dst_tok, label, arc_map (graph arc id),
  // score_bits (f32; graph+acoustic), t, lane, 0}. Token ids are arena-global; each
  // token belongs to one (lane, frame).
  const std::vector<int32_t>& LastLatticeRecords() const;

  // Device-memory accounting. The winners/lattice arenas and the log-prob staging are
  // lazily-committed regions: `reserved_bytes` is their stable virtual reservation,
  // `committed_bytes` the physical memory actually mapped; `fixed_bytes` covers the
  // eagerly-allocated workspace plus the graph image on device.
  struct MemStats {
    int64_t reserved_bytes = 0;
    int64_t committed_bytes = 0;
    int64_t fixed_bytes = 0;
    int64_t arena_high_water = 0;  // max end-of-batch winners-arena entries observed
  };
  MemStats GetMemStats() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace oasr::wfst
