#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "oasr/wfst/config.h"
#include "decoder/wfst/graph.h"

namespace oasr::wfst {

// Exact-semantics reference decoder (host, unoptimized). Defines the target behavior the
// GPU decoder must reproduce: k2 intersect_dense_pruned forward-pass semantics
// (see docs/DESIGN.md and k2/csrc/intersect_dense_pruned.cu):
//   - scores are log-likes, maximized (Viterbi in the max-tropical sense)
//   - per-frame: expand emitting arcs of the frontier; exact per-frame best; dynamic-beam
//     update (verbatim k2 formula, offline flavor) -> cutoff = best - beam; keep
//     candidates with end > cutoff (strict); recombine per dest state by max
//   - final step (t == T): if any active state has a real final (-1) arc or
//     !allow_partial: expand only final arcs (acoustic 0); else redirect ALL arcs of
//     active states to the super-final state with acoustic 0 (k2 allow_partial rule);
//     beam = 1e10 (no pruning)
// Unlike k2 (which derives 1-best from the lattice), this tracks Viterbi argmax
// backpointers; path scores match k2 shortest_path scores exactly, paths match up to
// exact score ties.
struct CpuDecodeResult {
  bool ok = false;                 // false: frontier died and allow_partial produced nothing
  bool reached_final = false;      // ended via a real -1 arc (vs allow_partial redirect)
  double score = 0.0;              // total best-path score (graph + acoustic)
  std::vector<int32_t> arc_path;   // graph arc ids along the best path, in time order
  std::vector<int32_t> words;      // aux_pool values > 0 along the path
  // Per-frame surviving frontier AFTER recombination, sorted by state id:
  // frames[t] = {(state, forward_score)}; frames[0] is the pre-frame-0 start frontier.
  std::vector<std::vector<std::pair<int32_t, float>>> frames;
};

// log_probs: [T, vocab_stride] row-major fp32, labels index columns directly.
// online=true mirrors the streaming decoder (k2 online beam semantics mid-stream; the
// final step still applies the final-frame rules).
CpuDecodeResult CpuDecode(const GraphImage& g, const float* log_probs, int32_t num_frames,
                          int32_t vocab_stride, const DecoderConfig& cfg,
                          bool online = false);

}  // namespace oasr::wfst
