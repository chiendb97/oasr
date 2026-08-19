#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "decoder/wfst/config.h"
#include "decoder/wfst/graph.h"

namespace oasr::wfst {

// Unoptimized host oracle for pruned-intersection semantics. It applies strict
// dynamic-beam pruning, destination-state max recombination, and final/partial
// expansion rules. Viterbi backpointers preserve reference scores; tied paths
// may differ.
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
