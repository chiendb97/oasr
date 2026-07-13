#pragma once

#include <cstdint>

namespace oasr::wfst {

// Decoding configuration. Names/defaults mirror k2 intersect_dense_pruned / oasr.
struct DecoderConfig {
  float search_beam = 20.0f;
  float output_beam = 8.0f;
  int32_t min_active_states = 30;
  int32_t max_active_states = 10000;
  bool allow_partial = true;
  int32_t eps_iterations = 3;  // epsilon-closure passes per frame (TLG graphs; no-op on
                               // epsilon-free graphs). Bounds eps-chain depth.

  // Capacity policy (per lane). main_q bounds the per-frame surviving frontier;
  // cand_cap bounds raw in-beam expansion candidates per frame.
  int32_t main_q_factor = 4;   // main_q = main_q_factor * max_active_states
  int32_t cand_factor = 6;     // cand_cap = cand_factor * main_q  (u2pp HLG mean degree 18.8;
                               // loose-admit superset needs headroom over the exact set)

  int32_t max_lanes = 32;      // max concurrent utterances per batch
  int32_t max_frames = 4096;   // max T per utterance (final step excluded)

  int32_t gc_interval = 0;  // >0 (even): winners-log GC every N offline steps. 1-best
                            // runs a segmented host loop; interval-lattice mode GCs at
                            // its prune points. Frees the finalized log prefix, so long
                            // audio needs O(live window) winners memory instead of O(T).

  // Winners-log budgets (entries of 8 bytes). 0 keeps the built-in formulas:
  // arena budget = min(512Mi, max(64Mi, 16Mi * max_lanes)); per-channel streaming
  // region = arena budget / max_lanes. The streaming region is rounded up to whole
  // physical mapping chunks (its slices commit/unmap per channel).
  int64_t arena_budget_entries = 0;
  int64_t stream_log_entries = 0;

  bool lattice = false;        // persist per-frame candidates for lattice extraction
  int32_t lat_prune_interval = 0;  // >0 (even): prune the lattice arena every N frames
                                   // (k2-style window-loose rule; bounds memory for long
                                   // audio; the final exact pass is unchanged)
  bool fp16_logprobs = false;  // consume fp16 log-probs (f32 accumulation inside)
  bool streaming = false;      // chunked decoding (k2 online beam semantics; per-lane
                               // winners regions; max_frames = max CHUNK length)

  int32_t main_q_capacity() const { return main_q_factor * max_active_states; }
  int32_t cand_capacity() const { return cand_factor * main_q_capacity(); }
};

}  // namespace oasr::wfst
