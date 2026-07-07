#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace oasr::wfst {

// Host-side graph image, loaded from an hlg.img file (see python/wfst_decoder/
// graph_image.py for the canonical format definition). Arc order is identical to the
// source k2 FSA; arc index == k2 graph arc index.
struct GraphImage {
  int64_t num_states = 0;
  int64_t num_arcs = 0;
  int32_t vocab_size = 0;
  int32_t start_state = 0;
  bool finals_at_end = false;
  bool has_eps = false;   // epsilon (non-emitting) arcs present (TLG graphs)
  bool eps_first = false; // eps arcs at the START of the non-final range

  std::vector<int32_t> row_splits;       // [num_states + 1]
  std::vector<int32_t> final_count;      // [num_states]
  std::vector<int32_t> arc_dest_ilabel;  // [2 * num_arcs] interleaved {dest, ilabel}
  std::vector<float> arc_weight;         // [num_arcs]
  std::vector<int32_t> aux_row_splits;   // [num_arcs + 1]
  std::vector<int32_t> aux_pool;
  std::vector<int32_t> eps_count;        // [num_states] (empty == epsilon-free)

  // Per-state layout: finals at one end (finals_at_end); within the remaining range,
  // eps arcs sit at its start (eps_first) or end; emitting arcs fill the rest.
  int32_t RestBegin(int32_t s) const {
    return row_splits[s] + (finals_at_end ? 0 : final_count[s]);
  }
  int32_t RestEnd(int32_t s) const {
    return row_splits[s + 1] - (finals_at_end ? final_count[s] : 0);
  }
  int32_t EpsCountOf(int32_t s) const { return has_eps ? eps_count[s] : 0; }
  int32_t EpsBegin(int32_t s) const {
    return eps_first ? RestBegin(s) : RestEnd(s) - EpsCountOf(s);
  }
  int32_t EpsEnd(int32_t s) const {
    return eps_first ? RestBegin(s) + EpsCountOf(s) : RestEnd(s);
  }
  int32_t EmitBegin(int32_t s) const {
    return RestBegin(s) + (eps_first ? EpsCountOf(s) : 0);
  }
  int32_t EmitEnd(int32_t s) const {
    return RestEnd(s) - (eps_first ? 0 : EpsCountOf(s));
  }
  int32_t FinalBegin(int32_t s) const {
    return finals_at_end ? row_splits[s + 1] - final_count[s] : row_splits[s];
  }
  int32_t FinalEnd(int32_t s) const {
    return finals_at_end ? row_splits[s + 1] : row_splits[s] + final_count[s];
  }
  int32_t Dest(int32_t arc) const { return arc_dest_ilabel[2 * arc]; }
  int32_t Ilabel(int32_t arc) const { return arc_dest_ilabel[2 * arc + 1]; }
};

// Loads and validates an hlg.img file. Throws std::runtime_error on malformed input.
std::unique_ptr<GraphImage> LoadGraphImage(const std::string& path);

}  // namespace oasr::wfst
