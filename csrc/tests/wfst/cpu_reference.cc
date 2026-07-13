#include "tests/wfst/cpu_reference.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace oasr::wfst {
namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

struct Token {
  int32_t state;
  float score;
  int32_t prev_tok;  // index into the previous frame's tokens — or the SAME frame's for
                     // an epsilon hop; -1 at start
  int32_t arc;       // incoming graph arc id; -1 at start
  bool eps_hop = false;
};

struct Candidate {
  int32_t dest;
  float end;
  int32_t prev_tok;
  int32_t arc;
};

// Verbatim k2 dynamic-beam update, intersect_dense_pruned.cu
// lambda_set_beam_and_cutoffs. `t` counts real frames; the final step is t == num_frames
// (k2's final_t - 1 with final_t = num_frames + 1). Online (streaming) semantics: the
// device carries T = INT32_MAX mid-stream, making every final_t clause inert until the
// finalize step — mirrored here via an effectively-infinite final_t for t < num_frames.
float UpdateBeam(float dynamic_beam, const DecoderConfig& cfg, int32_t active_states,
                 int64_t t, int32_t num_frames, bool online) {
  const float default_beam = cfg.search_beam;
  const int64_t final_t =
      (online && t < num_frames) ? (static_cast<int64_t>(1) << 40) : num_frames + 1;
  float current_min_active = static_cast<float>(cfg.min_active_states);
  if (t + 5 >= final_t) {
    current_min_active = std::max(cfg.min_active_states, cfg.max_active_states / 2);
  }
  if (active_states <= cfg.max_active_states) {
    if (active_states >= current_min_active || active_states == 0) {
      dynamic_beam = 0.8f * dynamic_beam + 0.2f * default_beam;
    } else {
      if (dynamic_beam < default_beam) dynamic_beam = default_beam;
      dynamic_beam *= 1.25f;
    }
  } else if (t + 5 < final_t) {
    if (dynamic_beam > default_beam) dynamic_beam = default_beam;
    dynamic_beam *= 0.8f;
  }
  if (t == final_t - 1) dynamic_beam = 1.0e10f;
  return dynamic_beam;
}

// Recombine in-beam candidates by dest state (max score). Candidate order breaks exact
// ties (first max wins) — matches the documented tie policy.
std::vector<Token> Recombine(const std::vector<Candidate>& cands, float cutoff) {
  std::vector<Token> next;
  std::unordered_map<int32_t, int32_t> state_to_tok;
  for (const Candidate& c : cands) {
    if (!(c.end > cutoff)) continue;  // strict >, k2 lambda_set_state_map
    auto [it, inserted] = state_to_tok.try_emplace(c.dest, static_cast<int32_t>(next.size()));
    if (inserted) {
      next.push_back({c.dest, c.end, c.prev_tok, c.arc});
    } else if (c.end > next[it->second].score) {
      next[it->second] = {c.dest, c.end, c.prev_tok, c.arc};
    }
  }
  return next;
}

// Epsilon closure (TLG graphs): fixed passes; each pass expands the frontier snapshot's
// eps arcs, prunes against the pass's exact max with the frame's dynamic beam, and
// merges (append new states / update improved ones). Mirrors the device exactly.
void EpsClosure(const GraphImage& g, std::vector<Token>& toks, float beam, int32_t iters) {
  if (!g.has_eps || toks.empty()) return;
  std::unordered_map<int32_t, int32_t> pos;
  for (int32_t i = 0; i < static_cast<int32_t>(toks.size()); ++i) pos[toks[i].state] = i;
  for (int32_t it = 0; it < iters; ++it) {
    std::vector<Candidate> cands;
    const int32_t n = static_cast<int32_t>(toks.size());
    for (int32_t i = 0; i < n; ++i) {
      const Token& tok = toks[i];
      for (int32_t a = g.EpsBegin(tok.state); a < g.EpsEnd(tok.state); ++a) {
        cands.push_back({g.Dest(a), tok.score + g.arc_weight[a], i, a});
      }
    }
    if (cands.empty()) return;
    float best = kNegInf;
    for (const Candidate& c : cands) best = std::max(best, c.end);
    const float cutoff = best - beam;
    bool changed = false;
    for (const Candidate& c : cands) {
      if (!(c.end > cutoff)) continue;
      auto [it2, inserted] = pos.try_emplace(c.dest, static_cast<int32_t>(toks.size()));
      if (inserted) {
        toks.push_back({c.dest, c.end, c.prev_tok, c.arc, true});
        changed = true;
      } else if (c.end > toks[it2->second].score) {
        toks[it2->second] = {c.dest, c.end, c.prev_tok, c.arc, true};
        changed = true;
      }
    }
    if (!changed) return;
  }
}

}  // namespace

CpuDecodeResult CpuDecode(const GraphImage& g, const float* log_probs, int32_t num_frames,
                          int32_t vocab_stride, const DecoderConfig& cfg, bool online) {
  CpuDecodeResult res;
  std::vector<std::vector<Token>> toks(num_frames + 2);
  toks[0].push_back({g.start_state, 0.0f, -1, -1});
  float dynamic_beam = cfg.search_beam;
  EpsClosure(g, toks[0], dynamic_beam, cfg.eps_iterations);  // initial closure

  for (int32_t t = 0; t <= num_frames; ++t) {
    const std::vector<Token>& cur = toks[t];
    const bool final_step = (t == num_frames);
    std::vector<Candidate> cands;

    if (!final_step) {
      const float* lp = log_probs + static_cast<int64_t>(t) * vocab_stride;
      for (int32_t i = 0; i < static_cast<int32_t>(cur.size()); ++i) {
        const Token& tok = cur[i];
        for (int32_t a = g.EmitBegin(tok.state); a < g.EmitEnd(tok.state); ++a) {
          const float end = tok.score + g.arc_weight[a] + lp[g.Ilabel(a)];
          cands.push_back({g.Dest(a), end, i, a});
        }
      }
    } else {
      // k2 final step: acoustic 0 for -1 arcs; real-label arcs get -inf (die), except
      // under allow_partial with no reachable final arc, where ALL arcs are redirected
      // to the super-final state with acoustic 0.
      bool has_valid_final = false;
      for (const Token& tok : cur) {
        if (g.FinalEnd(tok.state) > g.FinalBegin(tok.state)) {
          has_valid_final = true;
          break;
        }
      }
      const int32_t super_final = static_cast<int32_t>(g.num_states) - 1;
      for (int32_t i = 0; i < static_cast<int32_t>(cur.size()); ++i) {
        const Token& tok = cur[i];
        if (has_valid_final || !cfg.allow_partial) {
          for (int32_t a = g.FinalBegin(tok.state); a < g.FinalEnd(tok.state); ++a) {
            cands.push_back({g.Dest(a), tok.score + g.arc_weight[a], i, a});
          }
        } else {
          for (int32_t a = g.row_splits[tok.state]; a < g.row_splits[tok.state + 1]; ++a) {
            cands.push_back({super_final, tok.score + g.arc_weight[a], i, a});
          }
        }
      }
      if (!has_valid_final && !cfg.allow_partial) cands.clear();
      res.reached_final = has_valid_final;
    }

    float best = kNegInf;
    for (const Candidate& c : cands) best = std::max(best, c.end);
    dynamic_beam =
        UpdateBeam(dynamic_beam, cfg, static_cast<int32_t>(cur.size()), t, num_frames,
                   online);
    const float cutoff = best - dynamic_beam;
    toks[t + 1] = Recombine(cands, cutoff);
    if (!final_step) EpsClosure(g, toks[t + 1], dynamic_beam, cfg.eps_iterations);
  }

  // Backtrack from the best final token.
  const std::vector<Token>& finals = toks[num_frames + 1];
  if (finals.empty()) return res;  // ok == false
  int32_t best_i = 0;
  for (int32_t i = 1; i < static_cast<int32_t>(finals.size()); ++i) {
    if (finals[i].score > finals[best_i].score) best_i = i;
  }
  res.ok = true;
  res.score = finals[best_i].score;

  int32_t frame = num_frames + 1, tok_i = best_i;
  while (tok_i >= 0 && frame >= 0) {
    const Token& tok = toks[frame][tok_i];
    if (tok.arc >= 0) res.arc_path.push_back(tok.arc);
    tok_i = tok.prev_tok;
    if (!tok.eps_hop) --frame;  // epsilon hops chain within the same frame
  }
  std::reverse(res.arc_path.begin(), res.arc_path.end());
  for (int32_t a : res.arc_path) {
    for (int32_t j = g.aux_row_splits[a]; j < g.aux_row_splits[a + 1]; ++j) {
      if (g.aux_pool[j] > 0) res.words.push_back(g.aux_pool[j]);
    }
  }

  res.frames.resize(toks.size());
  for (size_t f = 0; f < toks.size(); ++f) {
    for (const Token& tok : toks[f]) res.frames[f].push_back({tok.state, tok.score});
    std::sort(res.frames[f].begin(), res.frames[f].end());
  }
  return res;
}

}  // namespace oasr::wfst
