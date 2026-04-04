// Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "decoder/ctc_prefix_beam_search.h"

#include <algorithm>
#include <cassert>
#include <unordered_map>
#include <utility>

#include "decoder/common/utils.h"

namespace oasr {
namespace decoder {

CtcPrefixBeamSearch::CtcPrefixBeamSearch(const CtcPrefixBeamSearchOptions& opts) : opts_(opts) {
    Reset();
}

void CtcPrefixBeamSearch::Reset() {
    hypotheses_.clear();
    likelihood_.clear();
    cur_hyps_.clear();
    viterbi_likelihood_.clear();
    times_.clear();
    outputs_.clear();

    abs_time_step_ = 0;
    PrefixScore prefix_score;
    prefix_score.s = 0.0;
    prefix_score.ns = -kFloatMax;
    prefix_score.v_s = 0.0;
    prefix_score.v_ns = 0.0;

    std::vector<int> empty;
    cur_hyps_[empty] = prefix_score;
    outputs_.emplace_back(empty);
    hypotheses_.emplace_back(empty);
    likelihood_.emplace_back(prefix_score.score());
    times_.emplace_back(empty);
}

// Beam pruning uses total_score() (= acoustic score + context bonus).
static bool PrefixScoreCompare(const std::pair<std::vector<int>, PrefixScore>& a,
                               const std::pair<std::vector<int>, PrefixScore>& b) {
    return a.second.total_score() > b.second.total_score();
}

void CtcPrefixBeamSearch::UpdateHypotheses(
    const std::vector<std::pair<std::vector<int>, PrefixScore>>& hpys) {
    cur_hyps_.clear();
    outputs_.clear();
    hypotheses_.clear();
    likelihood_.clear();
    viterbi_likelihood_.clear();
    times_.clear();
    for (auto& item : hpys) {
        cur_hyps_[item.first] = item.second;
        hypotheses_.emplace_back(item.first);
        outputs_.emplace_back(std::move(item.first));
        likelihood_.emplace_back(item.second.total_score());
        viterbi_likelihood_.emplace_back(item.second.viterbi_score());
        times_.emplace_back(item.second.times());
    }
}

// Please refer https://robin1001.github.io/2020/12/11/ctc-search
// for how CTC prefix beam search works, and there is a simple graph demo in
// it.
void CtcPrefixBeamSearch::Search(const std::vector<std::vector<float>>& logp) {
    if (logp.size() == 0)
        return;
    int first_beam_size = std::min(static_cast<int>(logp[0].size()), opts_.first_beam_size);
    for (int t = 0; t < static_cast<int>(logp.size()); ++t, ++abs_time_step_) {
        const std::vector<float>& logp_t = logp[t];
        std::unordered_map<std::vector<int>, PrefixScore, PrefixHash> next_hyps;
        // 1. First beam prune, only select topk candidates
        std::vector<float> topk_score;
        std::vector<int32_t> topk_index;
        TopK(logp_t, first_beam_size, &topk_score, &topk_index);

        // 2. Token passing
        for (int i = 0; i < static_cast<int>(topk_index.size()); ++i) {
            int id = topk_index[i];
            auto prob = topk_score[i];
            for (const auto& it : cur_hyps_) {
                const std::vector<int>& prefix = it.first;
                const PrefixScore& prefix_score = it.second;
                // If prefix doesn't exist in next_hyps, next_hyps[prefix] will insert
                // PrefixScore(-inf, -inf) by default, since the default constructor
                // of PrefixScore will set fields s(blank ending score) and
                // ns(none blank ending score) to -inf, respectively.
                if (id == opts_.blank) {
                    // Case 0: *a + ε => *a
                    PrefixScore& next_score = next_hyps[prefix];
                    next_score.s = LogAdd(next_score.s, prefix_score.score() + prob);
                    next_score.v_s = prefix_score.viterbi_score() + prob;
                    next_score.times_s = prefix_score.times();
                    // Context state is unchanged when a blank is emitted
                    next_score.context_state = prefix_score.context_state;
                    next_score.context_score = prefix_score.context_score;
                } else if (!prefix.empty() && id == prefix.back()) {
                    // Case 1: *a + a => *a
                    PrefixScore& next_score1 = next_hyps[prefix];
                    next_score1.ns = LogAdd(next_score1.ns, prefix_score.ns + prob);
                    if (next_score1.v_ns < prefix_score.v_ns + prob) {
                        next_score1.v_ns = prefix_score.v_ns + prob;
                        if (next_score1.cur_token_prob < prob) {
                            next_score1.cur_token_prob = prob;
                            next_score1.times_ns = prefix_score.times_ns;
                            assert(next_score1.times_ns.size() > 0);
                            next_score1.times_ns.back() = abs_time_step_;
                        }
                    }
                    // Context state stays the same (no new token emitted)
                    next_score1.context_state = prefix_score.context_state;
                    next_score1.context_score = prefix_score.context_score;

                    // Case 2: *aε + a => *aa
                    std::vector<int> new_prefix(prefix);
                    new_prefix.emplace_back(id);
                    PrefixScore& next_score2 = next_hyps[new_prefix];
                    next_score2.ns = LogAdd(next_score2.ns, prefix_score.s + prob);
                    if (next_score2.v_ns < prefix_score.v_s + prob) {
                        next_score2.v_ns = prefix_score.v_s + prob;
                        next_score2.cur_token_prob = prob;
                        next_score2.times_ns = prefix_score.times_s;
                        next_score2.times_ns.emplace_back(abs_time_step_);
                    }
                    // New token emitted: advance context state
                    if (context_graph_) {
                        auto [new_ctx_state, delta] =
                            context_graph_->GetNextState(prefix_score.context_state, id);
                        next_score2.context_state = new_ctx_state;
                        next_score2.context_score = prefix_score.context_score + delta;
                    }
                } else {
                    // Case 3: *a + b => *ab, *aε + b => *ab
                    std::vector<int> new_prefix(prefix);
                    new_prefix.emplace_back(id);
                    PrefixScore& next_score = next_hyps[new_prefix];
                    next_score.ns = LogAdd(next_score.ns, prefix_score.score() + prob);
                    if (next_score.v_ns < prefix_score.viterbi_score() + prob) {
                        next_score.v_ns = prefix_score.viterbi_score() + prob;
                        next_score.cur_token_prob = prob;
                        next_score.times_ns = prefix_score.times();
                        next_score.times_ns.emplace_back(abs_time_step_);
                    }
                    // New token emitted: advance context state
                    if (context_graph_) {
                        auto [new_ctx_state, delta] =
                            context_graph_->GetNextState(prefix_score.context_state, id);
                        next_score.context_state = new_ctx_state;
                        next_score.context_score = prefix_score.context_score + delta;
                    }
                }
            }
        }

        // 3. Second beam prune, only keep top n best paths
        std::vector<std::pair<std::vector<int>, PrefixScore>> arr(next_hyps.begin(),
                                                                  next_hyps.end());
        int second_beam_size = std::min(static_cast<int>(arr.size()), opts_.second_beam_size);
        std::nth_element(arr.begin(), arr.begin() + second_beam_size, arr.end(),
                         PrefixScoreCompare);
        arr.resize(second_beam_size);
        std::sort(arr.begin(), arr.end(), PrefixScoreCompare);

        // 4. Update cur_hyps_ and get new result
        UpdateHypotheses(arr);
    }
}

void CtcPrefixBeamSearch::FinalizeSearch() {
    assert(hypotheses_.size() == cur_hyps_.size());
    assert(hypotheses_.size() == likelihood_.size());
    std::vector<std::pair<std::vector<int>, PrefixScore>> arr(cur_hyps_.begin(), cur_hyps_.end());

    // Subtract any partial-match context bonus at finalization to avoid
    // rewarding an incomplete phrase match.
    if (context_graph_) {
        for (auto& item : arr) {
            item.second.context_score -= context_graph_->GetBackoffScore(item.second.context_state);
        }
    }

    std::sort(arr.begin(), arr.end(), PrefixScoreCompare);
    UpdateHypotheses(arr);
}

}  // namespace decoder
}  // namespace oasr
