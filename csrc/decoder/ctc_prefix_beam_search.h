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

#ifndef DECODER_CTC_PREFIX_BEAM_SEARCH_H_
#define DECODER_CTC_PREFIX_BEAM_SEARCH_H_

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "decoder/common/utils.h"
#include "decoder/context_graph.h"
#include "decoder/search_interface.h"

namespace oasr {
namespace decoder {

struct CtcPrefixBeamSearchOptions {
    int blank = 0;  // blank id
    int first_beam_size = 10;
    int second_beam_size = 10;
};

struct PrefixScore {
    float s = -kFloatMax;               // blank ending score
    float ns = -kFloatMax;              // none blank ending score
    float v_s = -kFloatMax;             // viterbi blank ending score
    float v_ns = -kFloatMax;            // viterbi none blank ending score
    float cur_token_prob = -kFloatMax;  // prob of current token
    std::vector<int> times_s;           // times of viterbi blank path
    std::vector<int> times_ns;          // times of viterbi none blank path

    // Context biasing state (zero-overhead when no ContextGraph is set)
    int context_state = 0;
    float context_score = 0.0f;

    float score() const { return LogAdd(s, ns); }
    float viterbi_score() const { return v_s > v_ns ? v_s : v_ns; }
    const std::vector<int>& times() const { return v_s > v_ns ? times_s : times_ns; }
    // Total score including phrase-boosting bonus; used for beam pruning.
    float total_score() const { return score() + context_score; }
};

struct PrefixHash {
    size_t operator()(const std::vector<int>& prefix) const {
        size_t hash_code = 0;
        // here we use KB&DR hash code
        for (int id : prefix) {
            hash_code = id + 31 * hash_code;
        }
        return hash_code;
    }
};

class CtcPrefixBeamSearch : public SearchInterface {
public:
    explicit CtcPrefixBeamSearch(const CtcPrefixBeamSearchOptions& opts);

    void Search(const std::vector<std::vector<float>>& logp) override;
    void Reset() override;
    void FinalizeSearch() override;
    SearchType Type() const override { return SearchType::kPrefixBeamSearch; }
    void UpdateHypotheses(const std::vector<std::pair<std::vector<int>, PrefixScore>>& hpys);

    // Set an optional ContextGraph for phrase boosting. Pass nullptr to disable.
    void SetContextGraph(std::shared_ptr<ContextGraph> context_graph) {
        context_graph_ = std::move(context_graph);
    }

    const std::vector<float>& viterbi_likelihood() const { return viterbi_likelihood_; }
    const std::vector<std::vector<int>>& Inputs() const override { return hypotheses_; }
    const std::vector<std::vector<int>>& Outputs() const override { return outputs_; }
    const std::vector<float>& Likelihood() const override { return likelihood_; }
    const std::vector<std::vector<int>>& Times() const override { return times_; }

private:
    int abs_time_step_ = 0;

    // N-best list and corresponding likelihood_, in sorted order
    std::vector<std::vector<int>> hypotheses_;
    std::vector<float> likelihood_;
    std::vector<float> viterbi_likelihood_;
    std::vector<std::vector<int>> times_;

    std::unordered_map<std::vector<int>, PrefixScore, PrefixHash> cur_hyps_;
    std::vector<std::vector<int>> outputs_;
    CtcPrefixBeamSearchOptions opts_;
    std::shared_ptr<ContextGraph> context_graph_;

public:
    OASR_DISALLOW_COPY_AND_ASSIGN(CtcPrefixBeamSearch);
};

}  // namespace decoder
}  // namespace oasr

#endif  // OASR_DECODER_CTC_PREFIX_BEAM_SEARCH_H_