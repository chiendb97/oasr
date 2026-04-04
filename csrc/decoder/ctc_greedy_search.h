// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef DECODER_CTC_GREEDY_SEARCH_H_
#define DECODER_CTC_GREEDY_SEARCH_H_

#include <memory>
#include <vector>

#include "decoder/common/utils.h"
#include "decoder/context_graph.h"
#include "decoder/search_interface.h"

namespace oasr {
namespace decoder {

struct CtcGreedySearchOptions {
    int blank = 0;  // blank token id
};

// CTC greedy search (best-path decoding).
//
// At each timestep takes the most likely token (argmax), then collapses
// consecutive duplicates and removes blanks — the standard CTC decoding rule.
// Supports both offline (one call to Search) and chunk-based streaming
// (multiple Search calls; prev_token_ preserves CTC collapse state across
// chunks). Returns a single hypothesis in the N-best interface.
class CtcGreedySearch : public SearchInterface {
public:
    explicit CtcGreedySearch(const CtcGreedySearchOptions& opts);

    void Search(const std::vector<std::vector<float>>& logp) override;
    void Reset() override;
    // No-op for greedy; results are already final after each Search call.
    void FinalizeSearch() override;

    SearchType Type() const override { return SearchType::kGreedySearch; }

    // Set an optional ContextGraph for phrase boosting. Pass nullptr to disable.
    void SetContextGraph(std::shared_ptr<ContextGraph> context_graph) {
        context_graph_ = std::move(context_graph);
    }

    // Returns a size-1 N-best list to match SearchInterface.
    const std::vector<std::vector<int>>& Inputs() const override { return hypotheses_; }
    const std::vector<std::vector<int>>& Outputs() const override { return hypotheses_; }
    const std::vector<float>& Likelihood() const override { return likelihood_vec_; }
    const std::vector<std::vector<int>>& Times() const override { return times_; }

private:
    CtcGreedySearchOptions opts_;
    std::shared_ptr<ContextGraph> context_graph_;

    // Context state for the single hypothesis (reset to 0 on Reset())
    int context_state_ = 0;
    float context_score_ = 0.0f;

    int abs_time_step_ = 0;
    int prev_token_ = -1;   // last emitted (non-blank) token, for CTC collapse across chunks

    std::vector<int> hypothesis_;   // accumulated decoded token ids
    std::vector<int> token_times_;  // per-token timestamp (argmax frame index)
    float likelihood_ = 0.0f;       // sum of argmax log-probs across all frames

    // N-best wrappers (size-1) — updated after every Search call.
    std::vector<std::vector<int>> hypotheses_;
    std::vector<float> likelihood_vec_;
    std::vector<std::vector<int>> times_;

    void PackResults();

public:
    OASR_DISALLOW_COPY_AND_ASSIGN(CtcGreedySearch);
};

}  // namespace decoder
}  // namespace oasr

#endif  // DECODER_CTC_GREEDY_SEARCH_H_
