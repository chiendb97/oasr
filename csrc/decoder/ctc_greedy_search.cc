// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#include "decoder/ctc_greedy_search.h"

#include <algorithm>

namespace oasr {
namespace decoder {

CtcGreedySearch::CtcGreedySearch(const CtcGreedySearchOptions& opts) : opts_(opts) {
    Reset();
}

void CtcGreedySearch::Reset() {
    abs_time_step_ = 0;
    prev_token_ = -1;
    hypothesis_.clear();
    token_times_.clear();
    likelihood_ = 0.0f;
    context_state_ = 0;
    context_score_ = 0.0f;
    PackResults();
}

void CtcGreedySearch::Search(const std::vector<std::vector<float>>& logp) {
    if (logp.empty()) return;

    for (int t = 0; t < static_cast<int>(logp.size()); ++t, ++abs_time_step_) {
        const std::vector<float>& logp_t = logp[t];

        // Argmax over vocabulary
        int best_id = static_cast<int>(
            std::max_element(logp_t.begin(), logp_t.end()) - logp_t.begin());
        float best_prob = logp_t[best_id];

        // Accumulate log-likelihood of the best path
        likelihood_ += best_prob;

        // CTC collapse: skip blank and consecutive repeated tokens
        if (best_id != opts_.blank && best_id != prev_token_) {
            hypothesis_.push_back(best_id);
            token_times_.push_back(abs_time_step_);

            // Advance context state for the new token
            if (context_graph_) {
                auto [new_state, delta] = context_graph_->GetNextState(context_state_, best_id);
                context_state_ = new_state;
                context_score_ += delta;
            }
        }

        prev_token_ = best_id;
    }

    PackResults();
}

void CtcGreedySearch::FinalizeSearch() {
    // Results are already up-to-date; nothing extra to do for greedy search.
}

void CtcGreedySearch::PackResults() {
    // Reported likelihood includes the context bonus
    float reported_likelihood = likelihood_ + context_score_;
    hypotheses_ = {hypothesis_};
    likelihood_vec_ = {reported_likelihood};
    times_ = {token_times_};
}

}  // namespace decoder
}  // namespace oasr
