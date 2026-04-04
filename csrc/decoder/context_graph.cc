// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#include "decoder/context_graph.h"

#include <queue>

namespace oasr {
namespace decoder {

ContextGraph::ContextGraph(const std::vector<std::vector<int>>& phrases,
                           float context_score)
    : context_score_(context_score) {
    // Root node (state 0)
    nodes_.emplace_back();

    // Insert all phrases into the trie
    for (const auto& phrase : phrases) {
        if (phrase.empty()) continue;
        int cur = 0;
        for (int token : phrase) {
            auto it = nodes_[cur].children.find(token);
            if (it == nodes_[cur].children.end()) {
                int new_state = static_cast<int>(nodes_.size());
                nodes_[cur].children[token] = new_state;
                nodes_.emplace_back();
                // Inherit accumulated score from parent
                nodes_[new_state].score = nodes_[cur].score + context_score_;
                cur = new_state;
            } else {
                cur = it->second;
            }
        }
        nodes_[cur].is_end = true;
    }

    BuildFailureLinks();
}

void ContextGraph::BuildFailureLinks() {
    // BFS from root to compute Aho-Corasick failure links.
    // Root's children have fail_state = 0 (root).
    std::queue<int> q;
    for (auto& [token, child] : nodes_[0].children) {
        nodes_[child].fail_state = 0;
        q.push(child);
    }

    while (!q.empty()) {
        int state = q.front();
        q.pop();
        for (auto& [token, child] : nodes_[state].children) {
            // Follow failure links from parent until we find a state that has
            // a transition for this token, or we reach root.
            int fail = nodes_[state].fail_state;
            while (fail != 0 && nodes_[fail].children.count(token) == 0) {
                fail = nodes_[fail].fail_state;
            }
            auto it = nodes_[fail].children.find(token);
            if (it != nodes_[fail].children.end() && it->second != child) {
                nodes_[child].fail_state = it->second;
            } else {
                nodes_[child].fail_state = 0;
            }
            q.push(child);
        }
    }
}

std::pair<int, float> ContextGraph::GetNextState(int state, int token) const {
    // Follow transitions (with Aho-Corasick fall-back on failure) to find
    // the longest suffix of (current-path + token) that is a prefix of some phrase.
    int cur = state;
    while (cur != 0 && nodes_[cur].children.count(token) == 0) {
        cur = nodes_[cur].fail_state;
    }

    auto it = nodes_[cur].children.find(token);
    int new_state = (it != nodes_[cur].children.end()) ? it->second : 0;

    // If the previous state was a completed phrase (is_end=true), the phrase
    // bonus was already fully granted — use 0 as the base so the next partial
    // match starts fresh instead of subtracting the completed phrase's score.
    float base_score = nodes_[state].is_end ? 0.0f : nodes_[state].score;
    float delta = nodes_[new_state].score - base_score;
    return {new_state, delta};
}

float ContextGraph::GetBackoffScore(int state) const {
    // A completed phrase (is_end=true) was fully credited; no backoff needed.
    // A partial match (is_end=false) should be penalized by its accumulated score.
    if (nodes_[state].is_end) return 0.0f;
    return nodes_[state].score;
}

}  // namespace decoder
}  // namespace oasr
