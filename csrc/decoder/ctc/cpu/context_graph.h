// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef DECODER_CONTEXT_GRAPH_H_
#define DECODER_CONTEXT_GRAPH_H_

#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

namespace oasr {
namespace decoder {

// Aho-Corasick trie for phrase/context biasing in CTC decoding.
//
// Build the graph once from a list of token-ID phrases. Then during decoding,
// each hypothesis tracks a (state, accumulated_context_score) pair. At every
// token emission call GetNextState() to advance the state and receive an
// incremental score bonus. At finalization call GetBackoffScore() to subtract
// any uncommitted partial-match bonus.
//
// Thread-safety: immutable after construction; safe for concurrent read access.
class ContextGraph {
public:
    // Construct from a list of phrases (each phrase is a sequence of token IDs)
    // and a per-token bonus score applied when traversing toward a phrase end.
    // A completed phrase awards `context_score * phrase_length` total bonus.
    explicit ContextGraph(const std::vector<std::vector<int>>& phrases,
                          float context_score = 3.0f);

    // Given the current state and a new token, return {new_state, score_delta}.
    // score_delta is positive when advancing along a phrase, zero or negative
    // when failing back to a shorter match.
    std::pair<int, float> GetNextState(int state, int token) const;

    // Returns the accumulated partial-match score that should be subtracted
    // at finalization to penalize an uncommitted phrase prefix.
    float GetBackoffScore(int state) const;

    int start_state() const { return 0; }

private:
    struct TrieNode {
        std::unordered_map<int, int> children;  // token -> child node index
        int fail_state = 0;       // Aho-Corasick failure link (BFS-computed)
        float score = 0.0f;       // accumulated path score at this node
        bool is_end = false;      // true when a phrase ends here
    };

    std::vector<TrieNode> nodes_;
    float context_score_;

    void BuildFailureLinks();
};

}  // namespace decoder
}  // namespace oasr

#endif  // DECODER_CONTEXT_GRAPH_H_
