// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#include "decoder/ctc_wfst_beam_search.h"

#ifdef OASR_USE_K2

#include <algorithm>

namespace oasr {
namespace decoder {

CtcWfstBeamSearch::CtcWfstBeamSearch(const CtcWfstBeamSearchOptions& opts,
                                     k2::FsaClassPtr decoding_graph)
    : opts_(opts), decoding_graph_(std::move(decoding_graph)) {
    Reset();
}

/* static */
std::unique_ptr<CtcWfstBeamSearch> CtcWfstBeamSearch::FromFile(
    const CtcWfstBeamSearchOptions& opts, const std::string& fst_path,
    torch::Device device) {
    k2::FsaClassPtr graph = k2::LoadFsaClass(fst_path, device);
    return std::make_unique<CtcWfstBeamSearch>(opts, std::move(graph));
}

void CtcWfstBeamSearch::Reset() {
    frame_logprobs_.clear();
    hypotheses_.clear();
    outputs_.clear();
    likelihood_.clear();
    times_.clear();
}

void CtcWfstBeamSearch::Search(const std::vector<std::vector<float>>& logp) {
    if (logp.empty()) return;

    const int V = static_cast<int>(logp[0].size());
    for (const auto& logp_t : logp) {
        // Blank-skipping: skip frames where P(blank) > threshold.
        float blank_log_prob = logp_t[opts_.blank];
        if (std::exp(blank_log_prob) > opts_.blank_skip_thresh) continue;

        // Clone to own the data (from_blob does not copy).
        torch::Tensor frame =
            torch::from_blob(const_cast<float*>(logp_t.data()), {V},
                             torch::TensorOptions().dtype(torch::kFloat32))
                .clone();
        frame_logprobs_.push_back(frame);
    }
}

void CtcWfstBeamSearch::FinalizeSearch() {
    if (frame_logprobs_.empty()) return;
    DecodeAccumulated();
}

void CtcWfstBeamSearch::DecodeAccumulated() {
    hypotheses_.clear();
    outputs_.clear();
    likelihood_.clear();
    times_.clear();

    // Stack all frames into [T, V] then add batch dim → [1, T, V].
    torch::Tensor logp = torch::stack(frame_logprobs_, /*dim=*/0);  // [T, V]
    const int T = static_cast<int>(logp.size(0));
    logp = logp.unsqueeze(0);  // [1, T, V]

    // Sequence length for the single utterance.
    torch::Tensor lens =
        torch::tensor({T}, torch::TensorOptions().dtype(torch::kInt32));

    // k2 CTC lattice decoding.
    k2::FsaClassPtr lattice =
        k2::GetLattice(logp, lens, decoding_graph_, opts_.search_beam, opts_.output_beam,
                       opts_.min_active_states, opts_.max_active_states,
                       opts_.subsampling_factor);

    // Extract best path token IDs (one per batch element; batch size = 1).
    std::vector<std::vector<int32_t>> paths = k2::BestPath(lattice);

    for (const auto& path : paths) {
        std::vector<int> tokens(path.begin(), path.end());
        hypotheses_.push_back(tokens);
        outputs_.push_back(tokens);
        likelihood_.push_back(0.0f);  // Per-path score not exposed by public k2 API.
        times_.push_back({});
    }
}

}  // namespace decoder
}  // namespace oasr

#endif  // OASR_USE_K2
