// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef DECODER_CTC_WFST_BEAM_SEARCH_H_
#define DECODER_CTC_WFST_BEAM_SEARCH_H_

#ifdef OASR_USE_K2

#include <memory>
#include <string>
#include <vector>

// k2's public C API (only installed header in the pip package)
#include "k2/torch_api.h"

#include "decoder/common/utils.h"
#include "decoder/search_interface.h"

namespace oasr {
namespace decoder {

struct CtcWfstBeamSearchOptions {
    int blank = 0;
    float search_beam = 20.0f;        // decoding beam (larger = more exact, slower)
    float output_beam = 8.0f;         // lattice pruning beam (similar to lattice-beam in Kaldi)
    int min_active_states = 30;       // min active FSA states per frame
    int max_active_states = 10000;    // max active FSA states per frame
    int subsampling_factor = 1;       // encoder subsampling factor (used for timestamps)
    int nbest = 10;                   // number of N-best paths to extract
    // Skip frames where P(blank) > blank_skip_thresh to speed up decoding.
    float blank_skip_thresh = 0.98f;
};

// WFST-based CTC decoding using K2's public torch_api.h interface.
//
// Accepts a decoding graph as a k2::FsaClassPtr (shared pointer to FsaClass).
// Use the static FromFile() factory to load a graph saved with torch.save().
//
// Streaming mode: Search() accumulates log-prob frames; FinalizeSearch() runs
// the actual k2 intersection and N-best extraction over all accumulated frames.
// For truly incremental streaming call FinalizeSearch() after each chunk — the
// state is reset between calls (re-initialize with Reset() to start a new utt).
//
// Build requirements:
//   cmake -DOASR_USE_K2=ON -Dk2_DIR=<k2Config.cmake dir> ...
//   or:  OASR_USE_K2=1 pip install -e .
class CtcWfstBeamSearch : public SearchInterface {
public:
    CtcWfstBeamSearch(const CtcWfstBeamSearchOptions& opts,
                      k2::FsaClassPtr decoding_graph);

    // Convenience: load a decoding graph saved with
    //   torch.save(fsa.as_dict(), path, _use_new_zipfile_serialization=True)
    static std::unique_ptr<CtcWfstBeamSearch> FromFile(
        const CtcWfstBeamSearchOptions& opts,
        const std::string& fst_path,
        torch::Device device = torch::kCPU);

    void Search(const std::vector<std::vector<float>>& logp) override;
    void Reset() override;
    void FinalizeSearch() override;

    SearchType Type() const override { return SearchType::kWfstBeamSearch; }

    const std::vector<std::vector<int>>& Inputs() const override { return hypotheses_; }
    const std::vector<std::vector<int>>& Outputs() const override { return outputs_; }
    const std::vector<float>& Likelihood() const override { return likelihood_; }
    const std::vector<std::vector<int>>& Times() const override { return times_; }

private:
    CtcWfstBeamSearchOptions opts_;
    k2::FsaClassPtr decoding_graph_;

    // Accumulated per-frame log-probs (appended by each Search() call).
    // Shape after stacking: [T_total, V]
    std::vector<torch::Tensor> frame_logprobs_;

    // N-best results (populated by FinalizeSearch).
    std::vector<std::vector<int>> hypotheses_;
    std::vector<std::vector<int>> outputs_;
    std::vector<float> likelihood_;
    std::vector<std::vector<int>> times_;

    void DecodeAccumulated();

public:
    OASR_DISALLOW_COPY_AND_ASSIGN(CtcWfstBeamSearch);
};

}  // namespace decoder
}  // namespace oasr

#endif  // OASR_USE_K2
#endif  // DECODER_CTC_WFST_BEAM_SEARCH_H_
