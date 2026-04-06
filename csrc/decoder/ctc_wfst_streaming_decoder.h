// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef DECODER_CTC_WFST_STREAMING_DECODER_H_
#define DECODER_CTC_WFST_STREAMING_DECODER_H_

// Streaming WFST CTC decoder using k2::OnlineDenseIntersecter.
//
// Requires both OASR_USE_K2 and OASR_K2_STREAMING to be defined.
// OASR_K2_STREAMING is set by CMake when K2_SOURCE_DIR points to the k2
// source tree (internal headers are not shipped in the pip package).

#if defined(OASR_USE_K2)

#include <memory>
#include <string>
#include <vector>

// Public k2 API (shipped with pip package)
#include <k2/torch_api.h>

// Internal k2 headers (available when K2_SOURCE_DIR is set)
#include <k2/csrc/intersect_dense_pruned.h>  // OnlineDenseIntersecter, DecodeStateInfo
#include <k2/torch/csrc/fsa_class.h>          // full k2::FsaClass definition

#include "decoder/common/utils.h"
#include "decoder/ctc_wfst_beam_search.h"  // CtcWfstBeamSearchOptions

namespace oasr {
namespace decoder {

// Incremental (streaming) WFST CTC decoder for a single utterance.
//
// Follows the online_decode.cu pattern from k2:
//   https://github.com/k2-fsa/k2/blob/master/k2/torch/bin/online_decode.cu
//
// The key difference from CtcWfstBeamSearch (offline):
//   - Offline: accumulates ALL frames then decodes once in FinalizeSearch().
//   - Streaming: each Search() call feeds a chunk and decodes it incrementally
//     via k2::OnlineDenseIntersecter, which carries FSA states between chunks.
//
// Usage:
//   auto dec = CtcWfstStreamingDecoder::FromFile(opts, fst_path, device);
//   while (chunk available) {
//       dec->Search(chunk_logp);        // decode chunk, update partial result
//       auto partial = dec->Outputs();
//   }
//   dec->FinalizeSearch();              // (no-op; results updated per-chunk)
//   auto final_result = dec->Outputs();
//   dec->Reset();                       // ready for next utterance
//
// Build requirements:
//   cmake -DOASR_USE_K2=ON -DK2_SOURCE_DIR=<k2 source root> ...
class CtcWfstStreamingDecoder {
public:
    CtcWfstStreamingDecoder(const CtcWfstBeamSearchOptions& opts,
                             k2::FsaClassPtr decoding_graph,
                             torch::Device device = torch::kCPU);

    // Convenience: load a decoding graph saved with
    //   torch.save(fsa.as_dict(), path, _use_new_zipfile_serialization=True)
    static std::unique_ptr<CtcWfstStreamingDecoder> FromFile(
        const CtcWfstBeamSearchOptions& opts,
        const std::string& fst_path,
        torch::Device device = torch::kCPU);

    // Accept one chunk of encoder log-probs and decode incrementally.
    // logp[t][v] = log P(v | frame t), already post-subsampling.
    // Blank-skipping (blank_skip_thresh) is applied before decoding.
    // Outputs() / Inputs() are updated after each call.
    void Search(const std::vector<std::vector<float>>& logp);

    // Reset all state; call before decoding a new utterance.
    void Reset();

    // No-op: results are already updated after each Search() call.
    void FinalizeSearch() {}

    const std::vector<std::vector<int>>& Inputs()     const { return hypotheses_; }
    const std::vector<std::vector<int>>& Outputs()    const { return outputs_; }
    const std::vector<float>&            Likelihood() const { return likelihood_; }
    const std::vector<std::vector<int>>& Times()      const { return times_; }

private:
    CtcWfstBeamSearchOptions opts_;
    k2::FsaClassPtr decoding_graph_ptr_;
    torch::Device device_;

    // 3-axis FsaVec (fsa, state, arc) required by OnlineDenseIntersecter.
    // Derived from decoding_graph_ptr_->fsa in BuildIntersecter().
    k2::FsaVec decoding_fsa_vec_;

    // Stateful online decoder.  The intersecter itself is reused across
    // utterances; per-utterance FSA state lives in stream_state_.
    std::unique_ptr<k2::OnlineDenseIntersecter> intersecter_;

    // Accumulated FSA states for the current utterance (reset to {} in Reset()).
    // A default-constructed DecodeStateInfo signals "new stream" to Decode().
    k2::DecodeStateInfo stream_state_;

    // Latest best-path results (updated after every Search() call).
    std::vector<std::vector<int>> hypotheses_;
    std::vector<std::vector<int>> outputs_;
    std::vector<float>             likelihood_;
    std::vector<std::vector<int>> times_;

    void BuildIntersecter();
    void DecodeChunk(const std::vector<torch::Tensor>& frames);

public:
    OASR_DISALLOW_COPY_AND_ASSIGN(CtcWfstStreamingDecoder);
};

}  // namespace decoder
}  // namespace oasr

#endif  // OASR_USE_K2
#endif  // DECODER_CTC_WFST_STREAMING_DECODER_H_
