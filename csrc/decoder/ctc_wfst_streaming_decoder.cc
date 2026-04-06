// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#include "decoder/ctc_wfst_streaming_decoder.h"

#if defined(OASR_USE_K2)

#include <algorithm>
#include <cmath>
#include <vector>

// Internal k2 headers (available when K2_SOURCE_DIR is set)
#include <k2/csrc/fsa.h>             // k2::FsaToFsaVec
#include <k2/torch/csrc/decode.h>     // k2::GetTexts
#include <k2/torch/csrc/dense_fsa_vec.h>  // k2::DenseFsaVec, k2::CreateDenseFsaVec
#include <k2/torch/csrc/fsa_algo.h>   // k2::ShortestPath
#include <k2/torch/csrc/utils.h>      // k2::Array1ToTorch

namespace oasr {
namespace decoder {

CtcWfstStreamingDecoder::CtcWfstStreamingDecoder(
    const CtcWfstBeamSearchOptions& opts,
    k2::FsaClassPtr decoding_graph,
    torch::Device device)
    : opts_(opts),
      decoding_graph_ptr_(std::move(decoding_graph)),
      device_(device) {
    BuildIntersecter();
}

/* static */
std::unique_ptr<CtcWfstStreamingDecoder> CtcWfstStreamingDecoder::FromFile(
    const CtcWfstBeamSearchOptions& opts,
    const std::string& fst_path,
    torch::Device device) {
    k2::FsaClassPtr graph = k2::LoadFsaClass(fst_path, device);
    return std::make_unique<CtcWfstStreamingDecoder>(opts, std::move(graph), device);
}

void CtcWfstStreamingDecoder::BuildIntersecter() {
    // OnlineDenseIntersecter requires a 3-axis FsaVec (fsa, state, arc).
    // The loaded graph may be 2-axis (single Fsa); wrap it if needed.
    k2::FsaOrVec& foa = decoding_graph_ptr_->fsa;
    if (foa.NumAxes() == 2) {
        decoding_fsa_vec_ = k2::FsaToFsaVec(foa);
    } else {
        decoding_fsa_vec_ = foa;
    }

    intersecter_ = std::make_unique<k2::OnlineDenseIntersecter>(
        decoding_fsa_vec_,
        /*num_seqs=*/1,
        opts_.search_beam,
        opts_.output_beam,
        opts_.min_active_states,
        opts_.max_active_states);
}

void CtcWfstStreamingDecoder::Reset() {
    // A default-constructed DecodeStateInfo means "new stream" to Decode().
    stream_state_ = k2::DecodeStateInfo{};
    hypotheses_.clear();
    outputs_.clear();
    likelihood_.clear();
    times_.clear();
}

void CtcWfstStreamingDecoder::Search(
    const std::vector<std::vector<float>>& logp) {
    if (logp.empty()) return;

    const int V = static_cast<int>(logp[0].size());
    std::vector<torch::Tensor> frames;
    frames.reserve(logp.size());

    for (const auto& logp_t : logp) {
        // Blank-skipping: skip frames where P(blank) > threshold.
        if (std::exp(logp_t[opts_.blank]) > opts_.blank_skip_thresh) continue;

        // Clone to own the data (from_blob does not copy).
        frames.push_back(
            torch::from_blob(const_cast<float*>(logp_t.data()), {V},
                             torch::TensorOptions().dtype(torch::kFloat32))
                .clone());
    }

    if (!frames.empty()) DecodeChunk(frames);
}

void CtcWfstStreamingDecoder::DecodeChunk(
    const std::vector<torch::Tensor>& frames) {
    const int T = static_cast<int>(frames.size());

    // Stack frames to [T, V] then add batch dimension → [1, T, V].
    torch::Tensor logp = torch::stack(frames, /*dim=*/0).unsqueeze(0);
    if (logp.device() != device_) logp = logp.to(device_);

    // supervision_segments: [sequence_idx=0, start_frame=0, duration=T].
    // Duration is in encoder-output (post-subsampled) frames, same scale as T.
    // Must be on CPU per CreateDenseFsaVec requirements.
    torch::Tensor seg = torch::tensor(
        {{0, 0, T}},
        torch::TensorOptions().dtype(torch::kInt32));

    // allow_truncate handles edge cases when T is not divisible by
    // subsampling_factor (same value used in online_decode.cu).
    const int allow_truncate = std::max(0, opts_.subsampling_factor - 1);
    k2::DenseFsaVec dense = k2::CreateDenseFsaVec(logp, seg, allow_truncate);

    // Run online intersection.  stream_state_ is updated in-place:
    // it holds the active FSA states at the end of this chunk, ready for
    // the next call.  An empty stream_state_ means this is a new stream.
    k2::FsaVec ofsa;
    k2::Array1<int32_t> arc_map;
    std::vector<k2::DecodeStateInfo*> states = {&stream_state_};
    intersecter_->Decode(dense, &states, &ofsa, &arc_map);

    // Propagate decoding-graph attributes (e.g. aux_labels / word IDs for HLG)
    // to the output lattice arcs via the arc_map.
    k2::FsaClass lattice(ofsa);
    lattice.CopyAttrs(*decoding_graph_ptr_,
                      k2::Array1ToTorch<int32_t>(arc_map));

    // Extract current best path via ShortestPath (1-best Viterbi).
    k2::FsaClass best = k2::ShortestPath(lattice);

    // GetTexts returns aux_labels (word IDs for HLG, token IDs for CTC topo).
    k2::Ragged<int32_t> texts = k2::GetTexts(best);
    auto texts_vec = texts.ToVecVec();

    // Update outputs with the latest best hypothesis.
    hypotheses_.clear();
    outputs_.clear();
    likelihood_.clear();
    times_.clear();
    for (const auto& ids : texts_vec) {
        std::vector<int> tokens(ids.begin(), ids.end());
        hypotheses_.push_back(tokens);
        outputs_.push_back(tokens);
        likelihood_.push_back(0.0f);  // Per-path score not exposed here.
        times_.push_back({});
    }
}

}  // namespace decoder
}  // namespace oasr

#endif  // OASR_USE_K2
