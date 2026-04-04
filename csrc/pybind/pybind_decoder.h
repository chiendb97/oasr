// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/extension.h>

#include "decoder/context_graph.h"
#include "decoder/ctc_greedy_search.h"
#include "decoder/ctc_prefix_beam_search.h"
#ifdef OASR_USE_K2
#include "decoder/ctc_wfst_beam_search.h"
#endif

namespace py = pybind11;

namespace oasr {
namespace pybind {

inline std::vector<std::vector<float>> tensorToLogProbs(const torch::Tensor& logp) {
    TORCH_CHECK(logp.dim() == 2, "logp must be a 2D tensor [T, V]");
    TORCH_CHECK(logp.scalar_type() == torch::kFloat32,
                "logp must be a float32 tensor (got ", logp.scalar_type(), ")");

    auto logp_contig = logp.contiguous();
    const int64_t T = logp_contig.size(0);
    const int64_t V = logp_contig.size(1);

    auto accessor = logp_contig.accessor<float, 2>();
    std::vector<std::vector<float>> result;
    result.resize(static_cast<size_t>(T));
    for (int64_t t = 0; t < T; ++t) {
        auto& row = result[static_cast<size_t>(t)];
        row.resize(static_cast<size_t>(V));
        for (int64_t v = 0; v < V; ++v) {
            row[static_cast<size_t>(v)] = accessor[t][v];
        }
    }
    return result;
}

inline void registerDecoderBindings(py::module_& m) {
    using namespace oasr::decoder;

    py::module_ decoder_mod = m.def_submodule("decoder", "Decoder search algorithms");

    // -----------------------------------------------------------------
    // ContextGraph (phrase boosting)
    // -----------------------------------------------------------------
    py::class_<ContextGraph, std::shared_ptr<ContextGraph>>(decoder_mod, "ContextGraph")
        .def(py::init([](const std::vector<std::vector<int>>& phrases, float context_score) {
                 return std::make_shared<ContextGraph>(phrases, context_score);
             }),
             py::arg("phrases"), py::arg("context_score") = 3.0f,
             "Build a phrase-boosting context graph from a list of token-ID phrases.")
        .def_property_readonly("start_state", &ContextGraph::start_state,
                               "Index of the initial (root) state.")
        .def(
            "get_next_state",
            [](const ContextGraph& self, int state, int token) {
                auto [new_state, delta] = self.GetNextState(state, token);
                return py::make_tuple(new_state, delta);
            },
            py::arg("state"), py::arg("token"),
            "Advance the context state by one token. Returns (new_state, score_delta).")
        .def("get_backoff_score", &ContextGraph::GetBackoffScore, py::arg("state"),
             "Penalty for an uncommitted partial match at finalization.");

    py::enum_<SearchType>(decoder_mod, "SearchType")
        .value("kPrefixBeamSearch", SearchType::kPrefixBeamSearch)
        .value("kWfstBeamSearch", SearchType::kWfstBeamSearch)
        .value("kGreedySearch", SearchType::kGreedySearch)
        .export_values();

    // -----------------------------------------------------------------
    // CTC Greedy Search
    // -----------------------------------------------------------------
    py::class_<CtcGreedySearchOptions>(decoder_mod, "CtcGreedySearchOptions")
        .def(py::init<>())
        .def_readwrite("blank", &CtcGreedySearchOptions::blank);

    py::class_<CtcGreedySearch>(decoder_mod, "CtcGreedySearch")
        .def(py::init<const CtcGreedySearchOptions&>(), py::arg("options"))
        .def(py::init([](int blank) {
                 CtcGreedySearchOptions opts;
                 opts.blank = blank;
                 return std::make_unique<CtcGreedySearch>(opts);
             }),
             py::arg("blank") = 0)
        .def(
            "search",
            [](CtcGreedySearch& self, const torch::Tensor& logp) {
                if (logp.dim() != 2) {
                    throw py::value_error(
                        "logp must be a 2-D tensor [T, V], got " +
                        std::to_string(logp.dim()) + "-D");
                }
                auto logp_f32 = logp.detach().cpu().to(torch::kFloat32);
                auto logp_vec = tensorToLogProbs(logp_f32);
                self.Search(logp_vec);
            },
            py::arg("logp"),
            "Run CTC greedy search on a 2D log-probability tensor [T, V].")
        .def("reset", &CtcGreedySearch::Reset, "Reset internal decoder state.")
        .def("finalize_search", &CtcGreedySearch::FinalizeSearch,
             "Finalize search (no-op for greedy).")
        .def(
            "set_context_graph",
            [](CtcGreedySearch& self, std::shared_ptr<ContextGraph> ctx) {
                self.SetContextGraph(std::move(ctx));
            },
            py::arg("context_graph"),
            "Attach a ContextGraph for phrase boosting (pass None to disable).")
        .def_property_readonly(
            "outputs",
            &CtcGreedySearch::Outputs,
            py::return_value_policy::reference_internal,
            "Best output token ID sequence (wrapped in a size-1 list).")
        .def_property_readonly(
            "likelihood",
            &CtcGreedySearch::Likelihood,
            py::return_value_policy::reference_internal,
            "Cumulative log-likelihood of the best path (including context bonus).")
        .def_property_readonly(
            "times",
            &CtcGreedySearch::Times,
            py::return_value_policy::reference_internal,
            "Per-token frame timestamps.");

    // -----------------------------------------------------------------
    // CTC Prefix Beam Search
    // -----------------------------------------------------------------
    py::class_<CtcPrefixBeamSearchOptions>(decoder_mod, "CtcPrefixBeamSearchOptions")
        .def(py::init<>())
        .def_readwrite("blank", &CtcPrefixBeamSearchOptions::blank)
        .def_readwrite("first_beam_size", &CtcPrefixBeamSearchOptions::first_beam_size)
        .def_readwrite("second_beam_size", &CtcPrefixBeamSearchOptions::second_beam_size);

    py::class_<CtcPrefixBeamSearch>(decoder_mod, "CtcPrefixBeamSearch")
        .def(py::init<const CtcPrefixBeamSearchOptions&>(), py::arg("options"))
        .def(py::init([](int blank, int first_beam_size, int second_beam_size) {
                 CtcPrefixBeamSearchOptions opts;
                 opts.blank = blank;
                 opts.first_beam_size = first_beam_size;
                 opts.second_beam_size = second_beam_size;
                 return std::make_unique<CtcPrefixBeamSearch>(opts);
             }),
             py::arg("blank") = 0, py::arg("first_beam_size") = 10,
             py::arg("second_beam_size") = 10)
        .def(
            "search",
            [](CtcPrefixBeamSearch& self, const torch::Tensor& logp) {
                if (logp.dim() != 2) {
                    throw py::value_error(
                        "logp must be a 2-D tensor [T, V], got " +
                        std::to_string(logp.dim()) + "-D");
                }
                auto logp_f32 = logp.detach().cpu().to(torch::kFloat32);
                auto logp_vec = tensorToLogProbs(logp_f32);
                self.Search(logp_vec);
            },
            py::arg("logp"),
            "Run CTC prefix beam search on a 2D log-probability tensor [T, V].")
        .def("reset", &CtcPrefixBeamSearch::Reset, "Reset internal decoder state.")
        .def("finalize_search", &CtcPrefixBeamSearch::FinalizeSearch,
             "Finalize search and update hypotheses.")
        .def_property_readonly(
            "inputs",
            &CtcPrefixBeamSearch::Inputs,
            py::return_value_policy::reference_internal,
            "N-best input ID sequences.")
        .def_property_readonly(
            "outputs",
            &CtcPrefixBeamSearch::Outputs,
            py::return_value_policy::reference_internal,
            "N-best output ID sequences.")
        .def_property_readonly(
            "likelihood",
            &CtcPrefixBeamSearch::Likelihood,
            py::return_value_policy::reference_internal,
            "Likelihood scores for each hypothesis.")
        .def_property_readonly(
            "times",
            &CtcPrefixBeamSearch::Times,
            py::return_value_policy::reference_internal,
            "Per-token timestamps for each hypothesis.")
        .def_property_readonly(
            "viterbi_likelihood",
            &CtcPrefixBeamSearch::viterbi_likelihood,
            py::return_value_policy::reference_internal,
            "Viterbi path likelihoods for each hypothesis.")
        .def(
            "set_context_graph",
            [](CtcPrefixBeamSearch& self, std::shared_ptr<ContextGraph> ctx) {
                self.SetContextGraph(std::move(ctx));
            },
            py::arg("context_graph"),
            "Attach a ContextGraph for phrase boosting (pass None to disable).");

#ifdef OASR_USE_K2
    // -----------------------------------------------------------------
    // CTC WFST Beam Search (K2)
    // -----------------------------------------------------------------
    py::class_<CtcWfstBeamSearchOptions>(decoder_mod, "CtcWfstBeamSearchOptions")
        .def(py::init<>())
        .def_readwrite("blank", &CtcWfstBeamSearchOptions::blank)
        .def_readwrite("search_beam", &CtcWfstBeamSearchOptions::search_beam)
        .def_readwrite("output_beam", &CtcWfstBeamSearchOptions::output_beam)
        .def_readwrite("min_active_states", &CtcWfstBeamSearchOptions::min_active_states)
        .def_readwrite("max_active_states", &CtcWfstBeamSearchOptions::max_active_states)
        .def_readwrite("subsampling_factor", &CtcWfstBeamSearchOptions::subsampling_factor)
        .def_readwrite("nbest", &CtcWfstBeamSearchOptions::nbest)
        .def_readwrite("blank_skip_thresh", &CtcWfstBeamSearchOptions::blank_skip_thresh);

    py::class_<CtcWfstBeamSearch>(decoder_mod, "CtcWfstBeamSearch")
        // Primary constructor: load graph from a file saved with torch.save(fsa.as_dict(), path)
        .def_static(
            "from_file",
            [](const std::string& fst_path, const CtcWfstBeamSearchOptions& opts) {
                return CtcWfstBeamSearch::FromFile(opts, fst_path, torch::kCPU);
            },
            py::arg("fst_path"), py::arg("options") = CtcWfstBeamSearchOptions{},
            "Load a decoding graph from *fst_path* (saved with torch.save(fsa.as_dict(), ...)) "
            "and create a WFST beam-search decoder.")
        .def(
            "search",
            [](CtcWfstBeamSearch& self, const torch::Tensor& logp) {
                if (logp.dim() != 2) {
                    throw py::value_error(
                        "logp must be a 2-D tensor [T, V], got " +
                        std::to_string(logp.dim()) + "-D");
                }
                auto logp_f32 = logp.detach().cpu().to(torch::kFloat32);
                auto logp_vec = tensorToLogProbs(logp_f32);
                self.Search(logp_vec);
            },
            py::arg("logp"),
            "Feed a chunk of log-probability frames [T, V] to the WFST decoder.")
        .def("reset", &CtcWfstBeamSearch::Reset, "Reset internal decoder state.")
        .def("finalize_search", &CtcWfstBeamSearch::FinalizeSearch,
             "Finalize WFST decoding and extract N-best paths.")
        .def_property_readonly(
            "outputs", &CtcWfstBeamSearch::Outputs,
            py::return_value_policy::reference_internal, "N-best decoded token sequences.")
        .def_property_readonly(
            "likelihood", &CtcWfstBeamSearch::Likelihood,
            py::return_value_policy::reference_internal, "N-best path scores.")
        .def_property_readonly(
            "times", &CtcWfstBeamSearch::Times,
            py::return_value_policy::reference_internal, "Per-token timestamps (if available).");

    decoder_mod.attr("k2_available") = true;
#else
    decoder_mod.attr("k2_available") = false;
#endif  // OASR_USE_K2
}

}  // namespace pybind
}  // namespace oasr

