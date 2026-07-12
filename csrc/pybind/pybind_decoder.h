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
    // ContextGraph (phrase boosting) — public C++ class, used from Python
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

    // -----------------------------------------------------------------
    // _GreedySearchCore — internal C++ compute core for greedy decoding.
    // Use oasr.decoder.CtcGreedySearch (the Python wrapper) instead.
    // -----------------------------------------------------------------
    py::class_<CtcGreedySearch>(decoder_mod, "_GreedySearchCore")
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
            py::arg("logp"))
        .def("reset", &CtcGreedySearch::Reset)
        .def("finalize_search", &CtcGreedySearch::FinalizeSearch)
        .def(
            "set_context_graph",
            [](CtcGreedySearch& self, std::shared_ptr<ContextGraph> ctx) {
                self.SetContextGraph(std::move(ctx));
            },
            py::arg("context_graph"))
        .def_property_readonly(
            "outputs",
            &CtcGreedySearch::Outputs,
            py::return_value_policy::reference_internal)
        .def_property_readonly(
            "likelihood",
            &CtcGreedySearch::Likelihood,
            py::return_value_policy::reference_internal)
        .def_property_readonly(
            "times",
            &CtcGreedySearch::Times,
            py::return_value_policy::reference_internal);

    // -----------------------------------------------------------------
    // _PrefixBeamSearchCore — internal C++ compute core for prefix beam search.
    // Use oasr.decoder.CtcPrefixBeamSearch (the Python wrapper) instead.
    // -----------------------------------------------------------------
    py::class_<CtcPrefixBeamSearch>(decoder_mod, "_PrefixBeamSearchCore")
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
            py::arg("logp"))
        .def("reset", &CtcPrefixBeamSearch::Reset)
        .def("finalize_search", &CtcPrefixBeamSearch::FinalizeSearch)
        .def(
            "set_context_graph",
            [](CtcPrefixBeamSearch& self, std::shared_ptr<ContextGraph> ctx) {
                self.SetContextGraph(std::move(ctx));
            },
            py::arg("context_graph"))
        .def_property_readonly(
            "inputs",
            &CtcPrefixBeamSearch::Inputs,
            py::return_value_policy::reference_internal)
        .def_property_readonly(
            "outputs",
            &CtcPrefixBeamSearch::Outputs,
            py::return_value_policy::reference_internal)
        .def_property_readonly(
            "likelihood",
            &CtcPrefixBeamSearch::Likelihood,
            py::return_value_policy::reference_internal)
        .def_property_readonly(
            "times",
            &CtcPrefixBeamSearch::Times,
            py::return_value_policy::reference_internal)
        .def_property_readonly(
            "viterbi_likelihood",
            &CtcPrefixBeamSearch::viterbi_likelihood,
            py::return_value_policy::reference_internal);

#ifdef OASR_USE_K2
    // -----------------------------------------------------------------
    // _WfstBeamSearchCore — internal C++ compute core for K2 WFST decoding.
    // Use oasr.decoder.CtcWfstBeamSearch (the Python wrapper) instead.
    // -----------------------------------------------------------------
    py::class_<CtcWfstBeamSearch>(decoder_mod, "_WfstBeamSearchCore")
        .def_static(
            "from_file",
            [](const std::string& fst_path,
               int blank, float search_beam, float output_beam,
               int min_active_states, int max_active_states,
               int subsampling_factor, int nbest, float blank_skip_thresh) {
                CtcWfstBeamSearchOptions opts;
                opts.blank = blank;
                opts.search_beam = search_beam;
                opts.output_beam = output_beam;
                opts.min_active_states = min_active_states;
                opts.max_active_states = max_active_states;
                opts.subsampling_factor = subsampling_factor;
                opts.nbest = nbest;
                opts.blank_skip_thresh = blank_skip_thresh;
                return CtcWfstBeamSearch::FromFile(opts, fst_path, torch::kCPU);
            },
            py::arg("fst_path"),
            py::arg("blank") = 0,
            py::arg("search_beam") = 20.0f,
            py::arg("output_beam") = 8.0f,
            py::arg("min_active_states") = 30,
            py::arg("max_active_states") = 10000,
            py::arg("subsampling_factor") = 1,
            py::arg("nbest") = 10,
            py::arg("blank_skip_thresh") = 0.98f,
            "Load a decoding graph from *fst_path* and create a WFST beam-search core.")
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
            py::arg("logp"))
        .def("reset", &CtcWfstBeamSearch::Reset)
        .def("finalize_search", &CtcWfstBeamSearch::FinalizeSearch)
        .def_property_readonly(
            "outputs", &CtcWfstBeamSearch::Outputs,
            py::return_value_policy::reference_internal)
        .def_property_readonly(
            "likelihood", &CtcWfstBeamSearch::Likelihood,
            py::return_value_policy::reference_internal)
        .def_property_readonly(
            "times", &CtcWfstBeamSearch::Times,
            py::return_value_policy::reference_internal);

    decoder_mod.attr("k2_available") = true;
#else
    decoder_mod.attr("k2_available") = false;
#endif  // OASR_USE_K2
}

}  // namespace pybind
}  // namespace oasr
