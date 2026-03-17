// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/extension.h>

#include "decoder/ctc_prefix_beam_search.h"

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
            "Viterbi path likelihoods for each hypothesis.");
}

}  // namespace pybind
}  // namespace oasr

