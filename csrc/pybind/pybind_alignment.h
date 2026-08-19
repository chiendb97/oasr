// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Bindings for the post-decode alignment pass and the beam read-back.
//
// Both are pure data shuffling that used to run in Python on the engine's
// step-loop thread — the one thread that holds the GIL for every request the
// engine finishes.  What crosses the boundary here is one call per micro-batch
// row instead of one interpreter operation per token and per character.

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <torch/extension.h>
#include <tuple>
#include <vector>

#include "alignment/word_timings.h"
#include "tokenizers/symbol_table.h"

namespace py = pybind11;

namespace oasr {
namespace pybind {

namespace detail {

/// ``AlignmentResult`` -> ``(words, timestamps, confidence)``.
///
/// Words cross as tuples and become the public Python type there. Empty
/// alignments use `None` confidence to distinguish them from uncertainty.
inline py::tuple toPython(const oasr::alignment::AlignmentResult& result) {
    py::list words;
    for (const auto& w : result.words) {
        words.append(py::make_tuple(py::str(w.word), w.start, w.end, w.confidence));
    }
    py::object confidence =
        result.has_confidence ? py::cast(result.confidence) : py::object(py::none());
    return py::make_tuple(std::move(words), py::cast(result.timestamps), std::move(confidence));
}

/// ``(b, k)`` row of a padded ``(B, K, L)`` token tensor as a Python list.
template <typename T>
inline py::list rowToList(const T* data, int64_t stride_b, int64_t stride_k, int64_t b, int64_t k,
                          int64_t length) {
    py::list row;
    const T* p = data + b * stride_b + k * stride_k;
    for (int64_t i = 0; i < length; ++i) {
        row.append(static_cast<int64_t>(p[i]));
    }
    return row;
}

inline void checkBeamTensors(const torch::Tensor& values, const torch::Tensor& lengths) {
    TORCH_CHECK(values.dim() == 3, "values must be [batch, beam, max_len], got ", values.dim(),
                " dims");
    TORCH_CHECK(lengths.dim() == 2, "lengths must be [batch, beam], got ", lengths.dim(), " dims");
    TORCH_CHECK(values.device().is_cpu() && lengths.device().is_cpu(),
                "values and lengths must already be on the host — this is the read-back, not a "
                "copy engine");
    TORCH_CHECK(values.size(0) == lengths.size(0) && values.size(1) == lengths.size(1),
                "values and lengths disagree on [batch, beam]");
    TORCH_CHECK(values.scalar_type() == torch::kInt32 || values.scalar_type() == torch::kInt64,
                "values must be int32 or int64, got ", values.scalar_type());
    TORCH_CHECK(lengths.scalar_type() == torch::kInt32 || lengths.scalar_type() == torch::kInt64,
                "lengths must be int32 or int64, got ", lengths.scalar_type());
}

/// Bounded, clamped length for row ``(b, k)`` — a decoder that overran its cap
/// must not make this read past the buffer.
inline int64_t rowLength(const torch::Tensor& lengths, int64_t b, int64_t k, int64_t max_len) {
    const int64_t raw = lengths.scalar_type() == torch::kInt32
                            ? static_cast<int64_t>(lengths.index({b, k}).item<int32_t>())
                            : lengths.index({b, k}).item<int64_t>();
    return std::max<int64_t>(0, std::min<int64_t>(raw, max_len));
}

}  // namespace detail

inline void registerAlignmentBindings(py::module_& m) {
    py::module_ am = m.def_submodule("alignment", "Post-decode alignment and beam read-back");

    am.def(
        "align_emissions",
        [](const std::vector<int64_t>& frames, const std::vector<double>& confidences,
           const std::vector<std::string>& pieces, double seconds_per_frame, int64_t frame_offset,
           double offset, bool want_words) {
            oasr::alignment::AlignmentResult result;
            {
                py::gil_scoped_release unlock;
                result =
                    oasr::alignment::align_emissions(frames, confidences, pieces, seconds_per_frame,
                                                     frame_offset, offset, want_words);
            }
            return detail::toPython(result);
        },
        py::arg("frames"), py::arg("confidences"), py::arg("pieces"), py::arg("seconds_per_frame"),
        py::arg("frame_offset") = 0, py::arg("offset") = 0.0, py::arg("want_words") = true,
        "Emission frames + rendered pieces -> (words, timestamps, confidence).\n\n"
        "The frame-synchronous families (CTC, transducer) in one call: no\n"
        "per-token Python object is created at any point.");

    am.def(
        "align_spans",
        [](const std::vector<std::tuple<double, double, double, double>>& spans,
           const std::vector<std::string>& pieces, double seconds_per_frame, double offset,
           bool want_words) {
            std::vector<oasr::alignment::TokenSpan> converted;
            converted.reserve(spans.size());
            for (const auto& s : spans) {
                converted.push_back(
                    oasr::alignment::TokenSpan{std::get<1>(s), std::get<2>(s), std::get<3>(s)});
            }
            oasr::alignment::AlignmentResult result;
            {
                py::gil_scoped_release unlock;
                result = oasr::alignment::align(converted, pieces, seconds_per_frame, offset,
                                                want_words);
            }
            return detail::toPython(result);
        },
        py::arg("spans"), py::arg("pieces"), py::arg("seconds_per_frame"), py::arg("offset") = 0.0,
        py::arg("want_words") = true,
        "Per-token spans + rendered pieces -> (words, timestamps, confidence).\n\n"
        "``spans`` is a sequence of ``TokenAlignment`` — a NamedTuple, so it\n"
        "crosses as a tuple with no Python-level unpacking.  For the families\n"
        "whose spans are not emission frames (Paraformer CIF, AED DTW).");

    am.def(
        "extract_beam_tokens",
        [](const torch::Tensor& values, const torch::Tensor& lengths, int64_t beams) {
            detail::checkBeamTensors(values, lengths);
            auto v = values.contiguous();
            const int64_t batch = v.size(0);
            const int64_t n_beam = v.size(1);
            const int64_t max_len = v.size(2);
            const int64_t take = beams < 0 ? n_beam : std::min(beams, n_beam);
            auto lengths_cpu = lengths.to(torch::kInt64).contiguous();
            const int64_t* len_p = lengths_cpu.data_ptr<int64_t>();

            py::list out;
            for (int64_t b = 0; b < batch; ++b) {
                py::list rows;
                for (int64_t k = 0; k < take; ++k) {
                    const int64_t raw = len_p[b * n_beam + k];
                    const int64_t length = std::max<int64_t>(0, std::min(raw, max_len));
                    if (v.scalar_type() == torch::kInt32) {
                        rows.append(detail::rowToList<int32_t>(
                            v.data_ptr<int32_t>(), n_beam * max_len, max_len, b, k, length));
                    } else {
                        rows.append(detail::rowToList<int64_t>(
                            v.data_ptr<int64_t>(), n_beam * max_len, max_len, b, k, length));
                    }
                }
                out.append(std::move(rows));
            }
            return out;
        },
        py::arg("values"), py::arg("lengths"), py::arg("beams") = -1,
        "Padded ``[batch, beam, max_len]`` -> nested Python lists, trimmed to\n"
        "``lengths``.  In Python this was two tensor operations per (row, beam)\n"
        "— an index producing a 0-d tensor for the length, then a slice — which\n"
        "at beam 16 cost more than the decode's own device->host copy.");

    am.def(
        "extract_beam_row",
        [](const torch::Tensor& values, const torch::Tensor& lengths, int64_t b, int64_t k) {
            detail::checkBeamTensors(values, lengths);
            auto v = values.contiguous();
            TORCH_CHECK(b >= 0 && b < v.size(0), "batch index ", b, " out of range");
            TORCH_CHECK(k >= 0 && k < v.size(1), "beam index ", k, " out of range");
            const int64_t max_len = v.size(2);
            const int64_t length = detail::rowLength(lengths, b, k, max_len);
            if (v.scalar_type() == torch::kInt32) {
                return detail::rowToList<int32_t>(v.data_ptr<int32_t>(), v.size(1) * max_len,
                                                  max_len, b, k, length);
            }
            return detail::rowToList<int64_t>(v.data_ptr<int64_t>(), v.size(1) * max_len, max_len,
                                              b, k, length);
        },
        py::arg("values"), py::arg("lengths"), py::arg("b"), py::arg("k"),
        "One ``(b, k)`` row of a padded beam tensor, trimmed to its length.");

    py::class_<oasr::tokenizers::SymbolTablePieces>(
        am, "SymbolTablePieces",
        "Id -> piece rendering for a flat symbol table (``units.txt`` /\n"
        "``tokens.txt``).  Built once per tokenizer; ``pieces`` is what the\n"
        "word grouping calls per finished hypothesis.")
        .def(py::init<std::unordered_map<int64_t, std::string>, std::unordered_set<int64_t>>(),
             py::arg("table"), py::arg("special_ids"))
        .def("pieces", &oasr::tokenizers::SymbolTablePieces::pieces, py::arg("ids"),
             "Per-token text contributions, concatenating to ``decode(ids)``.")
        .def("__len__", &oasr::tokenizers::SymbolTablePieces::size);

    // Exposed so the differential test can check the two classification
    // predicates directly against CPython's own answer, rather than only
    // inferring them from grouped output.
    am.def("is_space", &oasr::alignment::is_space, py::arg("codepoint"),
           "Python ``str.isspace()`` for one code point, as this module sees it.");
    am.def("is_spaceless", &oasr::alignment::is_spaceless, py::arg("codepoint"),
           "Whether a code point belongs to a script written without spaces.");
}

}  // namespace pybind
}  // namespace oasr
