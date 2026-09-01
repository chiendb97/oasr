// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Bindings for the streaming step loop's host-side data shuffling.
//
// One call per tick replaces ~85 Python->ATen crossings per tick at sixteen
// streams.  What is removed is the binding cost of each crossing, not the ATen
// dispatch underneath: the same slice measured 1.246 us from Python and
// 0.422 us from C++.

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <torch/extension.h>

#include <optional>
#include <vector>

#include "streaming/feature_ring.h"

namespace py = pybind11;

namespace oasr {
namespace pybind {

inline void registerStreamingBindings(py::module_& m) {
    py::module_ sm = m.def_submodule("streaming", "Streaming step-loop host helpers");

    sm.def(
        "append_features",
        [](const std::vector<std::optional<at::Tensor>>& buffers,
           const std::vector<int64_t>& frames,
           const std::vector<int64_t>& cursors,
           const at::Tensor& feats,
           const std::vector<int64_t>& new_lens,
           int64_t feat_dim,
           int64_t headroom_appends,
           int64_t headroom_max) {
            oasr::streaming::AppendResult r;
            {
                // The whole loop is ATen calls and integer arithmetic; nothing
                // in it touches a Python object, so the GIL is dead weight for
                // its duration and the engine's other threads can run.
                py::gil_scoped_release release;
                r = oasr::streaming::appendFeatures(buffers, frames, cursors, feats,
                                                    new_lens, feat_dim,
                                                    headroom_appends, headroom_max);
            }
            return py::make_tuple(std::move(r.buffers), std::move(r.frames),
                                  std::move(r.cursors), std::move(r.base_delta),
                                  r.reallocations);
        },
        py::arg("buffers"), py::arg("frames"), py::arg("cursors"), py::arg("feats"),
        py::arg("new_lens"), py::arg("feat_dim"), py::arg("headroom_appends"),
        py::arg("headroom_max"),
        R"doc(Append each stream's new feature frames to its ring buffer.

Returns ``(buffers, frames, cursors, base_delta, reallocations)``, each list
indexed by the caller's stream order.  A stream whose ``new_lens`` entry is zero
is passed through untouched.  ``base_delta`` is what to *add* to that stream's
``feature_base``; it is non-zero exactly when the buffer was compacted and the
cursor rebased to zero.)doc");
}

}  // namespace pybind
}  // namespace oasr
