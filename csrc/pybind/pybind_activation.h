// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/extension.h>

#include "kernels/activation/activation_kernels.h"

namespace py = pybind11;

namespace oasr {
namespace pybind {

inline void registerActivationBindings(py::module_& kernels) {
    py::module_ activation = kernels.def_submodule("activation", "Activation kernels");

    activation.def(
        "glu",
        [](const torch::Tensor& input) -> torch::Tensor {
            int batch_size = input.size(0);
            int seq_len = input.size(1);
            int channels = input.size(2) / 2;
            return oasr::kernels::invokeGLU(input, nullptr);
        },
        py::arg("input"), "GLU (Gated Linear Unit) activation kernel");

    activation.def(
        "swish",
        [](const torch::Tensor& input) -> torch::Tensor {
            return oasr::kernels::invokeSwish(input, nullptr);
        },
        py::arg("input"), "Swish activation kernel");
}

}  // namespace pybind
}  // namespace oasr
