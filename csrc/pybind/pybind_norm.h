// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/extension.h>
#include "kernels/norm/norm_kernels.h"

namespace py = pybind11;

namespace oasr {
namespace pybind {

inline void registerNormBindings(py::module_& kernels) {
    py::module_ norm = kernels.def_submodule("norm", "Normalization kernels");

    norm.def("layer_norm",
        [](const torch::Tensor& input,
           const torch::Tensor& weight, py::object bias_obj,
           float eps) -> torch::Tensor {
            auto output = torch::empty_like(input);
            torch::Tensor bias;
            if (!bias_obj.is_none()) {
                bias = bias_obj.cast<torch::Tensor>();
            }
            oasr::kernels::invokeLayerNorm(
                input, output, weight, bias,
                eps, nullptr);
            return output;
        },
        py::arg("input"), py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("eps") = 1e-5f,
        "Layer normalization kernel");

    norm.def("rms_norm",
        [](const torch::Tensor& input,
           const torch::Tensor& weight, py::object bias_obj,
           float eps) -> torch::Tensor {
            auto output = torch::empty_like(input);
            torch::Tensor bias;
            if (!bias_obj.is_none()) {
                bias = bias_obj.cast<torch::Tensor>();
            }
            oasr::kernels::invokeRMSNorm(
                input, output, weight, bias,
                eps, nullptr);
            return output;
        },
        py::arg("input"), py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("eps") = 1e-5f,
        "RMS normalization kernel");

    norm.def("batch_norm_1d",
        [](const torch::Tensor& input,
           const torch::Tensor& weight, py::object bias_obj,
           const torch::Tensor& running_mean, const torch::Tensor& running_var,
           float eps) -> torch::Tensor {
            auto output = torch::empty_like(input);
            torch::Tensor bias;
            if (!bias_obj.is_none()) {
                bias = bias_obj.cast<torch::Tensor>();
            }
            oasr::kernels::invokeBatchNorm1D(
                input, output, weight, bias,
                running_mean, running_var,
                eps, nullptr);
            return output;
        },
        py::arg("input"),
        py::arg("weight"), py::arg("bias"),
        py::arg("running_mean"), py::arg("running_var"),
        py::arg("eps") = 1e-5f,
        "1D batch normalization kernel (inference mode)");

    norm.def("group_norm",
        [](const torch::Tensor& input,
           const torch::Tensor& weight, py::object bias_obj,
           int num_groups,
           float eps) -> torch::Tensor {
            auto output = torch::empty_like(input);
            torch::Tensor bias;
            if (!bias_obj.is_none()) {
                bias = bias_obj.cast<torch::Tensor>();
            }
            oasr::kernels::invokeGroupNorm(
                input, output, weight, bias,
                num_groups,
                eps, nullptr);
            return output;
        },
        py::arg("input"),
        py::arg("weight"), py::arg("bias"),
        py::arg("num_groups"),
        py::arg("eps") = 1e-5f,
        "Group normalization kernel");

    norm.def("add_layer_norm",
        [](const torch::Tensor& input, const torch::Tensor& residual,
           const torch::Tensor& weight, py::object bias_obj,
           float eps) -> torch::Tensor {
            auto output = torch::empty_like(input);
            torch::Tensor bias;
            if (!bias_obj.is_none()) {
                bias = bias_obj.cast<torch::Tensor>();
            }
            oasr::kernels::invokeAddLayerNorm(
                input, residual, output,
                weight, bias,
                eps, nullptr);
            return output;
        },
        py::arg("input"), py::arg("residual"),
        py::arg("weight"), py::arg("bias") = py::none(),
        py::arg("eps") = 1e-5f,
        "Fused add + layer norm kernel");
}

} // namespace pybind
} // namespace oasr
