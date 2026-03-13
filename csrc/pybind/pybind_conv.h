// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/extension.h>

#include "kernels/conv/conv_kernels.h"

namespace py = pybind11;

namespace oasr {
namespace pybind {

inline void registerConvBindings(py::module_& kernels) {
    py::module_ conv = kernels.def_submodule("conv", "Convolution kernels");

    conv.def(
        "depthwise_conv1d",
        [](const torch::Tensor& input, const torch::Tensor& weight, py::object bias_obj,
           int padding) -> torch::Tensor {
            torch::Tensor bias;
            if (!bias_obj.is_none()) {
                bias = bias_obj.cast<torch::Tensor>();
            }
            return oasr::kernels::invokeDepthwiseConv1D(input, weight, bias, padding, nullptr);
        },
        py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(), py::arg("padding") = 0,
        "Depthwise 1D convolution kernel");

    conv.def(
        "depthwise_conv1d_silu",
        [](const torch::Tensor& input, const torch::Tensor& weight, py::object bias_obj,
           int padding) -> torch::Tensor {
            torch::Tensor bias;
            if (!bias_obj.is_none()) {
                bias = bias_obj.cast<torch::Tensor>();
            }
            return oasr::kernels::invokeDepthwiseConv1DSilu(input, weight, bias, padding, nullptr);
        },
        py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(), py::arg("padding") = 0,
        "Fused Depthwise 1D convolution + SiLU kernel");

    conv.def(
        "pointwise_conv1d",
        [](const torch::Tensor& input, const torch::Tensor& weight,
           py::object bias_obj) -> torch::Tensor {
            torch::Tensor bias;
            if (!bias_obj.is_none()) {
                bias = bias_obj.cast<torch::Tensor>();
            }
            return oasr::kernels::invokePointwiseConv1D(input, weight, bias, nullptr);
        },
        py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(),
        "Pointwise (1x1) convolution kernel");

    conv.def(
        "pointwise_conv1d_activation",
        [](const torch::Tensor& input, const torch::Tensor& weight, py::object bias_obj,
           ActivationType activation) -> torch::Tensor {
            torch::Tensor bias;
            if (!bias_obj.is_none()) {
                bias = bias_obj.cast<torch::Tensor>();
            }
            return oasr::kernels::invokePointwiseConv1DActivation(input, weight, bias, activation,
                                                                  nullptr);
        },
        py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(),
        py::arg("activation") = ActivationType::SWISH,
        "Pointwise (1x1) convolution with activation kernel");

    conv.def(
        "glu",
        [](const torch::Tensor& input) -> torch::Tensor {
            int batch_size = input.size(0);
            int seq_len = input.size(1);
            int channels = input.size(2) / 2;
            return oasr::kernels::invokeGLU(input, nullptr);
        },
        py::arg("input"),
        "GLU (Gated Linear Unit) activation kernel");

    conv.def(
        "swish",
        [](const torch::Tensor& input) -> torch::Tensor {
            return oasr::kernels::invokeSwish(input, nullptr);
        },
        py::arg("input"), "Swish activation kernel");

    conv.def(
        "batch_norm_swish",
        [](const torch::Tensor& input, const torch::Tensor& gamma, const torch::Tensor& beta,
           const torch::Tensor& running_mean, const torch::Tensor& running_var, float eps) -> torch::Tensor {
            return oasr::kernels::invokeBatchNormSwish(input, gamma, beta, running_mean,
                                                       running_var, eps, nullptr);
        },
        py::arg("input"), py::arg("gamma"), py::arg("beta"), py::arg("running_mean"),
        py::arg("running_var"), py::arg("eps") = 1e-5f,
        "Fused BatchNorm + Swish kernel");
}

}  // namespace pybind
}  // namespace oasr
