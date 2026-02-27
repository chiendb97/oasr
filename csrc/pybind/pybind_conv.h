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

    conv.def("depthwise_conv1d",
        [](const torch::Tensor& input, const torch::Tensor& weight,
           py::object bias_obj,
           int padding, DataType dtype) -> torch::Tensor {
            int batch_size = input.size(0);
            int seq_len = input.size(1);
            int channels = input.size(2);
            int kernel_size = weight.size(0);
            auto output_seq_len = seq_len + 2 * padding - kernel_size + 1;
            auto output = torch::empty(
                {batch_size, output_seq_len, channels}, input.options());
            torch::Tensor bias;
            if (!bias_obj.is_none()) {
                bias = bias_obj.cast<torch::Tensor>();
            }
            oasr::kernels::invokeDepthwiseConv1D(
                input, weight, bias,
                output, padding, dtype, nullptr);
            return output;
        },
        py::arg("input"), py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("padding") = 0, py::arg("dtype") = DataType::FP16,
        "Depthwise 1D convolution kernel");

        conv.def("depthwise_conv1d_silu",
            [](const torch::Tensor& input, const torch::Tensor& weight,
               py::object bias_obj,
               int padding, DataType dtype) -> torch::Tensor {
                int batch_size = input.size(0);
                int seq_len = input.size(1);
                int channels = input.size(2);
                int kernel_size = weight.size(0);
                auto output_seq_len = seq_len + 2 * padding - kernel_size + 1;
                auto output = torch::empty(
                    {batch_size, output_seq_len, channels}, input.options());
                torch::Tensor bias;
                if (!bias_obj.is_none()) {
                    bias = bias_obj.cast<torch::Tensor>();
                }
                oasr::kernels::invokeDepthwiseConv1DSilu(
                    input, weight, bias,
                    output, padding, dtype, nullptr);
                return output;
            },
            py::arg("input"), py::arg("weight"),
            py::arg("bias") = py::none(),
            py::arg("padding") = 0, py::arg("dtype") = DataType::FP16,
            "Fused Depthwise 1D convolution + SiLU kernel");

    conv.def("pointwise_conv1d",
        [](const torch::Tensor& input, const torch::Tensor& weight,
           py::object bias_obj,
           ActivationType activation, bool fuse_activation,
           DataType dtype) -> torch::Tensor {
            int batch_size = input.size(0);
            int seq_len = input.size(1);
            int out_channels = weight.size(0);
            auto output = torch::empty(
                {batch_size, seq_len, out_channels}, input.options());
            torch::Tensor bias;
            if (!bias_obj.is_none()) {
                bias = bias_obj.cast<torch::Tensor>();
            }
            oasr::kernels::invokePointwiseConv1D(
                input, weight, bias,
                output, activation, fuse_activation, dtype, nullptr);
            return output;
        },
        py::arg("input"), py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("activation") = ActivationType::SWISH,
        py::arg("fuse_activation") = false,
        py::arg("dtype") = DataType::FP16,
        "Pointwise (1x1) convolution kernel");

    conv.def("glu",
        [](const torch::Tensor& input,
           DataType dtype) -> torch::Tensor {
            int batch_size = input.size(0);
            int seq_len = input.size(1);
            int channels = input.size(2) / 2;
            auto output = torch::empty(
                {batch_size, seq_len, channels}, input.options());
            oasr::kernels::invokeGLU(
                input, output, dtype, nullptr);
            return output;
        },
        py::arg("input"),
        py::arg("dtype") = DataType::FP16,
        "GLU (Gated Linear Unit) activation kernel");

    conv.def("swish",
        [](const torch::Tensor& input,
           DataType dtype) -> torch::Tensor {
            auto output = torch::empty_like(input);
            oasr::kernels::invokeSwish(
                input, output, dtype, nullptr);
            return output;
        },
        py::arg("input"),
        py::arg("dtype") = DataType::FP16,
        "Swish activation kernel");

    conv.def("batch_norm_swish",
        [](const torch::Tensor& input,
           const torch::Tensor& gamma, const torch::Tensor& beta,
           const torch::Tensor& running_mean, const torch::Tensor& running_var,
           float eps, DataType dtype) -> torch::Tensor {
            auto output = torch::empty_like(input);
            oasr::kernels::invokeBatchNormSwish(
                input, output,
                gamma, beta,
                running_mean, running_var,
                eps, dtype, nullptr);
            return output;
        },
        py::arg("input"),
        py::arg("gamma"), py::arg("beta"),
        py::arg("running_mean"), py::arg("running_var"),
        py::arg("eps") = 1e-5f, py::arg("dtype") = DataType::FP16,
        "Fused BatchNorm + Swish kernel");

    conv.def("conv1d",
        [](const torch::Tensor& input, const torch::Tensor& weight,
           py::object bias_obj,
           int stride, int padding, int dilation, int groups,
           ConvType conv_type, DataType dtype, bool channels_last,
           bool is_causal, ActivationType activation, bool fuse_activation) -> torch::Tensor {
            int batch_size = input.size(0);
            int seq_len = input.size(1);
            int out_channels = weight.size(0);
            int kernel_size = weight.size(-1);
            int out_seq = (seq_len + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
            torch::Tensor output;
            if (channels_last) {
                output = torch::empty(
                    {batch_size, out_seq, out_channels}, input.options());
            } else {
                output = torch::empty(
                    {batch_size, out_channels, out_seq}, input.options());
            }
            torch::Tensor bias;
            if (!bias_obj.is_none()) {
                bias = bias_obj.cast<torch::Tensor>();
            }
            oasr::kernels::invokeConv1D(
                input, output, weight, bias,
                stride, padding, dilation, groups,
                conv_type, dtype, channels_last, is_causal,
                activation, fuse_activation, nullptr);
            return output;
        },
        py::arg("input"), py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0, py::arg("dilation") = 1, py::arg("groups") = 1,
        py::arg("conv_type") = ConvType::STANDARD,
        py::arg("dtype") = DataType::FP16,
        py::arg("channels_last") = true, py::arg("is_causal") = false,
        py::arg("activation") = ActivationType::SWISH,
        py::arg("fuse_activation") = false,
        "General 1D convolution kernel");
}

} // namespace pybind
} // namespace oasr
