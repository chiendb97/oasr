// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/extension.h>

#include "kernels/conv/conv1d_kernels.h"
#include "kernels/conv/conv2d_kernels.h"

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
        "conv2d",
        [](const torch::Tensor& input, const torch::Tensor& filter, py::object bias_obj, int pad_h,
           int pad_w, int stride_h, int stride_w, int dilation_h, int dilation_w,
           py::object stream_obj) -> torch::Tensor {
            torch::Tensor bias;
            if (!bias_obj.is_none()) {
                bias = bias_obj.cast<torch::Tensor>();
            }
            cudaStream_t stream = stream_obj.is_none()
                                      ? nullptr
                                      : reinterpret_cast<cudaStream_t>(stream_obj.cast<intptr_t>());
            return oasr::kernels::invokeConv2d(input, filter, bias, pad_h, pad_w, stride_h,
                                               stride_w, dilation_h, dilation_w, stream);
        },
        py::arg("input"), py::arg("filter"), py::arg("bias") = py::none(), py::arg("pad_h") = 0,
        py::arg("pad_w") = 0, py::arg("stride_h") = 1, py::arg("stride_w") = 1,
        py::arg("dilation_h") = 1, py::arg("dilation_w") = 1, py::arg("stream") = py::none(),
        "2D convolution (NHWC layout, FP16/BF16) using CUTLASS Ampere Tensor Core Implicit GEMM. "
        "input [N,H,W,IC], filter [K,R,S,IC], bias [K] (optional), output [N,P,Q,K]. "
        "Requires IC % 8 == 0 and K % 8 == 0.");

    conv.def(
        "conv2d_activation",
        [](const torch::Tensor& input, const torch::Tensor& filter, py::object bias_obj,
           ActivationType activation, int pad_h, int pad_w, int stride_h, int stride_w,
           int dilation_h, int dilation_w, py::object stream_obj) -> torch::Tensor {
            torch::Tensor bias;
            if (!bias_obj.is_none()) {
                bias = bias_obj.cast<torch::Tensor>();
            }
            cudaStream_t stream = stream_obj.is_none()
                                      ? nullptr
                                      : reinterpret_cast<cudaStream_t>(stream_obj.cast<intptr_t>());
            return oasr::kernels::invokeConv2dActivation(input, filter, bias, activation, pad_h,
                                                         pad_w, stride_h, stride_w, dilation_h,
                                                         dilation_w, stream);
        },
        py::arg("input"), py::arg("filter"), py::arg("bias") = py::none(),
        py::arg("activation") = ActivationType::SWISH, py::arg("pad_h") = 0, py::arg("pad_w") = 0,
        py::arg("stride_h") = 1, py::arg("stride_w") = 1, py::arg("dilation_h") = 1,
        py::arg("dilation_w") = 1, py::arg("stream") = py::none(),
        "2D convolution with fused activation (NHWC layout, FP16/BF16) using CUTLASS Ampere "
        "Tensor Core Implicit GEMM. "
        "input [N,H,W,IC], filter [K,R,S,IC], bias [K] (optional), output [N,P,Q,K]. "
        "Supported activations: RELU, GELU, SWISH.");
}

}  // namespace pybind
}  // namespace oasr
