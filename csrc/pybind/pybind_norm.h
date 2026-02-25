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
           const torch::Tensor& gamma, py::object beta_obj,
           int batch_size, int seq_len, int hidden_size,
           float eps, DataType dtype) -> torch::Tensor {
            auto output = torch::empty_like(input);
            const void* beta_ptr = beta_obj.is_none()
                ? nullptr : beta_obj.cast<torch::Tensor>().data_ptr();
            oasr::kernels::invokeLayerNorm(
                input.data_ptr(), output.data_ptr(),
                gamma.data_ptr(), beta_ptr,
                batch_size, seq_len, hidden_size,
                eps, dtype, nullptr);
            return output;
        },
        py::arg("input"), py::arg("gamma"),
        py::arg("beta") = py::none(),
        py::arg("batch_size"), py::arg("seq_len"), py::arg("hidden_size"),
        py::arg("eps") = 1e-5f, py::arg("dtype") = DataType::FP16,
        "Layer normalization kernel");

    norm.def("rms_norm",
        [](const torch::Tensor& input,
           const torch::Tensor& gamma, py::object beta_obj,
           int batch_size, int seq_len, int hidden_size,
           float eps, DataType dtype) -> torch::Tensor {
            auto output = torch::empty_like(input);
            const void* beta_ptr = beta_obj.is_none()
                ? nullptr : beta_obj.cast<torch::Tensor>().data_ptr();
            oasr::kernels::invokeRMSNorm(
                input.data_ptr(), output.data_ptr(),
                gamma.data_ptr(), beta_ptr,
                batch_size, seq_len, hidden_size,
                eps, dtype, nullptr);
            return output;
        },
        py::arg("input"), py::arg("gamma"),
        py::arg("beta") = py::none(),
        py::arg("batch_size"), py::arg("seq_len"), py::arg("hidden_size"),
        py::arg("eps") = 1e-5f, py::arg("dtype") = DataType::FP16,
        "RMS normalization kernel");

    norm.def("batch_norm_1d",
        [](const torch::Tensor& input,
           const torch::Tensor& gamma, py::object beta_obj,
           const torch::Tensor& running_mean, const torch::Tensor& running_var,
           int batch_size, int seq_len, int channels,
           float eps, DataType dtype) -> torch::Tensor {
            auto output = torch::empty_like(input);
            const void* beta_ptr = beta_obj.is_none()
                ? nullptr : beta_obj.cast<torch::Tensor>().data_ptr();
            oasr::kernels::invokeBatchNorm1D(
                input.data_ptr(), output.data_ptr(),
                gamma.data_ptr(), beta_ptr,
                running_mean.data_ptr(), running_var.data_ptr(),
                batch_size, seq_len, channels,
                eps, dtype, nullptr);
            return output;
        },
        py::arg("input"),
        py::arg("gamma"), py::arg("beta"),
        py::arg("running_mean"), py::arg("running_var"),
        py::arg("batch_size"), py::arg("seq_len"), py::arg("channels"),
        py::arg("eps") = 1e-5f, py::arg("dtype") = DataType::FP16,
        "1D batch normalization kernel (inference mode)");

    norm.def("group_norm",
        [](const torch::Tensor& input,
           const torch::Tensor& gamma, py::object beta_obj,
           int batch_size, int seq_len, int channels, int num_groups,
           float eps, DataType dtype) -> torch::Tensor {
            auto output = torch::empty_like(input);
            const void* beta_ptr = beta_obj.is_none()
                ? nullptr : beta_obj.cast<torch::Tensor>().data_ptr();
            oasr::kernels::invokeGroupNorm(
                input.data_ptr(), output.data_ptr(),
                gamma.data_ptr(), beta_ptr,
                batch_size, seq_len, channels, num_groups,
                eps, dtype, nullptr);
            return output;
        },
        py::arg("input"),
        py::arg("gamma"), py::arg("beta"),
        py::arg("batch_size"), py::arg("seq_len"),
        py::arg("channels"), py::arg("num_groups"),
        py::arg("eps") = 1e-5f, py::arg("dtype") = DataType::FP16,
        "Group normalization kernel");

    norm.def("add_layer_norm",
        [](const torch::Tensor& input, const torch::Tensor& residual,
           const torch::Tensor& gamma, py::object beta_obj,
           int batch_size, int seq_len, int hidden_size,
           float eps, DataType dtype) -> torch::Tensor {
            auto output = torch::empty_like(input);
            const void* beta_ptr = beta_obj.is_none()
                ? nullptr : beta_obj.cast<torch::Tensor>().data_ptr();
            oasr::kernels::invokeAddLayerNorm(
                input.data_ptr(), residual.data_ptr(), output.data_ptr(),
                gamma.data_ptr(), beta_ptr,
                batch_size, seq_len, hidden_size,
                eps, dtype, nullptr);
            return output;
        },
        py::arg("input"), py::arg("residual"),
        py::arg("gamma"), py::arg("beta") = py::none(),
        py::arg("batch_size"), py::arg("seq_len"), py::arg("hidden_size"),
        py::arg("eps") = 1e-5f, py::arg("dtype") = DataType::FP16,
        "Fused add + layer norm kernel");
}

} // namespace pybind
} // namespace oasr
