// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cutlass/cutlass.h>

#include <torch/extension.h>

#include "kernels/gemm/bmm_kernels.h"
#include "kernels/gemm/gemm_kernels.h"
#include "kernels/gemm/group_gemm_kernels.h"

namespace py = pybind11;

namespace oasr {
namespace pybind {

inline void registerGemmBindings(py::module_& kernels) {
    using namespace kernels::gemm;

    py::module_ gemm_mod = kernels.def_submodule("gemm", "GEMM/BMM/GroupGEMM kernels (CUTLASS)");

    // --- Enums ---

    // --- GEMM: D = alpha * A @ B + beta * C ---
    gemm_mod.def(
        "gemm",
        [](const torch::Tensor& a, const torch::Tensor& b, std::optional<const torch::Tensor>& c,
           py::object stream) -> torch::Tensor {
            cudaStream_t s = stream.is_none()
                                 ? nullptr
                                 : reinterpret_cast<cudaStream_t>(stream.cast<intptr_t>());

            torch::Tensor c_tensor = c.has_value() ? c.value() : torch::Tensor();
            auto d = invokeGemm(a, b, c_tensor, s);
            return d;
        },
        py::arg("a"), py::arg("b"), py::arg("c") = py::none(), py::arg("stream") = py::none(),
        "Execute GEMM: D = A @ B + C (C is optional). Returns (output, status).");

    gemm_mod.def(
        "gemm_activation",
        [](const torch::Tensor& a, const torch::Tensor& b, std::optional<const torch::Tensor>& c,
           ActivationType activation, py::object stream) -> torch::Tensor {
            cudaStream_t s = stream.is_none()
                                 ? nullptr
                                 : reinterpret_cast<cudaStream_t>(stream.cast<intptr_t>());
            torch::Tensor c_tensor = c.has_value() ? c.value() : torch::Tensor();
            auto d = invokeGemmActivation(a, b, c_tensor, activation, s);
            return d;
        },
        py::arg("a"), py::arg("b"), py::arg("c") = py::none(), py::arg("activation"),
        py::arg("stream") = py::none(),
        "Execute GEMM with activation: D = activation(A @ B + C). Returns output tensor.");

    // --- Batched GEMM (strided) ---
    gemm_mod.def(
        "bmm",
        [](const torch::Tensor& a, const torch::Tensor& b, py::object stream) -> torch::Tensor {
            cudaStream_t s = stream.is_none()
                                 ? nullptr
                                 : reinterpret_cast<cudaStream_t>(stream.cast<intptr_t>());
            auto d = invokeBmm(a, b, s);
            return d;
        },
        py::arg("a"), py::arg("b"), py::arg("stream") = py::none(),
        "Execute batched GEMM. Returns output tensor.");

    // --- Grouped GEMM (pointer arrays, kept low-level) ---
    gemm_mod.def(
        "group_gemm",
        [](const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& offset,
           py::object stream) -> torch::Tensor {
            cudaStream_t s = stream.is_none()
                                 ? nullptr
                                 : reinterpret_cast<cudaStream_t>(stream.cast<intptr_t>());
            return invokeGroupGemm(a, b, offset, s);
        },
        py::arg("a"), py::arg("b"), py::arg("offset"), py::arg("stream") = py::none(),
        "Execute grouped GEMM. Returns output tensor.");
}

}  // namespace pybind
}  // namespace oasr
