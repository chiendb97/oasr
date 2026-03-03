// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cutlass/cutlass.h>
#include <torch/extension.h>

#include "kernels/gemm/bmm_kernels.h"
#include "kernels/gemm/gemm_configs.h"
#include "kernels/gemm/gemm_kernels.h"
#include "kernels/gemm/group_gemm_kernels.h"

namespace py = pybind11;

namespace oasr {
namespace pybind {

inline void registerGemmBindings(py::module_& kernels) {
    using namespace kernels::gemm;

    py::module_ gemm_mod = kernels.def_submodule("gemm", "GEMM/BMM/GroupGEMM kernels (CUTLASS)");

    // --- Enums ---
    py::enum_<MatrixLayout>(gemm_mod, "MatrixLayout")
        .value("RowMajor", MatrixLayout::RowMajor)
        .value("ColumnMajor", MatrixLayout::ColumnMajor);

    py::enum_<EpilogueFusion>(gemm_mod, "EpilogueFusion")
        .value("NONE", EpilogueFusion::NONE)
        .value("BIAS", EpilogueFusion::BIAS)
        .value("BIAS_RELU", EpilogueFusion::BIAS_RELU)
        .value("BIAS_GELU", EpilogueFusion::BIAS_GELU)
        .value("BIAS_SWISH", EpilogueFusion::BIAS_SWISH)
        .value("GATED", EpilogueFusion::GATED);

    py::enum_<GemmStatus>(gemm_mod, "GemmStatus")
        .value("SUCCESS", GemmStatus::SUCCESS)
        .value("INVALID_ARGUMENT", GemmStatus::INVALID_ARGUMENT)
        .value("WORKSPACE_TOO_SMALL", GemmStatus::WORKSPACE_TOO_SMALL)
        .value("NOT_SUPPORTED", GemmStatus::NOT_SUPPORTED)
        .value("INTERNAL_ERROR", GemmStatus::INTERNAL_ERROR)
        .value("CUDA_ERROR", GemmStatus::CUDA_ERROR)
        .value("CUTLASS_ERROR", GemmStatus::CUTLASS_ERROR);

    py::enum_<SplitKStyle>(gemm_mod, "SplitKStyle")
        .value("NO_SPLIT_K", SplitKStyle::NO_SPLIT_K)
        .value("SPLIT_K_SERIAL", SplitKStyle::SPLIT_K_SERIAL)
        .value("STREAM_K", SplitKStyle::STREAM_K);

    py::enum_<MainloopScheduleType>(gemm_mod, "MainloopScheduleType")
        .value("AUTO", MainloopScheduleType::AUTO)
        .value("PINGPONG", MainloopScheduleType::PINGPONG)
        .value("COOPERATIVE", MainloopScheduleType::COOPERATIVE)
        .value("WARPSPECIALIZED", MainloopScheduleType::WARPSPECIALIZED);

    py::enum_<ClusterShape>(gemm_mod, "ClusterShape")
        .value("ClusterShape_1x1x1", ClusterShape::ClusterShape_1x1x1)
        .value("ClusterShape_2x1x1", ClusterShape::ClusterShape_2x1x1)
        .value("ClusterShape_1x2x1", ClusterShape::ClusterShape_1x2x1)
        .value("ClusterShape_2x2x1", ClusterShape::ClusterShape_2x2x1)
        .value("ClusterShape_1x4x1", ClusterShape::ClusterShape_1x4x1)
        .value("ClusterShape_4x1x1", ClusterShape::ClusterShape_4x1x1);

    py::enum_<TransposeOp>(gemm_mod, "TransposeOp")
        .value("NoTranspose", TransposeOp::NoTranspose)
        .value("Transpose", TransposeOp::Transpose)
        .value("ConjugateTranspose", TransposeOp::ConjugateTranspose);

    // --- Config types ---
    py::class_<GemmConfig>(gemm_mod, "GemmConfig")
        .def(py::init<>())
        .def_readwrite("split_k_style", &GemmConfig::split_k_style)
        .def_readwrite("split_k_factor", &GemmConfig::split_k_factor)
        .def_readwrite("stages", &GemmConfig::stages)
        .def_readwrite("mainloop_schedule", &GemmConfig::mainloop_schedule)
        .def_readwrite("cluster_shape", &GemmConfig::cluster_shape)
        .def_readwrite("sm_version", &GemmConfig::sm_version)
        .def_readwrite("is_tma_warp_specialized", &GemmConfig::is_tma_warp_specialized)
        .def("to_string", &GemmConfig::toString)
        .def("__repr__", &GemmConfig::toString);

    // --- GEMM: D = alpha * A @ B + beta * C ---
    gemm_mod.def(
        "invoke_gemm",
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
        "invoke_gemm_activation",
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
        "invoke_bmm",
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
        "invoke_group_gemm",
        [](const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& offset,
           py::object stream) -> torch::Tensor {
            cudaStream_t s = stream.is_none()
                                 ? nullptr
                                 : reinterpret_cast<cudaStream_t>(stream.cast<intptr_t>());
            return invokeGroupGemm(a, b, offset, s);
        },
        py::arg("a"), py::arg("b"), py::arg("offset"), py::arg("stream") = py::none(),
        "Execute grouped GEMM. Returns output tensor.");

    // --- Utility functions ---
    gemm_mod.def("get_sm_version", &getSMVersion, py::arg("device_id") = -1,
                 "Get SM version of the current or specified device");
    gemm_mod.def("supports_tma", &supportsTMA, py::arg("sm_version"),
                 "Check if SM version supports TMA");
    gemm_mod.def("supports_warp_specialization", &supportsWarpSpecialization, py::arg("sm_version"),
                 "Check if SM version supports warp specialization");
    gemm_mod.def("get_gemm_status_string", &getGemmStatusString, py::arg("status"),
                 "Convert GemmStatus to string");

    // --- Auto-tuning ---
    gemm_mod.def(
        "auto_tune_gemm",
        [](int M, int N, int K, DataType dtype, int num_warmup, int num_iter) {
            return autoTuneGemm(M, N, K, dtype, num_warmup, num_iter, nullptr);
        },
        py::arg("M"), py::arg("N"), py::arg("K"), py::arg("dtype"), py::arg("num_warmup") = 5,
        py::arg("num_iter") = 10, "Auto-tune and return best GEMM configuration");

    gemm_mod.def(
        "auto_tune_bmm",
        [](int batch, int M, int N, int K, DataType dtype, int num_warmup, int num_iter) {
            return autoTuneBmm(batch, M, N, K, dtype, num_warmup, num_iter, nullptr);
        },
        py::arg("batch"), py::arg("M"), py::arg("N"), py::arg("K"), py::arg("dtype"),
        py::arg("num_warmup") = 5, py::arg("num_iter") = 10,
        "Auto-tune and return best batched GEMM configuration");

    // --- Default configs ---
    gemm_mod.def("get_default_sm80_configs", &getDefaultSM80Configs,
                 "Get default GEMM configurations for SM80");
    gemm_mod.def("get_default_sm90_configs", &getDefaultSM90Configs,
                 "Get default GEMM configurations for SM90");
}

}  // namespace pybind
}  // namespace oasr
