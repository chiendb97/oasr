// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/extension.h>
#include <cutlass/half.h>
#include <cutlass/bfloat16.h>
#include "kernels/gemm/gemm_configs.h"
#include "kernels/gemm/gemm_kernels.h"
#include "kernels/gemm/bmm_kernels.h"
#include "kernels/gemm/group_gemm_kernels.h"

namespace py = pybind11;

namespace oasr {
namespace pybind {

inline void registerGemmBindings(py::module_& kernels) {
    using namespace kernels::gemm;

    py::module_ gemm_mod =
        kernels.def_submodule("gemm", "GEMM/BMM/GroupGEMM kernels (CUTLASS)");

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

    py::class_<GemmProblemDesc>(gemm_mod, "GemmProblemDesc")
        .def(py::init<>())
        .def(py::init<int, int, int>(), py::arg("m"), py::arg("n"), py::arg("k"))
        .def_readwrite("M", &GemmProblemDesc::M)
        .def_readwrite("N", &GemmProblemDesc::N)
        .def_readwrite("K", &GemmProblemDesc::K);

    // --- GEMM: D = alpha * A @ B + beta * C ---
    gemm_mod.def("invoke_gemm",
        [](const torch::Tensor& a, const torch::Tensor& b,
           std::optional<const torch::Tensor>& c,
           py::object stream) -> torch::Tensor {
            cudaStream_t s = stream.is_none()
                ? nullptr
                : reinterpret_cast<cudaStream_t>(stream.cast<intptr_t>());

            torch::Tensor c_tensor = c.has_value() ? c.value() : torch::Tensor();
            auto d = invokeGemm(a, b, c_tensor, s);
            return d;
        },
        py::arg("a"), py::arg("b"), py::arg("c") = py::none(),
        py::arg("stream") = py::none(),
        "Execute GEMM: D = A @ B + C (C is optional). Returns (output, status).");

    gemm_mod.def("invoke_gemm_activation",
        [](const torch::Tensor& a, const torch::Tensor& b,
           std::optional<const torch::Tensor>& c,
           ActivationType activation, py::object stream) -> torch::Tensor {
            cudaStream_t s = stream.is_none()
                ? nullptr
                : reinterpret_cast<cudaStream_t>(stream.cast<intptr_t>());
            torch::Tensor c_tensor = c.has_value() ? c.value() : torch::Tensor();
            auto d = invokeGemmActivation(a, b, c_tensor, activation, s);
            return d;
        },
        py::arg("a"), py::arg("b"), py::arg("c") = py::none(), py::arg("activation"), py::arg("stream") = py::none(),
        "Execute GEMM with activation: D = activation(A @ B + C). Returns output tensor.");

    // --- Batched GEMM (strided) ---
    gemm_mod.def("invoke_bmm",
        [](const torch::Tensor& a, const torch::Tensor& b,
           int batch_size, int M, int N, int K,
           int64_t lda, int64_t ldb, int64_t ldd,
           int64_t stride_a, int64_t stride_b, int64_t stride_d,
           float alpha, float beta,
           TransposeOp trans_a, TransposeOp trans_b,
           DataType dtype, py::object stream) -> py::tuple {
            auto d = torch::empty({batch_size, M, N}, a.options());
            cudaStream_t s = stream.is_none()
                ? nullptr
                : reinterpret_cast<cudaStream_t>(stream.cast<intptr_t>());
            auto status = invokeBmm(
                a.data_ptr(), b.data_ptr(), d.data_ptr(),
                batch_size, M, N, K, lda, ldb, ldd,
                stride_a, stride_b, stride_d, alpha, beta,
                trans_a, trans_b, dtype, s);
            return py::make_tuple(d, status);
        },
        py::arg("a"), py::arg("b"),
        py::arg("batch_size"), py::arg("M"), py::arg("N"), py::arg("K"),
        py::arg("lda"), py::arg("ldb"), py::arg("ldd"),
        py::arg("stride_a"), py::arg("stride_b"), py::arg("stride_d"),
        py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f,
        py::arg("trans_a") = TransposeOp::NoTranspose,
        py::arg("trans_b") = TransposeOp::NoTranspose,
        py::arg("dtype"), py::arg("stream") = py::none(),
        "Execute strided batched GEMM. Returns (output, status).");

    // --- Grouped GEMM (pointer arrays, kept low-level) ---
    gemm_mod.def("invoke_group_gemm",
        [](intptr_t problems, int num_problems,
           intptr_t a_array, intptr_t b_array, intptr_t d_array,
           intptr_t lda_array, intptr_t ldb_array, intptr_t ldd_array,
           DataType dtype, intptr_t workspace_float, size_t workspace_float_size,
           py::object stream) {
            cudaStream_t s = stream.is_none()
                ? nullptr
                : reinterpret_cast<cudaStream_t>(stream.cast<intptr_t>());
            return invokeGroupGemm(
                reinterpret_cast<const GemmProblemDesc*>(problems),
                num_problems,
                reinterpret_cast<const void* const*>(a_array),
                reinterpret_cast<const void* const*>(b_array),
                reinterpret_cast<void* const*>(d_array),
                reinterpret_cast<const int64_t*>(lda_array),
                reinterpret_cast<const int64_t*>(ldb_array),
                reinterpret_cast<const int64_t*>(ldd_array),
                dtype,
                reinterpret_cast<void*>(workspace_float),
                workspace_float_size, s);
        },
        py::arg("problems"), py::arg("num_problems"),
        py::arg("a_array"), py::arg("b_array"), py::arg("d_array"),
        py::arg("lda_array"), py::arg("ldb_array"), py::arg("ldd_array"),
        py::arg("dtype"), py::arg("workspace_float"), py::arg("workspace_float_size"),
        py::arg("stream") = py::none(),
        "Execute grouped GEMM (variable-sized problems)");

    // --- Workspace queries ---
    gemm_mod.def("query_group_gemm_workspace_size",
        [](int num_problems, DataType dtype, const GemmConfig& config) {
            size_t float_ws = 0, int_ws = 0;
            queryGroupGemmWorkspaceSize(num_problems, nullptr, dtype,
                                        float_ws, int_ws, config);
            return py::make_tuple(float_ws, int_ws);
        },
        py::arg("num_problems"), py::arg("dtype"),
        py::arg("config") = GemmConfig(),
        "Query workspace sizes for grouped GEMM (returns float_ws, int_ws)");

    gemm_mod.def("query_bmm_workspace_size", &queryBmmWorkspaceSize,
        py::arg("batch_size"), py::arg("M"), py::arg("N"), py::arg("K"),
        py::arg("dtype"), py::arg("config") = GemmConfig(),
        "Query workspace size for batched GEMM");

    // --- Utility functions ---
    gemm_mod.def("get_sm_version", &getSMVersion, py::arg("device_id") = -1,
        "Get SM version of the current or specified device");
    gemm_mod.def("supports_tma", &supportsTMA, py::arg("sm_version"),
        "Check if SM version supports TMA");
    gemm_mod.def("supports_warp_specialization", &supportsWarpSpecialization,
        py::arg("sm_version"),
        "Check if SM version supports warp specialization");
    gemm_mod.def("get_gemm_status_string", &getGemmStatusString, py::arg("status"),
        "Convert GemmStatus to string");

    // --- Auto-tuning ---
    gemm_mod.def("auto_tune_gemm",
        [](int M, int N, int K, DataType dtype, int num_warmup, int num_iter) {
            return autoTuneGemm(M, N, K, dtype, num_warmup, num_iter, nullptr);
        },
        py::arg("M"), py::arg("N"), py::arg("K"), py::arg("dtype"),
        py::arg("num_warmup") = 5, py::arg("num_iter") = 10,
        "Auto-tune and return best GEMM configuration");

    gemm_mod.def("auto_tune_bmm",
        [](int batch, int M, int N, int K, DataType dtype,
           int num_warmup, int num_iter) {
            return autoTuneBmm(batch, M, N, K, dtype, num_warmup, num_iter, nullptr);
        },
        py::arg("batch"), py::arg("M"), py::arg("N"), py::arg("K"),
        py::arg("dtype"), py::arg("num_warmup") = 5, py::arg("num_iter") = 10,
        "Auto-tune and return best batched GEMM configuration");

    // --- Default configs ---
    gemm_mod.def("get_default_sm80_configs", &getDefaultSM80Configs,
        "Get default GEMM configurations for SM80");
    gemm_mod.def("get_default_sm90_configs", &getDefaultSM90Configs,
        "Get default GEMM configurations for SM90");
}

} // namespace pybind
} // namespace oasr
