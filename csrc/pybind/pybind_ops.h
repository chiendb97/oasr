// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ops/cutlass/gemm_ops.h"

namespace py = pybind11;

namespace oasr {
namespace pybind {

/**
 * @brief Register operations-related Python bindings
 */
inline void registerOpsBindings(py::module_& m) {
    // Create ops submodule
    py::module_ ops = m.def_submodule("ops", "Optimized operations (CUTLASS, cuBLAS)");
    
    // GemmConfig
    py::class_<ops::GemmConfig>(ops, "GemmConfig")
        .def(py::init<>())
        .def_readwrite("M", &ops::GemmConfig::M)
        .def_readwrite("N", &ops::GemmConfig::N)
        .def_readwrite("K", &ops::GemmConfig::K)
        .def_readwrite("alpha", &ops::GemmConfig::alpha)
        .def_readwrite("beta", &ops::GemmConfig::beta)
        .def_readwrite("dtype_a", &ops::GemmConfig::dtype_a)
        .def_readwrite("dtype_b", &ops::GemmConfig::dtype_b)
        .def_readwrite("dtype_c", &ops::GemmConfig::dtype_c)
        .def_readwrite("dtype_compute", &ops::GemmConfig::dtype_compute)
        .def_readwrite("trans_a", &ops::GemmConfig::trans_a)
        .def_readwrite("trans_b", &ops::GemmConfig::trans_b)
        .def_readwrite("lda", &ops::GemmConfig::lda)
        .def_readwrite("ldb", &ops::GemmConfig::ldb)
        .def_readwrite("ldc", &ops::GemmConfig::ldc)
        .def_readwrite("batch_count", &ops::GemmConfig::batch_count)
        .def_readwrite("stride_a", &ops::GemmConfig::stride_a)
        .def_readwrite("stride_b", &ops::GemmConfig::stride_b)
        .def_readwrite("stride_c", &ops::GemmConfig::stride_c);
    
    // GemmConfig::Epilogue enum
    py::enum_<ops::GemmConfig::Epilogue>(ops, "GemmEpilogue")
        .value("NONE", ops::GemmConfig::Epilogue::NONE)
        .value("BIAS", ops::GemmConfig::Epilogue::BIAS)
        .value("RELU", ops::GemmConfig::Epilogue::RELU)
        .value("GELU", ops::GemmConfig::Epilogue::GELU)
        .value("SWISH", ops::GemmConfig::Epilogue::SWISH)
        .value("BIAS_RELU", ops::GemmConfig::Epilogue::BIAS_RELU)
        .value("BIAS_GELU", ops::GemmConfig::Epilogue::BIAS_GELU)
        .value("BIAS_SWISH", ops::GemmConfig::Epilogue::BIAS_SWISH)
        .export_values();
    
    // GemmOp base class
    py::class_<ops::GemmOp, std::shared_ptr<ops::GemmOp>>(ops, "GemmOp")
        .def("get_workspace_size", &ops::GemmOp::getWorkspaceSize, py::arg("config"))
        .def("set_workspace", &ops::GemmOp::setWorkspace, py::arg("workspace"), py::arg("size"));
    
    // CutlassGemm
    py::class_<ops::CutlassGemm, ops::GemmOp, std::shared_ptr<ops::CutlassGemm>>(ops, "CutlassGemm")
        .def(py::init<>())
        .def("auto_tune", &ops::CutlassGemm::autoTune, py::arg("config"), py::arg("stream") = nullptr);
    
    // CublasGemm
    py::class_<ops::CublasGemm, ops::GemmOp, std::shared_ptr<ops::CublasGemm>>(ops, "CublasGemm")
        .def(py::init<>());
    
    // Free functions
    ops.def("gemm", &ops::gemm,
            py::arg("a"), py::arg("b"), py::arg("c"),
            py::arg("config"), py::arg("stream") = nullptr,
            "GEMM with automatic backend selection");
    
    ops.def("batched_gemm", &ops::batchedGemm,
            py::arg("a"), py::arg("b"), py::arg("c"),
            py::arg("config"), py::arg("stream") = nullptr,
            "Batched GEMM");
    
    ops.def("create_optimal_gemm", &ops::createOptimalGemm,
            py::arg("config"),
            "Create optimal GEMM implementation for given configuration");
}

} // namespace pybind
} // namespace oasr
