// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>

#include <torch/extension.h>

#include <c10/cuda/CUDAFunctions.h>

#include "kernels/common/types.h"
#include "pybind_activation.h"
#include "pybind_conv.h"
#include "pybind_decoder.h"
#include "pybind_gemm.h"
#include "pybind_norm.h"

namespace py = pybind11;

PYBIND11_MODULE(_C, m) {
    m.doc() = "OASR C++ extension module";
    m.attr("__version__") = "0.1.0";

    // =========================================================================
    // CUDA utilities
    // =========================================================================
    m.def(
        "init_cuda",
        []() {
            int device_count = static_cast<int>(c10::cuda::device_count());
            if (device_count == 0) {
                throw std::runtime_error("No CUDA devices found");
            }
            return device_count;
        },
        "Initialize CUDA and return device count");

    m.def(
        "get_device_count",
        []() {
            return static_cast<int>(c10::cuda::device_count());
        },
        "Get number of CUDA devices");

    m.def("set_device",
          [](int device_id) { c10::cuda::set_device(device_id); },
          py::arg("device_id"), "Set current CUDA device");

    m.def(
        "get_device",
        []() {
            return static_cast<int>(c10::cuda::current_device());
        },
        "Get current CUDA device");

    m.def("synchronize", []() { c10::cuda::device_synchronize(); }, "Synchronize current CUDA device");

    // =========================================================================
    // Enums
    // =========================================================================
    py::enum_<oasr::DataType>(m, "DataType")
        .value("FP32", oasr::DataType::FP32)
        .value("FP16", oasr::DataType::FP16)
        .value("BF16", oasr::DataType::BF16)
        .value("INT8", oasr::DataType::INT8)
        .value("INT32", oasr::DataType::INT32)
        .export_values();

    py::enum_<oasr::NormType>(m, "NormType")
        .value("LAYER_NORM", oasr::NormType::LAYER_NORM)
        .value("RMS_NORM", oasr::NormType::RMS_NORM)
        .value("BATCH_NORM", oasr::NormType::BATCH_NORM)
        .value("GROUP_NORM", oasr::NormType::GROUP_NORM)
        .export_values();

    py::enum_<oasr::ConvType>(m, "ConvType")
        .value("DEPTHWISE", oasr::ConvType::DEPTHWISE)
        .value("POINTWISE", oasr::ConvType::POINTWISE)
        .value("CAUSAL", oasr::ConvType::CAUSAL)
        .value("STANDARD", oasr::ConvType::STANDARD)
        .export_values();

    py::enum_<oasr::ActivationType>(m, "ActivationType")
        .value("RELU", oasr::ActivationType::RELU)
        .value("GELU", oasr::ActivationType::GELU)
        .value("SWISH", oasr::ActivationType::SWISH)
        .export_values();

    // =========================================================================
    // Kernel submodule registrations
    // =========================================================================
    py::module_ kernels = m.def_submodule("kernels", "Low-level CUDA kernels");

    oasr::pybind::registerNormBindings(kernels);
    oasr::pybind::registerConvBindings(kernels);
    oasr::pybind::registerGemmBindings(kernels);
    oasr::pybind::registerActivationBindings(kernels);

    // =========================================================================
    // Decoder bindings
    // =========================================================================
    oasr::pybind::registerDecoderBindings(m);
}
