// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include <cuda_runtime.h>
#include "common/types.h"
#include "kernels/normalization/norm_kernels.h"
#include "kernels/convolution/conv_kernels.h"
#include "kernels/convolution/conv_params.h"

namespace py = pybind11;

// Helper to convert intptr_t (Python int from tensor.data_ptr()) to void*
inline void* to_ptr(intptr_t ptr) {
    return reinterpret_cast<void*>(ptr);
}

inline const void* to_const_ptr(intptr_t ptr) {
    return reinterpret_cast<const void*>(ptr);
}

PYBIND11_MODULE(_C, m) {
    m.doc() = "OASR C++ extension module";
    
    // Version info
    m.attr("__version__") = "0.1.0";
    
    // Initialize CUDA
    m.def("init_cuda", []() {
        // Initialize CUDA runtime
        int device_count;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) {
            throw std::runtime_error("No CUDA devices found");
        }
        return device_count;
    }, "Initialize CUDA and return device count");
    
    m.def("get_device_count", []() {
        int count;
        cudaGetDeviceCount(&count);
        return count;
    }, "Get number of CUDA devices");
    
    m.def("set_device", [](int device_id) {
        cudaSetDevice(device_id);
    }, py::arg("device_id"), "Set current CUDA device");
    
    m.def("get_device", []() {
        int device;
        cudaGetDevice(&device);
        return device;
    }, "Get current CUDA device");
    
    m.def("synchronize", []() {
        cudaDeviceSynchronize();
    }, "Synchronize current CUDA device");
    
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
        .value("GATED_GELU", oasr::ActivationType::GATED_GELU)
        .value("GATED_SWISH", oasr::ActivationType::GATED_SWISH)
        .export_values();
    
    // =========================================================================
    // Normalization Kernels
    // =========================================================================
    py::module_ kernels = m.def_submodule("kernels", "Low-level CUDA kernels");
    py::module_ norm = kernels.def_submodule("normalization", "Normalization kernels");
    
    // NormParams
    py::class_<oasr::kernels::NormParams>(norm, "NormParams")
        .def(py::init<>())
        .def_readwrite("batch_size", &oasr::kernels::NormParams::batch_size)
        .def_readwrite("seq_len", &oasr::kernels::NormParams::seq_len)
        .def_readwrite("hidden_size", &oasr::kernels::NormParams::hidden_size)
        .def_readwrite("eps", &oasr::kernels::NormParams::eps)
        .def_readwrite("norm_type", &oasr::kernels::NormParams::norm_type)
        .def_readwrite("dtype", &oasr::kernels::NormParams::dtype);
    
    // Normalization kernel functions with Python-friendly interface
    // Use intptr_t for pointers (compatible with tensor.data_ptr())
    norm.def("layer_norm", 
             [](intptr_t input, intptr_t output,
                intptr_t gamma, intptr_t beta,
                int batch_size, int seq_len, int hidden_size,
                float eps, oasr::DataType dtype) {
                 oasr::kernels::invokeLayerNorm(
                     to_const_ptr(input), to_ptr(output),
                     to_const_ptr(gamma), to_const_ptr(beta),
                     batch_size, seq_len, hidden_size,
                     eps, dtype, nullptr);
             },
             py::arg("input"), py::arg("output"),
             py::arg("gamma"), py::arg("beta"),
             py::arg("batch_size"), py::arg("seq_len"), py::arg("hidden_size"),
             py::arg("eps") = 1e-5f, py::arg("dtype") = oasr::DataType::FP16,
             "Layer normalization kernel");
    
    norm.def("rms_norm",
             [](intptr_t input, intptr_t output, intptr_t gamma,
                int batch_size, int seq_len, int hidden_size,
                float eps, oasr::DataType dtype) {
                 oasr::kernels::invokeRMSNorm(
                     to_const_ptr(input), to_ptr(output), to_const_ptr(gamma),
                     batch_size, seq_len, hidden_size,
                     eps, dtype, nullptr);
             },
             py::arg("input"), py::arg("output"), py::arg("gamma"),
             py::arg("batch_size"), py::arg("seq_len"), py::arg("hidden_size"),
             py::arg("eps") = 1e-5f, py::arg("dtype") = oasr::DataType::FP16,
             "RMS normalization kernel");
    
    norm.def("batch_norm_1d",
             [](intptr_t input, intptr_t output,
                intptr_t gamma, intptr_t beta,
                intptr_t running_mean, intptr_t running_var,
                int batch_size, int seq_len, int channels,
                float eps, oasr::DataType dtype) {
                 oasr::kernels::invokeBatchNorm1D(
                     to_const_ptr(input), to_ptr(output),
                     to_const_ptr(gamma), to_const_ptr(beta),
                     to_const_ptr(running_mean), to_const_ptr(running_var),
                     batch_size, seq_len, channels,
                     eps, dtype, nullptr);
             },
             py::arg("input"), py::arg("output"),
             py::arg("gamma"), py::arg("beta"),
             py::arg("running_mean"), py::arg("running_var"),
             py::arg("batch_size"), py::arg("seq_len"), py::arg("channels"),
             py::arg("eps") = 1e-5f, py::arg("dtype") = oasr::DataType::FP16,
             "1D batch normalization kernel (inference mode)");
    
    norm.def("group_norm",
             [](intptr_t input, intptr_t output,
                intptr_t gamma, intptr_t beta,
                int batch_size, int seq_len, int channels, int num_groups,
                float eps, oasr::DataType dtype) {
                 oasr::kernels::invokeGroupNorm(
                     to_const_ptr(input), to_ptr(output),
                     to_const_ptr(gamma), to_const_ptr(beta),
                     batch_size, seq_len, channels, num_groups,
                     eps, dtype, nullptr);
             },
             py::arg("input"), py::arg("output"),
             py::arg("gamma"), py::arg("beta"),
             py::arg("batch_size"), py::arg("seq_len"),
             py::arg("channels"), py::arg("num_groups"),
             py::arg("eps") = 1e-5f, py::arg("dtype") = oasr::DataType::FP16,
             "Group normalization kernel");
    
    norm.def("add_layer_norm",
             [](intptr_t input, intptr_t residual, intptr_t output,
                intptr_t gamma, intptr_t beta,
                int batch_size, int seq_len, int hidden_size,
                float eps, oasr::DataType dtype) {
                 oasr::kernels::invokeAddLayerNorm(
                     to_const_ptr(input), to_const_ptr(residual), to_ptr(output),
                     to_const_ptr(gamma), to_const_ptr(beta),
                     batch_size, seq_len, hidden_size,
                     eps, dtype, nullptr);
             },
             py::arg("input"), py::arg("residual"), py::arg("output"),
             py::arg("gamma"), py::arg("beta"),
             py::arg("batch_size"), py::arg("seq_len"), py::arg("hidden_size"),
             py::arg("eps") = 1e-5f, py::arg("dtype") = oasr::DataType::FP16,
             "Fused add + layer norm kernel");
    
    // =========================================================================
    // Convolution Kernels
    // =========================================================================
    py::module_ conv = kernels.def_submodule("convolution", "Convolution kernels");
    
    // Depthwise Conv1D
    conv.def("depthwise_conv1d",
             [](intptr_t input, intptr_t weight, intptr_t bias,
                intptr_t output, int batch_size, int seq_len, int channels,
                int kernel_size, int padding, bool is_causal, oasr::DataType dtype) {
                 oasr::kernels::invokeDepthwiseConv1D(
                     to_const_ptr(input), to_const_ptr(weight),
                     bias != 0 ? to_const_ptr(bias) : nullptr,
                     to_ptr(output), batch_size, seq_len, channels,
                     kernel_size, padding, is_causal, dtype, nullptr);
             },
             py::arg("input"), py::arg("weight"), py::arg("bias"),
             py::arg("output"), py::arg("batch_size"), py::arg("seq_len"),
             py::arg("channels"), py::arg("kernel_size"),
             py::arg("padding") = 0, py::arg("is_causal") = false,
             py::arg("dtype") = oasr::DataType::FP16,
             "Depthwise 1D convolution kernel");
    
    // Pointwise Conv1D
    conv.def("pointwise_conv1d",
             [](intptr_t input, intptr_t weight, intptr_t bias,
                intptr_t output, int batch_size, int seq_len,
                int in_channels, int out_channels,
                oasr::ActivationType activation, bool fuse_activation,
                oasr::DataType dtype) {
                 oasr::kernels::invokePointwiseConv1D(
                     to_const_ptr(input), to_const_ptr(weight),
                     bias != 0 ? to_const_ptr(bias) : nullptr,
                     to_ptr(output), batch_size, seq_len,
                     in_channels, out_channels,
                     activation, fuse_activation, dtype, nullptr);
             },
             py::arg("input"), py::arg("weight"), py::arg("bias"),
             py::arg("output"), py::arg("batch_size"), py::arg("seq_len"),
             py::arg("in_channels"), py::arg("out_channels"),
             py::arg("activation") = oasr::ActivationType::SWISH,
             py::arg("fuse_activation") = false,
             py::arg("dtype") = oasr::DataType::FP16,
             "Pointwise (1x1) convolution kernel");
    
    // GLU activation
    conv.def("glu",
             [](intptr_t input, intptr_t output,
                int batch_size, int seq_len, int channels, oasr::DataType dtype) {
                 oasr::kernels::invokeGLU(
                     to_const_ptr(input), to_ptr(output),
                     batch_size, seq_len, channels, dtype, nullptr);
             },
             py::arg("input"), py::arg("output"),
             py::arg("batch_size"), py::arg("seq_len"), py::arg("channels"),
             py::arg("dtype") = oasr::DataType::FP16,
             "GLU (Gated Linear Unit) activation kernel");
    
    // Swish activation
    conv.def("swish",
             [](intptr_t input, intptr_t output,
                int batch_size, int seq_len, int channels, oasr::DataType dtype) {
                 oasr::kernels::invokeSwish(
                     to_const_ptr(input), to_ptr(output),
                     batch_size, seq_len, channels, dtype, nullptr);
             },
             py::arg("input"), py::arg("output"),
             py::arg("batch_size"), py::arg("seq_len"), py::arg("channels"),
             py::arg("dtype") = oasr::DataType::FP16,
             "Swish activation kernel");
    
    // Fused BatchNorm + Swish
    conv.def("batch_norm_swish",
             [](intptr_t input, intptr_t output,
                intptr_t gamma, intptr_t beta,
                intptr_t running_mean, intptr_t running_var,
                int batch_size, int seq_len, int channels,
                float eps, oasr::DataType dtype) {
                 oasr::kernels::invokeBatchNormSwish(
                     to_const_ptr(input), to_ptr(output),
                     to_const_ptr(gamma), to_const_ptr(beta),
                     to_const_ptr(running_mean), to_const_ptr(running_var),
                     batch_size, seq_len, channels, eps, dtype, nullptr);
             },
             py::arg("input"), py::arg("output"),
             py::arg("gamma"), py::arg("beta"),
             py::arg("running_mean"), py::arg("running_var"),
             py::arg("batch_size"), py::arg("seq_len"), py::arg("channels"),
             py::arg("eps") = 1e-5f, py::arg("dtype") = oasr::DataType::FP16,
             "Fused BatchNorm + Swish kernel");
    
    // General Conv1D (with params)
    conv.def("conv1d",
             [](intptr_t input, intptr_t weight, intptr_t bias, intptr_t output,
                int batch_size, int seq_len, int in_channels, int out_channels,
                int kernel_size, int stride, int padding, int dilation, int groups,
                oasr::ConvType conv_type, oasr::DataType dtype, bool channels_last,
                bool is_causal, oasr::ActivationType activation, bool fuse_activation) {
                 oasr::kernels::Conv1DParams params;
                 params.input = to_const_ptr(input);
                 params.output = to_ptr(output);
                 params.weight = to_const_ptr(weight);
                 params.bias = bias != 0 ? to_const_ptr(bias) : nullptr;
                 params.batch_size = batch_size;
                 params.seq_len = seq_len;
                 params.in_channels = in_channels;
                 params.out_channels = out_channels;
                 params.kernel_size = kernel_size;
                 params.stride = stride;
                 params.padding = padding;
                 params.dilation = dilation;
                 params.groups = groups;
                 params.conv_type = conv_type;
                 params.dtype = dtype;
                 params.channels_last = channels_last;
                 params.is_causal = is_causal;
                 params.activation = activation;
                 params.fuse_activation = fuse_activation;
                 params.stream = nullptr;
                 
                 oasr::kernels::invokeConv1D(params);
             },
             py::arg("input"), py::arg("weight"), py::arg("bias"), py::arg("output"),
             py::arg("batch_size"), py::arg("seq_len"),
             py::arg("in_channels"), py::arg("out_channels"),
             py::arg("kernel_size"), py::arg("stride") = 1,
             py::arg("padding") = 0, py::arg("dilation") = 1, py::arg("groups") = 1,
             py::arg("conv_type") = oasr::ConvType::STANDARD,
             py::arg("dtype") = oasr::DataType::FP16,
             py::arg("channels_last") = true, py::arg("is_causal") = false,
             py::arg("activation") = oasr::ActivationType::SWISH,
             py::arg("fuse_activation") = false,
             "General 1D convolution kernel");
}
