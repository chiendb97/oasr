// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "common/tensor.h"
#include "common/types.h"
#include "common/cuda_utils.h"
#include "common/allocator.h"

namespace py = pybind11;

namespace oasr {
namespace pybind {

/**
 * @brief Register tensor-related Python bindings
 */
inline void registerTensorBindings(py::module_& m) {
    // DataType enum
    py::enum_<DataType>(m, "DataType")
        .value("FP32", DataType::FP32)
        .value("FP16", DataType::FP16)
        .value("BF16", DataType::BF16)
        .value("INT8", DataType::INT8)
        .value("INT32", DataType::INT32)
        .export_values();
    
    // MemoryType enum
    py::enum_<MemoryType>(m, "MemoryType")
        .value("CPU", MemoryType::CPU)
        .value("GPU", MemoryType::GPU)
        .value("PINNED", MemoryType::PINNED)
        .value("UNIFIED", MemoryType::UNIFIED)
        .export_values();
    
    // AttentionType enum
    py::enum_<AttentionType>(m, "AttentionType")
        .value("MULTI_HEAD", AttentionType::MULTI_HEAD)
        .value("RELATIVE_POSITION", AttentionType::RELATIVE_POSITION)
        .value("RELATIVE_SHIFT", AttentionType::RELATIVE_SHIFT)
        .value("ROTARY", AttentionType::ROTARY)
        .value("ALiBi", AttentionType::ALiBi)
        .export_values();
    
    // ActivationType enum
    py::enum_<ActivationType>(m, "ActivationType")
        .value("RELU", ActivationType::RELU)
        .value("GELU", ActivationType::GELU)
        .value("SWISH", ActivationType::SWISH)
        .value("GATED_GELU", ActivationType::GATED_GELU)
        .value("GATED_SWISH", ActivationType::GATED_SWISH)
        .export_values();
    
    // NormType enum
    py::enum_<NormType>(m, "NormType")
        .value("LAYER_NORM", NormType::LAYER_NORM)
        .value("RMS_NORM", NormType::RMS_NORM)
        .value("BATCH_NORM", NormType::BATCH_NORM)
        .value("GROUP_NORM", NormType::GROUP_NORM)
        .export_values();
    
    // ModelType enum
    py::enum_<ModelType>(m, "ModelType")
        .value("CONFORMER", ModelType::CONFORMER)
        .value("PARAFORMER", ModelType::PARAFORMER)
        .value("BRANCHFORMER", ModelType::BRANCHFORMER)
        .value("TRANSFORMER", ModelType::TRANSFORMER)
        .value("ZIPFORMER", ModelType::ZIPFORMER)
        .export_values();
    
    // Tensor class
    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        .def(py::init<const std::vector<int64_t>&, DataType, MemoryType, int>(),
             py::arg("shape"),
             py::arg("dtype") = DataType::FP32,
             py::arg("memory_type") = MemoryType::GPU,
             py::arg("device_id") = 0,
             "Create a tensor with given shape and data type")
        
        // Properties
        .def_property_readonly("shape", &Tensor::shape, "Get tensor shape")
        .def_property_readonly("ndim", &Tensor::ndim, "Get number of dimensions")
        .def_property_readonly("numel", &Tensor::numel, "Get number of elements")
        .def_property_readonly("nbytes", &Tensor::nbytes, "Get size in bytes")
        .def_property_readonly("dtype", &Tensor::dtype, "Get data type")
        .def_property_readonly("memory_type", &Tensor::memoryType, "Get memory type")
        .def_property_readonly("device_id", &Tensor::deviceId, "Get device ID")
        .def_property_readonly("is_contiguous", &Tensor::isContiguous, "Check if contiguous")
        
        // Shape operations
        .def("view", &Tensor::view, py::arg("shape"), "Create a view with new shape")
        .def("reshape", &Tensor::reshape, py::arg("shape"), "Reshape tensor")
        .def("squeeze", &Tensor::squeeze, py::arg("dim") = -1, "Remove dimension of size 1")
        .def("unsqueeze", &Tensor::unsqueeze, py::arg("dim"), "Add dimension of size 1")
        .def("transpose", &Tensor::transpose, py::arg("dim0"), py::arg("dim1"), "Transpose two dimensions")
        .def("permute", &Tensor::permute, py::arg("dims"), "Permute dimensions")
        
        // Memory operations
        .def("clone", &Tensor::clone, "Create a copy of the tensor")
        .def("contiguous", &Tensor::contiguous, "Return contiguous tensor")
        .def("to_gpu", &Tensor::toGPU, py::arg("device_id") = 0, "Move to GPU")
        .def("to_cpu", &Tensor::toCPU, "Move to CPU")
        
        // Type conversion
        .def("to_fp16", &Tensor::toFP16, "Convert to FP16")
        .def("to_fp32", &Tensor::toFP32, "Convert to FP32")
        .def("to_bf16", &Tensor::toBF16, "Convert to BF16")
        
        // Utility
        .def("fill", &Tensor::fill, py::arg("value"), "Fill with value")
        .def("zero", &Tensor::zero, "Fill with zeros")
        .def("__repr__", &Tensor::toString)
        
        // NumPy interop
        .def("to_numpy", [](const Tensor& t) {
            // Move to CPU if needed
            Tensor cpu_tensor = t.memoryType() == MemoryType::CPU ? t.clone() : t.toCPU();
            
            // Create numpy array based on dtype
            std::vector<ssize_t> shape(cpu_tensor.shape().begin(), cpu_tensor.shape().end());
            std::vector<ssize_t> strides;
            
            ssize_t elem_size = getDataTypeSize(cpu_tensor.dtype());
            ssize_t stride = elem_size;
            for (int i = shape.size() - 1; i >= 0; --i) {
                strides.insert(strides.begin(), stride);
                stride *= shape[i];
            }
            
            py::str dtype_str;
            switch (cpu_tensor.dtype()) {
                case DataType::FP32: dtype_str = py::str("float32"); break;
                case DataType::FP16: dtype_str = py::str("float16"); break;
                case DataType::BF16: dtype_str = py::str("bfloat16"); break;
                case DataType::INT8: dtype_str = py::str("int8"); break;
                case DataType::INT32: dtype_str = py::str("int32"); break;
                default: throw std::runtime_error("Unsupported dtype");
            }
            
            return py::array(py::dtype(dtype_str), shape, strides, cpu_tensor.data());
        }, "Convert to numpy array")
        
        // Static factory methods
        .def_static("empty", &Tensor::empty,
            py::arg("shape"),
            py::arg("dtype") = DataType::FP32,
            py::arg("memory_type") = MemoryType::GPU,
            py::arg("device_id") = 0,
            "Create empty tensor")
        .def_static("zeros", &Tensor::zeros,
            py::arg("shape"),
            py::arg("dtype") = DataType::FP32,
            py::arg("memory_type") = MemoryType::GPU,
            py::arg("device_id") = 0,
            "Create tensor filled with zeros")
        .def_static("ones", &Tensor::ones,
            py::arg("shape"),
            py::arg("dtype") = DataType::FP32,
            py::arg("memory_type") = MemoryType::GPU,
            py::arg("device_id") = 0,
            "Create tensor filled with ones")
        .def_static("from_numpy", [](py::array arr, int device_id) {
            // Get array info
            py::buffer_info info = arr.request();
            
            // Determine data type
            DataType dtype;
            if (info.format == py::format_descriptor<float>::format()) {
                dtype = DataType::FP32;
            } else if (info.format == "e") {  // float16
                dtype = DataType::FP16;
            } else if (info.format == py::format_descriptor<int32_t>::format()) {
                dtype = DataType::INT32;
            } else if (info.format == py::format_descriptor<int8_t>::format()) {
                dtype = DataType::INT8;
            } else {
                throw std::runtime_error("Unsupported numpy dtype");
            }
            
            // Create tensor
            std::vector<int64_t> shape(info.shape.begin(), info.shape.end());
            return Tensor::fromCPU(info.ptr, shape, dtype, device_id);
        }, py::arg("arr"), py::arg("device_id") = 0, "Create tensor from numpy array");
    
    // CudaStream class
    py::class_<CudaStream, std::shared_ptr<CudaStream>>(m, "CudaStream")
        .def(py::init<int, unsigned int>(),
             py::arg("device_id") = 0,
             py::arg("flags") = 0)
        .def("synchronize", &CudaStream::synchronize)
        .def("query", &CudaStream::query)
        .def_property_readonly("device_id", &CudaStream::deviceId);
    
    // CudaEvent class
    py::class_<CudaEvent, std::shared_ptr<CudaEvent>>(m, "CudaEvent")
        .def(py::init<unsigned int>(), py::arg("flags") = 0)
        .def("record", &CudaEvent::record, py::arg("stream") = nullptr)
        .def("synchronize", &CudaEvent::synchronize)
        .def("query", &CudaEvent::query)
        .def("elapsed_time", &CudaEvent::elapsedTime, py::arg("start"));
}

} // namespace pybind
} // namespace oasr
