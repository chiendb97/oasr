// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "types.h"
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include <numeric>
#include <cuda_runtime.h>

namespace oasr {

// Forward declarations
class Allocator;

/**
 * @brief Tensor class for representing multi-dimensional arrays
 * 
 * This is the fundamental data structure used throughout OASR.
 * It supports both CPU and GPU memory, with optional ownership semantics.
 */
class Tensor {
public:
    // Constructors
    Tensor() = default;
    
    /**
     * @brief Construct a tensor with given shape and data type
     * @param shape The dimensions of the tensor
     * @param dtype Data type (FP32, FP16, BF16, etc.)
     * @param memory_type Where to allocate (CPU, GPU, etc.)
     * @param device_id GPU device ID (ignored for CPU)
     */
    Tensor(const std::vector<int64_t>& shape,
           DataType dtype = DataType::FP32,
           MemoryType memory_type = MemoryType::GPU,
           int device_id = 0);
    
    /**
     * @brief Construct a tensor from external data (no ownership)
     * @param data Pointer to external data
     * @param shape The dimensions of the tensor
     * @param dtype Data type
     * @param memory_type Memory location of the data
     * @param device_id GPU device ID
     */
    Tensor(void* data,
           const std::vector<int64_t>& shape,
           DataType dtype,
           MemoryType memory_type,
           int device_id = 0);
    
    // Destructor
    ~Tensor();
    
    // Move semantics
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    
    // Disable copy (use clone() for explicit copying)
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    
    // Accessors
    void* data() { return data_; }
    const void* data() const { return data_; }
    
    template <typename T>
    T* data() { return static_cast<T*>(data_); }
    
    template <typename T>
    const T* data() const { return static_cast<const T*>(data_); }
    
    const std::vector<int64_t>& shape() const { return shape_; }
    int64_t shape(int dim) const;
    int64_t ndim() const { return static_cast<int64_t>(shape_.size()); }
    int64_t numel() const { return numel_; }
    size_t nbytes() const { return numel_ * getDataTypeSize(dtype_); }
    
    DataType dtype() const { return dtype_; }
    MemoryType memoryType() const { return memory_type_; }
    int deviceId() const { return device_id_; }
    bool ownsData() const { return owns_data_; }
    
    // Shape operations
    Tensor view(const std::vector<int64_t>& new_shape) const;
    Tensor reshape(const std::vector<int64_t>& new_shape);
    Tensor squeeze(int dim = -1) const;
    Tensor unsqueeze(int dim) const;
    Tensor transpose(int dim0, int dim1) const;
    Tensor permute(const std::vector<int>& dims) const;
    
    // Memory operations
    Tensor clone() const;
    Tensor to(MemoryType memory_type, int device_id = 0) const;
    Tensor toGPU(int device_id = 0) const { return to(MemoryType::GPU, device_id); }
    Tensor toCPU() const { return to(MemoryType::CPU, 0); }
    Tensor contiguous() const;
    bool isContiguous() const { return is_contiguous_; }
    
    // Type conversion
    Tensor to(DataType dtype) const;
    Tensor toFP16() const { return to(DataType::FP16); }
    Tensor toFP32() const { return to(DataType::FP32); }
    Tensor toBF16() const { return to(DataType::BF16); }
    
    // Utility
    void fill(float value);
    void zero() { fill(0.0f); }
    std::string toString() const;
    
    // Static factory methods
    static Tensor empty(const std::vector<int64_t>& shape,
                        DataType dtype = DataType::FP32,
                        MemoryType memory_type = MemoryType::GPU,
                        int device_id = 0);
    
    static Tensor zeros(const std::vector<int64_t>& shape,
                        DataType dtype = DataType::FP32,
                        MemoryType memory_type = MemoryType::GPU,
                        int device_id = 0);
    
    static Tensor ones(const std::vector<int64_t>& shape,
                       DataType dtype = DataType::FP32,
                       MemoryType memory_type = MemoryType::GPU,
                       int device_id = 0);
    
    static Tensor fromCPU(const void* data,
                          const std::vector<int64_t>& shape,
                          DataType dtype,
                          int device_id = 0);

private:
    void* data_ = nullptr;
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;
    int64_t numel_ = 0;
    DataType dtype_ = DataType::FP32;
    MemoryType memory_type_ = MemoryType::CPU;
    int device_id_ = 0;
    bool owns_data_ = false;
    bool is_contiguous_ = true;
    
    void allocate();
    void deallocate();
    void computeStrides();
    static int64_t computeNumel(const std::vector<int64_t>& shape);
};

/**
 * @brief Shared pointer wrapper for Tensor
 */
using TensorPtr = std::shared_ptr<Tensor>;

/**
 * @brief Create a shared tensor
 */
template <typename... Args>
TensorPtr makeTensor(Args&&... args) {
    return std::make_shared<Tensor>(std::forward<Args>(args)...);
}

} // namespace oasr
