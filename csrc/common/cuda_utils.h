// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <string>
#include <stdexcept>
#include <memory>

namespace oasr {

// CUDA error checking macro
#define OASR_CUDA_CHECK(call)                                                    \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            throw std::runtime_error(                                             \
                std::string("CUDA error at ") + __FILE__ + ":" +                  \
                std::to_string(__LINE__) + ": " + cudaGetErrorString(err));       \
        }                                                                          \
    } while (0)

// cuBLAS error checking macro
#define OASR_CUBLAS_CHECK(call)                                                  \
    do {                                                                          \
        cublasStatus_t status = call;                                             \
        if (status != CUBLAS_STATUS_SUCCESS) {                                    \
            throw std::runtime_error(                                             \
                std::string("cuBLAS error at ") + __FILE__ + ":" +                \
                std::to_string(__LINE__) + ": " + std::to_string(status));        \
        }                                                                          \
    } while (0)

// cuDNN error checking macro
#define OASR_CUDNN_CHECK(call)                                                   \
    do {                                                                          \
        cudnnStatus_t status = call;                                              \
        if (status != CUDNN_STATUS_SUCCESS) {                                     \
            throw std::runtime_error(                                             \
                std::string("cuDNN error at ") + __FILE__ + ":" +                 \
                std::to_string(__LINE__) + ": " + cudnnGetErrorString(status));   \
        }                                                                          \
    } while (0)

/**
 * @brief RAII wrapper for CUDA stream
 */
class CudaStream {
public:
    CudaStream(int device_id = 0, unsigned int flags = cudaStreamDefault);
    ~CudaStream();
    
    // Disable copy
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    
    // Enable move
    CudaStream(CudaStream&& other) noexcept;
    CudaStream& operator=(CudaStream&& other) noexcept;
    
    cudaStream_t get() const { return stream_; }
    operator cudaStream_t() const { return stream_; }
    
    void synchronize();
    bool query() const;
    
    int deviceId() const { return device_id_; }
    
private:
    cudaStream_t stream_ = nullptr;
    int device_id_ = 0;
};

/**
 * @brief RAII wrapper for CUDA event
 */
class CudaEvent {
public:
    CudaEvent(unsigned int flags = cudaEventDefault);
    ~CudaEvent();
    
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    
    CudaEvent(CudaEvent&& other) noexcept;
    CudaEvent& operator=(CudaEvent&& other) noexcept;
    
    cudaEvent_t get() const { return event_; }
    operator cudaEvent_t() const { return event_; }
    
    void record(cudaStream_t stream = nullptr);
    void synchronize();
    bool query() const;
    
    // Elapsed time in milliseconds between this event and other
    float elapsedTime(const CudaEvent& start) const;
    
private:
    cudaEvent_t event_ = nullptr;
};

/**
 * @brief cuBLAS handle manager
 */
class CublasHandle {
public:
    CublasHandle();
    ~CublasHandle();
    
    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;
    
    cublasHandle_t get() const { return handle_; }
    operator cublasHandle_t() const { return handle_; }
    
    void setStream(cudaStream_t stream);
    void setMathMode(cublasMath_t mode);
    
private:
    cublasHandle_t handle_ = nullptr;
};

/**
 * @brief cuDNN handle manager  
 */
class CudnnHandle {
public:
    CudnnHandle();
    ~CudnnHandle();
    
    CudnnHandle(const CudnnHandle&) = delete;
    CudnnHandle& operator=(const CudnnHandle&) = delete;
    
    cudnnHandle_t get() const { return handle_; }
    operator cudnnHandle_t() const { return handle_; }
    
    void setStream(cudaStream_t stream);
    
private:
    cudnnHandle_t handle_ = nullptr;
};

/**
 * @brief Device guard for temporarily switching CUDA device
 */
class DeviceGuard {
public:
    explicit DeviceGuard(int device_id);
    ~DeviceGuard();
    
    DeviceGuard(const DeviceGuard&) = delete;
    DeviceGuard& operator=(const DeviceGuard&) = delete;
    
private:
    int original_device_;
};

/**
 * @brief Get properties of a CUDA device
 */
cudaDeviceProp getDeviceProperties(int device_id = 0);

/**
 * @brief Get number of available CUDA devices
 */
int getDeviceCount();

/**
 * @brief Get current CUDA device
 */
int getCurrentDevice();

/**
 * @brief Set current CUDA device
 */
void setDevice(int device_id);

/**
 * @brief Synchronize all CUDA operations on current device
 */
void synchronizeDevice();

/**
 * @brief Get available GPU memory in bytes
 */
size_t getAvailableMemory(int device_id = 0);

/**
 * @brief Get total GPU memory in bytes
 */
size_t getTotalMemory(int device_id = 0);

} // namespace oasr
