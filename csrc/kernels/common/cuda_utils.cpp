// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/common/cuda_utils.h"

namespace oasr {

// =============================================================================
// CudaStream
// =============================================================================

CudaStream::CudaStream(int device_id, unsigned int flags)
    : device_id_(device_id) {
    DeviceGuard guard(device_id);
    OASR_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, flags));
}

CudaStream::~CudaStream() {
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
}

CudaStream::CudaStream(CudaStream&& other) noexcept
    : stream_(other.stream_), device_id_(other.device_id_) {
    other.stream_ = nullptr;
}

CudaStream& CudaStream::operator=(CudaStream&& other) noexcept {
    if (this != &other) {
        if (stream_) {
            cudaStreamDestroy(stream_);
        }
        stream_ = other.stream_;
        device_id_ = other.device_id_;
        other.stream_ = nullptr;
    }
    return *this;
}

void CudaStream::synchronize() {
    OASR_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

bool CudaStream::query() const {
    cudaError_t err = cudaStreamQuery(stream_);
    if (err == cudaSuccess) {
        return true;
    } else if (err == cudaErrorNotReady) {
        return false;
    }
    OASR_CUDA_CHECK(err);
    return false;
}

// =============================================================================
// CudaEvent
// =============================================================================

CudaEvent::CudaEvent(unsigned int flags) {
    OASR_CUDA_CHECK(cudaEventCreateWithFlags(&event_, flags));
}

CudaEvent::~CudaEvent() {
    if (event_) {
        cudaEventDestroy(event_);
    }
}

CudaEvent::CudaEvent(CudaEvent&& other) noexcept
    : event_(other.event_) {
    other.event_ = nullptr;
}

CudaEvent& CudaEvent::operator=(CudaEvent&& other) noexcept {
    if (this != &other) {
        if (event_) {
            cudaEventDestroy(event_);
        }
        event_ = other.event_;
        other.event_ = nullptr;
    }
    return *this;
}

void CudaEvent::record(cudaStream_t stream) {
    OASR_CUDA_CHECK(cudaEventRecord(event_, stream));
}

void CudaEvent::synchronize() {
    OASR_CUDA_CHECK(cudaEventSynchronize(event_));
}

bool CudaEvent::query() const {
    cudaError_t err = cudaEventQuery(event_);
    if (err == cudaSuccess) {
        return true;
    } else if (err == cudaErrorNotReady) {
        return false;
    }
    OASR_CUDA_CHECK(err);
    return false;
}

float CudaEvent::elapsedTime(const CudaEvent& start) const {
    float ms = 0.0f;
    OASR_CUDA_CHECK(cudaEventElapsedTime(&ms, start.event_, event_));
    return ms;
}

// =============================================================================
// CublasHandle
// =============================================================================

CublasHandle::CublasHandle() {
    OASR_CUBLAS_CHECK(cublasCreate(&handle_));
}

CublasHandle::~CublasHandle() {
    if (handle_) {
        cublasDestroy(handle_);
    }
}

void CublasHandle::setStream(cudaStream_t stream) {
    OASR_CUBLAS_CHECK(cublasSetStream(handle_, stream));
}

void CublasHandle::setMathMode(cublasMath_t mode) {
    OASR_CUBLAS_CHECK(cublasSetMathMode(handle_, mode));
}

// =============================================================================
// CudnnHandle
// =============================================================================

CudnnHandle::CudnnHandle() {
    OASR_CUDNN_CHECK(cudnnCreate(&handle_));
}

CudnnHandle::~CudnnHandle() {
    if (handle_) {
        cudnnDestroy(handle_);
    }
}

void CudnnHandle::setStream(cudaStream_t stream) {
    OASR_CUDNN_CHECK(cudnnSetStream(handle_, stream));
}

// =============================================================================
// DeviceGuard
// =============================================================================

DeviceGuard::DeviceGuard(int device_id) {
    OASR_CUDA_CHECK(cudaGetDevice(&original_device_));
    if (device_id != original_device_) {
        OASR_CUDA_CHECK(cudaSetDevice(device_id));
    }
}

DeviceGuard::~DeviceGuard() {
    cudaSetDevice(original_device_);
}

// =============================================================================
// Utility functions
// =============================================================================

cudaDeviceProp getDeviceProperties(int device_id) {
    cudaDeviceProp props;
    OASR_CUDA_CHECK(cudaGetDeviceProperties(&props, device_id));
    return props;
}

int getDeviceCount() {
    int count = 0;
    OASR_CUDA_CHECK(cudaGetDeviceCount(&count));
    return count;
}

int getCurrentDevice() {
    int device_id = 0;
    OASR_CUDA_CHECK(cudaGetDevice(&device_id));
    return device_id;
}

void setDevice(int device_id) {
    OASR_CUDA_CHECK(cudaSetDevice(device_id));
}

void synchronizeDevice() {
    OASR_CUDA_CHECK(cudaDeviceSynchronize());
}

size_t getAvailableMemory(int device_id) {
    DeviceGuard guard(device_id);
    size_t free_mem = 0, total_mem = 0;
    OASR_CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    return free_mem;
}

size_t getTotalMemory(int device_id) {
    DeviceGuard guard(device_id);
    size_t free_mem = 0, total_mem = 0;
    OASR_CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    return total_mem;
}

} // namespace oasr
