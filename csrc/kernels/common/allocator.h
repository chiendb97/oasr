// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "types.h"
#include <cstddef>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <cuda_runtime.h>

namespace oasr {

/**
 * @brief Abstract base class for memory allocators
 */
class Allocator {
public:
    virtual ~Allocator() = default;
    
    /**
     * @brief Allocate memory
     * @param size Number of bytes to allocate
     * @param alignment Memory alignment (default 256 for GPU)
     * @return Pointer to allocated memory
     */
    virtual void* allocate(size_t size, size_t alignment = 256) = 0;
    
    /**
     * @brief Free memory
     * @param ptr Pointer to free
     */
    virtual void free(void* ptr) = 0;
    
    /**
     * @brief Get memory type this allocator manages
     */
    virtual MemoryType memoryType() const = 0;
    
    /**
     * @brief Get total allocated bytes
     */
    virtual size_t allocatedBytes() const = 0;
    
    /**
     * @brief Get peak allocated bytes
     */
    virtual size_t peakAllocatedBytes() const = 0;
    
    /**
     * @brief Reset peak memory tracking
     */
    virtual void resetPeakStats() = 0;
};

/**
 * @brief CPU memory allocator using aligned_alloc
 */
class CpuAllocator : public Allocator {
public:
    CpuAllocator() = default;
    ~CpuAllocator() override;
    
    void* allocate(size_t size, size_t alignment = 64) override;
    void free(void* ptr) override;
    
    MemoryType memoryType() const override { return MemoryType::CPU; }
    size_t allocatedBytes() const override { return allocated_bytes_; }
    size_t peakAllocatedBytes() const override { return peak_allocated_bytes_; }
    void resetPeakStats() override { peak_allocated_bytes_ = allocated_bytes_; }
    
private:
    std::unordered_map<void*, size_t> allocations_;
    size_t allocated_bytes_ = 0;
    size_t peak_allocated_bytes_ = 0;
    mutable std::mutex mutex_;
};

/**
 * @brief CUDA device memory allocator
 */
class CudaAllocator : public Allocator {
public:
    explicit CudaAllocator(int device_id = 0);
    ~CudaAllocator() override;
    
    void* allocate(size_t size, size_t alignment = 256) override;
    void free(void* ptr) override;
    
    MemoryType memoryType() const override { return MemoryType::GPU; }
    size_t allocatedBytes() const override { return allocated_bytes_; }
    size_t peakAllocatedBytes() const override { return peak_allocated_bytes_; }
    void resetPeakStats() override { peak_allocated_bytes_ = allocated_bytes_; }
    
    int deviceId() const { return device_id_; }
    
private:
    int device_id_;
    std::unordered_map<void*, size_t> allocations_;
    size_t allocated_bytes_ = 0;
    size_t peak_allocated_bytes_ = 0;
    mutable std::mutex mutex_;
};

/**
 * @brief Pinned (page-locked) host memory allocator
 * 
 * Useful for faster CPU-GPU transfers
 */
class PinnedAllocator : public Allocator {
public:
    PinnedAllocator() = default;
    ~PinnedAllocator() override;
    
    void* allocate(size_t size, size_t alignment = 256) override;
    void free(void* ptr) override;
    
    MemoryType memoryType() const override { return MemoryType::PINNED; }
    size_t allocatedBytes() const override { return allocated_bytes_; }
    size_t peakAllocatedBytes() const override { return peak_allocated_bytes_; }
    void resetPeakStats() override { peak_allocated_bytes_ = allocated_bytes_; }
    
private:
    std::unordered_map<void*, size_t> allocations_;
    size_t allocated_bytes_ = 0;
    size_t peak_allocated_bytes_ = 0;
    mutable std::mutex mutex_;
};

/**
 * @brief Caching allocator to reduce allocation overhead
 * 
 * Based on PyTorch's CUDACachingAllocator design
 */
class CachingAllocator : public Allocator {
public:
    explicit CachingAllocator(std::unique_ptr<Allocator> base_allocator,
                               size_t max_cached_bytes = 0);
    ~CachingAllocator() override;
    
    void* allocate(size_t size, size_t alignment = 256) override;
    void free(void* ptr) override;
    
    MemoryType memoryType() const override { return base_allocator_->memoryType(); }
    size_t allocatedBytes() const override { return allocated_bytes_; }
    size_t peakAllocatedBytes() const override { return peak_allocated_bytes_; }
    void resetPeakStats() override { peak_allocated_bytes_ = allocated_bytes_; }
    
    /**
     * @brief Release all cached memory back to the system
     */
    void emptyCache();
    
    /**
     * @brief Get amount of cached memory
     */
    size_t cachedBytes() const { return cached_bytes_; }
    
private:
    std::unique_ptr<Allocator> base_allocator_;
    size_t max_cached_bytes_;
    
    // Cached blocks organized by size
    struct Block {
        void* ptr;
        size_t size;
    };
    std::unordered_multimap<size_t, Block> free_blocks_;
    std::unordered_map<void*, size_t> active_blocks_;
    
    size_t allocated_bytes_ = 0;
    size_t peak_allocated_bytes_ = 0;
    size_t cached_bytes_ = 0;
    mutable std::mutex mutex_;
    
    // Round size up to reduce fragmentation
    static size_t roundSize(size_t size);
};

/**
 * @brief Get the default allocator for a given memory type
 */
Allocator* getDefaultAllocator(MemoryType memory_type, int device_id = 0);

/**
 * @brief Set the default allocator for a given memory type
 */
void setDefaultAllocator(MemoryType memory_type, 
                         std::shared_ptr<Allocator> allocator,
                         int device_id = 0);

} // namespace oasr
