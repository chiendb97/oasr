#pragma once

// Lazily-committed device memory region (PagedAttention-style physical paging, flat
// virtual addressing). Reserves the full virtual address range up front — kernel
// pointers and captured CUDA graphs stay valid for the decoder's lifetime — and maps
// physical memory in fixed-size chunks only where the region is actually used. Falls
// back to an eager cudaMalloc when the driver lacks VMM support (same footprint as the
// pre-paging decoder).

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace oasr::wfst {

class LazyRegion {
 public:
  LazyRegion() = default;
  ~LazyRegion() { Release(); }
  LazyRegion(const LazyRegion&) = delete;
  LazyRegion& operator=(const LazyRegion&) = delete;

  // Reserves va_bytes of address space on `device`; commits nothing yet (VMM) or
  // everything (eager fallback). chunk_target sets the physical mapping unit (rounded
  // up to the driver granularity): prefix-committed regions want large chunks (fewer
  // driver calls), strided per-lane regions want the minimum so holes stay unmapped.
  void Reserve(size_t va_bytes, int device, size_t chunk_target = kChunkTarget) {
    if (base_ != nullptr) throw std::runtime_error("LazyRegion: already reserved");
    device_ = device;
    va_ = va_bytes;
    int vmm = 0;
    CUdevice dev = 0;
    lazy_ = cuDeviceGet(&dev, device) == CUDA_SUCCESS &&
            cuDeviceGetAttribute(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED,
                                 dev) == CUDA_SUCCESS &&
            vmm != 0;
    if (lazy_) {
      CUmemAllocationProp prop{};
      prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
      prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      prop.location.id = device_;
      size_t gran = 0;
      Check(cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
            "cuMemGetAllocationGranularity");
      chunk_ = ((chunk_target + gran - 1) / gran) * gran;
      va_ = ((va_bytes + chunk_ - 1) / chunk_) * chunk_;
      CUdeviceptr p = 0;
      Check(cuMemAddressReserve(&p, va_, 0, 0, 0), "cuMemAddressReserve");
      base_ = reinterpret_cast<void*>(p);
      mapped_.assign(va_ / chunk_, false);
      handles_.assign(va_ / chunk_, CUmemGenericAllocationHandle{});
    } else {
      cudaError_t err = cudaMalloc(&base_, va_bytes);
      if (err != cudaSuccess)
        throw std::runtime_error(std::string("LazyRegion fallback cudaMalloc: ") +
                                 cudaGetErrorString(err));
      committed_ = va_bytes;
    }
  }

  // Ensures [offset, offset + bytes) is physically backed.
  void EnsureRange(size_t offset, size_t bytes) {
    if (!lazy_ || bytes == 0) return;
    if (offset + bytes > va_) throw std::runtime_error("LazyRegion: range beyond reservation");
    const size_t k0 = offset / chunk_;
    const size_t k1 = (offset + bytes - 1) / chunk_;
    for (size_t k = k0; k <= k1; ++k) {
      if (mapped_[k]) continue;
      CUmemAllocationProp prop{};
      prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
      prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      prop.location.id = device_;
      CUmemGenericAllocationHandle h{};
      Check(cuMemCreate(&h, chunk_, &prop, 0), "cuMemCreate");
      const CUdeviceptr addr = reinterpret_cast<CUdeviceptr>(base_) + k * chunk_;
      CUresult r = cuMemMap(addr, chunk_, 0, h, 0);
      if (r != CUDA_SUCCESS) {
        cuMemRelease(h);
        Check(r, "cuMemMap");
      }
      CUmemAccessDesc acc{};
      acc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      acc.location.id = device_;
      acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
      Check(cuMemSetAccess(addr, chunk_, &acc, 1), "cuMemSetAccess");
      mapped_[k] = true;
      handles_[k] = h;
      committed_ += chunk_;
    }
  }

  // Prefix commit convenience: ensures [0, bytes).
  void EnsurePrefix(size_t bytes) { EnsureRange(0, bytes); }

  // Unmaps and releases the physical chunks FULLY contained in [offset, offset + bytes).
  // Partially-covered edge chunks stay mapped (callers wanting exact release must pass
  // chunk-aligned ranges). No-op on the eager fallback. The caller must guarantee no
  // in-flight or future GPU work touches the range until it is re-committed.
  void ReleaseRange(size_t offset, size_t bytes) {
    if (!lazy_ || bytes == 0) return;
    if (offset + bytes > va_) throw std::runtime_error("LazyRegion: range beyond reservation");
    const size_t k0 = (offset + chunk_ - 1) / chunk_;            // first fully-inside chunk
    const size_t k1_end = (offset + bytes) / chunk_;             // one past the last
    for (size_t k = k0; k < k1_end; ++k) {
      if (!mapped_[k]) continue;
      const CUdeviceptr addr = reinterpret_cast<CUdeviceptr>(base_) + k * chunk_;
      Check(cuMemUnmap(addr, chunk_), "cuMemUnmap");
      Check(cuMemRelease(handles_[k]), "cuMemRelease");
      mapped_[k] = false;
      handles_[k] = CUmemGenericAllocationHandle{};
      committed_ -= chunk_;
    }
  }

  // Physical mapping unit chosen at Reserve (0 before Reserve or on the eager fallback).
  size_t chunk() const { return lazy_ ? chunk_ : 0; }

  // The mapping unit Reserve would pick on `device` for `chunk_target` (0 when the
  // driver lacks VMM support). Lets callers size chunk-aligned sub-regions up front.
  static size_t ChunkBytesFor(int device, size_t chunk_target = kChunkTarget) {
    int vmm = 0;
    CUdevice dev = 0;
    if (cuDeviceGet(&dev, device) != CUDA_SUCCESS ||
        cuDeviceGetAttribute(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED,
                             dev) != CUDA_SUCCESS ||
        vmm == 0)
      return 0;
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    size_t gran = 0;
    if (cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) !=
            CUDA_SUCCESS ||
        gran == 0)
      return 0;
    return ((chunk_target + gran - 1) / gran) * gran;
  }

  void* ptr() const { return base_; }
  size_t reserved() const { return va_; }
  size_t committed() const { return committed_; }
  bool lazy() const { return lazy_; }

  void Release() {
    if (base_ == nullptr) return;
    if (lazy_) {
      for (size_t k = 0; k < mapped_.size(); ++k) {
        if (!mapped_[k]) continue;
        const CUdeviceptr addr = reinterpret_cast<CUdeviceptr>(base_) + k * chunk_;
        cuMemUnmap(addr, chunk_);
        cuMemRelease(handles_[k]);
      }
      cuMemAddressFree(reinterpret_cast<CUdeviceptr>(base_), va_);
    } else {
      cudaFree(base_);
    }
    base_ = nullptr;
    mapped_.clear();
    handles_.clear();
    committed_ = 0;
    va_ = 0;
  }

  static constexpr size_t kChunkTarget = 32ull << 20;  // default physical mapping unit

 private:
  static void Check(CUresult r, const char* what) {
    if (r == CUDA_SUCCESS) return;
    const char* msg = nullptr;
    cuGetErrorString(r, &msg);
    throw std::runtime_error(std::string("LazyRegion ") + what + ": " +
                             (msg != nullptr ? msg : "unknown CUDA driver error"));
  }

  void* base_ = nullptr;
  size_t va_ = 0;
  size_t committed_ = 0;
  size_t chunk_ = 0;
  int device_ = 0;
  bool lazy_ = false;
  std::vector<bool> mapped_;
  std::vector<CUmemGenericAllocationHandle> handles_;
};

}  // namespace oasr::wfst
