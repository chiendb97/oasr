// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Persistent, stream-keyed workspace cache for CUTLASS kernel workspaces.
//
// Per-call allocation, zeroing and release can dominate thin GEMMs, so reusable
// workspaces preserve the intended benefit of split-K and Stream-K kernels.
//
// Two pools with different contracts:
//
//  * ``kZeroedSemaphore`` — for CUTLASS *serial split-K* semaphores.  The
//    serial split-K kernel is self-restoring: the final K-slice releases the
//    per-tile lock back to 0 (see ``cutlass/gemm/kernel/gemm.h``), so a buffer
//    zeroed ONCE at allocation stays all-zero across launches and the
//    per-launch ``cudaMemsetAsync`` in ``device::Gemm::initialize`` can be
//    skipped entirely, making serial split-K a single-launch kernel like
//    cuBLAS's splitK kernels.  Users of this pool MUST restore zeros (i.e.
//    only the serial split-K semaphore qualifies).
//
//  * ``kScratch`` — plain reusable device scratch with no content contract
//    (Stream-K barrier+partials workspace, which is re-zeroed per launch by
//    ``initialize``; parallel split-K partials, which are fully overwritten).
//    Saves the alloc/free churn only.
//
// Properties:
//  * one grow-only buffer per (device, stream, pool); map is mutex-protected.
//  * outgrown buffers are intentionally retired WITHOUT freeing: a captured
//    CUDA graph may still reference their address.  Semaphore buffers are one
//    int per output tile and scratch is capped, so the leak is bounded.
//  * during CUDA graph capture no new allocation is attempted (a
//    ``cudaMallocAsync`` recorded into a graph is graph-owned and unsafe to
//    cache).  A pre-existing large-enough buffer IS returned — its address is
//    baked into the graph, which is safe because buffers are never freed.
//    Otherwise ``nullptr`` is returned and the caller must fall back to the
//    legacy per-call ``GraphSafeWorkspace`` path.
//  * rollback: set ``OASR_GEMM_WS_CACHE=0`` to always return ``nullptr``
//    (callers then behave exactly as before this optimisation).

#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <tuple>
#include <map>
#include <vector>

namespace oasr {

enum class WorkspacePool : int {
    kZeroedSemaphore = 0,  // must stay zeroed; users restore zeros after use
    kScratch = 1,          // no content contract
};

namespace detail {

struct WorkspaceCacheState {
    std::mutex mutex;
    // (device, stream, pool) → current buffer
    std::map<std::tuple<int, cudaStream_t, int>, std::pair<void*, size_t>> buffers;
    // Retired (outgrown) buffers — kept alive forever; may be referenced by
    // captured CUDA graphs.
    std::vector<void*> retired;
};

inline WorkspaceCacheState& workspaceCacheState() {
    static WorkspaceCacheState state;
    return state;
}

inline bool workspaceCacheEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("OASR_GEMM_WS_CACHE");
        return env == nullptr || env[0] != '0';
    }();
    return enabled;
}

}  // namespace detail

/// Return a device buffer of at least ``bytes`` for (current device, stream).
///
/// ``kZeroedSemaphore`` buffers are zero-filled at allocation (stream-ordered
/// on ``stream``) and rely on their users to restore zeros.  Returns
/// ``nullptr`` when ``bytes`` is 0, when the cache is disabled, when a stream
/// capture is in progress and no cached buffer fits, or on allocation failure
/// — callers must fall back to a per-call workspace in that case.
inline void* getCachedWorkspace(size_t bytes, cudaStream_t stream, WorkspacePool pool) {
    if (bytes == 0 || !detail::workspaceCacheEnabled()) {
        return nullptr;
    }

    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return nullptr;
    }

    auto& state = detail::workspaceCacheState();
    std::lock_guard<std::mutex> lock(state.mutex);

    auto key = std::make_tuple(device, stream, static_cast<int>(pool));
    auto it = state.buffers.find(key);
    if (it != state.buffers.end() && it->second.second >= bytes) {
        return it->second.first;
    }

    // Need to (re)allocate.  Not safe while this stream is being captured.
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess) {
        (void)cudaGetLastError();  // clear (e.g. legacy-stream capture error)
        return nullptr;
    }
    if (capture != cudaStreamCaptureStatusNone) {
        return nullptr;
    }

    // Grow-only: round up (min 4 KiB, 2× headroom) to avoid churn.
    size_t new_size = bytes < 4096 ? 4096 : bytes * 2;
    void* ptr = nullptr;
    if (cudaMallocAsync(&ptr, new_size, stream) != cudaSuccess) {
        (void)cudaGetLastError();
        return nullptr;
    }
    if (pool == WorkspacePool::kZeroedSemaphore) {
        if (cudaMemsetAsync(ptr, 0, new_size, stream) != cudaSuccess) {
            (void)cudaGetLastError();
            (void)cudaFreeAsync(ptr, stream);
            return nullptr;
        }
    }
    if (it != state.buffers.end()) {
        state.retired.push_back(it->second.first);  // never freed — see header
        it->second = {ptr, new_size};
    } else {
        state.buffers.emplace(key, std::make_pair(ptr, new_size));
    }
    return ptr;
}

}  // namespace oasr
