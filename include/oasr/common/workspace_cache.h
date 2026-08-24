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
//    CUDA graph may still reference their address.  So every byte this cache
//    hands out is held until the process exits -- it is a cache, not an
//    allocator, and what it caches has to be bounded in BYTES.  It is, by
//    ``kMaxBytesPerKey`` and ``kMaxTotalBytes``; past either it returns
//    ``nullptr``, which is what it also does when disabled: correct, just back
//    on the per-call workspace path.
//  * the bytes are what matter, not the number of keys.  ``torch.cuda.Stream()``
//    hands out a POOL of 32 stream handles per device and cycles through it, so
//    a caller that makes a stream per unit of work never exceeds 32 keys per
//    pool -- but each of those keys grows to hold the largest workspace any
//    shape ever asked for on it.  A parallel split-K workspace is
//    ``M*N*4*split`` bytes, so one 4096x5008 shape is 328 MiB per key: measured,
//    96 stream-per-iteration calls at that shape held 10,016 MiB and the next
//    ladder step failed the kernel outright.  ``scripts/tune_asr_gemm.py`` hit
//    exactly this over a 121-shape sweep (30 GiB of non-PyTorch device memory,
//    dying on a 66 MiB allocation while PyTorch's own allocator held 1.00 GiB).
//    Serving never sees it: every architecture in the tree, offline and
//    streaming, asks this cache for 152-296 bytes of semaphore and at most
//    1 MiB of scratch.  ``oasr::cachedWorkspaceBytes()`` is exported through the
//    gemm module so that is checkable rather than asserted.
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
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>

namespace oasr {

enum class WorkspacePool : int {
    kZeroedSemaphore = 0,  // must stay zeroed; users restore zeros after use
    kScratch = 1,          // no content contract
};

namespace detail {

// What this cache is for: removing a per-call allocation -- and, for the serial
// split-K semaphore, a per-launch memset that costs a whole kernel launch --
// from a THIN GEMM, where that ritual is a large fraction of the kernel's own
// time.  A workspace of tens of MiB belongs to a kernel long enough that the
// ritual is noise, and caching it would hold those bytes forever on every stream
// that ever ran the shape.  So: cache the small ones, decline the large ones.
//
// Sized from measurement, not headroom-guessing.  Over real checkpoints for
// every architecture in the tree, offline and streaming, the largest request
// this cache ever sees is 1 MiB (Conformer scratch, 2 keys); the semaphore pool
// asks for 152-296 bytes.  8 MiB per key is 8x the worst real request and 64 MiB
// total is 32x what a whole served process holds.
inline constexpr size_t kMaxBytesPerKey = size_t{8} << 20;
inline constexpr size_t kMaxTotalBytes = size_t{64} << 20;

struct WorkspaceCacheState {
    std::mutex mutex;
    // (device, stream, pool) → current buffer
    std::map<std::tuple<int, cudaStream_t, int>, std::pair<void*, size_t>> buffers;
    // Retired (outgrown) buffers — kept alive forever; may be referenced by
    // captured CUDA graphs.
    std::vector<void*> retired;
    // Every byte ever handed out and never freed, retired buffers included.
    // That is the number the budget has to be about: a grow reuses a key but
    // does not give the old allocation back.
    size_t held_bytes = 0;
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
/// capture is in progress and no cached buffer fits, when ``bytes`` exceeds
/// ``kMaxBytesPerKey``, when the cache is already holding ``kMaxTotalBytes``, or
/// on allocation failure — callers must fall back to a per-call workspace in
/// that case.
inline void* getCachedWorkspace(size_t bytes, cudaStream_t stream, WorkspacePool pool) {
    if (bytes == 0 || bytes > detail::kMaxBytesPerKey || !detail::workspaceCacheEnabled()) {
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

    // Grow-only: round up (min 4 KiB, 2× headroom) to avoid churn, but never
    // past the per-key ceiling -- a request that only fits without headroom is
    // still worth caching.
    size_t new_size = bytes < 4096 ? 4096 : bytes * 2;
    if (new_size > detail::kMaxBytesPerKey) {
        new_size = detail::kMaxBytesPerKey;
    }

    // Would this allocation put the cache over budget?  Decline; the caller's
    // per-call workspace path is correct, just without the saved ritual.
    //
    // Say so once: a process that gets here is holding tens of MiB of device
    // memory forever, or has quietly lost the cache, and either should be
    // visible rather than inferred from a slowdown or an OOM elsewhere.
    if (state.held_bytes + new_size > detail::kMaxTotalBytes) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            std::fprintf(stderr,
                         "[oasr] GEMM workspace cache is at its %zu MiB ceiling (request "
                         "%zu B); further workspaces use a per-call allocation. A caller is "
                         "running large split-K GEMMs across many CUDA streams.\n",
                         detail::kMaxTotalBytes >> 20, bytes);
        }
        return nullptr;
    }
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
    state.held_bytes += new_size;
    if (it != state.buffers.end()) {
        state.retired.push_back(it->second.first);  // never freed — see header
        it->second = {ptr, new_size};
    } else {
        state.buffers.emplace(key, std::make_pair(ptr, new_size));
    }
    return ptr;
}

/// Number of live (device, stream, pool) keys.  Diagnostic only.
inline int64_t cachedWorkspaceKeys() {
    auto& state = detail::workspaceCacheState();
    std::lock_guard<std::mutex> lock(state.mutex);
    return static_cast<int64_t>(state.buffers.size());
}

/// Device bytes this cache has handed out and will never free.  Diagnostic
/// only, and the reason it is exported: the invariant that matters here is a
/// byte count, and a test that has to infer it from ``cudaMemGetInfo`` cannot
/// tell "cached once" from "allocated and freed every call".
inline int64_t cachedWorkspaceBytes() {
    auto& state = detail::workspaceCacheState();
    std::lock_guard<std::mutex> lock(state.mutex);
    return static_cast<int64_t>(state.held_bytes);
}

}  // namespace oasr
