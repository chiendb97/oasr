// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// RAII workspace allocation that is safe under CUDA Graph stream capture.
//
// CUTLASS's stock ``cutlass::device_memory::allocation<uint8_t>`` calls
// synchronous ``cudaMalloc`` / ``cudaFree``. Synchronous allocator calls
// silently break ``cudaStreamCaptureModeRelaxed`` capture: the capture state
// transitions to "invalid" and subsequent kernel launches on the same stream
// stop being recorded, so the resulting CUDA Graph is partially populated --
// captured replays reproduce only some of the intended ops, which manifests
// as NaN / wrong values in downstream state (e.g. the engine's K/V pool and
// CNN cache once any cutlass-based op runs inside the captured encoder
// forward).
//
// ``cudaMallocAsync`` / ``cudaFreeAsync`` are stream-ordered allocator calls
// and are explicitly supported during stream capture (they're recorded into
// the graph alongside the kernel that uses the buffer). Using them keeps the
// capture's ops contiguous and the replay reproduces the captured execution.

#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace oasr {

// RAII wrapper around stream-ordered allocator that's safe to use inside a
// ``cudaStreamCapture`` region. Mirrors the subset of the
// ``cutlass::device_memory::allocation`` API the templates rely on (``get()``
// + RAII free).
class GraphSafeWorkspace {
public:
    GraphSafeWorkspace(std::size_t size, cudaStream_t stream)
        : ptr_(nullptr), stream_(stream) {
        if (size > 0) {
            // Failure here would be detected later by the kernel; mirror
            // cutlass's behaviour of leaving ``ptr_`` nullptr on failure so
            // ``conv_op.initialize(... workspace.get() ...)`` returns an
            // error status that the caller surfaces.
            (void)cudaMallocAsync(&ptr_, size, stream_);
        }
    }

    ~GraphSafeWorkspace() {
        if (ptr_ != nullptr) {
            (void)cudaFreeAsync(ptr_, stream_);
        }
    }

    GraphSafeWorkspace(const GraphSafeWorkspace&) = delete;
    GraphSafeWorkspace& operator=(const GraphSafeWorkspace&) = delete;

    std::uint8_t* get() const noexcept { return reinterpret_cast<std::uint8_t*>(ptr_); }

private:
    void* ptr_;
    cudaStream_t stream_;
};

}  // namespace oasr
