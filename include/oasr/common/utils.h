#pragma once

#include <algorithm>

namespace oasr {

constexpr int WARP_SIZE = 32;
constexpr int MAX_THREADS_PER_BLOCK = 1024;

// Compute warp-aligned block size clamped to [WARP_SIZE, max_threads].
inline int alignedBlockSize(int num_elements, int max_threads = MAX_THREADS_PER_BLOCK) {
    int block_size = std::min(num_elements, max_threads);
    return std::max(((block_size + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE, WARP_SIZE);
}

// Check pointer alignment for vectorized access of VecSize elements.
template <typename T, int VecSize>
inline bool isAligned(const void* ptr) {
    return reinterpret_cast<uintptr_t>(ptr) % (sizeof(T) * VecSize) == 0;
}

namespace gemm {

//==============================================================================
// GemmStatus -- operation status codes
//==============================================================================

enum class GemmStatus {
    SUCCESS = 0,
    INVALID_ARGUMENT,
    WORKSPACE_TOO_SMALL,
    NOT_SUPPORTED,
    INTERNAL_ERROR,
    CUDA_ERROR,
    CUTLASS_ERROR,
};

inline const char* getGemmStatusString(GemmStatus status) {
    switch (status) {
        case GemmStatus::SUCCESS:
            return "SUCCESS";
        case GemmStatus::INVALID_ARGUMENT:
            return "INVALID_ARGUMENT";
        case GemmStatus::WORKSPACE_TOO_SMALL:
            return "WORKSPACE_TOO_SMALL";
        case GemmStatus::NOT_SUPPORTED:
            return "NOT_SUPPORTED";
        case GemmStatus::INTERNAL_ERROR:
            return "INTERNAL_ERROR";
        case GemmStatus::CUDA_ERROR:
            return "CUDA_ERROR";
        case GemmStatus::CUTLASS_ERROR:
            return "CUTLASS_ERROR";
        default:
            return "UNKNOWN";
    }
}

}  // namespace gemm
}  // namespace oasr