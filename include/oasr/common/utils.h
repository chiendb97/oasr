#pragma once

namespace oasr {
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