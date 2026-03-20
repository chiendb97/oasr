// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Shared utilities for GEMM kernels

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace oasr {
namespace kernels {
namespace gemm {

//==============================================================================
// Status Codes
//==============================================================================

/**
 * @brief GEMM operation status codes
 */
enum class GemmStatus {
    SUCCESS = 0,
    INVALID_ARGUMENT,
    WORKSPACE_TOO_SMALL,
    NOT_SUPPORTED,
    INTERNAL_ERROR,
    CUDA_ERROR,
    CUTLASS_ERROR,
};

/**
 * @brief Convert status to string
 */
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

//==============================================================================
// Dispatch Macros
//==============================================================================

/**
 * @brief Dispatch based on data type
 */
#define OASR_DISPATCH_GEMM_DTYPE(dtype, DTYPE_TYPE, ...)            \
    do {                                                            \
        if ((dtype) == DataType::FP16) {                            \
            using DTYPE_TYPE = half;                                \
            __VA_ARGS__                                             \
        } else if ((dtype) == DataType::BF16) {                     \
            using DTYPE_TYPE = __nv_bfloat16;                       \
            __VA_ARGS__                                             \
        } else {                                                    \
            throw std::runtime_error("Unsupported dtype for GEMM"); \
        }                                                           \
    } while (0)

/**
 * @brief Dispatch based on transpose operation
 */
#define OASR_DISPATCH_TRANSPOSE(trans_a, trans_b, TA, TB, ...)                                     \
    do {                                                                                           \
        if ((trans_a) == TransposeOp::NoTranspose && (trans_b) == TransposeOp::NoTranspose) {      \
            constexpr bool TA = false;                                                             \
            constexpr bool TB = false;                                                             \
            __VA_ARGS__                                                                            \
        } else if ((trans_a) == TransposeOp::Transpose && (trans_b) == TransposeOp::NoTranspose) { \
            constexpr bool TA = true;                                                              \
            constexpr bool TB = false;                                                             \
            __VA_ARGS__                                                                            \
        } else if ((trans_a) == TransposeOp::NoTranspose && (trans_b) == TransposeOp::Transpose) { \
            constexpr bool TA = false;                                                             \
            constexpr bool TB = true;                                                              \
            __VA_ARGS__                                                                            \
        } else {                                                                                   \
            constexpr bool TA = true;                                                              \
            constexpr bool TB = true;                                                              \
            __VA_ARGS__                                                                            \
        }                                                                                          \
    } while (0)

/**
 * @brief Dispatch based on weight layout (column-major vs row-major)
 */
#define OASR_DISPATCH_WEIGHT_LAYOUT(is_column_major, WEIGHT_LAYOUT, ...) \
    do {                                                                 \
        if (is_column_major) {                                           \
            constexpr bool WEIGHT_LAYOUT = true;                         \
            __VA_ARGS__                                                  \
        } else {                                                         \
            constexpr bool WEIGHT_LAYOUT = false;                        \
            __VA_ARGS__                                                  \
        }                                                                \
    } while (0)

/**
 * @brief Dispatch based on SM version
 */
#define OASR_DISPATCH_SM_VERSION(sm_version, SM_VERSION, ...) \
    do {                                                      \
        if ((sm_version) >= 90) {                             \
            constexpr int SM_VERSION = 90;                    \
            __VA_ARGS__                                       \
        } else {                                              \
            constexpr int SM_VERSION = 80;                    \
            __VA_ARGS__                                       \
        }                                                     \
    } while (0)

//==============================================================================
// CUTLASS Helper Macros
//==============================================================================

#define CUTLASS_CHECK(status)                                                                 \
    do {                                                                                      \
        cutlass::Status _status = (status);                                                   \
        if (_status != cutlass::Status::kSuccess) {                                           \
            std::ostringstream oss;                                                           \
            oss << "CUTLASS error: " << cutlassGetStatusString(_status) << " at " << __FILE__ \
                << ":" << __LINE__;                                                           \
            throw std::runtime_error(oss.str());                                              \
        }                                                                                     \
    } while (0)

}  // namespace gemm
}  // namespace kernels
}  // namespace oasr
