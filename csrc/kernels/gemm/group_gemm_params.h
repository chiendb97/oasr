// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Parameter structures for Grouped GEMM operations

#pragma once

#include "gemm_configs.h"
#include "common/types.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace oasr {
namespace kernels {
namespace gemm {

//==============================================================================
// Grouped GEMM Parameters
//==============================================================================

/**
 * @brief Problem specification for a single GEMM in a group
 */
struct GemmProblemDesc {
    int M;
    int N;
    int K;
    
    GemmProblemDesc() : M(0), N(0), K(0) {}
    GemmProblemDesc(int m, int n, int k) : M(m), N(n), K(k) {}
};

/**
 * @brief Parameters for grouped GEMM operation
 * 
 * Computes: D[i] = alpha * A[i] @ B[i] + beta * C[i] for variable-sized problems
 * 
 * Each problem in the group can have different M, N, K dimensions.
 * This is useful for:
 *   - Variable-length sequence processing
 *   - LoRA with different ranks
 *   - Expert mixture models (MoE)
 */
struct GroupGemmParams {
    // Problem specifications (on device)
    const GemmProblemDesc* problems;  // [num_problems] on device
    int num_problems;
    
    // Input/output pointers (all on device)
    const void* const* A_array;   // [num_problems] array of A pointers
    const void* const* B_array;   // [num_problems] array of B pointers
    const void* const* C_array;   // [num_problems] array of C pointers (optional)
    void* const* D_array;         // [num_problems] array of D pointers
    
    // Leading dimensions (on device)
    const int64_t* lda_array;     // [num_problems]
    const int64_t* ldb_array;     // [num_problems]
    const int64_t* ldc_array;     // [num_problems]
    const int64_t* ldd_array;     // [num_problems]
    
    // Scaling factors
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Per-problem scaling (optional)
    const float* alpha_array = nullptr;  // [num_problems]
    const float* beta_array = nullptr;   // [num_problems]
    
    // Matrix layouts
    bool weight_column_major = false;  // If true, B matrices are column-major
    
    // Data types
    DataType dtype_a = DataType::FP16;
    DataType dtype_b = DataType::FP16;
    DataType dtype_c = DataType::FP16;
    DataType dtype_d = DataType::FP16;
    DataType dtype_accumulator = DataType::FP32;
    
    // Configuration
    GemmConfig config;
    
    // Workspace buffers
    void* workspace_float = nullptr;   // Float workspace for intermediate results
    size_t workspace_float_size = 0;
    void* workspace_int = nullptr;     // Int workspace for indices/offsets
    size_t workspace_int_size = 0;
    
    cudaStream_t stream = nullptr;
    
    GroupGemmParams() = default;
    
    /**
     * @brief Factory method for uniform problem sizes (same M, N, K for all)
     */
    static GroupGemmParams Uniform(
        const void* const* a_array,
        const void* const* b_array,
        void* const* d_array,
        int num_problems, int m, int n, int k,
        DataType dtype,
        cudaStream_t stream = nullptr)
    {
        GroupGemmParams params;
        params.A_array = a_array;
        params.B_array = b_array;
        params.D_array = d_array;
        params.num_problems = num_problems;
        params.dtype_a = dtype;
        params.dtype_b = dtype;
        params.dtype_d = dtype;
        params.stream = stream;
        // Note: problems, lda_array, etc. need to be set separately
        return params;
    }
};

//==============================================================================
// Workspace Size Query
//==============================================================================

/**
 * @brief Query workspace sizes for grouped GEMM
 */
inline void getGroupGemmWorkspaceSize(int num_problems,
                                      const GemmProblemDesc* problems,
                                      DataType dtype,
                                      size_t& float_workspace_size,
                                      size_t& int_workspace_size,
                                      const GemmConfig& config = GemmConfig()) {
    (void)num_problems; (void)problems; (void)dtype; (void)config;
    float_workspace_size = 16 * 1024 * 1024;  // 16MB
    int_workspace_size = 1 * 1024 * 1024;     // 1MB
}

} // namespace gemm
} // namespace kernels
} // namespace oasr
