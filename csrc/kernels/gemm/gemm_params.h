// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Parameter structures for standard GEMM operations

#pragma once

#include "gemm_configs.h"
#include "common/types.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace oasr {
namespace kernels {
namespace gemm {

//==============================================================================
// Standard GEMM Parameters
//==============================================================================

/**
 * @brief Parameters for standard GEMM operation
 * 
 * Computes: D = alpha * op(A) @ op(B) + beta * C
 * 
 * Where:
 *   - A is [M, K] (or [K, M] if transposed)
 *   - B is [K, N] (or [N, K] if transposed)
 *   - C is [M, N] (optional, for bias/residual)
 *   - D is [M, N] (output)
 */
struct GemmParams {
    // Input matrices
    const void* A;              // [M, K] or [K, M]
    const void* B;              // [K, N] or [N, K]
    const void* C;              // [M, N] optional bias/residual
    void* D;                    // [M, N] output
    
    // Problem dimensions
    int M;                      // Number of rows of output
    int N;                      // Number of columns of output
    int K;                      // Contraction dimension
    
    // Leading dimensions (strides)
    int64_t lda;                // Leading dimension of A
    int64_t ldb;                // Leading dimension of B
    int64_t ldc;                // Leading dimension of C
    int64_t ldd;                // Leading dimension of D
    
    // Scaling factors
    float alpha = 1.0f;
    float beta = 0.0f;          // beta = 0 means ignore C
    
    // Transpose operations
    TransposeOp trans_a = TransposeOp::NoTranspose;
    TransposeOp trans_b = TransposeOp::NoTranspose;
    
    // Layout
    MatrixLayout layout_a = MatrixLayout::RowMajor;
    MatrixLayout layout_b = MatrixLayout::RowMajor;
    MatrixLayout layout_c = MatrixLayout::RowMajor;
    MatrixLayout layout_d = MatrixLayout::RowMajor;
    
    // Data types
    DataType dtype_a = DataType::FP16;
    DataType dtype_b = DataType::FP16;
    DataType dtype_c = DataType::FP16;
    DataType dtype_d = DataType::FP16;
    DataType dtype_accumulator = DataType::FP32;  // Internal accumulator type
    
    // Configuration
    GemmConfig config;
    EpilogueFusion epilogue_fusion = EpilogueFusion::NONE;
    
    // Workspace
    void* workspace = nullptr;
    size_t workspace_size = 0;
    
    // CUDA stream
    cudaStream_t stream = nullptr;
    
    GemmParams() = default;
    
    /**
     * @brief Convenience constructor for common case
     */
    GemmParams(const void* a, const void* b, void* d,
               int m, int n, int k, DataType dtype,
               cudaStream_t stream = nullptr)
        : A(a), B(b), C(nullptr), D(d)
        , M(m), N(n), K(k)
        , lda(k), ldb(n), ldc(n), ldd(n)  // Row-major defaults
        , dtype_a(dtype), dtype_b(dtype), dtype_c(dtype), dtype_d(dtype)
        , stream(stream) {}
    
    /**
     * @brief Set bias vector
     */
    GemmParams& withBias(const void* bias) {
        C = bias;
        beta = 1.0f;
        epilogue_fusion = EpilogueFusion::BIAS;
        return *this;
    }
    
    /**
     * @brief Set workspace
     */
    GemmParams& withWorkspace(void* ws, size_t size) {
        workspace = ws;
        workspace_size = size;
        return *this;
    }
    
    /**
     * @brief Set transpose operations
     */
    GemmParams& withTranspose(TransposeOp ta, TransposeOp tb) {
        trans_a = ta;
        trans_b = tb;
        return *this;
    }
};

//==============================================================================
// Workspace Size Query
//==============================================================================

/**
 * @brief Query workspace size for GEMM
 */
inline size_t getGemmWorkspaceSize(int M, int N, int K, DataType dtype,
                                   const GemmConfig& config = GemmConfig()) {
    // Conservative estimate: 4MB should be sufficient for most configurations
    (void)M; (void)N; (void)K; (void)dtype; (void)config;
    return 4 * 1024 * 1024;  // 4MB
}

} // namespace gemm
} // namespace kernels
} // namespace oasr
