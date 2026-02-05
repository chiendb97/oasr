// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Parameter structures for Batched Matrix Multiplication (BMM)

#pragma once

#include "gemm_configs.h"
#include "common/types.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace oasr {
namespace kernels {
namespace gemm {

//==============================================================================
// Batched GEMM Parameters
//==============================================================================

/**
 * @brief Parameters for batched GEMM (BMM) operation
 * 
 * Computes: D[b] = alpha * op(A[b]) @ op(B[b]) + beta * C[b] for b in [0, batch_size)
 * 
 * Supports two modes:
 *   1. Strided batched: Regular strides between batches
 *   2. Array of pointers: Arbitrary pointer arrays
 */
struct BmmParams {
    // For strided mode
    const void* A;              // [batch, M, K]
    const void* B;              // [batch, K, N]
    const void* C;              // [batch, M, N] optional
    void* D;                    // [batch, M, N]
    
    // For pointer array mode
    const void* const* A_array = nullptr;  // Array of A pointers
    const void* const* B_array = nullptr;  // Array of B pointers
    const void* const* C_array = nullptr;  // Array of C pointers
    void* const* D_array = nullptr;        // Array of D pointers
    
    // Batch size
    int batch_size;
    
    // Problem dimensions (same for all batches)
    int M;
    int N;
    int K;
    
    // Leading dimensions
    int64_t lda;
    int64_t ldb;
    int64_t ldc;
    int64_t ldd;
    
    // Batch strides (for strided mode)
    int64_t stride_a;           // Stride between batches in A
    int64_t stride_b;           // Stride between batches in B
    int64_t stride_c;           // Stride between batches in C
    int64_t stride_d;           // Stride between batches in D
    
    // Scaling factors
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Per-tensor scaling (for FP8)
    const float* scale_a = nullptr;  // [batch] or [1] for broadcast
    const float* scale_b = nullptr;  // [batch] or [1] for broadcast
    
    // Transpose operations
    TransposeOp trans_a = TransposeOp::NoTranspose;
    TransposeOp trans_b = TransposeOp::NoTranspose;
    
    // Data types
    DataType dtype_a = DataType::FP16;
    DataType dtype_b = DataType::FP16;
    DataType dtype_c = DataType::FP16;
    DataType dtype_d = DataType::FP16;
    DataType dtype_accumulator = DataType::FP32;
    
    // Mode
    bool use_pointer_array = false;  // If true, use A_array/B_array/etc.
    
    // Configuration
    GemmConfig config;
    EpilogueFusion epilogue_fusion = EpilogueFusion::NONE;
    
    // Workspace
    void* workspace = nullptr;
    size_t workspace_size = 0;
    
    cudaStream_t stream = nullptr;
    
    BmmParams() = default;
    
    /**
     * @brief Factory method for strided batched GEMM
     */
    static BmmParams Strided(
        const void* a, const void* b, void* d,
        int batch, int m, int n, int k,
        DataType dtype,
        cudaStream_t stream = nullptr)
    {
        BmmParams params;
        params.A = a;
        params.B = b;
        params.D = d;
        params.batch_size = batch;
        params.M = m;
        params.N = n;
        params.K = k;
        params.lda = k;
        params.ldb = n;
        params.ldc = n;
        params.ldd = n;
        params.stride_a = static_cast<int64_t>(m) * k;
        params.stride_b = static_cast<int64_t>(k) * n;
        params.stride_c = static_cast<int64_t>(m) * n;
        params.stride_d = static_cast<int64_t>(m) * n;
        params.dtype_a = dtype;
        params.dtype_b = dtype;
        params.dtype_d = dtype;
        params.stream = stream;
        params.use_pointer_array = false;
        return params;
    }
    
    /**
     * @brief Factory method for strided batched GEMM with custom strides
     */
    static BmmParams StridedCustom(
        const void* a, const void* b, void* d,
        int batch, int m, int n, int k,
        int64_t stride_a, int64_t stride_b, int64_t stride_d,
        DataType dtype,
        cudaStream_t stream = nullptr)
    {
        BmmParams params;
        params.A = a;
        params.B = b;
        params.D = d;
        params.batch_size = batch;
        params.M = m;
        params.N = n;
        params.K = k;
        params.lda = k;
        params.ldb = n;
        params.ldd = n;
        params.stride_a = stride_a;
        params.stride_b = stride_b;
        params.stride_c = stride_d;
        params.stride_d = stride_d;
        params.dtype_a = dtype;
        params.dtype_b = dtype;
        params.dtype_d = dtype;
        params.stream = stream;
        params.use_pointer_array = false;
        return params;
    }
    
    /**
     * @brief Factory method for pointer array mode
     */
    static BmmParams PointerArray(
        const void* const* a_array,
        const void* const* b_array,
        void* const* d_array,
        int batch, int m, int n, int k,
        DataType dtype,
        cudaStream_t stream = nullptr)
    {
        BmmParams params;
        params.A_array = a_array;
        params.B_array = b_array;
        params.D_array = d_array;
        params.batch_size = batch;
        params.M = m;
        params.N = n;
        params.K = k;
        params.lda = k;
        params.ldb = n;
        params.ldd = n;
        params.dtype_a = dtype;
        params.dtype_b = dtype;
        params.dtype_d = dtype;
        params.stream = stream;
        params.use_pointer_array = true;
        return params;
    }
    
    /**
     * @brief Set transpose operations
     */
    BmmParams& withTranspose(TransposeOp ta, TransposeOp tb) {
        trans_a = ta;
        trans_b = tb;
        return *this;
    }
    
    /**
     * @brief Set scaling factors
     */
    BmmParams& withScaling(float a, float b) {
        alpha = a;
        beta = b;
        return *this;
    }
};

//==============================================================================
// Workspace Size Query
//==============================================================================

/**
 * @brief Query workspace size for batched GEMM
 */
inline size_t getBmmWorkspaceSize(int batch, int M, int N, int K,
                                  DataType dtype,
                                  const GemmConfig& config = GemmConfig()) {
    (void)batch; (void)M; (void)N; (void)K; (void)dtype; (void)config;
    return 8 * 1024 * 1024;  // 8MB
}

} // namespace gemm
} // namespace kernels
} // namespace oasr
