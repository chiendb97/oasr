// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Batched Matrix Multiplication (BMM) kernel interfaces using NVIDIA CUTLASS
// Supports BF16 and FP16 precision

#pragma once

#include "gemm_configs.h"
#include "gemm_utils.h"
#include "common/types.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <vector>
#include <memory>

namespace oasr {
namespace kernels {
namespace gemm {

//==============================================================================
// Batched GEMM (BMM) Interface
//==============================================================================

/**
 * @brief Execute strided batched GEMM operation
 *
 * Computes: D[b] = alpha * A[b] @ B[b] + beta * D[b] for b in [0, batch_size)
 *
 * @param A Input [batch_size, M, K] (strided)
 * @param B Input [batch_size, K, N] (strided)
 * @param D Output [batch_size, M, N] (strided)
 * @param batch_size Number of batches
 * @param M Rows of output per batch
 * @param N Columns of output per batch
 * @param K Contraction dimension
 * @param lda Leading dimension of A (within batch)
 * @param ldb Leading dimension of B (within batch)
 * @param ldd Leading dimension of D (within batch)
 * @param stride_a Stride between batches in A
 * @param stride_b Stride between batches in B
 * @param stride_d Stride between batches in D
 * @param alpha Scale factor
 * @param beta Scale factor for D
 * @param trans_a Transpose for A
 * @param trans_b Transpose for B
 * @param dtype Data type (FP16 or BF16)
 * @param stream CUDA stream
 * @return Status code
 */
GemmStatus invokeBmm(const void* A, const void* B, void* D,
                     int batch_size, int M, int N, int K,
                     int64_t lda, int64_t ldb, int64_t ldd,
                     int64_t stride_a, int64_t stride_b, int64_t stride_d,
                     float alpha, float beta,
                     TransposeOp trans_a, TransposeOp trans_b,
                     DataType dtype,
                     cudaStream_t stream = nullptr);

/**
 * @brief Strided batched GEMM with typed interface
 */
template <typename T>
GemmStatus invokeBmmStrided(
    const T* A, const T* B, T* D,
    int batch_size, int M, int N, int K,
    int64_t stride_a, int64_t stride_b, int64_t stride_d,
    float alpha, float beta,
    TransposeOp trans_a, TransposeOp trans_b,
    cudaStream_t stream);

/**
 * @brief Batched GEMM with pointer arrays
 */
template <typename T>
GemmStatus invokeBmmArray(
    const T* const* A_array, const T* const* B_array, T* const* D_array,
    int batch_size, int M, int N, int K,
    int64_t lda, int64_t ldb, int64_t ldd,
    float alpha, float beta,
    TransposeOp trans_a, TransposeOp trans_b,
    cudaStream_t stream);

/**
 * @brief Query required workspace size for batched GEMM
 */
size_t queryBmmWorkspaceSize(int batch_size, int M, int N, int K,
                             DataType dtype,
                             const GemmConfig& config = GemmConfig());

//==============================================================================
// BMM Runner Class
//==============================================================================

class BmmRunner {
public:
    explicit BmmRunner(DataType dtype, int sm_version = -1);
    ~BmmRunner();
    
    BmmRunner(const BmmRunner&) = delete;
    BmmRunner& operator=(const BmmRunner&) = delete;
    
    size_t getWorkspaceSize(int batch, int M, int N, int K) const;
    
    GemmStatus runStrided(
        const void* A, const void* B, void* D,
        int batch, int M, int N, int K,
        int64_t stride_a, int64_t stride_b, int64_t stride_d,
        float alpha, float beta,
        void* workspace, size_t workspace_size,
        cudaStream_t stream);
    
    GemmStatus runArray(
        const void* const* A_array, const void* const* B_array,
        void* const* D_array,
        int batch, int M, int N, int K,
        float alpha, float beta,
        void* workspace, size_t workspace_size,
        cudaStream_t stream);
    
    std::vector<GemmConfig> getConfigs() const;
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

GemmConfig autoTuneBmm(int batch, int M, int N, int K, DataType dtype,
                       int num_warmup = 5, int num_iter = 10,
                       cudaStream_t stream = nullptr);

} // namespace gemm
} // namespace kernels
} // namespace oasr
