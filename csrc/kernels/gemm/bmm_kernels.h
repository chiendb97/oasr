// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Batched Matrix Multiplication (BMM) kernel interfaces using NVIDIA CUTLASS
// Supports BF16 and FP16 precision

#pragma once

#include "bmm_params.h"
#include "gemm_configs.h"
#include "gemm_utils.h"
#include "common/types.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <vector>
#include <memory>

namespace oasr {
namespace kernels {
namespace gemm {

//==============================================================================
// Batched GEMM (BMM) Interface
//==============================================================================

/**
 * @brief Execute batched GEMM operation
 * 
 * Computes: D[b] = alpha * A[b] @ B[b] + beta * C[b]
 * 
 * @param params Batched GEMM parameters
 * @return Status code
 */
GemmStatus invokeBmm(const BmmParams& params);

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
