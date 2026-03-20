// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Batched Matrix Multiplication (BMM) kernel interfaces using NVIDIA CUTLASS
// Supports BF16 and FP16 precision

#pragma once

#include <cuda_runtime.h>

#include <torch/torch.h>

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
 * @param A Input [batch_size, M, K]
 * @param B Input [batch_size, N, K]
 * @param stream CUDA stream
 * @return Output [batch_size, M, N]
 */
torch::Tensor invokeBmm(const torch::Tensor& A, const torch::Tensor& B,
                        cudaStream_t stream = nullptr);

}  // namespace gemm
}  // namespace kernels
}  // namespace oasr
