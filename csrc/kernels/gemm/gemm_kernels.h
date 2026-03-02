// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Standard GEMM kernel interfaces using NVIDIA CUTLASS
// Supports BF16 and FP16 precision
// Reference: https://github.com/flashinfer-ai/flashinfer/tree/main/include/flashinfer/gemm

#pragma once

#include "gemm_configs.h"
#include "gemm_utils.h"
#include "common/types.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <cstdint>
#include <vector>
#include <memory>

namespace oasr {
namespace kernels {
namespace gemm {

//==============================================================================
// Standard GEMM Interface
//==============================================================================

/**
 * @brief Execute GEMM operation
 *
 * Computes: D = A @ B + C
 * When C is undefined, computes: D = A @ B
 *
 * Dimensions are derived from tensor shapes:
 *   K = A.size(-1), M = A.numel() / K, N = B.size(0)
 *
 * @param A Input tensor [M, K] or [batch, M', K]
 * @param B Input tensor [N, K]
 * @param C Optional bias tensor [M, N] (undefined tensor to skip)
 * @param stream CUDA stream
 * @return Output tensor [M, N]
 */
torch::Tensor invokeGemm(const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& C,
                      cudaStream_t stream = nullptr);

//==============================================================================
// GEMM with Fused Operations
//==============================================================================

/**
 * @brief GEMM with fused activation
 * 
 * Computes: D = activation(A @ B + C)
 *
 * @param A Input tensor [M, K] or [batch, M', K]
 * @param B Input tensor [N, K]
 * @param C Optional bias tensor [M, N] (undefined tensor to skip)
 * @param activation Activation type
 * @param stream CUDA stream
 * @return Output tensor [M, N]
 */
torch::Tensor invokeGemmActivation(
    const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& C,
    ActivationType activation,
    cudaStream_t stream = nullptr);

//==============================================================================
// Auto-Tuning
//==============================================================================

/**
 * @brief Profile and select best GEMM configuration
 * 
 * @param M, N, K Problem dimensions
 * @param dtype Data type
 * @param num_warmup Number of warmup iterations
 * @param num_iter Number of timing iterations
 * @return Best configuration
 */
GemmConfig autoTuneGemm(int M, int N, int K, DataType dtype,
                        int num_warmup = 5, int num_iter = 10,
                        cudaStream_t stream = nullptr);

} // namespace gemm
} // namespace kernels
} // namespace oasr
