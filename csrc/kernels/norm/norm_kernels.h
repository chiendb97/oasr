// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/types.h"
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace oasr {
namespace kernels {

/**
 * @brief Layer normalization kernel
 * 
 * Computes: output = (input - mean) / sqrt(var + eps) * gamma + beta
 * where mean and var are computed over the last dimension.
 * 
 * @param input Input tensor [batch, seq_len, hidden_size]
 * @param output Output tensor [batch, seq_len, hidden_size]
 * @param weight Scale parameter [hidden_size]
 * @param bias Offset parameter [hidden_size]
 * @param eps Epsilon for numerical stability
 * @param stream CUDA stream
 */
torch::Tensor invokeLayerNorm(const torch::Tensor& input,
                     const torch::Tensor& weight, const torch::Tensor& bias,
                     float eps, cudaStream_t stream);

/**
 * @brief RMS normalization kernel
 * 
 * Computes: output = input / sqrt(mean(input^2) + eps) * gamma
 * Used in some modern architectures (LLaMA, etc.)
 * 
 * @param input Input tensor
 * @param output Output tensor
 * @param weight Scale parameter
 * @param bias Offset parameter
 * @param eps Epsilon for numerical stability
 * @param stream CUDA stream
 */
torch::Tensor invokeRMSNorm(const torch::Tensor& input,
                   const torch::Tensor& weight, const torch::Tensor& bias,
                   float eps, cudaStream_t stream);

/**
 * @brief Batch normalization kernel (inference mode)
 * 
 * Computes: output = (input - running_mean) / sqrt(running_var + eps) * gamma + beta
 * 
 * Used in Conformer convolution module.
 * 
 * @param input Input tensor [batch, seq_len, channels]
 * @param output Output tensor
 * @param weight Scale parameter [channels]
 * @param bias Offset parameter [channels]
 * @param running_mean Running mean [channels]
 * @param running_var Running variance [channels]
 * @param eps Epsilon for numerical stability
 * @param stream CUDA stream
 */
torch::Tensor invokeBatchNorm1D(const torch::Tensor& input,
                       const torch::Tensor& weight, const torch::Tensor& bias,
                       const torch::Tensor& running_mean, const torch::Tensor& running_var,
                       float eps, cudaStream_t stream);

/**
 * @brief Group normalization kernel
 * 
 * @param input Input tensor [batch, seq_len, channels]
 * @param output Output tensor
 * @param weight Scale [channels]
 * @param bias Offset [channels]
 * @param num_groups Number of groups
 * @param eps Epsilon for numerical stability
 * @param stream CUDA stream
 */
torch::Tensor invokeGroupNorm(const torch::Tensor& input,
                     const torch::Tensor& weight, const torch::Tensor& bias,
                     int num_groups,
                     float eps, cudaStream_t stream);


/**
 * @brief Fused Add + LayerNorm kernel
 * 
 * Computes: output = LayerNorm(input + residual)
 * Common pattern after attention and FFN.
 */
torch::Tensor invokeAddLayerNorm(const torch::Tensor& input, const torch::Tensor& residual,
                        const torch::Tensor& weight, const torch::Tensor& bias,
                        float eps, cudaStream_t stream);

// Template specializations
template <typename T>
void invokeLayerNormTyped(const torch::Tensor& input, torch::Tensor& output,
                          const torch::Tensor& weight, const torch::Tensor& bias,
                          float eps, cudaStream_t stream);

template <typename T>
void invokeRMSNormTyped(const torch::Tensor& input, torch::Tensor& output,
                        const torch::Tensor& weight, const torch::Tensor& bias,
                        float eps, cudaStream_t stream);

} // namespace kernels
} // namespace oasr
