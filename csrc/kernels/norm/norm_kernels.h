// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/types.h"
#include <cuda_runtime.h>

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
 * @param gamma Scale parameter [hidden_size]
 * @param beta Offset parameter [hidden_size]
 * @param batch_size Batch size
 * @param seq_len Sequence length
 * @param hidden_size Hidden dimension
 * @param eps Epsilon for numerical stability
 * @param dtype Data type
 * @param stream CUDA stream
 */
void invokeLayerNorm(const void* input, void* output,
                     const void* gamma, const void* beta,
                     int batch_size, int seq_len, int hidden_size,
                     float eps, DataType dtype, cudaStream_t stream);

/**
 * @brief RMS normalization kernel
 * 
 * Computes: output = input / sqrt(mean(input^2) + eps) * gamma
 * Used in some modern architectures (LLaMA, etc.)
 * 
 * @param input Input tensor
 * @param output Output tensor
 * @param gamma Scale parameter
 * @param batch_size Batch size
 * @param seq_len Sequence length
 * @param hidden_size Hidden dimension
 * @param eps Epsilon for numerical stability
 * @param dtype Data type
 * @param stream CUDA stream
 */
void invokeRMSNorm(const void* input, void* output,
                   const void* gamma, const void* beta,
                   int batch_size, int seq_len, int hidden_size,
                   float eps, DataType dtype, cudaStream_t stream);

/**
 * @brief Batch normalization kernel (inference mode)
 * 
 * Computes: output = (input - running_mean) / sqrt(running_var + eps) * gamma + beta
 * 
 * Used in Conformer convolution module.
 * 
 * @param input Input tensor [batch, seq_len, channels]
 * @param output Output tensor
 * @param gamma Scale parameter [channels]
 * @param beta Offset parameter [channels]
 * @param running_mean Running mean [channels]
 * @param running_var Running variance [channels]
 */
void invokeBatchNorm1D(const void* input, void* output,
                       const void* gamma, const void* beta,
                       const void* running_mean, const void* running_var,
                       int batch_size, int seq_len, int channels,
                       float eps, DataType dtype, cudaStream_t stream);

/**
 * @brief Group normalization kernel
 * 
 * @param input Input tensor [batch, seq_len, channels]
 * @param output Output tensor
 * @param gamma Scale [channels]
 * @param beta Offset [channels]
 * @param num_groups Number of groups
 */
void invokeGroupNorm(const void* input, void* output,
                     const void* gamma, const void* beta,
                     int batch_size, int seq_len, int channels, int num_groups,
                     float eps, DataType dtype, cudaStream_t stream);

/**
 * @brief Fused LayerNorm + Linear kernel
 * 
 * Computes: output = LayerNorm(input) @ weight.T + bias
 * Common pattern in transformer blocks.
 */
void invokeLayerNormLinear(const void* input, void* output,
                           const void* ln_gamma, const void* ln_beta,
                           const void* weight, const void* bias,
                           int batch_size, int seq_len,
                           int in_features, int out_features,
                           float eps, DataType dtype, cudaStream_t stream);

/**
 * @brief Fused Add + LayerNorm kernel
 * 
 * Computes: output = LayerNorm(input + residual)
 * Common pattern after attention and FFN.
 */
void invokeAddLayerNorm(const void* input, const void* residual, void* output,
                        const void* gamma, const void* beta,
                        int batch_size, int seq_len, int hidden_size,
                        float eps, DataType dtype, cudaStream_t stream);

/**
 * @brief Fused Add + LayerNorm + Linear kernel
 * 
 * Computes: output = Linear(LayerNorm(input + residual))
 */
void invokeAddLayerNormLinear(const void* input, const void* residual, void* output,
                              const void* ln_gamma, const void* ln_beta,
                              const void* weight, const void* bias,
                              int batch_size, int seq_len,
                              int in_features, int out_features,
                              float eps, DataType dtype, cudaStream_t stream);

// Template specializations
template <typename T>
void invokeLayerNormTyped(const void* input, void* output,
                          const void* gamma, const void* beta,
                          int batch_size, int seq_len, int hidden_size,
                          float eps, cudaStream_t stream);

template <typename T>
void invokeRMSNormTyped(const void* input, void* output,
                        const void* gamma, const void* beta,
                        int batch_size, int seq_len, int hidden_size,
                        float eps, cudaStream_t stream);

} // namespace kernels
} // namespace oasr
