// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime.h>

#include <torch/extension.h>

#include "kernels/common/types.h"

namespace oasr {
namespace kernels {

/**
 * @brief Depthwise separable 1D convolution
 *
 * Efficient implementation for Conformer-style depthwise convolutions.
 * Each input channel is convolved with its own filter.
 * Dimensions derived from tensors: batch_size=input.size(0), seq_len=input.size(1),
 * channels=input.size(2), kernel_size=weight.size(0).
 *
 * @param input Input [batch, seq_len, channels]
 * @param weight Weight [kernel_size, channels]
 * @param bias Optional bias [channels]
 * @param padding Padding size
 * @param stream CUDA stream
 * @return Output [batch, seq_len, channels]
 */
torch::Tensor invokeDepthwiseConv1D(const torch::Tensor& input, const torch::Tensor& weight,
                                    const torch::Tensor& bias, int padding, cudaStream_t stream);

/**
 * @brief Fused Depthwise 1D convolution + SiLU kernel
 *
 * Dimensions derived from tensors: batch_size=input.size(0), seq_len=input.size(1),
 * channels=input.size(2), kernel_size=weight.size(0).
 *
 * @param input Input [batch, seq_len, channels]
 * @param weight Weight [kernel_size, channels]
 * @param bias Optional bias [channels]
 * @param padding Padding size
 * @param stream CUDA stream
 * @return Output [batch, seq_len, channels]
 */
torch::Tensor invokeDepthwiseConv1DSilu(const torch::Tensor& input, const torch::Tensor& weight,
                                        const torch::Tensor& bias, int padding,
                                        cudaStream_t stream);

/**
 * @brief Pointwise (1x1) convolution
 *
 * Essentially a linear projection. Can be fused with activation.
 * Dimensions derived from tensors: batch_size=input.size(0), seq_len=input.size(1),
 * in_channels=input.size(2), out_channels=weight.size(0).
 *
 * @param input Input [batch, seq_len, in_channels]
 * @param weight Weight [out_channels, in_channels]
 * @param bias Optional bias [out_channels]
 * @param stream CUDA stream
 * @return Output [batch, seq_len, out_channels]
 */
torch::Tensor invokePointwiseConv1D(const torch::Tensor& input, const torch::Tensor& weight,
                                    const torch::Tensor& bias, cudaStream_t stream);

/**
 * @brief Pointwise (1x1) convolution with activation
 *
 * Essentially a linear projection. Can be fused with activation.
 * Dimensions derived from tensors: batch_size=input.size(0), seq_len=input.size(1),
 * in_channels=input.size(2), out_channels=weight.size(0).
 *
 * @param input Input [batch, seq_len, in_channels]
 * @param weight Weight [out_channels, in_channels]
 * @param bias Optional bias [out_channels]
 * @param activation Activation type
 * @param stream CUDA stream
 * @return Output [batch, seq_len, out_channels]
 */
torch::Tensor invokePointwiseConv1DActivation(const torch::Tensor& input,
                                              const torch::Tensor& weight,
                                              const torch::Tensor& bias, ActivationType activation,
                                              cudaStream_t stream);
/**
 * @brief Causal convolution with state management (streaming)
 *
 * Dimensions derived from tensors: batch_size=input.size(0), chunk_len=input.size(1),
 * channels=input.size(2), kernel_size=weight.size(-1).
 *
 * @param input Current input chunk [batch, chunk_len, channels]
 * @param weight Convolution weight
 * @param bias Optional bias
 * @param stream CUDA stream
 * @return Output [batch, chunk_len, channels]
 */
torch::Tensor invokeCausalConv1D(const torch::Tensor& input, void* state_buffer,
                                 const torch::Tensor& weight, const torch::Tensor& bias,
                                 cudaStream_t stream);

// GLU (Gated Linear Unit) activation
// output = input[:, :half] * sigmoid(input[:, half:])
// Dimensions derived: batch_size=input.size(0), seq_len=input.size(1), channels=input.size(2)/2
//
// @param input Input [batch, seq_len, channels]
// @param stream CUDA stream
// @return Output [batch, seq_len, channels]
torch::Tensor invokeGLU(const torch::Tensor& input, cudaStream_t stream);

// Swish activation: x * sigmoid(x)
// Dimensions derived: batch_size=input.size(0), seq_len=input.size(1), channels=input.size(2)
// @param input Input [batch, seq_len, channels]
// @param stream CUDA stream
// @return Output [batch, seq_len, channels]
torch::Tensor invokeSwish(const torch::Tensor& input, cudaStream_t stream);

// Fused BatchNorm + Swish (inference mode)
// Dimensions derived: batch_size=input.size(0), seq_len=input.size(1), channels=input.size(2)
// @param input Input [batch, seq_len, channels]
// @param gamma Scale tensor [channels]
// @param beta Bias tensor [channels]
// @param running_mean Running mean tensor [channels]
// @param running_var Running variance tensor [channels]
// @param eps Epsilon for numerical stability
// @param stream CUDA stream
// @return Output [batch, seq_len, channels]
torch::Tensor invokeBatchNormSwish(const torch::Tensor& input, const torch::Tensor& gamma,
                                   const torch::Tensor& beta, const torch::Tensor& running_mean,
                                   const torch::Tensor& running_var, float eps,
                                   cudaStream_t stream);

}  // namespace kernels
}  // namespace oasr
