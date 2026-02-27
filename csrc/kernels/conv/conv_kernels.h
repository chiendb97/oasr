// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/types.h"
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace oasr {
namespace kernels {

/**
 * @brief 1D convolution kernel
 *
 * Supports standard, depthwise, and pointwise convolutions.
 * Optimized implementations selected based on parameters.
 * Dimensions derived from tensors: batch_size=input.size(0), seq_len=input.size(1),
 * in_channels=input.size(2), out_channels=weight.size(0), kernel_size=weight.size(-1).
 */
void invokeConv1D(const torch::Tensor& input, torch::Tensor& output,
                  const torch::Tensor& weight, const torch::Tensor& bias,
                  int stride, int padding, int dilation, int groups,
                  ConvType conv_type, DataType dtype, bool channels_last, bool is_causal,
                  ActivationType activation, bool fuse_activation, cudaStream_t stream);

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
 * @param output Output [batch, out_seq_len, channels]
 * @param padding Padding size
 * @param dtype Data type
 * @param stream CUDA stream
 */
void invokeDepthwiseConv1D(const torch::Tensor& input, const torch::Tensor& weight,
                           const torch::Tensor& bias, torch::Tensor& output,
                           int padding, DataType dtype, cudaStream_t stream);

/**
 * @brief Fused Depthwise 1D convolution + SiLU kernel
 *
 * Dimensions derived from tensors: batch_size=input.size(0), seq_len=input.size(1),
 * channels=input.size(2), kernel_size=weight.size(0).
 *
 * @param input Input [batch, seq_len, channels]
 * @param weight Weight [kernel_size, channels]
 * @param bias Optional bias [channels]
 * @param output Output [batch, out_seq_len, channels]
 * @param padding Padding size
 * @param dtype Data type
 * @param stream CUDA stream
 */
void invokeDepthwiseConv1DSilu(const torch::Tensor& input, const torch::Tensor& weight,
                               const torch::Tensor& bias, torch::Tensor& output,
                               int padding, DataType dtype, cudaStream_t stream);

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
 * @param output Output [batch, seq_len, out_channels]
 * @param activation Optional fused activation
 * @param fuse_activation Whether to fuse activation
 * @param dtype Data type
 * @param stream CUDA stream
 */
void invokePointwiseConv1D(const torch::Tensor& input, const torch::Tensor& weight,
                           const torch::Tensor& bias, torch::Tensor& output,
                           ActivationType activation, bool fuse_activation,
                           DataType dtype, cudaStream_t stream);

/**
 * @brief Causal convolution with state management (streaming)
 *
 * Dimensions derived from tensors: batch_size=input.size(0), chunk_len=input.size(1),
 * channels=input.size(2), kernel_size=weight.size(-1).
 *
 * @param input Current input chunk [batch, chunk_len, channels]
 * @param state_buffer State buffer (updated in-place) [batch, kernel_size-1, channels]
 * @param weight Convolution weight
 * @param bias Optional bias
 * @param output Output [batch, chunk_len, channels]
 * @param dtype Data type
 * @param stream CUDA stream
 */
void invokeCausalConv1D(const torch::Tensor& input, void* state_buffer,
                        const torch::Tensor& weight, const torch::Tensor& bias,
                        torch::Tensor& output, DataType dtype, cudaStream_t stream);

/**
 * @brief Allocate and initialize convolution state buffer for streaming.
 * Caller must cudaFree the returned buffer when done.
 */
void* initConvState(int batch_size, int kernel_size, int channels, DataType dtype);

/**
 * @brief Reset convolution state (zero out buffer)
 */
void resetConvState(void* state_buffer, int batch_size, int kernel_size, int channels,
                    DataType dtype, cudaStream_t stream);

/**
 * @brief Free convolution state buffer allocated by initConvState
 */
void freeConvState(void* state_buffer);

// GLU (Gated Linear Unit) activation
// output = input[:, :half] * sigmoid(input[:, half:])
// Dimensions derived: batch_size=input.size(0), seq_len=input.size(1), channels=input.size(2)/2
void invokeGLU(const torch::Tensor& input, torch::Tensor& output,
               DataType dtype, cudaStream_t stream);

// Swish activation: x * sigmoid(x)
// Dimensions derived: batch_size=input.size(0), seq_len=input.size(1), channels=input.size(2)
void invokeSwish(const torch::Tensor& input, torch::Tensor& output,
                 DataType dtype, cudaStream_t stream);

// Fused BatchNorm + Swish (inference mode)
// Dimensions derived: batch_size=input.size(0), seq_len=input.size(1), channels=input.size(2)
void invokeBatchNormSwish(const torch::Tensor& input, torch::Tensor& output,
                          const torch::Tensor& gamma, const torch::Tensor& beta,
                          const torch::Tensor& running_mean, const torch::Tensor& running_var,
                          float eps, DataType dtype, cudaStream_t stream);

// Template specializations
template <typename T>
void invokeDepthwiseConv1DTyped(const torch::Tensor& input, const torch::Tensor& weight,
                                const torch::Tensor& bias, torch::Tensor& output,
                                int padding, cudaStream_t stream);

template <typename T>
void invokeDepthwiseConv1DSiluTyped(const torch::Tensor& input, const torch::Tensor& weight,
                                    const torch::Tensor& bias, torch::Tensor& output,
                                    int padding, cudaStream_t stream);

} // namespace kernels
} // namespace oasr
