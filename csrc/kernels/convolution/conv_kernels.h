// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "conv_params.h"
#include "common/tensor.h"

namespace oasr {
namespace kernels {

/**
 * @brief 1D convolution kernel
 * 
 * Supports standard, depthwise, and pointwise convolutions.
 * Optimized implementations selected based on parameters.
 * 
 * @param params Convolution parameters
 */
void invokeConv1D(const Conv1DParams& params);

/**
 * @brief Depthwise separable 1D convolution
 * 
 * Efficient implementation for Conformer-style depthwise convolutions.
 * Each input channel is convolved with its own filter.
 * 
 * @param input Input [batch, seq_len, channels]
 * @param weight Weight [channels, 1, kernel_size]
 * @param bias Optional bias [channels]
 * @param output Output [batch, seq_len, channels]
 * @param params Additional parameters
 */
void invokeDepthwiseConv1D(const void* input, const void* weight, const void* bias,
                           void* output, int batch_size, int seq_len, int channels,
                           int kernel_size, int padding, bool is_causal,
                           DataType dtype, cudaStream_t stream);

/**
 * @brief Pointwise (1x1) convolution
 * 
 * Essentially a linear projection, but kept as conv for consistency.
 * Can be fused with activation.
 * 
 * @param input Input [batch, seq_len, in_channels]
 * @param weight Weight [out_channels, in_channels]
 * @param bias Optional bias [out_channels]
 * @param output Output [batch, seq_len, out_channels]
 * @param activation Optional fused activation
 */
void invokePointwiseConv1D(const void* input, const void* weight, const void* bias,
                           void* output, int batch_size, int seq_len,
                           int in_channels, int out_channels,
                           ActivationType activation, bool fuse_activation,
                           DataType dtype, cudaStream_t stream);

/**
 * @brief Complete Conformer convolution module
 * 
 * Fused implementation of the full Conformer conv module:
 * - Pointwise conv (1 -> 2 channels with GLU)
 * - Depthwise conv
 * - BatchNorm + Swish
 * - Pointwise conv (back to original channels)
 * 
 * @param params Conformer conv parameters
 */
void invokeConformerConvModule(const ConformerConvParams& params);

/**
 * @brief Causal convolution with state management
 * 
 * For streaming inference, maintains state from previous chunks.
 * 
 * @param input Current input chunk [batch, chunk_len, channels]
 * @param state Previous state (updated in-place)
 * @param weight Convolution weight
 * @param bias Optional bias
 * @param output Output [batch, chunk_len, channels]
 * @param kernel_size Convolution kernel size
 * @param dtype Data type
 * @param stream CUDA stream
 */
void invokeCausalConv1D(const void* input, ConvState& state,
                        const void* weight, const void* bias,
                        void* output, int batch_size, int chunk_len, int channels,
                        int kernel_size, DataType dtype, cudaStream_t stream);

/**
 * @brief Initialize convolution state for streaming
 * 
 * @param state State to initialize
 * @param batch_size Batch size
 * @param kernel_size Kernel size
 * @param channels Number of channels
 * @param dtype Data type
 */
void initConvState(ConvState& state, int batch_size, int kernel_size, int channels,
                   DataType dtype);

/**
 * @brief Reset convolution state (zero out buffer)
 * 
 * @param state State to reset
 * @param stream CUDA stream
 */
void resetConvState(ConvState& state, cudaStream_t stream);

// GLU (Gated Linear Unit) activation
// output = input[:, :half] * sigmoid(input[:, half:])
void invokeGLU(const void* input, void* output,
               int batch_size, int seq_len, int channels,
               DataType dtype, cudaStream_t stream);

// Swish activation: x * sigmoid(x)
void invokeSwish(const void* input, void* output,
                 int batch_size, int seq_len, int channels,
                 DataType dtype, cudaStream_t stream);

// Fused BatchNorm + Swish (inference mode)
void invokeBatchNormSwish(const void* input, void* output,
                          const void* gamma, const void* beta,
                          const void* running_mean, const void* running_var,
                          int batch_size, int seq_len, int channels,
                          float eps, DataType dtype, cudaStream_t stream);

// Template specializations
template <typename T>
void invokeConv1DTyped(const Conv1DParams& params);

template <typename T>
void invokeDepthwiseConv1DTyped(const void* input, const void* weight, const void* bias,
                                void* output, int batch_size, int seq_len, int channels,
                                int kernel_size, int padding, bool is_causal,
                                cudaStream_t stream);

} // namespace kernels
} // namespace oasr
