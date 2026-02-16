// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/types.h"
#include <cuda_runtime.h>

namespace oasr {
namespace kernels {

/**
 * @brief 1D convolution kernel
 *
 * Supports standard, depthwise, and pointwise convolutions.
 * Optimized implementations selected based on parameters.
 */
void invokeConv1D(const void* input, void* output, const void* weight, const void* bias,
                  int batch_size, int seq_len, int in_channels, int out_channels,
                  int kernel_size, int stride, int padding, int dilation, int groups,
                  ConvType conv_type, DataType dtype, bool channels_last, bool is_causal,
                  ActivationType activation, bool fuse_activation, cudaStream_t stream);

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
 * Essentially a linear projection. Can be fused with activation.
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
 * @brief Causal convolution with state management (streaming)
 *
 * @param input Current input chunk [batch, chunk_len, channels]
 * @param state_buffer State buffer (updated in-place) [batch, kernel_size-1, channels]
 * @param weight Convolution weight
 * @param bias Optional bias
 * @param output Output [batch, chunk_len, channels]
 */
void invokeCausalConv1D(const void* input, void* state_buffer,
                        const void* weight, const void* bias,
                        void* output, int batch_size, int chunk_len, int channels,
                        int kernel_size, DataType dtype, cudaStream_t stream);

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
void invokeDepthwiseConv1DTyped(const void* input, const void* weight, const void* bias,
                                void* output, int batch_size, int seq_len, int channels,
                                int kernel_size, int padding, bool is_causal,
                                cudaStream_t stream);

} // namespace kernels
} // namespace oasr
