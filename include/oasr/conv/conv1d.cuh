// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Pure CUDA conv1d kernels — no framework dependencies.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>

#include <oasr/common/math.h>
#include <oasr/common/types.h>
#include <oasr/common/vec_dtypes.h>

namespace oasr {
namespace conv {

// =============================================================================
// Depthwise 1D Convolution Kernel
// =============================================================================

template <typename T>
__global__ void depthwiseConv1DKernel(const T* __restrict__ input,   // [batch, seq_len, channels]
                                      const T* __restrict__ weight,  // [kernel_size, channels]
                                      const T* __restrict__ bias,    // [channels] or nullptr
                                      T* __restrict__ output,        // [batch, seq_len, channels]
                                      int batch_size, int seq_len, int channels, int kernel_size,
                                      int padding) {
    int c_id = threadIdx.x;
    int s_id = blockIdx.x;
    int b_id = blockIdx.y;
    int o_id = (blockIdx.y * gridDim.x + blockIdx.x) * channels + c_id;

    int s_start = s_id - padding;
    int s_end = min(s_start + kernel_size, seq_len);
    s_start = max(s_start, 0);

    int k_start = max(padding - s_id, 0);

    input += b_id * seq_len * channels + c_id;
    weight += c_id;

    float val = 0.0f;
    for (int i = s_start; i < s_end; i++) {
        val += (float)input[i * channels] * (float)weight[(k_start + i - s_start) * channels];
    }

    if (bias != nullptr) {
        val += (float)bias[c_id];
    }

    output[o_id] = (T)val;
}

// =============================================================================
// Depthwise 1D Convolution Kernel (vectorized)
// =============================================================================
// Each thread processes VecSize channels using 128-bit vector loads/stores.
// Requires channels % VecSize == 0.
//
// Grid:  (out_len, batch_size)
// Block: (channels / VecSize)

template <typename T, int VecSize>
__global__ void depthwiseConv1DVecKernel(const T* __restrict__ input,  // [batch, seq_len, channels]
                                         const T* __restrict__ weight,  // [kernel_size, channels]
                                         const T* __restrict__ bias,    // [channels] or nullptr
                                         T* __restrict__ output,  // [batch, out_len, channels]
                                         int batch_size, int seq_len, int channels, int kernel_size,
                                         int padding) {
    // Thread ID in the vectorized channel dimension
    const int vec_id = threadIdx.x;  // which vector chunk [0, channels/VecSize)
    const int s_id = blockIdx.x;     // output sequence position
    const int b_id = blockIdx.y;     // batch index

    const int c_offset = vec_id * VecSize;  // starting channel for this thread

    // Compute the valid input range for this output position
    int s_start = s_id - padding;
    int s_end = min(s_start + kernel_size, seq_len);
    s_start = max(s_start, 0);

    int k_start = max(padding - s_id, 0);

    // Pointers for this batch element, offset to the vector chunk
    const T* input_base = input + b_id * seq_len * channels + c_offset;
    const T* weight_base = weight + c_offset;

    // Accumulate in float for numerical stability
    float acc[VecSize];
#pragma unroll
    for (int v = 0; v < VecSize; v++) {
        acc[v] = 0.0f;
    }

    // Main convolution loop
    for (int i = s_start; i < s_end; i++) {
        Vec<T, VecSize> in_vec;
        in_vec.load(input_base + i * channels);

        Vec<T, VecSize> w_vec;
        w_vec.load(weight_base + (k_start + i - s_start) * channels);

#pragma unroll
        for (int v = 0; v < VecSize; v++) {
            acc[v] += static_cast<float>(in_vec[v]) * static_cast<float>(w_vec[v]);
        }
    }

    // Add bias
    if (bias != nullptr) {
        Vec<T, VecSize> bias_vec;
        bias_vec.load(bias + c_offset);
#pragma unroll
        for (int v = 0; v < VecSize; v++) {
            acc[v] += static_cast<float>(bias_vec[v]);
        }
    }

    // Store result
    const int out_offset = (b_id * gridDim.x + s_id) * channels + c_offset;
    Vec<T, VecSize> out_vec;
#pragma unroll
    for (int v = 0; v < VecSize; v++) {
        out_vec[v] = static_cast<T>(acc[v]);
    }
    out_vec.store(output + out_offset);
}

// =============================================================================
// Fused Depthwise 1D Convolution + SiLU Kernel
// =============================================================================

template <typename T>
__global__ void depthwiseConv1DSiluKernel(
    const T* __restrict__ input,   // [batch, seq_len, channels]
    const T* __restrict__ weight,  // [kernel_size, channels]
    const T* __restrict__ bias,    // [channels] or nullptr
    T* __restrict__ output,        // [batch, seq_len, channels]
    int batch_size, int seq_len, int channels, int kernel_size, int padding) {
    int c_id = threadIdx.x;
    int s_id = blockIdx.x;
    int b_id = blockIdx.y;
    int o_id = (blockIdx.y * gridDim.x + blockIdx.x) * channels + c_id;

    int s_start = s_id - padding;
    int s_end = min(s_start + kernel_size, seq_len);
    s_start = max(s_start, 0);

    int k_start = max(padding - s_id, 0);

    input += b_id * seq_len * channels + c_id;
    weight += c_id;

    float val = 0.0f;
    for (int i = s_start; i < s_end; i++) {
        val += (float)input[i * channels] * (float)weight[(k_start + i - s_start) * channels];
    }

    if (bias != nullptr) {
        val += (float)bias[c_id];
    }

    val = oasr::swish(val);

    output[o_id] = (T)val;
}

// =============================================================================
// Depthwise 1D Convolution + SiLU Kernel (vectorized)
// =============================================================================
// Each thread processes VecSize channels using 128-bit vector loads/stores.
// Requires channels % VecSize == 0.
//
// Grid:  (out_len, batch_size)
// Block: (channels / VecSize)

template <typename T, int VecSize>
__global__ void depthwiseConv1DVecSiluKernel(
    const T* __restrict__ input,   // [batch, seq_len, channels]
    const T* __restrict__ weight,  // [kernel_size, channels]
    const T* __restrict__ bias,    // [channels] or nullptr
    T* __restrict__ output,        // [batch, out_len, channels]
    int batch_size, int seq_len, int channels, int kernel_size, int padding) {
    // Thread ID in the vectorized channel dimension
    const int vec_id = threadIdx.x;  // which vector chunk [0, channels/VecSize)
    const int s_id = blockIdx.x;     // output sequence position
    const int b_id = blockIdx.y;     // batch index

    const int c_offset = vec_id * VecSize;  // starting channel for this thread

    // Compute the valid input range for this output position
    int s_start = s_id - padding;
    int s_end = min(s_start + kernel_size, seq_len);
    s_start = max(s_start, 0);

    int k_start = max(padding - s_id, 0);

    // Pointers for this batch element, offset to the vector chunk
    const T* input_base = input + b_id * seq_len * channels + c_offset;
    const T* weight_base = weight + c_offset;

    // Accumulate in float for numerical stability
    float acc[VecSize];
#pragma unroll
    for (int v = 0; v < VecSize; v++) {
        acc[v] = 0.0f;
    }

    // Main convolution loop
    for (int i = s_start; i < s_end; i++) {
        Vec<T, VecSize> in_vec;
        in_vec.load(input_base + i * channels);

        Vec<T, VecSize> w_vec;
        w_vec.load(weight_base + (k_start + i - s_start) * channels);

#pragma unroll
        for (int v = 0; v < VecSize; v++) {
            acc[v] += static_cast<float>(in_vec[v]) * static_cast<float>(w_vec[v]);
        }
    }

    // Add bias
    if (bias != nullptr) {
        Vec<T, VecSize> bias_vec;
        bias_vec.load(bias + c_offset);
#pragma unroll
        for (int v = 0; v < VecSize; v++) {
            acc[v] += static_cast<float>(bias_vec[v]);
        }
    }

    // Apply SiLU activation
    for (int v = 0; v < VecSize; v++) {
        acc[v] = oasr::swish(acc[v]);
    }

    // Store result
    const int out_offset = (b_id * gridDim.x + s_id) * channels + c_offset;
    Vec<T, VecSize> out_vec;
#pragma unroll
    for (int v = 0; v < VecSize; v++) {
        out_vec[v] = static_cast<T>(acc[v]);
    }
    out_vec.store(output + out_offset);
}

// =============================================================================
// Causal Conv1D with State Kernel
// =============================================================================

template <typename T>
__global__ void causalConv1DKernel(const T* __restrict__ input,  // [batch, chunk_len, channels]
                                   T* __restrict__ state,        // [batch, kernel_size-1, channels]
                                   const T* __restrict__ weight,  // [channels, 1, kernel_size]
                                   const T* __restrict__ bias,    // [channels] or nullptr
                                   T* __restrict__ output,        // [batch, chunk_len, channels]
                                   int batch_size, int chunk_len, int channels, int kernel_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * chunk_len * channels;

    if (idx >= total_elements)
        return;

    int c = idx % channels;
    int t = (idx / channels) % chunk_len;
    int b = idx / (channels * chunk_len);

    int state_len = kernel_size - 1;

    float sum = 0.0f;

    // Compute convolution using state and current input
    for (int k = 0; k < kernel_size; k++) {
        int input_pos = t - (kernel_size - 1) + k;
        float val;

        if (input_pos < 0) {
            // Read from state buffer
            int state_pos = state_len + input_pos;  // Maps -state_len to 0, etc.
            int state_idx = b * state_len * channels + state_pos * channels + c;
            val = static_cast<float>(state[state_idx]);
        } else {
            // Read from current input
            int input_idx = b * chunk_len * channels + input_pos * channels + c;
            val = static_cast<float>(input[input_idx]);
        }

        int weight_idx = c * kernel_size + k;
        sum += val * static_cast<float>(weight[weight_idx]);
    }

    if (bias != nullptr) {
        sum += static_cast<float>(bias[c]);
    }

    output[idx] = static_cast<T>(sum);
}

// Update state buffer after processing chunk
template <typename T>
__global__ void updateConvStateKernel(const T* __restrict__ input,  // [batch, chunk_len, channels]
                                      T* __restrict__ state,        // [batch, state_len, channels]
                                      int batch_size, int chunk_len, int channels, int state_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * state_len * channels;

    if (idx >= total_elements)
        return;

    int c = idx % channels;
    int s = (idx / channels) % state_len;
    int b = idx / (channels * state_len);

    // New state comes from:
    // - Old state shifted left (if chunk_len < state_len)
    // - Or entirely from new input (if chunk_len >= state_len)
    int source_pos;
    if (chunk_len >= state_len) {
        // Take from input: last state_len positions
        source_pos = chunk_len - state_len + s;
        int input_idx = b * chunk_len * channels + source_pos * channels + c;
        state[idx] = input[input_idx];
    } else {
        // Mix: shift old state and add new input
        int shift = state_len - chunk_len;
        if (s < shift) {
            // From old state
            int old_state_idx = b * state_len * channels + (s + chunk_len) * channels + c;
            state[idx] = state[old_state_idx];
        } else {
            // From new input
            source_pos = s - shift;
            int input_idx = b * chunk_len * channels + source_pos * channels + c;
            state[idx] = input[input_idx];
        }
    }
}

// =============================================================================
// Typed Launchers — raw pointer interface, returns cudaError_t
// =============================================================================

/**
 * @brief Depthwise separable 1D convolution
 *
 * Efficient implementation for Conformer-style depthwise convolutions.
 * Each input channel is convolved with its own filter.
 * Automatically selects vectorized kernel path when channels are aligned.
 *
 * @param input   Input [batch, seq_len, channels]
 * @param weight  Weight [kernel_size, channels]
 * @param bias    Optional bias [channels], nullptr to skip
 * @param output  Output [batch, out_len, channels] where out_len = seq_len + 2*padding - kernel_size + 1
 * @param batch_size  Batch dimension
 * @param seq_len     Sequence length
 * @param channels    Number of channels
 * @param kernel_size Convolution kernel size
 * @param padding     Padding size
 * @param stream      CUDA stream
 */
template <typename T>
cudaError_t DepthwiseConv1D(const T* input, const T* weight, const T* bias, T* output,
                            int batch_size, int seq_len, int channels, int kernel_size, int padding,
                            cudaStream_t stream) {
    const int out_len = seq_len + 2 * padding - kernel_size + 1;
    dim3 grid_size(out_len, batch_size);

    constexpr int kVecSize = VecTypeTrait<T>::VecSize;

    // Use vectorized kernel when channels are aligned to VecSize
    // and the thread count fits within hardware limits
    if (channels % kVecSize == 0 && (channels / kVecSize) <= 1024) {
        dim3 block_size(channels / kVecSize);
        depthwiseConv1DVecKernel<T, kVecSize><<<grid_size, block_size, 0, stream>>>(
            input, weight, bias, output, batch_size, seq_len, channels, kernel_size, padding);
    } else {
        dim3 block_size(channels);
        depthwiseConv1DKernel<T><<<grid_size, block_size, 0, stream>>>(
            input, weight, bias, output, batch_size, seq_len, channels, kernel_size, padding);
    }

    return cudaGetLastError();
}

/**
 * @brief Fused Depthwise 1D convolution + SiLU activation
 *
 * Automatically selects vectorized kernel path when channels are aligned.
 *
 * @param input   Input [batch, seq_len, channels]
 * @param weight  Weight [kernel_size, channels]
 * @param bias    Optional bias [channels], nullptr to skip
 * @param output  Output [batch, out_len, channels] where out_len = seq_len + 2*padding - kernel_size + 1
 * @param batch_size  Batch dimension
 * @param seq_len     Sequence length
 * @param channels    Number of channels
 * @param kernel_size Convolution kernel size
 * @param padding     Padding size
 * @param stream      CUDA stream
 */
template <typename T>
cudaError_t DepthwiseConv1DSilu(const T* input, const T* weight, const T* bias, T* output,
                                int batch_size, int seq_len, int channels, int kernel_size,
                                int padding, cudaStream_t stream) {
    const int out_len = seq_len + 2 * padding - kernel_size + 1;
    dim3 grid_size(out_len, batch_size);

    constexpr int kVecSize = VecTypeTrait<T>::VecSize;

    // Use vectorized kernel when channels are aligned to VecSize
    // and the thread count fits within hardware limits
    if (channels % kVecSize == 0 && (channels / kVecSize) <= 1024) {
        dim3 block_size(channels / kVecSize);
        depthwiseConv1DVecSiluKernel<T, kVecSize><<<grid_size, block_size, 0, stream>>>(
            input, weight, bias, output, batch_size, seq_len, channels, kernel_size, padding);
    } else {
        dim3 block_size(channels);
        depthwiseConv1DSiluKernel<T><<<grid_size, block_size, 0, stream>>>(
            input, weight, bias, output, batch_size, seq_len, channels, kernel_size, padding);
    }

    return cudaGetLastError();
}

/**
 * @brief Pointwise (1x1) convolution
 *
 * Essentially a GEMM: output = input * weight^T + bias.
 * Input is reshaped to [batch*seq_len, in_channels] and multiplied by weight [out_channels, in_channels].
 *
 * @param input        Input [batch * seq_len, in_channels] (pre-reshaped)
 * @param weight       Weight [out_channels, in_channels]
 * @param bias         Optional bias [out_channels], nullptr to skip
 * @param output       Output [batch * seq_len, out_channels]
 * @param batch_size   Batch dimension
 * @param seq_len      Sequence length
 * @param in_channels  Number of input channels
 * @param out_channels Number of output channels
 * @param stream       CUDA stream
 */
template <typename T>
cudaError_t PointwiseConv1D(const T* input, const T* weight, const T* bias, T* output,
                            int batch_size, int seq_len, int in_channels, int out_channels,
                            cudaStream_t stream);

/**
 * @brief Pointwise (1x1) convolution with fused activation
 *
 * Essentially a GEMM with fused activation: output = activation(input * weight^T + bias).
 *
 * @param input        Input [batch * seq_len, in_channels] (pre-reshaped)
 * @param weight       Weight [out_channels, in_channels]
 * @param bias         Optional bias [out_channels], nullptr to skip
 * @param output       Output [batch * seq_len, out_channels]
 * @param batch_size   Batch dimension
 * @param seq_len      Sequence length
 * @param in_channels  Number of input channels
 * @param out_channels Number of output channels
 * @param activation   Activation type (RELU, GELU, or SWISH)
 * @param stream       CUDA stream
 */
template <typename T>
cudaError_t PointwiseConv1DActivation(const T* input, const T* weight, const T* bias, T* output,
                                      int batch_size, int seq_len, int in_channels,
                                      int out_channels, ActivationType activation,
                                      cudaStream_t stream);

/**
 * @brief Causal convolution with state management (streaming)
 *
 * Performs causal conv1d using a rolling state buffer for streaming inference.
 * After computing the output, the state buffer is automatically updated with
 * the latest input positions.
 *
 * @param input       Current input chunk [batch, chunk_len, channels]
 * @param state       State buffer [batch, kernel_size-1, channels] (read/write)
 * @param weight      Convolution weight [channels, 1, kernel_size]
 * @param bias        Optional bias [channels], nullptr to skip
 * @param output      Output [batch, chunk_len, channels]
 * @param batch_size  Batch dimension
 * @param chunk_len   Current chunk length
 * @param channels    Number of channels
 * @param kernel_size Convolution kernel size
 * @param stream      CUDA stream
 */
template <typename T>
cudaError_t CausalConv1D(const T* input, T* state, const T* weight, const T* bias, T* output,
                         int batch_size, int chunk_len, int channels, int kernel_size,
                         cudaStream_t stream) {
    int total_elements = batch_size * chunk_len * channels;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    int state_len = kernel_size - 1;

    causalConv1DKernel<T><<<grid_size, block_size, 0, stream>>>(
        input, state, weight, bias, output, batch_size, chunk_len, channels, kernel_size);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return err;
    }

    // Update state buffer
    int state_elements = batch_size * state_len * channels;
    int state_grid = (state_elements + block_size - 1) / block_size;
    updateConvStateKernel<T><<<state_grid, block_size, 0, stream>>>(
        input, state, batch_size, chunk_len, channels, state_len);

    return cudaGetLastError();
}

}  // namespace conv
}  // namespace oasr
