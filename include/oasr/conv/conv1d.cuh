// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Pure CUDA conv1d kernels — no framework dependencies.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <oasr/common/math.h>
#include <oasr/common/types.h>
#include <oasr/common/vec_dtypes.h>

namespace oasr {
namespace conv {

// =============================================================================
// Depthwise 1D Convolution Kernel
// =============================================================================

template <typename T, typename MaskT, int VecSize, bool FuseMask, bool AddInput>
__global__ void depthwiseConv1DKernel(
    const T* __restrict__ input,     // [batch, seq_len, channels]
    const T* __restrict__ weight,    // [kernel_size, channels]
    const T* __restrict__ bias,      // [channels] or nullptr
    const MaskT* __restrict__ mask,  // [batch, seq_len, 1] or nullptr
    T* __restrict__ output,          // [batch, out_len, channels]
    int batch_size, int seq_len, int channels, int kernel_size, int padding_left) {
    // Thread ID in the vectorized channel dimension
    const int vec_id = threadIdx.x;  // which vector chunk [0, channels/VecSize)
    const int s_id = blockIdx.x;     // output sequence position
    const int b_id = blockIdx.y;     // batch index

    const int c_offset = vec_id * VecSize;  // starting channel for this thread

    // Compute the valid input range for this output position
    int s_start = s_id - padding_left;
    int s_end = min(s_start + kernel_size, seq_len);
    s_start = max(s_start, 0);

    int k_start = max(padding_left - s_id, 0);

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
            float in_value = static_cast<float>(in_vec[v]);
            if constexpr (FuseMask) {
                in_value *= static_cast<float>(mask[b_id * seq_len + i]);
            }
            acc[v] += in_value * static_cast<float>(w_vec[v]);
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

    // Paraformer's FSMN memory block computes
    //   (depthwise(input * mask) + input * mask) * mask.
    // Specializing both flags at launch keeps the ordinary depthwise path free
    // of mask/residual branches while folding that whole expression into this
    // one kernel for the FSMN path.
    if constexpr (AddInput) {
        Vec<T, VecSize> input_vec;
        input_vec.load(input_base + s_id * channels);
        float input_scale = 1.0f;
        if constexpr (FuseMask) {
            input_scale = static_cast<float>(mask[b_id * seq_len + s_id]);
        }
#pragma unroll
        for (int v = 0; v < VecSize; v++) {
            acc[v] += static_cast<float>(input_vec[v]) * input_scale;
        }
    }

    if constexpr (FuseMask) {
        const float output_scale = static_cast<float>(mask[b_id * seq_len + s_id]);
#pragma unroll
        for (int v = 0; v < VecSize; v++) {
            acc[v] *= output_scale;
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

template <typename T, int VecSize>
__global__ void depthwiseConv1DSiluKernel(
    const T* __restrict__ input,   // [batch, seq_len, channels]
    const T* __restrict__ weight,  // [kernel_size, channels]
    const T* __restrict__ bias,    // [channels] or nullptr
    T* __restrict__ output,        // [batch, out_len, channels]
    int batch_size, int seq_len, int channels, int kernel_size, int padding_left) {
    // Thread ID in the vectorized channel dimension
    const int vec_id = threadIdx.x;  // which vector chunk [0, channels/VecSize)
    const int s_id = blockIdx.x;     // output sequence position
    const int b_id = blockIdx.y;     // batch index

    const int c_offset = vec_id * VecSize;  // starting channel for this thread

    // Compute the valid input range for this output position
    int s_start = s_id - padding_left;
    int s_end = min(s_start + kernel_size, seq_len);
    s_start = max(s_start, 0);

    int k_start = max(padding_left - s_id, 0);

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
 * @param mask    Optional multiplicative mask [batch, seq_len, 1]
 * @param output  Output [batch, out_len, channels]
 * @param batch_size  Batch dimension
 * @param seq_len     Sequence length
 * @param channels    Number of channels
 * @param kernel_size Convolution kernel size
 * @param padding_left  Zero padding before the sequence
 * @param padding_right Zero padding after the sequence
 * @param add_input     Add the (optionally masked) input before the output mask
 * @param stream      CUDA stream
 */
template <typename T, typename MaskT = T>
cudaError_t DepthwiseConv1D(const T* input, const T* weight, const T* bias, const MaskT* mask,
                            T* output, int batch_size, int seq_len, int channels, int kernel_size,
                            int padding_left, int padding_right, bool add_input,
                            cudaStream_t stream) {
    const int out_len = seq_len + padding_left + padding_right - kernel_size + 1;
    dim3 grid_size(out_len, batch_size);

    constexpr int kVecSize = VecTypeTrait<T>::VecSize;

    // Use vectorized kernels when channels are 128-bit aligned.  Mask and
    // residual modes are compile-time specializations so existing Conformer /
    // Zipformer calls do not pay for Paraformer's fused FSMN contract.
#define OASR_LAUNCH_DEPTHWISE(vec_size)                                                            \
    do {                                                                                           \
        dim3 block_size(channels / (vec_size));                                                    \
        if (mask != nullptr) {                                                                     \
            if (add_input) {                                                                       \
                depthwiseConv1DKernel<T, MaskT, (vec_size), true, true>                            \
                    <<<grid_size, block_size, 0, stream>>>(input, weight, bias, mask, output,      \
                                                           batch_size, seq_len, channels,          \
                                                           kernel_size, padding_left);             \
            } else {                                                                               \
                depthwiseConv1DKernel<T, MaskT, (vec_size), true, false>                           \
                    <<<grid_size, block_size, 0, stream>>>(input, weight, bias, mask, output,      \
                                                           batch_size, seq_len, channels,          \
                                                           kernel_size, padding_left);             \
            }                                                                                      \
        } else if (add_input) {                                                                    \
            depthwiseConv1DKernel<T, MaskT, (vec_size), false, true>                               \
                <<<grid_size, block_size, 0, stream>>>(input, weight, bias, mask, output,          \
                                                       batch_size, seq_len, channels, kernel_size, \
                                                       padding_left);                              \
        } else {                                                                                   \
            depthwiseConv1DKernel<T, MaskT, (vec_size), false, false>                              \
                <<<grid_size, block_size, 0, stream>>>(input, weight, bias, mask, output,          \
                                                       batch_size, seq_len, channels, kernel_size, \
                                                       padding_left);                              \
        }                                                                                          \
    } while (0)

    if (channels % kVecSize == 0 && (channels / kVecSize) <= 1024) {
        OASR_LAUNCH_DEPTHWISE(kVecSize);
    } else {
        OASR_LAUNCH_DEPTHWISE(1);
    }

#undef OASR_LAUNCH_DEPTHWISE

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
 * @param output  Output [batch, out_len, channels]
 * @param batch_size  Batch dimension
 * @param seq_len     Sequence length
 * @param channels    Number of channels
 * @param kernel_size Convolution kernel size
 * @param padding_left  Zero padding before the sequence
 * @param padding_right Zero padding after the sequence
 * @param stream      CUDA stream
 */
template <typename T>
cudaError_t DepthwiseConv1DSilu(const T* input, const T* weight, const T* bias, T* output,
                                int batch_size, int seq_len, int channels, int kernel_size,
                                int padding_left, int padding_right, cudaStream_t stream) {
    const int out_len = seq_len + padding_left + padding_right - kernel_size + 1;
    dim3 grid_size(out_len, batch_size);

    constexpr int kVecSize = VecTypeTrait<T>::VecSize;

    // Use vectorized kernel when channels are aligned to VecSize
    // and the thread count fits within hardware limits
    if (channels % kVecSize == 0 && (channels / kVecSize) <= 1024) {
        dim3 block_size(channels / kVecSize);
        depthwiseConv1DSiluKernel<T, kVecSize><<<grid_size, block_size, 0, stream>>>(
            input, weight, bias, output, batch_size, seq_len, channels, kernel_size, padding_left);
    } else {
        dim3 block_size(channels);
        depthwiseConv1DSiluKernel<T, 1><<<grid_size, block_size, 0, stream>>>(
            input, weight, bias, output, batch_size, seq_len, channels, kernel_size, padding_left);
    }

    return cudaGetLastError();
}

/**
 * @brief Pointwise (1x1) convolution
 *
 * Essentially a GEMM: output = input * weight^T + bias.
 * Input is reshaped to [batch*seq_len, in_channels] and multiplied by weight [out_channels,
 * in_channels].
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
    updateConvStateKernel<T><<<state_grid, block_size, 0, stream>>>(input, state, batch_size,
                                                                    chunk_len, channels, state_len);

    return cudaGetLastError();
}

}  // namespace conv
}  // namespace oasr
