// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <torch/extension.h>

#include "kernels/common/math.h"
#include "kernels/common/vec_dtypes.h"
#include "kernels/conv/conv1d_kernels.h"
#include "kernels/gemm/gemm_kernels.h"

namespace oasr {
namespace kernels {

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

    val = kernels::swish(val);

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
        acc[v] = kernels::swish(acc[v]);
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
// Dispatcher functions
// =============================================================================

template <typename T>
void invokeDepthwiseConv1DTyped(const torch::Tensor& input, const torch::Tensor& weight,
                                const torch::Tensor& bias, torch::Tensor& output, int padding,
                                cudaStream_t stream) {
    const int batch_size = input.size(0);
    const int seq_len = input.size(1);
    const int channels = input.size(2);
    const int kernel_size = weight.size(0);

    const T* input_ptr = static_cast<const T*>(input.data_ptr());
    const T* weight_ptr = static_cast<const T*>(weight.data_ptr());
    const T* bias_ptr = bias.defined() ? static_cast<const T*>(bias.data_ptr()) : nullptr;
    T* output_ptr = static_cast<T*>(output.data_ptr());

    const int out_len = seq_len + 2 * padding - kernel_size + 1;
    dim3 grid_size(out_len, batch_size);

    constexpr int kVecSize = VecTypeTrait<T>::VecSize;

    // Use vectorized kernel when channels are aligned to VecSize
    // and the thread count fits within hardware limits
    if (channels % kVecSize == 0 && (channels / kVecSize) <= 1024) {
        dim3 block_size(channels / kVecSize);
        depthwiseConv1DVecKernel<T, kVecSize><<<grid_size, block_size, 0, stream>>>(
            input_ptr, weight_ptr, bias_ptr, output_ptr, batch_size, seq_len, channels, kernel_size,
            padding);
    } else {
        dim3 block_size(channels);
        depthwiseConv1DKernel<T><<<grid_size, block_size, 0, stream>>>(
            input_ptr, weight_ptr, bias_ptr, output_ptr, batch_size, seq_len, channels, kernel_size,
            padding);
    }
}

template <typename T>
void invokeDepthwiseConv1DSiluTyped(const torch::Tensor& input, const torch::Tensor& weight,
                                    const torch::Tensor& bias, torch::Tensor& output, int padding,
                                    cudaStream_t stream) {
    const int batch_size = input.size(0);
    const int seq_len = input.size(1);
    const int channels = input.size(2);
    const int kernel_size = weight.size(0);

    const T* input_ptr = static_cast<const T*>(input.data_ptr());
    const T* weight_ptr = static_cast<const T*>(weight.data_ptr());
    const T* bias_ptr = bias.defined() ? static_cast<const T*>(bias.data_ptr()) : nullptr;
    T* output_ptr = static_cast<T*>(output.data_ptr());

    const int out_len = seq_len + 2 * padding - kernel_size + 1;
    dim3 grid_size(out_len, batch_size);

    constexpr int kVecSize = VecTypeTrait<T>::VecSize;

    // Use vectorized kernel when channels are aligned to VecSize
    // and the thread count fits within hardware limits
    if (channels % kVecSize == 0 && (channels / kVecSize) <= 1024) {
        dim3 block_size(channels / kVecSize);
        depthwiseConv1DVecSiluKernel<T, kVecSize><<<grid_size, block_size, 0, stream>>>(
            input_ptr, weight_ptr, bias_ptr, output_ptr, batch_size, seq_len, channels, kernel_size,
            padding);
    } else {
        dim3 block_size(channels);
        depthwiseConv1DSiluKernel<T><<<grid_size, block_size, 0, stream>>>(
            input_ptr, weight_ptr, bias_ptr, output_ptr, batch_size, seq_len, channels, kernel_size,
            padding);
    }
}

// =============================================================================
// Public API implementations
// =============================================================================

torch::Tensor invokeDepthwiseConv1D(const torch::Tensor& input, const torch::Tensor& weight,
                                    const torch::Tensor& bias, int padding, cudaStream_t stream) {
    int batch_size = input.size(0);
    int seq_len = input.size(1);
    int channels = input.size(2);
    int kernel_size = weight.size(0);
    auto output = torch::empty({batch_size, seq_len + 2 * padding - kernel_size + 1, channels},
                               input.options());
    switch (input.scalar_type()) {
        case torch::ScalarType::Float:
            invokeDepthwiseConv1DTyped<float>(input, weight, bias, output, padding, stream);
            break;
        case torch::ScalarType::Half:
            invokeDepthwiseConv1DTyped<half>(input, weight, bias, output, padding, stream);
            break;
        case torch::ScalarType::BFloat16:
            invokeDepthwiseConv1DTyped<__nv_bfloat16>(input, weight, bias, output, padding, stream);
            break;
        default:
            throw std::runtime_error("Unsupported data type for DepthwiseConv1D");
    }
    return output;
}

torch::Tensor invokeDepthwiseConv1DSilu(const torch::Tensor& input, const torch::Tensor& weight,
                                        const torch::Tensor& bias, int padding,
                                        cudaStream_t stream) {
    int batch_size = input.size(0);
    int seq_len = input.size(1);
    int channels = input.size(2);
    int kernel_size = weight.size(0);
    auto output = torch::empty({batch_size, seq_len + 2 * padding - kernel_size + 1, channels},
                               input.options());

    switch (input.scalar_type()) {
        case torch::ScalarType::Float:
            invokeDepthwiseConv1DSiluTyped<float>(input, weight, bias, output, padding, stream);
            break;
        case torch::ScalarType::Half:
            invokeDepthwiseConv1DSiluTyped<half>(input, weight, bias, output, padding, stream);
            break;
        case torch::ScalarType::BFloat16:
            invokeDepthwiseConv1DSiluTyped<__nv_bfloat16>(input, weight, bias, output, padding,
                                                          stream);
            break;
        default:
            throw std::runtime_error("Unsupported data type for DepthwiseConv1DSilu");
            break;
    }
    return output;
}

torch::Tensor invokePointwiseConv1D(const torch::Tensor& input, const torch::Tensor& weight,
                                    const torch::Tensor& bias, cudaStream_t stream) {
    const int batch_size = input.size(0);
    const int seq_len = input.size(1);
    const int out_channels = weight.size(0);

    using namespace oasr::kernels::gemm;

    auto output = gemm::invokeGemm(input, weight, bias, stream);
    output = output.view({batch_size, seq_len, out_channels});
    return output;
}

torch::Tensor invokePointwiseConv1DActivation(const torch::Tensor& input,
                                              const torch::Tensor& weight,
                                              const torch::Tensor& bias, ActivationType activation,
                                              cudaStream_t stream) {
    const int batch_size = input.size(0);
    const int seq_len = input.size(1);
    const int out_channels = weight.size(0);

    using namespace oasr::kernels::gemm;

    auto output = gemm::invokeGemmActivation(input, weight, bias, activation, stream);
    output = output.view({batch_size, seq_len, out_channels});
    return output;
}

torch::Tensor invokeCausalConv1D(const torch::Tensor& input, void* state_buffer,
                                 const torch::Tensor& weight, const torch::Tensor& bias,
                                 cudaStream_t stream) {
    const int batch_size = input.size(0);
    const int chunk_len = input.size(1);
    const int channels = input.size(2);
    const int kernel_size = weight.size(-1);

    auto output = torch::empty_like(input);

    const void* input_ptr = input.data_ptr();
    const void* weight_ptr = weight.data_ptr();
    const void* bias_ptr = bias.defined() ? bias.data_ptr() : nullptr;
    void* output_ptr = output.data_ptr();

    int total_elements = batch_size * chunk_len * channels;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    int state_len = kernel_size - 1;

    switch (input.scalar_type()) {
        case torch::ScalarType::Float:
            causalConv1DKernel<float><<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(input_ptr), static_cast<float*>(state_buffer),
                static_cast<const float*>(weight_ptr), static_cast<const float*>(bias_ptr),
                static_cast<float*>(output_ptr), batch_size, chunk_len, channels, kernel_size);
            {
                int state_elements = batch_size * state_len * channels;
                int state_grid = (state_elements + block_size - 1) / block_size;
                updateConvStateKernel<float><<<state_grid, block_size, 0, stream>>>(
                    static_cast<const float*>(input_ptr), static_cast<float*>(state_buffer),
                    batch_size, chunk_len, channels, state_len);
            }
            break;
        case torch::ScalarType::Half:
            causalConv1DKernel<half><<<grid_size, block_size, 0, stream>>>(
                static_cast<const half*>(input_ptr), static_cast<half*>(state_buffer),
                static_cast<const half*>(weight_ptr), static_cast<const half*>(bias_ptr),
                static_cast<half*>(output_ptr), batch_size, chunk_len, channels, kernel_size);
            {
                int state_elements = batch_size * state_len * channels;
                int state_grid = (state_elements + block_size - 1) / block_size;
                updateConvStateKernel<half><<<state_grid, block_size, 0, stream>>>(
                    static_cast<const half*>(input_ptr), static_cast<half*>(state_buffer),
                    batch_size, chunk_len, channels, state_len);
            }
            break;
        case torch::ScalarType::BFloat16:
            causalConv1DKernel<__nv_bfloat16><<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(input_ptr),
                static_cast<__nv_bfloat16*>(state_buffer),
                static_cast<const __nv_bfloat16*>(weight_ptr),
                static_cast<const __nv_bfloat16*>(bias_ptr),
                static_cast<__nv_bfloat16*>(output_ptr), batch_size, chunk_len, channels,
                kernel_size);
            {
                int state_elements = batch_size * state_len * channels;
                int state_grid = (state_elements + block_size - 1) / block_size;
                updateConvStateKernel<__nv_bfloat16><<<state_grid, block_size, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(input_ptr),
                    static_cast<__nv_bfloat16*>(state_buffer), batch_size, chunk_len, channels,
                    state_len);
            }
            break;
        default:
            throw std::runtime_error("Unsupported data type for CausalConv1D");
            break;
    }

    return output;
}

}  // namespace kernels
}  // namespace oasr