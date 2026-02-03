// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/convolution/conv_kernels.h"
#include "common/cuda_utils.h"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>

namespace oasr {
namespace kernels {

// =============================================================================
// Constants and helpers
// =============================================================================

constexpr int MAX_THREADS_PER_BLOCK = 1024;

// Sigmoid function
template <typename T>
__device__ __forceinline__ T sigmoid(T x) {
    return T(1.0f) / (T(1.0f) + expf(-float(x)));
}

// Swish activation: x * sigmoid(x)
template <typename T>
__device__ __forceinline__ T swish(T x) {
    return x * sigmoid(x);
}

// =============================================================================
// Depthwise 1D Convolution Kernel
// =============================================================================

template <typename T>
__global__ void depthwiseConv1DKernel(
    const T* __restrict__ input,     // [batch, seq_len, channels]
    const T* __restrict__ weight,    // [channels, 1, kernel_size]
    const T* __restrict__ bias,      // [channels] or nullptr
    T* __restrict__ output,          // [batch, seq_len, channels]
    int batch_size,
    int seq_len,
    int channels,
    int kernel_size,
    int padding,
    bool is_causal
) {
    // Each thread handles one output element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len * channels;
    
    if (idx >= total_elements) return;
    
    // Compute indices
    int c = idx % channels;
    int t = (idx / channels) % seq_len;
    int b = idx / (channels * seq_len);
    
    // Compute convolution
    float sum = 0.0f;
    int half_kernel = kernel_size / 2;
    
    for (int k = 0; k < kernel_size; k++) {
        int input_t;
        if (is_causal) {
            // Causal: only look at past and current
            input_t = t - (kernel_size - 1) + k;
        } else {
            // Standard: symmetric padding
            input_t = t - half_kernel + k;
        }
        
        if (input_t >= 0 && input_t < seq_len) {
            int input_idx = b * seq_len * channels + input_t * channels + c;
            int weight_idx = c * kernel_size + k;
            sum += static_cast<float>(input[input_idx]) * static_cast<float>(weight[weight_idx]);
        }
    }
    
    // Add bias
    if (bias != nullptr) {
        sum += static_cast<float>(bias[c]);
    }
    
    output[idx] = static_cast<T>(sum);
}

// Optimized depthwise conv with shared memory
template <typename T, int KERNEL_SIZE>
__global__ void depthwiseConv1DKernelOptimized(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ output,
    int batch_size,
    int seq_len,
    int channels,
    int padding,
    bool is_causal
) {
    extern __shared__ char shared_mem[];
    T* shared_input = reinterpret_cast<T*>(shared_mem);
    
    const int TILE_SIZE = blockDim.x;
    const int half_kernel = KERNEL_SIZE / 2;
    const int halo = is_causal ? (KERNEL_SIZE - 1) : half_kernel;
    
    int b = blockIdx.z;
    int c = blockIdx.y;
    int tile_start = blockIdx.x * TILE_SIZE;
    int local_t = threadIdx.x;
    int global_t = tile_start + local_t;
    
    // Load input tile with halo
    int shared_idx = local_t + halo;
    if (global_t < seq_len) {
        int input_idx = b * seq_len * channels + global_t * channels + c;
        shared_input[shared_idx] = input[input_idx];
    } else {
        shared_input[shared_idx] = T(0);
    }
    
    // Load halo regions
    if (local_t < halo) {
        int left_t = tile_start - halo + local_t;
        if (left_t >= 0 && left_t < seq_len) {
            int input_idx = b * seq_len * channels + left_t * channels + c;
            shared_input[local_t] = input[input_idx];
        } else {
            shared_input[local_t] = T(0);
        }
        
        if (!is_causal) {
            int right_t = tile_start + TILE_SIZE + local_t;
            if (right_t < seq_len) {
                int input_idx = b * seq_len * channels + right_t * channels + c;
                shared_input[TILE_SIZE + halo + local_t] = input[input_idx];
            } else {
                shared_input[TILE_SIZE + halo + local_t] = T(0);
            }
        }
    }
    
    __syncthreads();
    
    if (global_t >= seq_len) return;
    
    // Load weight into registers
    float w[KERNEL_SIZE];
    #pragma unroll
    for (int k = 0; k < KERNEL_SIZE; k++) {
        w[k] = static_cast<float>(weight[c * KERNEL_SIZE + k]);
    }
    
    // Compute convolution
    float sum = 0.0f;
    if (is_causal) {
        #pragma unroll
        for (int k = 0; k < KERNEL_SIZE; k++) {
            sum += static_cast<float>(shared_input[shared_idx - (KERNEL_SIZE - 1) + k]) * w[k];
        }
    } else {
        #pragma unroll
        for (int k = 0; k < KERNEL_SIZE; k++) {
            sum += static_cast<float>(shared_input[shared_idx - half_kernel + k]) * w[k];
        }
    }
    
    // Add bias
    if (bias != nullptr) {
        sum += static_cast<float>(bias[c]);
    }
    
    int output_idx = b * seq_len * channels + global_t * channels + c;
    output[output_idx] = static_cast<T>(sum);
}

// =============================================================================
// Pointwise (1x1) Convolution Kernel
// =============================================================================

template <typename T>
__global__ void pointwiseConv1DKernel(
    const T* __restrict__ input,      // [batch, seq_len, in_channels]
    const T* __restrict__ weight,     // [out_channels, in_channels]
    const T* __restrict__ bias,       // [out_channels] or nullptr
    T* __restrict__ output,           // [batch, seq_len, out_channels]
    int batch_size,
    int seq_len,
    int in_channels,
    int out_channels,
    ActivationType activation,
    bool fuse_activation
) {
    // Each block handles one output position, threads handle different output channels
    int pos = blockIdx.x;  // batch * seq_len position
    int out_c = threadIdx.x;
    
    if (pos >= batch_size * seq_len || out_c >= out_channels) return;
    
    const T* input_ptr = input + pos * in_channels;
    
    // Compute dot product
    float sum = 0.0f;
    for (int ic = 0; ic < in_channels; ic++) {
        sum += static_cast<float>(input_ptr[ic]) * 
               static_cast<float>(weight[out_c * in_channels + ic]);
    }
    
    // Add bias
    if (bias != nullptr) {
        sum += static_cast<float>(bias[out_c]);
    }
    
    // Apply activation
    if (fuse_activation) {
        switch (activation) {
            case ActivationType::RELU:
                sum = fmaxf(sum, 0.0f);
                break;
            case ActivationType::GELU:
                sum = 0.5f * sum * (1.0f + tanhf(0.7978845608f * (sum + 0.044715f * sum * sum * sum)));
                break;
            case ActivationType::SWISH:
                sum = sum / (1.0f + expf(-sum));
                break;
            default:
                break;
        }
    }
    
    output[pos * out_channels + out_c] = static_cast<T>(sum);
}

// =============================================================================
// GLU (Gated Linear Unit) Kernel
// =============================================================================

template <typename T>
__global__ void gluKernel(
    const T* __restrict__ input,   // [batch, seq_len, 2 * channels]
    T* __restrict__ output,        // [batch, seq_len, channels]
    int batch_size,
    int seq_len,
    int channels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len * channels;
    
    if (idx >= total_elements) return;
    
    int c = idx % channels;
    int pos = idx / channels;
    
    // input[:, :, :channels] * sigmoid(input[:, :, channels:])
    int input_idx1 = pos * (2 * channels) + c;
    int input_idx2 = pos * (2 * channels) + channels + c;
    
    float x = static_cast<float>(input[input_idx1]);
    float gate = static_cast<float>(input[input_idx2]);
    float sigmoid_gate = 1.0f / (1.0f + expf(-gate));
    
    output[idx] = static_cast<T>(x * sigmoid_gate);
}

// =============================================================================
// Swish Kernel
// =============================================================================

template <typename T>
__global__ void swishKernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int batch_size,
    int seq_len,
    int channels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len * channels;
    
    if (idx >= total_elements) return;
    
    float x = static_cast<float>(input[idx]);
    float result = x / (1.0f + expf(-x));
    output[idx] = static_cast<T>(result);
}

// =============================================================================
// Fused BatchNorm + Swish Kernel
// =============================================================================

template <typename T>
__global__ void batchNormSwishKernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    const T* __restrict__ running_mean,
    const T* __restrict__ running_var,
    int batch_size,
    int seq_len,
    int channels,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len * channels;
    
    if (idx >= total_elements) return;
    
    int c = idx % channels;
    
    float x = static_cast<float>(input[idx]);
    float mean = static_cast<float>(running_mean[c]);
    float var = static_cast<float>(running_var[c]);
    float g = static_cast<float>(gamma[c]);
    float b = static_cast<float>(beta[c]);
    
    // BatchNorm
    float inv_std = rsqrtf(var + eps);
    float normalized = (x - mean) * inv_std * g + b;
    
    // Swish
    float result = normalized / (1.0f + expf(-normalized));
    
    output[idx] = static_cast<T>(result);
}

// =============================================================================
// Causal Conv1D with State Kernel
// =============================================================================

template <typename T>
__global__ void causalConv1DKernel(
    const T* __restrict__ input,      // [batch, chunk_len, channels]
    T* __restrict__ state,            // [batch, kernel_size-1, channels]
    const T* __restrict__ weight,     // [channels, 1, kernel_size]
    const T* __restrict__ bias,       // [channels] or nullptr
    T* __restrict__ output,           // [batch, chunk_len, channels]
    int batch_size,
    int chunk_len,
    int channels,
    int kernel_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * chunk_len * channels;
    
    if (idx >= total_elements) return;
    
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
__global__ void updateConvStateKernel(
    const T* __restrict__ input,  // [batch, chunk_len, channels]
    T* __restrict__ state,        // [batch, state_len, channels]
    int batch_size,
    int chunk_len,
    int channels,
    int state_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * state_len * channels;
    
    if (idx >= total_elements) return;
    
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
// General 1D Convolution Kernel (for standard convolutions)
// =============================================================================

template <typename T>
__global__ void conv1DKernel(
    const T* __restrict__ input,      // [batch, seq_len, in_channels]
    const T* __restrict__ weight,     // [out_channels, in_channels/groups, kernel_size]
    const T* __restrict__ bias,       // [out_channels] or nullptr
    T* __restrict__ output,           // [batch, out_len, out_channels]
    int batch_size,
    int seq_len,
    int in_channels,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int groups,
    ActivationType activation,
    bool fuse_activation
) {
    // Output length
    int out_len = (seq_len + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_len * out_channels;
    
    if (idx >= total_elements) return;
    
    int out_c = idx % out_channels;
    int out_t = (idx / out_channels) % out_len;
    int b = idx / (out_channels * out_len);
    
    // Group info
    int group = out_c / (out_channels / groups);
    int in_channels_per_group = in_channels / groups;
    
    float sum = 0.0f;
    
    for (int k = 0; k < kernel_size; k++) {
        int input_t = out_t * stride - padding + k * dilation;
        
        if (input_t >= 0 && input_t < seq_len) {
            for (int ic = 0; ic < in_channels_per_group; ic++) {
                int global_ic = group * in_channels_per_group + ic;
                int input_idx = b * seq_len * in_channels + input_t * in_channels + global_ic;
                int weight_idx = out_c * in_channels_per_group * kernel_size + ic * kernel_size + k;
                
                sum += static_cast<float>(input[input_idx]) * static_cast<float>(weight[weight_idx]);
            }
        }
    }
    
    if (bias != nullptr) {
        sum += static_cast<float>(bias[out_c]);
    }
    
    // Apply activation
    if (fuse_activation) {
        switch (activation) {
            case ActivationType::RELU:
                sum = fmaxf(sum, 0.0f);
                break;
            case ActivationType::GELU:
                sum = 0.5f * sum * (1.0f + tanhf(0.7978845608f * (sum + 0.044715f * sum * sum * sum)));
                break;
            case ActivationType::SWISH:
                sum = sum / (1.0f + expf(-sum));
                break;
            default:
                break;
        }
    }
    
    int output_idx = b * out_len * out_channels + out_t * out_channels + out_c;
    output[output_idx] = static_cast<T>(sum);
}

// =============================================================================
// Dispatcher functions
// =============================================================================

template <typename T>
void invokeDepthwiseConv1DTyped(const void* input, const void* weight, const void* bias,
                                void* output, int batch_size, int seq_len, int channels,
                                int kernel_size, int padding, bool is_causal,
                                cudaStream_t stream) {
    int total_elements = batch_size * seq_len * channels;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    // Use optimized kernel for common kernel sizes
    constexpr int tile_size = 128;
    
    // Dispatch based on kernel size
    switch (kernel_size) {
        case 3: {
            dim3 grid((seq_len + tile_size - 1) / tile_size, channels, batch_size);
            int halo = is_causal ? 2 : 1;
            int shared_size = (tile_size + 2 * halo) * sizeof(T);
            depthwiseConv1DKernelOptimized<T, 3><<<grid, tile_size, shared_size, stream>>>(
                static_cast<const T*>(input), static_cast<const T*>(weight),
                static_cast<const T*>(bias), static_cast<T*>(output),
                batch_size, seq_len, channels, padding, is_causal);
            break;
        }
        case 7: {
            dim3 grid((seq_len + tile_size - 1) / tile_size, channels, batch_size);
            int halo = is_causal ? 6 : 3;
            int shared_size = (tile_size + 2 * halo) * sizeof(T);
            depthwiseConv1DKernelOptimized<T, 7><<<grid, tile_size, shared_size, stream>>>(
                static_cast<const T*>(input), static_cast<const T*>(weight),
                static_cast<const T*>(bias), static_cast<T*>(output),
                batch_size, seq_len, channels, padding, is_causal);
            break;
        }
        case 15: {
            dim3 grid((seq_len + tile_size - 1) / tile_size, channels, batch_size);
            int halo = is_causal ? 14 : 7;
            int shared_size = (tile_size + 2 * halo) * sizeof(T);
            depthwiseConv1DKernelOptimized<T, 15><<<grid, tile_size, shared_size, stream>>>(
                static_cast<const T*>(input), static_cast<const T*>(weight),
                static_cast<const T*>(bias), static_cast<T*>(output),
                batch_size, seq_len, channels, padding, is_causal);
            break;
        }
        case 31: {
            dim3 grid((seq_len + tile_size - 1) / tile_size, channels, batch_size);
            int halo = is_causal ? 30 : 15;
            int shared_size = (tile_size + 2 * halo) * sizeof(T);
            depthwiseConv1DKernelOptimized<T, 31><<<grid, tile_size, shared_size, stream>>>(
                static_cast<const T*>(input), static_cast<const T*>(weight),
                static_cast<const T*>(bias), static_cast<T*>(output),
                batch_size, seq_len, channels, padding, is_causal);
            break;
        }
        default:
            // Fall back to general kernel
            depthwiseConv1DKernel<T><<<grid_size, block_size, 0, stream>>>(
                static_cast<const T*>(input),
                static_cast<const T*>(weight),
                static_cast<const T*>(bias),
                static_cast<T*>(output),
                batch_size, seq_len, channels, kernel_size, padding, is_causal
            );
            break;
    }
}

// =============================================================================
// Public API implementations
// =============================================================================

void invokeConv1D(const Conv1DParams& params) {
    int out_len = (params.seq_len + 2 * params.padding - 
                   params.dilation * (params.kernel_size - 1) - 1) / params.stride + 1;
    int total_elements = params.batch_size * out_len * params.out_channels;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    switch (params.dtype) {
        case DataType::FP32:
            conv1DKernel<float><<<grid_size, block_size, 0, params.stream>>>(
                static_cast<const float*>(params.input),
                static_cast<const float*>(params.weight),
                static_cast<const float*>(params.bias),
                static_cast<float*>(params.output),
                params.batch_size, params.seq_len, params.in_channels,
                params.out_channels, params.kernel_size, params.stride,
                params.padding, params.dilation, params.groups,
                params.activation, params.fuse_activation
            );
            break;
        case DataType::FP16:
            conv1DKernel<half><<<grid_size, block_size, 0, params.stream>>>(
                static_cast<const half*>(params.input),
                static_cast<const half*>(params.weight),
                static_cast<const half*>(params.bias),
                static_cast<half*>(params.output),
                params.batch_size, params.seq_len, params.in_channels,
                params.out_channels, params.kernel_size, params.stride,
                params.padding, params.dilation, params.groups,
                params.activation, params.fuse_activation
            );
            break;
        case DataType::BF16:
            conv1DKernel<__nv_bfloat16><<<grid_size, block_size, 0, params.stream>>>(
                static_cast<const __nv_bfloat16*>(params.input),
                static_cast<const __nv_bfloat16*>(params.weight),
                static_cast<const __nv_bfloat16*>(params.bias),
                static_cast<__nv_bfloat16*>(params.output),
                params.batch_size, params.seq_len, params.in_channels,
                params.out_channels, params.kernel_size, params.stride,
                params.padding, params.dilation, params.groups,
                params.activation, params.fuse_activation
            );
            break;
        default:
            throw std::runtime_error("Unsupported data type for Conv1D");
    }
}

void invokeDepthwiseConv1D(const void* input, const void* weight, const void* bias,
                           void* output, int batch_size, int seq_len, int channels,
                           int kernel_size, int padding, bool is_causal,
                           DataType dtype, cudaStream_t stream) {
    switch (dtype) {
        case DataType::FP32:
            invokeDepthwiseConv1DTyped<float>(input, weight, bias, output,
                                              batch_size, seq_len, channels,
                                              kernel_size, padding, is_causal, stream);
            break;
        case DataType::FP16:
            invokeDepthwiseConv1DTyped<half>(input, weight, bias, output,
                                             batch_size, seq_len, channels,
                                             kernel_size, padding, is_causal, stream);
            break;
        case DataType::BF16:
            invokeDepthwiseConv1DTyped<__nv_bfloat16>(input, weight, bias, output,
                                                      batch_size, seq_len, channels,
                                                      kernel_size, padding, is_causal, stream);
            break;
        default:
            throw std::runtime_error("Unsupported data type for DepthwiseConv1D");
    }
}

void invokePointwiseConv1D(const void* input, const void* weight, const void* bias,
                           void* output, int batch_size, int seq_len,
                           int in_channels, int out_channels,
                           ActivationType activation, bool fuse_activation,
                           DataType dtype, cudaStream_t stream) {
    int num_positions = batch_size * seq_len;
    int block_size = std::min(out_channels, MAX_THREADS_PER_BLOCK);
    
    switch (dtype) {
        case DataType::FP32:
            pointwiseConv1DKernel<float><<<num_positions, block_size, 0, stream>>>(
                static_cast<const float*>(input),
                static_cast<const float*>(weight),
                static_cast<const float*>(bias),
                static_cast<float*>(output),
                batch_size, seq_len, in_channels, out_channels,
                activation, fuse_activation
            );
            break;
        case DataType::FP16:
            pointwiseConv1DKernel<half><<<num_positions, block_size, 0, stream>>>(
                static_cast<const half*>(input),
                static_cast<const half*>(weight),
                static_cast<const half*>(bias),
                static_cast<half*>(output),
                batch_size, seq_len, in_channels, out_channels,
                activation, fuse_activation
            );
            break;
        case DataType::BF16:
            pointwiseConv1DKernel<__nv_bfloat16><<<num_positions, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(input),
                static_cast<const __nv_bfloat16*>(weight),
                static_cast<const __nv_bfloat16*>(bias),
                static_cast<__nv_bfloat16*>(output),
                batch_size, seq_len, in_channels, out_channels,
                activation, fuse_activation
            );
            break;
        default:
            throw std::runtime_error("Unsupported data type for PointwiseConv1D");
    }
}

void invokeGLU(const void* input, void* output,
               int batch_size, int seq_len, int channels,
               DataType dtype, cudaStream_t stream) {
    int total_elements = batch_size * seq_len * channels;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    switch (dtype) {
        case DataType::FP32:
            gluKernel<float><<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(input),
                static_cast<float*>(output),
                batch_size, seq_len, channels
            );
            break;
        case DataType::FP16:
            gluKernel<half><<<grid_size, block_size, 0, stream>>>(
                static_cast<const half*>(input),
                static_cast<half*>(output),
                batch_size, seq_len, channels
            );
            break;
        case DataType::BF16:
            gluKernel<__nv_bfloat16><<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(input),
                static_cast<__nv_bfloat16*>(output),
                batch_size, seq_len, channels
            );
            break;
        default:
            throw std::runtime_error("Unsupported data type for GLU");
    }
}

void invokeSwish(const void* input, void* output,
                 int batch_size, int seq_len, int channels,
                 DataType dtype, cudaStream_t stream) {
    int total_elements = batch_size * seq_len * channels;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    switch (dtype) {
        case DataType::FP32:
            swishKernel<float><<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(input),
                static_cast<float*>(output),
                batch_size, seq_len, channels
            );
            break;
        case DataType::FP16:
            swishKernel<half><<<grid_size, block_size, 0, stream>>>(
                static_cast<const half*>(input),
                static_cast<half*>(output),
                batch_size, seq_len, channels
            );
            break;
        case DataType::BF16:
            swishKernel<__nv_bfloat16><<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(input),
                static_cast<__nv_bfloat16*>(output),
                batch_size, seq_len, channels
            );
            break;
        default:
            throw std::runtime_error("Unsupported data type for Swish");
    }
}

void invokeBatchNormSwish(const void* input, void* output,
                          const void* gamma, const void* beta,
                          const void* running_mean, const void* running_var,
                          int batch_size, int seq_len, int channels,
                          float eps, DataType dtype, cudaStream_t stream) {
    int total_elements = batch_size * seq_len * channels;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    switch (dtype) {
        case DataType::FP32:
            batchNormSwishKernel<float><<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(input),
                static_cast<float*>(output),
                static_cast<const float*>(gamma),
                static_cast<const float*>(beta),
                static_cast<const float*>(running_mean),
                static_cast<const float*>(running_var),
                batch_size, seq_len, channels, eps
            );
            break;
        case DataType::FP16:
            batchNormSwishKernel<half><<<grid_size, block_size, 0, stream>>>(
                static_cast<const half*>(input),
                static_cast<half*>(output),
                static_cast<const half*>(gamma),
                static_cast<const half*>(beta),
                static_cast<const half*>(running_mean),
                static_cast<const half*>(running_var),
                batch_size, seq_len, channels, eps
            );
            break;
        case DataType::BF16:
            batchNormSwishKernel<__nv_bfloat16><<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(input),
                static_cast<__nv_bfloat16*>(output),
                static_cast<const __nv_bfloat16*>(gamma),
                static_cast<const __nv_bfloat16*>(beta),
                static_cast<const __nv_bfloat16*>(running_mean),
                static_cast<const __nv_bfloat16*>(running_var),
                batch_size, seq_len, channels, eps
            );
            break;
        default:
            throw std::runtime_error("Unsupported data type for BatchNormSwish");
    }
}

void invokeCausalConv1D(const void* input, ConvState& state,
                        const void* weight, const void* bias,
                        void* output, int batch_size, int chunk_len, int channels,
                        int kernel_size, DataType dtype, cudaStream_t stream) {
    int total_elements = batch_size * chunk_len * channels;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    int state_len = kernel_size - 1;
    
    switch (dtype) {
        case DataType::FP32:
            causalConv1DKernel<float><<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(input),
                static_cast<float*>(state.buffer),
                static_cast<const float*>(weight),
                static_cast<const float*>(bias),
                static_cast<float*>(output),
                batch_size, chunk_len, channels, kernel_size
            );
            // Update state
            {
                int state_elements = batch_size * state_len * channels;
                int state_grid = (state_elements + block_size - 1) / block_size;
                updateConvStateKernel<float><<<state_grid, block_size, 0, stream>>>(
                    static_cast<const float*>(input),
                    static_cast<float*>(state.buffer),
                    batch_size, chunk_len, channels, state_len
                );
            }
            break;
        case DataType::FP16:
            causalConv1DKernel<half><<<grid_size, block_size, 0, stream>>>(
                static_cast<const half*>(input),
                static_cast<half*>(state.buffer),
                static_cast<const half*>(weight),
                static_cast<const half*>(bias),
                static_cast<half*>(output),
                batch_size, chunk_len, channels, kernel_size
            );
            {
                int state_elements = batch_size * state_len * channels;
                int state_grid = (state_elements + block_size - 1) / block_size;
                updateConvStateKernel<half><<<state_grid, block_size, 0, stream>>>(
                    static_cast<const half*>(input),
                    static_cast<half*>(state.buffer),
                    batch_size, chunk_len, channels, state_len
                );
            }
            break;
        case DataType::BF16:
            causalConv1DKernel<__nv_bfloat16><<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(input),
                static_cast<__nv_bfloat16*>(state.buffer),
                static_cast<const __nv_bfloat16*>(weight),
                static_cast<const __nv_bfloat16*>(bias),
                static_cast<__nv_bfloat16*>(output),
                batch_size, chunk_len, channels, kernel_size
            );
            {
                int state_elements = batch_size * state_len * channels;
                int state_grid = (state_elements + block_size - 1) / block_size;
                updateConvStateKernel<__nv_bfloat16><<<state_grid, block_size, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(input),
                    static_cast<__nv_bfloat16*>(state.buffer),
                    batch_size, chunk_len, channels, state_len
                );
            }
            break;
        default:
            throw std::runtime_error("Unsupported data type for CausalConv1D");
    }
}

void initConvState(ConvState& state, int batch_size, int kernel_size, int channels,
                   DataType dtype) {
    int state_len = kernel_size - 1;
    size_t buffer_size = batch_size * state_len * channels * getDataTypeSize(dtype);
    
    OASR_CUDA_CHECK(cudaMalloc(&state.buffer, buffer_size));
    OASR_CUDA_CHECK(cudaMemset(state.buffer, 0, buffer_size));
    
    state.buffer_size = state_len;
    state.channels = channels;
    state.dtype = dtype;
}

void resetConvState(ConvState& state, cudaStream_t stream) {
    if (state.buffer != nullptr && state.buffer_size > 0) {
        size_t bytes = state.buffer_size * state.channels * getDataTypeSize(state.dtype);
        OASR_CUDA_CHECK(cudaMemsetAsync(state.buffer, 0, bytes, stream));
    }
}

void invokeConformerConvModule(const ConformerConvParams& params) {
    // TODO: Implement fused Conformer conv module
    // For now, users should call individual kernels in sequence
    throw std::runtime_error("invokeConformerConvModule not yet implemented - use individual kernels");
}

// Explicit template instantiations
template void invokeDepthwiseConv1DTyped<float>(const void*, const void*, const void*,
                                                 void*, int, int, int, int, int, bool, cudaStream_t);
template void invokeDepthwiseConv1DTyped<half>(const void*, const void*, const void*,
                                                void*, int, int, int, int, int, bool, cudaStream_t);
template void invokeDepthwiseConv1DTyped<__nv_bfloat16>(const void*, const void*, const void*,
                                                         void*, int, int, int, int, int, bool, cudaStream_t);

} // namespace kernels
} // namespace oasr
