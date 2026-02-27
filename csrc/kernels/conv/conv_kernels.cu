// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/conv/conv_kernels.h"
#include "kernels/gemm/gemm_kernels.h"
#include "common/cuda_utils.h"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
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
    const T* __restrict__ weight,    // [kernel_size, channels]
    const T* __restrict__ bias,      // [channels] or nullptr
    T* __restrict__ output,          // [batch, seq_len, channels]
    int batch_size,
    int seq_len,
    int channels,
    int kernel_size,
    int padding
) {
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
// Fused Depthwise 1D Convolution + SiLU Kernel
// =============================================================================

template <typename T>
__global__ void depthwiseConv1DSiluKernel(
    const T* __restrict__ input,     // [batch, seq_len, channels]
    const T* __restrict__ weight,    // [kernel_size, channels]
    const T* __restrict__ bias,      // [channels] or nullptr
    T* __restrict__ output,          // [batch, seq_len, channels]
    int batch_size,
    int seq_len,
    int channels,
    int kernel_size,
    int padding
) {
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

    val *= sigmoid(val);
    
    output[o_id] = (T)val;
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

// Bias + optional activation for pointwise GEMM output [batch*seq_len, out_channels].
template <typename T>
__global__ void pointwiseBiasActivationKernel(
    T* __restrict__ output,       // [batch*seq_len, out_channels]
    const T* __restrict__ bias,   // [out_channels] or nullptr
    int batch_size,
    int seq_len,
    int out_channels,
    ActivationType activation,
    bool fuse_activation
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * seq_len * out_channels;
    if (idx >= total) return;

    int out_c = idx % out_channels;
    float val = static_cast<float>(output[idx]);
    if (bias != nullptr) {
        val += static_cast<float>(bias[out_c]);
    }
    if (fuse_activation) {
        switch (activation) {
            case ActivationType::RELU:
                val = fmaxf(val, 0.0f);
                break;
            case ActivationType::GELU:
                val = 0.5f * val * (1.0f + tanhf(0.7978845608f * (val + 0.044715f * val * val * val)));
                break;
            case ActivationType::SWISH:
                val = val / (1.0f + expf(-val));
                break;
            default:
                break;
        }
    }
    output[idx] = static_cast<T>(val);
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
    const T* __restrict__ weight,
    const T* __restrict__ bias,
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
    float g = static_cast<float>(weight[c]);
    float b = static_cast<float>(bias[c]);
    
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
void invokeDepthwiseConv1DTyped(const torch::Tensor& input, const torch::Tensor& weight,
                                const torch::Tensor& bias, torch::Tensor& output,
                                int padding, cudaStream_t stream) {
    const int batch_size = input.size(0);
    const int seq_len = input.size(1);
    const int channels = input.size(2);
    const int kernel_size = weight.size(0);

    const T* input_ptr = static_cast<const T*>(input.data_ptr());
    const T* weight_ptr = static_cast<const T*>(weight.data_ptr());
    const T* bias_ptr = bias.defined() ? static_cast<const T*>(bias.data_ptr()) : nullptr;
    T* output_ptr = static_cast<T*>(output.data_ptr());

    dim3 block_size(channels);
    dim3 grid_size(seq_len + 2 * padding - kernel_size + 1, batch_size);
    depthwiseConv1DKernel<T><<<grid_size, block_size, 0, stream>>>(
        input_ptr, weight_ptr, bias_ptr, output_ptr,
        batch_size, seq_len, channels, kernel_size, padding);
}

template <typename T>
void invokeDepthwiseConv1DSiluTyped(const torch::Tensor& input, const torch::Tensor& weight,
                                    const torch::Tensor& bias, torch::Tensor& output,
                                    int padding, cudaStream_t stream) {
    const int batch_size = input.size(0);
    const int seq_len = input.size(1);
    const int channels = input.size(2);
    const int kernel_size = weight.size(0);

    const T* input_ptr = static_cast<const T*>(input.data_ptr());
    const T* weight_ptr = static_cast<const T*>(weight.data_ptr());
    const T* bias_ptr = bias.defined() ? static_cast<const T*>(bias.data_ptr()) : nullptr;
    T* output_ptr = static_cast<T*>(output.data_ptr());

    dim3 block_size(channels);
    dim3 grid_size(seq_len + 2 * padding - kernel_size + 1, batch_size);
    depthwiseConv1DSiluKernel<T><<<grid_size, block_size, 0, stream>>>(
        input_ptr, weight_ptr, bias_ptr, output_ptr,
        batch_size, seq_len, channels, kernel_size, padding);
}

// =============================================================================
// Public API implementations
// =============================================================================

void invokeConv1D(const torch::Tensor& input, torch::Tensor& output,
                  const torch::Tensor& weight, const torch::Tensor& bias,
                  int stride, int padding, int dilation, int groups,
                  ConvType conv_type, DataType dtype, bool channels_last, bool is_causal,
                  ActivationType activation, bool fuse_activation, cudaStream_t stream) {
    (void)conv_type;
    (void)channels_last;
    (void)is_causal;

    const int batch_size = input.size(0);
    const int seq_len = input.size(1);
    const int in_channels = input.size(2);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(-1);

    const void* input_ptr = input.data_ptr();
    const void* weight_ptr = weight.data_ptr();
    const void* bias_ptr = bias.defined() ? bias.data_ptr() : nullptr;
    void* output_ptr = output.data_ptr();

    int out_len = (seq_len + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int total_elements = batch_size * out_len * out_channels;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    switch (dtype) {
        case DataType::FP32:
            conv1DKernel<float><<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(input_ptr),
                static_cast<const float*>(weight_ptr),
                static_cast<const float*>(bias_ptr),
                static_cast<float*>(output_ptr),
                batch_size, seq_len, in_channels, out_channels,
                kernel_size, stride, padding, dilation, groups,
                activation, fuse_activation
            );
            break;
        case DataType::FP16:
            conv1DKernel<half><<<grid_size, block_size, 0, stream>>>(
                static_cast<const half*>(input_ptr),
                static_cast<const half*>(weight_ptr),
                static_cast<const half*>(bias_ptr),
                static_cast<half*>(output_ptr),
                batch_size, seq_len, in_channels, out_channels,
                kernel_size, stride, padding, dilation, groups,
                activation, fuse_activation
            );
            break;
        case DataType::BF16:
            conv1DKernel<__nv_bfloat16><<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(input_ptr),
                static_cast<const __nv_bfloat16*>(weight_ptr),
                static_cast<const __nv_bfloat16*>(bias_ptr),
                static_cast<__nv_bfloat16*>(output_ptr),
                batch_size, seq_len, in_channels, out_channels,
                kernel_size, stride, padding, dilation, groups,
                activation, fuse_activation
            );
            break;
        default:
            throw std::runtime_error("Unsupported data type for Conv1D");
    }
}

void invokeDepthwiseConv1D(const torch::Tensor& input, const torch::Tensor& weight,
                           const torch::Tensor& bias, torch::Tensor& output,
                           int padding, DataType dtype, cudaStream_t stream) {
    switch (dtype) {
        case DataType::FP32:
            invokeDepthwiseConv1DTyped<float>(input, weight, bias, output,
                                              padding, stream);
            break;
        case DataType::FP16:
            invokeDepthwiseConv1DTyped<half>(input, weight, bias, output,
                                             padding, stream);
            break;
        case DataType::BF16:
            invokeDepthwiseConv1DTyped<__nv_bfloat16>(input, weight, bias, output,
                                                      padding, stream);
            break;
        default:
            throw std::runtime_error("Unsupported data type for DepthwiseConv1D");
    }
}

void invokeDepthwiseConv1DSilu(const torch::Tensor& input, const torch::Tensor& weight,
                               const torch::Tensor& bias, torch::Tensor& output,
                               int padding, DataType dtype, cudaStream_t stream) {
    switch (dtype) {
        case DataType::FP32:
            invokeDepthwiseConv1DSiluTyped<float>(input, weight, bias, output,
                                                  padding, stream);
            break;
        case DataType::FP16:
            invokeDepthwiseConv1DSiluTyped<half>(input, weight, bias, output,
                                                 padding, stream);
            break;
        case DataType::BF16:
            invokeDepthwiseConv1DSiluTyped<__nv_bfloat16>(input, weight, bias, output,
                                                          padding, stream);
            break;
        default:
            throw std::runtime_error("Unsupported data type for DepthwiseConv1DSilu");
    }
}

void invokePointwiseConv1D(const torch::Tensor& input, const torch::Tensor& weight,
                           const torch::Tensor& bias, torch::Tensor& output,
                           ActivationType activation, bool fuse_activation,
                           DataType dtype, cudaStream_t stream) {
    const int batch_size = input.size(0);
    const int seq_len = input.size(1);
    const int in_channels = input.size(2);
    const int out_channels = weight.size(0);

    const int M = batch_size * seq_len;
    const int N = out_channels;
    const int K = in_channels;
    int block_size   = std::min(out_channels, MAX_THREADS_PER_BLOCK);
    cudaStream_t s   = (stream != nullptr) ? stream : 0;

    if (dtype != DataType::FP16 && dtype != DataType::BF16) {
        throw std::runtime_error("PointwiseConv1D CUTLASS backend requires FP16 or BF16");
    }
    using namespace oasr::kernels::gemm;

    torch::Tensor empty_c;
    gemm::invokeGemm(input, weight, empty_c, output, stream);

    const void* bias_ptr = bias.defined() ? bias.data_ptr() : nullptr;
    void* output_ptr = output.data_ptr();

    int total_elements = M * N;
    int grid_size      = (total_elements + block_size - 1) / block_size;
    if (dtype == DataType::FP16) {
        pointwiseBiasActivationKernel<half><<<grid_size, block_size, 0, s>>>(
            static_cast<half*>(output_ptr), static_cast<const half*>(bias_ptr),
            batch_size, seq_len, out_channels, activation, fuse_activation);
    } else {
        pointwiseBiasActivationKernel<__nv_bfloat16><<<grid_size, block_size, 0, s>>>(
            static_cast<__nv_bfloat16*>(output_ptr), static_cast<const __nv_bfloat16*>(bias_ptr),
            batch_size, seq_len, out_channels, activation, fuse_activation);
    }
}

void invokeGLU(const torch::Tensor& input, torch::Tensor& output,
               DataType dtype, cudaStream_t stream) {
    const int batch_size = input.size(0);
    const int seq_len = input.size(1);
    const int channels = input.size(2) / 2;

    const void* input_ptr = input.data_ptr();
    void* output_ptr = output.data_ptr();

    int total_elements = batch_size * seq_len * channels;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    switch (dtype) {
        case DataType::FP32:
            gluKernel<float><<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(input_ptr),
                static_cast<float*>(output_ptr),
                batch_size, seq_len, channels
            );
            break;
        case DataType::FP16:
            gluKernel<half><<<grid_size, block_size, 0, stream>>>(
                static_cast<const half*>(input_ptr),
                static_cast<half*>(output_ptr),
                batch_size, seq_len, channels
            );
            break;
        case DataType::BF16:
            gluKernel<__nv_bfloat16><<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(input_ptr),
                static_cast<__nv_bfloat16*>(output_ptr),
                batch_size, seq_len, channels
            );
            break;
        default:
            throw std::runtime_error("Unsupported data type for GLU");
    }
}

void invokeSwish(const torch::Tensor& input, torch::Tensor& output,
                 DataType dtype, cudaStream_t stream) {
    const int batch_size = input.size(0);
    const int seq_len = input.size(1);
    const int channels = input.size(2);

    const void* input_ptr = input.data_ptr();
    void* output_ptr = output.data_ptr();

    int total_elements = batch_size * seq_len * channels;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    switch (dtype) {
        case DataType::FP32:
            swishKernel<float><<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(input_ptr),
                static_cast<float*>(output_ptr),
                batch_size, seq_len, channels
            );
            break;
        case DataType::FP16:
            swishKernel<half><<<grid_size, block_size, 0, stream>>>(
                static_cast<const half*>(input_ptr),
                static_cast<half*>(output_ptr),
                batch_size, seq_len, channels
            );
            break;
        case DataType::BF16:
            swishKernel<__nv_bfloat16><<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(input_ptr),
                static_cast<__nv_bfloat16*>(output_ptr),
                batch_size, seq_len, channels
            );
            break;
        default:
            throw std::runtime_error("Unsupported data type for Swish");
    }
}

void invokeBatchNormSwish(const torch::Tensor& input, torch::Tensor& output,
                          const torch::Tensor& weight, const torch::Tensor& bias,
                          const torch::Tensor& running_mean, const torch::Tensor& running_var,
                          float eps, DataType dtype, cudaStream_t stream) {
    const int batch_size = input.size(0);
    const int seq_len = input.size(1);
    const int channels = input.size(2);
    const void* input_ptr = input.data_ptr();
    void* output_ptr = output.data_ptr();
    const void* weight_ptr = weight.data_ptr();
    const void* bias_ptr = bias.data_ptr();
    const void* running_mean_ptr = running_mean.data_ptr();
    const void* running_var_ptr = running_var.data_ptr();

    int total_elements = batch_size * seq_len * channels;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    switch (dtype) {
        case DataType::FP32:
            batchNormSwishKernel<float><<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(input_ptr),
                static_cast<float*>(output_ptr),
                static_cast<const float*>(weight_ptr),
                static_cast<const float*>(bias_ptr),
                static_cast<const float*>(running_mean_ptr),
                static_cast<const float*>(running_var_ptr),
                batch_size, seq_len, channels, eps
            );
            break;
        case DataType::FP16:
            batchNormSwishKernel<half><<<grid_size, block_size, 0, stream>>>(
                static_cast<const half*>(input_ptr),
                static_cast<half*>(output_ptr),
                static_cast<const half*>(weight_ptr),
                static_cast<const half*>(bias_ptr),
                static_cast<const half*>(running_mean_ptr),
                static_cast<const half*>(running_var_ptr),
                batch_size, seq_len, channels, eps
            );
            break;
        case DataType::BF16:
            batchNormSwishKernel<__nv_bfloat16><<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(input_ptr),
                static_cast<__nv_bfloat16*>(output_ptr),
                static_cast<const __nv_bfloat16*>(weight_ptr),
                static_cast<const __nv_bfloat16*>(bias_ptr),
                static_cast<const __nv_bfloat16*>(running_mean_ptr),
                static_cast<const __nv_bfloat16*>(running_var_ptr),
                batch_size, seq_len, channels, eps
            );
            break;
        default:
            throw std::runtime_error("Unsupported data type for BatchNormSwish");
    }
}

void invokeCausalConv1D(const torch::Tensor& input, void* state_buffer,
                        const torch::Tensor& weight, const torch::Tensor& bias,
                        torch::Tensor& output, DataType dtype, cudaStream_t stream) {
    const int batch_size = input.size(0);
    const int chunk_len = input.size(1);
    const int channels = input.size(2);
    const int kernel_size = weight.size(-1);
    const void* input_ptr = input.data_ptr();
    const void* weight_ptr = weight.data_ptr();
    const void* bias_ptr = bias.defined() ? bias.data_ptr() : nullptr;
    void* output_ptr = output.data_ptr();

    int total_elements = batch_size * chunk_len * channels;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    int state_len = kernel_size - 1;
    
    switch (dtype) {
        case DataType::FP32:
            causalConv1DKernel<float><<<grid_size, block_size, 0, stream>>>(
                static_cast<const float*>(input_ptr),
                static_cast<float*>(state_buffer),
                static_cast<const float*>(weight_ptr),
                static_cast<const float*>(bias_ptr),
                static_cast<float*>(output_ptr),
                batch_size, chunk_len, channels, kernel_size
            );
            {
                int state_elements = batch_size * state_len * channels;
                int state_grid = (state_elements + block_size - 1) / block_size;
                updateConvStateKernel<float><<<state_grid, block_size, 0, stream>>>(
                    static_cast<const float*>(input_ptr),
                    static_cast<float*>(state_buffer),
                    batch_size, chunk_len, channels, state_len
                );
            }
            break;
        case DataType::FP16:
            causalConv1DKernel<half><<<grid_size, block_size, 0, stream>>>(
                static_cast<const half*>(input_ptr),
                static_cast<half*>(state_buffer),
                static_cast<const half*>(weight_ptr),
                static_cast<const half*>(bias_ptr),
                static_cast<half*>(output_ptr),
                batch_size, chunk_len, channels, kernel_size
            );
            {
                int state_elements = batch_size * state_len * channels;
                int state_grid = (state_elements + block_size - 1) / block_size;
                updateConvStateKernel<half><<<state_grid, block_size, 0, stream>>>(
                    static_cast<const half*>(input_ptr),
                    static_cast<half*>(state_buffer),
                    batch_size, chunk_len, channels, state_len
                );
            }
            break;
        case DataType::BF16:
            causalConv1DKernel<__nv_bfloat16><<<grid_size, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(input_ptr),
                static_cast<__nv_bfloat16*>(state_buffer),
                static_cast<const __nv_bfloat16*>(weight_ptr),
                static_cast<const __nv_bfloat16*>(bias_ptr),
                static_cast<__nv_bfloat16*>(output_ptr),
                batch_size, chunk_len, channels, kernel_size
            );
            {
                int state_elements = batch_size * state_len * channels;
                int state_grid = (state_elements + block_size - 1) / block_size;
                updateConvStateKernel<__nv_bfloat16><<<state_grid, block_size, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(input_ptr),
                    static_cast<__nv_bfloat16*>(state_buffer),
                    batch_size, chunk_len, channels, state_len
                );
            }
            break;
        default:
            throw std::runtime_error("Unsupported data type for CausalConv1D");
    }
}

void* initConvState(int batch_size, int kernel_size, int channels, DataType dtype) {
    int state_len = kernel_size - 1;
    size_t buffer_size = batch_size * state_len * channels * getDataTypeSize(dtype);
    
    void* buffer = nullptr;
    OASR_CUDA_CHECK(cudaMalloc(&buffer, buffer_size));
    OASR_CUDA_CHECK(cudaMemset(buffer, 0, buffer_size));
    return buffer;
}

void resetConvState(void* state_buffer, int batch_size, int kernel_size, int channels,
                    DataType dtype, cudaStream_t stream) {
    if (state_buffer != nullptr) {
        int state_len = kernel_size - 1;
        size_t bytes = batch_size * state_len * channels * getDataTypeSize(dtype);
        OASR_CUDA_CHECK(cudaMemsetAsync(state_buffer, 0, bytes, stream));
    }
}

void freeConvState(void* state_buffer) {
    if (state_buffer != nullptr) {
        OASR_CUDA_CHECK(cudaFree(state_buffer));
    }
}

// Explicit template instantiations
template void invokeDepthwiseConv1DTyped<float>(const torch::Tensor&, const torch::Tensor&,
                                                 const torch::Tensor&, torch::Tensor&,
                                                 int, cudaStream_t);
template void invokeDepthwiseConv1DTyped<half>(const torch::Tensor&, const torch::Tensor&,
                                                const torch::Tensor&, torch::Tensor&,
                                                int, cudaStream_t);
template void invokeDepthwiseConv1DTyped<__nv_bfloat16>(const torch::Tensor&, const torch::Tensor&,
                                                         const torch::Tensor&, torch::Tensor&,
                                                         int, cudaStream_t);

} // namespace kernels
} // namespace oasr
