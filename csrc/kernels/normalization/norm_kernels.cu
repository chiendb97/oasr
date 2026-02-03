// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/normalization/norm_kernels.h"
#include "common/cuda_utils.h"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <algorithm>

namespace oasr {
namespace kernels {

// =============================================================================
// Constants and helpers
// =============================================================================

constexpr int WARP_SIZE = 32;
constexpr int MAX_THREADS_PER_BLOCK = 1024;

// Warp-level reduction for sum
template <typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction for sum
template <typename T>
__device__ __forceinline__ T blockReduceSum(T val) {
    __shared__ T shared[32];  // One slot per warp
    
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    val = warpReduceSum(val);
    
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    
    // Only first warp does the final reduction
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : T(0);
    if (wid == 0) {
        val = warpReduceSum(val);
    }
    
    return val;
}

// =============================================================================
// LayerNorm Kernels
// =============================================================================

template <typename T>
__global__ void layerNormKernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    int hidden_size,
    float eps
) {
    // Each block processes one row (one token)
    const int row_idx = blockIdx.x;
    const T* row_input = input + row_idx * hidden_size;
    T* row_output = output + row_idx * hidden_size;
    
    // Compute mean
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        local_sum += static_cast<float>(row_input[i]);
    }
    float mean = blockReduceSum(local_sum) / static_cast<float>(hidden_size);
    __syncthreads();
    
    // Broadcast mean to all threads
    __shared__ float shared_mean;
    if (threadIdx.x == 0) {
        shared_mean = mean;
    }
    __syncthreads();
    mean = shared_mean;
    
    // Compute variance
    float local_var_sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float diff = static_cast<float>(row_input[i]) - mean;
        local_var_sum += diff * diff;
    }
    float variance = blockReduceSum(local_var_sum) / static_cast<float>(hidden_size);
    __syncthreads();
    
    // Broadcast variance
    __shared__ float shared_var;
    if (threadIdx.x == 0) {
        shared_var = variance;
    }
    __syncthreads();
    variance = shared_var;
    
    // Compute rsqrt(variance + eps)
    float inv_std = rsqrtf(variance + eps);
    
    // Normalize and scale
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float normalized = (static_cast<float>(row_input[i]) - mean) * inv_std;
        float scaled = normalized * static_cast<float>(gamma[i]);
        if (beta != nullptr) {
            scaled += static_cast<float>(beta[i]);
        }
        row_output[i] = static_cast<T>(scaled);
    }
}

// Optimized kernel for hidden_size <= 1024 (fits in one block)
template <typename T, int HIDDEN_SIZE>
__global__ void layerNormKernelSmall(
    const T* __restrict__ input,
    T* __restrict__ output,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    float eps
) {
    const int row_idx = blockIdx.x;
    const T* row_input = input + row_idx * HIDDEN_SIZE;
    T* row_output = output + row_idx * HIDDEN_SIZE;
    
    // Each thread handles multiple elements
    constexpr int ELEMENTS_PER_THREAD = (HIDDEN_SIZE + 255) / 256;
    
    float vals[ELEMENTS_PER_THREAD];
    float local_sum = 0.0f;
    
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int idx = threadIdx.x + i * blockDim.x;
        if (idx < HIDDEN_SIZE) {
            vals[i] = static_cast<float>(row_input[idx]);
            local_sum += vals[i];
        }
    }
    
    // Reduce to get mean
    float mean = blockReduceSum(local_sum) / static_cast<float>(HIDDEN_SIZE);
    __syncthreads();
    
    __shared__ float shared_mean;
    if (threadIdx.x == 0) shared_mean = mean;
    __syncthreads();
    mean = shared_mean;
    
    // Compute variance
    float local_var = 0.0f;
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int idx = threadIdx.x + i * blockDim.x;
        if (idx < HIDDEN_SIZE) {
            float diff = vals[i] - mean;
            local_var += diff * diff;
        }
    }
    
    float variance = blockReduceSum(local_var) / static_cast<float>(HIDDEN_SIZE);
    __syncthreads();
    
    __shared__ float shared_var;
    if (threadIdx.x == 0) shared_var = variance;
    __syncthreads();
    variance = shared_var;
    
    float inv_std = rsqrtf(variance + eps);
    
    // Write output
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int idx = threadIdx.x + i * blockDim.x;
        if (idx < HIDDEN_SIZE) {
            float normalized = (vals[i] - mean) * inv_std;
            float g = static_cast<float>(gamma[idx]);
            float scaled = normalized * g;
            if (beta != nullptr) {
                scaled += static_cast<float>(beta[idx]);
            }
            row_output[idx] = static_cast<T>(scaled);
        }
    }
}

// =============================================================================
// RMSNorm Kernels
// =============================================================================

template <typename T>
__global__ void rmsNormKernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const T* __restrict__ gamma,
    int hidden_size,
    float eps
) {
    const int row_idx = blockIdx.x;
    const T* row_input = input + row_idx * hidden_size;
    T* row_output = output + row_idx * hidden_size;
    
    // Compute mean of squares
    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = static_cast<float>(row_input[i]);
        local_sum_sq += val * val;
    }
    
    float mean_sq = blockReduceSum(local_sum_sq) / static_cast<float>(hidden_size);
    __syncthreads();
    
    __shared__ float shared_mean_sq;
    if (threadIdx.x == 0) {
        shared_mean_sq = mean_sq;
    }
    __syncthreads();
    mean_sq = shared_mean_sq;
    
    float inv_rms = rsqrtf(mean_sq + eps);
    
    // Normalize and scale
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float normalized = static_cast<float>(row_input[i]) * inv_rms;
        row_output[i] = static_cast<T>(normalized * static_cast<float>(gamma[i]));
    }
}

// =============================================================================
// BatchNorm1D Kernel (inference mode)
// =============================================================================

template <typename T>
__global__ void batchNorm1DKernel(
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
    // Each thread handles one element
    // Input shape: [batch, seq_len, channels]
    const int total_elements = batch_size * seq_len * channels;
    
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < total_elements; 
         idx += blockDim.x * gridDim.x) {
        
        int c = idx % channels;
        
        float mean = static_cast<float>(running_mean[c]);
        float var = static_cast<float>(running_var[c]);
        float g = static_cast<float>(gamma[c]);
        float b = static_cast<float>(beta[c]);
        
        float inv_std = rsqrtf(var + eps);
        float x = static_cast<float>(input[idx]);
        float normalized = (x - mean) * inv_std;
        
        output[idx] = static_cast<T>(normalized * g + b);
    }
}

// =============================================================================
// GroupNorm Kernel
// =============================================================================

template <typename T>
__global__ void groupNormKernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    int batch_size,
    int seq_len,
    int channels,
    int num_groups,
    float eps
) {
    // Each block processes one (batch, seq, group) combination
    int channels_per_group = channels / num_groups;
    
    int idx = blockIdx.x;
    int group_idx = idx % num_groups;
    int seq_idx = (idx / num_groups) % seq_len;
    int batch_idx = idx / (num_groups * seq_len);
    
    const T* group_input = input + batch_idx * seq_len * channels + 
                           seq_idx * channels + 
                           group_idx * channels_per_group;
    T* group_output = output + batch_idx * seq_len * channels + 
                      seq_idx * channels + 
                      group_idx * channels_per_group;
    
    // Compute mean
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < channels_per_group; i += blockDim.x) {
        local_sum += static_cast<float>(group_input[i]);
    }
    float mean = blockReduceSum(local_sum) / static_cast<float>(channels_per_group);
    __syncthreads();
    
    __shared__ float shared_mean;
    if (threadIdx.x == 0) shared_mean = mean;
    __syncthreads();
    mean = shared_mean;
    
    // Compute variance
    float local_var = 0.0f;
    for (int i = threadIdx.x; i < channels_per_group; i += blockDim.x) {
        float diff = static_cast<float>(group_input[i]) - mean;
        local_var += diff * diff;
    }
    float variance = blockReduceSum(local_var) / static_cast<float>(channels_per_group);
    __syncthreads();
    
    __shared__ float shared_var;
    if (threadIdx.x == 0) shared_var = variance;
    __syncthreads();
    variance = shared_var;
    
    float inv_std = rsqrtf(variance + eps);
    
    // Normalize and scale
    for (int i = threadIdx.x; i < channels_per_group; i += blockDim.x) {
        int channel_idx = group_idx * channels_per_group + i;
        float normalized = (static_cast<float>(group_input[i]) - mean) * inv_std;
        float g = static_cast<float>(gamma[channel_idx]);
        float b = static_cast<float>(beta[channel_idx]);
        group_output[i] = static_cast<T>(normalized * g + b);
    }
}

// =============================================================================
// Add + LayerNorm Fused Kernel
// =============================================================================

template <typename T>
__global__ void addLayerNormKernel(
    const T* __restrict__ input,
    const T* __restrict__ residual,
    T* __restrict__ output,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    int hidden_size,
    float eps
) {
    const int row_idx = blockIdx.x;
    const T* row_input = input + row_idx * hidden_size;
    const T* row_residual = residual + row_idx * hidden_size;
    T* row_output = output + row_idx * hidden_size;
    
    // Compute mean of (input + residual)
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = static_cast<float>(row_input[i]) + static_cast<float>(row_residual[i]);
        local_sum += val;
    }
    float mean = blockReduceSum(local_sum) / static_cast<float>(hidden_size);
    __syncthreads();
    
    __shared__ float shared_mean;
    if (threadIdx.x == 0) shared_mean = mean;
    __syncthreads();
    mean = shared_mean;
    
    // Compute variance
    float local_var = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = static_cast<float>(row_input[i]) + static_cast<float>(row_residual[i]);
        float diff = val - mean;
        local_var += diff * diff;
    }
    float variance = blockReduceSum(local_var) / static_cast<float>(hidden_size);
    __syncthreads();
    
    __shared__ float shared_var;
    if (threadIdx.x == 0) shared_var = variance;
    __syncthreads();
    variance = shared_var;
    
    float inv_std = rsqrtf(variance + eps);
    
    // Normalize and scale
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = static_cast<float>(row_input[i]) + static_cast<float>(row_residual[i]);
        float normalized = (val - mean) * inv_std;
        float scaled = normalized * static_cast<float>(gamma[i]);
        if (beta != nullptr) {
            scaled += static_cast<float>(beta[i]);
        }
        row_output[i] = static_cast<T>(scaled);
    }
}

// =============================================================================
// Dispatcher functions
// =============================================================================

template <typename T>
void invokeLayerNormTyped(const void* input, void* output,
                          const void* gamma, const void* beta,
                          int batch_size, int seq_len, int hidden_size,
                          float eps, cudaStream_t stream) {
    int num_rows = batch_size * seq_len;
    
    // Choose block size based on hidden_size
    int block_size = std::min(hidden_size, MAX_THREADS_PER_BLOCK);
    block_size = ((block_size + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    
    layerNormKernel<T><<<num_rows, block_size, 0, stream>>>(
        static_cast<const T*>(input),
        static_cast<T*>(output),
        static_cast<const T*>(gamma),
        static_cast<const T*>(beta),
        hidden_size,
        eps
    );
}

template <typename T>
void invokeRMSNormTyped(const void* input, void* output,
                        const void* gamma,
                        int batch_size, int seq_len, int hidden_size,
                        float eps, cudaStream_t stream) {
    int num_rows = batch_size * seq_len;
    
    int block_size = std::min(hidden_size, MAX_THREADS_PER_BLOCK);
    block_size = ((block_size + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    
    rmsNormKernel<T><<<num_rows, block_size, 0, stream>>>(
        static_cast<const T*>(input),
        static_cast<T*>(output),
        static_cast<const T*>(gamma),
        hidden_size,
        eps
    );
}

// =============================================================================
// Public API implementations
// =============================================================================

void invokeLayerNorm(const void* input, void* output,
                     const void* gamma, const void* beta,
                     int batch_size, int seq_len, int hidden_size,
                     float eps, DataType dtype, cudaStream_t stream) {
    switch (dtype) {
        case DataType::FP32:
            invokeLayerNormTyped<float>(input, output, gamma, beta,
                                        batch_size, seq_len, hidden_size,
                                        eps, stream);
            break;
        case DataType::FP16:
            invokeLayerNormTyped<half>(input, output, gamma, beta,
                                       batch_size, seq_len, hidden_size,
                                       eps, stream);
            break;
        case DataType::BF16:
            invokeLayerNormTyped<__nv_bfloat16>(input, output, gamma, beta,
                                                batch_size, seq_len, hidden_size,
                                                eps, stream);
            break;
        default:
            throw std::runtime_error("Unsupported data type for LayerNorm");
    }
}

void invokeRMSNorm(const void* input, void* output,
                   const void* gamma,
                   int batch_size, int seq_len, int hidden_size,
                   float eps, DataType dtype, cudaStream_t stream) {
    switch (dtype) {
        case DataType::FP32:
            invokeRMSNormTyped<float>(input, output, gamma,
                                      batch_size, seq_len, hidden_size,
                                      eps, stream);
            break;
        case DataType::FP16:
            invokeRMSNormTyped<half>(input, output, gamma,
                                     batch_size, seq_len, hidden_size,
                                     eps, stream);
            break;
        case DataType::BF16:
            invokeRMSNormTyped<__nv_bfloat16>(input, output, gamma,
                                              batch_size, seq_len, hidden_size,
                                              eps, stream);
            break;
        default:
            throw std::runtime_error("Unsupported data type for RMSNorm");
    }
}

void invokeBatchNorm1D(const void* input, void* output,
                       const void* gamma, const void* beta,
                       const void* running_mean, const void* running_var,
                       int batch_size, int seq_len, int channels,
                       float eps, DataType dtype, cudaStream_t stream) {
    int total_elements = batch_size * seq_len * channels;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    switch (dtype) {
        case DataType::FP32:
            batchNorm1DKernel<float><<<grid_size, block_size, 0, stream>>>(
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
            batchNorm1DKernel<half><<<grid_size, block_size, 0, stream>>>(
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
            batchNorm1DKernel<__nv_bfloat16><<<grid_size, block_size, 0, stream>>>(
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
            throw std::runtime_error("Unsupported data type for BatchNorm1D");
    }
}

void invokeGroupNorm(const void* input, void* output,
                     const void* gamma, const void* beta,
                     int batch_size, int seq_len, int channels, int num_groups,
                     float eps, DataType dtype, cudaStream_t stream) {
    int num_blocks = batch_size * seq_len * num_groups;
    int channels_per_group = channels / num_groups;
    int block_size = std::min(channels_per_group, MAX_THREADS_PER_BLOCK);
    block_size = ((block_size + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    
    switch (dtype) {
        case DataType::FP32:
            groupNormKernel<float><<<num_blocks, block_size, 0, stream>>>(
                static_cast<const float*>(input),
                static_cast<float*>(output),
                static_cast<const float*>(gamma),
                static_cast<const float*>(beta),
                batch_size, seq_len, channels, num_groups, eps
            );
            break;
        case DataType::FP16:
            groupNormKernel<half><<<num_blocks, block_size, 0, stream>>>(
                static_cast<const half*>(input),
                static_cast<half*>(output),
                static_cast<const half*>(gamma),
                static_cast<const half*>(beta),
                batch_size, seq_len, channels, num_groups, eps
            );
            break;
        case DataType::BF16:
            groupNormKernel<__nv_bfloat16><<<num_blocks, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(input),
                static_cast<__nv_bfloat16*>(output),
                static_cast<const __nv_bfloat16*>(gamma),
                static_cast<const __nv_bfloat16*>(beta),
                batch_size, seq_len, channels, num_groups, eps
            );
            break;
        default:
            throw std::runtime_error("Unsupported data type for GroupNorm");
    }
}

void invokeAddLayerNorm(const void* input, const void* residual, void* output,
                        const void* gamma, const void* beta,
                        int batch_size, int seq_len, int hidden_size,
                        float eps, DataType dtype, cudaStream_t stream) {
    int num_rows = batch_size * seq_len;
    int block_size = std::min(hidden_size, MAX_THREADS_PER_BLOCK);
    block_size = ((block_size + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    
    switch (dtype) {
        case DataType::FP32:
            addLayerNormKernel<float><<<num_rows, block_size, 0, stream>>>(
                static_cast<const float*>(input),
                static_cast<const float*>(residual),
                static_cast<float*>(output),
                static_cast<const float*>(gamma),
                static_cast<const float*>(beta),
                hidden_size, eps
            );
            break;
        case DataType::FP16:
            addLayerNormKernel<half><<<num_rows, block_size, 0, stream>>>(
                static_cast<const half*>(input),
                static_cast<const half*>(residual),
                static_cast<half*>(output),
                static_cast<const half*>(gamma),
                static_cast<const half*>(beta),
                hidden_size, eps
            );
            break;
        case DataType::BF16:
            addLayerNormKernel<__nv_bfloat16><<<num_rows, block_size, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(input),
                static_cast<const __nv_bfloat16*>(residual),
                static_cast<__nv_bfloat16*>(output),
                static_cast<const __nv_bfloat16*>(gamma),
                static_cast<const __nv_bfloat16*>(beta),
                hidden_size, eps
            );
            break;
        default:
            throw std::runtime_error("Unsupported data type for AddLayerNorm");
    }
}

// Stub implementations for fused kernels (TODO: implement with CUTLASS)
void invokeLayerNormLinear(const void* input, void* output,
                           const void* ln_gamma, const void* ln_beta,
                           const void* weight, const void* bias,
                           int batch_size, int seq_len,
                           int in_features, int out_features,
                           float eps, DataType dtype, cudaStream_t stream) {
    // TODO: Implement fused LayerNorm + Linear kernel
    throw std::runtime_error("invokeLayerNormLinear not yet implemented");
}

void invokeAddLayerNormLinear(const void* input, const void* residual, void* output,
                              const void* ln_gamma, const void* ln_beta,
                              const void* weight, const void* bias,
                              int batch_size, int seq_len,
                              int in_features, int out_features,
                              float eps, DataType dtype, cudaStream_t stream) {
    // TODO: Implement fused Add + LayerNorm + Linear kernel
    throw std::runtime_error("invokeAddLayerNormLinear not yet implemented");
}

// Explicit template instantiations
template void invokeLayerNormTyped<float>(const void*, void*, const void*, const void*,
                                          int, int, int, float, cudaStream_t);
template void invokeLayerNormTyped<half>(const void*, void*, const void*, const void*,
                                         int, int, int, float, cudaStream_t);
template void invokeLayerNormTyped<__nv_bfloat16>(const void*, void*, const void*, const void*,
                                                  int, int, int, float, cudaStream_t);

template void invokeRMSNormTyped<float>(const void*, void*, const void*,
                                        int, int, int, float, cudaStream_t);
template void invokeRMSNormTyped<half>(const void*, void*, const void*,
                                       int, int, int, float, cudaStream_t);
template void invokeRMSNormTyped<__nv_bfloat16>(const void*, void*, const void*,
                                                int, int, int, float, cudaStream_t);

} // namespace kernels
} // namespace oasr
