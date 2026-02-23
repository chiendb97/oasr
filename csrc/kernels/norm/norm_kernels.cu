// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/norm/norm_kernels.h"
#include "kernels/reduction/allreduce.h"
#include "common/vec_dtypes.h"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <stdexcept>

namespace oasr {
namespace kernels {

// Import vector types from oasr namespace
using oasr::Vec;
using oasr::VecTypeTrait;
using oasr::vecSum;
using oasr::vecSumSquares;
using oasr::vecSumSquaredDiff;
using oasr::floatToVec;
using oasr::loadVec;
using oasr::storeVec;

// =============================================================================
// Constants
// =============================================================================

constexpr int MAX_THREADS_PER_BLOCK = 1024;

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
    int channels_per_group = channels / num_groups;
    float inv_channels_per_group = 1.0f / static_cast<float>(channels_per_group);

    int idx = blockIdx.x;
    int seq_idx = idx % seq_len;
    int batch_idx = idx / seq_len;

    const T* row     = input  + (batch_idx * seq_len + seq_idx) * channels;
    T*       out_row = output + (batch_idx * seq_len + seq_idx) * channels;

    // threads_per_group: largest power-of-2 <= blockDim.x / num_groups
    // int raw_tpg = blockDim.x / num_groups;
    // int tpg = 1;
    // while ((tpg << 1) <= raw_tpg) tpg <<= 1;

    // thread_per_group should be a power-of-2
    int threads_per_group = blockDim.x / num_groups;

    int my_group  = threadIdx.x / threads_per_group;
    int local_tid = threadIdx.x % threads_per_group;
    int lane = threadIdx.x & (WARP_SIZE - 1);

    // Shared memory: [group_mean | group_inv_std | warp_buf (inter-warp case)]
    extern __shared__ float smem[];
    float* group_mean    = smem;
    float* group_inv_std = smem + num_groups;

    // ---- Phase 1: thread-local accumulation for assigned group ----
    float psum = 0.0f, psq = 0.0f;
    const T* gptr = row + my_group * channels_per_group;
    for (int i = local_tid; i < channels_per_group; i += threads_per_group) {
        float v = static_cast<float>(gptr[i]);
        psum += v;
        psq  += v * v;
    }

    // ---- Phase 2: CUB-style tree reduction ----
    // Warp-level segmented reduction: XOR with offset < seg stays in the
    // same power-of-2 aligned segment, so each group reduces independently.
    int seg = (threads_per_group < WARP_SIZE) ? threads_per_group : WARP_SIZE;
    for (int off = seg >> 1; off > 0; off >>= 1) {
        psum += __shfl_xor_sync(0xffffffff, psum, off);
        psq  += __shfl_xor_sync(0xffffffff, psq,  off);
    }

    if (threads_per_group <= WARP_SIZE) {
        // Each group's segment fits in one warp -- leader already has the total.
        if (local_tid == 0) {
            float mean = psum * inv_channels_per_group;
            group_mean[my_group]    = mean;
            group_inv_std[my_group] = rsqrtf(psq * inv_channels_per_group - mean * mean + eps);
        }
    } else {
        // Multiple warps per group: write warp partials, then reduce across warps.
        int warps_per_group = threads_per_group / WARP_SIZE;
        int warp_idx = local_tid / WARP_SIZE;
        float* wbuf = smem + 2 * num_groups;

        if (lane == 0) {
            int bi = my_group * warps_per_group + warp_idx;
            wbuf[bi]                    = psum;
            wbuf[num_groups * warps_per_group + bi] = psq;
        }
        __syncthreads();

        if (local_tid == 0) {
            float s = 0.0f, sq = 0.0f;
            int base    = my_group * warps_per_group;
            int sq_base = num_groups * warps_per_group + base;
            for (int w = 0; w < warps_per_group; ++w) {
                s  += wbuf[base + w];
                sq += wbuf[sq_base + w];
            }
            float mean = s * inv_channels_per_group;
            group_mean[my_group]    = mean;
            group_inv_std[my_group] = rsqrtf(sq * inv_channels_per_group - mean * mean + eps);
        }
    }
    __syncthreads();

    // ---- Phase 3: normalize and apply per-channel affine transform ----
    for (int c = threadIdx.x; c < channels; c += blockDim.x) {
        int   g = c / channels_per_group;
        float v = static_cast<float>(row[c]);
        float n = (v - group_mean[g]) * group_inv_std[g];
        out_row[c] = static_cast<T>(
            n * static_cast<float>(gamma[c]) + static_cast<float>(beta[c])
        );
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
// Vectorized LayerNorm Kernel
// =============================================================================

template <typename T, int VecSize>
__global__ void layerNormKernelVectorized(
    const T* __restrict__ input,
    T* __restrict__ output,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    int hidden_size,
    float eps
) {
    using VecT = Vec<T, VecSize>;
    
    const int row_idx = blockIdx.x;
    const T* row_input = input + row_idx * hidden_size;
    T* row_output = output + row_idx * hidden_size;
    
    const int vec_hidden_size = hidden_size / VecSize;
    
    // Phase 1: Compute mean using vectorized loads
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < vec_hidden_size; i += blockDim.x) {
        VecT v;
        v.load(row_input + i * VecSize);
        local_sum += vecSum(v);
    }
    
    float mean = blockReduceSum(local_sum) / static_cast<float>(hidden_size);
    __syncthreads();
    
    __shared__ float shared_mean;
    if (threadIdx.x == 0) shared_mean = mean;
    __syncthreads();
    mean = shared_mean;
    
    // Phase 2: Compute variance using vectorized loads
    float local_var = 0.0f;
    for (int i = threadIdx.x; i < vec_hidden_size; i += blockDim.x) {
        VecT v;
        v.load(row_input + i * VecSize);
        local_var += vecSumSquaredDiff(v, mean);
    }
    
    float variance = blockReduceSum(local_var) / static_cast<float>(hidden_size);
    __syncthreads();
    
    __shared__ float shared_var;
    if (threadIdx.x == 0) shared_var = variance;
    __syncthreads();
    variance = shared_var;
    
    float inv_std = rsqrtf(variance + eps);
    
    // Phase 3: Normalize and scale using vectorized load/store
    for (int i = threadIdx.x; i < vec_hidden_size; i += blockDim.x) {
        VecT v_in, v_gamma, v_out;
        v_in.load(row_input + i * VecSize);
        v_gamma.load(gamma + i * VecSize);
        
        float vals[VecSize];
        #pragma unroll
        for (int j = 0; j < VecSize; j++) {
            float normalized = (static_cast<float>(v_in[j]) - mean) * inv_std;
            vals[j] = normalized * static_cast<float>(v_gamma[j]);
        }
        
        if (beta != nullptr) {
            VecT v_beta;
            v_beta.load(beta + i * VecSize);
            #pragma unroll
            for (int j = 0; j < VecSize; j++) {
                vals[j] += static_cast<float>(v_beta[j]);
            }
        }
        
        floatToVec<T, VecSize>(vals, v_out);
        v_out.store(row_output + i * VecSize);
    }
}

// =============================================================================
// Vectorized RMSNorm Kernel
// =============================================================================

template <typename T, int VecSize>
__global__ void rmsNormKernelVectorized(
    const T* __restrict__ input,
    T* __restrict__ output,
    const T* __restrict__ gamma,
    int hidden_size,
    float eps
) {
    using VecT = Vec<T, VecSize>;
    
    const int row_idx = blockIdx.x;
    const T* row_input = input + row_idx * hidden_size;
    T* row_output = output + row_idx * hidden_size;
    
    const int vec_hidden_size = hidden_size / VecSize;
    
    // Phase 1: Compute mean of squares using vectorized loads
    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < vec_hidden_size; i += blockDim.x) {
        VecT v;
        v.load(row_input + i * VecSize);
        local_sum_sq += vecSumSquares(v);
    }
    
    float mean_sq = blockReduceSum(local_sum_sq) / static_cast<float>(hidden_size);
    __syncthreads();
    
    __shared__ float shared_mean_sq;
    if (threadIdx.x == 0) shared_mean_sq = mean_sq;
    __syncthreads();
    mean_sq = shared_mean_sq;
    
    float inv_rms = rsqrtf(mean_sq + eps);
    
    // Phase 2: Normalize and scale using vectorized load/store
    for (int i = threadIdx.x; i < vec_hidden_size; i += blockDim.x) {
        VecT v_in, v_gamma, v_out;
        v_in.load(row_input + i * VecSize);
        v_gamma.load(gamma + i * VecSize);
        
        float vals[VecSize];
        #pragma unroll
        for (int j = 0; j < VecSize; j++) {
            vals[j] = static_cast<float>(v_in[j]) * inv_rms * static_cast<float>(v_gamma[j]);
        }
        
        floatToVec<T, VecSize>(vals, v_out);
        v_out.store(row_output + i * VecSize);
    }
}

// =============================================================================
// Vectorized BatchNorm1D Kernel
// =============================================================================

template <typename T, int VecSize>
__global__ void batchNorm1DKernelVectorized(
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
    using VecT = Vec<T, VecSize>;
    
    const int total_elements = batch_size * seq_len * channels;
    const int vec_channels = channels / VecSize;
    
    // Each thread processes VecSize elements at a time
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < total_elements / VecSize; 
         idx += blockDim.x * gridDim.x) {
        
        // Calculate position
        int flat_idx = idx * VecSize;
        int c_start = flat_idx % channels;
        
        // Check if this is an aligned vector access within channels
        if (c_start + VecSize <= channels && c_start % VecSize == 0) {
            VecT v_in, v_gamma, v_beta, v_mean, v_var, v_out;
            v_in.load(input + flat_idx);
            v_gamma.load(gamma + c_start);
            v_beta.load(beta + c_start);
            v_mean.load(running_mean + c_start);
            v_var.load(running_var + c_start);
            
            float vals[VecSize];
            #pragma unroll
            for (int j = 0; j < VecSize; j++) {
                float inv_std = rsqrtf(static_cast<float>(v_var[j]) + eps);
                float x = static_cast<float>(v_in[j]);
                float normalized = (x - static_cast<float>(v_mean[j])) * inv_std;
                vals[j] = normalized * static_cast<float>(v_gamma[j]) + static_cast<float>(v_beta[j]);
            }
            
            floatToVec<T, VecSize>(vals, v_out);
            v_out.store(output + flat_idx);
        } else {
            // Fall back to scalar for unaligned accesses
            for (int j = 0; j < VecSize && flat_idx + j < total_elements; j++) {
                int c = (flat_idx + j) % channels;
                float mean = static_cast<float>(running_mean[c]);
                float var = static_cast<float>(running_var[c]);
                float g = static_cast<float>(gamma[c]);
                float b = static_cast<float>(beta[c]);
                float inv_std = rsqrtf(var + eps);
                float x = static_cast<float>(input[flat_idx + j]);
                float normalized = (x - mean) * inv_std;
                output[flat_idx + j] = static_cast<T>(normalized * g + b);
            }
        }
    }
}

// =============================================================================
// Vectorized GroupNorm Kernel
// =============================================================================

template <typename T, int VecSize>
__global__ void groupNormKernelVectorized(
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
    using VecT = Vec<T, VecSize>;

    int channels_per_group = channels / num_groups;
    float inv_channels_per_group = 1.0f / static_cast<float>(channels_per_group);

    int idx = blockIdx.x;
    int seq_idx = idx % seq_len;
    int batch_idx = idx / seq_len;

    const T* row     = input  + (batch_idx * seq_len + seq_idx) * channels;
    T*       out_row = output + (batch_idx * seq_len + seq_idx) * channels;

    // threads_per_group: largest power-of-2 <= blockDim.x / num_groups
    // int raw_tpg = blockDim.x / num_groups;
    // int tpg = 1;
    // while ((tpg << 1) <= raw_tpg) tpg <<= 1;

    // thread_per_group should be a power-of-2
    int threads_per_group = blockDim.x / num_groups;

    int my_group  = threadIdx.x / threads_per_group;
    int local_tid = threadIdx.x % threads_per_group;

    int lane = threadIdx.x & (WARP_SIZE - 1);

    // Shared memory: [group_mean | group_inv_std | warp_buf (inter-warp case)]
    extern __shared__ float smem[];
    float* group_mean    = smem;
    float* group_inv_std = smem + num_groups;

    // ---- Phase 1: thread-local accumulation for assigned group (vectorized) ----
    float psum = 0.0f, psq = 0.0f;
    const T* gptr = row + my_group * channels_per_group;
    int vec_channels_per_group = channels_per_group / VecSize;
    for (int vec_i = local_tid; vec_i < vec_channels_per_group; vec_i += threads_per_group) {
        VecT v;
        v.load(gptr + vec_i * VecSize);
        psum += vecSum(v);
        psq  += vecSumSquares(v);
    }
    for (int i = vec_channels_per_group * VecSize + local_tid; i < channels_per_group; i += threads_per_group) {
        float v = static_cast<float>(gptr[i]);
        psum += v;
        psq  += v * v;
    }

    // ---- Phase 2: CUB-style tree reduction ----
    int seg = (threads_per_group < WARP_SIZE) ? threads_per_group : WARP_SIZE;
    for (int off = seg >> 1; off > 0; off >>= 1) {
        psum += __shfl_xor_sync(0xffffffff, psum, off);
        psq  += __shfl_xor_sync(0xffffffff, psq,  off);
    }

    if (threads_per_group <= WARP_SIZE) {
        if (local_tid == 0) {
            float mean = psum * inv_channels_per_group;
            group_mean[my_group]    = mean;
            group_inv_std[my_group] = rsqrtf(psq * inv_channels_per_group - mean * mean + eps);
        }
    } else {
        int warps_per_group = threads_per_group / WARP_SIZE;
        int warp_idx = local_tid / WARP_SIZE;
        float* wbuf = smem + 2 * num_groups;

        if (lane == 0) {
            int bi = my_group * warps_per_group + warp_idx;
            wbuf[bi]                    = psum;
            wbuf[num_groups * warps_per_group + bi] = psq;
        }
        __syncthreads();

        if (local_tid == 0) {
            float s = 0.0f, sq = 0.0f;
            int base    = my_group * warps_per_group;
            int sq_base = num_groups * warps_per_group + base;
            for (int w = 0; w < warps_per_group; ++w) {
                s  += wbuf[base + w];
                sq += wbuf[sq_base + w];
            }
            float mean = s * inv_channels_per_group;
            group_mean[my_group]    = mean;
            group_inv_std[my_group] = rsqrtf(sq * inv_channels_per_group - mean * mean + eps);
        }
    }
    __syncthreads();

    // ---- Phase 3: normalize and apply per-channel affine (vectorized where possible) ----
    int vec_channels = channels / VecSize;

    for (int vec_c = threadIdx.x; vec_c < vec_channels; vec_c += blockDim.x) {
        int c = vec_c * VecSize;
        int g = c / channels_per_group;
        VecT v_in, v_gamma, v_beta, v_out;
        v_in.load(row + c);
        v_gamma.load(gamma + c);
        v_beta.load(beta + c);
        float mean_g = group_mean[g];
        float inv_std_g = group_inv_std[g];
        float vals[VecSize];
        #pragma unroll
        for (int j = 0; j < VecSize; j++) {
            float n = (static_cast<float>(v_in[j]) - mean_g) * inv_std_g;
            vals[j] = n * static_cast<float>(v_gamma[j]) + static_cast<float>(v_beta[j]);
        }
        floatToVec<T, VecSize>(vals, v_out);
        v_out.store(out_row + c);
    }
}

// =============================================================================
// Vectorized Add + LayerNorm Fused Kernel
// =============================================================================

template <typename T, int VecSize>
__global__ void addLayerNormKernelVectorized(
    const T* __restrict__ input,
    const T* __restrict__ residual,
    T* __restrict__ output,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    int hidden_size,
    float eps
) {
    using VecT = Vec<T, VecSize>;
    
    const int row_idx = blockIdx.x;
    const T* row_input = input + row_idx * hidden_size;
    const T* row_residual = residual + row_idx * hidden_size;
    T* row_output = output + row_idx * hidden_size;
    
    const int vec_hidden_size = hidden_size / VecSize;
    
    // Phase 1: Compute mean using vectorized loads
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < vec_hidden_size; i += blockDim.x) {
        VecT v_in, v_res;
        v_in.load(row_input + i * VecSize);
        v_res.load(row_residual + i * VecSize);
        
        #pragma unroll
        for (int j = 0; j < VecSize; j++) {
            local_sum += static_cast<float>(v_in[j]) + static_cast<float>(v_res[j]);
        }
    }
    
    float mean = blockReduceSum(local_sum) / static_cast<float>(hidden_size);
    __syncthreads();
    
    __shared__ float shared_mean;
    if (threadIdx.x == 0) shared_mean = mean;
    __syncthreads();
    mean = shared_mean;
    
    // Phase 2: Compute variance using vectorized loads
    float local_var = 0.0f;
    for (int i = threadIdx.x; i < vec_hidden_size; i += blockDim.x) {
        VecT v_in, v_res;
        v_in.load(row_input + i * VecSize);
        v_res.load(row_residual + i * VecSize);
        
        #pragma unroll
        for (int j = 0; j < VecSize; j++) {
            float val = static_cast<float>(v_in[j]) + static_cast<float>(v_res[j]);
            float diff = val - mean;
            local_var += diff * diff;
        }
    }
    
    float variance = blockReduceSum(local_var) / static_cast<float>(hidden_size);
    __syncthreads();
    
    __shared__ float shared_var;
    if (threadIdx.x == 0) shared_var = variance;
    __syncthreads();
    variance = shared_var;
    
    float inv_std = rsqrtf(variance + eps);
    
    // Phase 3: Normalize and scale using vectorized load/store
    for (int i = threadIdx.x; i < vec_hidden_size; i += blockDim.x) {
        VecT v_in, v_res, v_gamma, v_out;
        v_in.load(row_input + i * VecSize);
        v_res.load(row_residual + i * VecSize);
        v_gamma.load(gamma + i * VecSize);
        
        float vals[VecSize];
        #pragma unroll
        for (int j = 0; j < VecSize; j++) {
            float val = static_cast<float>(v_in[j]) + static_cast<float>(v_res[j]);
            float normalized = (val - mean) * inv_std;
            vals[j] = normalized * static_cast<float>(v_gamma[j]);
        }
        
        if (beta != nullptr) {
            VecT v_beta;
            v_beta.load(beta + i * VecSize);
            #pragma unroll
            for (int j = 0; j < VecSize; j++) {
                vals[j] += static_cast<float>(v_beta[j]);
            }
        }
        
        floatToVec<T, VecSize>(vals, v_out);
        v_out.store(row_output + i * VecSize);
    }
}

// =============================================================================
// Dispatcher functions
// =============================================================================

// Helper to check if we can use vectorized kernels
template <typename T>
__host__ constexpr int getVecSize() {
    return VecTypeTrait<T>::VecSize;
}

template <typename T>
void invokeLayerNormTyped(const void* input, void* output,
                          const void* gamma, const void* beta,
                          int batch_size, int seq_len, int hidden_size,
                          float eps, cudaStream_t stream) {
    int num_rows = batch_size * seq_len;
    
    constexpr int VecSize = VecTypeTrait<T>::VecSize;
    
    // Choose block size based on hidden_size
    int block_size = std::min(hidden_size, MAX_THREADS_PER_BLOCK);
    block_size = ((block_size + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    
    // Use vectorized kernel if hidden_size is large enough and alignment is good
    // Vectorization requires hidden_size >= VecSize and pointer alignment
    bool use_vec = (hidden_size >= VecSize) && 
                   (reinterpret_cast<uintptr_t>(input) % (sizeof(T) * VecSize) == 0) &&
                   (reinterpret_cast<uintptr_t>(output) % (sizeof(T) * VecSize) == 0);
    
    if (use_vec) {
        // Adjust block size for vectorized kernel (fewer iterations needed)
        int vec_hidden = hidden_size / VecSize;
        block_size = std::min(vec_hidden, MAX_THREADS_PER_BLOCK);
        block_size = std::max(((block_size + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE, WARP_SIZE);
        
        layerNormKernelVectorized<T, VecSize><<<num_rows, block_size, 0, stream>>>(
            static_cast<const T*>(input),
            static_cast<T*>(output),
            static_cast<const T*>(gamma),
            static_cast<const T*>(beta),
            hidden_size,
            eps
        );
    } else {
        layerNormKernel<T><<<num_rows, block_size, 0, stream>>>(
            static_cast<const T*>(input),
            static_cast<T*>(output),
            static_cast<const T*>(gamma),
            static_cast<const T*>(beta),
            hidden_size,
            eps
        );
    }
}

template <typename T>
void invokeRMSNormTyped(const void* input, void* output,
                        const void* gamma,
                        int batch_size, int seq_len, int hidden_size,
                        float eps, cudaStream_t stream) {
    int num_rows = batch_size * seq_len;
    
    constexpr int VecSize = VecTypeTrait<T>::VecSize;
    
    int block_size = std::min(hidden_size, MAX_THREADS_PER_BLOCK);
    block_size = ((block_size + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    
    bool use_vec = (hidden_size >= VecSize) && 
                   (reinterpret_cast<uintptr_t>(input) % (sizeof(T) * VecSize) == 0) &&
                   (reinterpret_cast<uintptr_t>(output) % (sizeof(T) * VecSize) == 0);
    
    if (use_vec) {
        int vec_hidden = hidden_size / VecSize;
        block_size = std::min(vec_hidden, MAX_THREADS_PER_BLOCK);
        block_size = std::max(((block_size + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE, WARP_SIZE);
        
        rmsNormKernelVectorized<T, VecSize><<<num_rows, block_size, 0, stream>>>(
            static_cast<const T*>(input),
            static_cast<T*>(output),
            static_cast<const T*>(gamma),
            hidden_size,
            eps
        );
    } else {
        rmsNormKernel<T><<<num_rows, block_size, 0, stream>>>(
            static_cast<const T*>(input),
            static_cast<T*>(output),
            static_cast<const T*>(gamma),
            hidden_size,
            eps
        );
    }
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

template <typename T>
void invokeBatchNorm1DTyped(const void* input, void* output,
                            const void* gamma, const void* beta,
                            const void* running_mean, const void* running_var,
                            int batch_size, int seq_len, int channels,
                            float eps, cudaStream_t stream) {
    int total_elements = batch_size * seq_len * channels;
    int block_size = 256;
    
    constexpr int VecSize = VecTypeTrait<T>::VecSize;
    
    // Check if we can use vectorized kernel
    bool use_vec = (channels >= VecSize) && (channels % VecSize == 0) &&
                   (reinterpret_cast<uintptr_t>(input) % (sizeof(T) * VecSize) == 0) &&
                   (reinterpret_cast<uintptr_t>(output) % (sizeof(T) * VecSize) == 0);
    
    if (use_vec) {
        int vec_total = total_elements / VecSize;
        int grid_size = (vec_total + block_size - 1) / block_size;
        
        batchNorm1DKernelVectorized<T, VecSize><<<grid_size, block_size, 0, stream>>>(
            static_cast<const T*>(input),
            static_cast<T*>(output),
            static_cast<const T*>(gamma),
            static_cast<const T*>(beta),
            static_cast<const T*>(running_mean),
            static_cast<const T*>(running_var),
            batch_size, seq_len, channels, eps
        );
    } else {
        int grid_size = (total_elements + block_size - 1) / block_size;
        
        batchNorm1DKernel<T><<<grid_size, block_size, 0, stream>>>(
            static_cast<const T*>(input),
            static_cast<T*>(output),
            static_cast<const T*>(gamma),
            static_cast<const T*>(beta),
            static_cast<const T*>(running_mean),
            static_cast<const T*>(running_var),
            batch_size, seq_len, channels, eps
        );
    }
}

void invokeBatchNorm1D(const void* input, void* output,
                       const void* gamma, const void* beta,
                       const void* running_mean, const void* running_var,
                       int batch_size, int seq_len, int channels,
                       float eps, DataType dtype, cudaStream_t stream) {
    switch (dtype) {
        case DataType::FP32:
            invokeBatchNorm1DTyped<float>(input, output, gamma, beta,
                                          running_mean, running_var,
                                          batch_size, seq_len, channels, eps, stream);
            break;
        case DataType::FP16:
            invokeBatchNorm1DTyped<half>(input, output, gamma, beta,
                                         running_mean, running_var,
                                         batch_size, seq_len, channels, eps, stream);
            break;
        case DataType::BF16:
            invokeBatchNorm1DTyped<__nv_bfloat16>(input, output, gamma, beta,
                                                  running_mean, running_var,
                                                  batch_size, seq_len, channels, eps, stream);
            break;
        default:
            throw std::runtime_error("Unsupported data type for BatchNorm1D");
    }
}

template <typename T>
void invokeGroupNormTyped(const void* input, void* output,
                          const void* gamma, const void* beta,
                          int batch_size, int seq_len, int channels, int num_groups,
                          float eps, cudaStream_t stream) {
    int channels_per_group = channels / num_groups;

    constexpr int VecSize = VecTypeTrait<T>::VecSize;

    int num_blocks = batch_size * seq_len;
    int block_size = std::min(channels, MAX_THREADS_PER_BLOCK);
    block_size = std::max(((block_size + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE,
                          WARP_SIZE);
    int num_warps = block_size / WARP_SIZE;
    size_t smem_bytes = sizeof(float) * (2 * num_groups + 2 * num_warps);

    bool use_vec = (channels_per_group >= VecSize) &&
                   (reinterpret_cast<uintptr_t>(input) % (sizeof(T) * VecSize) == 0) &&
                   (reinterpret_cast<uintptr_t>(output) % (sizeof(T) * VecSize) == 0) &&
                   (reinterpret_cast<uintptr_t>(gamma) % (sizeof(T) * VecSize) == 0) &&
                   (reinterpret_cast<uintptr_t>(beta) % (sizeof(T) * VecSize) == 0);

    if (use_vec) {
        groupNormKernelVectorized<T, VecSize><<<num_blocks, block_size, smem_bytes, stream>>>(
            static_cast<const T*>(input),
            static_cast<T*>(output),
            static_cast<const T*>(gamma),
            static_cast<const T*>(beta),
            batch_size, seq_len, channels, num_groups, eps
        );
    } else {
        groupNormKernel<T><<<num_blocks, block_size, smem_bytes, stream>>>(
            static_cast<const T*>(input),
            static_cast<T*>(output),
            static_cast<const T*>(gamma),
            static_cast<const T*>(beta),
            batch_size, seq_len, channels, num_groups, eps
        );
    }
}

void invokeGroupNorm(const void* input, void* output,
                     const void* gamma, const void* beta,
                     int batch_size, int seq_len, int channels, int num_groups,
                     float eps, DataType dtype, cudaStream_t stream) {
    switch (dtype) {
        case DataType::FP32:
            invokeGroupNormTyped<float>(input, output, gamma, beta,
                                        batch_size, seq_len, channels, num_groups, eps, stream);
            break;
        case DataType::FP16:
            invokeGroupNormTyped<half>(input, output, gamma, beta,
                                       batch_size, seq_len, channels, num_groups, eps, stream);
            break;
        case DataType::BF16:
            invokeGroupNormTyped<__nv_bfloat16>(input, output, gamma, beta,
                                                batch_size, seq_len, channels, num_groups, eps, stream);
            break;
        default:
            throw std::runtime_error("Unsupported data type for GroupNorm");
    }
}

template <typename T>
void invokeAddLayerNormTyped(const void* input, const void* residual, void* output,
                             const void* gamma, const void* beta,
                             int batch_size, int seq_len, int hidden_size,
                             float eps, cudaStream_t stream) {
    int num_rows = batch_size * seq_len;
    
    constexpr int VecSize = VecTypeTrait<T>::VecSize;
    
    int block_size = std::min(hidden_size, MAX_THREADS_PER_BLOCK);
    block_size = ((block_size + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    
    // Check if we can use vectorized kernel
    bool use_vec = (hidden_size >= VecSize) &&
                   (reinterpret_cast<uintptr_t>(input) % (sizeof(T) * VecSize) == 0) &&
                   (reinterpret_cast<uintptr_t>(residual) % (sizeof(T) * VecSize) == 0) &&
                   (reinterpret_cast<uintptr_t>(output) % (sizeof(T) * VecSize) == 0);
    
    if (use_vec) {
        int vec_hidden = hidden_size / VecSize;
        block_size = std::min(vec_hidden, MAX_THREADS_PER_BLOCK);
        block_size = std::max(((block_size + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE, WARP_SIZE);
        
        addLayerNormKernelVectorized<T, VecSize><<<num_rows, block_size, 0, stream>>>(
            static_cast<const T*>(input),
            static_cast<const T*>(residual),
            static_cast<T*>(output),
            static_cast<const T*>(gamma),
            static_cast<const T*>(beta),
            hidden_size, eps
        );
    } else {
        addLayerNormKernel<T><<<num_rows, block_size, 0, stream>>>(
            static_cast<const T*>(input),
            static_cast<const T*>(residual),
            static_cast<T*>(output),
            static_cast<const T*>(gamma),
            static_cast<const T*>(beta),
            hidden_size, eps
        );
    }
}

void invokeAddLayerNorm(const void* input, const void* residual, void* output,
                        const void* gamma, const void* beta,
                        int batch_size, int seq_len, int hidden_size,
                        float eps, DataType dtype, cudaStream_t stream) {
    switch (dtype) {
        case DataType::FP32:
            invokeAddLayerNormTyped<float>(input, residual, output, gamma, beta,
                                           batch_size, seq_len, hidden_size, eps, stream);
            break;
        case DataType::FP16:
            invokeAddLayerNormTyped<half>(input, residual, output, gamma, beta,
                                          batch_size, seq_len, hidden_size, eps, stream);
            break;
        case DataType::BF16:
            invokeAddLayerNormTyped<__nv_bfloat16>(input, residual, output, gamma, beta,
                                                   batch_size, seq_len, hidden_size, eps, stream);
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
