// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Pure CUDA normalization kernels — no framework dependencies.
// Includes LayerNorm, RMSNorm, BatchNorm1D, GroupNorm, AddLayerNorm,
// and fused norm+activation variants.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <oasr/common/math.h>
#include <oasr/common/types.h>
#include <oasr/common/vec_dtypes.h>

namespace oasr {
namespace norm {

// =============================================================================
// Reduction Utilities (inlined from allreduce.h)
// =============================================================================

constexpr int WARP_SIZE = 32;

template <typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

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
// Architecture Configuration & Helpers
// =============================================================================

constexpr int MAX_THREADS_PER_BLOCK = 1024;

// Broadcast a scalar from thread 0 to all threads via shared memory.
// Includes barriers before (flush prior shared-mem ops) and after (publish).
__device__ __forceinline__ float broadcastFromThread0(float value, float* smem) {
    __syncthreads();
    if (threadIdx.x == 0)
        *smem = value;
    __syncthreads();
    return *smem;
}

// Compute warp-aligned block size clamped to [WARP_SIZE, max_threads].
inline int alignedBlockSize(int num_elements, int max_threads = MAX_THREADS_PER_BLOCK) {
    int bs = std::min(num_elements, max_threads);
    return std::max(((bs + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE, WARP_SIZE);
}

// Check pointer alignment for vectorized access of VecSize elements.
template <typename T, int VecSize>
inline bool isAligned(const void* ptr) {
    return reinterpret_cast<uintptr_t>(ptr) % (sizeof(T) * VecSize) == 0;
}

// Type tag for dispatch helpers.
template <typename T>
struct TypeTag {
    using type = T;
};

// =============================================================================
// LayerNorm Kernel
// =============================================================================

template <typename T, int VecSize>
__global__ void layerNormKernel(const T* __restrict__ input, T* __restrict__ output,
                                const T* __restrict__ weight, const T* __restrict__ bias,
                                int hidden_size, float eps) {
    using VecT = oasr::Vec<T, VecSize>;

    const int row_idx = blockIdx.x;
    const T* row_input = input + row_idx * hidden_size;
    T* row_output = output + row_idx * hidden_size;

    const int vec_hidden_size = hidden_size / VecSize;

    __shared__ float smem[2];

    // Phase 1: Compute mean using vectorized loads
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < vec_hidden_size; i += blockDim.x) {
        VecT v;
        v.load(row_input + i * VecSize);
        local_sum += oasr::vecSum(v);
    }

    float mean =
        broadcastFromThread0(blockReduceSum(local_sum) / static_cast<float>(hidden_size), &smem[0]);

    // Phase 2: Compute variance using vectorized loads
    float local_var = 0.0f;
    for (int i = threadIdx.x; i < vec_hidden_size; i += blockDim.x) {
        VecT v;
        v.load(row_input + i * VecSize);
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            float diff = static_cast<float>(v[j]) - mean;
            local_var = std::fmaf(diff, diff, local_var);
        }
    }

    float inv_std =
        rsqrtf(broadcastFromThread0(blockReduceSum(local_var) / static_cast<float>(hidden_size),
                                    &smem[1]) +
               eps);

    // Phase 3: Normalize and scale using vectorized load/store
    for (int i = threadIdx.x; i < vec_hidden_size; i += blockDim.x) {
        VecT v_in, v_weight;
        v_in.load(row_input + i * VecSize);
        v_weight.load(weight + i * VecSize);

        oasr::Vec<float, VecSize> vals;
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            vals[j] =
                (static_cast<float>(v_in[j]) - mean) * inv_std * static_cast<float>(v_weight[j]);
        }

        if (bias != nullptr) {
            VecT v_bias;
            v_bias.load(bias + i * VecSize);
#pragma unroll
            for (int j = 0; j < VecSize; j++) {
                vals[j] += static_cast<float>(v_bias[j]);
            }
        }

        oasr::vecCast<T>(vals).store(row_output + i * VecSize);
    }
}

// =============================================================================
// RMSNorm Kernel
// =============================================================================

template <typename T, int VecSize>
__global__ void rmsNormKernel(const T* __restrict__ input, T* __restrict__ output,
                              const T* __restrict__ weight, const T* __restrict__ bias,
                              int hidden_size, float eps) {
    using VecT = oasr::Vec<T, VecSize>;

    const int row_idx = blockIdx.x;
    const T* row_input = input + row_idx * hidden_size;
    T* row_output = output + row_idx * hidden_size;

    const int vec_hidden_size = hidden_size / VecSize;

    __shared__ float smem;

    // Phase 1: Compute mean of squares using vectorized loads
    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < vec_hidden_size; i += blockDim.x) {
        VecT v;
        v.load(row_input + i * VecSize);
        local_sum_sq += oasr::vecSumSquares(v);
    }

    float inv_rms =
        rsqrtf(broadcastFromThread0(blockReduceSum(local_sum_sq) / static_cast<float>(hidden_size),
                                    &smem) +
               eps);

    // Phase 2: Normalize and scale using vectorized load/store
    for (int i = threadIdx.x; i < vec_hidden_size; i += blockDim.x) {
        VecT v_in, v_weight;
        v_in.load(row_input + i * VecSize);
        v_weight.load(weight + i * VecSize);

        oasr::Vec<float, VecSize> vals;
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            vals[j] = static_cast<float>(v_in[j]) * inv_rms * static_cast<float>(v_weight[j]);
        }

        if (bias != nullptr) {
            VecT v_bias;
            v_bias.load(bias + i * VecSize);
#pragma unroll
            for (int j = 0; j < VecSize; j++) {
                vals[j] += static_cast<float>(v_bias[j]);
            }
        }
        oasr::vecCast<T>(vals).store(row_output + i * VecSize);
    }
}

// =============================================================================
// BatchNorm1D Kernel (inference mode)
// =============================================================================

template <typename T, int VecSize>
__global__ void batchNorm1DKernel(const T* __restrict__ input, T* __restrict__ output,
                                  const T* __restrict__ weight, const T* __restrict__ bias,
                                  const T* __restrict__ running_mean,
                                  const T* __restrict__ running_var, int batch_size, int seq_len,
                                  int channels, float eps) {
    using VecT = oasr::Vec<T, VecSize>;

    const int total_elements = batch_size * seq_len * channels;

    // Each thread processes VecSize elements at a time
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements / VecSize;
         idx += blockDim.x * gridDim.x) {
        // Calculate position
        int flat_idx = idx * VecSize;
        int c_start = flat_idx % channels;

        VecT v_in, v_weight, v_bias, v_mean, v_var;
        v_in.load(input + flat_idx);
        v_weight.load(weight + c_start);
        v_bias.load(bias + c_start);
        v_mean.load(running_mean + c_start);
        v_var.load(running_var + c_start);

        oasr::Vec<float, VecSize> vals;
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            vals[j] = (static_cast<float>(v_in[j]) - static_cast<float>(v_mean[j])) *
                          rsqrtf(static_cast<float>(v_var[j]) + eps) *
                          static_cast<float>(v_weight[j]) +
                      static_cast<float>(v_bias[j]);
        }

        oasr::vecCast<T>(vals).store(output + flat_idx);
    }
}

// =============================================================================
// GroupNorm Kernel
// =============================================================================

template <typename T, int VecSize>
__global__ void groupNormKernel(const T* __restrict__ input, T* __restrict__ output,
                                const T* __restrict__ weight, const T* __restrict__ bias,
                                int batch_size, int seq_len, int channels, int num_groups,
                                float eps) {
    using VecT = oasr::Vec<T, VecSize>;

    int channels_per_group = channels / num_groups;
    float inv_channels_per_group = 1.0f / static_cast<float>(channels_per_group);

    int idx = blockIdx.x;
    int seq_idx = idx % seq_len;
    int batch_idx = idx / seq_len;

    const T* row = input + (batch_idx * seq_len + seq_idx) * channels;
    T* out_row = output + (batch_idx * seq_len + seq_idx) * channels;

    // thread_per_group should be a power-of-2
    int threads_per_group = blockDim.x / num_groups;

    int my_group = threadIdx.x / threads_per_group;
    int local_tid = threadIdx.x % threads_per_group;

    int lane = threadIdx.x & (WARP_SIZE - 1);

    // Shared memory: [group_mean | group_inv_std | warp_buf (inter-warp case)]
    extern __shared__ float smem[];
    float* group_mean = smem;
    float* group_inv_std = smem + num_groups;

    // ---- Phase 1: thread-local accumulation for assigned group (vectorized) ----
    float psum = 0.0f, psq = 0.0f;
    const T* gptr = row + my_group * channels_per_group;
    int vec_channels_per_group = channels_per_group / VecSize;
    for (int vec_i = local_tid; vec_i < vec_channels_per_group; vec_i += threads_per_group) {
        VecT v;
        v.load(gptr + vec_i * VecSize);
        psum += oasr::vecSum(v);
        psq += oasr::vecSumSquares(v);
    }
    for (int i = vec_channels_per_group * VecSize + local_tid; i < channels_per_group;
         i += threads_per_group) {
        float v = static_cast<float>(gptr[i]);
        psum += v;
        psq += v * v;
    }

    // ---- Phase 2: Tree reduction ----
    int seg = (threads_per_group < WARP_SIZE) ? threads_per_group : WARP_SIZE;
    for (int off = seg >> 1; off > 0; off >>= 1) {
        psum += __shfl_xor_sync(0xffffffff, psum, off);
        psq += __shfl_xor_sync(0xffffffff, psq, off);
    }

    if (threads_per_group <= WARP_SIZE) {
        if (local_tid == 0) {
            float mean = psum * inv_channels_per_group;
            group_mean[my_group] = mean;
            group_inv_std[my_group] = rsqrtf(psq * inv_channels_per_group - mean * mean + eps);
        }
    } else {
        int warps_per_group = threads_per_group / WARP_SIZE;
        int warp_idx = local_tid / WARP_SIZE;
        float* wbuf = smem + 2 * num_groups;

        if (lane == 0) {
            int bi = my_group * warps_per_group + warp_idx;
            wbuf[bi] = psum;
            wbuf[num_groups * warps_per_group + bi] = psq;
        }
        __syncthreads();

        if (local_tid == 0) {
            float s = 0.0f, sq = 0.0f;
            int base = my_group * warps_per_group;
            int sq_base = num_groups * warps_per_group + base;
            for (int w = 0; w < warps_per_group; ++w) {
                s += wbuf[base + w];
                sq += wbuf[sq_base + w];
            }
            float mean = s * inv_channels_per_group;
            group_mean[my_group] = mean;
            group_inv_std[my_group] = rsqrtf(sq * inv_channels_per_group - mean * mean + eps);
        }
    }
    __syncthreads();

    // ---- Phase 3: normalize and apply per-channel affine (vectorized where possible) ----
    int vec_channels = channels / VecSize;

    for (int vec_c = threadIdx.x; vec_c < vec_channels; vec_c += blockDim.x) {
        int c = vec_c * VecSize;
        int g = c / channels_per_group;
        VecT v_in, v_weight, v_bias;
        v_in.load(row + c);
        v_weight.load(weight + c);
        v_bias.load(bias + c);
        float mean_g = group_mean[g];
        float inv_std_g = group_inv_std[g];
        oasr::Vec<float, VecSize> vals;
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            vals[j] = (static_cast<float>(v_in[j]) - mean_g) * inv_std_g *
                          static_cast<float>(v_weight[j]) +
                      static_cast<float>(v_bias[j]);
        }
        oasr::vecCast<T>(vals).store(out_row + c);
    }
}

// =============================================================================
// Add + LayerNorm Fused Kernel
// =============================================================================

template <typename T, int VecSize>
__global__ void addLayerNormKernel(const T* __restrict__ input, const T* __restrict__ residual,
                                   T* __restrict__ output, const T* __restrict__ weight,
                                   const T* __restrict__ bias, int hidden_size, float eps) {
    using VecT = oasr::Vec<T, VecSize>;

    const int row_idx = blockIdx.x;
    const T* row_input = input + row_idx * hidden_size;
    const T* row_residual = residual + row_idx * hidden_size;
    T* row_output = output + row_idx * hidden_size;

    const int vec_hidden_size = hidden_size / VecSize;

    __shared__ float smem[2];

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

    float mean =
        broadcastFromThread0(blockReduceSum(local_sum) / static_cast<float>(hidden_size), &smem[0]);

    // Phase 2: Compute variance using vectorized loads
    float local_var = 0.0f;
    for (int i = threadIdx.x; i < vec_hidden_size; i += blockDim.x) {
        VecT v_in, v_res;
        v_in.load(row_input + i * VecSize);
        v_res.load(row_residual + i * VecSize);

#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            float diff = static_cast<float>(v_in[j]) + static_cast<float>(v_res[j]) - mean;
            local_var = std::fmaf(diff, diff, local_var);
        }
    }

    float inv_std =
        rsqrtf(broadcastFromThread0(blockReduceSum(local_var) / static_cast<float>(hidden_size),
                                    &smem[1]) +
               eps);

    // Phase 3: Normalize and scale using vectorized load/store
    for (int i = threadIdx.x; i < vec_hidden_size; i += blockDim.x) {
        VecT v_in, v_res, v_weight;
        v_in.load(row_input + i * VecSize);
        v_res.load(row_residual + i * VecSize);
        v_weight.load(weight + i * VecSize);

        oasr::Vec<float, VecSize> vals;
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            vals[j] = (static_cast<float>(v_in[j]) + static_cast<float>(v_res[j]) - mean) *
                      inv_std * static_cast<float>(v_weight[j]);
        }

        if (bias != nullptr) {
            VecT v_bias;
            v_bias.load(bias + i * VecSize);
#pragma unroll
            for (int j = 0; j < VecSize; j++) {
                vals[j] += static_cast<float>(v_bias[j]);
            }
        }

        oasr::vecCast<T>(vals).store(row_output + i * VecSize);
    }
}

// =============================================================================
// Activation Dispatch Helper
// =============================================================================

template <typename Fn>
void dispatchActivation(ActivationType act, Fn&& fn) {
    switch (act) {
        case ActivationType::RELU:
            fn(oasr::ReluActivation{});
            break;
        case ActivationType::GELU:
            fn(oasr::GeluActivation{});
            break;
        case ActivationType::SWISH:
            fn(oasr::SwishActivation{});
            break;
        default:
            break;
    }
}

// =============================================================================
// Fused LayerNorm + Activation Kernel
// =============================================================================

template <typename T, int VecSize, typename ActFn>
__global__ void layerNormActKernel(const T* __restrict__ input, T* __restrict__ output,
                                   const T* __restrict__ weight, const T* __restrict__ bias,
                                   int hidden_size, float eps) {
    using VecT = oasr::Vec<T, VecSize>;
    ActFn act;

    const int row_idx = blockIdx.x;
    const T* row_input = input + row_idx * hidden_size;
    T* row_output = output + row_idx * hidden_size;

    const int vec_hidden_size = hidden_size / VecSize;

    __shared__ float smem[2];

    // Phase 1: Compute mean
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < vec_hidden_size; i += blockDim.x) {
        VecT v;
        v.load(row_input + i * VecSize);
        local_sum += oasr::vecSum(v);
    }

    float mean =
        broadcastFromThread0(blockReduceSum(local_sum) / static_cast<float>(hidden_size), &smem[0]);

    // Phase 2: Compute variance
    float local_var = 0.0f;
    for (int i = threadIdx.x; i < vec_hidden_size; i += blockDim.x) {
        VecT v;
        v.load(row_input + i * VecSize);
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            float diff = static_cast<float>(v[j]) - mean;
            local_var = std::fmaf(diff, diff, local_var);
        }
    }

    float inv_std =
        rsqrtf(broadcastFromThread0(blockReduceSum(local_var) / static_cast<float>(hidden_size),
                                    &smem[1]) +
               eps);

    // Phase 3: Normalize, scale, and apply activation
    for (int i = threadIdx.x; i < vec_hidden_size; i += blockDim.x) {
        VecT v_in, v_weight;
        v_in.load(row_input + i * VecSize);
        v_weight.load(weight + i * VecSize);

        oasr::Vec<float, VecSize> vals;
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            vals[j] =
                (static_cast<float>(v_in[j]) - mean) * inv_std * static_cast<float>(v_weight[j]);
        }

        if (bias != nullptr) {
            VecT v_bias;
            v_bias.load(bias + i * VecSize);
#pragma unroll
            for (int j = 0; j < VecSize; j++) {
                vals[j] += static_cast<float>(v_bias[j]);
            }
        }

#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            vals[j] = act(vals[j]);
        }

        oasr::vecCast<T>(vals).store(row_output + i * VecSize);
    }
}

// =============================================================================
// Fused RMSNorm + Activation Kernel
// =============================================================================

template <typename T, int VecSize, typename ActFn>
__global__ void rmsNormActKernel(const T* __restrict__ input, T* __restrict__ output,
                                 const T* __restrict__ weight, const T* __restrict__ bias,
                                 int hidden_size, float eps) {
    using VecT = oasr::Vec<T, VecSize>;
    ActFn act;

    const int row_idx = blockIdx.x;
    const T* row_input = input + row_idx * hidden_size;
    T* row_output = output + row_idx * hidden_size;

    const int vec_hidden_size = hidden_size / VecSize;

    __shared__ float smem;

    // Phase 1: Compute mean of squares
    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < vec_hidden_size; i += blockDim.x) {
        VecT v;
        v.load(row_input + i * VecSize);
        local_sum_sq += oasr::vecSumSquares(v);
    }

    float inv_rms =
        rsqrtf(broadcastFromThread0(blockReduceSum(local_sum_sq) / static_cast<float>(hidden_size),
                                    &smem) +
               eps);

    // Phase 2: Normalize, scale, and apply activation
    for (int i = threadIdx.x; i < vec_hidden_size; i += blockDim.x) {
        VecT v_in, v_weight;
        v_in.load(row_input + i * VecSize);
        v_weight.load(weight + i * VecSize);

        oasr::Vec<float, VecSize> vals;
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            vals[j] = static_cast<float>(v_in[j]) * inv_rms * static_cast<float>(v_weight[j]);
        }

        if (bias != nullptr) {
            VecT v_bias;
            v_bias.load(bias + i * VecSize);
#pragma unroll
            for (int j = 0; j < VecSize; j++) {
                vals[j] += static_cast<float>(v_bias[j]);
            }
        }

#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            vals[j] = act(vals[j]);
        }

        oasr::vecCast<T>(vals).store(row_output + i * VecSize);
    }
}

// =============================================================================
// Fused BatchNorm + Activation Kernel (vectorized)
// =============================================================================

template <typename T, int VecSize, typename ActFn>
__global__ void batchNormActKernel(const T* __restrict__ input, T* __restrict__ output,
                                   const T* __restrict__ weight, const T* __restrict__ bias,
                                   const T* __restrict__ running_mean,
                                   const T* __restrict__ running_var, int batch_size, int seq_len,
                                   int channels, float eps) {
    using VecT = oasr::Vec<T, VecSize>;
    ActFn act;

    int total_elements = batch_size * seq_len * channels;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements / VecSize;
         idx += blockDim.x * gridDim.x) {
        int flat_idx = idx * VecSize;
        int c_start = flat_idx % channels;

        VecT v_in, v_weight, v_bias, v_mean, v_var;
        v_in.load(input + flat_idx);
        v_weight.load(weight + c_start);
        v_bias.load(bias + c_start);
        v_mean.load(running_mean + c_start);
        v_var.load(running_var + c_start);

        oasr::Vec<float, VecSize> vals;
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            vals[j] = act((static_cast<float>(v_in[j]) - static_cast<float>(v_mean[j])) *
                              rsqrtf(static_cast<float>(v_var[j]) + eps) *
                              static_cast<float>(v_weight[j]) +
                          static_cast<float>(v_bias[j]));
        }

        oasr::vecCast<T>(vals).store(output + flat_idx);
    }
}

// =============================================================================
// CMVN (Cepstral Mean and Variance Normalization) Kernel
// =============================================================================

template <typename T, int VecSize>
__global__ void cmvnKernel(const T* __restrict__ input, T* __restrict__ output,
                           const T* __restrict__ mean, const T* __restrict__ istd, int num_rows,
                           int num_cols) {
    using VecT = oasr::Vec<T, VecSize>;

    const int total_elements = num_rows * num_cols;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements / VecSize;
         idx += blockDim.x * gridDim.x) {
        int flat_idx = idx * VecSize;
        int c_start = flat_idx % num_cols;

        VecT v_in, v_mean, v_istd;
        v_in.load(input + flat_idx);
        v_mean.load(mean + c_start);
        v_istd.load(istd + c_start);

        oasr::Vec<float, VecSize> vals;
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            vals[j] = (static_cast<float>(v_in[j]) - static_cast<float>(v_mean[j])) *
                      static_cast<float>(v_istd[j]);
        }

        oasr::vecCast<T>(vals).store(output + flat_idx);
    }
}

// =============================================================================
// Typed Launcher Functions (raw pointer interface, returning cudaError_t)
// =============================================================================

// ---- LayerNorm ----

template <typename T>
cudaError_t LayerNorm(const T* input, const T* weight, const T* bias, T* output,
                      unsigned int num_rows, unsigned int hidden_size, float eps,
                      cudaStream_t stream) {
    constexpr int VecSize = oasr::VecTypeTrait<T>::VecSize;

    bool use_vec = (hidden_size >= static_cast<unsigned int>(VecSize)) &&
                   isAligned<T, VecSize>(input) && isAligned<T, VecSize>(output);

    if (use_vec) {
        int block_size = alignedBlockSize(static_cast<int>(hidden_size) / VecSize);
        layerNormKernel<T, VecSize><<<num_rows, block_size, 0, stream>>>(
            input, output, weight, bias, static_cast<int>(hidden_size), eps);
    } else {
        int block_size = alignedBlockSize(static_cast<int>(hidden_size));
        layerNormKernel<T, 1><<<num_rows, block_size, 0, stream>>>(
            input, output, weight, bias, static_cast<int>(hidden_size), eps);
    }
    return cudaGetLastError();
}

// ---- RMSNorm ----

template <typename T>
cudaError_t RMSNorm(const T* input, const T* weight, const T* bias, T* output,
                    unsigned int num_rows, unsigned int hidden_size, float eps,
                    cudaStream_t stream) {
    constexpr int VecSize = oasr::VecTypeTrait<T>::VecSize;

    bool use_vec = (hidden_size >= static_cast<unsigned int>(VecSize)) &&
                   isAligned<T, VecSize>(input) && isAligned<T, VecSize>(output);

    if (use_vec) {
        int block_size = alignedBlockSize(static_cast<int>(hidden_size) / VecSize);
        rmsNormKernel<T, VecSize><<<num_rows, block_size, 0, stream>>>(
            input, output, weight, bias, static_cast<int>(hidden_size), eps);
    } else {
        int block_size = alignedBlockSize(static_cast<int>(hidden_size));
        rmsNormKernel<T, 1><<<num_rows, block_size, 0, stream>>>(
            input, output, weight, bias, static_cast<int>(hidden_size), eps);
    }
    return cudaGetLastError();
}

// ---- BatchNorm1D ----

template <typename T>
cudaError_t BatchNorm1D(const T* input, const T* weight, const T* bias, const T* running_mean,
                        const T* running_var, T* output, unsigned int batch_size,
                        unsigned int seq_len, unsigned int channels, float eps,
                        cudaStream_t stream) {
    constexpr int VecSize = oasr::VecTypeTrait<T>::VecSize;

    int total_elements = static_cast<int>(batch_size * seq_len * channels);
    int block_size = 256;

    bool use_vec = (channels >= static_cast<unsigned int>(VecSize)) && (channels % VecSize == 0) &&
                   isAligned<T, VecSize>(input) && isAligned<T, VecSize>(output);

    if (use_vec) {
        int grid_size = (total_elements / VecSize + block_size - 1) / block_size;
        batchNorm1DKernel<T, VecSize><<<grid_size, block_size, 0, stream>>>(
            input, output, weight, bias, running_mean, running_var, static_cast<int>(batch_size),
            static_cast<int>(seq_len), static_cast<int>(channels), eps);
    } else {
        int grid_size = (total_elements + block_size - 1) / block_size;
        batchNorm1DKernel<T, 1><<<grid_size, block_size, 0, stream>>>(
            input, output, weight, bias, running_mean, running_var, static_cast<int>(batch_size),
            static_cast<int>(seq_len), static_cast<int>(channels), eps);
    }
    return cudaGetLastError();
}

// ---- GroupNorm ----

template <typename T>
cudaError_t GroupNorm(const T* input, const T* weight, const T* bias, T* output,
                      unsigned int batch_size, unsigned int seq_len, unsigned int channels,
                      unsigned int num_groups, float eps, cudaStream_t stream) {
    constexpr int VecSize = oasr::VecTypeTrait<T>::VecSize;

    int channels_per_group = static_cast<int>(channels / num_groups);
    int num_blocks = static_cast<int>(batch_size * seq_len);
    int block_size = alignedBlockSize(static_cast<int>(channels));
    int num_warps = block_size / WARP_SIZE;
    size_t smem_bytes = sizeof(float) * (2 * num_groups + 2 * static_cast<unsigned int>(num_warps));

    bool use_vec = (static_cast<int>(channels_per_group) >= VecSize) &&
                   isAligned<T, VecSize>(input) && isAligned<T, VecSize>(output) &&
                   isAligned<T, VecSize>(weight) && isAligned<T, VecSize>(bias);

    if (use_vec) {
        groupNormKernel<T, VecSize><<<num_blocks, block_size, smem_bytes, stream>>>(
            input, output, weight, bias, static_cast<int>(batch_size), static_cast<int>(seq_len),
            static_cast<int>(channels), static_cast<int>(num_groups), eps);
    } else {
        groupNormKernel<T, 1><<<num_blocks, block_size, smem_bytes, stream>>>(
            input, output, weight, bias, static_cast<int>(batch_size), static_cast<int>(seq_len),
            static_cast<int>(channels), static_cast<int>(num_groups), eps);
    }
    return cudaGetLastError();
}

// ---- AddLayerNorm ----

template <typename T>
cudaError_t AddLayerNorm(const T* input, const T* residual, const T* weight, const T* bias,
                         T* output, unsigned int num_rows, unsigned int hidden_size, float eps,
                         cudaStream_t stream) {
    constexpr int VecSize = oasr::VecTypeTrait<T>::VecSize;

    bool use_vec = (hidden_size >= static_cast<unsigned int>(VecSize)) &&
                   isAligned<T, VecSize>(input) && isAligned<T, VecSize>(residual) &&
                   isAligned<T, VecSize>(output);

    if (use_vec) {
        int block_size = alignedBlockSize(static_cast<int>(hidden_size) / VecSize);
        addLayerNormKernel<T, VecSize><<<num_rows, block_size, 0, stream>>>(
            input, residual, output, weight, bias, static_cast<int>(hidden_size), eps);
    } else {
        int block_size = alignedBlockSize(static_cast<int>(hidden_size));
        addLayerNormKernel<T, 1><<<num_rows, block_size, 0, stream>>>(
            input, residual, output, weight, bias, static_cast<int>(hidden_size), eps);
    }
    return cudaGetLastError();
}

// ---- LayerNormActivation ----

template <typename T>
cudaError_t LayerNormActivation(const T* input, const T* weight, const T* bias, T* output,
                                unsigned int num_rows, unsigned int hidden_size, float eps,
                                ActivationType activation, cudaStream_t stream) {
    constexpr int VecSize = oasr::VecTypeTrait<T>::VecSize;

    bool use_vec = (hidden_size >= static_cast<unsigned int>(VecSize)) &&
                   isAligned<T, VecSize>(input) && isAligned<T, VecSize>(output);

    cudaError_t err = cudaSuccess;
    dispatchActivation(activation, [&](auto act) {
        using ActFn = decltype(act);
        if (use_vec) {
            int block_size = alignedBlockSize(static_cast<int>(hidden_size) / VecSize);
            layerNormActKernel<T, VecSize, ActFn><<<num_rows, block_size, 0, stream>>>(
                input, output, weight, bias, static_cast<int>(hidden_size), eps);
        } else {
            int block_size = alignedBlockSize(static_cast<int>(hidden_size));
            layerNormActKernel<T, 1, ActFn><<<num_rows, block_size, 0, stream>>>(
                input, output, weight, bias, static_cast<int>(hidden_size), eps);
        }
        err = cudaGetLastError();
    });
    return err;
}

// ---- RMSNormActivation ----

template <typename T>
cudaError_t RMSNormActivation(const T* input, const T* weight, const T* bias, T* output,
                              unsigned int num_rows, unsigned int hidden_size, float eps,
                              ActivationType activation, cudaStream_t stream) {
    constexpr int VecSize = oasr::VecTypeTrait<T>::VecSize;

    bool use_vec = (hidden_size >= static_cast<unsigned int>(VecSize)) &&
                   isAligned<T, VecSize>(input) && isAligned<T, VecSize>(output);

    cudaError_t err = cudaSuccess;
    dispatchActivation(activation, [&](auto act) {
        using ActFn = decltype(act);
        if (use_vec) {
            int block_size = alignedBlockSize(static_cast<int>(hidden_size) / VecSize);
            rmsNormActKernel<T, VecSize, ActFn><<<num_rows, block_size, 0, stream>>>(
                input, output, weight, bias, static_cast<int>(hidden_size), eps);
        } else {
            int block_size = alignedBlockSize(static_cast<int>(hidden_size));
            rmsNormActKernel<T, 1, ActFn><<<num_rows, block_size, 0, stream>>>(
                input, output, weight, bias, static_cast<int>(hidden_size), eps);
        }
        err = cudaGetLastError();
    });
    return err;
}

// ---- CMVN ----

template <typename T>
cudaError_t CMVN(const T* input, const T* mean, const T* istd, T* output, unsigned int num_rows,
                 unsigned int num_cols, cudaStream_t stream) {
    constexpr int VecSize = oasr::VecTypeTrait<T>::VecSize;

    int total_elements = static_cast<int>(num_rows * num_cols);
    int block_size = 256;

    bool use_vec = (num_cols >= static_cast<unsigned int>(VecSize)) && (num_cols % VecSize == 0) &&
                   isAligned<T, VecSize>(input) && isAligned<T, VecSize>(output) &&
                   isAligned<T, VecSize>(mean) && isAligned<T, VecSize>(istd);

    if (use_vec) {
        int grid_size = (total_elements / VecSize + block_size - 1) / block_size;
        cmvnKernel<T, VecSize><<<grid_size, block_size, 0, stream>>>(
            input, output, mean, istd, static_cast<int>(num_rows), static_cast<int>(num_cols));
    } else {
        int grid_size = (total_elements + block_size - 1) / block_size;
        cmvnKernel<T, 1><<<grid_size, block_size, 0, stream>>>(
            input, output, mean, istd, static_cast<int>(num_rows), static_cast<int>(num_cols));
    }
    return cudaGetLastError();
}

// ---- BatchNormActivation ----

template <typename T>
cudaError_t BatchNormActivation(const T* input, const T* weight, const T* bias,
                                const T* running_mean, const T* running_var, T* output,
                                unsigned int batch_size, unsigned int seq_len,
                                unsigned int channels, float eps, ActivationType activation,
                                cudaStream_t stream) {
    constexpr int VecSize = oasr::VecTypeTrait<T>::VecSize;

    int total_elements = static_cast<int>(batch_size * seq_len * channels);
    int block_size = 256;

    bool use_vec = (channels >= static_cast<unsigned int>(VecSize)) && (channels % VecSize == 0) &&
                   isAligned<T, VecSize>(input) && isAligned<T, VecSize>(output);

    cudaError_t err = cudaSuccess;
    dispatchActivation(activation, [&](auto act) {
        using ActFn = decltype(act);
        if (use_vec) {
            int grid_size = (total_elements / VecSize + block_size - 1) / block_size;
            batchNormActKernel<T, VecSize, ActFn><<<grid_size, block_size, 0, stream>>>(
                input, output, weight, bias, running_mean, running_var,
                static_cast<int>(batch_size), static_cast<int>(seq_len), static_cast<int>(channels),
                eps);
        } else {
            int grid_size = (total_elements + block_size - 1) / block_size;
            batchNormActKernel<T, 1, ActFn><<<grid_size, block_size, 0, stream>>>(
                input, output, weight, bias, running_mean, running_var,
                static_cast<int>(batch_size), static_cast<int>(seq_len), static_cast<int>(channels),
                eps);
        }
        err = cudaGetLastError();
    });
    return err;
}

// ---- BatchNormSwish (convenience wrapper) ----

template <typename T>
cudaError_t BatchNormSwish(const T* input, const T* weight, const T* bias, const T* running_mean,
                           const T* running_var, T* output, unsigned int batch_size,
                           unsigned int seq_len, unsigned int channels, float eps,
                           cudaStream_t stream) {
    return BatchNormActivation<T>(input, weight, bias, running_mean, running_var, output,
                                  batch_size, seq_len, channels, eps, ActivationType::SWISH,
                                  stream);
}

}  // namespace norm
}  // namespace oasr
