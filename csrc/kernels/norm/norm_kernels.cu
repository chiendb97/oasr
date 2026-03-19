// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <torch/extension.h>

#include "kernels/common/math.h"
#include "kernels/common/vec_dtypes.h"
#include "kernels/norm/norm_kernels.h"
#include "kernels/reduction/allreduce.h"

namespace oasr {
namespace kernels {

// Import vector types from oasr namespace
using oasr::Vec;
using oasr::vecCast;
using oasr::vecSum;
using oasr::vecSumSquares;
using oasr::VecTypeTrait;

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

// Dispatch over float / half / bfloat16 scalar types.
template <typename T>
struct TypeTag {
    using type = T;
};

template <typename Fn>
void dispatchFloatTypes(torch::ScalarType dtype, const char* op_name, Fn&& fn) {
    switch (dtype) {
        case torch::kFloat32:
            fn(TypeTag<float>{});
            break;
        case torch::kHalf:
            fn(TypeTag<half>{});
            break;
        case torch::kBFloat16:
            fn(TypeTag<__nv_bfloat16>{});
            break;
        default:
            throw std::runtime_error(std::string("Unsupported data type for ") + op_name);
    }
}

// =============================================================================
// LayerNorm Kernel
// =============================================================================

template <typename T, int VecSize>
__global__ void layerNormKernel(const T* __restrict__ input, T* __restrict__ output,
                                const T* __restrict__ weight, const T* __restrict__ bias,
                                int hidden_size, float eps) {
    using VecT = Vec<T, VecSize>;

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
        local_sum += vecSum(v);
    }

    float mean = broadcastFromThread0(
        blockReduceSum(local_sum) / static_cast<float>(hidden_size), &smem[0]);

    // Phase 2: Compute variance using vectorized loads
    float local_var = 0.0f;
    for (int i = threadIdx.x; i < vec_hidden_size; i += blockDim.x) {
        VecT v;
        v.load(row_input + i * VecSize);
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            float diff = static_cast<float>(v[j]) - mean;
            local_var += diff * diff;
        }
    }

    float inv_std = rsqrtf(
        broadcastFromThread0(
            blockReduceSum(local_var) / static_cast<float>(hidden_size), &smem[1])
        + eps);

    // Phase 3: Normalize and scale using vectorized load/store
    for (int i = threadIdx.x; i < vec_hidden_size; i += blockDim.x) {
        VecT v_in, v_weight;
        v_in.load(row_input + i * VecSize);
        v_weight.load(weight + i * VecSize);

        Vec<float, VecSize> vals;
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            float normalized = (static_cast<float>(v_in[j]) - mean) * inv_std;
            vals[j] = normalized * static_cast<float>(v_weight[j]);
        }

        if (bias != nullptr) {
            VecT v_bias;
            v_bias.load(bias + i * VecSize);
#pragma unroll
            for (int j = 0; j < VecSize; j++) {
                vals[j] += static_cast<float>(v_bias[j]);
            }
        }

        vecCast<T>(vals).store(row_output + i * VecSize);
    }
}

// Optimized kernel for hidden_size <= 1024 (fits in one block)
template <typename T, int HIDDEN_SIZE>
__global__ void layerNormKernelSmall(const T* __restrict__ input, T* __restrict__ output,
                                     const T* __restrict__ weight, const T* __restrict__ bias,
                                     float eps) {
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

    __shared__ float smem[2];

    // Reduce to get mean
    float mean = broadcastFromThread0(
        blockReduceSum(local_sum) / static_cast<float>(HIDDEN_SIZE), &smem[0]);

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

    float inv_std = rsqrtf(
        broadcastFromThread0(
            blockReduceSum(local_var) / static_cast<float>(HIDDEN_SIZE), &smem[1])
        + eps);

// Write output
#pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int idx = threadIdx.x + i * blockDim.x;
        if (idx < HIDDEN_SIZE) {
            float normalized = (vals[i] - mean) * inv_std;
            float w = static_cast<float>(weight[idx]);
            float scaled = normalized * w;
            if (bias != nullptr) {
                scaled += static_cast<float>(bias[idx]);
            }
            row_output[idx] = static_cast<T>(scaled);
        }
    }
}

// =============================================================================
// RMSNorm Kernel
// =============================================================================

template <typename T, int VecSize>
__global__ void rmsNormKernel(const T* __restrict__ input, T* __restrict__ output,
                              const T* __restrict__ weight, const T* __restrict__ bias,
                              int hidden_size, float eps) {
    using VecT = Vec<T, VecSize>;

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
        local_sum_sq += vecSumSquares(v);
    }

    float inv_rms = rsqrtf(
        broadcastFromThread0(
            blockReduceSum(local_sum_sq) / static_cast<float>(hidden_size), &smem)
        + eps);

    // Phase 2: Normalize and scale using vectorized load/store
    for (int i = threadIdx.x; i < vec_hidden_size; i += blockDim.x) {
        VecT v_in, v_weight;
        v_in.load(row_input + i * VecSize);
        v_weight.load(weight + i * VecSize);

        Vec<float, VecSize> vals;
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
        vecCast<T>(vals).store(row_output + i * VecSize);
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
    using VecT = Vec<T, VecSize>;

    const int total_elements = batch_size * seq_len * channels;

    // Each thread processes VecSize elements at a time
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements / VecSize;
         idx += blockDim.x * gridDim.x) {
        // Calculate position
        int flat_idx = idx * VecSize;
        int c_start = flat_idx % channels;

        // Check if this is an aligned vector access within channels
        if (c_start + VecSize <= channels && c_start % VecSize == 0) {
            VecT v_in, v_weight, v_bias, v_mean, v_var;
            v_in.load(input + flat_idx);
            v_weight.load(weight + c_start);
            v_bias.load(bias + c_start);
            v_mean.load(running_mean + c_start);
            v_var.load(running_var + c_start);

            Vec<float, VecSize> vals;
#pragma unroll
            for (int j = 0; j < VecSize; j++) {
                float inv_std = rsqrtf(static_cast<float>(v_var[j]) + eps);
                float x = static_cast<float>(v_in[j]);
                float normalized = (x - static_cast<float>(v_mean[j])) * inv_std;
                vals[j] =
                    normalized * static_cast<float>(v_weight[j]) + static_cast<float>(v_bias[j]);
            }

            vecCast<T>(vals).store(output + flat_idx);
        } else {
            // Fall back to scalar for unaligned accesses
            for (int j = 0; j < VecSize && flat_idx + j < total_elements; j++) {
                int c = (flat_idx + j) % channels;
                float mean = static_cast<float>(running_mean[c]);
                float var = static_cast<float>(running_var[c]);
                float g = static_cast<float>(weight[c]);
                float b = static_cast<float>(bias[c]);
                float inv_std = rsqrtf(var + eps);
                float x = static_cast<float>(input[flat_idx + j]);
                float normalized = (x - mean) * inv_std;
                output[flat_idx + j] = static_cast<T>(normalized * g + b);
            }
        }
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
    using VecT = Vec<T, VecSize>;

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
        psum += vecSum(v);
        psq += vecSumSquares(v);
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
        Vec<float, VecSize> vals;
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            float n = (static_cast<float>(v_in[j]) - mean_g) * inv_std_g;
            vals[j] = n * static_cast<float>(v_weight[j]) + static_cast<float>(v_bias[j]);
        }
        vecCast<T>(vals).store(out_row + c);
    }
}

// =============================================================================
// Add + LayerNorm Fused Kernel
// =============================================================================

template <typename T, int VecSize>
__global__ void addLayerNormKernel(const T* __restrict__ input, const T* __restrict__ residual,
                                   T* __restrict__ output, const T* __restrict__ weight,
                                   const T* __restrict__ bias, int hidden_size, float eps) {
    using VecT = Vec<T, VecSize>;

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

    float mean = broadcastFromThread0(
        blockReduceSum(local_sum) / static_cast<float>(hidden_size), &smem[0]);

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

    float inv_std = rsqrtf(
        broadcastFromThread0(
            blockReduceSum(local_var) / static_cast<float>(hidden_size), &smem[1])
        + eps);

    // Phase 3: Normalize and scale using vectorized load/store
    for (int i = threadIdx.x; i < vec_hidden_size; i += blockDim.x) {
        VecT v_in, v_res, v_weight;
        v_in.load(row_input + i * VecSize);
        v_res.load(row_residual + i * VecSize);
        v_weight.load(weight + i * VecSize);

        Vec<float, VecSize> vals;
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            float val = static_cast<float>(v_in[j]) + static_cast<float>(v_res[j]);
            float normalized = (val - mean) * inv_std;
            vals[j] = normalized * static_cast<float>(v_weight[j]);
        }

        if (bias != nullptr) {
            VecT v_bias;
            v_bias.load(bias + i * VecSize);
#pragma unroll
            for (int j = 0; j < VecSize; j++) {
                vals[j] += static_cast<float>(v_bias[j]);
            }
        }

        vecCast<T>(vals).store(row_output + i * VecSize);
    }
}

// Dispatch ActivationType enum to compile-time activation functor.
template <typename Fn>
void dispatchActivation(ActivationType act, Fn&& fn) {
    switch (act) {
        case ActivationType::RELU:
            fn(ReluActivation{});
            break;
        case ActivationType::GELU:
            fn(GeluActivation{});
            break;
        case ActivationType::SWISH:
            fn(SwishActivation{});
            break;
        default:
            throw std::runtime_error("Unsupported activation type");
    }
}

// =============================================================================
// Fused LayerNorm + Activation Kernel
// =============================================================================

template <typename T, int VecSize, typename ActFn>
__global__ void layerNormActKernel(const T* __restrict__ input, T* __restrict__ output,
                                   const T* __restrict__ weight, const T* __restrict__ bias,
                                   int hidden_size, float eps) {
    using VecT = Vec<T, VecSize>;
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
        local_sum += vecSum(v);
    }

    float mean = broadcastFromThread0(
        blockReduceSum(local_sum) / static_cast<float>(hidden_size), &smem[0]);

    // Phase 2: Compute variance
    float local_var = 0.0f;
    for (int i = threadIdx.x; i < vec_hidden_size; i += blockDim.x) {
        VecT v;
        v.load(row_input + i * VecSize);
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            float diff = static_cast<float>(v[j]) - mean;
            local_var += diff * diff;
        }
    }

    float inv_std = rsqrtf(
        broadcastFromThread0(
            blockReduceSum(local_var) / static_cast<float>(hidden_size), &smem[1])
        + eps);

    // Phase 3: Normalize, scale, and apply activation
    for (int i = threadIdx.x; i < vec_hidden_size; i += blockDim.x) {
        VecT v_in, v_weight;
        v_in.load(row_input + i * VecSize);
        v_weight.load(weight + i * VecSize);

        Vec<float, VecSize> vals;
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            float normalized = (static_cast<float>(v_in[j]) - mean) * inv_std;
            vals[j] = normalized * static_cast<float>(v_weight[j]);
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

        vecCast<T>(vals).store(row_output + i * VecSize);
    }
}

// =============================================================================
// Fused RMSNorm + Activation Kernel
// =============================================================================

template <typename T, int VecSize, typename ActFn>
__global__ void rmsNormActKernel(const T* __restrict__ input, T* __restrict__ output,
                                 const T* __restrict__ weight, const T* __restrict__ bias,
                                 int hidden_size, float eps) {
    using VecT = Vec<T, VecSize>;
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
        local_sum_sq += vecSumSquares(v);
    }

    float inv_rms = rsqrtf(
        broadcastFromThread0(
            blockReduceSum(local_sum_sq) / static_cast<float>(hidden_size), &smem)
        + eps);

    // Phase 2: Normalize, scale, and apply activation
    for (int i = threadIdx.x; i < vec_hidden_size; i += blockDim.x) {
        VecT v_in, v_weight;
        v_in.load(row_input + i * VecSize);
        v_weight.load(weight + i * VecSize);

        Vec<float, VecSize> vals;
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

        vecCast<T>(vals).store(row_output + i * VecSize);
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
    using VecT = Vec<T, VecSize>;
    ActFn act;

    int total_elements = batch_size * seq_len * channels;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements / VecSize;
         idx += blockDim.x * gridDim.x) {
        int flat_idx = idx * VecSize;
        int c_start = flat_idx % channels;

        if (c_start + VecSize <= channels && c_start % VecSize == 0) {
            VecT v_in, v_weight, v_bias, v_mean, v_var;
            v_in.load(input + flat_idx);
            v_weight.load(weight + c_start);
            v_bias.load(bias + c_start);
            v_mean.load(running_mean + c_start);
            v_var.load(running_var + c_start);

            Vec<float, VecSize> vals;
#pragma unroll
            for (int j = 0; j < VecSize; j++) {
                float inv_std = rsqrtf(static_cast<float>(v_var[j]) + eps);
                float x = static_cast<float>(v_in[j]);
                float bn_out = (x - static_cast<float>(v_mean[j])) * inv_std
                               * static_cast<float>(v_weight[j]) + static_cast<float>(v_bias[j]);
                vals[j] = act(bn_out);
            }

            vecCast<T>(vals).store(output + flat_idx);
        } else {
            // Fall back to scalar for unaligned accesses
            for (int j = 0; j < VecSize && flat_idx + j < total_elements; j++) {
                int c = (flat_idx + j) % channels;
                float inv_std = rsqrtf(static_cast<float>(running_var[c]) + eps);
                float x = static_cast<float>(input[flat_idx + j]);
                float bn_out = (x - static_cast<float>(running_mean[c])) * inv_std
                               * static_cast<float>(weight[c]) + static_cast<float>(bias[c]);
                output[flat_idx + j] = static_cast<T>(act(bn_out));
            }
        }
    }
}

// =============================================================================
// Typed Dispatcher Functions
// =============================================================================

template <typename T>
void invokeLayerNormTyped(const torch::Tensor& input, torch::Tensor& output,
                          const torch::Tensor& weight, const torch::Tensor& bias, float eps,
                          cudaStream_t stream) {
    int batch_size = input.size(0);
    int seq_len = input.size(1);
    int hidden_size = input.size(2);
    int num_rows = batch_size * seq_len;

    constexpr int VecSize = VecTypeTrait<T>::VecSize;

    const T* input_ptr = static_cast<const T*>(input.data_ptr());
    T* output_ptr = static_cast<T*>(output.data_ptr());
    const T* weight_ptr = static_cast<const T*>(weight.data_ptr());
    const T* bias_ptr = bias.defined() ? static_cast<const T*>(bias.data_ptr()) : nullptr;

    bool use_vec = (hidden_size >= VecSize) && isAligned<T, VecSize>(input_ptr)
                   && isAligned<T, VecSize>(output_ptr);

    if (use_vec) {
        int block_size = alignedBlockSize(hidden_size / VecSize);
        layerNormKernel<T, VecSize><<<num_rows, block_size, 0, stream>>>(
            input_ptr, output_ptr, weight_ptr, bias_ptr, hidden_size, eps);
    } else {
        int block_size = alignedBlockSize(hidden_size);
        layerNormKernel<T, 1><<<num_rows, block_size, 0, stream>>>(
            input_ptr, output_ptr, weight_ptr, bias_ptr, hidden_size, eps);
    }
}

template <typename T>
void invokeRMSNormTyped(const torch::Tensor& input, torch::Tensor& output,
                        const torch::Tensor& weight, const torch::Tensor& bias, float eps,
                        cudaStream_t stream) {
    int batch_size = input.size(0);
    int seq_len = input.size(1);
    int hidden_size = input.size(2);
    int num_rows = batch_size * seq_len;

    constexpr int VecSize = VecTypeTrait<T>::VecSize;

    const T* input_ptr = static_cast<const T*>(input.data_ptr());
    T* output_ptr = static_cast<T*>(output.data_ptr());
    const T* weight_ptr = static_cast<const T*>(weight.data_ptr());
    const T* bias_ptr = bias.defined() ? static_cast<const T*>(bias.data_ptr()) : nullptr;

    bool use_vec = (hidden_size >= VecSize) && isAligned<T, VecSize>(input_ptr)
                   && isAligned<T, VecSize>(output_ptr);

    if (use_vec) {
        int block_size = alignedBlockSize(hidden_size / VecSize);
        rmsNormKernel<T, VecSize><<<num_rows, block_size, 0, stream>>>(
            input_ptr, output_ptr, weight_ptr, bias_ptr, hidden_size, eps);
    } else {
        int block_size = alignedBlockSize(hidden_size);
        rmsNormKernel<T, 1><<<num_rows, block_size, 0, stream>>>(input_ptr, output_ptr, weight_ptr,
                                                                   bias_ptr, hidden_size, eps);
    }
}

template <typename T>
void invokeBatchNorm1DTyped(const torch::Tensor& input, torch::Tensor& output,
                            const torch::Tensor& weight, const torch::Tensor& bias,
                            const torch::Tensor& running_mean, const torch::Tensor& running_var,
                            float eps, cudaStream_t stream) {
    int batch_size = input.size(0);
    int seq_len = input.size(1);
    int channels = input.size(2);
    int total_elements = batch_size * seq_len * channels;
    int block_size = 256;

    constexpr int VecSize = VecTypeTrait<T>::VecSize;

    const T* input_ptr = static_cast<const T*>(input.data_ptr());
    T* output_ptr = static_cast<T*>(output.data_ptr());
    const T* weight_ptr = static_cast<const T*>(weight.data_ptr());
    const T* bias_ptr = bias.defined() ? static_cast<const T*>(bias.data_ptr()) : nullptr;
    const T* running_mean_ptr = static_cast<const T*>(running_mean.data_ptr());
    const T* running_var_ptr = static_cast<const T*>(running_var.data_ptr());

    bool use_vec = (channels >= VecSize) && (channels % VecSize == 0)
                   && isAligned<T, VecSize>(input_ptr) && isAligned<T, VecSize>(output_ptr);

    if (use_vec) {
        int grid_size = (total_elements / VecSize + block_size - 1) / block_size;
        batchNorm1DKernel<T, VecSize><<<grid_size, block_size, 0, stream>>>(
            input_ptr, output_ptr, weight_ptr, bias_ptr, running_mean_ptr, running_var_ptr,
            batch_size, seq_len, channels, eps);
    } else {
        int grid_size = (total_elements + block_size - 1) / block_size;
        batchNorm1DKernel<T, 1><<<grid_size, block_size, 0, stream>>>(
            input_ptr, output_ptr, weight_ptr, bias_ptr, running_mean_ptr, running_var_ptr,
            batch_size, seq_len, channels, eps);
    }
}

template <typename T>
void invokeGroupNormTyped(const torch::Tensor& input, torch::Tensor& output,
                          const torch::Tensor& weight, const torch::Tensor& bias, int num_groups,
                          float eps, cudaStream_t stream) {
    int batch_size = input.size(0);
    int seq_len = input.size(1);
    int channels = input.size(2);
    int channels_per_group = channels / num_groups;

    constexpr int VecSize = VecTypeTrait<T>::VecSize;

    const T* input_ptr = static_cast<const T*>(input.data_ptr());
    T* output_ptr = static_cast<T*>(output.data_ptr());
    const T* weight_ptr = static_cast<const T*>(weight.data_ptr());
    const T* bias_ptr = static_cast<const T*>(bias.data_ptr());

    int num_blocks = batch_size * seq_len;
    int block_size = alignedBlockSize(channels);
    int num_warps = block_size / WARP_SIZE;
    size_t smem_bytes = sizeof(float) * (2 * num_groups + 2 * num_warps);

    bool use_vec = (channels_per_group >= VecSize) && isAligned<T, VecSize>(input_ptr)
                   && isAligned<T, VecSize>(output_ptr) && isAligned<T, VecSize>(weight_ptr)
                   && isAligned<T, VecSize>(bias_ptr);

    if (use_vec) {
        groupNormKernel<T, VecSize><<<num_blocks, block_size, smem_bytes, stream>>>(
            input_ptr, output_ptr, weight_ptr, bias_ptr, batch_size, seq_len, channels, num_groups,
            eps);
    } else {
        groupNormKernel<T, 1><<<num_blocks, block_size, smem_bytes, stream>>>(
            input_ptr, output_ptr, weight_ptr, bias_ptr, batch_size, seq_len, channels, num_groups,
            eps);
    }
}

template <typename T>
void invokeAddLayerNormTyped(const torch::Tensor& input, const torch::Tensor& residual,
                             torch::Tensor& output, const torch::Tensor& weight,
                             const torch::Tensor& bias, float eps, cudaStream_t stream) {
    int batch_size = input.size(0);
    int seq_len = input.size(1);
    int hidden_size = input.size(2);
    int num_rows = batch_size * seq_len;

    constexpr int VecSize = VecTypeTrait<T>::VecSize;

    const T* input_ptr = static_cast<const T*>(input.data_ptr());
    const T* residual_ptr = static_cast<const T*>(residual.data_ptr());
    T* output_ptr = static_cast<T*>(output.data_ptr());
    const T* weight_ptr = static_cast<const T*>(weight.data_ptr());
    const T* bias_ptr = bias.defined() ? static_cast<const T*>(bias.data_ptr()) : nullptr;

    bool use_vec = (hidden_size >= VecSize) && isAligned<T, VecSize>(input_ptr)
                   && isAligned<T, VecSize>(residual_ptr) && isAligned<T, VecSize>(output_ptr);

    if (use_vec) {
        int block_size = alignedBlockSize(hidden_size / VecSize);
        addLayerNormKernel<T, VecSize><<<num_rows, block_size, 0, stream>>>(
            input_ptr, residual_ptr, output_ptr, weight_ptr, bias_ptr, hidden_size, eps);
    } else {
        int block_size = alignedBlockSize(hidden_size);
        addLayerNormKernel<T, 1><<<num_rows, block_size, 0, stream>>>(
            input_ptr, residual_ptr, output_ptr, weight_ptr, bias_ptr, hidden_size, eps);
    }
}

template <typename T, typename ActFn>
void invokeLayerNormActTyped(const torch::Tensor& input, torch::Tensor& output,
                             const torch::Tensor& weight, const torch::Tensor& bias, float eps,
                             ActFn, cudaStream_t stream) {
    int batch_size = input.size(0);
    int seq_len = input.size(1);
    int hidden_size = input.size(2);
    int num_rows = batch_size * seq_len;

    constexpr int VecSize = VecTypeTrait<T>::VecSize;

    const T* input_ptr = static_cast<const T*>(input.data_ptr());
    T* output_ptr = static_cast<T*>(output.data_ptr());
    const T* weight_ptr = static_cast<const T*>(weight.data_ptr());
    const T* bias_ptr = bias.defined() ? static_cast<const T*>(bias.data_ptr()) : nullptr;

    bool use_vec = (hidden_size >= VecSize) && isAligned<T, VecSize>(input_ptr)
                   && isAligned<T, VecSize>(output_ptr);

    if (use_vec) {
        int block_size = alignedBlockSize(hidden_size / VecSize);
        layerNormActKernel<T, VecSize, ActFn><<<num_rows, block_size, 0, stream>>>(
            input_ptr, output_ptr, weight_ptr, bias_ptr, hidden_size, eps);
    } else {
        int block_size = alignedBlockSize(hidden_size);
        layerNormActKernel<T, 1, ActFn><<<num_rows, block_size, 0, stream>>>(
            input_ptr, output_ptr, weight_ptr, bias_ptr, hidden_size, eps);
    }
}

template <typename T, typename ActFn>
void invokeRMSNormActTyped(const torch::Tensor& input, torch::Tensor& output,
                           const torch::Tensor& weight, const torch::Tensor& bias, float eps,
                           ActFn, cudaStream_t stream) {
    int batch_size = input.size(0);
    int seq_len = input.size(1);
    int hidden_size = input.size(2);
    int num_rows = batch_size * seq_len;

    constexpr int VecSize = VecTypeTrait<T>::VecSize;

    const T* input_ptr = static_cast<const T*>(input.data_ptr());
    T* output_ptr = static_cast<T*>(output.data_ptr());
    const T* weight_ptr = static_cast<const T*>(weight.data_ptr());
    const T* bias_ptr = bias.defined() ? static_cast<const T*>(bias.data_ptr()) : nullptr;

    bool use_vec = (hidden_size >= VecSize) && isAligned<T, VecSize>(input_ptr)
                   && isAligned<T, VecSize>(output_ptr);

    if (use_vec) {
        int block_size = alignedBlockSize(hidden_size / VecSize);
        rmsNormActKernel<T, VecSize, ActFn><<<num_rows, block_size, 0, stream>>>(
            input_ptr, output_ptr, weight_ptr, bias_ptr, hidden_size, eps);
    } else {
        int block_size = alignedBlockSize(hidden_size);
        rmsNormActKernel<T, 1, ActFn><<<num_rows, block_size, 0, stream>>>(
            input_ptr, output_ptr, weight_ptr, bias_ptr, hidden_size, eps);
    }
}

template <typename T, typename ActFn>
void invokeBatchNormActTyped(const torch::Tensor& input, torch::Tensor& output,
                             const torch::Tensor& weight, const torch::Tensor& bias,
                             const torch::Tensor& running_mean, const torch::Tensor& running_var,
                             float eps, ActFn, cudaStream_t stream) {
    int batch_size = input.size(0);
    int seq_len = input.size(1);
    int channels = input.size(2);
    int total_elements = batch_size * seq_len * channels;
    int block_size = 256;

    constexpr int VecSize = VecTypeTrait<T>::VecSize;

    const T* input_ptr = static_cast<const T*>(input.data_ptr());
    T* output_ptr = static_cast<T*>(output.data_ptr());
    const T* weight_ptr = static_cast<const T*>(weight.data_ptr());
    const T* bias_ptr = static_cast<const T*>(bias.data_ptr());
    const T* running_mean_ptr = static_cast<const T*>(running_mean.data_ptr());
    const T* running_var_ptr = static_cast<const T*>(running_var.data_ptr());

    bool use_vec = (channels >= VecSize) && (channels % VecSize == 0)
                   && isAligned<T, VecSize>(input_ptr) && isAligned<T, VecSize>(output_ptr);

    if (use_vec) {
        int grid_size = (total_elements / VecSize + block_size - 1) / block_size;
        batchNormActKernel<T, VecSize, ActFn><<<grid_size, block_size, 0, stream>>>(
            input_ptr, output_ptr, weight_ptr, bias_ptr, running_mean_ptr, running_var_ptr,
            batch_size, seq_len, channels, eps);
    } else {
        int grid_size = (total_elements + block_size - 1) / block_size;
        batchNormActKernel<T, 1, ActFn><<<grid_size, block_size, 0, stream>>>(
            input_ptr, output_ptr, weight_ptr, bias_ptr, running_mean_ptr, running_var_ptr,
            batch_size, seq_len, channels, eps);
    }
}

// =============================================================================
// Public API implementations
// =============================================================================

torch::Tensor invokeLayerNorm(const torch::Tensor& input, const torch::Tensor& weight,
                              const torch::Tensor& bias, float eps, cudaStream_t stream) {
    auto output = torch::empty_like(input);
    dispatchFloatTypes(input.scalar_type(), "LayerNorm", [&](auto tag) {
        using T = typename decltype(tag)::type;
        invokeLayerNormTyped<T>(input, output, weight, bias, eps, stream);
    });
    return output;
}

torch::Tensor invokeRMSNorm(const torch::Tensor& input, const torch::Tensor& weight,
                            const torch::Tensor& bias, float eps, cudaStream_t stream) {
    auto output = torch::empty_like(input);
    dispatchFloatTypes(input.scalar_type(), "RMSNorm", [&](auto tag) {
        using T = typename decltype(tag)::type;
        invokeRMSNormTyped<T>(input, output, weight, bias, eps, stream);
    });
    return output;
}

torch::Tensor invokeBatchNorm1D(const torch::Tensor& input, const torch::Tensor& weight,
                                const torch::Tensor& bias, const torch::Tensor& running_mean,
                                const torch::Tensor& running_var, float eps, cudaStream_t stream) {
    auto output = torch::empty_like(input);
    dispatchFloatTypes(input.scalar_type(), "BatchNorm1D", [&](auto tag) {
        using T = typename decltype(tag)::type;
        invokeBatchNorm1DTyped<T>(input, output, weight, bias, running_mean, running_var, eps,
                                  stream);
    });
    return output;
}

torch::Tensor invokeGroupNorm(const torch::Tensor& input, const torch::Tensor& weight,
                              const torch::Tensor& bias, int num_groups, float eps,
                              cudaStream_t stream) {
    auto output = torch::empty_like(input);
    dispatchFloatTypes(input.scalar_type(), "GroupNorm", [&](auto tag) {
        using T = typename decltype(tag)::type;
        invokeGroupNormTyped<T>(input, output, weight, bias, num_groups, eps, stream);
    });
    return output;
}

torch::Tensor invokeAddLayerNorm(const torch::Tensor& input, const torch::Tensor& residual,
                                 const torch::Tensor& weight, const torch::Tensor& bias, float eps,
                                 cudaStream_t stream) {
    auto output = torch::empty_like(input);
    dispatchFloatTypes(input.scalar_type(), "AddLayerNorm", [&](auto tag) {
        using T = typename decltype(tag)::type;
        invokeAddLayerNormTyped<T>(input, residual, output, weight, bias, eps, stream);
    });
    return output;
}

torch::Tensor invokeLayerNormActivation(const torch::Tensor& input, const torch::Tensor& weight,
                                        const torch::Tensor& bias, float eps,
                                        ActivationType activation, cudaStream_t stream) {
    auto output = torch::empty_like(input);
    dispatchFloatTypes(input.scalar_type(), "LayerNormActivation", [&](auto tag) {
        using T = typename decltype(tag)::type;
        dispatchActivation(activation, [&](auto act) {
            invokeLayerNormActTyped<T>(input, output, weight, bias, eps, act, stream);
        });
    });
    return output;
}

torch::Tensor invokeRMSNormActivation(const torch::Tensor& input, const torch::Tensor& weight,
                                       const torch::Tensor& bias, float eps,
                                       ActivationType activation, cudaStream_t stream) {
    auto output = torch::empty_like(input);
    dispatchFloatTypes(input.scalar_type(), "RMSNormActivation", [&](auto tag) {
        using T = typename decltype(tag)::type;
        dispatchActivation(activation, [&](auto act) {
            invokeRMSNormActTyped<T>(input, output, weight, bias, eps, act, stream);
        });
    });
    return output;
}

torch::Tensor invokeBatchNormActivation(const torch::Tensor& input, const torch::Tensor& weight,
                                         const torch::Tensor& bias, const torch::Tensor& running_mean,
                                         const torch::Tensor& running_var, float eps,
                                         ActivationType activation, cudaStream_t stream) {
    auto output = torch::empty_like(input);
    dispatchFloatTypes(input.scalar_type(), "BatchNormActivation", [&](auto tag) {
        using T = typename decltype(tag)::type;
        dispatchActivation(activation, [&](auto act) {
            invokeBatchNormActTyped<T>(input, output, weight, bias, running_mean, running_var, eps,
                                       act, stream);
        });
    });
    return output;
}

torch::Tensor invokeBatchNormSwish(const torch::Tensor& input, const torch::Tensor& weight,
                                   const torch::Tensor& bias, const torch::Tensor& running_mean,
                                   const torch::Tensor& running_var, float eps,
                                   cudaStream_t stream) {
    return invokeBatchNormActivation(input, weight, bias, running_mean, running_var, eps,
                                     ActivationType::SWISH, stream);
}

// Explicit template instantiations
template void invokeLayerNormTyped<float>(const torch::Tensor&, torch::Tensor&,
                                          const torch::Tensor&, const torch::Tensor&, float,
                                          cudaStream_t);
template void invokeLayerNormTyped<half>(const torch::Tensor&, torch::Tensor&, const torch::Tensor&,
                                         const torch::Tensor&, float, cudaStream_t);
template void invokeLayerNormTyped<__nv_bfloat16>(const torch::Tensor&, torch::Tensor&,
                                                  const torch::Tensor&, const torch::Tensor&, float,
                                                  cudaStream_t);

template void invokeRMSNormTyped<float>(const torch::Tensor&, torch::Tensor&, const torch::Tensor&,
                                        const torch::Tensor&, float, cudaStream_t);
template void invokeRMSNormTyped<half>(const torch::Tensor&, torch::Tensor&, const torch::Tensor&,
                                       const torch::Tensor&, float, cudaStream_t);
template void invokeRMSNormTyped<__nv_bfloat16>(const torch::Tensor&, torch::Tensor&,
                                                const torch::Tensor&, const torch::Tensor&, float,
                                                cudaStream_t);

}  // namespace kernels
}  // namespace oasr
