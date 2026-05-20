// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Pure CUDA softmax kernel — no framework dependencies.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cuda/std/limits>
#include <oasr/common/math.h>
#include <oasr/common/types.h>
#include <oasr/common/utils.h>
#include <oasr/common/vec_dtypes.h>
#include <oasr/reduction.cuh>

namespace oasr {
namespace softmax {
using namespace oasr::reduction;

// =============================================================================
// Softmax Kernel
// =============================================================================

// One block per row. Three passes: find max, compute exp(x-max) + sum, normalize.
template <typename T, int VecSize>
__global__ void softmaxKernel(const T* __restrict__ input, T* __restrict__ output, int num_cols) {
    using VecT = oasr::Vec<T, VecSize>;

    const int row_idx = blockIdx.x;
    const T* row_input = input + row_idx * num_cols;
    T* row_output = output + row_idx * num_cols;

    const int vec_num_cols = num_cols / VecSize;

    __shared__ float smem[2];  // workspace for blockBroadcast: [max, sum]

    // Phase 1: find row maximum for numerical stability
    float local_max = cuda::std::numeric_limits<float>::lowest();
    for (int i = threadIdx.x; i < vec_num_cols; i += blockDim.x) {
        VecT v;
        v.load(row_input + i * VecSize);
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            local_max = max(local_max, static_cast<float>(v[j]));
        }
    }
    float row_max = blockBroadcast(blockReduceMax(local_max), &smem[0]);

    // Phase 2: compute exp(x - max) for each element, store to output, accumulate sum
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < vec_num_cols; i += blockDim.x) {
        VecT v;
        v.load(row_input + i * VecSize);

        oasr::Vec<float, VecSize> vals;
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            vals[j] = expf(static_cast<float>(v[j]) - row_max);
            local_sum += vals[j];
        }
    }
    float inv_sum = 1.0f / blockBroadcast(blockReduceSum(local_sum), &smem[1]);

    // Phase 3: normalize; each thread reads back its own phase-2 output
    for (int i = threadIdx.x; i < vec_num_cols; i += blockDim.x) {
        VecT v;
        v.load(row_input + i * VecSize);

        oasr::Vec<float, VecSize> vals;
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            vals[j] = expf(static_cast<float>(v[j]) - row_max) * inv_sum;
        }
        oasr::vecCast<T>(vals).store(row_output + i * VecSize);
    }
}

// Online softmax pairwise merge over a warp.
// Treats lanes as partials (m, s) where s = sum_i exp(x_i - m); combines two
// partials (m1, s1), (m2, s2) into (max(m1,m2), s1*exp(m1-new_m) + s2*exp(m2-new_m)).
__device__ __forceinline__ float2 warpReduce(float2 val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other_max = __shfl_xor_sync(0xffffffff, val.x, offset);
        float other_sum = __shfl_xor_sync(0xffffffff, val.y, offset);
        float new_max = max(val.x, other_max);
        val.y = val.y * expf(val.x - new_max) + other_sum * expf(other_max - new_max);
        val.x = new_max;
    }
    return val;
}

__device__ __forceinline__ float2 blockReduce(float2 val) {
    __shared__ float2 shared[32];  // One slot per warp
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warpReduce(val);
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    val.x = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane].x
                                                   : cuda::std::numeric_limits<float>::lowest();

    val.y = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane].y : 0.0f;
    // Only first warp does the final reduction
    if (wid == 0) {
        val = warpReduce(val);
    }
    return val;
}


// One block per row. Two passes: online (max, sum) accumulation, then emit
// log_softmax(x) = x - row_max - log(sum_exp). The first phase mirrors
// onlineSoftmaxKernel; the second writes the log form directly.
template <typename T, int VecSize>
__global__ void onlineLogSoftmaxKernel(const T* __restrict__ input, T* __restrict__ output,
                                       int num_cols) {
    using VecT = oasr::Vec<T, VecSize>;

    const int row_idx = blockIdx.x;
    const T* row_input = input + row_idx * num_cols;
    T* row_output = output + row_idx * num_cols;

    const int vec_num_cols = num_cols / VecSize;

    __shared__ float2 smem;

    float2 local_val = make_float2(cuda::std::numeric_limits<float>::lowest(), 0.0f);
    for (int i = threadIdx.x; i < vec_num_cols; i += blockDim.x) {
        VecT v;
        v.load(row_input + i * VecSize);

        float vec_max = static_cast<float>(v[0]);
#pragma unroll
        for (int j = 1; j < VecSize; j++) {
            vec_max = max(vec_max, static_cast<float>(v[j]));
        }
        float new_max = max(local_val.x, vec_max);
        float vec_sum = 0.0f;
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            vec_sum += expf(static_cast<float>(v[j]) - new_max);
        }
        local_val.y = local_val.y * expf(local_val.x - new_max) + vec_sum;
        local_val.x = new_max;
    }
    float2 row_val = blockBroadcast(blockReduce(local_val), &smem);
    const float row_max = row_val.x;
    const float log_norm = logf(row_val.y);  // log(sum_exp(x - row_max))

    // Phase 2: emit log_softmax(x) = (x - row_max) - log_norm.
    for (int i = threadIdx.x; i < vec_num_cols; i += blockDim.x) {
        VecT v;
        v.load(row_input + i * VecSize);

        oasr::Vec<float, VecSize> vals;
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            vals[j] = static_cast<float>(v[j]) - row_max - log_norm;
        }
        oasr::vecCast<T>(vals).store(row_output + i * VecSize);
    }
}

// One block per row. Two passes: online (max, sum) accumulation, then normalize.
template <typename T, int VecSize>
__global__ void onlineSoftmaxKernel(const T* __restrict__ input, T* __restrict__ output,
                                    int num_cols) {
    using VecT = oasr::Vec<T, VecSize>;

    const int row_idx = blockIdx.x;
    const T* row_input = input + row_idx * num_cols;
    T* row_output = output + row_idx * num_cols;

    const int vec_num_cols = num_cols / VecSize;

    __shared__ float2 smem;  // workspace for blockBroadcast

    // Phase 1: single-pass online (max, sum) over the row.
    // Per loaded vector, fold elements into the per-thread partial in two steps:
    //   1) update running max,
    //   2) rescale running sum by exp(old_max - new_max) and add sum_j exp(x_j - new_max).
    // This costs one extra exp per vector instead of one extra exp per element.
    float2 local_val = make_float2(cuda::std::numeric_limits<float>::lowest(), 0.0f);
    for (int i = threadIdx.x; i < vec_num_cols; i += blockDim.x) {
        VecT v;
        v.load(row_input + i * VecSize);

        float vec_max = static_cast<float>(v[0]);
#pragma unroll
        for (int j = 1; j < VecSize; j++) {
            vec_max = max(vec_max, static_cast<float>(v[j]));
        }
        float new_max = max(local_val.x, vec_max);
        float vec_sum = 0.0f;
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            vec_sum += expf(static_cast<float>(v[j]) - new_max);
        }
        local_val.y = local_val.y * expf(local_val.x - new_max) + vec_sum;
        local_val.x = new_max;
    }
    float2 row_val = blockBroadcast(blockReduce(local_val), &smem);
    const float row_max = row_val.x;
    const float inv_sum = 1.0f / row_val.y;

    // Phase 2: normalize and write output.
    for (int i = threadIdx.x; i < vec_num_cols; i += blockDim.x) {
        VecT v;
        v.load(row_input + i * VecSize);

        oasr::Vec<float, VecSize> vals;
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            vals[j] = expf(static_cast<float>(v[j]) - row_max) * inv_sum;
        }
        oasr::vecCast<T>(vals).store(row_output + i * VecSize);
    }
}

// =============================================================================
// Typed Launcher (raw pointer interface, returns cudaError_t)
// =============================================================================

template <typename T>
cudaError_t Softmax(const T* input, T* output, unsigned int num_rows, unsigned int num_cols,
                    cudaStream_t stream) {
    constexpr int VecSize = oasr::VecTypeTrait<T>::VecSize;

    bool use_vec = (num_cols >= static_cast<unsigned int>(VecSize)) && (num_cols % VecSize == 0) &&
                   isAligned<T, VecSize>(input) && isAligned<T, VecSize>(output);

    if (use_vec) {
        int block_size = alignedBlockSize(static_cast<int>(num_cols) / VecSize);
        onlineSoftmaxKernel<T, VecSize>
            <<<num_rows, block_size, 0, stream>>>(input, output, static_cast<int>(num_cols));
    } else {
        int block_size = alignedBlockSize(static_cast<int>(num_cols));
        onlineSoftmaxKernel<T, 1>
            <<<num_rows, block_size, 0, stream>>>(input, output, static_cast<int>(num_cols));
    }
    return cudaGetLastError();
}

template <typename T>
cudaError_t LogSoftmax(const T* input, T* output, unsigned int num_rows, unsigned int num_cols,
                       cudaStream_t stream) {
    constexpr int VecSize = oasr::VecTypeTrait<T>::VecSize;

    bool use_vec = (num_cols >= static_cast<unsigned int>(VecSize)) && (num_cols % VecSize == 0) &&
                   isAligned<T, VecSize>(input) && isAligned<T, VecSize>(output);

    if (use_vec) {
        int block_size = alignedBlockSize(static_cast<int>(num_cols) / VecSize);
        onlineLogSoftmaxKernel<T, VecSize>
            <<<num_rows, block_size, 0, stream>>>(input, output, static_cast<int>(num_cols));
    } else {
        int block_size = alignedBlockSize(static_cast<int>(num_cols));
        onlineLogSoftmaxKernel<T, 1>
            <<<num_rows, block_size, 0, stream>>>(input, output, static_cast<int>(num_cols));
    }
    return cudaGetLastError();
}

}  // namespace softmax
}  // namespace oasr
