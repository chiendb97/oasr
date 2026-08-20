// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime.h>

#include <cuda/std/limits>
#include <oasr/common/utils.h>

namespace oasr {
namespace reduction {

template <typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// Down-shuffle variant for reductions whose accumulation order is part of
// their numerical contract. Only lane zero is guaranteed to hold the sum.
template <typename T>
__device__ __forceinline__ T warpReduceSumDown(T val) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
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

// Reduce a small fixed-size accumulator array while sharing one caller-owned
// scratch allocation. This avoids serial blockReduceSum calls racing on its
// internal shared storage and preserves down-shuffle accumulation order.
template <typename T, int NumValues>
__device__ __forceinline__ void blockReduceSums(T (&values)[NumValues], T* scratch) {
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

#pragma unroll
    for (int i = 0; i < NumValues; ++i) {
        values[i] = warpReduceSumDown(values[i]);
    }
    if (lane == 0) {
#pragma unroll
        for (int i = 0; i < NumValues; ++i) {
            scratch[i * WARP_SIZE + warp] = values[i];
        }
    }
    __syncthreads();

    if (warp == 0) {
#pragma unroll
        for (int i = 0; i < NumValues; ++i) {
            values[i] = lane < num_warps ? scratch[i * WARP_SIZE + lane] : T(0);
            values[i] = warpReduceSumDown(values[i]);
        }
    }
}

template <typename T>
__device__ __forceinline__ T warpReduceMax(T val) {
    T max_val = val;
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        max_val = max(max_val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return max_val;
}

template <typename T>
__device__ __forceinline__ T blockReduceMax(T val) {
    __shared__ T shared[32];  // One slot per warp
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warpReduceMax(val);
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane]
                                                 : cuda::std::numeric_limits<T>::lowest();

    // Only first warp does the final reduction
    if (wid == 0) {
        val = warpReduceMax(val);
    }
    return val;
}

template <typename T>
__device__ __forceinline__ T blockBroadcast(T value, T* workspace) {
    if (threadIdx.x == 0) {
        *workspace = value;
    }
    __syncthreads();
    return *workspace;
}

}  // namespace reduction
}  // namespace oasr
