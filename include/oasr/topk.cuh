// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Pure CUDA top-k kernel — no framework dependencies.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cuda/std/limits>
#include <oasr/common/utils.h>

namespace oasr {
namespace topk {

/*!
 * \brief One-block-per-row top-k extraction kernel.
 *
 * Each thread loads a strided slice of the row into PER_THREAD float registers.
 * For each of the k iterations, we:
 *   1. Find the local argmax over each thread's registers.
 *   2. Reduce across the warp via shuffle.
 *   3. Sync, then have the first warp reduce the per-warp results in parallel
 *      (no thread-0 serial loop).
 *   4. Sync, then every thread masks its own register if it owns the global max.
 *
 * No shared memory is used for the row data — it lives entirely in registers,
 * so the inner loop has no shared-memory reads.
 *
 * \tparam T          Input/output scalar type (half, __nv_bfloat16, float).
 * \tparam BLOCK_SIZE Threads per block; must be a multiple of 32.
 * \tparam PER_THREAD Number of row elements held per thread (BLOCK_SIZE * PER_THREAD ≥ num_cols).
 */
template <typename T, int BLOCK_SIZE, int PER_THREAD>
__global__ void topkKernel(const T* __restrict__ input, T* __restrict__ values,
                           int32_t* __restrict__ indices, int num_cols, int k) {
    static constexpr int NWARP = BLOCK_SIZE / WARP_SIZE;

    __shared__ float warp_vals[NWARP];
    __shared__ int32_t warp_idxs[NWARP];
    __shared__ int32_t global_idx_smem;

    const int row = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;

    const T* row_in = input + (int64_t)row * num_cols;
    T* row_out = values + (int64_t)row * k;
    int32_t* idx_out = indices + (int64_t)row * k;

    const float neg_inf = cuda::std::numeric_limits<float>::lowest();

    // Load row into registers (cast to float for comparison). Out-of-range
    // slots are set to -inf so they never win the argmax.
    float regs[PER_THREAD];
#pragma unroll
    for (int j = 0; j < PER_THREAD; j++) {
        int i = threadIdx.x + j * BLOCK_SIZE;
        regs[j] = (i < num_cols) ? static_cast<float>(row_in[i]) : neg_inf;
    }

    for (int iter = 0; iter < k; iter++) {
        // Each thread finds its argmax over its register strip.
        float local_val = neg_inf;
        int local_idx = 0;
#pragma unroll
        for (int j = 0; j < PER_THREAD; j++) {
            int i = threadIdx.x + j * BLOCK_SIZE;
            if (regs[j] > local_val) {
                local_val = regs[j];
                local_idx = i;
            }
        }

// Warp-level argmax reduce via shuffle.
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_val = __shfl_xor_sync(0xffffffff, local_val, offset);
            int other_idx = __shfl_xor_sync(0xffffffff, local_idx, offset);
            if (other_val > local_val) {
                local_val = other_val;
                local_idx = other_idx;
            }
        }

        // Lane 0 of each warp publishes its warp result.
        if (lane == 0) {
            warp_vals[wid] = local_val;
            warp_idxs[wid] = local_idx;
        }
        __syncthreads();

        // First warp does the cross-warp reduction in parallel (replaces the
        // thread-0 serial sweep — big win at NWARP=32).
        if (wid == 0) {
            float v = (lane < NWARP) ? warp_vals[lane] : neg_inf;
            int32_t i = (lane < NWARP) ? warp_idxs[lane] : 0;

#pragma unroll
            for (int offset = NWARP / 2; offset > 0; offset >>= 1) {
                float ov = __shfl_xor_sync(0xffffffff, v, offset);
                int32_t oi = __shfl_xor_sync(0xffffffff, i, offset);
                if (ov > v) {
                    v = ov;
                    i = oi;
                }
            }

            if (lane == 0) {
                row_out[iter] = row_in[i];  // exact original value (avoids float roundtrip)
                idx_out[iter] = i;
                global_idx_smem = i;
            }
        }
        __syncthreads();

        // Every thread masks the register that owned the global max so it
        // won't be selected again. Cheap because it touches registers only.
        int gidx = global_idx_smem;
#pragma unroll
        for (int j = 0; j < PER_THREAD; j++) {
            int i = threadIdx.x + j * BLOCK_SIZE;
            if (i == gidx) {
                regs[j] = neg_inf;
            }
        }
    }
}

/*!
 * \brief Launch top-k kernel for a batch of rows.
 *
 * \param input    Input pointer [num_rows × num_cols] in row-major order.
 * \param values   Output values pointer [num_rows × k].
 * \param indices  Output indices pointer [num_rows × k] (int32).
 * \param num_rows Number of independent rows.
 * \param num_cols Length of each row (≤ 16384).
 * \param k        Number of top elements per row (must satisfy 1 ≤ k ≤ num_cols).
 * \param stream   CUDA stream.
 * \return cudaError_t
 */
template <typename T>
cudaError_t TopK(const T* input, T* values, int32_t* indices, unsigned int num_rows,
                 unsigned int num_cols, int k, cudaStream_t stream = nullptr) {
    // Pick (BLOCK_SIZE, PER_THREAD) such that BLOCK_SIZE * PER_THREAD ≥ num_cols.
    int block_size, per_thread;
    if (num_cols <= 32) {
        block_size = 32;
        per_thread = 1;
    } else if (num_cols <= 64) {
        block_size = 64;
        per_thread = 1;
    } else if (num_cols <= 128) {
        block_size = 128;
        per_thread = 1;
    } else if (num_cols <= 256) {
        block_size = 256;
        per_thread = 1;
    } else if (num_cols <= 512) {
        block_size = 512;
        per_thread = 1;
    } else if (num_cols <= 1024) {
        block_size = 1024;
        per_thread = 1;
    } else if (num_cols <= 2048) {
        block_size = 1024;
        per_thread = 2;
    } else if (num_cols <= 4096) {
        block_size = 1024;
        per_thread = 4;
    } else if (num_cols <= 8192) {
        block_size = 1024;
        per_thread = 8;
    } else {
        block_size = 1024;
        per_thread = 16;
    }

    const int num_cols_int = static_cast<int>(num_cols);

#define LAUNCH_TOPK(BS, PT) \
    topkKernel<T, BS, PT><<<num_rows, BS, 0, stream>>>(input, values, indices, num_cols_int, k)

    if (block_size == 32 && per_thread == 1) {
        LAUNCH_TOPK(32, 1);
    } else if (block_size == 64 && per_thread == 1) {
        LAUNCH_TOPK(64, 1);
    } else if (block_size == 128 && per_thread == 1) {
        LAUNCH_TOPK(128, 1);
    } else if (block_size == 256 && per_thread == 1) {
        LAUNCH_TOPK(256, 1);
    } else if (block_size == 512 && per_thread == 1) {
        LAUNCH_TOPK(512, 1);
    } else if (block_size == 1024 && per_thread == 1) {
        LAUNCH_TOPK(1024, 1);
    } else if (block_size == 1024 && per_thread == 2) {
        LAUNCH_TOPK(1024, 2);
    } else if (block_size == 1024 && per_thread == 4) {
        LAUNCH_TOPK(1024, 4);
    } else if (block_size == 1024 && per_thread == 8) {
        LAUNCH_TOPK(1024, 8);
    } else {
        LAUNCH_TOPK(1024, 16);
    }
#undef LAUNCH_TOPK

    return cudaGetLastError();
}

}  // namespace topk
}  // namespace oasr
