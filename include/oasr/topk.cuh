// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Pure CUDA top-k kernel — no framework dependencies.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cub/block/block_radix_sort.cuh>
#include <cuda/std/limits>
#include <oasr/common/utils.h>
#include <oasr/sort/bitonic.cuh>

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
 * \brief One-block-per-row top-k via block-wide radix sort.
 *
 * The iterative kernel above is O(k) in syncs; for large k the sync cost
 * dominates. This variant pays a single sort (O(N · radix_passes)) and
 * extracts the leading k elements, beating the iterative path once k is
 * larger than ~8–16.
 *
 * Layout: load striped (coalesced), sort blocked→striped so the top-k land
 * in register 0 of threads 0..min(k,BS)-1, giving coalesced output writes.
 */
template <typename T, int BLOCK_SIZE, int PER_THREAD>
__global__ void topkSortKernel(const T* __restrict__ input, T* __restrict__ values,
                               int32_t* __restrict__ indices, int num_cols, int k) {
    using BlockRadixSort = cub::BlockRadixSort<float, BLOCK_SIZE, PER_THREAD, int32_t>;
    __shared__ typename BlockRadixSort::TempStorage temp_storage;

    const int row = blockIdx.x;

    const T* row_in = input + (int64_t)row * num_cols;
    T* row_out = values + (int64_t)row * k;
    int32_t* idx_out = indices + (int64_t)row * k;

    const float neg_inf = cuda::std::numeric_limits<float>::lowest();

    float keys[PER_THREAD];
    int32_t idx[PER_THREAD];

#pragma unroll
    for (int j = 0; j < PER_THREAD; j++) {
        int i = threadIdx.x + j * BLOCK_SIZE;
        keys[j] = (i < num_cols) ? static_cast<float>(row_in[i]) : neg_inf;
        idx[j] = i;
    }

    BlockRadixSort(temp_storage).SortDescendingBlockedToStriped(keys, idx);

    // Striped output: rank (j*BS + tx) lives in register j of thread tx.
    // half/bf16/float survive a float roundtrip exactly when the source was
    // already that type, so we can write keys[j] directly without a gather.
#pragma unroll
    for (int j = 0; j < PER_THREAD; j++) {
        int rank = j * BLOCK_SIZE + threadIdx.x;
        if (rank < k) {
            row_out[rank] = static_cast<T>(keys[j]);
            idx_out[rank] = idx[j];
        }
    }
}

/*!
 * \brief One-block-per-row top-k via bitonic top-k merge.
 *
 * Algorithm:
 *   1. Each thread loads PER_THREAD elements (PT must be a multiple of K).
 *   2. Per-thread top-k: bitonic_sort the first K, then for each subsequent
 *      K-chunk bitonic_sort + bitonic_topk_merge into the running top-K.
 *   3. Warp-level top-k: log2(WARP_SIZE) shuffle stages of bitonic_topk_merge.
 *   4. Cross-warp top-k (if NWARP > 1): each warp's lane 0 publishes its
 *      top-K to shared memory; warp 0 reads back NWARP per-warp top-Ks (one
 *      per lane, padding with -inf) and runs log2(NWARP) shuffle merges.
 *   5. Lane 0 of warp 0 writes the leading k_actual values to global memory.
 *
 * Compared to topkKernel (iter), the merge cost per stage is O(K log K)
 * rather than O(k) full reductions, so total work scales with K log² K once
 * the per-thread sort dominates — much better than O(k) for moderate k.
 *
 * \tparam T          Input/output scalar type.
 * \tparam BLOCK_SIZE Threads per block; must be a power of two ≥ WARP_SIZE.
 * \tparam PER_THREAD Elements held per thread; must be a multiple of K.
 * \tparam K          Top-k width (power of two). k_actual ≤ K is the runtime k.
 */
template <typename T, int BLOCK_SIZE, int PER_THREAD, int K>
__global__ void topkBitonicKernel(const T* __restrict__ input, T* __restrict__ values,
                                  int32_t* __restrict__ indices, int num_cols, int k_actual) {
    static_assert((K & (K - 1)) == 0, "K must be a power of two");
    static_assert(PER_THREAD >= K && (PER_THREAD % K) == 0,
                  "PER_THREAD must be a multiple of K and ≥ K");
    static_assert(BLOCK_SIZE >= WARP_SIZE && (BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0,
                  "BLOCK_SIZE must be a power of two ≥ WARP_SIZE");

    constexpr int kNumWarps = BLOCK_SIZE / WARP_SIZE;
    constexpr int kNumChunks = PER_THREAD / K;

    __shared__ float warp_topk_smem[kNumWarps * K];
    __shared__ int32_t warp_idx_smem[kNumWarps * K];

    const int row = blockIdx.x;
    const int tx = threadIdx.x;
    const int lane = tx & (WARP_SIZE - 1);
    const int wid = tx >> 5;

    const T* row_in = input + (int64_t)row * num_cols;
    T* row_out = values + (int64_t)row * k_actual;
    int32_t* idx_out = indices + (int64_t)row * k_actual;

    const float neg_inf = cuda::std::numeric_limits<float>::lowest();

    // Per-thread register strip: striped layout for coalesced loads.
    float regs[PER_THREAD];
    int32_t regs_idx[PER_THREAD];
#pragma unroll
    for (int j = 0; j < PER_THREAD; ++j) {
        int i = tx + j * BLOCK_SIZE;
        if (i < num_cols) {
            regs[j] = static_cast<float>(row_in[i]);
            regs_idx[j] = i;
        } else {
            regs[j] = neg_inf;
            regs_idx[j] = -1;
        }
    }

    // Per-thread top-K: sort the first chunk, then merge subsequent chunks.
    float topk_vals[K];
    int32_t topk_idx[K];
#pragma unroll
    for (int j = 0; j < K; ++j) {
        topk_vals[j] = regs[j];
        topk_idx[j] = regs_idx[j];
    }
    sort::bitonic_sort<K, /*Ascending=*/false>(topk_vals, topk_idx);

#pragma unroll
    for (int chunk = 1; chunk < kNumChunks; ++chunk) {
        float other_vals[K];
        int32_t other_idx[K];
#pragma unroll
        for (int j = 0; j < K; ++j) {
            other_vals[j] = regs[chunk * K + j];
            other_idx[j] = regs_idx[chunk * K + j];
        }
        sort::bitonic_sort<K, /*Ascending=*/false>(other_vals, other_idx);
        sort::bitonic_topk_merge<K, /*Ascending=*/false>(topk_vals, topk_idx, other_vals,
                                                        other_idx);
    }

    // Warp-level top-k merge: log2(WARP_SIZE) butterfly shuffles.
#pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        float other_vals[K];
        int32_t other_idx[K];
#pragma unroll
        for (int j = 0; j < K; ++j) {
            other_vals[j] = __shfl_xor_sync(0xffffffff, topk_vals[j], offset);
            other_idx[j] = __shfl_xor_sync(0xffffffff, topk_idx[j], offset);
        }
        sort::bitonic_topk_merge<K, /*Ascending=*/false>(topk_vals, topk_idx, other_vals,
                                                        other_idx);
    }

    // Cross-warp top-k merge via shared memory.
    if constexpr (kNumWarps > 1) {
        if (lane == 0) {
#pragma unroll
            for (int j = 0; j < K; ++j) {
                warp_topk_smem[wid * K + j] = topk_vals[j];
                warp_idx_smem[wid * K + j] = topk_idx[j];
            }
        }
        __syncthreads();

        if (wid == 0) {
            // Lane l reads warp l's top-K. Lanes ≥ kNumWarps load -inf so
            // the butterfly merges below treat them as no-ops.
#pragma unroll
            for (int j = 0; j < K; ++j) {
                if (lane < kNumWarps) {
                    topk_vals[j] = warp_topk_smem[lane * K + j];
                    topk_idx[j] = warp_idx_smem[lane * K + j];
                } else {
                    topk_vals[j] = neg_inf;
                    topk_idx[j] = -1;
                }
            }

#pragma unroll
            for (int offset = 1; offset < kNumWarps; offset <<= 1) {
                float other_vals[K];
                int32_t other_idx[K];
#pragma unroll
                for (int j = 0; j < K; ++j) {
                    other_vals[j] = __shfl_xor_sync(0xffffffff, topk_vals[j], offset);
                    other_idx[j] = __shfl_xor_sync(0xffffffff, topk_idx[j], offset);
                }
                sort::bitonic_topk_merge<K, /*Ascending=*/false>(topk_vals, topk_idx, other_vals,
                                                                other_idx);
            }
        }
    }

    // Output: lane 0 of warp 0 holds the global top-K (sorted descending).
    // K is small (≤ 128) and writes are uncoalesced from one thread, but the
    // total bytes are tiny compared to the sort work.
    if (wid == 0 && lane == 0) {
#pragma unroll
        for (int j = 0; j < K; ++j) {
            if (j < k_actual) {
                row_out[j] = static_cast<T>(topk_vals[j]);
                idx_out[j] = topk_idx[j];
            }
        }
    }
}

// Bitonic launcher for a fixed K_PADDED: picks the smallest BLOCK_SIZE such
// that BLOCK_SIZE * K_PADDED ≥ num_cols. PER_THREAD = K_PADDED (one chunk per
// thread), which minimizes per-thread sort cost. Returns false (and does not
// launch) when num_cols would require BLOCK_SIZE > 1024.
template <typename T, int K_PADDED>
inline bool LaunchTopKBitonicForK(const T* input, T* values, int32_t* indices,
                                  unsigned int num_rows, int num_cols, int k,
                                  cudaStream_t stream) {
    const int needed_bs = (num_cols + K_PADDED - 1) / K_PADDED;
#define LAUNCH_BITONIC(BS)                                                           \
    topkBitonicKernel<T, BS, K_PADDED, K_PADDED>                                     \
        <<<num_rows, BS, 0, stream>>>(input, values, indices, num_cols, k)

    if (needed_bs <= 32) {
        LAUNCH_BITONIC(32);
    } else if (needed_bs <= 64) {
        LAUNCH_BITONIC(64);
    } else if (needed_bs <= 128) {
        LAUNCH_BITONIC(128);
    } else if (needed_bs <= 256) {
        LAUNCH_BITONIC(256);
    } else if (needed_bs <= 512) {
        LAUNCH_BITONIC(512);
    } else if (needed_bs <= 1024) {
        LAUNCH_BITONIC(1024);
    } else {
        return false;  // Row too wide for this K_PADDED — caller falls back.
    }
#undef LAUNCH_BITONIC
    return true;
}

// Top-level bitonic dispatch: rounds k up to the next power of two ≥ 8, picks
// the per-K launcher. Returns false if the request can't be served by bitonic
// (k > 128 or row width exceeds what the chosen K can cover with BS ≤ 1024).
template <typename T>
inline bool LaunchTopKBitonic(const T* input, T* values, int32_t* indices,
                              unsigned int num_rows, int num_cols, int k,
                              cudaStream_t stream) {
    if (k > 128 || num_cols < 32) return false;
    if (k <= 8) {
        return LaunchTopKBitonicForK<T, 8>(input, values, indices, num_rows, num_cols, k, stream);
    } else if (k <= 16) {
        return LaunchTopKBitonicForK<T, 16>(input, values, indices, num_rows, num_cols, k, stream);
    } else if (k <= 32) {
        return LaunchTopKBitonicForK<T, 32>(input, values, indices, num_rows, num_cols, k, stream);
    } else if (k <= 64) {
        return LaunchTopKBitonicForK<T, 64>(input, values, indices, num_rows, num_cols, k, stream);
    } else {
        return LaunchTopKBitonicForK<T, 128>(input, values, indices, num_rows, num_cols, k, stream);
    }
}

// Per-(BS, PT) launcher: picks the sort kernel when k is large enough AND
// the config benefits from it.
//
// Two constexpr gates:
//   - shmem fits: CUB exchange storage is ~4 bytes/element; (BS=1024, PT=16)
//     would blow the 48 KB static-shmem limit, so sort is never instantiated
//     for it.
//   - PT >= 4: CUB BlockRadixSort has high fixed overhead per radix pass.
//     Empirically, only PT >= 4 (rows ≥ 4096) sees sort beat the iterative
//     kernel in practice — at smaller PT the sort costs more than the
//     equivalent iterative reduction even at k = 50.
template <typename T, int BLOCK_SIZE, int PER_THREAD>
inline void LaunchTopK(const T* input, T* values, int32_t* indices, unsigned int num_rows,
                       int num_cols, int k, bool use_sort, cudaStream_t stream) {
    constexpr bool kSortFitsShmem =
        BLOCK_SIZE * PER_THREAD * static_cast<int>(sizeof(float)) <= 48 * 1024;
    constexpr bool kSortAmortizes = PER_THREAD >= 4;
    if constexpr (kSortFitsShmem && kSortAmortizes) {
        if (use_sort) {
            topkSortKernel<T, BLOCK_SIZE, PER_THREAD>
                <<<num_rows, BLOCK_SIZE, 0, stream>>>(input, values, indices, num_cols, k);
            return;
        }
    }
    topkKernel<T, BLOCK_SIZE, PER_THREAD>
        <<<num_rows, BLOCK_SIZE, 0, stream>>>(input, values, indices, num_cols, k);
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

    // Bitonic top-k handles k ≤ 128 with per-stage cost O(K log K) — much
    // better than the iterative kernel's O(k) full reductions, AND it lets us
    // use small blocks (BS = num_cols / K_padded) so many rows pack onto one
    // SM at once. It wins decisively at BS ≤ 128 (lots of blocks per SM); we
    // only stay on the iter kernel for k=1 (degenerate sort) and very small
    // num_cols where the iter overhead is already negligible.
    constexpr int K_BITONIC_MIN = 1;
    constexpr int K_BITONIC_MAX = 128;
    if (k > K_BITONIC_MIN && k <= K_BITONIC_MAX && num_cols_int >= 32) {
        if (LaunchTopKBitonic<T>(input, values, indices, num_rows, num_cols_int, k, stream)) {
            return cudaGetLastError();
        }
        // else: row too wide for the chosen K — fall through to iter/sort.
    }

    // For small k, the iterative kernel wins (less constant overhead).
    // For larger k (k > 128, beyond bitonic's range), CUB's block-wide sort
    // beats the iterative path's per-iteration sync cost.
    constexpr int K_SORT_THRESHOLD = 32;
    const bool use_sort = (k > K_SORT_THRESHOLD);

#define LAUNCH_TOPK(BS, PT)                                                                  \
    LaunchTopK<T, BS, PT>(input, values, indices, num_rows, num_cols_int, k, use_sort, stream)

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
