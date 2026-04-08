// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// GPU-accelerated CTC prefix beam search decoder.
//
// Algorithm based on torchaudio's ctc_prefix_decoder_kernel_v2.cu
// (BSD 3-Clause License, NVIDIA CORPORATION & AFFILIATES).
//
// Supports batched offline decoding and streaming (chunk-by-chunk) decoding.

#pragma once

#include <cuda_runtime.h>

#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <cub/cub.cuh>

// ---------------------------------------------------------------------------
// Paged memory primitives (folded in from paged_memory.cuh)
//
// Implements a paged-attention style memory system for storing variable-length
// decoded token sequences. Instead of pre-allocating batch * beam * max_seq_len
// tokens, sequences are stored in fixed-size pages with a block table for
// indirection, enabling prefix sharing across beams via reference counting.
// ---------------------------------------------------------------------------

namespace oasr {
namespace paged_memory {

static constexpr int DEFAULT_PAGE_SIZE = 16;  // tokens per page (64 bytes, 1 cache line)
static constexpr int INVALID_PAGE = -1;
static constexpr int PAGED_ALIGN_BYTES = 128;

struct PagedSequenceState {
    // GPU: [num_pages * page_size] — contiguous token storage pool
    int* page_storage;

    // GPU: [batch * beam * max_logical_pages] * 2 — double-buffered block tables.
    // block_table[p][(bid * beam + k) * max_logical_pages + lp] = physical page index
    int* block_table[2];

    // GPU: [num_pages] — per-page reference count (CPU-driven during init/step)
    int* ref_counts;

    // GPU: [1] — atomic counter for page allocation (next available physical page)
    int* next_free_page;

    // GPU: [num_pages] — stack of recycled physical page indices (ref_count dropped to 0)
    int* free_pool;

    // GPU: [1] — number of valid entries in free_pool (top of stack)
    int* free_pool_size;

    // Scalars (host-accessible after copying StreamingState from GPU)
    int page_size;
    int max_logical_pages;  // = ceil(max_seq_len / page_size)
    int num_pages;
    int batch;
    int beam;

    // Zero-initialize: indicates flat mode when page_storage == nullptr
    PagedSequenceState()
        : page_storage(nullptr),
          ref_counts(nullptr),
          next_free_page(nullptr),
          free_pool(nullptr),
          free_pool_size(nullptr),
          page_size(0),
          max_logical_pages(0),
          num_pages(0),
          batch(0),
          beam(0) {
        block_table[0] = nullptr;
        block_table[1] = nullptr;
    }

    __host__ __device__ bool is_enabled() const { return page_storage != nullptr; }
};

inline constexpr size_t paged_align_size(size_t n) {
    return (n + PAGED_ALIGN_BYTES - 1) / PAGED_ALIGN_BYTES * PAGED_ALIGN_BYTES;
}

inline int default_num_pages(int batch, int beam, int max_seq_len, int page_size) {
    // Allocate (max_lp + 1) pages per beam: max_lp for the steady-state double-buffered
    // block tables (completed pages are shared across both slots) plus 1 extra batch*beam
    // slab to cover the step-1 CoW burst before any recycling has happened.
    int max_lp = (max_seq_len + page_size - 1) / page_size;
    return batch * beam * (max_lp + 1);
}

inline size_t calculate_paged_region_size(int batch, int beam, int max_seq_len,
                                          int page_size = DEFAULT_PAGE_SIZE,
                                          int num_pages = 0) {
    if (num_pages <= 0)
        num_pages = default_num_pages(batch, beam, max_seq_len, page_size);
    int max_lp = (max_seq_len + page_size - 1) / page_size;

    size_t total = 0;
    total += paged_align_size(sizeof(int) * num_pages * page_size);
    total += paged_align_size(sizeof(int) * batch * beam * max_lp) * 2;
    total += paged_align_size(sizeof(int) * num_pages);  // ref_counts
    total += paged_align_size(sizeof(int));              // next_free_page
    total += paged_align_size(sizeof(int) * num_pages);  // free_pool
    total += paged_align_size(sizeof(int));              // free_pool_size
    return total;
}

inline void init_paged_state(PagedSequenceState* ps, void* workspace,
                             int batch, int beam, int max_seq_len,
                             int page_size, int num_pages,
                             cudaStream_t stream) {
    if (num_pages <= 0)
        num_pages = default_num_pages(batch, beam, max_seq_len, page_size);

    ps->page_size = page_size;
    ps->max_logical_pages = (max_seq_len + page_size - 1) / page_size;
    ps->num_pages = num_pages;
    ps->batch = batch;
    ps->beam = beam;

    int max_lp = ps->max_logical_pages;

    char* ptr = reinterpret_cast<char*>(workspace);

#define PAGED_ALLOC(field, type, count)               \
    ps->field = reinterpret_cast<type*>(ptr);         \
    ptr += paged_align_size(sizeof(type) * (count));

    PAGED_ALLOC(page_storage, int, num_pages * page_size)
    PAGED_ALLOC(block_table[0], int, batch * beam * max_lp)
    PAGED_ALLOC(block_table[1], int, batch * beam * max_lp)
    PAGED_ALLOC(ref_counts, int, num_pages)
    PAGED_ALLOC(next_free_page, int, 1)
    PAGED_ALLOC(free_pool, int, num_pages)
    PAGED_ALLOC(free_pool_size, int, 1)

#undef PAGED_ALLOC

    cudaMemsetAsync(ps->page_storage, 0, sizeof(int) * num_pages * page_size, stream);

    {
        int total_bt = batch * beam * max_lp;
        int* h_bt = new int[total_bt];
        for (int b = 0; b < batch; ++b) {
            for (int k = 0; k < beam; ++k) {
                int bk = b * beam + k;
                for (int lp = 0; lp < max_lp; ++lp) {
                    h_bt[bk * max_lp + lp] = (lp == 0) ? bk : INVALID_PAGE;
                }
            }
        }
        cudaMemcpyAsync(ps->block_table[0], h_bt, sizeof(int) * total_bt,
                        cudaMemcpyHostToDevice, stream);
        delete[] h_bt;
    }

    {
        int total_bt = batch * beam * max_lp;
        int* h_bt = new int[total_bt];
        for (int i = 0; i < total_bt; ++i)
            h_bt[i] = INVALID_PAGE;
        cudaMemcpyAsync(ps->block_table[1], h_bt, sizeof(int) * total_bt,
                        cudaMemcpyHostToDevice, stream);
        delete[] h_bt;
    }

    {
        int* h_refs = new int[num_pages];
        for (int i = 0; i < batch * beam; ++i)
            h_refs[i] = 1;
        for (int i = batch * beam; i < num_pages; ++i)
            h_refs[i] = 0;
        cudaMemcpyAsync(ps->ref_counts, h_refs, sizeof(int) * num_pages,
                        cudaMemcpyHostToDevice, stream);
        delete[] h_refs;
    }

    {
        int nfp = batch * beam;
        cudaMemcpyAsync(ps->next_free_page, &nfp, sizeof(int),
                        cudaMemcpyHostToDevice, stream);
    }

    {
        int zero = 0;
        cudaMemcpyAsync(ps->free_pool_size, &zero, sizeof(int),
                        cudaMemcpyHostToDevice, stream);
    }
    // free_pool contents are uninitialised (only valid indices are used via free_pool_size)
}

__device__ __forceinline__ int paged_read(const int* __restrict__ page_storage,
                                          const int* __restrict__ bt,
                                          int bk_idx, int pos,
                                          int page_size, int max_lp) {
    int lp = pos / page_size;
    int off = pos - lp * page_size;
    int phys = bt[bk_idx * max_lp + lp];
    return page_storage[phys * page_size + off];
}

__device__ __forceinline__ bool paged_seq_compare(const int* __restrict__ page_storage,
                                                   const int* __restrict__ bt,
                                                   int bid, int beam_a, int beam_b,
                                                   int len, int beam,
                                                   int page_size, int max_lp) {
    int full_pages = len / page_size;
    int bk_a = bid * beam + beam_a;
    int bk_b = bid * beam + beam_b;

    for (int p = 0; p < full_pages; ++p) {
        int phys_a = bt[bk_a * max_lp + p];
        int phys_b = bt[bk_b * max_lp + p];
        if (phys_a == phys_b)
            continue;
        for (int i = 0; i < page_size; ++i) {
            if (page_storage[phys_a * page_size + i] != page_storage[phys_b * page_size + i])
                return false;
        }
    }

    int rem = len - full_pages * page_size;
    if (rem > 0) {
        int phys_a = bt[bk_a * max_lp + full_pages];
        int phys_b = bt[bk_b * max_lp + full_pages];
        if (phys_a != phys_b) {
            for (int i = 0; i < rem; ++i) {
                if (page_storage[phys_a * page_size + i] != page_storage[phys_b * page_size + i])
                    return false;
            }
        }
    }
    return true;
}

}  // namespace paged_memory
}  // namespace oasr

namespace oasr {
namespace ctc_decoder {

// =============================================================================
// Constants
// =============================================================================

static constexpr int ALIGN_BYTES = 128;
static constexpr int MAX_BLOCKS = 800;
static constexpr int MAX_BLOCKS_PER_BATCH = 16;
// Use -FLT_MAX to match the reference and for correct CUB sort sentinel behavior.
static constexpr float NEG_INF = -FLT_MAX;

// =============================================================================
// FastDivmod — precomputed fast integer division
// =============================================================================

struct FastDivmod {
    int divisor;
    unsigned int multiplier;
    unsigned int shift_right;

    __host__ __device__ FastDivmod() : divisor(0), multiplier(0), shift_right(0) {}

    __host__ FastDivmod(int d) : divisor(d) {
        if (d == 0) {
            multiplier = 0;
            shift_right = 0;
            return;
        }
        unsigned int p = 31;
        while ((1u << p) < (unsigned int)d)
            ++p;
        uint64_t m = ((1ULL << (32 + p)) + d - 1) / d;
        multiplier = (unsigned int)(m - (1ULL << 32));
        shift_right = p;
    }

    __host__ __device__ void operator()(int& quotient, int& remainder, int dividend) const {
        if (divisor == 0) {
            quotient = 0;
            remainder = 0;
            return;
        }
#ifdef __CUDA_ARCH__
        quotient = dividend / divisor;
#else
        unsigned int t = (unsigned int)((uint64_t)multiplier * (unsigned int)dividend >> 32);
        t = ((unsigned int)dividend - t) / 2 + t;
        quotient = (int)(t >> shift_right);
#endif
        remainder = dividend - quotient * divisor;
    }

    __host__ __device__ int div(int dividend) const {
        int q, r;
        (*this)(q, r, dividend);
        return q;
    }

    __host__ __device__ int mod(int dividend) const {
        int q, r;
        (*this)(q, r, dividend);
        return r;
    }
};

// =============================================================================
// InternalData — all GPU state carved from a single workspace buffer
// =============================================================================

struct InternalData {
    int batch;
    int beam;
    int vocab_size;
    int ldc;        // = vocab_size
    int ldbeam;     // 16-aligned beam
    int ldseq_len;  // 16-aligned max_seq_len
    int max_seq_len;

    // Double-buffered decoded sequences
    int* clen[2];   // [batch * ldbeam] decoded sequence lengths
    int* clist[2];  // [batch * beam * ldseq_len] decoded token sequences

    // Beam state
    float2* pprev;   // [batch * ldbeam] (blank_score, nonblank_score) per beam
    float* ptable;   // [batch * beam * ldc] blank-path probability table
    float* ptablen;  // [batch * beam * ldc] nonblank-path probability table
    int* clast;      // [batch * ldbeam] last character in each beam
    int* ptid;       // [batch * ldbeam] (unused, kept for ABI compatibility)
    float* score;    // [batch * ldbeam] beam scores

    // Top-K buffers (Phase 1 output: [batch * MAX_BLOCKS_PER_BATCH * beam])
    float* topk_key_buffer;
    int* topk_value_buffer;

    // Sequence selection (blank threshold filtering)
    int* select_seqs;      // [batch * max_seq_len]
    int* select_seq_lens;  // [batch]

    FastDivmod ldc_divmod;
    int max_select_seq_len;

    // Paged memory state (all null pointers = flat mode; page_storage != nullptr = paged mode)
    paged_memory::PagedSequenceState paged;
};

// =============================================================================
// Device helper functions (matching reference semantics exactly)
// =============================================================================

__inline__ __device__ float logprob_add(float a, float b) {
    return a + b;
}

// =============================================================================
// Paged memory helpers — allocate/free a physical page atomically.
//
// alloc_page: prefer recycled pages from free_pool over fresh allocation.
//   Call only AFTER __syncthreads() that follows the fork phase so that all
//   free_page pushes are visible before any pop.
//
// free_page: decrement ref_count; if it reaches zero, push the page index onto
//   the free_pool stack.  The push is safe to call from multiple concurrent
//   threads as long as each physical page is freed by exactly one logical
//   "owner" (ensured by the fork loop which handles each bk_dst slot once).
// =============================================================================

__device__ __forceinline__ int alloc_page(int* __restrict__ free_pool,
                                          int* __restrict__ free_pool_size,
                                          int* __restrict__ next_free_page) {
    // Atomically pop one index from the free_pool stack.
    int idx = atomicAdd(free_pool_size, -1) - 1;
    if (idx >= 0) {
        return free_pool[idx];
    }
    // Pool was empty (or raced to empty); undo the decrement and allocate fresh.
    atomicAdd(free_pool_size, 1);
    return atomicAdd(next_free_page, 1);
}

__device__ __forceinline__ void free_page(int phys,
                                          int* __restrict__ free_pool,
                                          int* __restrict__ free_pool_size,
                                          int* __restrict__ ref_counts) {
    // Decrement ref_count; when it hits 0 the page becomes available.
    if (atomicSub(&ref_counts[phys], 1) == 1) {
        int slot = atomicAdd(free_pool_size, 1);
        free_pool[slot] = phys;
    }
}

// logsumexp matching torchaudio reference:
//   _logsumexp(a, b) = max(a,b) + log(1 + exp(-|a-b|))
__inline__ __device__ float logsumexp(float a, float b) {
    float max_ab = (a > b) ? a : b;
    float neg_abs = (a - b) > 0.0f ? (b - a) : (a - b);
    return max_ab + __logf(1.0f + __expf(neg_abs));
}

// Compare two integer sequences. Returns true if equal.
__inline__ __device__ bool seq_compare(int len, const int* a, const int* b) {
    for (int i = 0; i < len; ++i)
        if (a[i] != b[i])
            return false;
    return true;
}

// =============================================================================
// Kernel: Initialize sequence selection via BlockScan (matching reference)
// =============================================================================
// Selects frames where blank prob (in log space) < threshold.
// Uses CUB BlockScan to produce ordered output without atomics.

template <int BLOCK_SIZE, int ITEMS_PER_THREAD>
__global__ void init_select_kernel(const float* __restrict__ log_prob, int batch_stride,
                                   int seq_stride, int vocab_stride,
                                   const int* __restrict__ seq_lengths, int batch, int max_seq_len,
                                   int blank_id, float log_threshold, int* __restrict__ select_seqs,
                                   int* __restrict__ select_seq_lens) {
    int bid = blockIdx.x;
    if (bid >= batch)
        return;

    using BlockScanT = cub::BlockScan<int, BLOCK_SIZE>;
    __shared__ typename BlockScanT::TempStorage temp_storage;

    int selected[ITEMS_PER_THREAD];
    int selected_scan[ITEMS_PER_THREAD];
    const int tx = threadIdx.x;
    int actual_len = seq_lengths[bid];
    int block_agg = 0;

    for (int t_offset = 0; t_offset < actual_len; t_offset += BLOCK_SIZE * ITEMS_PER_THREAD) {
#pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
            int t = t_offset + ITEMS_PER_THREAD * tx + ITEM;
            if (t < actual_len) {
                float lp = log_prob[bid * batch_stride + t * seq_stride + blank_id * vocab_stride];
                selected[ITEM] = (lp < log_threshold) ? 1 : 0;
            } else {
                selected[ITEM] = 0;
            }
        }
        __syncthreads();

        int block_agg_this_iter = 0;
        BlockScanT{temp_storage}.ExclusiveSum(selected, selected_scan, block_agg_this_iter);
        __syncthreads();

#pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
            if (selected[ITEM]) {
                int t = t_offset + ITEMS_PER_THREAD * tx + ITEM;
                select_seqs[bid * max_seq_len + selected_scan[ITEM] + block_agg] = t;
            }
        }
        block_agg += block_agg_this_iter;
    }

    if (tx == 0)
        select_seq_lens[bid] = block_agg;
}

// =============================================================================
// Kernel: First step — select initial top-K beams from vocabulary
//
// Uses a simple two-phase block reduction (serial merge by thread 0) since
// this runs only once per decode, not in the hot loop.
// =============================================================================

template <int BLOCK_SIZE>
__global__ void first_step_kernel(const float* __restrict__ log_prob, int batch_stride,
                                  int seq_stride, int vocab_stride,
                                  const int* __restrict__ select_seqs,
                                  const int* __restrict__ select_seq_lens,
                                  float2* __restrict__ pprev, int* __restrict__ clast,
                                  int* __restrict__ clen, int* __restrict__ clist,
                                  float* __restrict__ score, int beam, int ldbeam, int ldseq_len,
                                  int vocab_size, int blank_id, int batch, int max_seq_len) {
    int bid = blockIdx.x;
    if (bid >= batch)
        return;
    if (select_seq_lens[bid] == 0)
        return;

    int first_t = select_seqs[bid * max_seq_len];
    // need_add_blank: true when there are leading blank frames before the first
    // selected frame (matching reference's need_add_blank(batch_id, 0) logic).
    bool need_add_blank = (first_t > 0);

    // Phase 1: each thread finds its local best non-blank token.
    extern __shared__ char smem[];
    float* s_keys = reinterpret_cast<float*>(smem);
    int* s_vals = reinterpret_cast<int*>(s_keys + BLOCK_SIZE);

    float local_best = NEG_INF;
    int local_best_id = -1;
    for (int c = threadIdx.x; c < vocab_size; c += BLOCK_SIZE) {
        if (c == blank_id)
            continue;
        float p = log_prob[bid * batch_stride + first_t * seq_stride + c * vocab_stride];
        if (p > local_best) {
            local_best = p;
            local_best_id = c;
        }
    }
    s_keys[threadIdx.x] = local_best;
    s_vals[threadIdx.x] = local_best_id;
    __syncthreads();

    // Phase 2: thread 0 merges per-thread bests into global top-beam.
    // Note: each thread may cover multiple vocab items and only contributed its
    // single best. For top-1 this is exact; for top-beam we do a full second
    // pass below to ensure correctness when beam candidates span the same thread.
    if (threadIdx.x == 0) {
        // Full serial top-beam scan over all non-blank vocab items.
        float topk_keys[128];
        int topk_vals[128];
        for (int k = 0; k < beam; ++k) {
            topk_keys[k] = NEG_INF;
            topk_vals[k] = -1;
        }

        for (int c = 0; c < vocab_size; ++c) {
            if (c == blank_id)
                continue;
            float p = log_prob[bid * batch_stride + first_t * seq_stride + c * vocab_stride];
            // Replace the minimum in the top-beam heap.
            int min_idx = 0;
            for (int k = 1; k < beam; ++k)
                if (topk_keys[k] < topk_keys[min_idx])
                    min_idx = k;
            if (p > topk_keys[min_idx]) {
                topk_keys[min_idx] = p;
                topk_vals[min_idx] = c;
            }
        }

        // Sort top-beam descending (insertion sort).
        for (int i = 1; i < beam; ++i) {
            float ki = topk_keys[i];
            int vi = topk_vals[i];
            int j = i - 1;
            while (j >= 0 && topk_keys[j] < ki) {
                topk_keys[j + 1] = topk_keys[j];
                topk_vals[j + 1] = topk_vals[j];
                --j;
            }
            topk_keys[j + 1] = ki;
            topk_vals[j + 1] = vi;
        }

        // Write results (matching reference first_matrix__bitonic_topk_kernel).
        for (int k = 0; k < beam; ++k) {
            int base = bid * ldbeam + k;
            int token = topk_vals[k];
            float key = topk_keys[k];

            if (token >= 0 && token != blank_id) {
                // Non-blank top-K token: initialise pprev per need_add_blank flag.
                // If need_add_blank=true: blank path already accumulated → (key, NEG_INF)
                // If need_add_blank=false: fresh start, nonblank path → (NEG_INF, key)
                float2 xy = need_add_blank ? make_float2(key, NEG_INF) : make_float2(NEG_INF, key);
                pprev[base] = xy;
                int shift = clen[base];  // should be 0 on first step
                clist[bid * beam * ldseq_len + k * ldseq_len + shift] = token;
                clen[base] += 1;
                clast[base] = token;
                score[base] = key;
            } else {
                // No valid token (e.g., beam > vocab_size - 1 non-blank tokens)
                pprev[base] = make_float2(NEG_INF, NEG_INF);
                clast[base] = blank_id;
                clen[base] = 0;
                score[base] = NEG_INF;
            }
        }
    }
}

// =============================================================================
// Kernel: Probability matrix computation — v2 semantics
//
// For each (beam, non-blank char) pair:
//   ptable[idout]  = NEG_INF  (non-blank chars never carry a blank-ending path)
//   ptablen[idout] = log-prob of extending beam by this char (non-blank path)
//
// Special case — char == last_char of beam:
//   ptablen[blank_pos_of_beam] += cur_prob + prev_nonblank
//   (the "same char again" path folds into the blank slot of this beam)
//
// =============================================================================

__global__ void prob_matrix_kernel(const float* __restrict__ log_prob, int batch_stride,
                                   int seq_stride, int vocab_stride,
                                   const int* __restrict__ select_seqs,
                                   const int* __restrict__ select_seq_lens, int step,
                                   float2* __restrict__ pprev, float* __restrict__ ptable,
                                   float* __restrict__ ptablen, const int* __restrict__ clast,
                                   int ldc, int beam, int ldbeam, int batch, int blank_id,
                                   int space_id, int max_seq_len) {
    const int bid = blockIdx.y;
    if (bid >= batch)
        return;
    if (step >= select_seq_lens[bid])
        return;

    int t = select_seqs[bid * max_seq_len + step];

    // When there are skipped (blank-dominant) frames between the previous
    // selected frame and this one, we must account for the blank path that
    // passes through them.  The effective pprev after one or more blank frames
    // collapses both blank and non-blank paths into the blank slot:
    //   effective_blank    = logsumexp(prev_blank, prev_nonblank)
    //   effective_nonblank = NEG_INF
    bool need_add_blank =
        (select_seqs[bid * max_seq_len + step] > select_seqs[bid * max_seq_len + step - 1] + 1);

    int total = ldc * beam;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (; tid < total; tid += stride) {
        int beam_idx = tid / ldc;
        int char_idx = tid - beam_idx * ldc;

        if (beam_idx >= beam)
            continue;
        // Blank and space are handled by prob_space_blank_kernel.
        if (char_idx == blank_id)
            continue;
        if (space_id >= 0 && char_idx == space_id)
            continue;

        int pprev_idx = bid * ldbeam + beam_idx;
        float2 raw_prev = pprev[pprev_idx];
        int last_char = clast[pprev_idx];

        // Apply blank-frame adjustment when intermediate blank frames were skipped.
        float2 prev;
        if (need_add_blank) {
            prev = make_float2(logsumexp(raw_prev.x, raw_prev.y), NEG_INF);
        } else {
            prev = raw_prev;
        }

        float cur_prob = log_prob[bid * batch_stride + t * seq_stride + char_idx * vocab_stride];

        int idout = char_idx + (beam_idx + bid * beam) * ldc;

        float out_prob;
        if (last_char == char_idx) {
            // Same char as last: only blank path can transition (prevents CTC repeat
            // collapse). Also write the "repeat without blank" contribution to the
            // blank slot of this beam (will be merged into the blank prob later).
            out_prob = logprob_add(cur_prob, prev.x);  // cur_prob + prev_blank

            // prev_nonblank + cur_prob → folds into blank slot (reference semantics).
            int blank_slot = blank_id + (bid * beam + beam_idx) * ldc;
            ptablen[blank_slot] = logprob_add(cur_prob, prev.y);
        } else {
            // Different char: extend from both blank and nonblank paths.
            out_prob = logprob_add(cur_prob, logsumexp(prev.x, prev.y));
        }

        // Non-blank chars never carry a blank-ending probability.
        ptable[idout] = NEG_INF;
        ptablen[idout] = out_prob;
    }
}

// =============================================================================
// Kernel: Blank and space probability update — v2 semantics
//
// Blank:
//   ptable[blank_slot]  = blank_prob + logsumexp(prev_blank, prev_nonblank)
//   ptablen[blank_slot] = NEG_INF  iff  last_char == blank_id
//   (when last_char != blank_id, ptablen[blank_slot] was already written by
//    prob_matrix_kernel for the char matching last_char)
//
// Space (if enabled):
//   ptablen[space_slot] = space_prob + logsumexp(prev_blank, prev_nonblank)
//   ptable[space_slot]  = NEG_INF
//
// =============================================================================

__global__ void prob_space_blank_kernel(const float* __restrict__ log_prob, int batch_stride,
                                        int seq_stride, int vocab_stride,
                                        const int* __restrict__ select_seqs,
                                        const int* __restrict__ select_seq_lens, int step,
                                        float2* __restrict__ pprev, float* __restrict__ ptable,
                                        float* __restrict__ ptablen, const int* __restrict__ clast,
                                        int ldc, int beam, int ldbeam, int batch, int blank_id,
                                        int space_id, int max_seq_len) {
    const int bid = blockIdx.x;
    if (bid >= batch)
        return;
    if (step >= select_seq_lens[bid])
        return;

    int t = select_seqs[bid * max_seq_len + step];
    int beam_idx = threadIdx.x;
    if (beam_idx >= beam)
        return;

    // Apply the same blank-frame adjustment as prob_matrix_kernel.
    bool need_add_blank =
        (select_seqs[bid * max_seq_len + step] > select_seqs[bid * max_seq_len + step - 1] + 1);

    int pprev_idx = bid * ldbeam + beam_idx;
    float2 raw_prev = pprev[pprev_idx];
    int last_char = clast[pprev_idx];

    float2 prev;
    if (need_add_blank) {
        prev = make_float2(logsumexp(raw_prev.x, raw_prev.y), NEG_INF);
    } else {
        prev = raw_prev;
    }

    // --- Blank ---
    float blank_prob = log_prob[bid * batch_stride + t * seq_stride + blank_id * vocab_stride];
    int blank_slot = blank_id + (bid * beam + beam_idx) * ldc;
    ptable[blank_slot] = logprob_add(blank_prob, logsumexp(prev.x, prev.y));
    // Only write ptablen[blank_slot] = NEG_INF when last_char == blank_id.
    // Otherwise prob_matrix_kernel already wrote ptablen[blank_slot] for the
    // char that equals last_char (the "same-char without blank" contribution).
    if (need_add_blank || last_char == blank_id) {
        // After blank-frame adjustment, prev_nonblank is NEG_INF, so the
        // same-char contribution to ptablen[blank_slot] (which folds
        // prev_nonblank × cur_prob) is already negligible.  Clear it to avoid
        // stale values from the previous step.
        ptablen[blank_slot] = NEG_INF;
    }

    // --- Space (optional) ---
    if (space_id >= 0 && space_id != blank_id) {
        float space_prob = log_prob[bid * batch_stride + t * seq_stride + space_id * vocab_stride];
        int space_slot = space_id + (bid * beam + beam_idx) * ldc;
        ptablen[space_slot] = logprob_add(space_prob, logsumexp(prev.x, prev.y));
        ptable[space_slot] = NEG_INF;
    }
}

// =============================================================================
// Kernel: Merge duplicate prefixes — v2 semantics
//
// For each pair (shorter_beam = blockIdx.x, longer_beam = threadIdx.x):
//   If clen[longer] - 1 == clen[shorter] AND
//      clist[longer][0..clen[shorter]-1] == clist[shorter]:
//     then longer_beam = shorter_beam + clast[longer_beam]
//
// In that case, the extension of shorter_beam by clast[longer_beam] is
// the same CTC prefix as longer_beam. Fold it into longer_beam's blank slot:
//   ptable[blank_of_longer]  = logsumexp(ptable[blank_of_longer],
//                                         ptable[clast_in_shorter_slot])
//   ptablen[blank_of_longer] = logsumexp(ptablen[blank_of_longer],
//                                         ptablen[clast_in_shorter_slot])
//   ptable/ptablen[clast_in_shorter_slot] = NEG_INF
//
// Grid: (beam, batch), Block: (ldbeam, 1)
// =============================================================================

__global__ void merge_kernel(const int* __restrict__ select_seq_lens, int step,
                             float* __restrict__ ptable, float* __restrict__ ptablen,
                             const int* __restrict__ clast, const int* __restrict__ clist,
                             const int* __restrict__ clen, int ldc, int beam, int ldbeam,
                             int ldseq_len, int batch, int blank_id) {
    const int bid = blockIdx.y;
    if (bid >= batch)
        return;
    if (step >= select_seq_lens[bid])
        return;

    // Cache all beam lengths in shared memory.
    __shared__ int smem_clen[128];  // beam <= 128
    if (threadIdx.x < beam)
        smem_clen[threadIdx.x] = clen[threadIdx.x + bid * ldbeam];
    __syncthreads();

    int shorter_beam = blockIdx.x;  // i
    int longer_beam = threadIdx.x;  // j

    if (longer_beam < beam && (smem_clen[longer_beam] - 1) == smem_clen[shorter_beam]) {
        // j is exactly one token longer than i.
        if (seq_compare(smem_clen[shorter_beam],
                        clist + longer_beam * ldseq_len + bid * beam * ldseq_len,
                        clist + shorter_beam * ldseq_len + bid * beam * ldseq_len)) {
            // j's prefix == i's sequence: j = i + clast[j]
            // Merge i's "extension by clast[j]" into j's blank slot.
            int tidin = clast[longer_beam + bid * ldbeam] + (shorter_beam + bid * beam) * ldc;
            int tidout = blank_id + (longer_beam + bid * beam) * ldc;

            ptable[tidout] = logsumexp(ptable[tidout], ptable[tidin]);
            ptablen[tidout] = logsumexp(ptablen[tidout], ptablen[tidin]);
            ptable[tidin] = NEG_INF;
            ptablen[tidin] = NEG_INF;
        }
    }
}

// =============================================================================
// Top-K Phase 1: per-batch, multi-block streaming sort
//
// Each block independently finds its local top-beam from a contiguous chunk of
// beam*ldc items using the "streaming BlockRadixSort" pattern from the
// reference: keep the current top-K in the sorted array and slide in new
// items replacing the non-top-K positions, re-sorting each iteration.
//
// Output: topk_key_buffer  [batch * bxs * beam]
//         topk_value_buffer [batch * bxs * beam]
// where bxs = gridDim.x.
// =============================================================================

template <int BLOCK_SIZE, int ITEMS_PER_THREAD>
__global__ __launch_bounds__(BLOCK_SIZE) void topk_phase1_kernel(
    const int* __restrict__ select_seq_lens, int step, const float* __restrict__ ptable,
    const float* __restrict__ ptablen, int ldc, int beam, int batch, float* topk_key_buffer,
    int* topk_value_buffer) {
    const int bid = blockIdx.y;
    if (bid >= batch)
        return;
    if (step >= select_seq_lens[bid])
        return;

    const int bx = blockIdx.x;
    const int bxs = gridDim.x;
    const int all_items = ldc * beam;
    const int tx = threadIdx.x;

    // Chunk assigned to this block (contiguous range).
    const int chunk_size = (all_items + bxs - 1) / bxs;
    const int chunk_start = bx * chunk_size;
    const int chunk_end = min(chunk_start + chunk_size, all_items);
    const int my_count = chunk_end - chunk_start;

    typedef cub::BlockRadixSort<float, BLOCK_SIZE, ITEMS_PER_THREAD, int> BlockSortT;
    __shared__ typename BlockSortT::TempStorage temp_storage;

    float keys[ITEMS_PER_THREAD];
    int values[ITEMS_PER_THREAD];

    const int items_per_iter = BLOCK_SIZE * ITEMS_PER_THREAD;

    // First iteration: load first min(items_per_iter, my_count) items.
#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        int pos = BLOCK_SIZE * ITEM + tx;  // striped position
        int global_idx = chunk_start + pos;
        if (pos < my_count && global_idx < all_items) {
            float p = ptable[bid * all_items + global_idx];
            float pn = ptablen[bid * all_items + global_idx];
            keys[ITEM] = logsumexp(p, pn);
            values[ITEM] = global_idx;
        } else {
            keys[ITEM] = NEG_INF;
            values[ITEM] = chunk_start + pos;
        }
    }
    BlockSortT{temp_storage}.SortDescendingBlockedToStriped(keys, values);
    __syncthreads();

    // Subsequent iterations: replace non-top-K positions with new items.
    const int stride = items_per_iter - beam;
    for (int offset = items_per_iter; offset < my_count; offset += stride) {
#pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
            int striped_pos = BLOCK_SIZE * ITEM + tx;
            int new_local = striped_pos - beam;
            int new_global = chunk_start + offset + new_local;
            if (new_local >= 0) {
                if (new_global < chunk_end) {
                    float p = ptable[bid * all_items + new_global];
                    float pn = ptablen[bid * all_items + new_global];
                    keys[ITEM] = logsumexp(p, pn);
                    values[ITEM] = new_global;
                } else {
                    keys[ITEM] = NEG_INF;
                    values[ITEM] = new_global;
                }
            }
            // striped_pos < beam → keep previous top-K value (no overwrite).
        }
        BlockSortT{temp_storage}.SortDescendingBlockedToStriped(keys, values);
        __syncthreads();
    }

    // Write local top-beam to output buffer (striped position k → smem_keys[k]).
    const int out_offset = (bid * bxs + bx) * beam;
#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        int striped_pos = BLOCK_SIZE * ITEM + tx;
        if (striped_pos < beam) {
            topk_key_buffer[out_offset + striped_pos] = keys[ITEM];
            topk_value_buffer[out_offset + striped_pos] = values[ITEM];
        }
    }
}

// =============================================================================
// Top-K Phase 2: reduce Phase-1 results + update beam state
//
// One block per batch. Reads bxs*beam items from Phase-1 buffers, finds the
// global top-beam using the same streaming sort, then updates pprev, clast,
// clen_dst, clist_dst, and score.
//
// WRITE_THREADS: sub-warp size for parallel clist copy.
// =============================================================================

template <int BLOCK_SIZE, int ITEMS_PER_THREAD, int WRITE_THREADS = 8>
__global__ __launch_bounds__(BLOCK_SIZE) void topk_phase2_kernel(
    const int* __restrict__ select_seq_lens, int step,
    int items_per_batch,  // = bxs * beam
    int beam, int batch, float* __restrict__ topk_key_buffer, int* __restrict__ topk_value_buffer,
    int ldc, int ldbeam, int ldseq_len, float2* __restrict__ pprev,
    const float* __restrict__ ptable, const float* __restrict__ ptablen, int* __restrict__ clast,
    int* __restrict__ clen_src, int* __restrict__ clen_dst, int* __restrict__ clist_src,
    int* __restrict__ clist_dst, float* __restrict__ score, int blank_id,
    const int* __restrict__ select_seqs, int max_seq_len) {
    const int bid = blockIdx.x;
    if (bid >= batch)
        return;
    if (step >= select_seq_lens[bid])
        return;

    // need_add_blank: true when there are blank frames between the previous
    // selected step and this one (or leading blanks for step==0).
    bool need_add_blank = false;
    if (step == 0) {
        need_add_blank = (select_seqs[bid * max_seq_len] > 0);
    } else {
        int cur_t = select_seqs[bid * max_seq_len + step];
        int prev_t = select_seqs[bid * max_seq_len + step - 1];
        need_add_blank = (cur_t > prev_t + 1);
    }

    const int tx = threadIdx.x;
    const int rw_offset = bid * items_per_batch;

    typedef cub::BlockRadixSort<float, BLOCK_SIZE, ITEMS_PER_THREAD, int> BlockSortT;
    // Shared memory for sort temp storage + top-beam results.
    __shared__ union {
        typename BlockSortT::TempStorage temp_storage;
        struct {
            float keys[128];
            int vals[128];
            int src_clast[128];
            int src_clen[128];
        } topk;
    } smem;

    float keys[ITEMS_PER_THREAD];
    int values[ITEMS_PER_THREAD];

    const int items_per_iter = BLOCK_SIZE * ITEMS_PER_THREAD;

    // First iteration.
#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        int pos = BLOCK_SIZE * ITEM + tx;
        if (pos < items_per_batch) {
            keys[ITEM] = topk_key_buffer[rw_offset + pos];
            values[ITEM] = topk_value_buffer[rw_offset + pos];
        } else {
            keys[ITEM] = NEG_INF;
            values[ITEM] = pos;
        }
    }
    BlockSortT{smem.temp_storage}.SortDescendingBlockedToStriped(keys, values);
    __syncthreads();

    const int stride = items_per_iter - beam;
    for (int offset = items_per_iter; offset < items_per_batch; offset += stride) {
#pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
            int striped_pos = BLOCK_SIZE * ITEM + tx;
            int new_local = striped_pos - beam;
            int new_idx = rw_offset + offset + new_local;
            if (new_local >= 0) {
                if ((offset + new_local) < items_per_batch) {
                    keys[ITEM] = topk_key_buffer[new_idx];
                    values[ITEM] = topk_value_buffer[new_idx];
                } else {
                    keys[ITEM] = NEG_INF;
                }
            }
        }
        BlockSortT{smem.temp_storage}.SortDescendingBlockedToStriped(keys, values);
        __syncthreads();
    }

    // Write top-beam to shared memory.
#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        int striped_pos = BLOCK_SIZE * ITEM + tx;
        if (striped_pos < beam) {
            smem.topk.keys[striped_pos] = keys[ITEM];
            smem.topk.vals[striped_pos] = values[ITEM];
        }
    }
    __syncthreads();

    // Cache source clast/clen (prevent write-before-read races when src_beam is
    // also a dst_beam).
    for (int k = tx; k < beam; k += BLOCK_SIZE) {
        smem.topk.src_clast[k] = clast[bid * ldbeam + k];
        smem.topk.src_clen[k] = clen_src[bid * ldbeam + k];
    }
    __syncthreads();

    // Update state using sub-warp parallelism for clist copy.
    const int sub_warp_id = tx / WRITE_THREADS;
    const int tid_in_sub = tx % WRITE_THREADS;
    const int sub_warps = BLOCK_SIZE / WRITE_THREADS;

    for (int out_beam = sub_warp_id; out_beam < beam; out_beam += sub_warps) {
        int id = smem.topk.vals[out_beam];
        int src_beam = id / ldc;
        int char_id = id - src_beam * ldc;
        float new_score = smem.topk.keys[out_beam];
        int prevlen = smem.topk.src_clen[src_beam];

        // Parallel clist copy (WRITE_THREADS threads per output beam).
        for (int s = tid_in_sub; s < prevlen; s += WRITE_THREADS) {
            clist_dst[bid * beam * ldseq_len + out_beam * ldseq_len + s] =
                clist_src[bid * beam * ldseq_len + src_beam * ldseq_len + s];
        }

        if (tid_in_sub == 0) {
            int dst_base = bid * ldbeam + out_beam;

            if (char_id == blank_id) {
                // Blank extension: keep same prefix, propagate last char.
                clast[dst_base] = smem.topk.src_clast[src_beam];
                clen_dst[dst_base] = prevlen;
            } else {
                // Non-blank extension: append new character.
                clast[dst_base] = char_id;
                clen_dst[dst_base] = prevlen + 1;
                if (prevlen < ldseq_len) {
                    clist_dst[bid * beam * ldseq_len + out_beam * ldseq_len + prevlen] = char_id;
                }
            }

            score[dst_base] = new_score;

            // pprev for the next step (matching reference topk_reduce logic):
            //   need_add_blank=true  → collapsed score in blank slot: {score, NEG_INF}
            //   need_add_blank=false → preserve the (p_blank, p_nonblank) split
            float p = ptable[bid * ldc * beam + id];
            float pn = ptablen[bid * ldc * beam + id];
            float2 pprev_next;
            if (need_add_blank) {
                pprev_next = make_float2(new_score, NEG_INF);
            } else {
                pprev_next = make_float2(p, pn);
            }
            pprev[dst_base] = pprev_next;
        }
    }
}

// =============================================================================
// Kernel: Copy beam state for batches with different double-buffer parity
//
// When the global max_select_seq_len has parity P but a specific batch's
// select_seq_len has parity Q != P, that batch's results were left in the
// wrong (1 - P) buffer. Copy them to buffer P.
//
// Matches copy_list_len_for_diff_parity_simple_kernel from reference.
// =============================================================================

__global__ void fixup_parity_kernel(const int* __restrict__ select_seq_lens, int max_select_seq_len,
                                    int* __restrict__ clen0, int* __restrict__ clen1,
                                    int* __restrict__ clist0, int* __restrict__ clist1, int ldbeam,
                                    int ldseq_len, int beam, int batch, int final_parity) {
    const int bid = blockIdx.x;
    if (bid >= batch)
        return;

    int nsteps = select_seq_lens[bid];
    // For step=0 (first_step_kernel), results go to clen[0]/clist[0].
    // For subsequent steps, dst_parity alternates: step % 2.
    int batch_parity;
    if (nsteps <= 1) {
        batch_parity = 0;
    } else {
        // Last active step index = nsteps - 1; that step writes to dst parity.
        batch_parity = (nsteps - 1) % 2;
    }
    if (batch_parity == final_parity)
        return;

    int* src_clen = (batch_parity == 0) ? clen0 : clen1;
    int* dst_clen = (final_parity == 0) ? clen0 : clen1;
    int* src_clist = (batch_parity == 0) ? clist0 : clist1;
    int* dst_clist = (final_parity == 0) ? clist0 : clist1;

    for (int k = threadIdx.x; k < beam; k += blockDim.x) {
        int idx = bid * ldbeam + k;
        int len = src_clen[idx];
        dst_clen[idx] = len;
        for (int s = 0; s < len && s < ldseq_len; ++s) {
            dst_clist[bid * beam * ldseq_len + k * ldseq_len + s] =
                src_clist[bid * beam * ldseq_len + k * ldseq_len + s];
        }
    }
}

// =============================================================================
// Host-side workspace management
// =============================================================================

constexpr size_t align_size(size_t size) {
    return (size + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
}

inline int align16(int val) {
    return ((val - 1) / 16 + 1) * 16;
}

inline size_t calculate_workspace_size(int batch, int beam, int vocab_size, int max_seq_len) {
    int ldbeam = align16(beam);
    int ldseq_len = align16(max_seq_len);
    int ldc = vocab_size;

    size_t total = 0;
    total += align_size(sizeof(float2) * batch * ldbeam);             // pprev
    total += align_size(sizeof(float) * batch * beam * ldc);          // ptable
    total += align_size(sizeof(float) * batch * beam * ldc);          // ptablen
    total += align_size(sizeof(int) * batch * ldbeam);                // clast
    total += align_size(sizeof(int) * batch * ldbeam) * 2;            // clen[0..1]
    total += align_size(sizeof(int) * batch * beam * ldseq_len) * 2;  // clist[0..1]
    total += align_size(sizeof(int) * batch * ldbeam);                // ptid (unused)
    total += align_size(sizeof(float) * batch * ldbeam);              // score
    // topk buffers: batch * MAX_BLOCKS_PER_BATCH * beam (Phase 1 output)
    total += align_size(sizeof(float) * batch * MAX_BLOCKS_PER_BATCH * beam);
    total += align_size(sizeof(int) * batch * MAX_BLOCKS_PER_BATCH * beam);
    total += align_size(sizeof(int) * batch * max_seq_len);  // select_seqs
    total += align_size(sizeof(int) * batch);                // select_seq_lens
    total += ALIGN_BYTES;
    return total;
}

inline void init_internal_data(InternalData* data, void* workspace, int batch, int beam,
                               int vocab_size, int max_seq_len) {
    data->batch = batch;
    data->beam = beam;
    data->vocab_size = vocab_size;
    data->ldc = vocab_size;
    data->ldbeam = align16(beam);
    data->ldseq_len = align16(max_seq_len);
    data->max_seq_len = max_seq_len;
    data->ldc_divmod = FastDivmod(vocab_size);
    data->paged = paged_memory::PagedSequenceState();  // all null (flat mode)

    int ldbeam = data->ldbeam;
    int ldseq_len = data->ldseq_len;
    int ldc = data->ldc;

    char* ptr = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(workspace) + ALIGN_BYTES - 1) /
                                        ALIGN_BYTES * ALIGN_BYTES);

#define ALLOC_BUF(name, type, count)           \
    data->name = reinterpret_cast<type*>(ptr); \
    ptr += align_size(sizeof(type) * (count));

    ALLOC_BUF(pprev, float2, batch * ldbeam)
    ALLOC_BUF(ptable, float, batch * beam * ldc)
    ALLOC_BUF(ptablen, float, batch * beam * ldc)
    ALLOC_BUF(clast, int, batch* ldbeam)
    ALLOC_BUF(clen[0], int, batch* ldbeam)
    ALLOC_BUF(clen[1], int, batch* ldbeam)
    ALLOC_BUF(clist[0], int, batch * beam * ldseq_len)
    ALLOC_BUF(clist[1], int, batch * beam * ldseq_len)
    ALLOC_BUF(ptid, int, batch* ldbeam)
    ALLOC_BUF(score, float, batch* ldbeam)
    ALLOC_BUF(topk_key_buffer, float, batch * MAX_BLOCKS_PER_BATCH * beam)
    ALLOC_BUF(topk_value_buffer, int, batch * MAX_BLOCKS_PER_BATCH * beam)
    ALLOC_BUF(select_seqs, int, batch* max_seq_len)
    ALLOC_BUF(select_seq_lens, int, batch)

#undef ALLOC_BUF
}

// =============================================================================
// Paged workspace size (replaces the two flat clist arrays with a paged region)
// =============================================================================

inline size_t calculate_paged_workspace_size(int batch, int beam, int vocab_size, int max_seq_len,
                                             int page_size = paged_memory::DEFAULT_PAGE_SIZE,
                                             int num_pages = 0) {
    int ldbeam = align16(beam);
    int ldc = vocab_size;

    size_t total = 0;
    total += align_size(sizeof(float2) * batch * ldbeam);
    total += align_size(sizeof(float) * batch * beam * ldc);
    total += align_size(sizeof(float) * batch * beam * ldc);
    total += align_size(sizeof(int) * batch * ldbeam);
    total += align_size(sizeof(int) * batch * ldbeam) * 2;  // clen[0..1]
    // clist[0..1] are OMITTED — replaced by paged region
    total += align_size(sizeof(int) * batch * ldbeam);      // ptid
    total += align_size(sizeof(float) * batch * ldbeam);
    total += align_size(sizeof(float) * batch * MAX_BLOCKS_PER_BATCH * beam);
    total += align_size(sizeof(int) * batch * MAX_BLOCKS_PER_BATCH * beam);
    total += align_size(sizeof(int) * batch * max_seq_len);
    total += align_size(sizeof(int) * batch);
    total += ALIGN_BYTES;
    // Paged region
    total += paged_memory::calculate_paged_region_size(batch, beam, max_seq_len, page_size,
                                                        num_pages);
    return total;
}

// =============================================================================
// Paged init_internal_data — bump-allocates base fields then initialises
// PagedSequenceState from the remaining workspace. `clist[0/1]` are left null.
// =============================================================================

inline void init_internal_data_paged(InternalData* data, void* workspace,
                                     int batch, int beam, int vocab_size, int max_seq_len,
                                     int page_size, int num_pages,
                                     cudaStream_t stream) {
    data->batch = batch;
    data->beam = beam;
    data->vocab_size = vocab_size;
    data->ldc = vocab_size;
    data->ldbeam = align16(beam);
    data->ldseq_len = align16(max_seq_len);
    data->max_seq_len = max_seq_len;
    data->ldc_divmod = FastDivmod(vocab_size);
    // clist stays null in paged mode
    data->clist[0] = nullptr;
    data->clist[1] = nullptr;

    int ldbeam = data->ldbeam;
    int ldc = data->ldc;

    char* ptr = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(workspace) + ALIGN_BYTES - 1) /
                                        ALIGN_BYTES * ALIGN_BYTES);

#define ALLOC_BUF_P(name, type, count)         \
    data->name = reinterpret_cast<type*>(ptr); \
    ptr += align_size(sizeof(type) * (count));

    ALLOC_BUF_P(pprev, float2, batch * ldbeam)
    ALLOC_BUF_P(ptable, float, batch * beam * ldc)
    ALLOC_BUF_P(ptablen, float, batch * beam * ldc)
    ALLOC_BUF_P(clast, int, batch * ldbeam)
    ALLOC_BUF_P(clen[0], int, batch * ldbeam)
    ALLOC_BUF_P(clen[1], int, batch * ldbeam)
    // clist[0] and clist[1] are NOT allocated; paged region follows instead
    ALLOC_BUF_P(ptid, int, batch * ldbeam)
    ALLOC_BUF_P(score, float, batch * ldbeam)
    ALLOC_BUF_P(topk_key_buffer, float, batch * MAX_BLOCKS_PER_BATCH * beam)
    ALLOC_BUF_P(topk_value_buffer, int, batch * MAX_BLOCKS_PER_BATCH * beam)
    ALLOC_BUF_P(select_seqs, int, batch * max_seq_len)
    ALLOC_BUF_P(select_seq_lens, int, batch)

#undef ALLOC_BUF_P

    // Initialise paged state from remaining workspace
    paged_memory::init_paged_state(&data->paged, ptr, batch, beam, max_seq_len,
                                   page_size, num_pages, stream);
}

// =============================================================================
// Host launcher: single step of CTC prefix beam search
// =============================================================================

inline cudaError_t ctc_prefix_beam_search_step(InternalData* data, const float* log_prob,
                                               int batch_stride, int seq_stride, int vocab_stride,
                                               int step, bool is_last_step, int blank_id,
                                               int space_id, cudaStream_t stream) {
    int batch = data->batch;
    int beam = data->beam;
    int ldc = data->ldc;
    int ldbeam = data->ldbeam;
    int ldseq_len = data->ldseq_len;
    int max_seq_len = data->max_seq_len;

    // Double-buffer parity: step=0 writes to clen[0]/clist[0] (via first_step_kernel).
    // Steps 1,2,... alternate src/dst: dst_parity = step % 2.
    int src_parity = (step == 0) ? 0 : ((step - 1) % 2);
    int dst_parity = (step == 0) ? 0 : (step % 2);

    if (step == 0) {
        // Initialise beam state from the first selected frame.
        int smem_size = (sizeof(float) + sizeof(int)) * 256;  // s_keys + s_vals for BLOCK_SIZE=256
        first_step_kernel<256><<<batch, 256, smem_size, stream>>>(
            log_prob, batch_stride, seq_stride, vocab_stride, data->select_seqs,
            data->select_seq_lens, data->pprev, data->clast, data->clen[0], data->clist[0],
            data->score, beam, ldbeam, ldseq_len, data->vocab_size, blank_id, batch, max_seq_len);
    } else {
        // --- 1. Compute probability matrix (non-blank chars) ---
        {
            int total = ldc * beam;
            int threads = 256;
            int bx = min((total + threads - 1) / threads, MAX_BLOCKS / max(batch, 1));
            dim3 grid(bx, batch);
            prob_matrix_kernel<<<grid, threads, 0, stream>>>(
                log_prob, batch_stride, seq_stride, vocab_stride, data->select_seqs,
                data->select_seq_lens, step, data->pprev, data->ptable, data->ptablen, data->clast,
                ldc, beam, ldbeam, batch, blank_id, space_id, max_seq_len);
        }

        // --- 2. Blank / space probability ---
        prob_space_blank_kernel<<<batch, ldbeam, 0, stream>>>(
            log_prob, batch_stride, seq_stride, vocab_stride, data->select_seqs,
            data->select_seq_lens, step, data->pprev, data->ptable, data->ptablen, data->clast, ldc,
            beam, ldbeam, batch, blank_id, space_id, max_seq_len);

        // --- 3. Merge duplicate prefixes ---
        {
            dim3 merge_grid(beam, batch);
            merge_kernel<<<merge_grid, ldbeam, 0, stream>>>(
                data->select_seq_lens, step, data->ptable, data->ptablen, data->clast,
                data->clist[src_parity], data->clen[src_parity], ldc, beam, ldbeam, ldseq_len,
                batch, blank_id);
        }

        // --- 4. Top-K selection: Phase 1 (multi-block per batch) ---
        {
            constexpr int P1_BLOCK = 128;
            constexpr int P1_IPT = 4;
            int all_items = ldc * beam;
            int bxs = min(MAX_BLOCKS_PER_BATCH,
                          max(1, (all_items + P1_BLOCK * P1_IPT - 1) / (P1_BLOCK * P1_IPT)));
            bxs = min(bxs, MAX_BLOCKS / max(batch, 1));
            dim3 p1_grid(bxs, batch);
            topk_phase1_kernel<P1_BLOCK, P1_IPT><<<p1_grid, P1_BLOCK, 0, stream>>>(
                data->select_seq_lens, step, data->ptable, data->ptablen, ldc, beam, batch,
                data->topk_key_buffer, data->topk_value_buffer);

            // --- 5. Top-K Phase 2: reduce + state update ---
            constexpr int P2_BLOCK = 128;
            constexpr int P2_IPT = 2;
            int items_per_batch = bxs * beam;
            topk_phase2_kernel<P2_BLOCK, P2_IPT><<<batch, P2_BLOCK, 0, stream>>>(
                data->select_seq_lens, step, items_per_batch, beam, batch, data->topk_key_buffer,
                data->topk_value_buffer, ldc, ldbeam, ldseq_len, data->pprev, data->ptable,
                data->ptablen, data->clast, data->clen[src_parity], data->clen[dst_parity],
                data->clist[src_parity], data->clist[dst_parity], data->score, blank_id,
                data->select_seqs, max_seq_len);
        }
    }

    return cudaGetLastError();
}

// =============================================================================
// Host launcher: full batch decode (offline)
// =============================================================================

inline cudaError_t ctc_beam_search_decode_batch(
    const float* log_prob,  // [batch, seq_len, vocab_size]
    int batch_stride, int seq_stride, int vocab_stride,
    const int* seq_lengths,  // [batch]
    int* out_tokens,         // [batch, beam, max_out_len]
    int* out_lengths,        // [batch, beam]
    float* out_scores,       // [batch, beam]
    void* workspace, int batch, int beam, int vocab_size, int max_seq_len, int max_out_len,
    int blank_id, int space_id, float blank_threshold, cudaStream_t stream) {
    int ws_seq_len = max_seq_len > max_out_len ? max_seq_len : max_out_len;

    InternalData data;
    init_internal_data(&data, workspace, batch, beam, vocab_size, ws_seq_len);

    // Initialise beam-state buffers.
    cudaMemsetAsync(data.clast, 0, sizeof(int) * batch * data.ldbeam, stream);
    cudaMemsetAsync(data.clen[0], 0, sizeof(int) * batch * data.ldbeam, stream);
    cudaMemsetAsync(data.clen[1], 0, sizeof(int) * batch * data.ldbeam, stream);
    cudaMemsetAsync(data.clist[0], 0xff, sizeof(int) * batch * beam * data.ldseq_len, stream);
    cudaMemsetAsync(data.clist[1], 0xff, sizeof(int) * batch * beam * data.ldseq_len, stream);
    // ptable and ptablen must start at NEG_INF so that stale entries from
    // previous steps (or the initial state) don't corrupt probability lookups.
    // 0xcc → float bit pattern ≈ -1.7e38, which is effectively -FLT_MAX.
    cudaMemsetAsync(data.ptable, 0xcc, sizeof(float) * batch * beam * data.ldc, stream);
    cudaMemsetAsync(data.ptablen, 0xcc, sizeof(float) * batch * beam * data.ldc, stream);
    cudaMemsetAsync(data.select_seq_lens, 0, sizeof(int) * batch, stream);

    // Blank threshold: convert from probability space to log space.
    // User passes blank_threshold as a probability (e.g. 0.99).
    // We compare log_prob(blank) < log(threshold) in the kernel.
    float log_threshold = (blank_threshold <= 0.0f)   ? NEG_INF
                          : (blank_threshold >= 1.0f) ? 0.0f
                                                      : logf(blank_threshold);

    // Select frames: filter out blank-dominant frames.
    constexpr int SEL_BLOCK = 128;
    constexpr int SEL_IPT = 4;
    init_select_kernel<SEL_BLOCK, SEL_IPT><<<batch, SEL_BLOCK, 0, stream>>>(
        log_prob, batch_stride, seq_stride, vocab_stride, seq_lengths, batch, data.max_seq_len,
        blank_id, log_threshold, data.select_seqs, data.select_seq_lens);

    // Read select_seq_lens to determine the main loop bound.
    int* h_select_lens = new int[batch];
    cudaMemcpyAsync(h_select_lens, data.select_seq_lens, sizeof(int) * batch,
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    int max_select = 0;
    for (int b = 0; b < batch; ++b)
        if (h_select_lens[b] > max_select)
            max_select = h_select_lens[b];
    delete[] h_select_lens;
    data.max_select_seq_len = max_select;

    // Main decode loop.
    for (int step = 0; step < max_select; ++step) {
        bool is_last = (step == max_select - 1);
        cudaError_t err =
            ctc_prefix_beam_search_step(&data, log_prob, batch_stride, seq_stride, vocab_stride,
                                        step, is_last, blank_id, space_id, stream);
        if (err != cudaSuccess)
            return err;
    }

    // Determine which double-buffer holds final results.
    // step=0 → clen[0]/clist[0]; step=N-1 where N>1 → clen[(N-1)%2]/clist[(N-1)%2].
    int final_parity = (max_select <= 1) ? 0 : ((max_select - 1) % 2);

    // Fix batches whose last active step has different parity from final_parity.
    fixup_parity_kernel<<<batch, 32, 0, stream>>>(
        data.select_seq_lens, max_select, data.clen[0], data.clen[1], data.clist[0], data.clist[1],
        data.ldbeam, data.ldseq_len, beam, batch, final_parity);

    // Copy results to output tensors (strided memcpy).
    cudaMemcpy2DAsync(out_lengths, sizeof(int) * beam, data.clen[final_parity],
                      sizeof(int) * data.ldbeam, sizeof(int) * beam, batch,
                      cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(out_tokens, sizeof(int) * max_out_len, data.clist[final_parity],
                      sizeof(int) * data.ldseq_len, sizeof(int) * max_out_len, batch * beam,
                      cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(out_scores, sizeof(float) * beam, data.score, sizeof(float) * data.ldbeam,
                      sizeof(float) * beam, batch, cudaMemcpyDeviceToDevice, stream);

    return cudaGetLastError();
}

// =============================================================================
// Forward declarations for paged variants (defined after streaming functions)
// =============================================================================

__global__ void gather_paged_results_kernel(
    const int* page_storage,
    const int* block_table0, const int* block_table1,
    const int* select_seq_lens, int max_select_seq_len,
    const int* clen0, const int* clen1,
    const float* score, int ldbeam,
    int* out_tokens, int* out_lengths, float* out_scores,
    int batch, int beam, int max_out_len,
    int page_size, int max_lp);

inline cudaError_t ctc_prefix_beam_search_step_paged(
    InternalData* data, const float* log_prob,
    int batch_stride, int seq_stride, int vocab_stride,
    int step, int blank_id, int space_id, cudaStream_t stream);

// =============================================================================
// Streaming state
// =============================================================================

static constexpr size_t STATE_HEADER_SIZE = align_size(sizeof(InternalData) + sizeof(int) * 4);

inline size_t calculate_state_buffer_size(int batch, int beam, int vocab_size, int max_seq_len) {
    return STATE_HEADER_SIZE + calculate_workspace_size(batch, beam, vocab_size, max_seq_len);
}

// Paged streaming state buffer: same header layout, smaller workspace (no clist), plus paged region.
inline size_t calculate_paged_state_buffer_size(int batch, int beam, int vocab_size, int max_seq_len,
                                                int page_size = paged_memory::DEFAULT_PAGE_SIZE,
                                                int num_pages = 0) {
    return STATE_HEADER_SIZE +
           calculate_paged_workspace_size(batch, beam, vocab_size, max_seq_len, page_size, num_pages);
}

struct StreamingState {
    InternalData data;
    int current_step;
    int space_id;
    int blank_id;
    int use_paged_memory;  // 0 = flat (replaces _pad), 1 = paged
};

inline void init_streaming_state(void* state_buffer, int batch, int beam, int vocab_size,
                                 int max_seq_len, int blank_id, cudaStream_t stream) {
    StreamingState state;
    state.current_step = 0;
    state.space_id = -1;
    state.blank_id = blank_id;
    state.use_paged_memory = 0;

    void* workspace = reinterpret_cast<char*>(state_buffer) + STATE_HEADER_SIZE;
    init_internal_data(&state.data, workspace, batch, beam, vocab_size, max_seq_len);
    state.data.max_select_seq_len = max_seq_len;

    cudaMemcpyAsync(state_buffer, &state, sizeof(StreamingState), cudaMemcpyHostToDevice, stream);

    // Initialise GPU buffers.
    cudaMemsetAsync(state.data.clast, 0, sizeof(int) * batch * state.data.ldbeam, stream);
    cudaMemsetAsync(state.data.clen[0], 0, sizeof(int) * batch * state.data.ldbeam, stream);
    cudaMemsetAsync(state.data.clen[1], 0, sizeof(int) * batch * state.data.ldbeam, stream);
    cudaMemsetAsync(state.data.clist[0], 0xff, sizeof(int) * batch * beam * state.data.ldseq_len,
                    stream);
    cudaMemsetAsync(state.data.clist[1], 0xff, sizeof(int) * batch * beam * state.data.ldseq_len,
                    stream);
    cudaMemsetAsync(state.data.ptable, 0xcc, sizeof(float) * batch * beam * state.data.ldc, stream);
    cudaMemsetAsync(state.data.ptablen, 0xcc, sizeof(float) * batch * beam * state.data.ldc,
                    stream);

    // In streaming mode all frames are selected (no blank filtering); set up
    // select_seqs as identity mapping [0, 1, ..., max_seq_len-1].
    int* h_sel_lens = new int[batch];
    for (int b = 0; b < batch; ++b)
        h_sel_lens[b] = max_seq_len;
    cudaMemcpyAsync(state.data.select_seq_lens, h_sel_lens, sizeof(int) * batch,
                    cudaMemcpyHostToDevice, stream);
    delete[] h_sel_lens;

    int* h_sel_seqs = new int[batch * max_seq_len];
    for (int b = 0; b < batch; ++b)
        for (int t = 0; t < max_seq_len; ++t)
            h_sel_seqs[b * max_seq_len + t] = t;
    cudaMemcpyAsync(state.data.select_seqs, h_sel_seqs, sizeof(int) * batch * max_seq_len,
                    cudaMemcpyHostToDevice, stream);
    delete[] h_sel_seqs;

    cudaStreamSynchronize(stream);
}

inline cudaError_t streaming_step(void* state_buffer, const float* log_prob_frame, int batch_stride,
                                  int vocab_stride, int step, int blank_id, int space_id,
                                  cudaStream_t stream) {
    StreamingState state;
    cudaMemcpyAsync(&state, state_buffer, sizeof(StreamingState), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // log_prob_frame is [batch, vocab_size] (single frame).
    // With seq_stride=0, the kernel always accesses index 0 along the seq dim.
    if (state.use_paged_memory) {
        return ctc_prefix_beam_search_step_paged(&state.data, log_prob_frame,
                                                 batch_stride, 0, vocab_stride,
                                                 step, blank_id, space_id, stream);
    } else {
        bool is_last = false;
        return ctc_prefix_beam_search_step(&state.data, log_prob_frame, batch_stride, 0, vocab_stride,
                                           step, is_last, blank_id, space_id, stream);
    }
}

inline cudaError_t read_streaming_results(void* state_buffer, int* out_tokens, int* out_lengths,
                                          float* out_scores, int max_out_len, cudaStream_t stream) {
    StreamingState state;
    cudaMemcpyAsync(&state, state_buffer, sizeof(StreamingState), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    InternalData* data = &state.data;
    int batch = data->batch;
    int beam = data->beam;
    int step = state.current_step;
    int final_parity = (step <= 1) ? 0 : ((step - 1) % 2);

    if (state.use_paged_memory) {
        auto& ps = data->paged;
        // Use gather kernel; treat step as max_select for parity computation
        dim3 gather_grid(batch, beam);
        gather_paged_results_kernel<<<gather_grid, 256, 0, stream>>>(
            ps.page_storage, ps.block_table[0], ps.block_table[1],
            data->select_seq_lens, step,
            data->clen[0], data->clen[1],
            data->score, data->ldbeam,
            out_tokens, out_lengths, out_scores,
            batch, beam, max_out_len,
            ps.page_size, ps.max_logical_pages);
        return cudaGetLastError();
    }

    cudaMemcpy2DAsync(out_lengths, sizeof(int) * beam, data->clen[final_parity],
                      sizeof(int) * data->ldbeam, sizeof(int) * beam, batch,
                      cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(out_tokens, sizeof(int) * max_out_len, data->clist[final_parity],
                      sizeof(int) * data->ldseq_len, sizeof(int) * max_out_len, batch * beam,
                      cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(out_scores, sizeof(float) * beam, data->score, sizeof(float) * data->ldbeam,
                      sizeof(float) * beam, batch, cudaMemcpyDeviceToDevice, stream);

    return cudaGetLastError();
}

// =============================================================================
// Paged kernel: first step — write initial tokens into pre-allocated page 0
//
// Each (batch, beam) pair has physical page = bid * beam + k pre-allocated in
// block_table[0]. The initial token is written directly to page_storage.
// =============================================================================

template <int BLOCK_SIZE>
__global__ void first_step_paged_kernel(
    const float* __restrict__ log_prob, int batch_stride, int seq_stride, int vocab_stride,
    const int* __restrict__ select_seqs, const int* __restrict__ select_seq_lens,
    float2* __restrict__ pprev, int* __restrict__ clast, int* __restrict__ clen,
    float* __restrict__ score,
    int* __restrict__ page_storage, int page_size,
    int beam, int ldbeam, int vocab_size, int blank_id, int batch, int max_seq_len) {
    int bid = blockIdx.x;
    if (bid >= batch)
        return;
    if (select_seq_lens[bid] == 0)
        return;

    int first_t = select_seqs[bid * max_seq_len];
    bool need_add_blank = (first_t > 0);

    extern __shared__ char smem[];
    float* s_keys = reinterpret_cast<float*>(smem);
    int* s_vals = reinterpret_cast<int*>(s_keys + BLOCK_SIZE);

    float local_best = NEG_INF;
    int local_best_id = -1;
    for (int c = threadIdx.x; c < vocab_size; c += BLOCK_SIZE) {
        if (c == blank_id)
            continue;
        float p = log_prob[bid * batch_stride + first_t * seq_stride + c * vocab_stride];
        if (p > local_best) {
            local_best = p;
            local_best_id = c;
        }
    }
    s_keys[threadIdx.x] = local_best;
    s_vals[threadIdx.x] = local_best_id;
    __syncthreads();

    if (threadIdx.x == 0) {
        float topk_keys[128];
        int topk_vals[128];
        for (int k = 0; k < beam; ++k) {
            topk_keys[k] = NEG_INF;
            topk_vals[k] = -1;
        }
        for (int c = 0; c < vocab_size; ++c) {
            if (c == blank_id)
                continue;
            float p = log_prob[bid * batch_stride + first_t * seq_stride + c * vocab_stride];
            int min_idx = 0;
            for (int k = 1; k < beam; ++k)
                if (topk_keys[k] < topk_keys[min_idx])
                    min_idx = k;
            if (p > topk_keys[min_idx]) {
                topk_keys[min_idx] = p;
                topk_vals[min_idx] = c;
            }
        }
        for (int i = 1; i < beam; ++i) {
            float ki = topk_keys[i];
            int vi = topk_vals[i];
            int j = i - 1;
            while (j >= 0 && topk_keys[j] < ki) {
                topk_keys[j + 1] = topk_keys[j];
                topk_vals[j + 1] = topk_vals[j];
                --j;
            }
            topk_keys[j + 1] = ki;
            topk_vals[j + 1] = vi;
        }

        for (int k = 0; k < beam; ++k) {
            int base = bid * ldbeam + k;
            int token = topk_vals[k];
            float key = topk_keys[k];

            if (token >= 0 && token != blank_id) {
                float2 xy = need_add_blank ? make_float2(key, NEG_INF) : make_float2(NEG_INF, key);
                pprev[base] = xy;
                // Physical page for (bid, k) is pre-allocated as bid * beam + k.
                // Token goes to offset 0 (clen starts at 0).
                int phys = bid * beam + k;
                page_storage[phys * page_size + 0] = token;
                clen[base] = 1;
                clast[base] = token;
                score[base] = key;
            } else {
                pprev[base] = make_float2(NEG_INF, NEG_INF);
                clast[base] = blank_id;
                clen[base] = 0;
                score[base] = NEG_INF;
            }
        }
    }
}

// =============================================================================
// Paged kernel: merge duplicate prefixes
//
// Same logic as merge_kernel but reads sequences via paged_seq_compare.
// =============================================================================

__global__ void merge_paged_kernel(
    const int* __restrict__ select_seq_lens, int step,
    float* __restrict__ ptable, float* __restrict__ ptablen,
    const int* __restrict__ clast, const int* __restrict__ clen,
    int ldc, int beam, int ldbeam, int batch, int blank_id,
    const int* __restrict__ page_storage, const int* __restrict__ block_table,
    int page_size, int max_lp) {
    const int bid = blockIdx.y;
    if (bid >= batch)
        return;
    if (step >= select_seq_lens[bid])
        return;

    __shared__ int smem_clen[128];
    if (threadIdx.x < beam)
        smem_clen[threadIdx.x] = clen[threadIdx.x + bid * ldbeam];
    __syncthreads();

    int shorter_beam = blockIdx.x;
    int longer_beam = threadIdx.x;

    if (longer_beam < beam && (smem_clen[longer_beam] - 1) == smem_clen[shorter_beam]) {
        if (paged_memory::paged_seq_compare(page_storage, block_table,
                                            bid, longer_beam, shorter_beam,
                                            smem_clen[shorter_beam],
                                            beam, page_size, max_lp)) {
            int tidin = clast[longer_beam + bid * ldbeam] + (shorter_beam + bid * beam) * ldc;
            int tidout = blank_id + (longer_beam + bid * beam) * ldc;
            ptable[tidout] = logsumexp(ptable[tidout], ptable[tidin]);
            ptablen[tidout] = logsumexp(ptablen[tidout], ptablen[tidin]);
            ptable[tidin] = NEG_INF;
            ptablen[tidin] = NEG_INF;
        }
    }
}

// =============================================================================
// Paged kernel: Top-K Phase 2 with block table fork and copy-on-write
//
// Execution is split into two passes separated by __syncthreads():
//
//   Pass 1 – Fork (parallelised over WRITE_THREADS per output beam):
//     For each logical page p of the source beam, copy the physical page index
//     from block_table_src to block_table_dst, releasing (via free_page) any
//     entry that block_table_dst previously held, and acquiring (via
//     atomicAdd on ref_counts) the new entry.
//
//   __syncthreads()  — ensures all free_page pushes are visible before any pop.
//
//   Pass 2 – Append / CoW (tid_in_sub == 0 only):
//     Non-blank token: allocate or copy-on-write the last logical page with
//     alloc_page (which prefers recycled pages from free_pool).
//
// =============================================================================

template <int BLOCK_SIZE, int ITEMS_PER_THREAD, int WRITE_THREADS = 8>
__global__ __launch_bounds__(BLOCK_SIZE) void topk_phase2_paged_kernel(
    const int* __restrict__ select_seq_lens, int step,
    int items_per_batch, int beam, int batch,
    float* __restrict__ topk_key_buffer, int* __restrict__ topk_value_buffer,
    int ldc, int ldbeam,
    float2* __restrict__ pprev,
    const float* __restrict__ ptable, const float* __restrict__ ptablen,
    int* __restrict__ clast,
    int* __restrict__ clen_src, int* __restrict__ clen_dst,
    float* __restrict__ score,
    int blank_id, const int* __restrict__ select_seqs, int max_seq_len,
    int* __restrict__ page_storage,
    int* __restrict__ block_table_src, int* __restrict__ block_table_dst,
    int* __restrict__ ref_counts, int* __restrict__ next_free_page,
    int* __restrict__ free_pool, int* __restrict__ free_pool_size,
    int page_size, int max_lp) {
    const int bid = blockIdx.x;
    if (bid >= batch)
        return;
    if (step >= select_seq_lens[bid])
        return;

    bool need_add_blank = false;
    if (step == 0) {
        need_add_blank = (select_seqs[bid * max_seq_len] > 0);
    } else {
        int cur_t = select_seqs[bid * max_seq_len + step];
        int prev_t = select_seqs[bid * max_seq_len + step - 1];
        need_add_blank = (cur_t > prev_t + 1);
    }

    const int tx = threadIdx.x;
    const int rw_offset = bid * items_per_batch;

    typedef cub::BlockRadixSort<float, BLOCK_SIZE, ITEMS_PER_THREAD, int> BlockSortT;
    __shared__ union {
        typename BlockSortT::TempStorage temp_storage;
        struct {
            float keys[128];
            int vals[128];
            int src_clast[128];
            int src_clen[128];
        } topk;
    } smem;

    float keys[ITEMS_PER_THREAD];
    int values[ITEMS_PER_THREAD];

    const int items_per_iter = BLOCK_SIZE * ITEMS_PER_THREAD;

#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        int pos = BLOCK_SIZE * ITEM + tx;
        if (pos < items_per_batch) {
            keys[ITEM] = topk_key_buffer[rw_offset + pos];
            values[ITEM] = topk_value_buffer[rw_offset + pos];
        } else {
            keys[ITEM] = NEG_INF;
            values[ITEM] = pos;
        }
    }
    BlockSortT{smem.temp_storage}.SortDescendingBlockedToStriped(keys, values);
    __syncthreads();

    const int stride = items_per_iter - beam;
    for (int offset = items_per_iter; offset < items_per_batch; offset += stride) {
#pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
            int striped_pos = BLOCK_SIZE * ITEM + tx;
            int new_local = striped_pos - beam;
            int new_idx = rw_offset + offset + new_local;
            if (new_local >= 0) {
                if ((offset + new_local) < items_per_batch) {
                    keys[ITEM] = topk_key_buffer[new_idx];
                    values[ITEM] = topk_value_buffer[new_idx];
                } else {
                    keys[ITEM] = NEG_INF;
                }
            }
        }
        BlockSortT{smem.temp_storage}.SortDescendingBlockedToStriped(keys, values);
        __syncthreads();
    }

#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        int striped_pos = BLOCK_SIZE * ITEM + tx;
        if (striped_pos < beam) {
            smem.topk.keys[striped_pos] = keys[ITEM];
            smem.topk.vals[striped_pos] = values[ITEM];
        }
    }
    __syncthreads();

    for (int k = tx; k < beam; k += BLOCK_SIZE) {
        smem.topk.src_clast[k] = clast[bid * ldbeam + k];
        smem.topk.src_clen[k] = clen_src[bid * ldbeam + k];
    }
    __syncthreads();

    const int sub_warp_id = tx / WRITE_THREADS;
    const int tid_in_sub = tx % WRITE_THREADS;
    const int sub_warps = BLOCK_SIZE / WRITE_THREADS;

    // =========================================================================
    // Pass 1: Fork — copy block table entries from src to dst.
    //   For each slot that block_table_dst previously held a valid page,
    //   release that reference via free_page (which may push the page onto
    //   free_pool if ref_count drops to zero).
    //   Then acquire the new reference from block_table_src via atomicAdd.
    // =========================================================================
    for (int out_beam = sub_warp_id; out_beam < beam; out_beam += sub_warps) {
        int id = smem.topk.vals[out_beam];
        int src_beam = id / ldc;
        int prevlen = smem.topk.src_clen[src_beam];
        int num_pages_used = (prevlen > 0) ? (prevlen + page_size - 1) / page_size : 0;

        int bk_src = bid * beam + src_beam;
        int bk_dst = bid * beam + out_beam;

        for (int p = tid_in_sub; p < num_pages_used; p += WRITE_THREADS) {
            int old_phys = block_table_dst[bk_dst * max_lp + p];
            if (old_phys != paged_memory::INVALID_PAGE)
                free_page(old_phys, free_pool, free_pool_size, ref_counts);
            int phys = block_table_src[bk_src * max_lp + p];
            block_table_dst[bk_dst * max_lp + p] = phys;
            atomicAdd(&ref_counts[phys], 1);
        }
    }

    // Barrier: all free_page pushes must be complete before alloc_page pops.
    __syncthreads();

    // =========================================================================
    // Pass 2: Append / CoW — scalar per output beam (tid_in_sub == 0 only).
    //   Allocate or copy-on-write the last logical page, then write the token.
    // =========================================================================
    for (int out_beam = sub_warp_id; out_beam < beam; out_beam += sub_warps) {
        if (tid_in_sub != 0) continue;

        int id = smem.topk.vals[out_beam];
        int src_beam = id / ldc;
        int char_id = id - src_beam * ldc;
        float new_score = smem.topk.keys[out_beam];
        int prevlen = smem.topk.src_clen[src_beam];

        int bk_dst = bid * beam + out_beam;
        int dst_base = bid * ldbeam + out_beam;

        if (char_id == blank_id) {
            clast[dst_base] = smem.topk.src_clast[src_beam];
            clen_dst[dst_base] = prevlen;
        } else {
            // Non-blank: append char_id at position prevlen
            clast[dst_base] = char_id;
            clen_dst[dst_base] = prevlen + 1;

            int write_pos = prevlen;
            int last_lp = write_pos / page_size;
            int off = write_pos - last_lp * page_size;

            if (off == 0) {
                // Starting a new logical page: allocate a fresh or recycled page.
                int new_phys = alloc_page(free_pool, free_pool_size, next_free_page);
                block_table_dst[bk_dst * max_lp + last_lp] = new_phys;
                ref_counts[new_phys] = 1;
                page_storage[new_phys * page_size + 0] = char_id;
            } else {
                // Writing within an existing page: CoW if shared.
                int bt_idx = bk_dst * max_lp + last_lp;
                int phys = block_table_dst[bt_idx];
                if (ref_counts[phys] > 1) {
                    // Copy-on-write: allocate a fresh/recycled page and copy.
                    int new_phys = alloc_page(free_pool, free_pool_size, next_free_page);
                    for (int i = 0; i < page_size; ++i)
                        page_storage[new_phys * page_size + i] = page_storage[phys * page_size + i];
                    block_table_dst[bt_idx] = new_phys;
                    // Release one reference from the old page (the bt_dst slot).
                    free_page(phys, free_pool, free_pool_size, ref_counts);
                    ref_counts[new_phys] = 1;
                    phys = new_phys;
                }
                page_storage[phys * page_size + off] = char_id;
            }
        }

        score[dst_base] = new_score;

        float p = ptable[bid * ldc * beam + id];
        float pn = ptablen[bid * ldc * beam + id];
        float2 pprev_next;
        if (need_add_blank) {
            pprev_next = make_float2(new_score, NEG_INF);
        } else {
            pprev_next = make_float2(p, pn);
        }
        pprev[dst_base] = pprev_next;
    }
}

// =============================================================================
// Paged kernel: fixup parity — copy block table entries for batches whose
// last-step parity differs from the global final parity.
//
// Unlike the flat fixup which copies token data, this copies block table
// entries and increments ref counts on the referenced physical pages.
// =============================================================================

__global__ void fixup_parity_paged_kernel(
    const int* __restrict__ select_seq_lens, int max_select_seq_len,
    int* __restrict__ clen0, int* __restrict__ clen1,
    int* __restrict__ block_table0, int* __restrict__ block_table1,
    int* __restrict__ ref_counts,
    int ldbeam, int beam, int batch, int final_parity, int max_lp) {
    const int bid = blockIdx.x;
    if (bid >= batch)
        return;

    int nsteps = select_seq_lens[bid];
    int batch_parity;
    if (nsteps <= 1) {
        batch_parity = 0;
    } else {
        batch_parity = (nsteps - 1) % 2;
    }
    if (batch_parity == final_parity)
        return;

    int* src_clen = (batch_parity == 0) ? clen0 : clen1;
    int* dst_clen = (final_parity == 0) ? clen0 : clen1;
    int* src_bt = (batch_parity == 0) ? block_table0 : block_table1;
    int* dst_bt = (final_parity == 0) ? block_table0 : block_table1;

    for (int k = threadIdx.x; k < beam; k += blockDim.x) {
        int base = bid * ldbeam + k;
        int len = src_clen[base];
        dst_clen[base] = len;

        int num_pages_used = (len > 0) ? (len + 31) / 32 : 0;  // safe upper bound
        // Use actual page_size via max_lp and len
        // Recompute: num_pages = ceil(len / page_size), but we don't have page_size here.
        // Instead, iterate over all max_lp entries and check for INVALID_PAGE.
        int bk = bid * beam + k;
        for (int lp = 0; lp < max_lp; ++lp) {
            int phys = src_bt[bk * max_lp + lp];
            dst_bt[bk * max_lp + lp] = phys;
            if (phys != paged_memory::INVALID_PAGE)
                atomicAdd(&ref_counts[phys], 1);
        }
    }
}

// =============================================================================
// Paged kernel: gather results — read token sequences via block table and
// write to flat output tensor [batch, beam, max_out_len].
// =============================================================================

__global__ void gather_paged_results_kernel(
    const int* __restrict__ page_storage,
    const int* __restrict__ block_table0, const int* __restrict__ block_table1,
    const int* __restrict__ select_seq_lens, int max_select_seq_len,
    const int* __restrict__ clen0, const int* __restrict__ clen1,
    const float* __restrict__ score, int ldbeam,
    int* __restrict__ out_tokens, int* __restrict__ out_lengths,
    float* __restrict__ out_scores,
    int batch, int beam, int max_out_len,
    int page_size, int max_lp) {
    // One block per (batch, beam); threads iterate over token positions.
    const int bid = blockIdx.x;
    const int k = blockIdx.y;
    if (bid >= batch || k >= beam)
        return;

    // Determine which parity holds this batch's results.
    // In streaming mode select_seq_lens[bid] = max_seq_len (identity mapping),
    // so clamp to max_select_seq_len (= actual step count) for correct parity.
    int nsteps = select_seq_lens[bid];
    if (nsteps > max_select_seq_len)
        nsteps = max_select_seq_len;
    int batch_parity = (nsteps <= 1) ? 0 : (nsteps - 1) % 2;
    int final_parity = (max_select_seq_len <= 1) ? 0 : (max_select_seq_len - 1) % 2;
    int use_parity = (batch_parity == final_parity) ? final_parity : batch_parity;

    const int* clen = (use_parity == 0) ? clen0 : clen1;
    const int* bt = (use_parity == 0) ? block_table0 : block_table1;

    int base = bid * ldbeam + k;
    int len = clen[base];
    int copy_len = (len < max_out_len) ? len : max_out_len;

    // Write tokens
    int bk = bid * beam + k;
    for (int pos = threadIdx.x; pos < copy_len; pos += blockDim.x) {
        int lp = pos / page_size;
        int off = pos - lp * page_size;
        int phys = bt[bk * max_lp + lp];
        out_tokens[bid * beam * max_out_len + k * max_out_len + pos] =
            page_storage[phys * page_size + off];
    }

    if (threadIdx.x == 0) {
        out_lengths[bid * beam + k] = copy_len;
        out_scores[bid * beam + k] = score[base];
    }
}

// =============================================================================
// Paged step launcher — dispatches paged kernels for one beam search step
// =============================================================================

inline cudaError_t ctc_prefix_beam_search_step_paged(
    InternalData* data, const float* log_prob,
    int batch_stride, int seq_stride, int vocab_stride,
    int step, int blank_id, int space_id, cudaStream_t stream) {
    auto& ps = data->paged;
    int batch = data->batch;
    int beam = data->beam;
    int ldc = data->ldc;
    int ldbeam = data->ldbeam;
    int max_seq_len = data->max_seq_len;

    int src_parity = (step == 0) ? 0 : ((step - 1) % 2);
    int dst_parity = (step == 0) ? 0 : (step % 2);

    if (step == 0) {
        int smem_size = (sizeof(float) + sizeof(int)) * 256;
        first_step_paged_kernel<256><<<batch, 256, smem_size, stream>>>(
            log_prob, batch_stride, seq_stride, vocab_stride,
            data->select_seqs, data->select_seq_lens,
            data->pprev, data->clast, data->clen[0], data->score,
            ps.page_storage, ps.page_size,
            beam, ldbeam, data->vocab_size, blank_id, batch, max_seq_len);
    } else {
        // --- Probability matrix (same as flat) ---
        {
            int total = ldc * beam;
            int threads = 256;
            int batch_nz = (batch > 0) ? batch : 1;
            int bx = min((total + threads - 1) / threads, MAX_BLOCKS / batch_nz);
            dim3 grid(bx, batch);
            prob_matrix_kernel<<<grid, threads, 0, stream>>>(
                log_prob, batch_stride, seq_stride, vocab_stride,
                data->select_seqs, data->select_seq_lens, step,
                data->pprev, data->ptable, data->ptablen, data->clast,
                ldc, beam, ldbeam, batch, blank_id, space_id, max_seq_len);
        }

        // --- Blank / space (same as flat) ---
        prob_space_blank_kernel<<<batch, ldbeam, 0, stream>>>(
            log_prob, batch_stride, seq_stride, vocab_stride,
            data->select_seqs, data->select_seq_lens, step,
            data->pprev, data->ptable, data->ptablen, data->clast,
            ldc, beam, ldbeam, batch, blank_id, space_id, max_seq_len);

        // --- Merge duplicates (paged) ---
        {
            dim3 merge_grid(beam, batch);
            merge_paged_kernel<<<merge_grid, ldbeam, 0, stream>>>(
                data->select_seq_lens, step,
                data->ptable, data->ptablen, data->clast, data->clen[src_parity],
                ldc, beam, ldbeam, batch, blank_id,
                ps.page_storage, ps.block_table[src_parity],
                ps.page_size, ps.max_logical_pages);
        }

        // --- Top-K Phase 1 (same as flat) ---
        int bxs;
        {
            constexpr int P1_BLOCK = 128;
            constexpr int P1_IPT = 4;
            int all_items = ldc * beam;
            bxs = min(MAX_BLOCKS_PER_BATCH,
                      max(1, (all_items + P1_BLOCK * P1_IPT - 1) / (P1_BLOCK * P1_IPT)));
            bxs = min(bxs, MAX_BLOCKS / max(batch, 1));
            dim3 p1_grid(bxs, batch);
            topk_phase1_kernel<P1_BLOCK, P1_IPT><<<p1_grid, P1_BLOCK, 0, stream>>>(
                data->select_seq_lens, step, data->ptable, data->ptablen,
                ldc, beam, batch, data->topk_key_buffer, data->topk_value_buffer);
        }

        // --- Top-K Phase 2 (paged) ---
        {
            constexpr int P2_BLOCK = 128;
            constexpr int P2_IPT = 2;
            int items_per_batch = bxs * beam;
            topk_phase2_paged_kernel<P2_BLOCK, P2_IPT><<<batch, P2_BLOCK, 0, stream>>>(
                data->select_seq_lens, step, items_per_batch, beam, batch,
                data->topk_key_buffer, data->topk_value_buffer,
                ldc, ldbeam,
                data->pprev, data->ptable, data->ptablen, data->clast,
                data->clen[src_parity], data->clen[dst_parity],
                data->score,
                blank_id, data->select_seqs, max_seq_len,
                ps.page_storage,
                ps.block_table[src_parity], ps.block_table[dst_parity],
                ps.ref_counts, ps.next_free_page,
                ps.free_pool, ps.free_pool_size,
                ps.page_size, ps.max_logical_pages);
        }
    }

    return cudaGetLastError();
}

// =============================================================================
// Paged offline decode (analogous to ctc_beam_search_decode_batch)
// =============================================================================

inline cudaError_t ctc_beam_search_decode_batch_paged(
    const float* log_prob, int batch_stride, int seq_stride, int vocab_stride,
    const int* seq_lengths,
    int* out_tokens, int* out_lengths, float* out_scores,
    void* workspace, int batch, int beam, int vocab_size,
    int max_seq_len, int max_out_len,
    int blank_id, int space_id, float blank_threshold,
    int page_size, int num_pages,
    cudaStream_t stream) {
    int ws_seq_len = (max_seq_len > max_out_len) ? max_seq_len : max_out_len;

    InternalData data;
    init_internal_data_paged(&data, workspace, batch, beam, vocab_size, ws_seq_len,
                             page_size, num_pages, stream);

    // Initialise beam-state buffers (same as flat, minus clist)
    cudaMemsetAsync(data.clast, 0, sizeof(int) * batch * data.ldbeam, stream);
    cudaMemsetAsync(data.clen[0], 0, sizeof(int) * batch * data.ldbeam, stream);
    cudaMemsetAsync(data.clen[1], 0, sizeof(int) * batch * data.ldbeam, stream);
    cudaMemsetAsync(data.ptable, 0xcc, sizeof(float) * batch * beam * data.ldc, stream);
    cudaMemsetAsync(data.ptablen, 0xcc, sizeof(float) * batch * beam * data.ldc, stream);
    cudaMemsetAsync(data.select_seq_lens, 0, sizeof(int) * batch, stream);

    float log_threshold = (blank_threshold <= 0.0f)   ? NEG_INF
                          : (blank_threshold >= 1.0f) ? 0.0f
                                                      : logf(blank_threshold);

    constexpr int SEL_BLOCK = 128;
    constexpr int SEL_IPT = 4;
    init_select_kernel<SEL_BLOCK, SEL_IPT><<<batch, SEL_BLOCK, 0, stream>>>(
        log_prob, batch_stride, seq_stride, vocab_stride, seq_lengths,
        batch, data.max_seq_len, blank_id, log_threshold,
        data.select_seqs, data.select_seq_lens);

    int* h_select_lens = new int[batch];
    cudaMemcpyAsync(h_select_lens, data.select_seq_lens, sizeof(int) * batch,
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    int max_select = 0;
    for (int b = 0; b < batch; ++b)
        if (h_select_lens[b] > max_select)
            max_select = h_select_lens[b];
    delete[] h_select_lens;
    data.max_select_seq_len = max_select;

    for (int step = 0; step < max_select; ++step) {
        cudaError_t err = ctc_prefix_beam_search_step_paged(
            &data, log_prob, batch_stride, seq_stride, vocab_stride,
            step, blank_id, space_id, stream);
        if (err != cudaSuccess)
            return err;
    }

    auto& ps = data.paged;

    // Launch gather kernel to read paged sequences into flat output
    dim3 gather_grid(batch, beam);
    gather_paged_results_kernel<<<gather_grid, 256, 0, stream>>>(
        ps.page_storage, ps.block_table[0], ps.block_table[1],
        data.select_seq_lens, max_select,
        data.clen[0], data.clen[1],
        data.score, data.ldbeam,
        out_tokens, out_lengths, out_scores,
        batch, beam, max_out_len,
        ps.page_size, ps.max_logical_pages);

    return cudaGetLastError();
}

// =============================================================================
// Paged streaming state init
// =============================================================================

inline void init_streaming_state_paged(void* state_buffer, int batch, int beam,
                                       int vocab_size, int max_seq_len,
                                       int blank_id, int page_size, int num_pages,
                                       cudaStream_t stream) {
    StreamingState state;
    state.current_step = 0;
    state.space_id = -1;
    state.blank_id = blank_id;
    state.use_paged_memory = 1;

    void* workspace = reinterpret_cast<char*>(state_buffer) + STATE_HEADER_SIZE;
    init_internal_data_paged(&state.data, workspace, batch, beam, vocab_size, max_seq_len,
                             page_size, num_pages, stream);
    state.data.max_select_seq_len = max_seq_len;

    cudaMemcpyAsync(state_buffer, &state, sizeof(StreamingState), cudaMemcpyHostToDevice, stream);

    // Initialise beam-state buffers
    cudaMemsetAsync(state.data.clast, 0, sizeof(int) * batch * state.data.ldbeam, stream);
    cudaMemsetAsync(state.data.clen[0], 0, sizeof(int) * batch * state.data.ldbeam, stream);
    cudaMemsetAsync(state.data.clen[1], 0, sizeof(int) * batch * state.data.ldbeam, stream);
    cudaMemsetAsync(state.data.ptable, 0xcc, sizeof(float) * batch * beam * state.data.ldc, stream);
    cudaMemsetAsync(state.data.ptablen, 0xcc, sizeof(float) * batch * beam * state.data.ldc, stream);

    // Identity frame mapping (all frames selected in streaming mode)
    int* h_sel_lens = new int[batch];
    for (int b = 0; b < batch; ++b)
        h_sel_lens[b] = max_seq_len;
    cudaMemcpyAsync(state.data.select_seq_lens, h_sel_lens, sizeof(int) * batch,
                    cudaMemcpyHostToDevice, stream);
    delete[] h_sel_lens;

    int* h_sel_seqs = new int[batch * max_seq_len];
    for (int b = 0; b < batch; ++b)
        for (int t = 0; t < max_seq_len; ++t)
            h_sel_seqs[b * max_seq_len + t] = t;
    cudaMemcpyAsync(state.data.select_seqs, h_sel_seqs, sizeof(int) * batch * max_seq_len,
                    cudaMemcpyHostToDevice, stream);
    delete[] h_sel_seqs;

    cudaStreamSynchronize(stream);
}

}  // namespace ctc_decoder
}  // namespace oasr
