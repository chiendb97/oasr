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
#include <climits>
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

    // GPU: [num_pages * page_size] — the encoder frame each stored token was
    // emitted at, in EXACTLY page_storage's layout: same physical page index,
    // same offset, same block table, same reference counts.  A parallel array
    // rather than a wider page entry keeps every existing index computation
    // valid — a page copy copies both with the same loop, and the allocator
    // never learns that a second array exists.
    int* time_storage;

    // GPU: [batch * beam * max_logical_pages] * 2 — double-buffered block tables.
    // block_table[p][(bid * beam + k) * max_logical_pages + lp] = physical page index
    int* block_table[2];

    // GPU: [num_pages] — per-page reference count (CPU-driven during init/step)
    int* ref_counts;

    // Allocator state is PARTITIONED PER BATCH ROW: pages are never shared
    // across rows, and a shared pool would let one row's pop race another
    // row's in-flight push (free_pool_size is bumped before free_pool[slot]
    // is written), handing out stale page indices.  Row b owns physical pages
    // [b * ppr, (b+1) * ppr) with ppr = num_pages / batch.

    // GPU: [batch] — per-row bump counter (next fresh physical page of the row)
    int* next_free_page;

    // GPU: [num_pages] — per-row stacks of recycled physical page indices;
    // row b uses free_pool[b * ppr .. (b+1) * ppr)
    int* free_pool;

    // GPU: [batch] — per-row number of valid entries in the row's free_pool stack
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
          time_storage(nullptr),
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
    total += paged_align_size(sizeof(int) * num_pages * page_size);  // page_storage
    total += paged_align_size(sizeof(int) * num_pages * page_size);  // time_storage
    total += paged_align_size(sizeof(int) * batch * beam * max_lp) * 2;
    total += paged_align_size(sizeof(int) * num_pages);  // ref_counts
    total += paged_align_size(sizeof(int) * batch);      // next_free_page (per row)
    total += paged_align_size(sizeof(int) * num_pages);  // free_pool (per-row stacks)
    total += paged_align_size(sizeof(int) * batch);      // free_pool_size (per row)
    return total;
}

// Device-side initialisation of PagedSequenceState.  With ppr = pages per
// row (= num_pages / batch), a single kernel launch writes:
//   block_table[0][(b * beam + k)][lp] = (lp == 0) ? b * ppr + k : INVALID_PAGE
//   block_table[1][bk][lp] = INVALID_PAGE
//   ref_counts[i] = (i % ppr < beam && i < batch * ppr) ? 1 : 0
//   next_free_page[b] = b * ppr + beam      (absolute page index, per row)
//   free_pool_size[b] = 0
// This replaces the previous host-side new[] + cudaMemcpyAsync + sync sequence.
__global__ void init_paged_state_kernel(int* __restrict__ block_table0,
                                        int* __restrict__ block_table1,
                                        int* __restrict__ ref_counts,
                                        int* __restrict__ next_free_page,
                                        int* __restrict__ free_pool_size,
                                        int batch, int beam, int max_lp, int num_pages,
                                        int pages_per_row) {
    const int total_bt = batch * beam * max_lp;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < total_bt; i += stride) {
        int lp = i % max_lp;
        int bk = i / max_lp;
        int b = bk / beam;
        int k = bk - b * beam;
        block_table0[i] = (lp == 0) ? (b * pages_per_row + k) : INVALID_PAGE;
        block_table1[i] = INVALID_PAGE;
    }

    for (int i = tid; i < num_pages; i += stride) {
        ref_counts[i] =
            (i < batch * pages_per_row && (i % pages_per_row) < beam) ? 1 : 0;
    }

    for (int b = tid; b < batch; b += stride) {
        next_free_page[b] = b * pages_per_row + beam;
        free_pool_size[b] = 0;
    }
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
    PAGED_ALLOC(time_storage, int, num_pages * page_size)
    PAGED_ALLOC(block_table[0], int, batch * beam * max_lp)
    PAGED_ALLOC(block_table[1], int, batch * beam * max_lp)
    PAGED_ALLOC(ref_counts, int, num_pages)
    PAGED_ALLOC(next_free_page, int, batch)
    PAGED_ALLOC(free_pool, int, num_pages)
    PAGED_ALLOC(free_pool_size, int, batch)

#undef PAGED_ALLOC

    cudaMemsetAsync(ps->page_storage, 0, sizeof(int) * num_pages * page_size, stream);
    cudaMemsetAsync(ps->time_storage, 0, sizeof(int) * num_pages * page_size, stream);

    int total_work = max(batch * beam * max_lp, num_pages);
    int threads = 256;
    int blocks = min(1024, (total_work + threads - 1) / threads);
    if (blocks < 1) blocks = 1;
    init_paged_state_kernel<<<blocks, threads, 0, stream>>>(
        ps->block_table[0], ps->block_table[1], ps->ref_counts,
        ps->next_free_page, ps->free_pool_size,
        batch, beam, max_lp, num_pages, num_pages / max(batch, 1));
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
        while ((1u << p) < (unsigned int)d) {
            ++p;
        }
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
    // [batch * beam * ldseq_len] the encoder frame each token was emitted at,
    // written where clist is written and copied where clist is copied.  This is
    // what a CTC beam already knows and used to throw away: recovering it after
    // the fact means a forced-alignment DP over the whole log-prob sequence,
    // which costs an order of magnitude more than the decode it decorates and
    // is impossible for a stream that no longer holds those log-probs.
    // Null in paged mode — PagedSequenceState::time_storage takes over.
    int* ctime[2];

    // Beam state
    float2* pprev;   // [batch * ldbeam] (blank_score, nonblank_score) per beam
    float* ptable;   // [batch * beam * ldc] blank-path probability table
    float* ptablen;  // [batch * beam * ldc] nonblank-path probability table
    int* clast;      // [batch * ldbeam] last character in each beam
    int* ptid;       // [batch * ldbeam] (unused; offline reuses it as d_step scratch)
    float* score;    // [batch * ldbeam] beam scores

    // Top-K buffers (Phase 1 output: [batch * MAX_BLOCKS_PER_BATCH * beam])
    float* topk_key_buffer;
    int* topk_value_buffer;

    // Sequence selection (blank threshold filtering)
    int* select_seqs;      // [batch * max_seq_len]
    int* select_seq_lens;  // [batch]

    // Per-state log-prob frame buffer.  Captured streaming graphs (Step 4)
    // read the current frame's log-probs from this stable address; the host
    // loop refreshes it via a single ``cudaMemcpyAsync`` before each graph
    // replay so the captured kernel arg never changes across frames.
    // Size: ``batch * vocab_size`` floats — the captured graphs read all
    // ``batch`` rows in one go.
    float* d_lp_frame_buf;

    // Chunk-level vocab top-K pre-pass buffers (fused path only; null on the
    // legacy layout).  The pre-pass kernel ranks each frame's vocab once with
    // full (frames x batch) grid parallelism so the sequential fused step —
    // latency-bound at <2% SM utilisation — no longer scans the vocab on its
    // critical path.  Layout: row-major [tile_row][batch][stride] with
    // stride = fused_prepass_stride(beam) and tile_row < PREPASS_TILE.
    int* pre_chars;   // [PREPASS_TILE * batch * stride] candidate char ids
    float* pre_lp;    // [PREPASS_TILE * batch * stride] matching log-probs
    int* pre_cnt;     // [PREPASS_TILE * batch] valid candidates per row

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
// All three pointers are ROW-LOCAL: callers pass free_pool + bid * ppr,
// free_pool_size + bid, next_free_page + bid.  Only the batch row's own block
// ever touches them, so pushes and pops never race across blocks.
//
// alloc_page: prefer recycled pages from free_pool over fresh allocation.
//   Call only AFTER a __syncthreads() that follows the last free_page push so
//   the pushed pool entries are visible before any pop (free_pool_size is
//   bumped before free_pool[slot] is written; the barrier orders the pair).
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
// Kernel: identity-fill select_seqs / select_seq_lens for streaming mode.
// select_seqs[b * max_seq_len + t] = t, select_seq_lens[b] = max_seq_len.
// Replaces a host-side new[]/cudaMemcpy(H2D)/sync sequence.
// =============================================================================

__global__ void init_streaming_select_kernel(int* __restrict__ select_seqs,
                                             int* __restrict__ select_seq_lens, int batch,
                                             int max_seq_len) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int total = batch * max_seq_len;
    for (int i = tid; i < total; i += stride) {
        select_seqs[i] = i % max_seq_len;
    }
    for (int b = tid; b < batch; b += stride) {
        // Streaming has no fixed select length — every fed frame is decoded and
        // ``step`` may exceed ``max_seq_len`` (the output-token cap) on long
        // streams.  The per-step kernels guard on ``step >= select_seq_lens``,
        // so set it to INT_MAX to disable that bound; select_seqs itself is a
        // ring of width max_seq_len.  (Paged read passes the real step count as
        // max_select_seq_len, so gather still derives the right parity.)
        select_seq_lens[b] = INT_MAX;
    }
}

// =============================================================================
// Streaming counters living at the front of the state buffer
// -----------------------------------------------------------------------------
// The streaming chunk launcher used to advance ``step`` and ``actual_frame_index``
// on the host and pass them as kernel-launch scalars per frame.  That prevents
// CUDA-Graph capture from replaying a single ``streaming_step`` graph across
// multiple frames — the captured launch would bake in the capture-time value.
// We move both counters into device-resident int32 scalars at offsets 0/4 of
// the (otherwise unused) state-buffer header so kernels can read them via a
// single ``__ldg`` at block entry and the host loop can advance them with a
// trivial captureable kernel.
//
// STATE_HEADER_SIZE is large enough to hold these 8 bytes already — the
// previous comment in ``init_streaming_state`` noted the header region was
// reserved but unused.
// =============================================================================

__host__ __device__ inline int* device_step_ptr(void* state_buffer) {
    return reinterpret_cast<int*>(state_buffer);
}
__host__ __device__ inline int* device_frame_idx_ptr(void* state_buffer) {
    return reinterpret_cast<int*>(reinterpret_cast<char*>(state_buffer) + sizeof(int));
}

// Single-thread kernels: cheap and captureable. Launch <<<1, 1>>>.
__global__ inline void set_counters_kernel(int* d_step, int* d_frame_idx, int step,
                                           int frame_idx) {
    *d_step = step;
    *d_frame_idx = frame_idx;
}
__global__ inline void advance_counters_kernel(int* d_step, int* d_frame_idx) {
    *d_step += 1;
    *d_frame_idx += 1;
}
__global__ inline void advance_frame_idx_kernel(int* d_frame_idx) { *d_frame_idx += 1; }

inline cudaError_t set_stream_counters(void* state_buffer, int step, int frame_idx,
                                       cudaStream_t stream) {
    set_counters_kernel<<<1, 1, 0, stream>>>(device_step_ptr(state_buffer),
                                             device_frame_idx_ptr(state_buffer), step, frame_idx);
    return cudaGetLastError();
}
inline cudaError_t advance_stream_counters(void* state_buffer, cudaStream_t stream) {
    advance_counters_kernel<<<1, 1, 0, stream>>>(device_step_ptr(state_buffer),
                                                 device_frame_idx_ptr(state_buffer));
    return cudaGetLastError();
}
inline cudaError_t advance_stream_frame_idx(void* state_buffer, cudaStream_t stream) {
    advance_frame_idx_kernel<<<1, 1, 0, stream>>>(device_frame_idx_ptr(state_buffer));
    return cudaGetLastError();
}

// =============================================================================
// Kernel: write actual_frame_index into select_seqs[b, step] for all batches.
// Tiny kernel that replaces a host-loop of one-byte cudaMemcpyAsync per batch.
//
// Device-resident counter form: reads ``*d_step`` and ``*d_frame_idx`` so the
// host loop / graph capture doesn't need to bake either into a launch arg.
// When ``*d_step == *d_frame_idx`` the kernel is a no-op (the identity mapping
// already placed in select_seqs by ``init_streaming_select_kernel`` is correct).
// =============================================================================

__global__ void set_select_seq_step_kernel(int* __restrict__ select_seqs,
                                           const int* __restrict__ d_step,
                                           const int* __restrict__ d_frame_idx, int batch,
                                           int max_seq_len) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch)
        return;
    int step = __ldg(d_step);
    int frame_idx = __ldg(d_frame_idx);
    // ``step`` is the running count of decoded frames and is *not* bounded by
    // ``max_seq_len`` (the output-token cap) in streaming — a long stream
    // decodes more frames than it emits tokens.  ``select_seqs`` is only read
    // for the need_add_blank gap test, which compares consecutive steps, so a
    // ring of width ``max_seq_len`` is sufficient: step and step-1 never alias
    // (max_seq_len >= 2) and are written every step once any blank is skipped.
    // Offline always has step < max_seq_len, so ``% max_seq_len`` is a no-op
    // there.
    if (step != frame_idx)
        select_seqs[b * max_seq_len + step % max_seq_len] = frame_idx;
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
// Uses block-wide radix sort with streaming-reduction (same pattern as
// topk_phase1) so the whole block contributes to the top-K computation
// instead of thread 0 serially scanning the vocabulary.
// =============================================================================

template <int BLOCK_SIZE, int ITEMS_PER_THREAD>
__global__ __launch_bounds__(BLOCK_SIZE) void first_step_kernel(
    const float* __restrict__ log_prob, int batch_stride, int seq_stride, int vocab_stride,
    const int* __restrict__ select_seqs, const int* __restrict__ select_seq_lens,
    float2* __restrict__ pprev, int* __restrict__ clast, int* __restrict__ clen,
    int* __restrict__ clist, int* __restrict__ ctime, float* __restrict__ score, int beam,
    int ldbeam, int ldseq_len, int vocab_size, int blank_id, int batch, int max_seq_len) {
    const int bid = blockIdx.x;
    if (bid >= batch)
        return;
    if (select_seq_lens[bid] == 0)
        return;

    const int first_t = select_seqs[bid * max_seq_len];

    // Reserve one beam slot for the empty prefix (blank-only path) when
    // beam > 1.  This prevents spurious initial tokens in streaming mode
    // where early frames may have moderate blank probability (< threshold)
    // but no confident non-blank tokens.
    const int nb_beams = (beam > 1) ? beam - 1 : beam;
    const int tx = threadIdx.x;

    typedef cub::BlockRadixSort<float, BLOCK_SIZE, ITEMS_PER_THREAD, int> BlockSortT;
    __shared__ union {
        typename BlockSortT::TempStorage temp_storage;
        struct {
            float keys[128];  // beam <= 128
            int vals[128];
        } topk;
    } smem;

    float keys[ITEMS_PER_THREAD];
    int values[ITEMS_PER_THREAD];

    const int items_per_iter = BLOCK_SIZE * ITEMS_PER_THREAD;
    const int lp_base = bid * batch_stride + first_t * seq_stride;

    // First iteration: load first items_per_iter vocabulary entries.
#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        int c = BLOCK_SIZE * ITEM + tx;
        if (c < vocab_size && c != blank_id) {
            keys[ITEM] = log_prob[lp_base + c * vocab_stride];
            values[ITEM] = c;
        } else {
            keys[ITEM] = NEG_INF;
            values[ITEM] = -1;
        }
    }
    BlockSortT{smem.temp_storage}.SortDescendingBlockedToStriped(keys, values);
    __syncthreads();

    // Subsequent iterations: replace non-top-nb_beams positions with new entries.
    const int stride = items_per_iter - nb_beams;
    for (int offset = items_per_iter; offset < vocab_size; offset += stride) {
#pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
            int striped_pos = BLOCK_SIZE * ITEM + tx;
            int new_local = striped_pos - nb_beams;
            if (new_local >= 0) {
                int c = offset + new_local;
                if (c < vocab_size && c != blank_id) {
                    keys[ITEM] = log_prob[lp_base + c * vocab_stride];
                    values[ITEM] = c;
                } else {
                    keys[ITEM] = NEG_INF;
                    values[ITEM] = -1;
                }
            }
            // striped_pos < nb_beams → keep previous top-K value (no overwrite).
        }
        BlockSortT{smem.temp_storage}.SortDescendingBlockedToStriped(keys, values);
        __syncthreads();
    }

    // Write top nb_beams to shared memory.
#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        int striped_pos = BLOCK_SIZE * ITEM + tx;
        if (striped_pos < nb_beams) {
            smem.topk.keys[striped_pos] = keys[ITEM];
            smem.topk.vals[striped_pos] = values[ITEM];
        }
    }
    __syncthreads();

    // Parallel beam state write (threads in [0, nb_beams) cooperate).
    for (int k = tx; k < nb_beams; k += BLOCK_SIZE) {
        int base = bid * ldbeam + k;
        int token = smem.topk.vals[k];
        float key = smem.topk.keys[k];
        if (token >= 0 && token != blank_id) {
            // The prefix [token] ends in a non-blank, so its probability mass
            // belongs in the non-blank slot regardless of any leading blank
            // frames skipped before first_t (those only affect the empty/blank
            // beam below).  Putting it in the blank slot would let the next
            // identical frame extend (CTC repeat) instead of collapsing.
            pprev[base] = make_float2(NEG_INF, key);
            // clen is memset to 0 before first_step; write token at position 0 directly.
            clist[bid * beam * ldseq_len + k * ldseq_len + 0] = token;
            if (ctime)
                ctime[bid * beam * ldseq_len + k * ldseq_len + 0] = first_t;
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

    // Write the blank/empty-prefix beam to the last slot.
    if (beam > 1 && tx == 0) {
        int base = bid * ldbeam + (beam - 1);
        float blank_prob = log_prob[lp_base + blank_id * vocab_stride];
        pprev[base] = make_float2(blank_prob, NEG_INF);
        clast[base] = blank_id;
        clen[base] = 0;
        score[base] = blank_prob;
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
                                   const int* __restrict__ select_seq_lens,
                                   const int* __restrict__ d_step,
                                   float2* __restrict__ pprev, float* __restrict__ ptable,
                                   float* __restrict__ ptablen, const int* __restrict__ clast,
                                   int ldc, int beam, int ldbeam, int batch, int blank_id,
                                   int space_id, int max_seq_len) {
    const int bid = blockIdx.y;
    if (bid >= batch)
        return;
    int step = __ldg(d_step);
    if (step >= select_seq_lens[bid])
        return;

    // ``select_seqs`` is a width-``max_seq_len`` ring indexed by step (offline
    // step < max_seq_len so the modulo is a no-op; streaming may run more
    // decoded frames than the output cap — see set_select_seq_step_kernel).
    int t = select_seqs[bid * max_seq_len + step % max_seq_len];

    // When there are skipped (blank-dominant) frames between the previous
    // selected frame and this one, we must account for the blank path that
    // passes through them.  The effective pprev after one or more blank frames
    // collapses both blank and non-blank paths into the blank slot:
    //   effective_blank    = logsumexp(prev_blank, prev_nonblank)
    //   effective_nonblank = NEG_INF
    bool need_add_blank = (t > select_seqs[bid * max_seq_len + (step - 1) % max_seq_len] + 1);

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
                                        const int* __restrict__ select_seq_lens,
                                        const int* __restrict__ d_step,
                                        float2* __restrict__ pprev, float* __restrict__ ptable,
                                        float* __restrict__ ptablen, const int* __restrict__ clast,
                                        int ldc, int beam, int ldbeam, int batch, int blank_id,
                                        int space_id, int max_seq_len) {
    const int bid = blockIdx.x;
    if (bid >= batch)
        return;
    int step = __ldg(d_step);
    if (step >= select_seq_lens[bid])
        return;

    // Width-``max_seq_len`` ring (no-op modulo for offline; see
    // set_select_seq_step_kernel for the streaming rationale).
    int t = select_seqs[bid * max_seq_len + step % max_seq_len];
    int beam_idx = threadIdx.x;
    if (beam_idx >= beam)
        return;

    // Apply the same blank-frame adjustment as prob_matrix_kernel.
    bool need_add_blank = (t > select_seqs[bid * max_seq_len + (step - 1) % max_seq_len] + 1);

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

__global__ void merge_kernel(const int* __restrict__ select_seq_lens,
                             const int* __restrict__ d_step,
                             float* __restrict__ ptable, float* __restrict__ ptablen,
                             const int* __restrict__ clast, const int* __restrict__ clist,
                             const int* __restrict__ clen, int ldc, int beam, int ldbeam,
                             int ldseq_len, int batch, int blank_id) {
    const int bid = blockIdx.y;
    if (bid >= batch)
        return;
    int step = __ldg(d_step);
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
    const int* __restrict__ select_seq_lens, const int* __restrict__ d_step,
    const float* __restrict__ ptable, const float* __restrict__ ptablen, int ldc, int beam,
    int batch, float* topk_key_buffer, int* topk_value_buffer) {
    const int bid = blockIdx.y;
    if (bid >= batch)
        return;
    int step = __ldg(d_step);
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
    const int* __restrict__ select_seq_lens, const int* __restrict__ d_step,
    int items_per_batch,  // = bxs * beam
    int beam, int batch, float* __restrict__ topk_key_buffer, int* __restrict__ topk_value_buffer,
    int ldc, int ldbeam, int ldseq_len, float2* __restrict__ pprev,
    const float* __restrict__ ptable, const float* __restrict__ ptablen, int* __restrict__ clast,
    int* __restrict__ clen_src, int* __restrict__ clen_dst, int* __restrict__ clist_src,
    int* __restrict__ clist_dst, int* __restrict__ ctime_src, int* __restrict__ ctime_dst,
    float* __restrict__ score, int blank_id, const int* __restrict__ select_seqs,
    int max_seq_len, const int* __restrict__ d_frame_idx) {
    const int bid = blockIdx.x;
    if (bid >= batch)
        return;
    int step = __ldg(d_step);
    if (step >= select_seq_lens[bid])
        return;

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

        // Parallel clist copy (WRITE_THREADS threads per output beam).  The
        // emission frames ride in the same loop: the addresses are already
        // computed, so this is one more load/store per element and no extra
        // synchronisation.
        for (int s = tid_in_sub; s < prevlen; s += WRITE_THREADS) {
            clist_dst[bid * beam * ldseq_len + out_beam * ldseq_len + s] =
                clist_src[bid * beam * ldseq_len + src_beam * ldseq_len + s];
            if (ctime_dst)
                ctime_dst[bid * beam * ldseq_len + out_beam * ldseq_len + s] =
                    ctime_src[bid * beam * ldseq_len + src_beam * ldseq_len + s];
        }

        if (tid_in_sub == 0) {
            int dst_base = bid * ldbeam + out_beam;

            if (char_id == blank_id) {
                // Blank extension: keep same prefix, propagate last char.
                clast[dst_base] = smem.topk.src_clast[src_beam];
                clen_dst[dst_base] = prevlen;
            } else {
                // Non-blank extension: append new character.  Cap the length at
                // the clist capacity (ldseq_len = output-token cap) so a stream
                // that decodes more frames than it can emit tokens never lets
                // clen run past this beam's clist region — merge_kernel's
                // seq_compare and the result copy both bound their reads by clen.
                clast[dst_base] = char_id;
                if (prevlen < ldseq_len) {
                    clen_dst[dst_base] = prevlen + 1;
                    clist_dst[bid * beam * ldseq_len + out_beam * ldseq_len + prevlen] = char_id;
                    if (ctime_dst) {
                        // The **absolute** frame, not ``select_seqs[step %
                        // max_seq_len]``: that array is a ring of width
                        // max_seq_len, and a stream decodes more frames than
                        // its output-token cap, so reading it back would wrap
                        // the recorded time once the stream passes max_seq_len.
                        // Offline has no device counter and step == frame there
                        // by construction, so the ring read is exact.
                        ctime_dst[bid * beam * ldseq_len + out_beam * ldseq_len + prevlen] =
                            d_frame_idx ? __ldg(d_frame_idx)
                                        : select_seqs[bid * max_seq_len + step % max_seq_len];
                    }
                } else {
                    clen_dst[dst_base] = ldseq_len;
                }
            }

            score[dst_base] = new_score;

            // pprev for the next step is just this state's (blank, nonblank)
            // split from ptable/ptablen.  The blank-frame collapse for any
            // skipped (blank-dominant) frames *before* this step is already
            // baked into ptable/ptablen by prob_matrix_kernel /
            // prob_space_blank_kernel (they collapse the *incoming* prev to
            // {logsumexp(pb,pn), NEG_INF}).  Do NOT additionally force the
            // *outgoing* state into the blank slot on need_add_blank steps: the
            // frame just emitted ``char_id``, so when it is non-blank the prefix
            // ends in non-blank (pn carries the mass, pb == NEG_INF).  Forcing
            // {new_score, NEG_INF} mislabels a freshly emitted non-blank token
            // as "ends in blank", which lets the next identical frame extend
            // (CTC repeat) instead of collapsing — duplicating the token.  For
            // a blank winner ptablen[blank_slot] is already NEG_INF and
            // p == new_score, so (p, pn) matches the old collapsed value too.
            float p = ptable[bid * ldc * beam + id];
            float pn = ptablen[bid * ldc * beam + id];
            pprev[dst_base] = make_float2(p, pn);
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
                                    int* __restrict__ clist0, int* __restrict__ clist1,
                                    int* __restrict__ ctime0, int* __restrict__ ctime1, int ldbeam,
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
    int* src_ctime = (batch_parity == 0) ? ctime0 : ctime1;
    int* dst_ctime = (final_parity == 0) ? ctime0 : ctime1;

    for (int k = threadIdx.x; k < beam; k += blockDim.x) {
        int idx = bid * ldbeam + k;
        int len = src_clen[idx];
        dst_clen[idx] = len;
        for (int s = 0; s < len && s < ldseq_len; ++s) {
            dst_clist[bid * beam * ldseq_len + k * ldseq_len + s] =
                src_clist[bid * beam * ldseq_len + k * ldseq_len + s];
            if (dst_ctime)
                dst_ctime[bid * beam * ldseq_len + k * ldseq_len + s] =
                    src_ctime[bid * beam * ldseq_len + k * ldseq_len + s];
        }
    }
}

// =============================================================================
// Fused single-kernel beam-search step (beam <= FUSED_MAX_BEAM)
// -----------------------------------------------------------------------------
// The legacy step pipeline materialises the full [beam x vocab] ptable/ptablen
// score matrix to global memory and extracts the top-`beam` entries with a
// streaming block radix sort over all beam*vocab items (5 kernels per frame;
// topk_phase1 alone is ~75% of decoder GPU time at vocab=5000).
//
// The score matrix is separable: for an "ordinary" char c of beam k,
//     score(k, c) = lp[c] + A_k,   A_k = logsumexp(prev_blank, prev_nonblank)
// so the per-beam ranking of ordinary chars is the ranking of lp itself.  The
// only per-beam exceptions are the blank slot, the repeated-last-char slot,
// the optional space slot, and slots zeroed by the duplicate-prefix merge.
// Hence the global top-`beam` over all (k, c) pairs is contained in
//     { top-K_all chars of lp } x beams  ∪  per-beam special slots,
// with K_all = beam + 3 + max_patches  (<= 2*beam + 2), because at most
// blank/last-char/space/patched chars can displace ordinary candidates.
//
// One fused kernel per (batch row, step) therefore: snapshots beam state,
// builds the merge map, radix-selects the top-K_all lp chars, scores ~beam *
// (K_all + 3) candidates with the exact legacy formulas, selects + bitonic-
// sorts the global top-`beam`, and applies the legacy phase-2 state update
// (flat clist copy or paged fork/CoW).  ptable/ptablen are never materialised
// (they are not allocated when the fused path is active; see
// layout_has_prob_tables).
//
// Determinism: candidate keys are (score, ~id) composites, so ties resolve by
// ascending id = beam_idx * ldc + char — matching the stable order of the
// legacy radix sort.  Unlike the legacy merge_kernel, the duplicate-prefix
// fold into a beam's blank slot is accumulated in ascending source-beam order
// instead of racy cross-block read-modify-writes, and the paged fork skips
// self-forks instead of free+re-acquire (which could push a still-referenced
// page onto the free pool).  Both legacy behaviours were nondeterministic.
//
// The legacy kernels remain for beam > FUSED_MAX_BEAM and for A/B validation
// (compile with -DOASR_CTC_DISABLE_FUSED, exposed via OASR_CTC_FUSED=0).
// =============================================================================

static constexpr int FUSED_MAX_BEAM = 32;

// Frames covered by one chunk-level top-K pre-pass launch.  Both the offline
// step loop and the streaming chunk launchers tile their frame loops by this,
// so the workspace only holds PREPASS_TILE rows of candidates.
static constexpr int PREPASS_TILE = 128;

inline bool step_uses_fused(int beam) {
#ifdef OASR_CTC_DISABLE_FUSED
    (void)beam;
    return false;
#else
    return beam <= FUSED_MAX_BEAM;
#endif
}

// ptable/ptablen (and the phase-1 top-k buffers) are only needed by the
// legacy step pipeline; the fused kernel computes scores analytically.
inline bool layout_has_prob_tables(int beam) { return !step_uses_fused(beam); }

namespace fused {

// --- monotone float <-> uint32 mapping (descending float == descending uint) --

__device__ __forceinline__ uint32_t f32_sortable(float f) {
    uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ __forceinline__ float sortable_f32(uint32_t s) {
    uint32_t u = (s & 0x80000000u) ? (s & 0x7fffffffu) : ~s;
    return __uint_as_float(u);
}

// Composite sort key: score in the high 32 bits, bit-flipped id in the low 32
// bits.  Descending key order == (score desc, id asc); keys are unique per id.
__device__ __forceinline__ uint64_t make_ckey(float score, uint32_t id) {
    return (uint64_t(f32_sortable(score)) << 32) | uint64_t(0xffffffffu - id);
}

__device__ __forceinline__ uint32_t ckey_id(uint64_t k) {
    return 0xffffffffu - uint32_t(k & 0xffffffffu);
}

__device__ __forceinline__ float ckey_score(uint64_t k) {
    return sortable_f32(uint32_t(k >> 32));
}

// --- block-wide top-K selection via byte-radix refinement --------------------

template <int BLOCK_SIZE>
struct SelectScratch {
    static constexpr int WARPS = BLOCK_SIZE / 32;
    int warp_hist[WARPS][256];
    int out_n;
    int bstar, gt, eq;
};

// Selects (a superset containing exactly) the top-K keys by descending u64 key
// from n items.  Keys must be unique (callers embed the item id in the low
// bits), which guarantees termination with an exact count by the last byte
// level.  Each selected item is emitted exactly once via emit(slot, i, key);
// slot order is arbitrary.  Returns the count: min(n, K) <= count <= MAX_OUT,
// where MAX_OUT (>= K) bounds the overshoot of the final whole-bin collect.
template <int BLOCK_SIZE, int MAX_OUT, typename KeyFn, typename EmitFn>
__device__ int block_topk_select(int n, int K, KeyFn key_at, EmitFn emit,
                                 SelectScratch<BLOCK_SIZE>* s) {
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    if (tid == 0) s->out_n = 0;
    __syncthreads();

    if (n <= K) {
        for (int i = tid; i < n; i += BLOCK_SIZE)
            emit(atomicAdd(&s->out_n, 1), i, key_at(i));
        __syncthreads();
        return n;
    }

    uint64_t prefix = 0;
    int plen = 0;  // bits matched from the MSB, multiple of 8
    int collected = 0;
    int K_rem = K;

    for (int level = 0; level < 8; ++level) {
        const int shift = 56 - 8 * level;
        for (int b = tid; b < SelectScratch<BLOCK_SIZE>::WARPS * 256; b += BLOCK_SIZE)
            (&s->warp_hist[0][0])[b] = 0;
        __syncthreads();
        for (int i = tid; i < n; i += BLOCK_SIZE) {
            uint64_t k = key_at(i);
            if (plen && (k >> (64 - plen)) != prefix)
                continue;
            atomicAdd(&s->warp_hist[warp][(k >> shift) & 0xff], 1);
        }
        __syncthreads();
        // Warp 0 reduces the per-warp histograms and finds b* = the highest
        // byte bin whose suffix-cumulative count reaches K_rem (gt = items
        // strictly above b*, always < K_rem).  Each lane owns 8 bins; lane
        // suffix-sums combine via shuffles — a serial 256-bin scan here costs
        // microseconds of dependent shared-memory latency per level.
        if (tid < 32) {
            const int base = tid * 8;
            int v[8], loc[8];
            int acc = 0;
#pragma unroll
            for (int q = 7; q >= 0; --q) {
                int sum = 0;
#pragma unroll
                for (int w = 0; w < SelectScratch<BLOCK_SIZE>::WARPS; ++w)
                    sum += s->warp_hist[w][base + q];
                v[q] = sum;
                acc += sum;
                loc[q] = acc;  // suffix-sum within this lane's bins
            }
            // Inclusive suffix-sum of lane totals over the warp.
            int lane_suffix = acc;
#pragma unroll
            for (int off = 1; off < 32; off <<= 1) {
                int x = __shfl_down_sync(0xffffffffu, lane_suffix, off);
                if (tid + off < 32)
                    lane_suffix += x;
            }
            const int suffix_higher = lane_suffix - acc;  // bins of higher lanes
#pragma unroll
            for (int q = 7; q >= 0; --q) {
                int s_q = loc[q] + suffix_higher;
                int s_q1 = (q == 7) ? suffix_higher : loc[q + 1] + suffix_higher;
                if (s_q >= K_rem && s_q1 < K_rem) {  // true for exactly one (lane, q)
                    s->bstar = base + q;
                    s->gt = s_q1;
                    s->eq = v[q];
                }
            }
        }
        __syncthreads();
        const int bstar = s->bstar, gt = s->gt, eq = s->eq;
        if (collected + gt + eq <= MAX_OUT) {
            // Everything in bins >= b* fits in the output budget: collect it
            // all and stop (may exceed K; callers sort and truncate).
            for (int i = tid; i < n; i += BLOCK_SIZE) {
                uint64_t k = key_at(i);
                if (plen && (k >> (64 - plen)) != prefix)
                    continue;
                if (int((k >> shift) & 0xff) >= bstar)
                    emit(atomicAdd(&s->out_n, 1), i, k);
            }
            __syncthreads();
            collected += gt + eq;
            break;
        }
        // Refine: collect bins > b* (gt < K_rem items), recurse into bin b*.
        for (int i = tid; i < n; i += BLOCK_SIZE) {
            uint64_t k = key_at(i);
            if (plen && (k >> (64 - plen)) != prefix)
                continue;
            if (int((k >> shift) & 0xff) > bstar)
                emit(atomicAdd(&s->out_n, 1), i, k);
        }
        __syncthreads();
        collected += gt;
        K_rem -= gt;
        prefix = (prefix << 8) | uint64_t(bstar);
        plen += 8;
    }
    return collected;
}

__device__ __forceinline__ uint64_t shfl_xor_u64(uint64_t v, int mask) {
    uint32_t hi = __shfl_xor_sync(0xffffffffu, uint32_t(v >> 32), mask);
    uint32_t lo = __shfl_xor_sync(0xffffffffu, uint32_t(v), mask);
    return (uint64_t(hi) << 32) | uint64_t(lo);
}

__device__ __forceinline__ uint64_t warp_cmpx_u64(uint64_t v, int j, bool keep_max) {
    uint64_t p = shfl_xor_u64(v, j);
    return keep_max ? (v > p ? v : p) : (v < p ? v : p);
}

// Bitonic sort of 32 keys, one per lane, by warp shuffles (15 compare-exchange
// stages, no barriers).  DESC selects the overall direction.
template <bool DESC>
__device__ __forceinline__ uint64_t warp_bitonic_sort32(uint64_t v) {
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int k = 2; k <= 32; k <<= 1) {
#pragma unroll
        for (int j = k >> 1; j > 0; j >>= 1) {
            const bool asc_block = DESC ? ((lane & k) != 0) : ((lane & k) == 0);
            const bool lower = ((lane & j) == 0);
            v = warp_cmpx_u64(v, j, /*keep_max=*/asc_block != lower);
        }
    }
    return v;
}

// Sort a bitonic 32-key sequence descending (5 stages).
__device__ __forceinline__ uint64_t warp_bitonic_merge32_desc(uint64_t v) {
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int j = 16; j > 0; j >>= 1)
        v = warp_cmpx_u64(v, j, /*keep_max=*/(lane & j) == 0);
    return v;
}

// In-place bitonic sort of 64 u64 keys in shared memory, descending.  Warp 0
// holds two elements per lane in registers and exchanges via shuffles — no
// block barriers inside the 21-stage network.  All threads must call
// (trailing __syncthreads publishes the result).
template <int BLOCK_SIZE>
__device__ void bitonic_sort64_desc(uint64_t* keys) {
    const int tid = threadIdx.x;
    if (tid < 32) {
        uint64_t a = keys[tid];        // element i = tid
        uint64_t b = keys[tid + 32];   // element i = tid + 32
        for (int k = 2; k <= 64; k <<= 1) {
            for (int j = k >> 1; j > 0; j >>= 1) {
                if (j < 32) {
                    // Partner is j apart within the same register vector.
                    uint64_t pa = shfl_xor_u64(a, j);
                    bool keep_max_a = (((tid & j) == 0) == ((tid & k) == 0));
                    a = keep_max_a ? (a > pa ? a : pa) : (a < pa ? a : pa);
                    uint64_t pb = shfl_xor_u64(b, j);
                    bool keep_max_b = (((tid & j) == 0) == (((tid + 32) & k) == 0));
                    b = keep_max_b ? (b > pb ? b : pb) : (b < pb ? b : pb);
                } else {
                    // j == 32 (k == 64): partner is the other register, same
                    // lane; the final merge block is descending for all i.
                    uint64_t mx = a > b ? a : b;
                    uint64_t mn = a < b ? a : b;
                    a = mx;
                    b = mn;
                }
            }
        }
        keys[tid] = a;
        keys[tid + 32] = b;
    }
    __syncthreads();
}

// --- fused first step ---------------------------------------------------------

template <int BLOCK_SIZE>
struct FirstStepSmem {
    static constexpr int RANK_BUF = 64;
    static constexpr int VKEY_CACHE = 6144;
    uint64_t rank[RANK_BUF];
    int n_sel;
    int beam_char[FUSED_MAX_BEAM];
    float beam_lp[FUSED_MAX_BEAM];
    // Sortable lp keys cached in shared memory (vocab <= VKEY_CACHE) so the
    // multi-pass radix select re-reads smem instead of cold global memory.
    uint32_t vkeys[VKEY_CACHE];
    SelectScratch<BLOCK_SIZE> sel;
};

// Drop-in replacement for first_step_kernel / first_step_paged_kernel
// (PAGED selects the token-write target).  Semantics identical: top
// nb_beams non-blank chars of the first selected frame become beams 0..nb-1
// (ties by ascending char id, like the stable legacy sort); slot beam-1 is
// the empty/blank prefix when beam > 1.
template <int BLOCK_SIZE, bool PAGED>
__global__ __launch_bounds__(BLOCK_SIZE) void fused_first_step_kernel(
    const float* __restrict__ log_prob, int batch_stride, int seq_stride, int vocab_stride,
    const int* __restrict__ select_seqs, const int* __restrict__ select_seq_lens,
    float2* __restrict__ pprev, int* __restrict__ clast, int* __restrict__ clen,
    int* __restrict__ clist, int* __restrict__ ctime, int* __restrict__ page_storage,
    int* __restrict__ time_storage, int page_size, int pages_per_row,
    float* __restrict__ score, int beam, int ldbeam, int ldseq_len, int vocab_size,
    int blank_id, int batch, int max_seq_len) {
    __shared__ FirstStepSmem<BLOCK_SIZE> s;
    const int bid = blockIdx.x;
    if (bid >= batch)
        return;
    if (select_seq_lens[bid] == 0)
        return;
    const int tid = threadIdx.x;
    const int first_t = select_seqs[bid * max_seq_len];
    const float* lp_row =
        log_prob + (size_t)bid * batch_stride + (size_t)first_t * seq_stride;
    const int nb_beams = (beam > 1) ? beam - 1 : beam;

    const bool use_vcache = (vocab_size <= FirstStepSmem<BLOCK_SIZE>::VKEY_CACHE);
    if (use_vcache) {
        for (int c = tid; c < vocab_size; c += BLOCK_SIZE)
            s.vkeys[c] = f32_sortable(lp_row[(size_t)c * vocab_stride]);
        __syncthreads();
    }

    // Top-(nb_beams + 1) so a high-probability blank can be discarded
    // post-sort while still leaving nb_beams non-blank chars.
    {
        auto key_at = [&](int c) -> uint64_t {
            uint32_t sk = use_vcache ? s.vkeys[c]
                                     : f32_sortable(lp_row[(size_t)c * vocab_stride]);
            return (uint64_t(sk) << 32) | uint64_t(0xffffffffu - uint32_t(c));
        };
        auto emit = [&](int slot, int /*c*/, uint64_t k) {
            if (slot < FirstStepSmem<BLOCK_SIZE>::RANK_BUF)
                s.rank[slot] = k;
        };
        int cnt = block_topk_select<BLOCK_SIZE, FirstStepSmem<BLOCK_SIZE>::RANK_BUF>(
            vocab_size, nb_beams + 1, key_at, emit, &s.sel);
        if (tid == 0)
            s.n_sel = cnt;
    }
    __syncthreads();
    for (int i = tid; i < FirstStepSmem<BLOCK_SIZE>::RANK_BUF; i += BLOCK_SIZE)
        if (i >= s.n_sel)
            s.rank[i] = 0;  // sorts last
    __syncthreads();
    bitonic_sort64_desc<BLOCK_SIZE>(s.rank);

    if (tid == 0) {
        int out = 0;
        for (int r = 0; r < s.n_sel && out < nb_beams; ++r) {
            int c = (int)ckey_id(s.rank[r]);
            if (c == blank_id)
                continue;
            s.beam_char[out] = c;
            s.beam_lp[out] = ckey_score(s.rank[r]);
            ++out;
        }
        for (; out < nb_beams; ++out)
            s.beam_char[out] = -1;
    }
    __syncthreads();

    for (int k = tid; k < nb_beams; k += BLOCK_SIZE) {
        int base = bid * ldbeam + k;
        int token = s.beam_char[k];
        if (token >= 0) {
            // Prefix [token] ends in a non-blank: mass goes in the non-blank
            // slot (see the legacy first_step_kernel for the full rationale).
            pprev[base] = make_float2(NEG_INF, s.beam_lp[k]);
            if (PAGED) {
                page_storage[(size_t)(bid * pages_per_row + k) * page_size] = token;
                time_storage[(size_t)(bid * pages_per_row + k) * page_size] = first_t;
            } else {
                clist[(size_t)(bid * beam + k) * ldseq_len] = token;
                if (ctime)
                    ctime[(size_t)(bid * beam + k) * ldseq_len] = first_t;
            }
            clen[base] = 1;
            clast[base] = token;
            score[base] = s.beam_lp[k];
        } else {
            pprev[base] = make_float2(NEG_INF, NEG_INF);
            clast[base] = blank_id;
            clen[base] = 0;
            score[base] = NEG_INF;
        }
    }

    if (beam > 1 && tid == 0) {
        int base = bid * ldbeam + (beam - 1);
        float blank_prob = lp_row[(size_t)blank_id * vocab_stride];
        pprev[base] = make_float2(blank_prob, NEG_INF);
        clast[base] = blank_id;
        clen[base] = 0;
        score[base] = blank_prob;
    }
}

// --- fused beam-search step ----------------------------------------------------

template <int BLOCK_SIZE, int BEAM_CAP>
struct FusedStepSmem {
    static constexpr int K_ALL_CAP = 2 * BEAM_CAP + 2;
    static constexpr int C_BUF = K_ALL_CAP + 8;  // slack lets the select stop early
    static constexpr int CAND_CAP = BEAM_CAP * (C_BUF + 3);
    static constexpr int RANK_BUF = 64;
    // Sortable lp keys cached in shared memory (vocab <= VKEY_CACHE) so the
    // multi-pass radix select re-reads smem instead of cold global memory.
    // The 32-beam variant shrinks the cache to stay under the 48 KB static
    // shared-memory limit.
    static constexpr int VKEY_CACHE = (BEAM_CAP <= 16) ? 6144 : 2048;

    // Per-beam state snapshot (after the blank-frame adjustment)
    float2 prev[BEAM_CAP];
    float A[BEAM_CAP];  // logsumexp(prev.x, prev.y)
    int s_clast[BEAM_CAP];
    int s_clen[BEAM_CAP];
    float lp_clast[BEAM_CAP];  // log_prob[t, clast[k]]
    // Duplicate-prefix merge map
    int patch_chars[BEAM_CAP][BEAM_CAP];  // chars zeroed in row k by a merge
    int patch_cnt[BEAM_CAP];
    // Bit i set = shorter row i folds into row k's blank slot; iterating set
    // bits ascending reproduces the (previously sorted) ascending source-beam
    // fold order with no sort pass.  BEAM_CAP <= 32 fits one word.
    unsigned int merge_mask[BEAM_CAP];
    // Shared vocab candidates (top-K_all lp chars of this frame)
    int c_chars[C_BUF];
    float c_lp[C_BUF];
    int c_n;
    // Scored candidates and final ranking.  ckeys doubles as the scatter
    // destination of the rank-by-count ordering (slots [0, RANK_BUF) are dead
    // as candidates by then).
    uint64_t ckeys[CAND_CAP];
    uint64_t rank[RANK_BUF];
    int rank_cnt[BLOCK_SIZE / RANK_BUF][RANK_BUF];  // per-part greater-counts
    // Per-warp streaming top-K survivors (Phase 5): warp w stashes its local
    // top-beam at wtop[w * beam ..); the rank-by-count merge orders the union.
    uint64_t wtop[(BLOCK_SIZE / 32) * BEAM_CAP];
    // Paged CoW releases deferred past the last alloc_page pop (see Phase 6)
    int defer_free[BEAM_CAP];
    int defer_n;
    // Broadcast scalars
    float lp_blank, lp_space;
    int k_all;
    uint32_t vkeys[VKEY_CACHE];
    SelectScratch<BLOCK_SIZE> sel;
};

// Order the RANK_BUF keys in ``s.rank`` descending into ``s.ckeys[0..63]`` by
// rank-by-counting: every key's position is the number of strictly greater
// keys (keys are unique, so positions are a bijection — identical output to
// a full sort, but one all-warp pass instead of a warp-0-only 21-stage
// bitonic network).  All threads must call; trailing barrier publishes.
template <int BLOCK_SIZE, int BEAM_CAP>
__device__ __forceinline__ void rank64_desc(FusedStepSmem<BLOCK_SIZE, BEAM_CAP>& s) {
    using Smem = FusedStepSmem<BLOCK_SIZE, BEAM_CAP>;
    constexpr int PARTS = BLOCK_SIZE / Smem::RANK_BUF;
    constexpr int SPAN = Smem::RANK_BUF / PARTS;
    static_assert(BLOCK_SIZE % Smem::RANK_BUF == 0, "rank64_desc thread partition");
    const int tid = threadIdx.x;
    const int i = tid % Smem::RANK_BUF;
    const int part = tid / Smem::RANK_BUF;
    const uint64_t ki = s.rank[i];
    int cnt = 0;
#pragma unroll
    for (int j = part * SPAN; j < (part + 1) * SPAN; ++j)
        cnt += (s.rank[j] > ki) ? 1 : 0;
    s.rank_cnt[part][i] = cnt;
    __syncthreads();
    if (tid < Smem::RANK_BUF) {
        int pos = 0;
#pragma unroll
        for (int p = 0; p < PARTS; ++p)
            pos += s.rank_cnt[p][tid];
        s.ckeys[pos] = s.rank[tid];
    }
    __syncthreads();
}

// One block per batch row; replaces prob_matrix + prob_space_blank + merge +
// topk_phase1 + topk_phase2 (and their paged variants).  Score formulas and
// the state-update logic mirror the legacy kernels exactly — see the section
// comment above for the candidate-set argument and the deterministic-ordering
// differences.
//
// Pre-pass mode: when ``pre_cnt_row >= 0`` the Phase-3 vocab select is
// replaced by a load of the frame's precomputed top-K candidates from
// ``pre_chars_row`` / ``pre_lp_row`` (see fused_topk_prepass_kernel).  The
// precomputed list is a superset of the in-kernel selection (K = 2*beam + 2
// >= any reachable k_all), and every scored candidate is an exact slot of the
// conceptual score matrix, so the global top-beam — and hence the decode — is
// unchanged.  ``pre_cnt_row < 0`` keeps the in-kernel vocab select.
//
// This body is shared by ``fused_step_kernel`` (one frame per launch) and
// ``fused_chunk_kernel`` (in-kernel frame loop).  Callers resolve the
// double-buffer parity (clen/clist/block_table src vs dst) and row-localize
// the paged allocator views; the body ends without a trailing barrier (the
// chunk loop inserts one between frames).
template <int BLOCK_SIZE, int BEAM_CAP, bool PAGED>
__device__ __forceinline__ void fused_step_body(
    FusedStepSmem<BLOCK_SIZE, BEAM_CAP>& s, int bid, const float* __restrict__ lp_row,
    int vocab_stride, bool need_add_blank, int pre_cnt_row,
    const int* __restrict__ pre_chars_row, const float* __restrict__ pre_lp_row,
    float2* __restrict__ pprev, int* __restrict__ clast, const int* __restrict__ clen_src,
    int* __restrict__ clen_dst, const int* __restrict__ clist_src, int* __restrict__ clist_dst,
    const int* __restrict__ ctime_src, int* __restrict__ ctime_dst, int frame,
    float* __restrict__ score, int ldc, int beam, int ldbeam, int ldseq_len, int blank_id,
    int space_id, int* __restrict__ page_storage, int* __restrict__ time_storage,
    const int* __restrict__ block_table_src,
    int* __restrict__ block_table_dst, int* __restrict__ ref_counts,
    int* __restrict__ next_free_page, int* __restrict__ free_pool,
    int* __restrict__ free_pool_size, int page_size, int max_lp) {
    using Smem = FusedStepSmem<BLOCK_SIZE, BEAM_CAP>;
    const int tid = threadIdx.x;

#ifdef OASR_CTC_PHASE_PROF
    // TEMPORARY phase-latency instrumentation (build with
    // -DOASR_CTC_PHASE_PROF); prints per-frame phase cycles from tid 0.
    unsigned long long prof_t = 0, prof_acc[8] = {0};
    if (tid == 0) prof_t = clock64();
#define OASR_PROF_T(slot)                       \
    __syncthreads();                            \
    if (tid == 0) {                             \
        unsigned long long t_ = clock64();      \
        prof_acc[slot] += t_ - prof_t;          \
        prof_t = t_;                            \
    }
#else
#define OASR_PROF_T(slot)
#endif

    // --- Phase 1: snapshot per-beam state (with blank-frame adjustment) -----
    for (int k = tid; k < beam; k += BLOCK_SIZE) {
        float2 raw = pprev[bid * ldbeam + k];
        float2 prev =
            need_add_blank ? make_float2(logsumexp(raw.x, raw.y), NEG_INF) : raw;
        s.prev[k] = prev;
        s.A[k] = logsumexp(prev.x, prev.y);
        int lc = clast[bid * ldbeam + k];
        s.s_clast[k] = lc;
        s.s_clen[k] = clen_src[bid * ldbeam + k];
        s.lp_clast[k] = lp_row[(size_t)lc * vocab_stride];
        s.patch_cnt[k] = 0;
        s.merge_mask[k] = 0u;
    }
    // Pre-pass candidate load rides in the same pass: it is independent of
    // beam state, so its global-memory latency hides behind the snapshot
    // reads and the former standalone Phase-3 barrier disappears.
    if (pre_cnt_row >= 0) {
        const int cn = min(pre_cnt_row, Smem::C_BUF);
        for (int i = tid; i < cn; i += BLOCK_SIZE) {
            s.c_chars[i] = pre_chars_row[i];
            s.c_lp[i] = pre_lp_row[i];
        }
    }
    if (tid == 0) {
        s.lp_blank = lp_row[(size_t)blank_id * vocab_stride];
        s.lp_space = (space_id >= 0) ? lp_row[(size_t)space_id * vocab_stride] : NEG_INF;
        s.defer_n = 0;
    }
    __syncthreads();
    OASR_PROF_T(0)

    // --- Phase 2: merge map (same pair condition as legacy merge_kernel) ----
    for (int p = tid; p < beam * beam; p += BLOCK_SIZE) {
        const int i = p / beam;        // shorter beam
        const int j = p - i * beam;    // longer beam (j = i + clast[j])
        if (s.s_clen[j] - 1 != s.s_clen[i])
            continue;
        bool eq;
        if (PAGED) {
            eq = paged_memory::paged_seq_compare(page_storage, block_table_src, bid, j, i,
                                                 s.s_clen[i], beam, page_size, max_lp);
        } else {
            eq = seq_compare(s.s_clen[i],
                             clist_src + (size_t)(bid * beam + j) * ldseq_len,
                             clist_src + (size_t)(bid * beam + i) * ldseq_len);
        }
        if (!eq)
            continue;
        int pp = atomicAdd(&s.patch_cnt[i], 1);
        s.patch_chars[i][pp] = s.s_clast[j];
        // Bitmask instead of an (atomically appended, then sorted) source
        // list: iterating set bits ascending IS the ascending-source fold
        // order, so the sort pass and its barrier disappear.
        atomicOr(&s.merge_mask[j], 1u << i);
    }
    __syncthreads();
    OASR_PROF_T(1)

    // --- Phase 3 (in-kernel fallback only): vocab top-K_all ------------------
    // Pre-pass mode skips this entirely (candidates already in smem, and
    // k_all is only consumed by this select).
    if (pre_cnt_row < 0) {
        if (tid == 0) {
            int mp = 0;
            for (int k = 0; k < beam; ++k)
                mp = max(mp, s.patch_cnt[k]);
            // beam ordinaries + blank + last-char + space, displaced by patches.
            s.k_all = min(beam + 3 + mp, Smem::K_ALL_CAP);
        }
        const bool use_vcache = (ldc <= Smem::VKEY_CACHE);
        if (use_vcache) {
            for (int c = tid; c < ldc; c += BLOCK_SIZE)
                s.vkeys[c] = f32_sortable(lp_row[(size_t)c * vocab_stride]);
        }
        __syncthreads();  // publishes k_all + the vkey cache
        auto key_at = [&](int c) -> uint64_t {
            uint32_t sk = use_vcache ? s.vkeys[c]
                                     : f32_sortable(lp_row[(size_t)c * vocab_stride]);
            return (uint64_t(sk) << 32) | uint64_t(0xffffffffu - uint32_t(c));
        };
        auto emit = [&](int slot, int c, uint64_t k) {
            if (slot < Smem::C_BUF) {
                s.c_chars[slot] = c;
                s.c_lp[slot] = ckey_score(k);
            }
        };
        int cnt = block_topk_select<BLOCK_SIZE, Smem::C_BUF>(ldc, s.k_all, key_at, emit,
                                                             &s.sel);
        if (tid == 0)
            s.c_n = cnt;
        __syncthreads();
    }

    // Candidate value helpers — these reproduce the legacy ptable/ptablen
    // entries bit-exactly (for every non-blank slot ptable == NEG_INF, so the
    // legacy key logsumexp(ptable, ptablen) equals ptablen).
    auto is_patched = [&](int k, int c) -> bool {
        int pc = s.patch_cnt[k];
        for (int a = 0; a < pc; ++a)
            if (s.patch_chars[k][a] == c)
                return true;
        return false;
    };
    // ptablen value of non-blank slot (k, c).
    auto nonblank_slot_pn = [&](int k, int c, float lp_c) -> float {
        if (is_patched(k, c))
            return NEG_INF;
        if (space_id >= 0 && c == space_id)
            return logprob_add(lp_c, s.A[k]);  // space ignores the same-char rule
        if (c == s.s_clast[k])
            return logprob_add(lp_c, s.prev[k].x);  // repeat needs a blank path
        return logprob_add(lp_c, s.A[k]);
    };
    // (ptable, ptablen) of the blank slot of row k, including the merge folds
    // (the legacy ptable fold adds only NEG_INF terms — an exact no-op).
    auto blank_slot = [&](int k, float* out_p) -> float {
        float p = logprob_add(s.lp_blank, s.A[k]);
        float pn = (need_add_blank || s.s_clast[k] == blank_id)
                       ? NEG_INF
                       : logprob_add(s.lp_clast[k], s.prev[k].y);
        unsigned int m = s.merge_mask[k];
        while (m) {
            const int i = __ffs(m) - 1;  // ascending source-beam fold order
            m &= m - 1;
            // Value of slot (i, clast[k]) as written by the legacy prob
            // kernels, before any patching.
            float contrib;
            if (space_id >= 0 && s.s_clast[k] == space_id)
                contrib = logprob_add(s.lp_space, s.A[i]);
            else if (s.s_clast[i] == s.s_clast[k])
                contrib = logprob_add(s.lp_clast[k], s.prev[i].x);
            else
                contrib = logprob_add(s.lp_clast[k], s.A[i]);
            pn = logsumexp(pn, contrib);
        }
        *out_p = p;
        return pn;
    };

    // --- Phase 4: enumerate + score candidates -------------------------------
    // Every slot e writes exactly one key: real candidates their composite
    // (score, ~id) key, skipped slots (blank/last-char/space duplicates) the
    // unique low value ``e`` — all real keys are >= 2^54 (any float score's
    // sortable form is >= 0x007fffff), so dummies sort strictly below every
    // real candidate and can never displace one from the top beam.  Fixed
    // slots replace the previous atomicAdd compaction, whose serialized
    // same-address increments dominated this phase.
    const int c_n = (pre_cnt_row >= 0) ? min(pre_cnt_row, Smem::C_BUF) : s.c_n;
    const int per_row = c_n + 3;
    const int n_cand = beam * per_row;  // <= CAND_CAP by construction
    for (int e = tid; e < n_cand; e += BLOCK_SIZE) {
        const int k = e / per_row;
        const int j = e - k * per_row;
        float key_val;
        int c;
        if (j < c_n) {
            c = s.c_chars[j];
            if (c == blank_id) {  // blank slot handled at j == c_n
                s.ckeys[e] = (uint64_t)e;
                continue;
            }
            key_val = nonblank_slot_pn(k, c, s.c_lp[j]);
        } else if (j == c_n) {
            c = blank_id;
            float p;
            float pn = blank_slot(k, &p);
            key_val = logsumexp(p, pn);
        } else if (j == c_n + 1) {
            // Last-char slot, unless already covered by the shared list.
            c = s.s_clast[k];
            bool in_c = (c == blank_id);
            for (int a = 0; !in_c && a < c_n; ++a)
                in_c = (s.c_chars[a] == c);
            if (in_c) {
                s.ckeys[e] = (uint64_t)e;
                continue;
            }
            key_val = nonblank_slot_pn(k, c, s.lp_clast[k]);
        } else {
            // Space slot, unless covered by the shared list or the last-char slot.
            c = space_id;
            bool in_c = (space_id < 0 || space_id == blank_id || c == s.s_clast[k]);
            for (int a = 0; !in_c && a < c_n; ++a)
                in_c = (s.c_chars[a] == c);
            if (in_c) {
                s.ckeys[e] = (uint64_t)e;
                continue;
            }
            key_val = nonblank_slot_pn(k, c, s.lp_space);
        }
        s.ckeys[e] = make_ckey(key_val, (uint32_t)(k * ldc + c));
    }
    __syncthreads();
    OASR_PROF_T(2)

    // --- Phase 5: global top-beam select + rank ------------------------------
    // Per-warp streaming top-32 over a contiguous candidate slice — register
    // bitonic networks, no block barriers — then one rank-by-count merge of
    // the per-warp top-beam survivors.  Two barriers replace the multi-level
    // byte-radix select (~4 barriers + a warp-0 histogram reduce per level)
    // that dominated this phase.  Exactness: every global top-beam key is in
    // its warp's top-beam (keys are unique, the order is total), and out-of-
    // range lanes load unique low pad keys (< 2^32, below every real key).
    {
        constexpr int WARPS = BLOCK_SIZE / 32;
        const int warp = tid >> 5;
        const int lane = tid & 31;
        const int span = (n_cand + WARPS - 1) / WARPS;
        const int begin = warp * span;
        const int end = min(begin + span, n_cand);
        auto load = [&](int idx) -> uint64_t {
            return (idx < end) ? s.ckeys[idx] : (uint64_t)(Smem::CAND_CAP + idx);
        };
        // cur: warp-local running top-32, sorted descending across lanes.
        uint64_t cur = warp_bitonic_sort32<true>(load(begin + lane));
        for (int base = begin + 32; base < end; base += 32) {
            // Classic bitonic top-K stream: sort the new batch ascending;
            // the elementwise max against the descending top-32 keeps the 32
            // largest of the union and is bitonic — a 5-stage merge re-sorts.
            uint64_t nv = warp_bitonic_sort32<false>(load(base + lane));
            cur = warp_bitonic_merge32_desc(cur > nv ? cur : nv);
        }
        if (lane < beam)
            s.wtop[warp * beam + lane] = cur;
        __syncthreads();
        OASR_PROF_T(3)
        // Rank the m = WARPS * beam survivors by counting strictly greater
        // keys (unique keys -> exact descending positions); the top RANK_BUF
        // land in s.rank, of which Phase 6 reads [0, beam).
        const int m = WARPS * beam;
        if (tid < m) {
            const uint64_t ki = s.wtop[tid];
            int pos = 0;
            for (int j = 0; j < m; ++j)
                pos += (s.wtop[j] > ki) ? 1 : 0;
            if (pos < Smem::RANK_BUF)
                s.rank[pos] = ki;
        }
    }
    __syncthreads();
    OASR_PROF_T(4)

    // --- Phase 6: state update (legacy topk_phase2 semantics) ----------------
    // beam <= BLOCK_SIZE / WRITE_THREADS, so every sub-warp owns at most ONE
    // output beam: the winner's identity is resolved once up front and the
    // scattered lp_row[char_id] load (needed only for the final pprev
    // recompute) is issued before the paged fork pass, hiding its latency
    // behind the fork's global RMW chains.  Straight-line (loop-free) form so
    // every warp lane reaches the CoW shuffles below.
    constexpr int WRITE_THREADS = 8;
    static_assert(BEAM_CAP <= BLOCK_SIZE / WRITE_THREADS, "one output beam per sub-warp");
    const int sub = tid / WRITE_THREADS;
    const int lane = tid % WRITE_THREADS;

    const int out_beam = sub;
    const bool own = (out_beam < beam);
    uint64_t kk = 0;
    int src_beam = 0, char_id = 0, prevlen = 0;
    float lp_c_pref = 0.f;
    if (own) {
        kk = s.rank[out_beam];
        const int id = (int)ckey_id(kk);
        src_beam = id / ldc;
        char_id = id - src_beam * ldc;
        prevlen = s.s_clen[src_beam];
        if (lane == 0 && char_id != blank_id)
            lp_c_pref = lp_row[(size_t)char_id * vocab_stride];
    }

    if (PAGED) {
        // Pass 1: fork block tables (release old dst refs, acquire src refs).
        if (own) {
            int n_pages = (prevlen > 0) ? (prevlen + page_size - 1) / page_size : 0;
            int bk_src = bid * beam + src_beam;
            int bk_dst = bid * beam + out_beam;
            for (int p = lane; p < n_pages; p += WRITE_THREADS) {
                int phys = block_table_src[bk_src * max_lp + p];
                int old_phys = block_table_dst[bk_dst * max_lp + p];
                if (old_phys == phys)
                    continue;  // self-fork: net refcount no-op (legacy
                               // free+re-acquire could push a live page)
                if (old_phys != paged_memory::INVALID_PAGE)
                    free_page(old_phys, free_pool, free_pool_size, ref_counts);
                block_table_dst[bk_dst * max_lp + p] = phys;
                atomicAdd(&ref_counts[phys], 1);
            }
        }
        // All free_page pushes must be visible before any alloc_page pop.
        __syncthreads();
    }

    // Pass 2: append / CoW + state writes.  A needed copy-on-write is split:
    // lane 0 takes the decision and allocates, all WRITE_THREADS lanes of the
    // sub-warp copy the page in parallel (the former lane-0 serial copy was a
    // page_size-long dependent global load/store chain), then lane 0 finishes
    // in the original order (block-table write, release, token write).
    int cow_old = -1, cow_new = -1, cow_bt = -1, cow_off = 0;
    const int dst = bid * ldbeam + out_beam;
    if (own && !PAGED) {
        // Parallel clist copy (WRITE_THREADS threads per output beam).  The
        // emission frames ride along in the same loop over the same addresses:
        // one more load/store per element, no extra synchronisation, and ~8 KiB
        // per frame at beam 10 — under a microsecond of DRAM traffic across a
        // whole utterance.
        for (int q = lane; q < prevlen; q += WRITE_THREADS) {
            clist_dst[(size_t)(bid * beam + out_beam) * ldseq_len + q] =
                clist_src[(size_t)(bid * beam + src_beam) * ldseq_len + q];
            if (ctime_dst)
                ctime_dst[(size_t)(bid * beam + out_beam) * ldseq_len + q] =
                    ctime_src[(size_t)(bid * beam + src_beam) * ldseq_len + q];
        }
    }
    if (own && lane == 0) {
        if (PAGED) {
            int bk_dst = bid * beam + out_beam;
            const int out_cap = max_lp * page_size;
            if (char_id == blank_id) {
                clast[dst] = s.s_clast[src_beam];
                clen_dst[dst] = prevlen;
            } else if (prevlen >= out_cap) {
                // Output cap reached: keep the prefix, stop appending tokens.
                clast[dst] = char_id;
                clen_dst[dst] = out_cap;
            } else {
                clast[dst] = char_id;
                clen_dst[dst] = prevlen + 1;
                int write_pos = prevlen;
                int last_lp = write_pos / page_size;
                int off = write_pos - last_lp * page_size;
                if (off == 0) {
                    int new_phys = alloc_page(free_pool, free_pool_size, next_free_page);
                    block_table_dst[bk_dst * max_lp + last_lp] = new_phys;
                    ref_counts[new_phys] = 1;
                    page_storage[new_phys * page_size + 0] = char_id;
                    time_storage[new_phys * page_size + 0] = frame;
                } else {
                    int bt_idx = bk_dst * max_lp + last_lp;
                    int phys = block_table_dst[bt_idx];
                    if (ref_counts[phys] > 1) {
                        // Copy-on-write: the last page is shared.  Allocate
                        // here; the copy + handover run below on the whole
                        // sub-warp.  (The pool push for the released
                        // reference stays DEFERRED past the last alloc_page
                        // pop of this step — a concurrent pop could otherwise
                        // read the reserved free_pool slot before this thread
                        // writes it.)
                        cow_new = alloc_page(free_pool, free_pool_size, next_free_page);
                        cow_old = phys;
                        cow_bt = bt_idx;
                        cow_off = off;
                    } else {
                        page_storage[phys * page_size + off] = char_id;
                        time_storage[phys * page_size + off] = frame;
                    }
                }
            }
        } else {
            if (char_id == blank_id) {
                // Blank extension: keep same prefix, propagate last char.
                clast[dst] = s.s_clast[src_beam];
                clen_dst[dst] = prevlen;
            } else {
                // Non-blank extension; cap clen at the clist capacity (see the
                // legacy topk_phase2_kernel for the streaming rationale).
                clast[dst] = char_id;
                if (prevlen < ldseq_len) {
                    clen_dst[dst] = prevlen + 1;
                    clist_dst[(size_t)(bid * beam + out_beam) * ldseq_len + prevlen] =
                        char_id;
                    if (ctime_dst)
                        ctime_dst[(size_t)(bid * beam + out_beam) * ldseq_len + prevlen] =
                            frame;
                } else {
                    clen_dst[dst] = ldseq_len;
                }
            }
        }
    }
    if (PAGED) {
        // Every warp lane executes the segment shuffles (sub-warps that own
        // no beam or need no CoW broadcast -1 and skip the copy).
        cow_old = __shfl_sync(0xffffffffu, cow_old, 0, WRITE_THREADS);
        cow_new = __shfl_sync(0xffffffffu, cow_new, 0, WRITE_THREADS);
        if (cow_new >= 0) {
            // Both arrays share the physical page index, so one loop forks the
            // page and its frames together.
            for (int q = lane; q < page_size; q += WRITE_THREADS) {
                page_storage[cow_new * page_size + q] =
                    page_storage[cow_old * page_size + q];
                time_storage[cow_new * page_size + q] =
                    time_storage[cow_old * page_size + q];
            }
        }
        __syncwarp();
        if (own && lane == 0 && cow_new >= 0) {
            block_table_dst[cow_bt] = cow_new;
            if (atomicSub(&ref_counts[cow_old], 1) == 1) {
                int dslot = atomicAdd(&s.defer_n, 1);
                s.defer_free[dslot] = cow_old;
            }
            ref_counts[cow_new] = 1;
            page_storage[cow_new * page_size + cow_off] = char_id;
            time_storage[cow_new * page_size + cow_off] = frame;
        }
    }

    if (own && lane == 0) {
        score[dst] = ckey_score(kk);

        // pprev for the next step is this slot's (ptable, ptablen) split,
        // recomputed analytically (identical to the legacy table read).
        float2 ppn;
        if (char_id == blank_id) {
            float p;
            float pn = blank_slot(src_beam, &p);
            ppn = make_float2(p, pn);
        } else {
            ppn = make_float2(NEG_INF, nonblank_slot_pn(src_beam, char_id, lp_c_pref));
        }
        pprev[dst] = ppn;
    }

    if (PAGED) {
        // Flush the deferred CoW releases now that no alloc_page pop can run.
        __syncthreads();
        for (int i = tid; i < s.defer_n; i += BLOCK_SIZE) {
            int slot = atomicAdd(free_pool_size, 1);
            free_pool[slot] = s.defer_free[i];
        }
    }
    OASR_PROF_T(5)
#ifdef OASR_CTC_PHASE_PROF
    if (tid == 0)
        printf("PROF p1=%llu p2=%llu p4=%llu sel=%llu rank=%llu p6=%llu\n", prof_acc[0],
               prof_acc[1], prof_acc[2], prof_acc[3], prof_acc[4], prof_acc[5]);
#endif
#undef OASR_PROF_T
}

// One frame per launch — used by the single-frame streaming API and as the
// step kernel captured into the legacy-path CUDA graphs.  Resolves the frame
// via select_seqs and keeps the in-kernel vocab select (no pre-pass).
template <int BLOCK_SIZE, int BEAM_CAP, bool PAGED>
__global__ __launch_bounds__(BLOCK_SIZE) void fused_step_kernel(
    const float* __restrict__ log_prob, int batch_stride, int seq_stride, int vocab_stride,
    const int* __restrict__ select_seqs, const int* __restrict__ select_seq_lens,
    const int* __restrict__ d_step, int step_const, float2* __restrict__ pprev,
    int* __restrict__ clast, const int* __restrict__ clen_src, int* __restrict__ clen_dst,
    const int* __restrict__ clist_src, int* __restrict__ clist_dst,
    const int* __restrict__ ctime_src, int* __restrict__ ctime_dst, float* __restrict__ score,
    int ldc, int beam, int ldbeam, int ldseq_len, int batch, int blank_id, int space_id,
    int max_seq_len, int* __restrict__ page_storage, int* __restrict__ time_storage,
    const int* __restrict__ block_table_src,
    int* __restrict__ block_table_dst, int* __restrict__ ref_counts,
    int* __restrict__ next_free_page, int* __restrict__ free_pool,
    int* __restrict__ free_pool_size, int page_size, int max_lp, int pages_per_row) {
    using Smem = FusedStepSmem<BLOCK_SIZE, BEAM_CAP>;
    __shared__ Smem s;

    const int bid = blockIdx.x;
    if (bid >= batch)
        return;
    // step_const >= 0 is the offline fast path (the host loop knows the step,
    // saving a per-step device-counter update); -1 reads the device-resident
    // counter so a captured streaming graph stays valid across frames.
    const int step = (step_const >= 0) ? step_const : __ldg(d_step);
    if (step >= select_seq_lens[bid])
        return;

    if (PAGED) {
        // Row-local allocator views (see the alloc_page/free_page comment).
        free_pool += bid * pages_per_row;
        free_pool_size += bid;
        next_free_page += bid;
    }

    // Width-max_seq_len ring (no-op modulo for offline); see
    // set_select_seq_step_kernel for the streaming rationale.
    const int t = select_seqs[bid * max_seq_len + step % max_seq_len];
    const int t_prev = select_seqs[bid * max_seq_len + (step - 1) % max_seq_len];
    const bool need_add_blank = (t > t_prev + 1);
    const float* lp_row = log_prob + (size_t)bid * batch_stride + (size_t)t * seq_stride;

    fused_step_body<BLOCK_SIZE, BEAM_CAP, PAGED>(
        s, bid, lp_row, vocab_stride, need_add_blank, /*pre_cnt_row=*/-1, nullptr, nullptr,
        pprev, clast, clen_src, clen_dst, clist_src, clist_dst, ctime_src, ctime_dst, t,
        score, ldc, beam, ldbeam, ldseq_len, blank_id, space_id, page_storage, time_storage,
        block_table_src, block_table_dst,
        ref_counts, next_free_page, free_pool, free_pool_size, page_size, max_lp);
}

// --- multi-frame fused chunk kernel --------------------------------------------
//
// One launch decodes a whole PREPASS_TILE tile of frames: each block owns one
// batch row and loops the tile's frames in-kernel, carrying beam state across
// iterations (global state writes by a block are visible to its own reads
// after __syncthreads()).  This removes the per-frame launch chain — counter
// kernels, d_lp_frame_buf copies and CUDA-graph replays — that dominated the
// latency-bound step after the pre-pass hoist.
//
// Two modes:
//   * streaming == 0 (offline): frame row r covers step_begin + r through the
//     select_seqs indirection; a row returns once step >= select_seq_lens.
//   * streaming == 1 (chunk): frame row r covers chunk position
//     chunk_frame_begin + r; bit r of mask_lo/mask_hi gates decoding (clear =
//     blank-skip frame, which advances only the frame index).  ``step`` and
//     the actual frame index start at step_begin / frame_begin and advance
//     in-kernel; the select_seqs ring is maintained inline (replacing
//     set_select_seq_step_kernel).
//
// step == 0 of a stream runs the first-step initialisation from the same
// pre-pass candidates (a superset of the top-(nb_beams + 1) chars the
// dedicated first-step kernels select, so the greedy non-blank pick below is
// identical).  Double-buffer parity is resolved per step in-kernel.
//
// The loop lives in ``fused_chunk_loop`` so the single-state kernel and the
// multi-stream batched kernel share it; callers pass the state's own buffer
// pointers (paged allocator views already row-localised) plus that block's
// ``bid``.  ``log_prob`` is the state's lp base — the loop adds
// ``bid * batch_stride`` itself.
template <int BLOCK_SIZE, int BEAM_CAP, bool PAGED>
__device__ __forceinline__ void fused_chunk_loop(
    FusedStepSmem<BLOCK_SIZE, BEAM_CAP>& s, int bid, const float* __restrict__ log_prob,
    int batch_stride, int seq_stride, int vocab_stride, int* __restrict__ select_seqs,
    const int* __restrict__ select_seq_lens, int step_begin, int frame_begin,
    int chunk_frame_begin, int tile_len, unsigned long long mask_lo,
    unsigned long long mask_hi, int streaming, float2* __restrict__ pprev,
    int* __restrict__ clast, int* __restrict__ clen0, int* __restrict__ clen1,
    int* __restrict__ clist0, int* __restrict__ clist1, int* __restrict__ ctime0,
    int* __restrict__ ctime1, float* __restrict__ score, int ldc,
    int beam, int ldbeam, int ldseq_len, int batch, int blank_id, int space_id,
    int max_seq_len, int* __restrict__ page_storage, int* __restrict__ time_storage,
    int* __restrict__ block_table0,
    int* __restrict__ block_table1, int* __restrict__ ref_counts,
    int* __restrict__ next_free_page, int* __restrict__ free_pool,
    int* __restrict__ free_pool_size, int page_size, int max_lp, int pages_per_row,
    const int* __restrict__ pre_chars, const float* __restrict__ pre_lp,
    const int* __restrict__ pre_cnt) {
    using Smem = FusedStepSmem<BLOCK_SIZE, BEAM_CAP>;
    const int tid = threadIdx.x;

    const int sel_len = select_seq_lens[bid];  // INT_MAX in streaming
    int step = step_begin;

    for (int r = 0; r < tile_len; ++r) {
        if (streaming) {
            // Blank-skip frames advance only the frame index (implicit:
            // frame_begin + r tracks it); uniform across the block.
            const bool dec =
                (r < 64) ? ((mask_lo >> r) & 1ull) : ((mask_hi >> (r - 64)) & 1ull);
            if (!dec)
                continue;
        }
        if (step >= sel_len)
            return;  // offline: this row decoded all its selected frames

        int frame_for_gap;  // index used for the skipped-blank-frame gap test
        const float* lp_row;
        if (streaming) {
            const int t = chunk_frame_begin + r;
            frame_for_gap = frame_begin + r;
            lp_row = log_prob + (size_t)bid * batch_stride + (size_t)t * seq_stride;
        } else {
            const int t = select_seqs[bid * max_seq_len + step % max_seq_len];
            frame_for_gap = t;
            lp_row = log_prob + (size_t)bid * batch_stride + (size_t)t * seq_stride;
        }

        // Pre-pass candidates for this frame (row r of the tile buffers).
        const int pre_off = (r * batch + bid) * Smem::C_BUF;
        const int cn = min(__ldg(&pre_cnt[r * batch + bid]), Smem::C_BUF);

        if (step == 0) {
            // --- First step: init beams from the top non-blank chars --------
            // (inline replacement for fused_first_step_kernel, fed by the
            // pre-pass superset instead of a fresh vocab scan).
            if (streaming && tid == 0)
                select_seqs[bid * max_seq_len] = frame_for_gap;
            const int nb_beams = (beam > 1) ? beam - 1 : beam;
            for (int i = tid; i < cn; i += BLOCK_SIZE) {
                s.c_chars[i] = pre_chars[pre_off + i];
                s.c_lp[i] = pre_lp[pre_off + i];
            }
            __syncthreads();
            {
                auto key_at = [&](int i) -> uint64_t {
                    return (uint64_t(f32_sortable(s.c_lp[i])) << 32) |
                           uint64_t(0xffffffffu - uint32_t(s.c_chars[i]));
                };
                auto emit = [&](int slot, int /*i*/, uint64_t k) {
                    if (slot < Smem::RANK_BUF)
                        s.rank[slot] = k;
                };
                block_topk_select<BLOCK_SIZE, Smem::RANK_BUF>(cn, nb_beams + 1, key_at, emit,
                                                              &s.sel);
            }
            __syncthreads();
            for (int i = tid; i < Smem::RANK_BUF; i += BLOCK_SIZE)
                if (i >= s.sel.out_n)
                    s.rank[i] = (uint64_t)(Smem::CAND_CAP + i);  // unique; sorts last
            __syncthreads();
            rank64_desc<BLOCK_SIZE, BEAM_CAP>(s);  // descending into s.ckeys
            if (tid == 0) {
                // Greedy non-blank pick in descending order; stage the new
                // beams in the (otherwise unused at step 0) snapshot arrays.
                int out = 0;
                for (int rr = 0; rr < s.sel.out_n && out < nb_beams; ++rr) {
                    int c = (int)ckey_id(s.ckeys[rr]);
                    if (c == blank_id)
                        continue;
                    s.s_clast[out] = c;
                    s.lp_clast[out] = ckey_score(s.ckeys[rr]);
                    ++out;
                }
                for (; out < nb_beams; ++out)
                    s.s_clast[out] = -1;
            }
            __syncthreads();
            for (int k = tid; k < nb_beams; k += BLOCK_SIZE) {
                int base = bid * ldbeam + k;
                int token = s.s_clast[k];
                if (token >= 0) {
                    // Prefix [token] ends in a non-blank: mass goes in the
                    // non-blank slot (see the legacy first_step_kernel).
                    pprev[base] = make_float2(NEG_INF, s.lp_clast[k]);
                    if (PAGED) {
                        page_storage[(size_t)(bid * pages_per_row + k) * page_size] = token;
                        time_storage[(size_t)(bid * pages_per_row + k) * page_size] =
                            frame_for_gap;
                    } else {
                        clist0[(size_t)(bid * beam + k) * ldseq_len] = token;
                        if (ctime0)
                            ctime0[(size_t)(bid * beam + k) * ldseq_len] = frame_for_gap;
                    }
                    clen0[base] = 1;
                    clast[base] = token;
                    score[base] = s.lp_clast[k];
                } else {
                    pprev[base] = make_float2(NEG_INF, NEG_INF);
                    clast[base] = blank_id;
                    clen0[base] = 0;
                    score[base] = NEG_INF;
                }
            }
            if (beam > 1 && tid == 0) {
                int base = bid * ldbeam + (beam - 1);
                float blank_prob = lp_row[(size_t)blank_id * vocab_stride];
                pprev[base] = make_float2(blank_prob, NEG_INF);
                clast[base] = blank_id;
                clen0[base] = 0;
                score[base] = blank_prob;
            }
        } else {
            // --- Regular step ------------------------------------------------
            // Gap test against the previous decoded frame (ring entry written
            // by the previous iteration, a previous chunk, or init_select).
            const int prev_frame =
                select_seqs[bid * max_seq_len + (step - 1) % max_seq_len];
            const bool need_add_blank = (frame_for_gap > prev_frame + 1);
            if (streaming && tid == 0)
                select_seqs[bid * max_seq_len + step % max_seq_len] = frame_for_gap;

            const int srcp = (step - 1) & 1;
            const int dstp = step & 1;
            fused_step_body<BLOCK_SIZE, BEAM_CAP, PAGED>(
                s, bid, lp_row, vocab_stride, need_add_blank, cn, pre_chars + pre_off,
                pre_lp + pre_off, pprev, clast, srcp ? clen1 : clen0, dstp ? clen1 : clen0,
                srcp ? clist1 : clist0, dstp ? clist1 : clist0, srcp ? ctime1 : ctime0,
                dstp ? ctime1 : ctime0, frame_for_gap, score, ldc, beam, ldbeam,
                ldseq_len, blank_id, space_id, page_storage, time_storage,
                srcp ? block_table1 : block_table0, dstp ? block_table1 : block_table0,
                ref_counts, next_free_page, free_pool, free_pool_size, page_size, max_lp);
        }

        // Publish this step's global-state writes to the next iteration.
        __syncthreads();
        ++step;
    }
}

// Single-state chunk kernel: one block per batch row of one stream.
template <int BLOCK_SIZE, int BEAM_CAP, bool PAGED>
__global__ __launch_bounds__(BLOCK_SIZE) void fused_chunk_kernel(
    const float* __restrict__ log_prob, int batch_stride, int seq_stride, int vocab_stride,
    int* __restrict__ select_seqs, const int* __restrict__ select_seq_lens, int step_begin,
    int frame_begin, int chunk_frame_begin, int tile_len, unsigned long long mask_lo,
    unsigned long long mask_hi, int streaming, float2* __restrict__ pprev,
    int* __restrict__ clast, int* __restrict__ clen0, int* __restrict__ clen1,
    int* __restrict__ clist0, int* __restrict__ clist1, int* __restrict__ ctime0,
    int* __restrict__ ctime1, float* __restrict__ score, int ldc,
    int beam, int ldbeam, int ldseq_len, int batch, int blank_id, int space_id,
    int max_seq_len, int* __restrict__ page_storage, int* __restrict__ time_storage,
    int* __restrict__ block_table0,
    int* __restrict__ block_table1, int* __restrict__ ref_counts,
    int* __restrict__ next_free_page, int* __restrict__ free_pool,
    int* __restrict__ free_pool_size, int page_size, int max_lp, int pages_per_row,
    const int* __restrict__ pre_chars, const float* __restrict__ pre_lp,
    const int* __restrict__ pre_cnt) {
    using Smem = FusedStepSmem<BLOCK_SIZE, BEAM_CAP>;
    __shared__ Smem s;

    const int bid = blockIdx.x;
    if (bid >= batch)
        return;

    if (PAGED) {
        // Row-local allocator views (see the alloc_page/free_page comment).
        free_pool += bid * pages_per_row;
        free_pool_size += bid;
        next_free_page += bid;
    }

    fused_chunk_loop<BLOCK_SIZE, BEAM_CAP, PAGED>(
        s, bid, log_prob, batch_stride, seq_stride, vocab_stride, select_seqs,
        select_seq_lens, step_begin, frame_begin, chunk_frame_begin, tile_len, mask_lo,
        mask_hi, streaming, pprev, clast, clen0, clen1, clist0, clist1, ctime0, ctime1,
        score, ldc, beam, ldbeam, ldseq_len, batch, blank_id, space_id, max_seq_len,
        page_storage, time_storage,
        block_table0, block_table1, ref_counts, next_free_page, free_pool, free_pool_size,
        page_size, max_lp, pages_per_row, pre_chars, pre_lp, pre_cnt);
}

// --- multi-stream batched chunk decode ------------------------------------------
//
// ``ctc_beam_search_chunk_batched`` used to launch the (pre-pass + chunk
// kernel) pair once per stream on one CUDA stream, leaving N independent
// streams to run as one serial chain N tiles deep.  The batched kernels fold
// a *group* of up to FusedStreamGroup::CAP streams into a single launch
// (grid = group_size x batch blocks), so the serial depth per tile drops from
// N to ceil(N / CAP).
//
// All streams of a group share one config (engine invariant) and therefore
// one workspace layout; per-stream state lives at ``base + delta[slot]``
// relative to the group's first stream, so the kernels take the first
// stream's pointers plus a by-value array of byte deltas — no device-side
// pointer table, no H2D upload.  Per-stream counters and blank-skip masks
// ride in the same by-value struct (32 B per stream; CAP = 64 keeps the
// kernel-parameter block ~2.4 KB, comfortably under the 4 KB CUDA limit).
struct FusedStreamGroup {
    static constexpr int CAP = 64;
    long long delta[CAP];  // state-buffer byte offset vs the group's first stream
    unsigned long long mask_lo[CAP];
    unsigned long long mask_hi[CAP];
    int step_begin[CAP];
    int frame_begin[CAP];
    int n;  // active streams in this group (<= CAP)
};

__device__ __forceinline__ char* group_shift(const void* p, long long d) {
    return const_cast<char*>(reinterpret_cast<const char*>(p)) + d;
}

// Grid: (group_size * batch) blocks; block i covers (stream slot i / batch,
// batch row i % batch).  ``log_prob`` points at the group's first stream's
// rows of the stacked (N, T, V) chunk tensor; state row ``bid`` of stream
// slot ``si`` reads lp row ``si + bid`` (same convention as the per-stream
// launcher, where each state's rows follow its stream index).
template <int BLOCK_SIZE, int BEAM_CAP, bool PAGED>
__global__ __launch_bounds__(BLOCK_SIZE) void fused_chunk_batched_kernel(
    const float* __restrict__ log_prob, int batch_stride, int seq_stride, int vocab_stride,
    FusedStreamGroup grp, int chunk_frame_begin, int tile_len, int* __restrict__ select_seqs0,
    const int* __restrict__ select_seq_lens0, float2* __restrict__ pprev0,
    int* __restrict__ clast0, int* __restrict__ clen00, int* __restrict__ clen10,
    int* __restrict__ clist00, int* __restrict__ clist10, int* __restrict__ ctime00,
    int* __restrict__ ctime10, float* __restrict__ score0,
    int ldc, int beam, int ldbeam, int ldseq_len, int batch, int blank_id, int space_id,
    int max_seq_len, int* __restrict__ page_storage0, int* __restrict__ time_storage0,
    int* __restrict__ block_table00,
    int* __restrict__ block_table10, int* __restrict__ ref_counts0,
    int* __restrict__ next_free_page0, int* __restrict__ free_pool0,
    int* __restrict__ free_pool_size0, int page_size, int max_lp, int pages_per_row,
    const int* __restrict__ pre_chars0, const float* __restrict__ pre_lp0,
    const int* __restrict__ pre_cnt0) {
    using Smem = FusedStepSmem<BLOCK_SIZE, BEAM_CAP>;
    __shared__ Smem s;

    const int si = blockIdx.x / batch;   // stream slot within the group
    const int bid = blockIdx.x - si * batch;
    if (si >= grp.n)
        return;
    const long long d = grp.delta[si];

    int* select_seqs = reinterpret_cast<int*>(group_shift(select_seqs0, d));
    const int* select_seq_lens = reinterpret_cast<const int*>(group_shift(select_seq_lens0, d));
    float2* pprev = reinterpret_cast<float2*>(group_shift(pprev0, d));
    int* clast = reinterpret_cast<int*>(group_shift(clast0, d));
    int* clen0 = reinterpret_cast<int*>(group_shift(clen00, d));
    int* clen1 = reinterpret_cast<int*>(group_shift(clen10, d));
    int* clist0 = PAGED ? nullptr : reinterpret_cast<int*>(group_shift(clist00, d));
    int* clist1 = PAGED ? nullptr : reinterpret_cast<int*>(group_shift(clist10, d));
    int* ctime0 = PAGED ? nullptr : reinterpret_cast<int*>(group_shift(ctime00, d));
    int* ctime1 = PAGED ? nullptr : reinterpret_cast<int*>(group_shift(ctime10, d));
    float* score = reinterpret_cast<float*>(group_shift(score0, d));
    const int* pre_chars = reinterpret_cast<const int*>(group_shift(pre_chars0, d));
    const float* pre_lp = reinterpret_cast<const float*>(group_shift(pre_lp0, d));
    const int* pre_cnt = reinterpret_cast<const int*>(group_shift(pre_cnt0, d));

    int* page_storage = nullptr;
    int* time_storage = nullptr;
    int* block_table0 = nullptr;
    int* block_table1 = nullptr;
    int* ref_counts = nullptr;
    int* next_free_page = nullptr;
    int* free_pool = nullptr;
    int* free_pool_size = nullptr;
    if (PAGED) {
        page_storage = reinterpret_cast<int*>(group_shift(page_storage0, d));
        time_storage = reinterpret_cast<int*>(group_shift(time_storage0, d));
        block_table0 = reinterpret_cast<int*>(group_shift(block_table00, d));
        block_table1 = reinterpret_cast<int*>(group_shift(block_table10, d));
        ref_counts = reinterpret_cast<int*>(group_shift(ref_counts0, d));
        // Row-local allocator views (see the alloc_page/free_page comment).
        next_free_page = reinterpret_cast<int*>(group_shift(next_free_page0, d)) + bid;
        free_pool =
            reinterpret_cast<int*>(group_shift(free_pool0, d)) + bid * pages_per_row;
        free_pool_size = reinterpret_cast<int*>(group_shift(free_pool_size0, d)) + bid;
    }

    fused_chunk_loop<BLOCK_SIZE, BEAM_CAP, PAGED>(
        s, bid, log_prob + (size_t)si * batch_stride, batch_stride, seq_stride, vocab_stride,
        select_seqs, select_seq_lens, grp.step_begin[si], grp.frame_begin[si],
        chunk_frame_begin, tile_len, grp.mask_lo[si], grp.mask_hi[si], /*streaming=*/1,
        pprev, clast, clen0, clen1, clist0, clist1, ctime0, ctime1, score, ldc, beam,
        ldbeam, ldseq_len, batch, blank_id, space_id, max_seq_len, page_storage,
        time_storage, block_table0, block_table1,
        ref_counts, next_free_page, free_pool, free_pool_size, page_size, max_lp,
        pages_per_row, pre_chars, pre_lp, pre_cnt);
}

// --- chunk-level vocab top-K pre-pass -----------------------------------------
//
// The fused step is sequential across frames (beam-state dependency) and
// latency-bound, but its Phase-3 vocab ranking depends only on the frame's
// log-probs.  This kernel hoists that ranking out of the step loop: one block
// per (frame row, batch row) — grid (tile_len, batch) — ranks the frame's
// vocab once and emits the top-K chars + log-probs (K = 2*beam + 2, an upper
// bound on any step's k_all = beam + 3 + max_patches) to the pre_chars /
// pre_lp / pre_cnt workspace buffers, layout [row][batch][MAX_OUT].
//
// Two frame-indexing modes:
//   * select_seqs != nullptr (offline): block row r covers step_begin + r via
//     the select_seqs indirection; rows with step >= select_seq_lens[bid] are
//     skipped (the step kernel guards identically, so those buffer rows are
//     never read).
//   * select_seqs == nullptr (streaming): block row r covers chunk frame
//     step_begin + r directly.
//
// Bit r of mask_lo/mask_hi gates row r (clear = blank-skip frame, never
// consumed by the chunk kernel — its block exits immediately).
template <int BLOCK_SIZE>
struct PrepassSmem {
    static constexpr int VKEY_CACHE = 6144;
    uint32_t vkeys[VKEY_CACHE];
    SelectScratch<BLOCK_SIZE> sel;
};

// Rank one frame's vocab and emit the top-K chars + log-probs; shared by the
// single-state and the multi-stream batched pre-pass kernels.
template <int BLOCK_SIZE, int MAX_OUT>
__device__ __forceinline__ void prepass_rank_row(PrepassSmem<BLOCK_SIZE>& s,
                                                 const float* __restrict__ lp_row,
                                                 int vocab_stride, int K, int vocab_size,
                                                 int* __restrict__ out_chars,
                                                 float* __restrict__ out_lp,
                                                 int* __restrict__ out_cnt) {
    const int tid = threadIdx.x;
    const bool use_vcache = (vocab_size <= PrepassSmem<BLOCK_SIZE>::VKEY_CACHE);
    if (use_vcache) {
        for (int c = tid; c < vocab_size; c += BLOCK_SIZE)
            s.vkeys[c] = f32_sortable(lp_row[(size_t)c * vocab_stride]);
        __syncthreads();
    }

    auto key_at = [&](int c) -> uint64_t {
        uint32_t sk = use_vcache ? s.vkeys[c] : f32_sortable(lp_row[(size_t)c * vocab_stride]);
        return (uint64_t(sk) << 32) | uint64_t(0xffffffffu - uint32_t(c));
    };
    auto emit = [&](int slot, int c, uint64_t k) {
        if (slot < MAX_OUT) {
            out_chars[slot] = c;
            out_lp[slot] = ckey_score(k);
        }
    };
    int cnt = block_topk_select<BLOCK_SIZE, MAX_OUT>(vocab_size, K, key_at, emit, &s.sel);
    if (tid == 0)
        *out_cnt = min(cnt, MAX_OUT);
}

template <int BLOCK_SIZE, int MAX_OUT>
__global__ __launch_bounds__(BLOCK_SIZE) void fused_topk_prepass_kernel(
    const float* __restrict__ log_prob, int batch_stride, int seq_stride, int vocab_stride,
    const int* __restrict__ select_seqs, const int* __restrict__ select_seq_lens,
    int step_begin, int K, unsigned long long mask_lo, unsigned long long mask_hi,
    int* __restrict__ pre_chars, float* __restrict__ pre_lp,
    int* __restrict__ pre_cnt, int vocab_size, int batch, int max_seq_len) {
    __shared__ PrepassSmem<BLOCK_SIZE> s;

    const int bid = blockIdx.y;
    const int row = blockIdx.x;
    if (!((row < 64) ? ((mask_lo >> row) & 1ull) : ((mask_hi >> (row - 64)) & 1ull)))
        return;
    int t;
    if (select_seqs != nullptr) {
        const int step = step_begin + row;
        if (step >= select_seq_lens[bid])
            return;
        // Width-max_seq_len ring (no-op modulo for offline, where step is
        // always < max_seq_len; see set_select_seq_step_kernel).
        t = select_seqs[bid * max_seq_len + step % max_seq_len];
    } else {
        t = step_begin + row;
    }
    const float* lp_row = log_prob + (size_t)bid * batch_stride + (size_t)t * seq_stride;

    const int out_off = (row * batch + bid) * MAX_OUT;
    prepass_rank_row<BLOCK_SIZE, MAX_OUT>(s, lp_row, vocab_stride, K, vocab_size,
                                          pre_chars + out_off, pre_lp + out_off,
                                          pre_cnt + row * batch + bid);
}

// Multi-stream batched pre-pass: grid (tile_len, group_size * batch).  Frame
// indexing is identity (streaming chunk positions); per-stream output buffers
// sit at ``ptr0 + grp.delta[slot]`` and per-stream mask bits gate each row.
template <int BLOCK_SIZE, int MAX_OUT>
__global__ __launch_bounds__(BLOCK_SIZE) void fused_topk_prepass_batched_kernel(
    const float* __restrict__ log_prob, int batch_stride, int seq_stride, int vocab_stride,
    FusedStreamGroup grp, int frame_begin, int K, int* __restrict__ pre_chars0,
    float* __restrict__ pre_lp0, int* __restrict__ pre_cnt0, int vocab_size, int batch) {
    __shared__ PrepassSmem<BLOCK_SIZE> s;

    const int row = blockIdx.x;
    const int si = blockIdx.y / batch;
    const int bid = blockIdx.y - si * batch;
    if (si >= grp.n)
        return;
    if (!((row < 64) ? ((grp.mask_lo[si] >> row) & 1ull)
                     : ((grp.mask_hi[si] >> (row - 64)) & 1ull)))
        return;

    const long long d = grp.delta[si];
    int* pre_chars = reinterpret_cast<int*>(group_shift(pre_chars0, d));
    float* pre_lp = reinterpret_cast<float*>(group_shift(pre_lp0, d));
    int* pre_cnt = reinterpret_cast<int*>(group_shift(pre_cnt0, d));

    const int t = frame_begin + row;
    const float* lp_row =
        log_prob + (size_t)(si + bid) * batch_stride + (size_t)t * seq_stride;

    const int out_off = (row * batch + bid) * MAX_OUT;
    prepass_rank_row<BLOCK_SIZE, MAX_OUT>(s, lp_row, vocab_stride, K, vocab_size,
                                          pre_chars + out_off, pre_lp + out_off,
                                          pre_cnt + row * batch + bid);
}

}  // namespace fused

// Per-row entry stride of the pre-pass buffers.  Must equal the C_BUF of the
// fused_step_kernel instantiation picked by the same beam bucket (the step
// kernel indexes the buffers with Smem::C_BUF directly).
inline constexpr int fused_prepass_stride(int beam) {
    return (beam <= 16) ? fused::FusedStepSmem<256, 16>::C_BUF
                        : fused::FusedStepSmem<256, 32>::C_BUF;
}

// Launch the chunk-level top-K pre-pass over ``tile_len`` frame rows
// (tile_len <= PREPASS_TILE).  Offline callers pass select_seqs /
// select_seq_lens and step_begin = the tile's first step; streaming callers
// pass select_seqs == nullptr and step_begin = the tile's first chunk frame.
// Bit r of mask_lo/mask_hi gates row r (streaming blank-skip; all-ones for
// offline).  No-op on the legacy layout (pre-pass buffers not allocated).
inline cudaError_t launch_fused_topk_prepass(const InternalData* data, const float* log_prob,
                                             int batch_stride, int seq_stride, int vocab_stride,
                                             const int* select_seqs, const int* select_seq_lens,
                                             int step_begin, int tile_len, cudaStream_t stream,
                                             unsigned long long mask_lo = ~0ull,
                                             unsigned long long mask_hi = ~0ull) {
    if (data->pre_chars == nullptr || tile_len <= 0)
        return cudaSuccess;
    constexpr int FUSED_BLOCK = 256;
    const int K = 2 * data->beam + 2;
    dim3 grid(tile_len, data->batch);
#define OASR_LAUNCH_FUSED_PREPASS(BEAM_CAP)                                                     \
    fused::fused_topk_prepass_kernel<FUSED_BLOCK,                                               \
                                     fused::FusedStepSmem<FUSED_BLOCK, BEAM_CAP>::C_BUF>        \
        <<<grid, FUSED_BLOCK, 0, stream>>>(log_prob, batch_stride, seq_stride, vocab_stride,    \
                                           select_seqs, select_seq_lens, step_begin, K,         \
                                           mask_lo, mask_hi,                                    \
                                           data->pre_chars, data->pre_lp, data->pre_cnt,        \
                                           data->vocab_size, data->batch, data->max_seq_len)
    if (data->beam <= 16) {
        OASR_LAUNCH_FUSED_PREPASS(16);
    } else {
        OASR_LAUNCH_FUSED_PREPASS(32);
    }
#undef OASR_LAUNCH_FUSED_PREPASS
    return cudaGetLastError();
}

// Launch the multi-frame fused chunk kernel over one pre-pass tile.  Offline:
// streaming=false, step_begin = the tile's first step, frames resolved via
// select_seqs, mask all-ones.  Streaming: streaming=true, frame row r covers
// chunk position chunk_frame_begin + r gated by mask bit r; step_begin /
// frame_begin are the stream's counters at row 0.  Callers guarantee
// step_uses_fused(beam) (pre-pass buffers present).
inline cudaError_t launch_fused_chunk(const InternalData* data, const float* log_prob,
                                      int batch_stride, int seq_stride, int vocab_stride,
                                      int step_begin, int frame_begin, int chunk_frame_begin,
                                      int tile_len, unsigned long long mask_lo,
                                      unsigned long long mask_hi, bool streaming, int blank_id,
                                      int space_id, cudaStream_t stream) {
    if (tile_len <= 0)
        return cudaSuccess;
    constexpr int FUSED_BLOCK = 256;
    const auto& ps = data->paged;
    const bool paged = ps.is_enabled();
    const int fused_ppr = paged ? ps.num_pages / (data->batch > 0 ? data->batch : 1) : 0;
#define OASR_LAUNCH_FUSED_CHUNK(BEAM_CAP, PAGED_)                                              \
    fused::fused_chunk_kernel<FUSED_BLOCK, BEAM_CAP, PAGED_>                                   \
        <<<data->batch, FUSED_BLOCK, 0, stream>>>(                                             \
            log_prob, batch_stride, seq_stride, vocab_stride, data->select_seqs,               \
            data->select_seq_lens, step_begin, frame_begin, chunk_frame_begin, tile_len,       \
            mask_lo, mask_hi, streaming ? 1 : 0, data->pprev, data->clast, data->clen[0],      \
            data->clen[1], data->clist[0], data->clist[1], data->ctime[0], data->ctime[1],    \
            data->score, data->ldc,                                                            \
            data->beam, data->ldbeam, data->ldseq_len, data->batch, blank_id, space_id,        \
            data->max_seq_len, ps.page_storage, ps.time_storage, ps.block_table[0],            \
            ps.block_table[1],                                                                 \
            ps.ref_counts, ps.next_free_page, ps.free_pool, ps.free_pool_size, ps.page_size,   \
            ps.max_logical_pages, fused_ppr, data->pre_chars, data->pre_lp, data->pre_cnt)
    if (data->beam <= 16) {
        if (paged) {
            OASR_LAUNCH_FUSED_CHUNK(16, true);
        } else {
            OASR_LAUNCH_FUSED_CHUNK(16, false);
        }
    } else {
        if (paged) {
            OASR_LAUNCH_FUSED_CHUNK(32, true);
        } else {
            OASR_LAUNCH_FUSED_CHUNK(32, false);
        }
    }
#undef OASR_LAUNCH_FUSED_CHUNK
    return cudaGetLastError();
}

// Launch one (batched pre-pass + batched chunk kernel) pair for a GROUP of
// streams over one tile.  ``data`` describes the group's first stream;
// per-stream byte deltas, counters and blank-skip masks ride in ``grp`` (by
// value — no device-side pointer table or H2D upload).  ``log_prob`` points
// at the group's first stream's rows of the stacked (N, T, V) chunk tensor;
// ``tile_begin`` is the tile's first chunk frame.
inline cudaError_t launch_fused_chunk_batched(const InternalData* data, const float* log_prob,
                                              int batch_stride, int seq_stride, int vocab_stride,
                                              const fused::FusedStreamGroup& grp, int tile_begin,
                                              int tile_len, int blank_id, int space_id,
                                              cudaStream_t stream) {
    if (tile_len <= 0 || grp.n <= 0)
        return cudaSuccess;
    constexpr int FUSED_BLOCK = 256;
    const auto& ps = data->paged;
    const bool paged = ps.is_enabled();
    const int fused_ppr = paged ? ps.num_pages / (data->batch > 0 ? data->batch : 1) : 0;
    const int K = 2 * data->beam + 2;
    dim3 pre_grid(tile_len, grp.n * data->batch);
    const int chunk_blocks = grp.n * data->batch;
#define OASR_LAUNCH_FUSED_BATCHED(BEAM_CAP, PAGED_)                                            \
    do {                                                                                       \
        fused::fused_topk_prepass_batched_kernel<                                              \
            FUSED_BLOCK, fused::FusedStepSmem<FUSED_BLOCK, BEAM_CAP>::C_BUF>                   \
            <<<pre_grid, FUSED_BLOCK, 0, stream>>>(                                            \
                log_prob, batch_stride, seq_stride, vocab_stride, grp, tile_begin, K,          \
                data->pre_chars, data->pre_lp, data->pre_cnt, data->vocab_size, data->batch);  \
        fused::fused_chunk_batched_kernel<FUSED_BLOCK, BEAM_CAP, PAGED_>                       \
            <<<chunk_blocks, FUSED_BLOCK, 0, stream>>>(                                        \
                log_prob, batch_stride, seq_stride, vocab_stride, grp, tile_begin, tile_len,   \
                data->select_seqs, data->select_seq_lens, data->pprev, data->clast,            \
                data->clen[0], data->clen[1], data->clist[0], data->clist[1],                  \
                data->ctime[0], data->ctime[1], data->score,                                   \
                data->ldc, data->beam, data->ldbeam, data->ldseq_len, data->batch, blank_id,   \
                space_id, data->max_seq_len, ps.page_storage, ps.time_storage,                 \
                ps.block_table[0],                                                             \
                ps.block_table[1], ps.ref_counts, ps.next_free_page, ps.free_pool,             \
                ps.free_pool_size, ps.page_size, ps.max_logical_pages, fused_ppr,              \
                data->pre_chars, data->pre_lp, data->pre_cnt);                                 \
    } while (0)
    if (data->beam <= 16) {
        if (paged) {
            OASR_LAUNCH_FUSED_BATCHED(16, true);
        } else {
            OASR_LAUNCH_FUSED_BATCHED(16, false);
        }
    } else {
        if (paged) {
            OASR_LAUNCH_FUSED_BATCHED(32, true);
        } else {
            OASR_LAUNCH_FUSED_BATCHED(32, false);
        }
    }
#undef OASR_LAUNCH_FUSED_BATCHED
    return cudaGetLastError();
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
    total += align_size(sizeof(float2) * batch * ldbeam);  // pprev
    if (layout_has_prob_tables(beam)) {
        total += align_size(sizeof(float) * batch * beam * ldc);  // ptable
        total += align_size(sizeof(float) * batch * beam * ldc);  // ptablen
    }
    total += align_size(sizeof(int) * batch * ldbeam);                // clast
    total += align_size(sizeof(int) * batch * ldbeam) * 2;            // clen[0..1]
    total += align_size(sizeof(int) * batch * beam * ldseq_len) * 2;  // clist[0..1]
    total += align_size(sizeof(int) * batch * beam * ldseq_len) * 2;  // ctime[0..1]
    total += align_size(sizeof(int) * batch * ldbeam);                // ptid (unused)
    total += align_size(sizeof(float) * batch * ldbeam);              // score
    if (layout_has_prob_tables(beam)) {
        // topk buffers: batch * MAX_BLOCKS_PER_BATCH * beam (Phase 1 output)
        total += align_size(sizeof(float) * batch * MAX_BLOCKS_PER_BATCH * beam);
        total += align_size(sizeof(int) * batch * MAX_BLOCKS_PER_BATCH * beam);
    }
    total += align_size(sizeof(int) * batch * max_seq_len);  // select_seqs
    total += align_size(sizeof(int) * batch);                // select_seq_lens
    total += align_size(sizeof(float) * batch * ldc);        // d_lp_frame_buf (Step 4 capture)
    if (step_uses_fused(beam)) {
        // Chunk-level top-K pre-pass buffers (fused path only)
        int pstride = fused_prepass_stride(beam);
        total += align_size(sizeof(int) * PREPASS_TILE * batch * pstride);    // pre_chars
        total += align_size(sizeof(float) * PREPASS_TILE * batch * pstride);  // pre_lp
        total += align_size(sizeof(int) * PREPASS_TILE * batch);              // pre_cnt
    }
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
    if (layout_has_prob_tables(beam)) {
        ALLOC_BUF(ptable, float, batch * beam * ldc)
        ALLOC_BUF(ptablen, float, batch * beam * ldc)
    } else {
        data->ptable = nullptr;
        data->ptablen = nullptr;
    }
    ALLOC_BUF(clast, int, batch* ldbeam)
    ALLOC_BUF(clen[0], int, batch* ldbeam)
    ALLOC_BUF(clen[1], int, batch* ldbeam)
    ALLOC_BUF(clist[0], int, batch * beam * ldseq_len)
    ALLOC_BUF(clist[1], int, batch * beam * ldseq_len)
    ALLOC_BUF(ctime[0], int, batch * beam * ldseq_len)
    ALLOC_BUF(ctime[1], int, batch * beam * ldseq_len)
    ALLOC_BUF(ptid, int, batch* ldbeam)
    ALLOC_BUF(score, float, batch* ldbeam)
    if (layout_has_prob_tables(beam)) {
        ALLOC_BUF(topk_key_buffer, float, batch * MAX_BLOCKS_PER_BATCH * beam)
        ALLOC_BUF(topk_value_buffer, int, batch * MAX_BLOCKS_PER_BATCH * beam)
    } else {
        data->topk_key_buffer = nullptr;
        data->topk_value_buffer = nullptr;
    }
    ALLOC_BUF(select_seqs, int, batch* max_seq_len)
    ALLOC_BUF(select_seq_lens, int, batch)
    ALLOC_BUF(d_lp_frame_buf, float, batch * ldc)
    if (step_uses_fused(beam)) {
        int pstride = fused_prepass_stride(beam);
        ALLOC_BUF(pre_chars, int, PREPASS_TILE * batch * pstride)
        ALLOC_BUF(pre_lp, float, PREPASS_TILE * batch * pstride)
        ALLOC_BUF(pre_cnt, int, PREPASS_TILE * batch)
    } else {
        data->pre_chars = nullptr;
        data->pre_lp = nullptr;
        data->pre_cnt = nullptr;
    }

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
    if (layout_has_prob_tables(beam)) {
        total += align_size(sizeof(float) * batch * beam * ldc);
        total += align_size(sizeof(float) * batch * beam * ldc);
    }
    total += align_size(sizeof(int) * batch * ldbeam);
    total += align_size(sizeof(int) * batch * ldbeam) * 2;  // clen[0..1]
    // clist[0..1] are OMITTED — replaced by paged region
    total += align_size(sizeof(int) * batch * ldbeam);      // ptid
    total += align_size(sizeof(float) * batch * ldbeam);
    if (layout_has_prob_tables(beam)) {
        total += align_size(sizeof(float) * batch * MAX_BLOCKS_PER_BATCH * beam);
        total += align_size(sizeof(int) * batch * MAX_BLOCKS_PER_BATCH * beam);
    }
    total += align_size(sizeof(int) * batch * max_seq_len);
    total += align_size(sizeof(int) * batch);
    total += align_size(sizeof(float) * batch * ldc);  // d_lp_frame_buf (Step 4 capture)
    if (step_uses_fused(beam)) {
        // Chunk-level top-K pre-pass buffers (fused path only)
        int pstride = fused_prepass_stride(beam);
        total += align_size(sizeof(int) * PREPASS_TILE * batch * pstride);    // pre_chars
        total += align_size(sizeof(float) * PREPASS_TILE * batch * pstride);  // pre_lp
        total += align_size(sizeof(int) * PREPASS_TILE * batch);              // pre_cnt
    }
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
    // clist / ctime stay null in paged mode (the paged region carries both)
    data->clist[0] = nullptr;
    data->clist[1] = nullptr;
    data->ctime[0] = nullptr;
    data->ctime[1] = nullptr;

    int ldbeam = data->ldbeam;
    int ldc = data->ldc;

    char* ptr = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(workspace) + ALIGN_BYTES - 1) /
                                        ALIGN_BYTES * ALIGN_BYTES);

#define ALLOC_BUF_P(name, type, count)         \
    data->name = reinterpret_cast<type*>(ptr); \
    ptr += align_size(sizeof(type) * (count));

    ALLOC_BUF_P(pprev, float2, batch * ldbeam)
    if (layout_has_prob_tables(beam)) {
        ALLOC_BUF_P(ptable, float, batch * beam * ldc)
        ALLOC_BUF_P(ptablen, float, batch * beam * ldc)
    } else {
        data->ptable = nullptr;
        data->ptablen = nullptr;
    }
    ALLOC_BUF_P(clast, int, batch * ldbeam)
    ALLOC_BUF_P(clen[0], int, batch * ldbeam)
    ALLOC_BUF_P(clen[1], int, batch * ldbeam)
    // clist[0] and clist[1] are NOT allocated; paged region follows instead
    ALLOC_BUF_P(ptid, int, batch * ldbeam)
    ALLOC_BUF_P(score, float, batch * ldbeam)
    if (layout_has_prob_tables(beam)) {
        ALLOC_BUF_P(topk_key_buffer, float, batch * MAX_BLOCKS_PER_BATCH * beam)
        ALLOC_BUF_P(topk_value_buffer, int, batch * MAX_BLOCKS_PER_BATCH * beam)
    } else {
        data->topk_key_buffer = nullptr;
        data->topk_value_buffer = nullptr;
    }
    ALLOC_BUF_P(select_seqs, int, batch * max_seq_len)
    ALLOC_BUF_P(select_seq_lens, int, batch)
    ALLOC_BUF_P(d_lp_frame_buf, float, batch * ldc)
    if (step_uses_fused(beam)) {
        int pstride = fused_prepass_stride(beam);
        ALLOC_BUF_P(pre_chars, int, PREPASS_TILE * batch * pstride)
        ALLOC_BUF_P(pre_lp, float, PREPASS_TILE * batch * pstride)
        ALLOC_BUF_P(pre_cnt, int, PREPASS_TILE * batch)
    } else {
        data->pre_chars = nullptr;
        data->pre_lp = nullptr;
        data->pre_cnt = nullptr;
    }

#undef ALLOC_BUF_P

    // Initialise paged state from remaining workspace
    paged_memory::init_paged_state(&data->paged, ptr, batch, beam, max_seq_len,
                                   page_size, num_pages, stream);
}

// =============================================================================
// Host launcher: single step of CTC prefix beam search
// =============================================================================

// ``d_frame_idx`` (streaming only, may be null) is the device-resident absolute
// frame counter.  It exists so the emission times this step records are frame
// indices rather than ring positions; offline passes null and the select_seqs
// entry is already the true frame.
inline cudaError_t ctc_prefix_beam_search_step(InternalData* data, const float* log_prob,
                                               int batch_stride, int seq_stride, int vocab_stride,
                                               int step, bool is_last_step, int blank_id,
                                               int space_id, cudaStream_t stream,
                                               const int* d_step, bool d_step_dynamic = true,
                                               const int* d_frame_idx = nullptr) {
    int batch = data->batch;
    int beam = data->beam;
    int ldc = data->ldc;
    int ldbeam = data->ldbeam;
    int ldseq_len = data->ldseq_len;
    int max_seq_len = data->max_seq_len;

    // Double-buffer parity: step=0 writes to clen[0]/clist[0] (via first_step_kernel).
    // Steps 1,2,... alternate src/dst: dst_parity = step % 2.
    // Host-side parity is used to pick the parity-dependent kernel ptr args;
    // when this launcher is called from a graph-capture host that wraps it,
    // each capture is parity-specific.  Per-frame ``step`` indexing inside
    // each kernel reads from ``*d_step`` so the captured graph stays valid
    // across multiple frames at the same parity.
    int src_parity = (step == 0) ? 0 : ((step - 1) % 2);
    int dst_parity = (step == 0) ? 0 : (step % 2);

    if (step_uses_fused(beam)) {
        constexpr int FUSED_BLOCK = 256;
        if (step == 0) {
            fused::fused_first_step_kernel<FUSED_BLOCK, false><<<batch, FUSED_BLOCK, 0, stream>>>(
                log_prob, batch_stride, seq_stride, vocab_stride, data->select_seqs,
                data->select_seq_lens, data->pprev, data->clast, data->clen[0],
                data->clist[0], data->ctime[0], /*page_storage=*/nullptr,
                /*time_storage=*/nullptr, /*page_size=*/0,
                /*pages_per_row=*/0, data->score, beam, ldbeam, ldseq_len,
                data->vocab_size, blank_id, batch, max_seq_len);
        } else {
            const int step_const = d_step_dynamic ? -1 : step;
#define OASR_LAUNCH_FUSED_STEP(BEAM_CAP)                                                       \
    fused::fused_step_kernel<FUSED_BLOCK, BEAM_CAP, false><<<batch, FUSED_BLOCK, 0, stream>>>( \
        log_prob, batch_stride, seq_stride, vocab_stride, data->select_seqs,                   \
        data->select_seq_lens, d_step, step_const, data->pprev, data->clast,                   \
        data->clen[src_parity], data->clen[dst_parity], data->clist[src_parity],               \
        data->clist[dst_parity], data->ctime[src_parity], data->ctime[dst_parity],             \
        data->score, ldc, beam, ldbeam, ldseq_len, batch, blank_id,                            \
        space_id, max_seq_len, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,  \
        nullptr, 0, 0, 0)
            if (beam <= 16) {
                OASR_LAUNCH_FUSED_STEP(16);
            } else {
                OASR_LAUNCH_FUSED_STEP(32);
            }
#undef OASR_LAUNCH_FUSED_STEP
        }
        return cudaGetLastError();
    }

    if (step == 0) {
        // Initialise beam state from the first selected frame.  Parallel block-wide
        // radix sort top-K (BLOCK_SIZE=128, ITEMS_PER_THREAD=4 → 512 items/iter).
        first_step_kernel<128, 4><<<batch, 128, 0, stream>>>(
            log_prob, batch_stride, seq_stride, vocab_stride, data->select_seqs,
            data->select_seq_lens, data->pprev, data->clast, data->clen[0], data->clist[0],
            data->ctime[0], data->score, beam, ldbeam, ldseq_len, data->vocab_size, blank_id,
            batch, max_seq_len);
    } else {
        // --- 1. Compute probability matrix (non-blank chars) ---
        {
            int total = ldc * beam;
            int threads = 256;
            int bx = min((total + threads - 1) / threads, MAX_BLOCKS / max(batch, 1));
            dim3 grid(bx, batch);
            prob_matrix_kernel<<<grid, threads, 0, stream>>>(
                log_prob, batch_stride, seq_stride, vocab_stride, data->select_seqs,
                data->select_seq_lens, d_step, data->pprev, data->ptable, data->ptablen,
                data->clast, ldc, beam, ldbeam, batch, blank_id, space_id, max_seq_len);
        }

        // --- 2. Blank / space probability ---
        prob_space_blank_kernel<<<batch, ldbeam, 0, stream>>>(
            log_prob, batch_stride, seq_stride, vocab_stride, data->select_seqs,
            data->select_seq_lens, d_step, data->pprev, data->ptable, data->ptablen, data->clast,
            ldc, beam, ldbeam, batch, blank_id, space_id, max_seq_len);

        // --- 3. Merge duplicate prefixes ---
        {
            dim3 merge_grid(beam, batch);
            merge_kernel<<<merge_grid, ldbeam, 0, stream>>>(
                data->select_seq_lens, d_step, data->ptable, data->ptablen, data->clast,
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
                data->select_seq_lens, d_step, data->ptable, data->ptablen, ldc, beam, batch,
                data->topk_key_buffer, data->topk_value_buffer);

            // --- 5. Top-K Phase 2: reduce + state update ---
            constexpr int P2_BLOCK = 128;
            constexpr int P2_IPT = 2;
            int items_per_batch = bxs * beam;
            topk_phase2_kernel<P2_BLOCK, P2_IPT><<<batch, P2_BLOCK, 0, stream>>>(
                data->select_seq_lens, d_step, items_per_batch, beam, batch,
                data->topk_key_buffer, data->topk_value_buffer, ldc, ldbeam, ldseq_len,
                data->pprev, data->ptable, data->ptablen, data->clast, data->clen[src_parity],
                data->clen[dst_parity], data->clist[src_parity], data->clist[dst_parity],
                data->ctime[src_parity], data->ctime[dst_parity],
                data->score, blank_id, data->select_seqs, max_seq_len, d_frame_idx);
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
    int blank_id, int space_id, float blank_threshold, cudaStream_t stream,
    int* out_times = nullptr) {
    int ws_seq_len = max_seq_len > max_out_len ? max_seq_len : max_out_len;

    InternalData data;
    init_internal_data(&data, workspace, batch, beam, vocab_size, ws_seq_len);

    // Initialise beam-state buffers.
    cudaMemsetAsync(data.clast, 0, sizeof(int) * batch * data.ldbeam, stream);
    cudaMemsetAsync(data.clen[0], 0, sizeof(int) * batch * data.ldbeam, stream);
    cudaMemsetAsync(data.clen[1], 0, sizeof(int) * batch * data.ldbeam, stream);
    cudaMemsetAsync(data.clist[0], 0xff, sizeof(int) * batch * beam * data.ldseq_len, stream);
    cudaMemsetAsync(data.clist[1], 0xff, sizeof(int) * batch * beam * data.ldseq_len, stream);
    if (data.ctime[0]) {
        // -1 = "no frame", so a length/time mismatch is visible rather than
        // reading as frame 0.
        cudaMemsetAsync(data.ctime[0], 0xff, sizeof(int) * batch * beam * data.ldseq_len, stream);
        cudaMemsetAsync(data.ctime[1], 0xff, sizeof(int) * batch * beam * data.ldseq_len, stream);
    }
    // ptable and ptablen must start at NEG_INF so that stale entries from
    // previous steps (or the initial state) don't corrupt probability lookups.
    // 0xcc → float bit pattern ≈ -1.7e38, which is effectively -FLT_MAX.
    // (Not allocated when the fused step path is active.)
    if (data.ptable) {
        cudaMemsetAsync(data.ptable, 0xcc, sizeof(float) * batch * beam * data.ldc, stream);
        cudaMemsetAsync(data.ptablen, 0xcc, sizeof(float) * batch * beam * data.ldc, stream);
    }
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

    // The fused path receives the step by value (d_step_dynamic=false), so no
    // per-step device-counter update is needed.  The legacy kernels read
    // ``*d_step``; reuse the otherwise-unused ``ptid`` buffer as the scratch
    // counter (saves a cudaMalloc/cudaFree per decode call) and refresh it
    // with a per-step H2D copy.
    int* d_step_scratch = data.ptid;
    const bool fused_path = step_uses_fused(beam);

    // Main decode loop, tiled by PREPASS_TILE.  Fused path: two launches per
    // tile — the parallel vocab top-K pre-pass (grid = tile_len x batch) and
    // one fused chunk kernel that loops the tile's steps in-kernel (first
    // step included).  Legacy path: one multi-kernel step per frame.
    for (int tile_begin = 0; tile_begin < max_select; tile_begin += PREPASS_TILE) {
        const int tile_len = min(PREPASS_TILE, max_select - tile_begin);
        if (fused_path) {
            cudaError_t perr = launch_fused_topk_prepass(
                &data, log_prob, batch_stride, seq_stride, vocab_stride, data.select_seqs,
                data.select_seq_lens, tile_begin, tile_len, stream);
            if (perr != cudaSuccess)
                return perr;
            cudaError_t cerr = launch_fused_chunk(
                &data, log_prob, batch_stride, seq_stride, vocab_stride,
                /*step_begin=*/tile_begin, /*frame_begin=*/0, /*chunk_frame_begin=*/0,
                tile_len, ~0ull, ~0ull, /*streaming=*/false, blank_id, space_id, stream);
            if (cerr != cudaSuccess)
                return cerr;
            continue;
        }
        for (int step = tile_begin; step < tile_begin + tile_len; ++step) {
            cudaMemcpyAsync(d_step_scratch, &step, sizeof(int), cudaMemcpyHostToDevice, stream);
            bool is_last = (step == max_select - 1);
            cudaError_t err = ctc_prefix_beam_search_step(
                &data, log_prob, batch_stride, seq_stride, vocab_stride, step, is_last, blank_id,
                space_id, stream, d_step_scratch, /*d_step_dynamic=*/false);
            if (err != cudaSuccess)
                return err;
        }
    }

    // Determine which double-buffer holds final results.
    // step=0 → clen[0]/clist[0]; step=N-1 where N>1 → clen[(N-1)%2]/clist[(N-1)%2].
    int final_parity = (max_select <= 1) ? 0 : ((max_select - 1) % 2);

    // Fix batches whose last active step has different parity from final_parity.
    fixup_parity_kernel<<<batch, 32, 0, stream>>>(
        data.select_seq_lens, max_select, data.clen[0], data.clen[1], data.clist[0], data.clist[1],
        data.ctime[0], data.ctime[1], data.ldbeam, data.ldseq_len, beam, batch, final_parity);

    // Copy results to output tensors (strided memcpy).
    cudaMemcpy2DAsync(out_lengths, sizeof(int) * beam, data.clen[final_parity],
                      sizeof(int) * data.ldbeam, sizeof(int) * beam, batch,
                      cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(out_tokens, sizeof(int) * max_out_len, data.clist[final_parity],
                      sizeof(int) * data.ldseq_len, sizeof(int) * max_out_len, batch * beam,
                      cudaMemcpyDeviceToDevice, stream);
    if (out_times && data.ctime[final_parity])
        cudaMemcpy2DAsync(out_times, sizeof(int) * max_out_len, data.ctime[final_parity],
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
    const int* page_storage, const int* time_storage,
    const int* block_table0, const int* block_table1,
    const int* select_seq_lens, int max_select_seq_len,
    const int* clen0, const int* clen1,
    const float* score, int ldbeam,
    int* out_tokens, int* out_times, int* out_lengths, float* out_scores,
    int batch, int beam, int max_out_len,
    int page_size, int max_lp);

inline cudaError_t ctc_prefix_beam_search_step_paged(
    InternalData* data, const float* log_prob,
    int batch_stride, int seq_stride, int vocab_stride,
    int step, int blank_id, int space_id, cudaStream_t stream,
    const int* d_step, bool d_step_dynamic = true, const int* d_frame_idx = nullptr);

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
    // The StreamingState header that used to live at the start of state_buffer
    // is no longer read by the hot-loop step or read-state functions (they
    // reconstruct InternalData pointers on the host from dimensions passed by
    // the caller).  The first 8 bytes are now repurposed for the streaming
    // counters (``device_step_ptr`` / ``device_frame_idx_ptr``); the rest of
    // the STATE_HEADER_SIZE region is still reserved padding.
    StreamingState state;
    state.current_step = 0;
    state.space_id = -1;
    state.blank_id = blank_id;
    state.use_paged_memory = 0;

    void* workspace = reinterpret_cast<char*>(state_buffer) + STATE_HEADER_SIZE;
    init_internal_data(&state.data, workspace, batch, beam, vocab_size, max_seq_len);
    state.data.max_select_seq_len = max_seq_len;

    // Zero the device-resident streaming counters (step / actual_frame_index)
    // so the first ``streaming_step`` call sees step==0 / frame_idx==0 unless
    // the caller writes different start values via ``set_stream_counters``.
    cudaMemsetAsync(state_buffer, 0, sizeof(int) * 2, stream);

    // Initialise GPU buffers.
    cudaMemsetAsync(state.data.clast, 0, sizeof(int) * batch * state.data.ldbeam, stream);
    cudaMemsetAsync(state.data.clen[0], 0, sizeof(int) * batch * state.data.ldbeam, stream);
    cudaMemsetAsync(state.data.clen[1], 0, sizeof(int) * batch * state.data.ldbeam, stream);
    cudaMemsetAsync(state.data.clist[0], 0xff, sizeof(int) * batch * beam * state.data.ldseq_len,
                    stream);
    cudaMemsetAsync(state.data.clist[1], 0xff, sizeof(int) * batch * beam * state.data.ldseq_len,
                    stream);
    if (state.data.ctime[0]) {
        cudaMemsetAsync(state.data.ctime[0], 0xff,
                        sizeof(int) * batch * beam * state.data.ldseq_len, stream);
        cudaMemsetAsync(state.data.ctime[1], 0xff,
                        sizeof(int) * batch * beam * state.data.ldseq_len, stream);
    }
    if (state.data.ptable) {
        cudaMemsetAsync(state.data.ptable, 0xcc, sizeof(float) * batch * beam * state.data.ldc,
                        stream);
        cudaMemsetAsync(state.data.ptablen, 0xcc, sizeof(float) * batch * beam * state.data.ldc,
                        stream);
    }

    // In streaming mode all frames are selected (no blank filtering); set up
    // select_seqs as identity mapping [0, 1, ..., max_seq_len-1] via kernel
    // (no host allocation, no stream sync).
    {
        int total = batch * max_seq_len;
        int threads = 256;
        int blocks = min(1024, (total + threads - 1) / threads);
        if (blocks < 1) blocks = 1;
        init_streaming_select_kernel<<<blocks, threads, 0, stream>>>(
            state.data.select_seqs, state.data.select_seq_lens, batch, max_seq_len);
    }
}

// Reconstruct InternalData pointers on the host from known dimensions. Mirrors
// init_internal_data's bump-allocator layout but performs NO initialisation —
// safe to call on an already-initialised state buffer to avoid a device→host
// sync per streaming step.
inline void setup_internal_data_pointers(InternalData* data, void* workspace, int batch, int beam,
                                         int vocab_size, int max_seq_len) {
    data->batch = batch;
    data->beam = beam;
    data->vocab_size = vocab_size;
    data->ldc = vocab_size;
    data->ldbeam = align16(beam);
    data->ldseq_len = align16(max_seq_len);
    data->max_seq_len = max_seq_len;
    data->ldc_divmod = FastDivmod(vocab_size);
    data->paged = paged_memory::PagedSequenceState();
    data->max_select_seq_len = max_seq_len;

    int ldbeam = data->ldbeam;
    int ldseq_len = data->ldseq_len;
    int ldc = data->ldc;

    char* ptr = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(workspace) + ALIGN_BYTES - 1) /
                                        ALIGN_BYTES * ALIGN_BYTES);

#define ALLOC_BUF(name, type, count)           \
    data->name = reinterpret_cast<type*>(ptr); \
    ptr += align_size(sizeof(type) * (count));

    ALLOC_BUF(pprev, float2, batch * ldbeam)
    if (layout_has_prob_tables(beam)) {
        ALLOC_BUF(ptable, float, batch * beam * ldc)
        ALLOC_BUF(ptablen, float, batch * beam * ldc)
    } else {
        data->ptable = nullptr;
        data->ptablen = nullptr;
    }
    ALLOC_BUF(clast, int, batch* ldbeam)
    ALLOC_BUF(clen[0], int, batch* ldbeam)
    ALLOC_BUF(clen[1], int, batch* ldbeam)
    ALLOC_BUF(clist[0], int, batch * beam * ldseq_len)
    ALLOC_BUF(clist[1], int, batch * beam * ldseq_len)
    ALLOC_BUF(ctime[0], int, batch * beam * ldseq_len)
    ALLOC_BUF(ctime[1], int, batch * beam * ldseq_len)
    ALLOC_BUF(ptid, int, batch* ldbeam)
    ALLOC_BUF(score, float, batch* ldbeam)
    if (layout_has_prob_tables(beam)) {
        ALLOC_BUF(topk_key_buffer, float, batch * MAX_BLOCKS_PER_BATCH * beam)
        ALLOC_BUF(topk_value_buffer, int, batch * MAX_BLOCKS_PER_BATCH * beam)
    } else {
        data->topk_key_buffer = nullptr;
        data->topk_value_buffer = nullptr;
    }
    ALLOC_BUF(select_seqs, int, batch* max_seq_len)
    ALLOC_BUF(select_seq_lens, int, batch)
    ALLOC_BUF(d_lp_frame_buf, float, batch * ldc)
    if (step_uses_fused(beam)) {
        int pstride = fused_prepass_stride(beam);
        ALLOC_BUF(pre_chars, int, PREPASS_TILE * batch * pstride)
        ALLOC_BUF(pre_lp, float, PREPASS_TILE * batch * pstride)
        ALLOC_BUF(pre_cnt, int, PREPASS_TILE * batch)
    } else {
        data->pre_chars = nullptr;
        data->pre_lp = nullptr;
        data->pre_cnt = nullptr;
    }

#undef ALLOC_BUF
}

// Reconstruct paged InternalData pointers on the host from known dimensions.
// NO device memory is touched — safe on an already-initialised paged state.
inline void setup_internal_data_paged_pointers(InternalData* data, void* workspace, int batch,
                                               int beam, int vocab_size, int max_seq_len,
                                               int page_size, int num_pages) {
    data->batch = batch;
    data->beam = beam;
    data->vocab_size = vocab_size;
    data->ldc = vocab_size;
    data->ldbeam = align16(beam);
    data->ldseq_len = align16(max_seq_len);
    data->max_seq_len = max_seq_len;
    data->ldc_divmod = FastDivmod(vocab_size);
    data->max_select_seq_len = max_seq_len;
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
    if (layout_has_prob_tables(beam)) {
        ALLOC_BUF_P(ptable, float, batch * beam * ldc)
        ALLOC_BUF_P(ptablen, float, batch * beam * ldc)
    } else {
        data->ptable = nullptr;
        data->ptablen = nullptr;
    }
    ALLOC_BUF_P(clast, int, batch * ldbeam)
    ALLOC_BUF_P(clen[0], int, batch * ldbeam)
    ALLOC_BUF_P(clen[1], int, batch * ldbeam)
    ALLOC_BUF_P(ptid, int, batch * ldbeam)
    ALLOC_BUF_P(score, float, batch * ldbeam)
    if (layout_has_prob_tables(beam)) {
        ALLOC_BUF_P(topk_key_buffer, float, batch * MAX_BLOCKS_PER_BATCH * beam)
        ALLOC_BUF_P(topk_value_buffer, int, batch * MAX_BLOCKS_PER_BATCH * beam)
    } else {
        data->topk_key_buffer = nullptr;
        data->topk_value_buffer = nullptr;
    }
    ALLOC_BUF_P(select_seqs, int, batch * max_seq_len)
    ALLOC_BUF_P(select_seq_lens, int, batch)
    ALLOC_BUF_P(d_lp_frame_buf, float, batch * ldc)
    if (step_uses_fused(beam)) {
        int pstride = fused_prepass_stride(beam);
        ALLOC_BUF_P(pre_chars, int, PREPASS_TILE * batch * pstride)
        ALLOC_BUF_P(pre_lp, float, PREPASS_TILE * batch * pstride)
        ALLOC_BUF_P(pre_cnt, int, PREPASS_TILE * batch)
    } else {
        data->pre_chars = nullptr;
        data->pre_lp = nullptr;
        data->pre_cnt = nullptr;
    }

#undef ALLOC_BUF_P

    // Re-derive PagedSequenceState pointers from the trailing paged region.
    if (num_pages <= 0)
        num_pages = paged_memory::default_num_pages(batch, beam, max_seq_len, page_size);
    auto& ps = data->paged;
    ps.page_size = page_size;
    ps.max_logical_pages = (max_seq_len + page_size - 1) / page_size;
    ps.num_pages = num_pages;
    ps.batch = batch;
    ps.beam = beam;

    int max_lp = ps.max_logical_pages;
#define PAGED_ALLOC_P(field, type, count)                   \
    ps.field = reinterpret_cast<type*>(ptr);                \
    ptr += paged_memory::paged_align_size(sizeof(type) * (count));

    PAGED_ALLOC_P(page_storage, int, num_pages * page_size)
    // Must mirror ``init_paged_state``'s order exactly: this is a *bump
    // re-derivation* of the same region, so a field missing here does not just
    // leave one pointer null — it shifts every pointer after it.
    PAGED_ALLOC_P(time_storage, int, num_pages * page_size)
    PAGED_ALLOC_P(block_table[0], int, batch * beam * max_lp)
    PAGED_ALLOC_P(block_table[1], int, batch * beam * max_lp)
    PAGED_ALLOC_P(ref_counts, int, num_pages)
    PAGED_ALLOC_P(next_free_page, int, batch)
    PAGED_ALLOC_P(free_pool, int, num_pages)
    PAGED_ALLOC_P(free_pool_size, int, batch)

#undef PAGED_ALLOC_P
}

// Streaming-step launcher that takes all sizes as arguments so the host state
// header never has to be read back with a blocking memcpy. Callers pass the
// same (batch, beam, vocab_size, max_seq_len, ...) they passed to init.
// `use_paged_memory=0` selects the flat path, nonzero the paged path (and then
// page_size/num_pages select the paged layout).
inline cudaError_t streaming_step(void* state_buffer, const float* log_prob_frame, int batch_stride,
                                  int vocab_stride, int step, int blank_id, int space_id,
                                  int actual_frame_index, int batch, int beam, int vocab_size,
                                  int max_seq_len, int use_paged_memory, int page_size,
                                  int num_pages, cudaStream_t stream) {
    void* workspace = reinterpret_cast<char*>(state_buffer) + STATE_HEADER_SIZE;

    InternalData data;
    if (use_paged_memory) {
        setup_internal_data_paged_pointers(&data, workspace, batch, beam, vocab_size, max_seq_len,
                                           page_size, num_pages);
    } else {
        setup_internal_data_pointers(&data, workspace, batch, beam, vocab_size, max_seq_len);
    }

    // Sync the device-resident counters with the host scalar args so kernels
    // that read ``*d_step`` / ``*d_frame_idx`` (currently only the unconditional
    // ``set_select_seq_step_kernel``; later steps will refactor more kernels)
    // see the caller's values. This is the host-int shim path; future
    // device-counter overload skips this and uses the state's current scalars.
    cudaError_t ce = set_stream_counters(state_buffer, step, actual_frame_index, stream);
    if (ce != cudaSuccess)
        return ce;

    // Update select_seqs[b, step] to reflect the caller's actual frame index.
    // Always launched; the kernel itself no-ops when step == frame_idx.
    {
        int threads = 128;
        int blocks = (batch + threads - 1) / threads;
        set_select_seq_step_kernel<<<blocks, threads, 0, stream>>>(
            data.select_seqs, device_step_ptr(state_buffer),
            device_frame_idx_ptr(state_buffer), batch, max_seq_len);
    }

    const int* d_step = device_step_ptr(state_buffer);
    if (use_paged_memory) {
        return ctc_prefix_beam_search_step_paged(&data, log_prob_frame, batch_stride, 0,
                                                 vocab_stride, step, blank_id, space_id, stream,
                                                 d_step, /*d_step_dynamic=*/true,
                                                 device_frame_idx_ptr(state_buffer));
    } else {
        bool is_last = false;
        return ctc_prefix_beam_search_step(&data, log_prob_frame, batch_stride, 0, vocab_stride,
                                           step, is_last, blank_id, space_id, stream, d_step,
                                           /*d_step_dynamic=*/true,
                                           device_frame_idx_ptr(state_buffer));
    }
}

// =============================================================================
// Streaming-step launcher — device-counter / captureable variant
// -----------------------------------------------------------------------------
// Same kernel sequence as ``streaming_step`` but expects the device-resident
// ``d_step`` / ``d_frame_idx`` counters to already hold the current step's
// values. No internal ``set_stream_counters`` write — counter advancement is
// the caller's responsibility (e.g. via ``advance_counters_kernel``).
//
// ``step_for_parity`` is a HOST int used only to pick the parity-dependent
// kernel ptr args (``clen[src/dst]``, ``clist[src/dst]``) and to branch into
// the ``step==0`` ``first_step_kernel`` path.  Each CUDA-Graph capture of
// this function is therefore parity-specific (Step 4 captures three graphs:
// ``step==0``, even-step, odd-step).  Per-frame indexing inside every kernel
// reads from ``*d_step`` so the captured launch is valid across all frames
// that share the same parity.
// =============================================================================

inline cudaError_t streaming_step_persistent(
    void* state_buffer, const float* log_prob_frame, int batch_stride, int vocab_stride,
    int blank_id, int space_id, int batch, int beam, int vocab_size, int max_seq_len,
    int use_paged_memory, int page_size, int num_pages, int step_for_parity,
    cudaStream_t stream) {
    void* workspace = reinterpret_cast<char*>(state_buffer) + STATE_HEADER_SIZE;

    InternalData data;
    if (use_paged_memory) {
        setup_internal_data_paged_pointers(&data, workspace, batch, beam, vocab_size, max_seq_len,
                                           page_size, num_pages);
    } else {
        setup_internal_data_pointers(&data, workspace, batch, beam, vocab_size, max_seq_len);
    }

    // Always launch — the kernel itself is a no-op when *d_step == *d_frame_idx.
    {
        int threads = 128;
        int blocks = (batch + threads - 1) / threads;
        set_select_seq_step_kernel<<<blocks, threads, 0, stream>>>(
            data.select_seqs, device_step_ptr(state_buffer),
            device_frame_idx_ptr(state_buffer), batch, max_seq_len);
    }

    const int* d_step = device_step_ptr(state_buffer);
    if (use_paged_memory) {
        return ctc_prefix_beam_search_step_paged(&data, log_prob_frame, batch_stride, 0,
                                                 vocab_stride, step_for_parity, blank_id,
                                                 space_id, stream, d_step,
                                                 /*d_step_dynamic=*/true,
                                                 device_frame_idx_ptr(state_buffer));
    } else {
        bool is_last = false;
        return ctc_prefix_beam_search_step(&data, log_prob_frame, batch_stride, 0, vocab_stride,
                                           step_for_parity, is_last, blank_id, space_id, stream,
                                           d_step, /*d_step_dynamic=*/true,
                                           device_frame_idx_ptr(state_buffer));
    }
}

// Decode one chunk tile for a streaming state on the fused path: one parallel
// vocab top-K pre-pass + one multi-frame fused chunk kernel — no per-frame
// launches, no d_lp_frame_buf copies, no CUDA graphs.  ``log_prob_chunk`` is
// the stream's chunk base; bit r of mask_lo/mask_hi gates chunk frame
// ``tile_begin + r`` (clear = blank-skip); ``step_begin`` / ``frame_begin``
// are the stream's counters at row 0 of the tile.  Caller guarantees
// step_uses_fused(beam).  Used by the chunk launchers in csrc.
inline cudaError_t streaming_decode_chunk_fused(
    void* state_buffer, const float* log_prob_chunk, int batch_stride, int seq_stride,
    int vocab_stride, int tile_begin, int tile_len, int step_begin, int frame_begin,
    unsigned long long mask_lo, unsigned long long mask_hi, int blank_id, int space_id,
    int batch, int beam, int vocab_size, int max_seq_len, int use_paged_memory, int page_size,
    int num_pages, cudaStream_t stream) {
    void* workspace = reinterpret_cast<char*>(state_buffer) + STATE_HEADER_SIZE;
    InternalData data;
    if (use_paged_memory) {
        setup_internal_data_paged_pointers(&data, workspace, batch, beam, vocab_size, max_seq_len,
                                           page_size, num_pages);
    } else {
        setup_internal_data_pointers(&data, workspace, batch, beam, vocab_size, max_seq_len);
    }
    cudaError_t err = launch_fused_topk_prepass(
        &data, log_prob_chunk, batch_stride, seq_stride, vocab_stride, /*select_seqs=*/nullptr,
        /*select_seq_lens=*/nullptr, tile_begin, tile_len, stream, mask_lo, mask_hi);
    if (err != cudaSuccess)
        return err;
    return launch_fused_chunk(&data, log_prob_chunk, batch_stride, seq_stride, vocab_stride,
                              step_begin, frame_begin, /*chunk_frame_begin=*/tile_begin,
                              tile_len, mask_lo, mask_hi, /*streaming=*/true, blank_id, space_id,
                              stream);
}

// Decode one chunk for N streams (shared config) with grouped launches: per
// group of up to FusedStreamGroup::CAP streams and PREPASS_TILE tile, ONE
// batched pre-pass + ONE batched chunk kernel.  Streams of a group decode
// concurrently (grid = group x batch blocks) instead of as a serial
// per-stream launch chain — states are fully independent (own buffers, own
// per-row paged-allocator partitions), so concurrency cannot change results.
//
// ``state_ptrs`` / ``masks`` / ``steps`` / ``frame_idxs`` are HOST arrays
// (steps / frame_idxs are updated in place, mirroring the per-stream path).
// State pointers must be distinct and share the buffer layout; the per-stream
// byte deltas must be ALIGN_BYTES-multiples for the shared-offset trick
// (torch CUDA allocations are 512-byte aligned, so this always holds for
// states allocated by the Python wrapper); other streams fall back to the
// single-state path.  Caller guarantees step_uses_fused(beam).
inline cudaError_t streaming_decode_chunk_fused_batched(
    const int64_t* state_ptrs, int n_streams, const float* log_prob, int batch_stride,
    int seq_stride, int vocab_stride, int chunk_t, const uint8_t* mask_data, int mask_stride0,
    int* steps, int* frame_idxs, int blank_id, int space_id, int batch, int beam,
    int vocab_size, int max_seq_len, int use_paged_memory, int page_size, int num_pages,
    cudaStream_t stream) {
    using fused::FusedStreamGroup;
    for (int g0 = 0; g0 < n_streams; g0 += FusedStreamGroup::CAP) {
        const int gn = min(FusedStreamGroup::CAP, n_streams - g0);

        // Group prototype: the first stream's pointers; every other stream's
        // state sits at a constant byte delta (identical layout).
        void* base0 = reinterpret_cast<void*>(state_ptrs[g0]);
        void* workspace = reinterpret_cast<char*>(base0) + STATE_HEADER_SIZE;
        InternalData data;
        if (use_paged_memory) {
            setup_internal_data_paged_pointers(&data, workspace, batch, beam, vocab_size,
                                               max_seq_len, page_size, num_pages);
        } else {
            setup_internal_data_pointers(&data, workspace, batch, beam, vocab_size, max_seq_len);
        }

        FusedStreamGroup grp;
        grp.n = gn;
        bool deltas_ok = true;
        for (int s = 0; s < gn; ++s) {
            const long long d = static_cast<long long>(state_ptrs[g0 + s] - state_ptrs[g0]);
            grp.delta[s] = d;
            // The internal-pointer round-up must cancel identically for every
            // stream; deltas that are ALIGN_BYTES-multiples guarantee it.
            if (d % ALIGN_BYTES != 0)
                deltas_ok = false;
        }
        if (!deltas_ok) {
            // Misaligned state buffers (non-torch embedder): per-stream path.
            for (int s = 0; s < gn; ++s) {
                const int i = g0 + s;
                const float* lp_base = log_prob + (size_t)i * batch_stride;
                for (int tile_begin = 0; tile_begin < chunk_t; tile_begin += PREPASS_TILE) {
                    const int tile_len = min(PREPASS_TILE, chunk_t - tile_begin);
                    unsigned long long lo = ~0ull, hi = ~0ull;
                    int n_active = tile_len;
                    if (mask_data) {
                        const uint8_t* m =
                            mask_data + (size_t)i * mask_stride0 + tile_begin;
                        lo = hi = 0;
                        n_active = 0;
                        for (int r = 0; r < tile_len; ++r) {
                            if (!m[r])
                                continue;
                            if (r < 64) {
                                lo |= 1ull << r;
                            } else {
                                hi |= 1ull << (r - 64);
                            }
                            ++n_active;
                        }
                    }
                    if (n_active > 0) {
                        cudaError_t err = streaming_decode_chunk_fused(
                            reinterpret_cast<void*>(state_ptrs[i]), lp_base, batch_stride,
                            seq_stride, vocab_stride, tile_begin, tile_len, steps[i],
                            frame_idxs[i], lo, hi, blank_id, space_id, batch, beam,
                            vocab_size, max_seq_len, use_paged_memory, page_size, num_pages,
                            stream);
                        if (err != cudaSuccess)
                            return err;
                        steps[i] += n_active;
                    }
                    frame_idxs[i] += tile_len;
                }
            }
            continue;
        }

        const float* lp_group = log_prob + (size_t)g0 * batch_stride;
        for (int tile_begin = 0; tile_begin < chunk_t; tile_begin += PREPASS_TILE) {
            const int tile_len = min(PREPASS_TILE, chunk_t - tile_begin);
            bool any_active = false;
            for (int s = 0; s < gn; ++s) {
                unsigned long long lo = ~0ull, hi = ~0ull;
                int n_active = tile_len;
                if (mask_data) {
                    const uint8_t* m =
                        mask_data + (size_t)(g0 + s) * mask_stride0 + tile_begin;
                    lo = hi = 0;
                    n_active = 0;
                    for (int r = 0; r < tile_len; ++r) {
                        if (!m[r])
                            continue;
                        if (r < 64) {
                            lo |= 1ull << r;
                        } else {
                            hi |= 1ull << (r - 64);
                        }
                        ++n_active;
                    }
                }
                grp.mask_lo[s] = lo;
                grp.mask_hi[s] = hi;
                grp.step_begin[s] = steps[g0 + s];
                grp.frame_begin[s] = frame_idxs[g0 + s];
                steps[g0 + s] += n_active;
                frame_idxs[g0 + s] += tile_len;
                any_active = any_active || (n_active > 0);
            }
            if (!any_active)
                continue;
            cudaError_t err =
                launch_fused_chunk_batched(&data, lp_group, batch_stride, seq_stride,
                                           vocab_stride, grp, tile_begin, tile_len, blank_id,
                                           space_id, stream);
            if (err != cudaSuccess)
                return err;
        }
    }
    return cudaSuccess;
}

// ``out_times`` (optional, may be null) receives the encoder frame each token
// was emitted at, in the same [batch, beam, max_out_len] layout as
// ``out_tokens`` — the beam recorded it as it decoded, so reading it out is a
// copy rather than a re-derivation.
inline cudaError_t read_streaming_results(void* state_buffer, int* out_tokens, int* out_lengths,
                                          float* out_scores, int max_out_len, int step, int batch,
                                          int beam, int vocab_size, int max_seq_len,
                                          int use_paged_memory, int page_size, int num_pages,
                                          cudaStream_t stream, int* out_times = nullptr) {
    void* workspace = reinterpret_cast<char*>(state_buffer) + STATE_HEADER_SIZE;

    InternalData data;
    if (use_paged_memory) {
        setup_internal_data_paged_pointers(&data, workspace, batch, beam, vocab_size, max_seq_len,
                                           page_size, num_pages);
    } else {
        setup_internal_data_pointers(&data, workspace, batch, beam, vocab_size, max_seq_len);
    }

    int final_parity = (step <= 1) ? 0 : ((step - 1) % 2);

    if (use_paged_memory) {
        auto& ps = data.paged;
        dim3 gather_grid(batch, beam);
        gather_paged_results_kernel<<<gather_grid, 256, 0, stream>>>(
            ps.page_storage, ps.time_storage, ps.block_table[0], ps.block_table[1],
            data.select_seq_lens, step,
            data.clen[0], data.clen[1],
            data.score, data.ldbeam,
            out_tokens, out_times, out_lengths, out_scores,
            batch, beam, max_out_len,
            ps.page_size, ps.max_logical_pages);
        return cudaGetLastError();
    }

    cudaMemcpy2DAsync(out_lengths, sizeof(int) * beam, data.clen[final_parity],
                      sizeof(int) * data.ldbeam, sizeof(int) * beam, batch,
                      cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(out_tokens, sizeof(int) * max_out_len, data.clist[final_parity],
                      sizeof(int) * data.ldseq_len, sizeof(int) * max_out_len, batch * beam,
                      cudaMemcpyDeviceToDevice, stream);
    if (out_times && data.ctime[final_parity])
        cudaMemcpy2DAsync(out_times, sizeof(int) * max_out_len, data.ctime[final_parity],
                          sizeof(int) * data.ldseq_len, sizeof(int) * max_out_len, batch * beam,
                          cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(out_scores, sizeof(float) * beam, data.score, sizeof(float) * data.ldbeam,
                      sizeof(float) * beam, batch, cudaMemcpyDeviceToDevice, stream);

    return cudaGetLastError();
}

// =============================================================================
// Paged kernel: first step — write initial tokens into pre-allocated page 0
//
// Each (batch, beam) pair has physical page = bid * beam + k pre-allocated in
// block_table[0]. The initial token is written directly to page_storage.
// =============================================================================

template <int BLOCK_SIZE, int ITEMS_PER_THREAD>
__global__ __launch_bounds__(BLOCK_SIZE) void first_step_paged_kernel(
    const float* __restrict__ log_prob, int batch_stride, int seq_stride, int vocab_stride,
    const int* __restrict__ select_seqs, const int* __restrict__ select_seq_lens,
    float2* __restrict__ pprev, int* __restrict__ clast, int* __restrict__ clen,
    float* __restrict__ score, int* __restrict__ page_storage,
    int* __restrict__ time_storage, int page_size,
    int pages_per_row, int beam, int ldbeam,
    int vocab_size, int blank_id, int batch, int max_seq_len) {
    const int bid = blockIdx.x;
    if (bid >= batch)
        return;
    if (select_seq_lens[bid] == 0)
        return;

    const int first_t = select_seqs[bid * max_seq_len];
    const int nb_beams = (beam > 1) ? beam - 1 : beam;
    const int tx = threadIdx.x;

    typedef cub::BlockRadixSort<float, BLOCK_SIZE, ITEMS_PER_THREAD, int> BlockSortT;
    __shared__ union {
        typename BlockSortT::TempStorage temp_storage;
        struct {
            float keys[128];
            int vals[128];
        } topk;
    } smem;

    float keys[ITEMS_PER_THREAD];
    int values[ITEMS_PER_THREAD];

    const int items_per_iter = BLOCK_SIZE * ITEMS_PER_THREAD;
    const int lp_base = bid * batch_stride + first_t * seq_stride;

#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        int c = BLOCK_SIZE * ITEM + tx;
        if (c < vocab_size && c != blank_id) {
            keys[ITEM] = log_prob[lp_base + c * vocab_stride];
            values[ITEM] = c;
        } else {
            keys[ITEM] = NEG_INF;
            values[ITEM] = -1;
        }
    }
    BlockSortT{smem.temp_storage}.SortDescendingBlockedToStriped(keys, values);
    __syncthreads();

    const int stride = items_per_iter - nb_beams;
    for (int offset = items_per_iter; offset < vocab_size; offset += stride) {
#pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
            int striped_pos = BLOCK_SIZE * ITEM + tx;
            int new_local = striped_pos - nb_beams;
            if (new_local >= 0) {
                int c = offset + new_local;
                if (c < vocab_size && c != blank_id) {
                    keys[ITEM] = log_prob[lp_base + c * vocab_stride];
                    values[ITEM] = c;
                } else {
                    keys[ITEM] = NEG_INF;
                    values[ITEM] = -1;
                }
            }
        }
        BlockSortT{smem.temp_storage}.SortDescendingBlockedToStriped(keys, values);
        __syncthreads();
    }

#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        int striped_pos = BLOCK_SIZE * ITEM + tx;
        if (striped_pos < nb_beams) {
            smem.topk.keys[striped_pos] = keys[ITEM];
            smem.topk.vals[striped_pos] = values[ITEM];
        }
    }
    __syncthreads();

    for (int k = tx; k < nb_beams; k += BLOCK_SIZE) {
        int base = bid * ldbeam + k;
        int token = smem.topk.vals[k];
        float key = smem.topk.keys[k];
        if (token >= 0 && token != blank_id) {
            // Prefix [token] ends in a non-blank: mass goes in the non-blank
            // slot (see flat first_step_kernel for the full rationale).
            pprev[base] = make_float2(NEG_INF, key);
            int phys = bid * pages_per_row + k;
            page_storage[phys * page_size + 0] = token;
            if (time_storage)
                time_storage[phys * page_size + 0] = first_t;
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

    if (beam > 1 && tx == 0) {
        int base = bid * ldbeam + (beam - 1);
        float blank_prob = log_prob[lp_base + blank_id * vocab_stride];
        pprev[base] = make_float2(blank_prob, NEG_INF);
        clast[base] = blank_id;
        clen[base] = 0;
        score[base] = blank_prob;
    }
}

// =============================================================================
// Paged kernel: merge duplicate prefixes
//
// Same logic as merge_kernel but reads sequences via paged_seq_compare.
// =============================================================================

__global__ void merge_paged_kernel(
    const int* __restrict__ select_seq_lens, const int* __restrict__ d_step,
    float* __restrict__ ptable, float* __restrict__ ptablen,
    const int* __restrict__ clast, const int* __restrict__ clen,
    int ldc, int beam, int ldbeam, int batch, int blank_id,
    const int* __restrict__ page_storage, const int* __restrict__ block_table,
    int page_size, int max_lp) {
    const int bid = blockIdx.y;
    if (bid >= batch)
        return;
    int step = __ldg(d_step);
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
    const int* __restrict__ select_seq_lens, const int* __restrict__ d_step,
    int items_per_batch, int beam, int batch,
    float* __restrict__ topk_key_buffer, int* __restrict__ topk_value_buffer,
    int ldc, int ldbeam,
    float2* __restrict__ pprev,
    const float* __restrict__ ptable, const float* __restrict__ ptablen,
    int* __restrict__ clast,
    int* __restrict__ clen_src, int* __restrict__ clen_dst,
    float* __restrict__ score,
    int blank_id, const int* __restrict__ select_seqs, int max_seq_len,
    const int* __restrict__ d_frame_idx,
    int* __restrict__ page_storage, int* __restrict__ time_storage,
    int* __restrict__ block_table_src, int* __restrict__ block_table_dst,
    int* __restrict__ ref_counts, int* __restrict__ next_free_page,
    int* __restrict__ free_pool, int* __restrict__ free_pool_size,
    int page_size, int max_lp, int pages_per_row) {
    const int bid = blockIdx.x;
    if (bid >= batch)
        return;
    int step = __ldg(d_step);
    if (step >= select_seq_lens[bid])
        return;
    // The **absolute** frame this step is decoding.  ``select_seqs`` is a ring
    // of width max_seq_len and a stream decodes more frames than its
    // output-token cap, so reading the frame back from it wraps; offline has no
    // device counter and the ring entry is the true frame there.
    const int emit_frame =
        d_frame_idx ? __ldg(d_frame_idx) : select_seqs[bid * max_seq_len + step % max_seq_len];

    // Row-local allocator views (see the alloc_page/free_page comment).
    free_pool += bid * pages_per_row;
    free_pool_size += bid;
    next_free_page += bid;

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

        // Output-token capacity = max_lp logical pages.  A stream may decode
        // more frames than it can emit tokens; cap clen at the capacity so the
        // last logical page index never exceeds max_lp (block_table OOB) and
        // gather/seq_compare stay in range.
        const int out_cap = max_lp * page_size;
        if (char_id == blank_id) {
            clast[dst_base] = smem.topk.src_clast[src_beam];
            clen_dst[dst_base] = prevlen;
        } else if (prevlen >= out_cap) {
            // Output cap reached: keep the prefix but stop appending tokens.
            clast[dst_base] = char_id;
            clen_dst[dst_base] = out_cap;
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
                if (time_storage)
                    time_storage[new_phys * page_size + 0] = emit_frame;
            } else {
                // Writing within an existing page: CoW if shared.
                int bt_idx = bk_dst * max_lp + last_lp;
                int phys = block_table_dst[bt_idx];
                if (ref_counts[phys] > 1) {
                    // Copy-on-write: allocate a fresh/recycled page and copy.
                    int new_phys = alloc_page(free_pool, free_pool_size, next_free_page);
                    for (int i = 0; i < page_size; ++i) {
                        page_storage[new_phys * page_size + i] =
                            page_storage[phys * page_size + i];
                        if (time_storage)
                            time_storage[new_phys * page_size + i] =
                                time_storage[phys * page_size + i];
                    }
                    block_table_dst[bt_idx] = new_phys;
                    // Release one reference from the old page (the bt_dst slot).
                    free_page(phys, free_pool, free_pool_size, ref_counts);
                    ref_counts[new_phys] = 1;
                    phys = new_phys;
                }
                page_storage[phys * page_size + off] = char_id;
                if (time_storage)
                    time_storage[phys * page_size + off] = emit_frame;
            }
        }

        score[dst_base] = new_score;

        // Outgoing (blank, nonblank) split straight from ptable/ptablen — the
        // incoming blank-frame collapse is already baked in upstream.  See the
        // flat topk_phase2_kernel for why we must NOT force {new_score,
        // NEG_INF} on need_add_blank steps (it duplicates repeated tokens).
        float p = ptable[bid * ldc * beam + id];
        float pn = ptablen[bid * ldc * beam + id];
        pprev[dst_base] = make_float2(p, pn);
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
    const int* __restrict__ page_storage, const int* __restrict__ time_storage,
    const int* __restrict__ block_table0, const int* __restrict__ block_table1,
    const int* __restrict__ select_seq_lens, int max_select_seq_len,
    const int* __restrict__ clen0, const int* __restrict__ clen1,
    const float* __restrict__ score, int ldbeam,
    int* __restrict__ out_tokens, int* __restrict__ out_times,
    int* __restrict__ out_lengths, float* __restrict__ out_scores,
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
        // Same physical page, same offset — the emission frame gathers with the
        // token it belongs to, through the identical block table.
        if (out_times)
            out_times[bid * beam * max_out_len + k * max_out_len + pos] =
                time_storage[phys * page_size + off];
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
    int step, int blank_id, int space_id, cudaStream_t stream,
    const int* d_step, bool d_step_dynamic, const int* d_frame_idx) {
    auto& ps = data->paged;
    int batch = data->batch;
    int beam = data->beam;
    int ldc = data->ldc;
    int ldbeam = data->ldbeam;
    int max_seq_len = data->max_seq_len;

    int src_parity = (step == 0) ? 0 : ((step - 1) % 2);
    int dst_parity = (step == 0) ? 0 : (step % 2);

    if (step_uses_fused(beam)) {
        constexpr int FUSED_BLOCK = 256;
        const int fused_ppr = ps.num_pages / max(batch, 1);
        if (step == 0) {
            fused::fused_first_step_kernel<FUSED_BLOCK, true><<<batch, FUSED_BLOCK, 0, stream>>>(
                log_prob, batch_stride, seq_stride, vocab_stride, data->select_seqs,
                data->select_seq_lens, data->pprev, data->clast, data->clen[0],
                /*clist=*/nullptr, /*ctime=*/nullptr, ps.page_storage, ps.time_storage,
                ps.page_size, fused_ppr, data->score,
                beam, ldbeam, data->ldseq_len, data->vocab_size, blank_id, batch, max_seq_len);
        } else {
            const int step_const = d_step_dynamic ? -1 : step;
#define OASR_LAUNCH_FUSED_STEP_PAGED(BEAM_CAP)                                                \
    fused::fused_step_kernel<FUSED_BLOCK, BEAM_CAP, true><<<batch, FUSED_BLOCK, 0, stream>>>( \
        log_prob, batch_stride, seq_stride, vocab_stride, data->select_seqs,                  \
        data->select_seq_lens, d_step, step_const, data->pprev, data->clast,                  \
        data->clen[src_parity], data->clen[dst_parity], /*clist_src=*/nullptr,                \
        /*clist_dst=*/nullptr, /*ctime_src=*/nullptr, /*ctime_dst=*/nullptr,                   \
        data->score, ldc, beam, ldbeam, data->ldseq_len, batch,                                \
        blank_id, space_id, max_seq_len, ps.page_storage, ps.time_storage,                     \
        ps.block_table[src_parity],                                                            \
        ps.block_table[dst_parity], ps.ref_counts, ps.next_free_page, ps.free_pool,          \
        ps.free_pool_size, ps.page_size, ps.max_logical_pages, fused_ppr)
            if (beam <= 16) {
                OASR_LAUNCH_FUSED_STEP_PAGED(16);
            } else {
                OASR_LAUNCH_FUSED_STEP_PAGED(32);
            }
#undef OASR_LAUNCH_FUSED_STEP_PAGED
        }
        return cudaGetLastError();
    }

    const int pages_per_row = ps.num_pages / max(batch, 1);

    if (step == 0) {
        first_step_paged_kernel<128, 4><<<batch, 128, 0, stream>>>(
            log_prob, batch_stride, seq_stride, vocab_stride,
            data->select_seqs, data->select_seq_lens,
            data->pprev, data->clast, data->clen[0], data->score,
            ps.page_storage, ps.time_storage, ps.page_size, pages_per_row,
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
                data->select_seqs, data->select_seq_lens, d_step,
                data->pprev, data->ptable, data->ptablen, data->clast,
                ldc, beam, ldbeam, batch, blank_id, space_id, max_seq_len);
        }

        // --- Blank / space (same as flat) ---
        prob_space_blank_kernel<<<batch, ldbeam, 0, stream>>>(
            log_prob, batch_stride, seq_stride, vocab_stride,
            data->select_seqs, data->select_seq_lens, d_step,
            data->pprev, data->ptable, data->ptablen, data->clast,
            ldc, beam, ldbeam, batch, blank_id, space_id, max_seq_len);

        // --- Merge duplicates (paged) ---
        {
            dim3 merge_grid(beam, batch);
            merge_paged_kernel<<<merge_grid, ldbeam, 0, stream>>>(
                data->select_seq_lens, d_step,
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
                data->select_seq_lens, d_step, data->ptable, data->ptablen,
                ldc, beam, batch, data->topk_key_buffer, data->topk_value_buffer);
        }

        // --- Top-K Phase 2 (paged) ---
        {
            constexpr int P2_BLOCK = 128;
            constexpr int P2_IPT = 2;
            int items_per_batch = bxs * beam;
            topk_phase2_paged_kernel<P2_BLOCK, P2_IPT><<<batch, P2_BLOCK, 0, stream>>>(
                data->select_seq_lens, d_step, items_per_batch, beam, batch,
                data->topk_key_buffer, data->topk_value_buffer,
                ldc, ldbeam,
                data->pprev, data->ptable, data->ptablen, data->clast,
                data->clen[src_parity], data->clen[dst_parity],
                data->score,
                blank_id, data->select_seqs, max_seq_len, d_frame_idx,
                ps.page_storage, ps.time_storage,
                ps.block_table[src_parity], ps.block_table[dst_parity],
                ps.ref_counts, ps.next_free_page,
                ps.free_pool, ps.free_pool_size,
                ps.page_size, ps.max_logical_pages, pages_per_row);
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
    cudaStream_t stream, int* out_times = nullptr) {
    int ws_seq_len = (max_seq_len > max_out_len) ? max_seq_len : max_out_len;

    InternalData data;
    init_internal_data_paged(&data, workspace, batch, beam, vocab_size, ws_seq_len,
                             page_size, num_pages, stream);

    // Initialise beam-state buffers (same as flat, minus clist)
    cudaMemsetAsync(data.clast, 0, sizeof(int) * batch * data.ldbeam, stream);
    cudaMemsetAsync(data.clen[0], 0, sizeof(int) * batch * data.ldbeam, stream);
    cudaMemsetAsync(data.clen[1], 0, sizeof(int) * batch * data.ldbeam, stream);
    if (data.ptable) {
        cudaMemsetAsync(data.ptable, 0xcc, sizeof(float) * batch * beam * data.ldc, stream);
        cudaMemsetAsync(data.ptablen, 0xcc, sizeof(float) * batch * beam * data.ldc, stream);
    }
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

    // See ctc_beam_search_decode_batch: ptid doubles as the legacy step
    // counter; the fused path runs two launches per PREPASS_TILE tile (vocab
    // top-K pre-pass + in-kernel multi-step chunk kernel).
    int* d_step_scratch = data.ptid;
    const bool fused_path = step_uses_fused(beam);

    for (int tile_begin = 0; tile_begin < max_select; tile_begin += PREPASS_TILE) {
        const int tile_len = min(PREPASS_TILE, max_select - tile_begin);
        if (fused_path) {
            cudaError_t perr = launch_fused_topk_prepass(
                &data, log_prob, batch_stride, seq_stride, vocab_stride, data.select_seqs,
                data.select_seq_lens, tile_begin, tile_len, stream);
            if (perr != cudaSuccess)
                return perr;
            cudaError_t cerr = launch_fused_chunk(
                &data, log_prob, batch_stride, seq_stride, vocab_stride,
                /*step_begin=*/tile_begin, /*frame_begin=*/0, /*chunk_frame_begin=*/0,
                tile_len, ~0ull, ~0ull, /*streaming=*/false, blank_id, space_id, stream);
            if (cerr != cudaSuccess)
                return cerr;
            continue;
        }
        for (int step = tile_begin; step < tile_begin + tile_len; ++step) {
            cudaMemcpyAsync(d_step_scratch, &step, sizeof(int), cudaMemcpyHostToDevice, stream);
            cudaError_t err = ctc_prefix_beam_search_step_paged(
                &data, log_prob, batch_stride, seq_stride, vocab_stride,
                step, blank_id, space_id, stream, d_step_scratch, /*d_step_dynamic=*/false);
            if (err != cudaSuccess)
                return err;
        }
    }

    auto& ps = data.paged;

    // Launch gather kernel to read paged sequences into flat output
    dim3 gather_grid(batch, beam);
    gather_paged_results_kernel<<<gather_grid, 256, 0, stream>>>(
        ps.page_storage, ps.time_storage, ps.block_table[0], ps.block_table[1],
        data.select_seq_lens, max_select,
        data.clen[0], data.clen[1],
        data.score, data.ldbeam,
        out_tokens, out_times, out_lengths, out_scores,
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

    // Header is no longer read; skip the StreamingState memcpy.  Zero the
    // device-resident streaming counters (mirrors the flat init path).
    cudaMemsetAsync(state_buffer, 0, sizeof(int) * 2, stream);

    // Initialise beam-state buffers
    cudaMemsetAsync(state.data.clast, 0, sizeof(int) * batch * state.data.ldbeam, stream);
    cudaMemsetAsync(state.data.clen[0], 0, sizeof(int) * batch * state.data.ldbeam, stream);
    cudaMemsetAsync(state.data.clen[1], 0, sizeof(int) * batch * state.data.ldbeam, stream);
    if (state.data.ptable) {
        cudaMemsetAsync(state.data.ptable, 0xcc, sizeof(float) * batch * beam * state.data.ldc,
                        stream);
        cudaMemsetAsync(state.data.ptablen, 0xcc, sizeof(float) * batch * beam * state.data.ldc,
                        stream);
    }

    // Identity frame mapping via kernel (no host allocation, no stream sync).
    {
        int total = batch * max_seq_len;
        int threads = 256;
        int blocks = min(1024, (total + threads - 1) / threads);
        if (blocks < 1) blocks = 1;
        init_streaming_select_kernel<<<blocks, threads, 0, stream>>>(
            state.data.select_seqs, state.data.select_seq_lens, batch, max_seq_len);
    }
}

}  // namespace ctc_decoder
}  // namespace oasr
