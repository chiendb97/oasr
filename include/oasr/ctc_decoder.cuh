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

namespace oasr {
namespace ctc_decoder {

// =============================================================================
// Constants
// =============================================================================

static constexpr int ALIGN_BYTES = 128;
static constexpr int MAX_BLOCKS = 800;
static constexpr float NEG_INF = -1e30f;

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
    // Find the shift amount
    unsigned int p = 31;
    while ((1u << p) < (unsigned int)d) ++p;
    // Compute multiplier: ceil(2^(32+p) / d)
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
    unsigned int t = __uint128_t(multiplier) * (unsigned int)dividend >> 32;
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
  // Dimensions
  int batch;
  int beam;
  int vocab_size;
  int ldc;       // = vocab_size
  int ldbeam;    // 16-aligned beam
  int ldseq_len; // 16-aligned max_seq_len
  int max_seq_len;

  // Double-buffered decoded sequences
  int* clen[2];  // [batch * ldbeam] decoded sequence lengths
  int* clist[2]; // [batch * beam * ldseq_len] decoded token sequences

  // Beam state
  float2* pprev; // [batch * ldbeam] (blank_score, nonblank_score) per beam
  float* ptable; // [batch * beam * ldc] probability table
  float* ptablen; // [batch * beam * ldc] merged probability table
  int* clast;    // [batch * ldbeam] last character in each beam
  int* ptid;     // [batch * ldbeam] prefix tracking id for merging
  float* score;  // [batch * ldbeam] beam scores

  // Top-K buffers
  float* topk_key_buffer;   // [beam * MAX_BLOCKS] sorted keys
  int* topk_value_buffer;   // [beam * MAX_BLOCKS] sorted values

  // Sequence selection (for blank threshold filtering)
  int* select_seqs;      // [batch * max_seq_len] selected sequence indices
  int* select_seq_lens;  // [batch] number of selected sequences per batch

  // FastDivmod for ldc
  FastDivmod ldc_divmod;

  // Current max selected sequence length (computed on init)
  int max_select_seq_len;
};

// =============================================================================
// Device helper functions
// =============================================================================

__inline__ __device__ float logprob_add(float a, float b) { return a + b; }

__inline__ __device__ float logsumexp(float a, float b) {
  if (a == NEG_INF && b == NEG_INF) return NEG_INF;
  float mx = fmaxf(a, b);
  return mx + __log2f(exp2f(a - mx) + exp2f(b - mx)) * 0.6931471805599453f;
  // log2 * ln2 = ln, using log2 for GPU fast math
}

// Compare two integer sequences of given length. Returns true if equal.
__inline__ __device__ bool seq_compare(int len, const int* a, const int* b) {
  for (int i = 0; i < len; ++i) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

// =============================================================================
// Log-probability access helpers
// =============================================================================

// Access log_prob[batch_id, seq_id, char_id] with strides
__inline__ __device__ float log_prob_at(const float* log_prob, int batch_stride,
                                         int seq_stride, int vocab_stride,
                                         int batch_id, int seq_id, int char_id) {
  return log_prob[batch_id * batch_stride + seq_id * seq_stride + char_id * vocab_stride];
}

// =============================================================================
// Kernel: Initialize sequence selection (blank threshold filtering)
// =============================================================================

__global__ void init_select_kernel(const float* __restrict__ log_prob,
                                   int batch_stride, int seq_stride, int vocab_stride,
                                   const int* __restrict__ seq_lengths, int batch, int max_seq_len,
                                   int blank_id, float threshold,
                                   int* __restrict__ select_seqs,
                                   int* __restrict__ select_seq_lens) {
  int bid = blockIdx.x;
  if (bid >= batch) return;

  int actual_len = seq_lengths[bid];
  int count = 0;

  // Each thread processes a range of timesteps
  for (int t = threadIdx.x; t < actual_len; t += blockDim.x) {
    float blank_prob = expf(log_prob[bid * batch_stride + t * seq_stride + blank_id * vocab_stride]);
    if (blank_prob <= threshold) {
      // Atomically add to selection list
      int idx = atomicAdd(&select_seq_lens[bid], 1);
      select_seqs[bid * max_seq_len + idx] = t;
    }
  }
}

// Sort the selected sequences (simple insertion sort, runs per batch on one block)
__global__ void sort_select_kernel(int* __restrict__ select_seqs,
                                   const int* __restrict__ select_seq_lens,
                                   int batch, int max_seq_len) {
  int bid = blockIdx.x * blockDim.x + threadIdx.x;
  if (bid >= batch) return;

  int len = select_seq_lens[bid];
  int* arr = select_seqs + bid * max_seq_len;

  // Simple insertion sort (len is typically small after threshold filtering)
  for (int i = 1; i < len; ++i) {
    int key = arr[i];
    int j = i - 1;
    while (j >= 0 && arr[j] > key) {
      arr[j + 1] = arr[j];
      --j;
    }
    arr[j + 1] = key;
  }
}

// =============================================================================
// Kernel: First step — select initial top-K tokens from vocabulary
// =============================================================================

template <int BLOCK_SIZE>
__global__ void first_step_kernel(const float* __restrict__ log_prob,
                                  int batch_stride, int seq_stride, int vocab_stride,
                                  const int* __restrict__ select_seqs,
                                  const int* __restrict__ select_seq_lens,
                                  float2* __restrict__ pprev, int* __restrict__ clast,
                                  int* __restrict__ clen, int* __restrict__ clist,
                                  float* __restrict__ score,
                                  int beam, int ldbeam, int ldseq_len,
                                  int vocab_size, int blank_id, int batch,
                                  int max_seq_len) {
  int bid = blockIdx.x;
  if (bid >= batch) return;
  if (select_seq_lens[bid] == 0) return;

  int first_t = select_seqs[bid * max_seq_len];

  // Use shared memory for top-K selection
  extern __shared__ char smem[];
  float* s_topk_keys = reinterpret_cast<float*>(smem);
  int* s_topk_vals = reinterpret_cast<int*>(s_topk_keys + beam);

  // Thread 0 performs serial top-K (beam is small, this runs once per utterance)
  if (threadIdx.x == 0) {
    for (int k = 0; k < beam; ++k) {
      s_topk_keys[k] = NEG_INF;
      s_topk_vals[k] = -1;
    }

    for (int c = 0; c < vocab_size; ++c) {
      if (c == blank_id) continue;

      float prob = log_prob[bid * batch_stride + first_t * seq_stride + c * vocab_stride];

      // Find minimum in current top-K
      float min_val = s_topk_keys[0];
      int min_idx = 0;
      for (int k = 1; k < beam; ++k) {
        if (s_topk_keys[k] < min_val) {
          min_val = s_topk_keys[k];
          min_idx = k;
        }
      }

      if (prob > min_val) {
        s_topk_keys[min_idx] = prob;
        s_topk_vals[min_idx] = c;
      }
    }

    // Sort the top-K by score (descending) using insertion sort
    for (int i = 1; i < beam; ++i) {
      float key_i = s_topk_keys[i];
      int val_i = s_topk_vals[i];
      int j = i - 1;
      while (j >= 0 && s_topk_keys[j] < key_i) {
        s_topk_keys[j + 1] = s_topk_keys[j];
        s_topk_vals[j + 1] = s_topk_vals[j];
        --j;
      }
      s_topk_keys[j + 1] = key_i;
      s_topk_vals[j + 1] = val_i;
    }

    // Write results
    float blank_prob = log_prob[bid * batch_stride + first_t * seq_stride +
                                blank_id * vocab_stride];

    for (int k = 0; k < beam; ++k) {
      int base = bid * ldbeam + k;
      if (s_topk_vals[k] >= 0) {
        pprev[base] = make_float2(NEG_INF, s_topk_keys[k]); // (blank, nonblank)
        clast[base] = s_topk_vals[k];
        clen[base] = 1;
        clist[bid * beam * ldseq_len + k * ldseq_len + 0] = s_topk_vals[k];
        score[base] = s_topk_keys[k];
      } else {
        pprev[base] = make_float2(NEG_INF, NEG_INF);
        clast[base] = -1;
        clen[base] = 0;
        score[base] = NEG_INF;
      }
    }

    // The first beam also gets the blank probability as its blank score
    // if blank_prob is significant
    if (beam > 0) {
      int base = bid * ldbeam + 0;
      float nonblank = pprev[base].y;
      pprev[base] = make_float2(blank_prob, nonblank);
      score[base] = logsumexp(blank_prob, nonblank);
    }
  }
}

// =============================================================================
// Kernel: Probability matrix computation (CTC recurrence)
// =============================================================================

__global__ void prob_matrix_kernel(const float* __restrict__ log_prob,
                                   int batch_stride, int seq_stride, int vocab_stride,
                                   const int* __restrict__ select_seqs,
                                   const int* __restrict__ select_seq_lens,
                                   int step,
                                   float2* __restrict__ pprev,
                                   float* __restrict__ ptable,
                                   float* __restrict__ ptablen,
                                   const int* __restrict__ clast,
                                   int ldc, int beam, int ldbeam, int batch,
                                   int blank_id, int max_seq_len,
                                   FastDivmod ldc_divmod) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int bid = blockIdx.y;
  if (bid >= batch) return;
  if (step >= select_seq_lens[bid]) return;

  int t = select_seqs[bid * max_seq_len + step];

  int total_items = ldc * beam;
  if (tid >= total_items) return;

  int beam_idx, char_idx;
  ldc_divmod(beam_idx, char_idx, tid);

  if (beam_idx >= beam) return;
  if (char_idx >= ldc) return;
  if (char_idx == blank_id) return; // blank handled separately

  int pprev_idx = bid * ldbeam + beam_idx;
  float2 prev = pprev[pprev_idx];
  float prev_blank = prev.x;
  float prev_nonblank = prev.y;

  float char_prob = log_prob[bid * batch_stride + t * seq_stride + char_idx * vocab_stride];

  int last_char = clast[pprev_idx];

  float prob;
  if (char_idx == last_char) {
    // Same character as last: only extend from blank path
    // (extending from non-blank would be a repeat, collapsed by CTC)
    prob = logprob_add(prev_blank, char_prob);
  } else {
    // Different character: extend from both blank and non-blank paths
    prob = logprob_add(logsumexp(prev_blank, prev_nonblank), char_prob);
  }

  int table_idx = bid * beam * ldc + beam_idx * ldc + char_idx;
  ptable[table_idx] = prob;
  ptablen[table_idx] = prob; // Will be updated by merge kernel
}

// =============================================================================
// Kernel: Blank and space probability update
// =============================================================================

__global__ void prob_space_blank_kernel(const float* __restrict__ log_prob,
                                        int batch_stride, int seq_stride, int vocab_stride,
                                        const int* __restrict__ select_seqs,
                                        const int* __restrict__ select_seq_lens,
                                        int step,
                                        float2* __restrict__ pprev,
                                        float* __restrict__ ptable,
                                        float* __restrict__ ptablen,
                                        const int* __restrict__ clast,
                                        int ldc, int beam, int ldbeam, int batch,
                                        int blank_id, int space_id, int max_seq_len) {
  int beam_idx = threadIdx.x;
  int bid = blockIdx.x;
  if (bid >= batch || beam_idx >= beam) return;
  if (step >= select_seq_lens[bid]) return;

  int t = select_seqs[bid * max_seq_len + step];
  int pprev_idx = bid * ldbeam + beam_idx;
  float2 prev = pprev[pprev_idx];
  float prev_blank = prev.x;
  float prev_nonblank = prev.y;

  float blank_prob = log_prob[bid * batch_stride + t * seq_stride + blank_id * vocab_stride];

  // Blank: extend from both blank and non-blank
  float new_blank = logprob_add(logsumexp(prev_blank, prev_nonblank), blank_prob);

  // Store blank prob in ptable at blank_id position
  int table_idx = bid * beam * ldc + beam_idx * ldc + blank_id;
  ptable[table_idx] = new_blank;
  ptablen[table_idx] = new_blank;

  // Handle space token if specified
  if (space_id >= 0 && space_id < ldc && space_id != blank_id) {
    float space_prob = log_prob[bid * batch_stride + t * seq_stride + space_id * vocab_stride];
    int last_char = clast[pprev_idx];

    float prob;
    if (space_id == last_char) {
      prob = logprob_add(prev_blank, space_prob);
    } else {
      prob = logprob_add(logsumexp(prev_blank, prev_nonblank), space_prob);
    }

    int space_table_idx = bid * beam * ldc + beam_idx * ldc + space_id;
    ptable[space_table_idx] = prob;
    ptablen[space_table_idx] = prob;
  }
}

// =============================================================================
// Kernel: Merge duplicate prefixes
// =============================================================================

__global__ void merge_kernel(const int* __restrict__ select_seq_lens,
                             int step,
                             float* __restrict__ ptable,
                             float* __restrict__ ptablen,
                             int* __restrict__ ptid,
                             const int* __restrict__ clast,
                             const int* __restrict__ clist,
                             const int* __restrict__ clen,
                             int ldc, int beam, int ldbeam, int ldseq_len,
                             int batch, int blank_id) {
  int bid = blockIdx.x;
  if (bid >= batch) return;
  if (step >= select_seq_lens[bid]) return;

  // Thread 0 per batch handles merging (beam is small)
  if (threadIdx.x != 0) return;

  // Initialize ptid to identity
  for (int k = 0; k < beam; ++k) {
    ptid[bid * ldbeam + k] = k;
  }

  // For each pair of beams, check if they have the same prefix
  // If so, merge probabilities
  for (int i = 0; i < beam; ++i) {
    int len_i = clen[bid * ldbeam + i];
    const int* list_i = clist + bid * beam * ldseq_len + i * ldseq_len;

    for (int j = i + 1; j < beam; ++j) {
      int len_j = clen[bid * ldbeam + j];
      if (len_i != len_j) continue;
      if (len_i == 0) continue;

      const int* list_j = clist + bid * beam * ldseq_len + j * ldseq_len;

      // Compare sequences
      bool same = true;
      for (int s = 0; s < len_i; ++s) {
        if (list_i[s] != list_j[s]) {
          same = false;
          break;
        }
      }

      if (same) {
        // Merge j into i: for each character, merge probabilities
        ptid[bid * ldbeam + j] = i; // j merged into i

        for (int c = 0; c < ldc; ++c) {
          int idx_i = bid * beam * ldc + i * ldc + c;
          int idx_j = bid * beam * ldc + j * ldc + c;
          ptablen[idx_i] = logsumexp(ptablen[idx_i], ptable[idx_j]);
          ptable[idx_j] = NEG_INF; // Invalidate merged beam
          ptablen[idx_j] = NEG_INF;
        }
      }
    }
  }
}

// =============================================================================
// Kernel: Top-K selection and state update
// =============================================================================

// Simple top-K: single block per batch, for beam * ldc candidates.
// Uses shared memory partial sort.
__global__ void topk_and_update_kernel(const int* __restrict__ select_seq_lens,
                                        int step, bool is_last_step,
                                        float2* __restrict__ pprev,
                                        float* __restrict__ ptable,
                                        float* __restrict__ ptablen,
                                        const int* __restrict__ ptid,
                                        int* __restrict__ clast,
                                        int* __restrict__ clen_src,
                                        int* __restrict__ clen_dst,
                                        int* __restrict__ clist_src,
                                        int* __restrict__ clist_dst,
                                        float* __restrict__ score,
                                        int ldc, int beam, int ldbeam, int ldseq_len,
                                        int batch, int blank_id) {
  int bid = blockIdx.x;
  if (bid >= batch) return;
  if (step >= select_seq_lens[bid]) return;

  // Thread 0 does serial top-K (beam and ldc are small enough for decoding)
  if (threadIdx.x != 0) return;

  // Collect top-beam candidates from ptablen across all beams and characters
  // Candidate = (score, beam_src, char_id)
  // We need beam results

  // Temporary storage: top-beam candidates
  // Using registers / local memory (beam <= 128)
  float best_scores[128]; // max beam
  int best_beams[128];
  int best_chars[128];

  for (int k = 0; k < beam; ++k) {
    best_scores[k] = NEG_INF;
    best_beams[k] = -1;
    best_chars[k] = -1;
  }

  // Scan all (beam_src, char) pairs
  for (int b = 0; b < beam; ++b) {
    int merged_to = ptid[bid * ldbeam + b];
    if (merged_to != b) continue; // Skip merged beams

    for (int c = 0; c < ldc; ++c) {
      float val = ptablen[bid * beam * ldc + b * ldc + c];
      if (val <= NEG_INF) continue;

      // Find minimum in current top-K
      float min_val = best_scores[0];
      int min_idx = 0;
      for (int k = 1; k < beam; ++k) {
        if (best_scores[k] < min_val) {
          min_val = best_scores[k];
          min_idx = k;
        }
      }

      if (val > min_val) {
        best_scores[min_idx] = val;
        best_beams[min_idx] = b;
        best_chars[min_idx] = c;
      }
    }
  }

  // Sort results by score (descending)
  for (int i = 1; i < beam; ++i) {
    float key_i = best_scores[i];
    int beam_i = best_beams[i];
    int char_i = best_chars[i];
    int j = i - 1;
    while (j >= 0 && best_scores[j] < key_i) {
      best_scores[j + 1] = best_scores[j];
      best_beams[j + 1] = best_beams[j];
      best_chars[j + 1] = best_chars[j];
      --j;
    }
    best_scores[j + 1] = key_i;
    best_beams[j + 1] = beam_i;
    best_chars[j + 1] = char_i;
  }

  // Update state: write new beams
  for (int k = 0; k < beam; ++k) {
    int dst_base = bid * ldbeam + k;

    if (best_beams[k] < 0) {
      // No valid candidate
      pprev[dst_base] = make_float2(NEG_INF, NEG_INF);
      clast[dst_base] = -1;
      clen_dst[dst_base] = 0;
      score[dst_base] = NEG_INF;
      continue;
    }

    int src_beam = best_beams[k];
    int new_char = best_chars[k];
    float new_score = best_scores[k];
    int src_base = bid * ldbeam + src_beam;

    if (new_char == blank_id) {
      // Blank extension: keep the same prefix, update blank score
      pprev[dst_base] = make_float2(new_score, NEG_INF);
      clast[dst_base] = clast[src_base];

      // Copy prefix from source beam
      int src_len = clen_src[src_base];
      clen_dst[dst_base] = src_len;
      for (int s = 0; s < src_len; ++s) {
        clist_dst[bid * beam * ldseq_len + k * ldseq_len + s] =
            clist_src[bid * beam * ldseq_len + src_beam * ldseq_len + s];
      }
      score[dst_base] = new_score;
    } else {
      // Non-blank extension: append character
      pprev[dst_base] = make_float2(NEG_INF, new_score);
      clast[dst_base] = new_char;

      // Copy prefix from source and append new char
      int src_len = clen_src[src_base];
      int dst_len = src_len + 1;
      if (dst_len > ldseq_len) dst_len = ldseq_len; // Clamp

      for (int s = 0; s < src_len && s < ldseq_len - 1; ++s) {
        clist_dst[bid * beam * ldseq_len + k * ldseq_len + s] =
            clist_src[bid * beam * ldseq_len + src_beam * ldseq_len + s];
      }
      if (src_len < ldseq_len) {
        clist_dst[bid * beam * ldseq_len + k * ldseq_len + src_len] = new_char;
      }
      clen_dst[dst_base] = dst_len;
      score[dst_base] = new_score;
    }
  }
}

// =============================================================================
// Kernel: Copy results for batches with different parity
// =============================================================================

// For batches whose last active step has different parity than the global final,
// copy their results from the "wrong" buffer to the "right" buffer.
__global__ void fixup_parity_kernel(const int* __restrict__ select_seq_lens,
                                     int max_select_seq_len,
                                     int* __restrict__ clen0, int* __restrict__ clen1,
                                     int* __restrict__ clist0, int* __restrict__ clist1,
                                     int ldbeam, int ldseq_len, int beam,
                                     int batch, int final_parity) {
  int bid = blockIdx.x;
  if (bid >= batch) return;
  if (threadIdx.x != 0) return;

  int nsteps = select_seq_lens[bid];
  if (nsteps == 0) return;

  // Determine which buffer this batch's results are in
  int batch_parity;
  if (nsteps == 1) {
    batch_parity = 0; // first_step writes to buffer 0
  } else {
    // Last active step for this batch is (nsteps - 1)
    // That step writes to dst_parity = (nsteps - 1) % 2
    batch_parity = (nsteps - 1) % 2;
  }

  if (batch_parity == final_parity) return; // No fixup needed

  // Copy from batch_parity to final_parity
  int* src_clen = (batch_parity == 0) ? clen0 : clen1;
  int* dst_clen = (final_parity == 0) ? clen0 : clen1;
  int* src_clist = (batch_parity == 0) ? clist0 : clist1;
  int* dst_clist = (final_parity == 0) ? clist0 : clist1;

  for (int k = 0; k < beam; ++k) {
    int idx = bid * ldbeam + k;
    dst_clen[idx] = src_clen[idx];

    int len = src_clen[idx];
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

inline int align16(int val) { return ((val - 1) / 16 + 1) * 16; }

inline size_t calculate_workspace_size(int batch, int beam, int vocab_size, int max_seq_len) {
  int ldbeam = align16(beam);
  int ldseq_len = align16(max_seq_len);
  int ldc = vocab_size;

  size_t total = 0;
  total += align_size(sizeof(float2) * batch * ldbeam);                // pprev
  total += align_size(sizeof(float) * batch * beam * ldc);             // ptable
  total += align_size(sizeof(float) * batch * beam * ldc);             // ptablen
  total += align_size(sizeof(int) * batch * ldbeam);                   // clast
  total += align_size(sizeof(int) * batch * ldbeam) * 2;              // clen[0], clen[1]
  total += align_size(sizeof(int) * batch * beam * ldseq_len) * 2;   // clist[0], clist[1]
  total += align_size(sizeof(int) * batch * ldbeam);                   // ptid
  total += align_size(sizeof(float) * batch * ldbeam);                 // score
  total += align_size(sizeof(float) * beam * MAX_BLOCKS);              // topk_key_buffer
  total += align_size(sizeof(int) * beam * MAX_BLOCKS);                // topk_value_buffer
  total += align_size(sizeof(int) * batch * max_seq_len);              // select_seqs
  total += align_size(sizeof(int) * batch);                            // select_seq_lens
  total += ALIGN_BYTES; // For initial alignment
  return total;
}

inline void init_internal_data(InternalData* data, void* workspace,
                               int batch, int beam, int vocab_size, int max_seq_len) {
  data->batch = batch;
  data->beam = beam;
  data->vocab_size = vocab_size;
  data->ldc = vocab_size;
  data->ldbeam = align16(beam);
  data->ldseq_len = align16(max_seq_len);
  data->max_seq_len = max_seq_len;
  data->ldc_divmod = FastDivmod(vocab_size);

  int ldbeam = data->ldbeam;
  int ldseq_len = data->ldseq_len;
  int ldc = data->ldc;

  // Align the workspace pointer
  char* ptr = reinterpret_cast<char*>(
      (reinterpret_cast<uintptr_t>(workspace) + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES);

#define ALLOC_BUF(name, type, count)                           \
  data->name = reinterpret_cast<type*>(ptr);                   \
  ptr += align_size(sizeof(type) * (count));

  ALLOC_BUF(pprev, float2, batch * ldbeam)
  ALLOC_BUF(ptable, float, batch * beam * ldc)
  ALLOC_BUF(ptablen, float, batch * beam * ldc)
  ALLOC_BUF(clast, int, batch * ldbeam)
  ALLOC_BUF(clen[0], int, batch * ldbeam)
  ALLOC_BUF(clen[1], int, batch * ldbeam)
  ALLOC_BUF(clist[0], int, batch * beam * ldseq_len)
  ALLOC_BUF(clist[1], int, batch * beam * ldseq_len)
  ALLOC_BUF(ptid, int, batch * ldbeam)
  ALLOC_BUF(score, float, batch * ldbeam)
  ALLOC_BUF(topk_key_buffer, float, beam * MAX_BLOCKS)
  ALLOC_BUF(topk_value_buffer, int, beam * MAX_BLOCKS)
  ALLOC_BUF(select_seqs, int, batch * max_seq_len)
  ALLOC_BUF(select_seq_lens, int, batch)

#undef ALLOC_BUF
}

// =============================================================================
// Host launcher: single step of CTC prefix beam search
// =============================================================================

inline cudaError_t ctc_prefix_beam_search_step(
    InternalData* data,
    const float* log_prob, // Full tensor [batch, seq_len, vocab_size]
    int batch_stride, int seq_stride, int vocab_stride,
    int step,
    bool is_last_step,
    int blank_id,
    int space_id,
    cudaStream_t stream) {

  int batch = data->batch;
  int beam = data->beam;
  int ldc = data->ldc;
  int ldbeam = data->ldbeam;
  int ldseq_len = data->ldseq_len;
  int max_seq_len = data->max_seq_len;

  // Determine source/dest buffer parity
  int src_parity = (step % 2) ^ 1;
  int dst_parity = step % 2;

  if (step == 0) {
    // First step: select initial top-K tokens
    int smem_size = sizeof(float) * beam + sizeof(int) * beam;
    first_step_kernel<256><<<batch, 256, smem_size, stream>>>(
        log_prob, batch_stride, seq_stride, vocab_stride,
        data->select_seqs, data->select_seq_lens,
        data->pprev, data->clast,
        data->clen[0], data->clist[0], data->score,
        beam, ldbeam, ldseq_len, data->vocab_size, blank_id, batch,
        max_seq_len);
  } else {
    // Step 1+: prob_matrix -> merge -> topk

    // 1. Compute probability matrix
    int total_items = ldc * beam;
    int threads = 256;
    int blocks_x = (total_items + threads - 1) / threads;
    blocks_x = min(blocks_x, MAX_BLOCKS / max(batch, 1));
    dim3 grid_prob(blocks_x, batch);

    prob_matrix_kernel<<<grid_prob, threads, 0, stream>>>(
        log_prob, batch_stride, seq_stride, vocab_stride,
        data->select_seqs, data->select_seq_lens,
        step,
        data->pprev, data->ptable, data->ptablen, data->clast,
        ldc, beam, ldbeam, batch, blank_id, max_seq_len,
        data->ldc_divmod);

    // 2. Handle blank/space probabilities
    prob_space_blank_kernel<<<batch, beam, 0, stream>>>(
        log_prob, batch_stride, seq_stride, vocab_stride,
        data->select_seqs, data->select_seq_lens,
        step,
        data->pprev, data->ptable, data->ptablen, data->clast,
        ldc, beam, ldbeam, batch, blank_id, space_id, max_seq_len);

    // 3. Merge duplicate prefixes
    merge_kernel<<<batch, 1, 0, stream>>>(
        data->select_seq_lens, step,
        data->ptable, data->ptablen, data->ptid, data->clast,
        data->clist[src_parity], data->clen[src_parity],
        ldc, beam, ldbeam, ldseq_len, batch, blank_id);

    // 4. Top-K selection and state update
    topk_and_update_kernel<<<batch, 1, 0, stream>>>(
        data->select_seq_lens, step, is_last_step,
        data->pprev, data->ptable, data->ptablen, data->ptid,
        data->clast,
        data->clen[src_parity], data->clen[dst_parity],
        data->clist[src_parity], data->clist[dst_parity],
        data->score,
        ldc, beam, ldbeam, ldseq_len, batch, blank_id);
  }

  return cudaGetLastError();
}

// =============================================================================
// Host launcher: full batch decode (offline)
// =============================================================================

inline cudaError_t ctc_beam_search_decode_batch(
    const float* log_prob, // [batch, seq_len, vocab_size]
    int batch_stride, int seq_stride, int vocab_stride,
    const int* seq_lengths, // [batch]
    int* out_tokens,        // [batch, beam, max_out_len] output
    int* out_lengths,       // [batch, beam] output
    float* out_scores,      // [batch, beam] output
    void* workspace,
    int batch, int beam, int vocab_size,
    int max_seq_len, int max_out_len,
    int blank_id, int space_id,
    float blank_threshold,
    cudaStream_t stream) {

  // Use max of input seq len and output seq len for workspace sizing,
  // since select_seqs needs max_seq_len entries and clist needs max_out_len entries.
  int ws_seq_len = max_seq_len > max_out_len ? max_seq_len : max_out_len;

  InternalData data;
  init_internal_data(&data, workspace, batch, beam, vocab_size, ws_seq_len);

  // Zero-initialize state
  cudaMemsetAsync(data.clast, 0, sizeof(int) * batch * data.ldbeam, stream);
  cudaMemsetAsync(data.clen[0], 0, sizeof(int) * batch * data.ldbeam, stream);
  cudaMemsetAsync(data.clen[1], 0, sizeof(int) * batch * data.ldbeam, stream);
  cudaMemsetAsync(data.clist[0], 0xff,
                  sizeof(int) * batch * beam * data.ldseq_len, stream); // -1
  cudaMemsetAsync(data.clist[1], 0xff,
                  sizeof(int) * batch * beam * data.ldseq_len, stream);
  cudaMemsetAsync(data.select_seq_lens, 0, sizeof(int) * batch, stream);

  // Initialize: filter sequences by blank threshold
  // Use data.max_seq_len as stride for select_seqs (consistent with kernels)
  init_select_kernel<<<batch, 128, 0, stream>>>(
      log_prob, batch_stride, seq_stride, vocab_stride,
      seq_lengths, batch, data.max_seq_len,
      blank_id, blank_threshold,
      data.select_seqs, data.select_seq_lens);

  // Sort selected sequences per batch
  int sort_blocks = (batch + 127) / 128;
  sort_select_kernel<<<sort_blocks, 128, 0, stream>>>(
      data.select_seqs, data.select_seq_lens, batch, data.max_seq_len);

  // Sync to read max_select_seq_len on host
  // We need to know the max number of selected steps for the main loop
  int* h_select_lens = new int[batch];
  cudaMemcpyAsync(h_select_lens, data.select_seq_lens,
                  sizeof(int) * batch, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  int max_select = 0;
  for (int b = 0; b < batch; ++b) {
    if (h_select_lens[b] > max_select) max_select = h_select_lens[b];
  }
  delete[] h_select_lens;
  data.max_select_seq_len = max_select;

  // Main decoding loop
  for (int step = 0; step < max_select; ++step) {
    bool is_last = (step == max_select - 1);
    cudaError_t err = ctc_prefix_beam_search_step(
        &data, log_prob, batch_stride, seq_stride, vocab_stride,
        step, is_last, blank_id, space_id, stream);
    if (err != cudaSuccess) return err;
  }

  // Determine final parity
  int final_parity = (max_select > 0) ? ((max_select - 1) % 2) : 0;
  if (max_select == 1) final_parity = 0; // first_step writes to clen[0]/clist[0]

  // Fix up parity for batches with fewer selected steps
  fixup_parity_kernel<<<batch, 1, 0, stream>>>(
      data.select_seq_lens, max_select,
      data.clen[0], data.clen[1],
      data.clist[0], data.clist[1],
      data.ldbeam, data.ldseq_len, beam,
      batch, final_parity);

  // Copy results to output
  // out_lengths: [batch, beam]
  cudaMemcpy2DAsync(out_lengths, sizeof(int) * beam,
                    data.clen[final_parity], sizeof(int) * data.ldbeam,
                    sizeof(int) * beam, batch,
                    cudaMemcpyDeviceToDevice, stream);

  // out_tokens: [batch, beam, max_out_len] from clist [batch * beam, ldseq_len]
  cudaMemcpy2DAsync(out_tokens, sizeof(int) * max_out_len,
                    data.clist[final_parity], sizeof(int) * data.ldseq_len,
                    sizeof(int) * max_out_len, batch * beam,
                    cudaMemcpyDeviceToDevice, stream);

  // out_scores: [batch, beam]
  cudaMemcpy2DAsync(out_scores, sizeof(float) * beam,
                    data.score, sizeof(float) * data.ldbeam,
                    sizeof(float) * beam, batch,
                    cudaMemcpyDeviceToDevice, stream);

  return cudaGetLastError();
}

// =============================================================================
// Host launcher: streaming init/step/read
// =============================================================================

// Layout of streaming state buffer:
// [InternalData struct (padded)] [workspace buffers...]
static constexpr size_t STATE_HEADER_SIZE = align_size(sizeof(InternalData) + sizeof(int) * 4);

inline size_t calculate_state_buffer_size(int batch, int beam, int vocab_size, int max_seq_len) {
  return STATE_HEADER_SIZE + calculate_workspace_size(batch, beam, vocab_size, max_seq_len);
}

// Stored at the beginning of state buffer, after InternalData
struct StreamingState {
  InternalData data;
  int current_step;
  int space_id;
  int blank_id;
  int _pad;
};

inline void init_streaming_state(void* state_buffer,
                                 int batch, int beam, int vocab_size,
                                 int max_seq_len, int blank_id,
                                 cudaStream_t stream) {
  StreamingState state;
  state.current_step = 0;
  state.space_id = -1;
  state.blank_id = blank_id;
  state._pad = 0;

  void* workspace = reinterpret_cast<char*>(state_buffer) + STATE_HEADER_SIZE;
  init_internal_data(&state.data, workspace, batch, beam, vocab_size, max_seq_len);
  state.data.max_select_seq_len = max_seq_len; // For streaming, process all steps

  // Copy StreamingState header to device
  cudaMemcpyAsync(state_buffer, &state, sizeof(StreamingState),
                  cudaMemcpyHostToDevice, stream);

  // Zero-initialize GPU buffers
  cudaMemsetAsync(state.data.clast, 0,
                  sizeof(int) * batch * state.data.ldbeam, stream);
  cudaMemsetAsync(state.data.clen[0], 0,
                  sizeof(int) * batch * state.data.ldbeam, stream);
  cudaMemsetAsync(state.data.clen[1], 0,
                  sizeof(int) * batch * state.data.ldbeam, stream);
  cudaMemsetAsync(state.data.clist[0], 0xff,
                  sizeof(int) * batch * beam * state.data.ldseq_len, stream);
  cudaMemsetAsync(state.data.clist[1], 0xff,
                  sizeof(int) * batch * beam * state.data.ldseq_len, stream);

  // For streaming, select_seq_lens = [max_seq_len] for all batches (no filtering)
  // and select_seqs = [0, 1, 2, ..., max_seq_len - 1] for each batch
  // We set this up so that step indexing works
  int* h_select_lens = new int[batch];
  for (int b = 0; b < batch; ++b) h_select_lens[b] = max_seq_len;
  cudaMemcpyAsync(state.data.select_seq_lens, h_select_lens,
                  sizeof(int) * batch, cudaMemcpyHostToDevice, stream);
  delete[] h_select_lens;

  // select_seqs: identity mapping [0, 1, 2, ...]
  int* h_select_seqs = new int[batch * max_seq_len];
  for (int b = 0; b < batch; ++b)
    for (int t = 0; t < max_seq_len; ++t)
      h_select_seqs[b * max_seq_len + t] = t;
  cudaMemcpyAsync(state.data.select_seqs, h_select_seqs,
                  sizeof(int) * batch * max_seq_len, cudaMemcpyHostToDevice, stream);
  delete[] h_select_seqs;

  cudaStreamSynchronize(stream);
}

inline cudaError_t streaming_step(void* state_buffer,
                                  const float* log_prob_frame, // [batch, vocab_size]
                                  int batch_stride, int vocab_stride,
                                  int step,
                                  int blank_id, int space_id,
                                  cudaStream_t stream) {
  // Read InternalData from device state (we reconstruct pointers on host side)
  StreamingState state;
  cudaMemcpyAsync(&state, state_buffer, sizeof(StreamingState),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  // For streaming: log_prob_frame is [batch, vocab_size], treated as
  // [batch, 1, vocab_size] where the "step" in select_seqs maps to seq_id=step
  // But we pass the frame pointer directly with seq_stride=0 since it's a single frame
  // The select_seqs[bid * max_seq_len + step] = step, so the kernel will access
  // log_prob[bid * batch_stride + step * seq_stride + c * vocab_stride]
  // For a single frame, seq_stride = 0 and we set select_seqs[step] = 0
  // Actually, simpler: pass log_prob_frame with seq_stride=0
  // and have select_seqs[*][step] = 0

  bool is_last = false; // In streaming, we don't know if it's the last step
  return ctc_prefix_beam_search_step(
      &state.data, log_prob_frame,
      batch_stride, 0, vocab_stride, // seq_stride=0 since it's a single frame
      step, is_last, blank_id, space_id, stream);
}

inline cudaError_t read_streaming_results(
    void* state_buffer,
    int* out_tokens, int* out_lengths, float* out_scores,
    int max_out_len,
    cudaStream_t stream) {

  StreamingState state;
  cudaMemcpyAsync(&state, state_buffer, sizeof(StreamingState),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  InternalData* data = &state.data;
  int batch = data->batch;
  int beam = data->beam;

  // Determine parity from current step
  int step = state.current_step;
  int final_parity = (step > 1) ? ((step - 1) % 2) : 0;

  cudaMemcpy2DAsync(out_lengths, sizeof(int) * beam,
                    data->clen[final_parity], sizeof(int) * data->ldbeam,
                    sizeof(int) * beam, batch,
                    cudaMemcpyDeviceToDevice, stream);

  cudaMemcpy2DAsync(out_tokens, sizeof(int) * max_out_len,
                    data->clist[final_parity], sizeof(int) * data->ldseq_len,
                    sizeof(int) * max_out_len, batch * beam,
                    cudaMemcpyDeviceToDevice, stream);

  cudaMemcpy2DAsync(out_scores, sizeof(float) * beam,
                    data->score, sizeof(float) * data->ldbeam,
                    sizeof(float) * beam, batch,
                    cudaMemcpyDeviceToDevice, stream);

  return cudaGetLastError();
}

}  // namespace ctc_decoder
}  // namespace oasr
