// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// The second half of split-KV: reduce `num_splits` partial attentions into one.
//
// Each split ran the full online softmax over its own contiguous chunk of the
// K range, so it produced a *correct* attention output for that chunk plus the
// log-sum-exp that says how much mass the chunk holds.  Combining them is a
// weighted mean whose weights are those masses renormalised:
//
//     O = sum_s  exp2(lse_s - lse_max) * O_s  /  sum_s exp2(lse_s - lse_max)
//
// which is just the online softmax's rescale step performed once more, one
// level up.  Subtracting `lse_max` first is the same guard as contract 1 in
// `fmha_softmax.h` and for the same reason: it keeps every exponent argument
// at or below zero.
//
// Two edge cases carry real weight:
//
//   * An **empty split** -- one whose K chunk lies past this stream's
//     `seqlen_k` -- writes `lse = -INFINITY`, so `exp2(-inf - lse_max)` is
//     exactly 0 and it contributes nothing.  This is why the grid can be a
//     function of shape alone and never of the per-stream lengths, which is
//     what makes a split launch capturable at all.
//   * **Every** split empty means the row has no live key, and the contract
//     (AGENTS.md rule 10's neighbourhood) says the answer is exactly zero, not
//     NaN.  `sum_w == 0` is the test, and it has to be made before the divide.

#pragma once

#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include "cutlass_fmha_configs.h"

namespace oasr {
namespace attention {

/*! \brief Reduce `(num_splits, B, H, T_q, D)` fp32 partials into `(B, H, T_q, D)`.
 *
 * One CTA per `(m_block, head, batch)`; `kBlockM` rows and `kThreads` threads.
 *
 * \tparam Element     the served dtype the output is cast back to
 * \tparam kBlockM     query rows per CTA -- the same tile the attention pass used, so
 *                     a row is combined by the CTA shaped like the one that
 *                     produced it
 * \tparam kHeadDim    padded value head dim
 */
//
// \note No `__restrict__` on the pointers, deliberately.  `cudafe` writes the
//   host-side registration stub as C, and for a `__restrict__` parameter it
//   spells the element type using whatever alias the calling translation unit
//   declared -- which here is an *anonymous-namespace* `using Element = ...`,
//   so it emits that alias's mangled internal-linkage name into a C file and
//   the stub fails to parse.  The aliasing hint is worth little in a kernel
//   whose loads are already independent by construction.
template <class Element, int kBlockM, int kHeadDim, int kThreads>
__global__ void fmha_combine_kernel(Element* out, float const* o_partial,
                                    float const* lse_partial, int const num_splits,
                                    int const seqlen_q, int const head_dim, int const num_heads,
                                    int const batch, int64_t const stride_o_row,
                                    int64_t const stride_o_head, int64_t const stride_o_batch) {
    int const m_block = int(blockIdx.x);
    int const bidh = int(blockIdx.y);
    int const bidb = int(blockIdx.z);
    int const tid = int(threadIdx.x);

    // Per-row reduction state.  Holding `lse_max` and `1/sum` rather than the
    // whole `(num_splits, kBlockM)` weight matrix keeps shared memory at two
    // floats per row whatever `num_splits` is -- the weights are re-derived in
    // the accumulation pass from an LSE read that is L1-resident by then.
    __shared__ float s_lse_max[kBlockM];
    __shared__ float s_inv_sum[kBlockM];

    int64_t const plane = int64_t(seqlen_q) * head_dim;
    int64_t const lse_stride_split = int64_t(batch) * num_heads * seqlen_q;
    int64_t const o_stride_split = int64_t(batch) * num_heads * plane;
    float const* lse_bh =
        lse_partial + (int64_t(bidb) * num_heads + bidh) * int64_t(seqlen_q);
    float const* o_bh = o_partial + (int64_t(bidb) * num_heads + bidh) * plane;

    // --- pass 1: per-row max and normaliser -------------------------------
    for (int r = tid; r < kBlockM; r += kThreads) {
        int const row = m_block * kBlockM + r;
        float lse_max = -INFINITY;
        if (row < seqlen_q) {
            for (int s = 0; s < num_splits; ++s) {
                float const l = lse_bh[s * lse_stride_split + row];
                lse_max = l > lse_max ? l : lse_max;
            }
        }
        float sum_w = 0.f;
        if (row < seqlen_q && lse_max != -INFINITY) {
            for (int s = 0; s < num_splits; ++s) {
                sum_w += ::exp2f(lse_bh[s * lse_stride_split + row] - lse_max);
            }
        }
        s_lse_max[r] = lse_max;
        // A row with no live key in any split: leave the normaliser at zero so
        // the accumulation pass writes zeros rather than dividing by it.
        s_inv_sum[r] = sum_w > 0.f ? 1.0f / sum_w : 0.0f;
    }
    __syncthreads();

    // --- pass 2: the weighted sum -----------------------------------------
    // Thread `tid` owns a strided set of `(row, col)` positions, so the loads
    // from one split are contiguous across the warp.
    int const total = kBlockM * head_dim;
    for (int idx = tid; idx < total; idx += kThreads) {
        int const r = idx / head_dim;
        int const col = idx - r * head_dim;
        int const row = m_block * kBlockM + r;
        if (row >= seqlen_q) {
            continue;
        }
        float const lse_max = s_lse_max[r];
        float const inv_sum = s_inv_sum[r];
        float acc = 0.f;
        if (inv_sum > 0.f) {
            for (int s = 0; s < num_splits; ++s) {
                float const w = ::exp2f(lse_bh[s * lse_stride_split + row] - lse_max);
                acc += w * o_bh[s * o_stride_split + int64_t(row) * head_dim + col];
            }
            acc *= inv_sum;
        }
        out[int64_t(bidb) * stride_o_batch + int64_t(bidh) * stride_o_head +
            int64_t(row) * stride_o_row + col] = static_cast<Element>(acc);
    }
}

/*! \brief Launch the combine pass.
 *
 * `head_dim` is the caller's real head dim, not the padded one: the partials
 * were written that wide and the output is that wide.
 */
template <class Element, int kBlockM, int kHeadDim>
cudaError_t run_fmha_combine(Element* out, float const* o_partial, float const* lse_partial,
                             int num_splits, int seqlen_q, int head_dim, int num_heads,
                             int batch, int64_t stride_o_row, int64_t stride_o_head,
                             int64_t stride_o_batch, cudaStream_t stream) {
    constexpr int kThreads = 128;
    dim3 const grid(uint32_t((seqlen_q + kBlockM - 1) / kBlockM), uint32_t(num_heads),
                    uint32_t(batch));
    fmha_combine_kernel<Element, kBlockM, kHeadDim, kThreads><<<grid, kThreads, 0, stream>>>(
        out, o_partial, lse_partial, num_splits, seqlen_q, head_dim, num_heads, batch,
        stride_o_row, stride_o_head, stride_o_batch);
    return cudaGetLastError();
}

}  // namespace attention
}  // namespace oasr
