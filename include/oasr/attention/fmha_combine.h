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

//! Most splits the combine reduces: each lane holds `kFmhaCombineMaxSplits / 32`
//! LSEs.  At least `kFmhaMaxSplits`; the decode kernel may cut more.
inline constexpr int kFmhaCombineMaxSplits = 64;
static_assert(kFmhaCombineMaxSplits % 32 == 0 && kFmhaCombineMaxSplits >= kFmhaMaxSplits,
              "the combine holds whole warps of LSEs and covers every split the lane cuts");

/*! \brief Reduce `(num_splits, B, H, T_q, D)` fp32 partials into `(B, H, T_q, D)`.
 *
 * **One warp per output row**, where a row is one `(batch, head, query)`
 * position and a CTA owns `kThreads / 32` consecutive rows of the flattened
 * `(batch, head, seqlen_q)` row space.
 *
 * That decomposition is the whole design, and it replaces one tiled by the
 * attention pass's `kBlockM`.  Tiling the combine by the *attention's* M tile
 * reproduces the attention's grid -- and the shapes that need splitting are
 * exactly the ones whose attention grid was too small to fill the machine, so
 * the combine inherited the starvation it was launched to cure.  Measured on
 * `B1 H8 Tq256 Tk1024 D64` at 4 splits (sm_120, 170 SMs): the split mainloop
 * ran 8968 ns at 0.75 waves and 29.3 % of compute SOL, and this pass took
 * **22952 ns at 0.02 waves and 1.7 %** -- 2.6x the kernel it reduces.  Per
 * row rather than per M tile, the same work is 512 CTAs instead of 32.
 *
 * The second cost was arithmetic, not occupancy: the weight
 * `exp2f(lse_s - lse_max)` depends only on `(row, split)`, but computing it
 * inside the column loop evaluated it -- and re-read its LSE -- `head_dim`
 * times per `(row, split)`.  Hoisting it out of a **split-outer, column-inner**
 * loop lets the accumulator be a `kHeadDim / 32` register array indexed only
 * by an unrolled loop.  A runtime-indexed local array here would spill the
 * whole thing to local memory, which is why `num_splits` never subscripts a
 * local.
 *
 * The third cost was latency.  Reading the LSEs one split at a time, in two
 * loops, made every row a chain of `2 * num_splits` dependent global loads:
 * at a small batch -- the only place that splits -- that chain *was* the pass
 * (6.1 us at B1 H6 and 12 splits, against 7.6 us for the split mainloop; ncu,
 * base clocks).  Now each lane reads one split's LSE and computes its weight,
 * the weights reach the other lanes by shuffle, and `kAhead` splits' partials
 * are loaded before any is accumulated.
 *
 * The arithmetic *order* is unchanged -- `sum_w` still accumulates over
 * ascending `s` one add at a time, the weighted sum still accumulates over
 * ascending `s` with the same expression, and `inv_sum` is still applied once
 * at the end -- so this produces bit-identical output to both versions it
 * replaces (`tests/kernels/test_fmha_cpp.py` pins the split path).
 *
 * \tparam Element   the served dtype the output is cast back to
 * \tparam kHeadDim  **padded** value head dim; `head_dim` is the real one and
 *                   bounds the store, so `d = 72` runs as 96 predicated to 72
 * \tparam kThreads  CTA width; `kThreads / 32` rows are combined per CTA
 */
//
// \note No `__restrict__` on the pointers, deliberately.  `cudafe` writes the
//   host-side registration stub as C, and for a `__restrict__` parameter it
//   spells the element type using whatever alias the calling translation unit
//   declared -- which here is an *anonymous-namespace* `using Element = ...`,
//   so it emits that alias's mangled internal-linkage name into a C file and
//   the stub fails to parse.  The aliasing hint is worth little in a kernel
//   whose loads are already independent by construction.
template <class Element, int kHeadDim, int kThreads>
__global__ void fmha_combine_kernel(Element* out, float const* o_partial,
                                    float const* lse_partial, int const num_splits,
                                    int const seqlen_q, int const head_dim,
                                    int const num_heads, int const batch,
                                    int64_t const stride_o_row, int64_t const stride_o_head,
                                    int64_t const stride_o_batch) {
    static_assert(kThreads % 32 == 0, "one warp per row");
    static_assert(kHeadDim % 32 == 0, "padded head dim is a multiple of the warp width");
    constexpr int kRowsPerCta = kThreads / 32;
    constexpr int kColsPerLane = kHeadDim / 32;

    int const lane = int(threadIdx.x) % 32;
    int const row = int(blockIdx.x) * kRowsPerCta + int(threadIdx.x) / 32;
    int const total_rows = batch * num_heads * seqlen_q;
    if (row >= total_rows) {
        return;
    }

    int64_t const lse_stride_split = int64_t(batch) * num_heads * seqlen_q;
    int64_t const o_stride_split = lse_stride_split * head_dim;
    float const* lse_row = lse_partial + row;

    // Lane `j` reads split `j` (and `j + 32`): every LSE in one round trip.  The
    // loop this replaces read them one split at a time, twice -- `2 * num_splits`
    // dependent global loads per row, which at small batch *was* the pass
    // (~6 us at 12 splits, more than the split mainloop it reduces).
    constexpr unsigned kFull = 0xffffffffu;
    constexpr int kSlots = kFmhaCombineMaxSplits / 32;
    float lse[kSlots];
    float lse_max = -INFINITY;
    CUTLASS_PRAGMA_UNROLL
    for (int r = 0; r < kSlots; ++r) {
        int const s = r * 32 + lane;
        lse[r] = s < num_splits ? lse_row[int64_t(s) * lse_stride_split] : -INFINITY;
        lse_max = lse[r] > lse_max ? lse[r] : lse_max;
    }
    // A max is exact in any order.
    CUTLASS_PRAGMA_UNROLL
    for (int o = 16; o > 0; o >>= 1) {
        float const other = __shfl_xor_sync(kFull, lse_max, o);
        lse_max = other > lse_max ? other : lse_max;
    }
    // Lane `j`'s weight is `exp2f(lse_j - lse_max)`, the value the old loop
    // recomputed; `weight(s)` hands split `s`'s to every lane.  The register is
    // picked by unrolled compares, never by a runtime subscript -- that would
    // put `w` in local memory.
    float w[kSlots];
    CUTLASS_PRAGMA_UNROLL
    for (int r = 0; r < kSlots; ++r) {
        w[r] = ::exp2f(lse[r] - lse_max);
    }
    auto weight = [&](int s) {
        float pick = w[0];
        CUTLASS_PRAGMA_UNROLL
        for (int r = 1; r < kSlots; ++r) {
            pick = s >= r * 32 ? w[r] : pick;
        }
        return __shfl_sync(kFull, pick, s & 31);
    };

    // Summed in ascending `s`, one add at a time, from zero: the same sequence of
    // fp32 operations as before, so the same bits.
    float sum_w = 0.f;
    if (lse_max != -INFINITY) {
        for (int s = 0; s < num_splits; ++s) {
            sum_w += weight(s);
        }
    }
    // Every split empty means the row saw no live key, and the contract says
    // the answer is exactly zero rather than a divide by zero.
    float const inv_sum = sum_w > 0.f ? 1.0f / sum_w : 0.0f;

    float acc[kColsPerLane];
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < kColsPerLane; ++c) {
        acc[c] = 0.f;
    }
    if (inv_sum > 0.f) {
        // `kAhead` splits' partials are loaded before any is accumulated, so their
        // latencies overlap; the accumulation itself stays in ascending `s` with
        // the same expression, hence the same rounding.
        constexpr int kAhead = 4;
        float const* o_row = o_partial + int64_t(row) * head_dim;
        for (int s0 = 0; s0 < num_splits; s0 += kAhead) {
            float vals[kAhead][kColsPerLane];
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < kAhead; ++j) {
                int const s = s0 + j;
                float const* src = o_row + int64_t(s) * o_stride_split;
                CUTLASS_PRAGMA_UNROLL
                for (int c = 0; c < kColsPerLane; ++c) {
                    int const col = lane + c * 32;
                    vals[j][c] = (s < num_splits && col < head_dim) ? src[col] : 0.f;
                }
            }
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < kAhead; ++j) {
                int const s = s0 + j;
                if (s < num_splits) {
                    float const ws = weight(s);
                    CUTLASS_PRAGMA_UNROLL
                    for (int c = 0; c < kColsPerLane; ++c) {
                        int const col = lane + c * 32;
                        if (col < head_dim) {
                            acc[c] += ws * vals[j][c];
                        }
                    }
                }
            }
        }
    }

    // `row` is flat over `(batch, head, query)`; the output carries arbitrary
    // strides, so it has to be decomposed.  Twice per warp, not per element.
    int const m = row % seqlen_q;
    int const bh = row / seqlen_q;
    Element* dst = out + int64_t(bh / num_heads) * stride_o_batch +
                   int64_t(bh % num_heads) * stride_o_head + int64_t(m) * stride_o_row;
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < kColsPerLane; ++c) {
        int const col = lane + c * 32;
        if (col < head_dim) {
            dst[col] = static_cast<Element>(acc[c] * inv_sum);
        }
    }
}

/*! \brief Launch the combine pass.
 *
 * `head_dim` is the caller's real head dim, not the padded one: the partials
 * were written that wide and the output is that wide.  The grid is sized from
 * the **row** count, never from the attention pass's tile -- see the kernel.
 */
template <class Element, int kHeadDim>
cudaError_t run_fmha_combine(Element* out, float const* o_partial, float const* lse_partial,
                             int num_splits, int seqlen_q, int head_dim, int num_heads,
                             int batch, int64_t stride_o_row, int64_t stride_o_head,
                             int64_t stride_o_batch, cudaStream_t stream) {
    constexpr int kThreads = 128;
    constexpr int kRowsPerCta = kThreads / 32;
    int const total_rows = batch * num_heads * seqlen_q;
    if (total_rows <= 0) {
        return cudaSuccess;
    }
    if (num_splits < 1 || num_splits > kFmhaCombineMaxSplits) {
        return cudaErrorInvalidValue;
    }
    dim3 const grid(uint32_t((total_rows + kRowsPerCta - 1) / kRowsPerCta));
    fmha_combine_kernel<Element, kHeadDim, kThreads><<<grid, kThreads, 0, stream>>>(
        out, o_partial, lse_partial, num_splits, seqlen_q, head_dim, num_heads, batch,
        stride_o_row, stride_o_head, stride_o_batch);
    return cudaGetLastError();
}

}  // namespace attention
}  // namespace oasr
