// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// One frame of the transducer's modified beam search after the joiner, fused:
// log-softmax of every hypothesis's logits, the score add, the top-k over the
// beam's k * V candidates, the split into (parent, label), the masking of rows
// past their utterance, and the reorder of the predictor label windows onto the
// new beam.  The torch composition of the same frame is about fourteen launches
// (float cast, log_softmax, add, topk's select and sort, div/mul/sub, a gather,
// a cat and five wheres), each a few microseconds of work on a (B * k, V) grid.
//
// Exactness.  The scores are bit-identical to that composition: the log-softmax
// reproduces torch's `softmax_warp_forward` -- the kernel torch dispatches for a
// row of at most 1024 floats -- element for element: the same lane-strided
// layout and padding, the same per-lane sequential max and sum, the same xor
// butterflies, and libdevice's precise `__nv_expf` / `__nv_logf`, called by name
// because `--use_fast_math` maps `expf` / `logf` to approximate intrinsics.  The
// log-prob is `(x - max) - log(sum)` and the total `score + log_prob`, the same
// two roundings torch takes.
//
// Selection order.  Candidates are ranked by total, higher first, and equal
// totals by flat index `j * V + v`, lower first: a total order, so the result
// does not depend on scheduling.  torch.topk keeps the same *set* -- values above
// the k-th, then values equal to it in index order -- but orders ties with an
// unstable bitonic network, so a tie inside the selected k can come out in a
// different slot order here.  Distinct totals select and order identically.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <cstdint>

// libdevice's precise exp / log (see the exactness note above).
extern "C" __device__ float __nv_expf(float);
extern "C" __device__ float __nv_logf(float);

namespace oasr {
namespace transducer {

//: Widest beam: the k x k candidate grid lives in shared memory.
constexpr int kBeamTopkMaxBeam = 32;
//: Largest vocabulary, as log2: torch's warp log-softmax covers 1024 floats.
constexpr int kBeamTopkMaxLog2Vocab = 10;
constexpr int kBeamTopkWarps = 8;

template <typename T>
struct BeamTopkParams {
    const T* logits;         // (B * k, ld): the first V columns are the vocabulary
    const float* scores;     // (B, k)
    const int64_t* context;  // (B, k, ctx) predictor label windows
    const bool* active;      // (B,): frame still inside the utterance
    float* scores_out;       // (B, k)
    int64_t* parent_out;     // (B, k): slot each new slot extended
    int64_t* label_out;      // (B, k): label it took (blank = none)
    int64_t* context_out;    // (B, k, ctx)
    int B;
    int k;
    int V;
    int64_t ld;
    int ctx;
    int64_t blank;
};

namespace beam_detail {

__device__ __forceinline__ float toFloat(float v) {
    return v;
}
__device__ __forceinline__ float toFloat(__half v) {
    return __half2float(v);
}
__device__ __forceinline__ float toFloat(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

// torch's `Max` (`a < b ? b : a`) and `Add`, reduced by xor butterflies of
// torch's warp width.
template <int kWidth>
__device__ __forceinline__ float warpMaxLikeTorch(float v) {
#pragma unroll
    for (int off = kWidth / 2; off > 0; off /= 2) {
        const float o = __shfl_xor_sync(0xffffffffu, v, off, kWidth);
        v = v < o ? o : v;
    }
    return v;
}

template <int kWidth>
__device__ __forceinline__ float warpSumLikeTorch(float v) {
#pragma unroll
    for (int off = kWidth / 2; off > 0; off /= 2) {
        v = v + __shfl_xor_sync(0xffffffffu, v, off, kWidth);
    }
    return v;
}

// The candidate order: higher total first, equal totals by lower flat index.
__device__ __forceinline__ bool ranksAbove(float va, int ia, float vb, int ib) {
    return va > vb || (va == vb && ia < ib);
}

// Warp-wide best (value, index) under `ranksAbove`; every lane ends with it.
__device__ __forceinline__ void warpBest(float& v, int& i) {
#pragma unroll
    for (int off = 16; off > 0; off /= 2) {
        const float ov = __shfl_xor_sync(0xffffffffu, v, off);
        const int oi = __shfl_xor_sync(0xffffffffu, i, off);
        if (ranksAbove(ov, oi, v, i)) {
            v = ov;
            i = oi;
        }
    }
}

}  // namespace beam_detail

// One CTA per utterance.  Each warp takes hypothesis rows j, j + kWarps, ...:
// it computes the row's log-softmax normaliser as torch does, then extracts
// the row's best k candidates in rank order (k rounds of a warp arg-best, the
// winner retired by the lane holding it).  The global top k is inside the union
// of the rows' top k, so warp 0 then merges the k sorted rows.
template <typename T, int kLog2V>
__global__ void __launch_bounds__(kBeamTopkWarps * 32)
    beamTopkKernel(const BeamTopkParams<T> p) {
    constexpr int kPow2 = 1 << kLog2V;
    // torch's WARP_SIZE and WARP_ITERATIONS for this row length.
    constexpr int kWidth = kPow2 < 32 ? kPow2 : 32;
    constexpr int kIter = kPow2 / kWidth;

    __shared__ float cand_val[kBeamTopkMaxBeam * kBeamTopkMaxBeam];
    __shared__ int cand_idx[kBeamTopkMaxBeam * kBeamTopkMaxBeam];
    __shared__ int sel_idx[kBeamTopkMaxBeam];
    __shared__ float sel_val[kBeamTopkMaxBeam];

    const int b = blockIdx.x;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int k = p.k;
    const int V = p.V;
    const bool active = p.active[b];

    if (active) {
        for (int j = warp; j < k; j += kBeamTopkWarps) {
            const T* row = p.logits + static_cast<int64_t>(b * k + j) * p.ld;
            float x[kIter];
#pragma unroll
            for (int it = 0; it < kIter; ++it) {
                const int v = lane + it * kWidth;
                x[it] = (lane < kWidth && v < V) ? beam_detail::toFloat(row[v]) : -INFINITY;
            }
            float mx = x[0];
#pragma unroll
            for (int it = 0; it < kIter; ++it) {
                mx = (mx > x[it]) ? mx : x[it];
            }
            mx = beam_detail::warpMaxLikeTorch<kWidth>(mx);
            float sum = 0.0f;
#pragma unroll
            for (int it = 0; it < kIter; ++it) {
                sum += __nv_expf(x[it] - mx);
            }
            sum = beam_detail::warpSumLikeTorch<kWidth>(sum);
            const float log_sum = __nv_logf(sum);
            const float score = p.scores[b * k + j];
#pragma unroll
            for (int it = 0; it < kIter; ++it) {
                const int v = lane + it * kWidth;
                x[it] = (lane < kWidth && v < V) ? score + ((x[it] - mx) - log_sum) : -INFINITY;
            }
            for (int r = 0; r < k; ++r) {
                float bv = -INFINITY;
                int bi = INT_MAX;
#pragma unroll
                for (int it = 0; it < kIter; ++it) {
                    const int idx = j * V + lane + it * kWidth;
                    if (beam_detail::ranksAbove(x[it], idx, bv, bi)) {
                        bv = x[it];
                        bi = idx;
                    }
                }
                beam_detail::warpBest(bv, bi);
                // The lane holding the winner retires it: compile-time indices
                // only, so `x` stays in registers.
#pragma unroll
                for (int it = 0; it < kIter; ++it) {
                    if (j * V + lane + it * kWidth == bi) {
                        x[it] = -INFINITY;
                    }
                }
                if (lane == 0) {
                    cand_val[j * k + r] = bv;
                    cand_idx[j * k + r] = bi;
                }
            }
        }
        __syncthreads();
        if (warp == 0) {
            // Lane j walks row j's sorted list; flat indices are unique, so the
            // lane whose head won is the one that advances.
            int head = 0;
            for (int r = 0; r < k; ++r) {
                float v = -INFINITY;
                int i = INT_MAX;
                if (lane < k && head < k) {
                    v = cand_val[lane * k + head];
                    i = cand_idx[lane * k + head];
                }
                float bv = v;
                int bi = i;
                beam_detail::warpBest(bv, bi);
                if (lane < k && head < k && i == bi) {
                    ++head;
                }
                if (lane == 0) {
                    sel_val[r] = bv;
                    sel_idx[r] = bi;
                }
            }
        }
        __syncthreads();
    }

    // The new beam.  A row past its utterance keeps every slot where it was,
    // emitting blank -- what the walk back through the frame expects.
    for (int r = threadIdx.x; r < k; r += blockDim.x) {
        const int o = b * k + r;
        if (active) {
            const int par = sel_idx[r] / V;
            p.scores_out[o] = sel_val[r];
            p.parent_out[o] = par;
            p.label_out[o] = sel_idx[r] - par * V;
        } else {
            p.scores_out[o] = p.scores[o];
            p.parent_out[o] = r;
            p.label_out[o] = p.blank;
        }
    }
    // Label windows onto the new slots: the parent's window, shifted to take the
    // label unless it is blank.
    const int ctx = p.ctx;
    for (int e = threadIdx.x; e < k * ctx; e += blockDim.x) {
        const int r = e / ctx;
        const int c = e - r * ctx;
        int par = r;
        int64_t label = p.blank;
        if (active) {
            par = sel_idx[r] / V;
            label = sel_idx[r] - par * V;
        }
        const int64_t* src = p.context + (static_cast<int64_t>(b) * k + par) * ctx;
        const int64_t value = (label == p.blank) ? src[c] : (c + 1 < ctx ? src[c + 1] : label);
        p.context_out[(static_cast<int64_t>(b) * k + r) * ctx + c] = value;
    }
}

template <typename T>
cudaError_t BeamTopk(const BeamTopkParams<T>& p, cudaStream_t stream) {
    if (p.B == 0) {
        return cudaSuccess;
    }
    if (p.k < 1 || p.k > kBeamTopkMaxBeam || p.k > p.V || p.V < 1 || p.ctx < 1) {
        return cudaErrorInvalidValue;
    }
    int log2v = 0;
    while ((1 << log2v) < p.V) {
        ++log2v;
    }
    const dim3 grid(p.B);
    const dim3 block(kBeamTopkWarps * 32);
    switch (log2v) {
#define OASR_BEAM_TOPK_CASE(L)                                         \
    case L:                                                            \
        beamTopkKernel<T, L><<<grid, block, 0, stream>>>(p);           \
        break;
        OASR_BEAM_TOPK_CASE(0)
        OASR_BEAM_TOPK_CASE(1)
        OASR_BEAM_TOPK_CASE(2)
        OASR_BEAM_TOPK_CASE(3)
        OASR_BEAM_TOPK_CASE(4)
        OASR_BEAM_TOPK_CASE(5)
        OASR_BEAM_TOPK_CASE(6)
        OASR_BEAM_TOPK_CASE(7)
        OASR_BEAM_TOPK_CASE(8)
        OASR_BEAM_TOPK_CASE(9)
        OASR_BEAM_TOPK_CASE(10)
#undef OASR_BEAM_TOPK_CASE
        default:
            return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

}  // namespace transducer
}  // namespace oasr
