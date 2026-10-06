// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Fused frame-synchronous greedy decode for a stateless-predictor transducer.
//
// One launch decodes a whole batch.  Each CTA owns `R` rows and runs the
// greedy loop for them on the device -- encoder-frame gather, joiner
// (add + activation + output GEMV + argmax), emit/advance bookkeeping, and on
// an emission the predictor step (label-window shift, embedding, grouped
// convolution over the window, ReLU, decoder projection GEMV) -- until every
// row has consumed its encoder frames.
//
// Why a persistent kernel rather than faster per-op launches: the per-step work
// is a few hundred kilo-MACs per row, so the batched op-by-op loop it replaces
// is latency-bound -- about 25 kernels per step, each 0.6-3 us, plus the gaps
// between them, plus a host round trip every few steps for termination.
// Measured on the icefall Zipformer transducer at B=64: ~8,000 kernels and
// ~15 ms of GPU time per micro-batch, almost independent of batch width.
// Here rows advance independently and the host waits once.
//
// The GEMVs, and what each layout measured (B=64, one row per CTA, 304 steps,
// RTX 5090, bf16):
//
//   * Row-major weights, lanes splitting K, a shuffle reduction per output:
//     14.7 us/step -- every output waited out a five-deep shuffle chain.
//   * K-major weights, a warp per K-slice, a lane per 16 outputs, K-slices
//     summed once through shared memory: 9.9 us/step.  Issue-bound: ~189
//     instructions per four K, half of them bf16->fp32 conversions, one per FMA.
//   * Tensor cores (`mma.sync.m16n8k16`): 12.7 us/step, *slower*.  With one live
//     row the MMA computes sixteen, and GeForce Blackwell's fp32-accumulate MMA
//     rate makes that 16x the work, not free padding.
//   * K-major as above, with sm_100+'s mixed-precision FMA (`FHFMA`: bf16/fp16
//     multiplicands, fp32 accumulator, a half of a packed register selected as
//     an operand modifier) instead of convert-then-FFMA, and eight K per stage:
//     no conversion instructions, and the bf16 x bf16 product is exact in fp32,
//     so the arithmetic is bit-identical to `fmaf(float(w), float(x), acc)`.
//
// Numerical contract.  Every rounding point of the op-by-op path is kept: the
// joiner input `enc + dec` and its activation are rounded to the activation
// dtype, logits are rounded to it before the argmax (ties resolve to the lowest
// token id, as `torch.argmax` does), the convolution accumulates in the same
// fmaf order as `oasr::conv::groupedConv2dKernel` / `depthwiseConv1DKernel`,
// ReLU is `oasr::relu`, and the decoder projection is rounded after its bias.
// What cannot be kept is the GEMV accumulation order of the library GEMMs it
// replaces (tensor-core MMAs), so a logit can land one ulp away from the
// reference -- which moves a decision only when the top two tokens tie at the
// activation dtype's resolution.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <oasr/common/math.h>
#include <oasr/common/vec_dtypes.h>
#include <type_traits>

namespace oasr {
namespace transducer {

//! Longest label window (predictor context) the kernel carries per row.
constexpr int kMaxContext = 8;
//! Outputs one GEMV pass produces: one per thread of the CTA.  Both K-major
//! weights are padded to a multiple of it on their output axis.
constexpr int kGemvTile = 512;
//! Threads per CTA (one per tile output) and the warps that split K.
constexpr int kThreads = kGemvTile;
constexpr int kWarps = kThreads / 32;
//! A lane owns outputs [8l, 8l+8) and [kHalfTile+8l, kHalfTile+8l+8) of a tile,
//! so each weight load instruction is 512 contiguous bytes.
constexpr int kHalfTile = kGemvTile / 2;
//! K per stage: every lane keeps 2*kStageK 16-byte weight loads in flight.
constexpr int kStageK = 8;

//! Joiner nonlinearity applied to `enc + dec` before the output projection.
enum JoinerActivation : int { kJoinerTanh = 0, kJoinerRelu = 1 };

template <typename T>
struct StatelessGreedyParams {
    // ---- per-row inputs ----
    const T* enc_proj;  // (B, T, J), last dim contiguous: encoder already in joiner space
    int64_t enc_stride_b;
    int64_t enc_stride_t;
    const int64_t* lengths;    // (B,) valid frames per row
    const int64_t* window_in;  // (B, ctx) label window the predictor state is
    const T* dec_proj_in;      // (B, J) decoder projection of that window
    // ---- weights ----
    const T* w_out_t;  // (J, ld_out) joiner output projection, K-major
    const T* b_out;    // (>= V,) or nullptr
    const T* emb;      // (V, D) predictor embedding
    const T* conv_w;   // group > 1: (D, 1, ctx, group) KRSC; group == 1: (ctx, 1, D); nullptr if ctx == 1
    const T* w_dp_t;   // (D, ld_dp) decoder projection, K-major
    const T* b_dp;     // (J,) or nullptr
    // ---- outputs ----
    int32_t* tokens;      // (B, cap) emitted tokens, in order
    int32_t* frames;      // (B, cap) encoder frame of each emission
    float* probs;         // (B, cap) posterior of each emission, or nullptr
    int32_t* counts;      // (B,) emissions per row (may exceed cap: overflow)
    int64_t* window_out;  // (B, ctx)
    T* dec_proj_out;      // (B, J)
    // ---- sizes / constants ----
    int B;
    int J;       // joiner dim (multiple of kStageK)
    int D;       // decoder dim (multiple of kStageK)
    int V;       // vocabulary: the argmax range
    int ld_out;  // multiple of kGemvTile, >= V
    int ld_dp;   // multiple of kGemvTile, >= J
    int ctx;     // predictor context size
    int group;   // input channels per conv group (1 == depthwise)
    int cap;
    int max_sym;
    int blank;
    int activation;
};

namespace detail {

template <typename T>
__device__ __forceinline__ float roundTo(float x) {
    return toFloat<T>(fromFloat<T>(x));
}

// (value, index) argmax order: larger value wins, NaN counts as the largest
// (torch.argmax propagates NaN), and equal keys resolve to the lowest index.
__device__ __forceinline__ bool argmaxBetter(float v, int i, float bv, int bi) {
    const bool vn = isnan(v), bn = isnan(bv);
    if (vn || bn) {
        if (vn && !bn) return true;
        if (!vn && bn) return false;
        return i < bi;
    }
    return v > bv || (v == bv && i < bi);
}

__device__ __forceinline__ void lseCombine(float& m, float& s, float om, float os) {
    if (om == -INFINITY) return;
    const float nm = fmaxf(m, om);
    s = s * __expf(m - nm) + os * __expf(om - nm);
    m = nm;
}

// `acc + w * x` for one half of each packed pair, in fp32.  On sm_100+ a single
// mixed-precision FMA (FHFMA): the product of two half-precision values is exact
// in fp32, so this is bit-identical to converting first and calling fmaf -- which
// is what older architectures do.
template <typename T, int HW, int HX>
__device__ __forceinline__ float fmaPair(uint32_t w2, uint32_t x2, float acc) {
    const unsigned short a = HW ? static_cast<unsigned short>(w2 >> 16)
                                : static_cast<unsigned short>(w2 & 0xffffu);
    const unsigned short b = HX ? static_cast<unsigned short>(x2 >> 16)
                                : static_cast<unsigned short>(x2 & 0xffffu);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    float d;
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        asm("fma.rn.f32.bf16 %0, %1, %2, %3;" : "=f"(d) : "h"(a), "h"(b), "f"(acc));
    } else {
        asm("fma.rn.f32.f16 %0, %1, %2, %3;" : "=f"(d) : "h"(a), "h"(b), "f"(acc));
    }
    return d;
#else
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return fmaf(__bfloat162float(__ushort_as_bfloat16(a)),
                    __bfloat162float(__ushort_as_bfloat16(b)), acc);
    } else {
        return fmaf(__half2float(__ushort_as_half(a)), __half2float(__ushort_as_half(b)), acc);
    }
#endif
}

__device__ __forceinline__ uint32_t word(const uint4& v, int i) {
    return i == 0 ? v.x : (i == 1 ? v.y : (i == 2 ? v.z : v.w));
}

// acc[r][i] += sum over K in [k_begin, k_end) of x[r][k] * wt[k][n0 + out(lane, i)],
// out(lane, i) = (i < 8 ? 0 : kHalfTile) + 8*lane + i % 8.  K advances kStageK
// at a time: all 2*kStageK weight loads of a stage are issued before its first
// FMA, and x is one 16-byte broadcast read per row.  k_begin, k_end and K are
// multiples of kStageK.
template <typename T, int R>
__device__ __forceinline__ void kSliceFma(float (&acc)[R][16], const T* wt, int ldw, int n0,
                                          int k_begin, int k_end, const T* xs, int K,
                                          int lane) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int i = 0; i < 16; ++i) acc[r][i] = 0.0f;
    }
    const T* base = wt + n0 + lane * 8;
    for (int k = k_begin; k < k_end; k += kStageK) {
        uint4 w[kStageK][2];
#pragma unroll
        for (int u = 0; u < kStageK; ++u) {
            const T* row = base + static_cast<int64_t>(k + u) * ldw;
            w[u][0] = *reinterpret_cast<const uint4*>(row);
            w[u][1] = *reinterpret_cast<const uint4*>(row + kHalfTile);
        }
        uint4 xv[R];
#pragma unroll
        for (int r = 0; r < R; ++r) xv[r] = *reinterpret_cast<const uint4*>(xs + r * K + k);
#pragma unroll
        for (int u = 0; u < kStageK; ++u) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const uint32_t x2 = word(xv[r], u >> 1);
#pragma unroll
                for (int h = 0; h < 2; ++h) {
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const uint32_t w2 = word(w[u][h], i);
                        float& lo = acc[r][8 * h + 2 * i];
                        float& hi = acc[r][8 * h + 2 * i + 1];
                        if (u & 1) {
                            lo = fmaPair<T, 0, 1>(w2, x2, lo);
                            hi = fmaPair<T, 1, 1>(w2, x2, hi);
                        } else {
                            lo = fmaPair<T, 0, 0>(w2, x2, lo);
                            hi = fmaPair<T, 1, 0>(w2, x2, hi);
                        }
                    }
                }
            }
        }
    }
}

// Sum one row's per-warp partials: every thread ends up with tile output `tid`.
// Two barriers: after the partials land, and before `part` is reused.
__device__ __forceinline__ float crossWarpSum(const float (&acc)[16], float* part, int warp,
                                              int lane, int tid) {
    float* dst = part + warp * kGemvTile + lane * 8;
    reinterpret_cast<float4*>(dst)[0] = make_float4(acc[0], acc[1], acc[2], acc[3]);
    reinterpret_cast<float4*>(dst)[1] = make_float4(acc[4], acc[5], acc[6], acc[7]);
    reinterpret_cast<float4*>(dst + kHalfTile)[0] = make_float4(acc[8], acc[9], acc[10], acc[11]);
    reinterpret_cast<float4*>(dst + kHalfTile)[1] =
        make_float4(acc[12], acc[13], acc[14], acc[15]);
    __syncthreads();
    float v = 0.0f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) v += part[w * kGemvTile + tid];
    __syncthreads();
    return v;
}

__device__ __forceinline__ void cpAsync16(void* smem_dst, const void* gmem_src) {
    const unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(gmem_src));
}

__device__ __forceinline__ void cpAsyncWaitAll() { asm volatile("cp.async.wait_all;\n" ::); }

}  // namespace detail

template <typename T, int R>
__host__ __device__ constexpr size_t statelessGreedySmemBytes(int J, int D) {
    // es: 2 x R x J raw encoder frames + xs: R x J joiner input + hs: R x D
    // predictor hidden (all T); dps: R x J decoder projection, K-slice partials
    // kWarps x tile, argmax / logsumexp partials kWarps x R x 4 (all float).
    return sizeof(T) * static_cast<size_t>(R) * (3 * J + D) +
           sizeof(float) * (static_cast<size_t>(R) * J + static_cast<size_t>(kWarps) * kGemvTile +
                            static_cast<size_t>(kWarps) * R * 4);
}

// One CTA per SM by construction (the grid is one CTA per row group, and a
// batch rarely exceeds the SM count), so the register budget is the whole
// 128/thread a 512-thread block allows.  Left to its default, ptxas held an
// earlier version at 64 to keep a second CTA resident -- which never happens --
// and spilled the GEMV accumulators on every K stage.
template <typename T, int R, bool kTrack>
__global__ void __launch_bounds__(kThreads, 1)
    statelessGreedyKernel(const StatelessGreedyParams<T> p) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    const int J = p.J, D = p.D, ctx = p.ctx;
    T* es = reinterpret_cast<T*>(smem_raw);  // 2 x R x J encoder frames (current / prefetched)
    T* xs = es + 2 * R * J;                  // R x J joiner input (after activation)
    T* hs = xs + R * J;                      // R x D predictor hidden (after ReLU)
    float* dps = reinterpret_cast<float*>(hs + R * D);  // R x J current decoder projection
    float* part = dps + R * J;                          // kWarps x tile K-slice partials
    float* red = part + kWarps * kGemvTile;  // kWarps x R x {best, idx, lse_m, lse_s}

    __shared__ int s_t[R], s_sym[R], s_len[R], s_cnt[R], s_emit[R], s_buf[R];
    __shared__ int64_t s_win[R][kMaxContext];
    __shared__ int s_any_active, s_any_emit;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int row0 = blockIdx.x * R;
    constexpr int kVec = 16 / sizeof(T);  // elements per 16-byte cp.async

    if (tid < R) {
        const int b = row0 + tid;
        const bool valid = b < p.B;
        s_t[tid] = 0;
        s_sym[tid] = 0;
        s_cnt[tid] = 0;
        s_buf[tid] = 0;
        s_len[tid] = valid ? static_cast<int>(p.lengths[b]) : 0;
        for (int i = 0; i < ctx; ++i) {
            s_win[tid][i] = valid ? p.window_in[static_cast<int64_t>(b) * ctx + i]
                                  : static_cast<int64_t>(p.blank);
        }
    }
    for (int idx = tid; idx < R * J; idx += kThreads) {
        const int r = idx / J;
        const int b = row0 + r;
        dps[idx] = b < p.B ? toFloat<T>(p.dec_proj_in[static_cast<int64_t>(b) * J + (idx - r * J)])
                           : 0.0f;
    }
    for (int idx = tid; idx < R * D; idx += kThreads) hs[idx] = fromFloat<T>(0.0f);
    __syncthreads();
    // Frame 0 of every live row into buffer 0.
    for (int idx = tid * kVec; idx < R * J; idx += kThreads * kVec) {
        const int r = idx / J;
        if (s_len[r] > 0) {
            detail::cpAsync16(
                es + idx, p.enc_proj + static_cast<int64_t>(row0 + r) * p.enc_stride_b + (idx - r * J));
        }
    }
    detail::cpAsyncWaitAll();
    if (tid == 0) {
        int any = 0;
#pragma unroll
        for (int r = 0; r < R; ++r) any |= s_len[r] > 0;
        s_any_active = any;
    }
    __syncthreads();

    // K-slices, rounded to whole stages.
    const int kj = ((J + kWarps - 1) / kWarps + kStageK - 1) / kStageK * kStageK;
    const int kd = ((D + kWarps - 1) / kWarps + kStageK - 1) / kStageK * kStageK;
    const int jb = min(J, warp * kj), je = min(J, jb + kj);
    const int db = min(D, warp * kd), de = min(D, db + kd);

    while (s_any_active) {
        // ---- joiner input: act(round(enc[t] + dec)), rounded ----
        for (int idx = tid; idx < R * J; idx += kThreads) {
            const int r = idx / J;
            const float e = toFloat<T>(es[s_buf[r] * R * J + idx]);
            const float s = detail::roundTo<T>(e + dps[idx]);
            const float a = p.activation == kJoinerTanh ? tanhf(s) : relu<float>(s);
            xs[idx] = fromFloat<T>(a);
        }
        __syncthreads();

        // ---- prefetch frame t+1 into the other buffer, under the joiner GEMV ----
        for (int idx = tid * kVec; idx < R * J; idx += kThreads * kVec) {
            const int r = idx / J;
            const int t1 = s_t[r] + 1;
            if (t1 < s_len[r]) {
                detail::cpAsync16(es + (1 - s_buf[r]) * R * J + idx,
                                  p.enc_proj + static_cast<int64_t>(row0 + r) * p.enc_stride_b +
                                      static_cast<int64_t>(t1) * p.enc_stride_t + (idx - r * J));
            }
        }

        // ---- joiner output GEMV, one tile of the vocabulary per pass ----
        float best[R], lse_m[R], lse_s[R];
        int best_i[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            best[r] = -INFINITY;
            best_i[r] = 0x7fffffff;
            lse_m[r] = -INFINITY;
            lse_s[r] = 0.0f;
        }
        for (int n0 = 0; n0 < p.V; n0 += kGemvTile) {
            float acc[R][16];
            detail::kSliceFma<T, R>(acc, p.w_out_t, p.ld_out, n0, jb, je, xs, J, lane);
            const int v = n0 + tid;
            const float bias = (v < p.V && p.b_out != nullptr) ? toFloat<T>(p.b_out[v]) : 0.0f;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const float sum = detail::crossWarpSum(acc[r], part, warp, lane, tid);
                if (v < p.V) {
                    const float logit = detail::roundTo<T>(sum + bias);
                    if (detail::argmaxBetter(logit, v, best[r], best_i[r])) {
                        best[r] = logit;
                        best_i[r] = v;
                    }
                    if constexpr (kTrack) detail::lseCombine(lse_m[r], lse_s[r], logit, 1.0f);
                }
            }
        }
        // Block argmax (+ logsumexp): warp shuffles, then one slot per warp.
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, best[r], off);
                const int oi = __shfl_xor_sync(0xffffffffu, best_i[r], off);
                if (detail::argmaxBetter(ov, oi, best[r], best_i[r])) {
                    best[r] = ov;
                    best_i[r] = oi;
                }
                if constexpr (kTrack) {
                    const float om = __shfl_xor_sync(0xffffffffu, lse_m[r], off);
                    const float os = __shfl_xor_sync(0xffffffffu, lse_s[r], off);
                    detail::lseCombine(lse_m[r], lse_s[r], om, os);
                }
            }
            if (lane == 0) {
                float* slot = red + (warp * R + r) * 4;
                slot[0] = best[r];
                slot[1] = __int_as_float(best_i[r]);
                slot[2] = lse_m[r];
                slot[3] = lse_s[r];
            }
        }
        detail::cpAsyncWaitAll();
        __syncthreads();

        // ---- per-row decision: warp 0, lanes [16r, 16r+16) combine row r ----
        if (warp == 0) {
            int emits = 0, actives = 0;
            for (int rb = 0; rb < R; rb += 2) {
                const int r = rb + (lane >> 4);
                const int w = lane & 15;
                float bv = -INFINITY, m = -INFINITY, s = 0.0f;
                int bi = 0x7fffffff;
                if (r < R) {
                    const float* slot = red + (w * R + r) * 4;
                    bv = slot[0];
                    bi = __float_as_int(slot[1]);
                    m = slot[2];
                    s = slot[3];
                }
#pragma unroll
                for (int off = 8; off > 0; off >>= 1) {
                    const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
                    const int oi = __shfl_xor_sync(0xffffffffu, bi, off);
                    if (detail::argmaxBetter(ov, oi, bv, bi)) {
                        bv = ov;
                        bi = oi;
                    }
                    if constexpr (kTrack) {
                        const float om = __shfl_xor_sync(0xffffffffu, m, off);
                        const float os = __shfl_xor_sync(0xffffffffu, s, off);
                        detail::lseCombine(m, s, om, os);
                    }
                }
                int emit = 0, active_next = 0;
                if (r < R && w == 0) {
                    if (s_t[r] < s_len[r]) {
                        const int b = row0 + r;
                        if (bi != p.blank && s_sym[r] < p.max_sym) {
                            const int cidx = s_cnt[r];
                            if (cidx < p.cap) {
                                const int64_t o = static_cast<int64_t>(b) * p.cap + cidx;
                                p.tokens[o] = bi;
                                p.frames[o] = s_t[r];
                                if constexpr (kTrack) p.probs[o] = __expf(bv - (m + __logf(s)));
                            }
                            s_cnt[r] = cidx + 1;
                            s_sym[r] += 1;
                            for (int i = 0; i + 1 < ctx; ++i) s_win[r][i] = s_win[r][i + 1];
                            s_win[r][ctx - 1] = bi;
                            emit = 1;
                        } else {
                            s_t[r] += 1;
                            s_sym[r] = 0;
                            s_buf[r] ^= 1;  // frame t+1 was prefetched into the other buffer
                        }
                    }
                    s_emit[r] = emit;
                    active_next = s_t[r] < s_len[r];
                }
                emits |= __ballot_sync(0xffffffffu, emit) != 0u;
                actives |= __ballot_sync(0xffffffffu, active_next) != 0u;
            }
            if (lane == 0) {
                s_any_emit = emits;
                s_any_active = actives;
            }
        }
        __syncthreads();
        if (!s_any_emit) continue;

        // ---- predictor hidden: relu(round(conv(emb[window]))) ----
        for (int idx = tid; idx < R * D; idx += kThreads) {
            const int r = idx / D;
            if (!s_emit[r]) continue;
            const int d = idx - r * D;
            T h;
            if (ctx == 1) {
                h = p.emb[s_win[r][0] * D + d];
            } else if (p.group == 1) {
                float acc = 0.0f;
                for (int s = 0; s < ctx; ++s) {
                    acc = fmaf(toFloat<T>(p.emb[s_win[r][s] * D + d]),
                               toFloat<T>(p.conv_w[static_cast<int64_t>(s) * D + d]), acc);
                }
                h = fromFloat<T>(acc);
            } else {
                const int gs = p.group;
                const int c0 = (d / gs) * gs;
                float acc = 0.0f;
                for (int s = 0; s < ctx; ++s) {
                    const T* e = p.emb + s_win[r][s] * D + c0;
                    const T* w = p.conv_w + (static_cast<int64_t>(d) * ctx + s) * gs;
                    for (int i = 0; i < gs; ++i) acc = fmaf(toFloat<T>(e[i]), toFloat<T>(w[i]), acc);
                }
                h = fromFloat<T>(acc);
            }
            hs[idx] = relu<T>(h);
        }
        __syncthreads();

        // ---- decoder projection GEMV for the rows that emitted ----
        for (int n0 = 0; n0 < J; n0 += kGemvTile) {
            float acc[R][16];
            detail::kSliceFma<T, R>(acc, p.w_dp_t, p.ld_dp, n0, db, de, hs, D, lane);
            const int n = n0 + tid;
            const float bias = (n < J && p.b_dp != nullptr) ? toFloat<T>(p.b_dp[n]) : 0.0f;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const float sum = detail::crossWarpSum(acc[r], part, warp, lane, tid);
                if (n < J && s_emit[r]) dps[r * J + n] = detail::roundTo<T>(sum + bias);
            }
        }
        __syncthreads();
    }

    // ---- write the carried state back ----
    if (tid < R) {
        const int b = row0 + tid;
        if (b < p.B) {
            p.counts[b] = s_cnt[tid];
            for (int i = 0; i < ctx; ++i) {
                p.window_out[static_cast<int64_t>(b) * ctx + i] = s_win[tid][i];
            }
        }
    }
    for (int idx = tid; idx < R * J; idx += kThreads) {
        const int r = idx / J;
        const int b = row0 + r;
        if (b < p.B) {
            p.dec_proj_out[static_cast<int64_t>(b) * J + (idx - r * J)] = fromFloat<T>(dps[idx]);
        }
    }
}

//! Rows per CTA for a batch: one row per SM while the batch fits on the GPU
//! (a CTA's step latency is one pass over the weights whatever its row count),
//! two once it does not, so a wide batch runs in fewer waves.
inline int statelessGreedyRowsPerCta(int batch, int num_sms) { return batch <= num_sms ? 1 : 2; }

template <typename T, int R, bool kTrack>
cudaError_t launchStatelessGreedy(const StatelessGreedyParams<T>& p, cudaStream_t stream) {
    const size_t smem = statelessGreedySmemBytes<T, R>(p.J, p.D);
    auto kernel = statelessGreedyKernel<T, R, kTrack>;
    if (smem > 48 * 1024) {
        cudaError_t st = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              static_cast<int>(smem));
        if (st != cudaSuccess) return st;
    }
    const int grid = (p.B + R - 1) / R;
    kernel<<<grid, kThreads, smem, stream>>>(p);
    return cudaGetLastError();
}

//! Launch the fused greedy decode.  `rows_per_cta` <= 0 picks
//! :func:`statelessGreedyRowsPerCta`.  Half precision only; `J` and `D` must
//! be multiples of `kStageK`, and both K-major weights padded on their output
//! axis to a multiple of `kGemvTile`.
template <typename T>
cudaError_t StatelessGreedyDecode(const StatelessGreedyParams<T>& p, int rows_per_cta,
                                  cudaStream_t stream) {
    if (p.B == 0) return cudaSuccess;
    if (p.ctx < 1 || p.ctx > kMaxContext || p.J % kStageK != 0 || p.D % kStageK != 0 ||
        p.ld_out % kGemvTile != 0 || p.ld_dp % kGemvTile != 0 || p.ld_out < p.V ||
        p.ld_dp < p.J) {
        return cudaErrorInvalidValue;
    }
    if (rows_per_cta <= 0) {
        int dev = 0, sms = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        rows_per_cta = statelessGreedyRowsPerCta(p.B, sms);
    }
    const bool track = p.probs != nullptr;
    switch (rows_per_cta) {
        case 1:
            return track ? launchStatelessGreedy<T, 1, true>(p, stream)
                         : launchStatelessGreedy<T, 1, false>(p, stream);
        case 2:
            return track ? launchStatelessGreedy<T, 2, true>(p, stream)
                         : launchStatelessGreedy<T, 2, false>(p, stream);
        default:
            return cudaErrorInvalidValue;
    }
}

}  // namespace transducer
}  // namespace oasr
