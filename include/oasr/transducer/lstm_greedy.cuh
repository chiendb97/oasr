// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Fused greedy decode for a transducer with an LSTM predictor (NeMo's RNN-T,
// Nemotron ASR): one cooperative launch decodes a small batch, every CTA of the
// GPU working on every step.
//
// What it replaces.  The strategy's greedy loop, replayed from CUDA graphs, runs
// each step as a dozen small kernels -- the joiner add and ReLU, a vocabulary
// GEMM, argmax, the masks, and on an emission an embedding gather, two LSTM
// layer steps and the decoder projection.  At batch 1 that is ~53 us a step and
// two thirds of a Nemotron request's GPU time, although a step's arithmetic is
// a few tens of microseconds of weight reads.
//
// Why the whole GPU, not a cluster (greedy_cluster.cuh).  A step reads the
// vocabulary head -- 13,088 x 640 on Nemotron, 16.75 MB -- and an emission the
// two LSTM layers (13.1 MB) and the decoder projection.  One SM streams ~78
// GB/s out of L2, so a 16-CTA cluster would spend ~13 us a step on the head
// alone; every SM together streams ~7 TB/s, ~2.3 us.  So the work is cut by
// output across the whole grid and the CTAs meet at grid-wide barriers:
//
//   joint     every CTA builds `relu(round(enc[t] + dec))` for each row and
//             computes its share of the vocabulary's logits; its best
//             (logit, token) per row goes into a global 64-bit atomicMax whose
//             high half is the round, so no reset is needed between rounds.
//   barrier   every CTA reads the winners and takes the same emit / advance
//             decision per row; the first CTA records the emissions.
//   layer l   (only when some row emitted) each CTA owns a share of the hidden
//             units -- all four gates of each, so the cell update is local --
//             and computes them from the layer's input and previous hidden
//             state, both read whole from global memory; it keeps its units'
//             cells and publishes their new hidden state.  A barrier per layer.
//   project   each CTA owns a share of the decoder projection's outputs.  A
//             barrier, and every CTA refreshes its copy of the projection.
//
// A step with no emission is one grid barrier; an emission adds L + 1.
//
// Weights are read as the model holds them (row-major, one warp per output row,
// lanes splitting K in 16-byte chunks), so nothing is re-laid-out.  Each layer's
// hidden state is double-buffered by emission parity: a CTA publishes the new
// state while others still read the previous one in the same phase.
//
// Numerical contract.  The rounding points of the op-by-op path are kept: the
// joiner input `enc + dec` is rounded before its ReLU, logits are rounded to the
// activation dtype before the argmax (ties to the lowest token, NaN above all,
// as torch.argmax), the LSTM gates stay fp32 with b_ih then b_hh added, the
// cell is fp32 and stored rounded while the hidden state is
// round(o * tanh(cell)) of the unrounded cell (the recurrent kernel's step),
// and the decoder projection is rounded after its bias.  What cannot be kept is
// the dot products' accumulation order (each library kernel has its own), so a
// value can land an ulp away, which moves a decision only at a near tie.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <oasr/common/math.h>
#include <oasr/common/vec_dtypes.h>
#include <oasr/transducer/greedy_cluster.cuh>
#include <oasr/transducer/greedy_decode.cuh>

namespace oasr {
namespace transducer {

//! Deepest LSTM the kernel takes.
constexpr int kLstmMaxLayers = 4;
//! Rows (utterances) one launch decodes, all in lockstep.
constexpr int kLstmMaxRows = 4;
constexpr int kLstmThreads = 512;
constexpr int kLstmWarps = kLstmThreads / 32;
//! Largest vocabulary: a candidate's token is the low half of its key.
constexpr int kLstmMaxVocab = 65535;
//! A dot product's K, in 16-byte chunks per lane, loaded before any FMA.
constexpr int kLstmChunks = 4;
//! Longest dot product (hidden size, joiner dim): one pass of every lane's chunks.
constexpr int kLstmMaxK = 32 * kLstmChunks * 8;

template <typename T>
struct LstmGreedyParams {
    // ---- per-row inputs ----
    const T* enc_proj;  // (B, frames, J), last dim contiguous: encoder in joiner space
    int64_t enc_stride_b;
    int64_t enc_stride_t;
    int num_frames;
    const int64_t* lengths;  // (B,) valid frames, clamped to [0, num_frames]
    const T* h_in;           // (L, B, H) hidden state the decode starts from
    const T* c_in;           // (L, B, H) cell state
    const T* dec_proj_in;    // (B, J) decoder projection of h_in[L - 1]
    // ---- weights, as the model holds them ----
    const T* emb;                     // (>= V, E) label embedding
    const T* w_ih[kLstmMaxLayers];    // (4H, E) for layer 0, (4H, H) after; gates i, f, g, o
    const T* w_hh[kLstmMaxLayers];    // (4H, H)
    const T* b_ih[kLstmMaxLayers];    // (4H,) or nullptr
    const T* b_hh[kLstmMaxLayers];    // (4H,) or nullptr
    const T* w_dp;                    // (J, H) decoder projection
    const T* b_dp;                    // (J,) or nullptr
    const T* w_out;                   // (>= V, J) joiner head
    const T* b_out;                   // (>= V,) or nullptr
    // ---- outputs ----
    int32_t* tokens;  // (B, cap)
    int32_t* frames;  // (B, cap)
    float* probs;     // (B, cap) posterior of each emission, or nullptr
    int32_t* counts;  // (B,) emissions per row (may exceed cap)
    T* h_out;         // (L, B, H)
    T* c_out;         // (L, B, H)
    T* dec_proj_out;  // (B, J)
    // ---- workspace (lstmGreedyWorkspace), carved by the launcher ----
    T* hx;                     // (L, 2, B, H) each layer's hidden state, by emission parity
    T* dpx;                    // (B, J) decoder projection
    float* lse;                // (2, G, B, 2) per-CTA logsumexp, by round parity (tracking)
    unsigned long long* best;  // (2, B) round-tagged argmax keys, by round parity; zeroed
    unsigned* bar;             // grid barrier arrivals; zeroed
    // ---- sizes ----
    int B;  // rows, <= kLstmMaxRows
    int E;  // embedding dim (layer 0 input)
    int H;  // hidden size
    int L;  // layers
    int J;  // joiner dim
    int V;  // vocabulary (argmax range)
    int cap;
    int max_sym;
    int blank;
    //! Head rows a CTA keeps in shared memory for the whole launch (the rest of
    //! its share streams from L2 every step); set by the launcher.
    int res_rows;
};

namespace lg {

//! [lo, hi) of `n` items for CTA `i` of `g`.
__device__ __forceinline__ void share(int n, int i, int g, int* lo, int* hi) {
    *lo = static_cast<int>(static_cast<int64_t>(n) * i / g);
    *hi = static_cast<int>(static_cast<int64_t>(n) * (i + 1) / g);
}

__device__ __forceinline__ uint4 ldWeight(const void* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

// acc[n][r] += w[n][0:K] . x[n][r][0:K] for the first `nw` of NW weight rows,
// this lane's 16-byte chunks (c = lane, lane + 32, ...) of every row loaded
// before the first FMA, so a warp keeps NW rows' worth of loads in flight:
// one row at a time left a warp waiting out a full L2 round trip per row, and
// the vocabulary head ran at a third of the bandwidth.  `x[n]` is in shared
// memory, input row r at x[n] + r * ldx.  K % 8 == 0, K <= 256 * kLstmChunks,
// weights 16-byte aligned.
template <typename T, int R, int NW>
__device__ __forceinline__ void laneDots(const T* const (&w)[NW], const T* const (&x)[NW], int K,
                                         int ldx, int nw, int nr, int lane, float (&acc)[NW][R]) {
    const int nc = K >> 3;
    uint4 wv[NW][kLstmChunks];
#pragma unroll
    for (int n = 0; n < NW; ++n) {
#pragma unroll
        for (int i = 0; i < kLstmChunks; ++i) {
            const int c = lane + 32 * i;
            if (n < nw && c < nc) wv[n][i] = ldWeight(w[n] + c * 8);
        }
    }
#pragma unroll
    for (int n = 0; n < NW; ++n) {
#pragma unroll
        for (int i = 0; i < kLstmChunks; ++i) {
            const int c = lane + 32 * i;
            if (n < nw && c < nc) {
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    if (r < nr) {
                        const uint4 xv = *reinterpret_cast<const uint4*>(x[n] + r * ldx + c * 8);
#pragma unroll
                        for (int q = 0; q < 4; ++q) {
                            const uint32_t w2 = detail::word(wv[n][i], q);
                            const uint32_t x2 = detail::word(xv, q);
                            acc[n][r] = detail::fmaPair<T, 0, 0>(w2, x2, acc[n][r]);
                            acc[n][r] = detail::fmaPair<T, 1, 1>(w2, x2, acc[n][r]);
                        }
                    }
                }
            }
        }
    }
}

template <int R>
__device__ __forceinline__ void warpSum(float (&acc)[R]) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], off);
    }
}

template <int NW, int R>
__device__ __forceinline__ void warpSum(float (&acc)[NW][R]) {
#pragma unroll
    for (int n = 0; n < NW; ++n) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                acc[n][r] += __shfl_xor_sync(0xffffffffu, acc[n][r], off);
            }
        }
    }
}

// Grid-wide barrier on a monotonic counter (zeroed before the launch): every
// CTA adds one with release semantics and waits, with acquire, for all of this
// barrier's arrivals.  `target` is this CTA's running count.
__device__ __forceinline__ void gridSync(unsigned* bar, unsigned& target) {
    __syncthreads();
    target += gridDim.x;
    if (threadIdx.x == 0) {
        asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(bar) : "memory");
        unsigned v;
        do {
            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(bar) : "memory");
        } while (static_cast<int>(v - target) < 0);
    }
    __syncthreads();
}

//! Exchange-buffer reads go to L2: a line cached in L1 from an earlier phase
//! would be stale.
template <typename T>
__device__ __forceinline__ uint4 ldShared16(const T* p) {
    return __ldcg(reinterpret_cast<const uint4*>(p));
}

}  // namespace lg

namespace lg {
__host__ __device__ constexpr size_t up256(size_t n) { return (n + 255) / 256 * 256; }
__host__ __device__ constexpr size_t up16(size_t n) { return (n + 15) / 16 * 16; }
}  // namespace lg

//! Byte offsets of the global workspace for `B` rows on a grid of `G` CTAs:
//! the hidden-state and projection exchanges, the tracked logsumexps, then the
//! argmax keys and the barrier counter -- last and contiguous, so the launcher
//! zeroes them with one memset.  Each part 256-byte aligned.
struct LstmGreedyWorkspace {
    size_t hx, dpx, lse, best, bar, total;
};

template <typename T>
__host__ __device__ constexpr LstmGreedyWorkspace lstmGreedyWorkspace(int B, int H, int L, int J,
                                                                       int G) {
    LstmGreedyWorkspace w{};
    w.hx = 0;
    w.dpx = w.hx + lg::up256(sizeof(T) * static_cast<size_t>(L) * 2 * B * H);
    w.lse = w.dpx + lg::up256(sizeof(T) * static_cast<size_t>(B) * J);
    w.best = w.lse + lg::up256(sizeof(float) * 2 * static_cast<size_t>(G) * B * 2);
    w.bar = w.best + lg::up256(sizeof(unsigned long long) * 2 * B);
    w.total = w.bar + 256;
    return w;
}

//! Byte offsets of one CTA's dynamic shared memory, for at most `units` hidden
//! units a CTA (16-byte aligned parts).
struct LstmGreedySmem {
    size_t xa, xb, fr, dps, gates, cells, hown, red, bo, bd, bg, wres, total;
    int W;
};

//! Byte offsets of one CTA's dynamic shared memory for `rows` rows, at most
//! `units` hidden units, `outs` projection outputs and `vocab` head rows a CTA,
//! `res` of them resident.
template <typename T>
__host__ __device__ constexpr LstmGreedySmem lstmGreedySmem(int rows, int E, int H, int L, int J,
                                                            int units, int outs, int vocab,
                                                            int res) {
    LstmGreedySmem s{};
    const int K = E > H ? E : H;
    s.W = K > J ? K : J;
    const size_t R = rows;
    s.xa = 0;                                                   // R x W: a GEMV input
    s.xb = s.xa + lg::up16(sizeof(T) * R * s.W);                // R x W: previous hidden
    s.fr = s.xb + lg::up16(sizeof(T) * R * s.W);                // 2 x R x J: frames by parity
    s.dps = s.fr + lg::up16(sizeof(T) * 2 * R * J);             // R x J: projection copy
    s.gates = s.dps + lg::up16(sizeof(float) * R * J);          // (units x 4) x R
    s.cells = s.gates + lg::up16(sizeof(float) * 4 * R * units);  // (L x units) x R
    s.hown = s.cells + lg::up16(sizeof(float) * L * R * units);   // (L x units) x R
    s.red = s.hown + lg::up16(sizeof(T) * L * R * units);         // warps x R x 4
    s.bo = s.red + lg::up16(sizeof(float) * kLstmWarps * R * 4);  // head bias share
    s.bd = s.bo + lg::up16(sizeof(float) * vocab);                // projection bias share
    s.bg = s.bd + lg::up16(sizeof(float) * outs);                 // (L x units x 4) x {ih, hh}
    s.wres = s.bg + lg::up16(sizeof(float) * L * units * 4 * 2);  // res x J resident head rows
    s.total = s.wres + lg::up16(sizeof(T) * static_cast<size_t>(res) * J);
    return s;
}

template <typename T, int R, bool kTrack>
__global__ void __launch_bounds__(kLstmThreads, 1) lstmGreedyKernel(const LstmGreedyParams<T> p) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    const int G = gridDim.x, cta = blockIdx.x;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int B = p.B, E = p.E, H = p.H, L = p.L, J = p.J;
    int u0, u1, d0, d1, v0, v1;
    lg::share(H, cta, G, &u0, &u1);
    lg::share(J, cta, G, &d0, &d1);
    lg::share(p.V, cta, G, &v0, &v1);
    const int nu = u1 - u0;
    const int units_max = (H + G - 1) / G;
    const int outs_max = (J + G - 1) / G;
    const int vocab_max = (p.V + G - 1) / G;

    // ---- shared memory ----
    const LstmGreedySmem S =
        lstmGreedySmem<T>(R, E, H, L, J, units_max, outs_max, vocab_max, p.res_rows);
    const int W = S.W;
    T* xa = reinterpret_cast<T*>(smem_raw + S.xa);
    T* xb = reinterpret_cast<T*>(smem_raw + S.xb);
    T* fr = reinterpret_cast<T*>(smem_raw + S.fr);
    float* dps = reinterpret_cast<float*>(smem_raw + S.dps);
    float* gates = reinterpret_cast<float*>(smem_raw + S.gates);
    float* cells = reinterpret_cast<float*>(smem_raw + S.cells);
    T* hown = reinterpret_cast<T*>(smem_raw + S.hown);
    float* red = reinterpret_cast<float*>(smem_raw + S.red);
    float* bo = reinterpret_cast<float*>(smem_raw + S.bo);
    float* bd = reinterpret_cast<float*>(smem_raw + S.bd);
    float* bg = reinterpret_cast<float*>(smem_raw + S.bg);
    T* wres = reinterpret_cast<T*>(smem_raw + S.wres);
    const int nres = min(p.res_rows, v1 - v0);  // resident head rows: [v0, v0 + nres)

    __shared__ int s_t[kLstmMaxRows], s_sym[kLstmMaxRows], s_len[kLstmMaxRows],
        s_cnt[kLstmMaxRows], s_tok[kLstmMaxRows], s_emit[kLstmMaxRows];
    __shared__ int s_active, s_any_emit;

    // ---- setup ----
    if (tid < R) {
        const int r = tid;
        int len = 0;
        if (r < B) {
            const int64_t l = p.lengths[r];
            len = l < 0 ? 0 : (l > p.num_frames ? p.num_frames : static_cast<int>(l));
        }
        s_len[r] = len;
        s_t[r] = 0;
        s_sym[r] = 0;
        s_cnt[r] = 0;
    }
    // Own units' state into shared memory and the parity-0 exchange; own
    // projection outputs into the projection exchange.
    for (int idx = tid; idx < L * nu * B; idx += kLstmThreads) {
        const int l = idx / (nu * B), rem = idx - l * nu * B;
        const int ju = rem / B, r = rem - ju * B;
        const int64_t src = (static_cast<int64_t>(l) * B + r) * H + u0 + ju;
        const T h = p.h_in[src];
        cells[(l * units_max + ju) * R + r] = toFloat<T>(p.c_in[src]);
        hown[(l * units_max + ju) * R + r] = h;
        p.hx[((static_cast<int64_t>(l) * 2 + 0) * B + r) * H + u0 + ju] = h;
    }
    for (int idx = tid; idx < (d1 - d0) * B; idx += kLstmThreads) {
        const int dn = idx / B, r = idx - dn * B;
        const int64_t at = static_cast<int64_t>(r) * J + d0 + dn;
        p.dpx[at] = p.dec_proj_in[at];
    }
    for (int idx = tid; idx < R * J; idx += kLstmThreads) {
        const int r = idx / J, j = idx - r * J;
        dps[idx] = r < B ? toFloat<T>(p.dec_proj_in[static_cast<int64_t>(r) * J + j]) : 0.0f;
    }
    // Every bias a step reads, staged once: each was a dependent global load
    // behind its row's dot product.
    for (int v = tid; v < v1 - v0; v += kLstmThreads) {
        bo[v] = p.b_out != nullptr ? toFloat<T>(p.b_out[v0 + v]) : 0.0f;
    }
    for (int n = tid; n < d1 - d0; n += kLstmThreads) {
        bd[n] = p.b_dp != nullptr ? toFloat<T>(p.b_dp[d0 + n]) : 0.0f;
    }
    for (int idx = tid; idx < L * nu * 4; idx += kLstmThreads) {
        const int l = idx / (nu * 4), q = idx - l * nu * 4;
        const int64_t row = static_cast<int64_t>(q & 3) * H + u0 + (q >> 2);
        float* b2 = bg + ((l * units_max) * 4 + q) * 2;
        b2[0] = p.b_ih[l] != nullptr ? toFloat<T>(p.b_ih[l][row]) : 0.0f;
        b2[1] = p.b_hh[l] != nullptr ? toFloat<T>(p.b_hh[l][row]) : 0.0f;
    }
    __syncthreads();
    constexpr int kVec = 16 / sizeof(T);
    // The resident head rows, once: rows [v0, v0 + nres) are contiguous.
    for (int64_t idx = static_cast<int64_t>(tid) * kVec; idx < static_cast<int64_t>(nres) * J;
         idx += kLstmThreads * kVec) {
        detail::cpAsync16(wres + idx, p.w_out + static_cast<int64_t>(v0) * J + idx);
    }
    // Frame 0 of every live row (parity 0).
    for (int idx = tid * kVec; idx < R * J; idx += kLstmThreads * kVec) {
        const int r = idx / J, j = idx - r * J;
        if (r < B && s_len[r] > 0) {
            detail::cpAsync16(fr + r * J + j, p.enc_proj + r * p.enc_stride_b + j);
        }
    }
    unsigned target = 0;
    lg::gridSync(p.bar, target);  // the exchanges hold the start state


    int round = 0, emissions = 0;
    while (true) {
        if (tid == 0) {
            int any = 0;
            for (int r = 0; r < B; ++r) any |= s_t[r] < s_len[r];
            s_active = any;
        }
        detail::cpAsyncWaitAll();
        __syncthreads();
        if (!s_active) break;

        // ---- joint: x = relu(round(enc[t] + dec)) per live row ----
        for (int idx = tid; idx < R * J; idx += kLstmThreads) {
            const int r = idx / J, j = idx - r * J;
            float x = 0.0f;
            if (r < B && s_t[r] < s_len[r]) {
                const float e = toFloat<T>(fr[((s_t[r] & 1) * R + r) * J + j]);
                x = relu<float>(detail::roundTo<T>(e + dps[idx]));
            }
            xa[r * W + j] = fromFloat<T>(x);
        }
        __syncthreads();
        // Prefetch frame t + 1 of every live row under the head.
        for (int idx = tid * kVec; idx < R * J; idx += kLstmThreads * kVec) {
            const int r = idx / J, j = idx - r * J;
            if (r < B) {
                const int t1 = s_t[r] + 1;
                if (t1 < s_len[r]) {
                    detail::cpAsync16(fr + ((t1 & 1) * R + r) * J + j,
                                      p.enc_proj + r * p.enc_stride_b +
                                          static_cast<int64_t>(t1) * p.enc_stride_t + j);
                }
            }
        }
        // ---- this CTA's logits: warp w takes vocabulary rows v0 + w, + 16, ... ----
        uint32_t best[R];
        float lm[R], ls[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            best[r] = 0u;
            lm[r] = -INFINITY;
            ls[r] = 0.0f;
        }
        auto take = [&](int v, const float (&acc)[R]) {
            const float bias = bo[v - v0];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const float logit = detail::roundTo<T>(acc[r] + bias);
                const uint32_t key = gc::packKey<T>(logit, v);
                best[r] = key > best[r] ? key : best[r];
                if constexpr (kTrack) detail::lseCombine(lm[r], ls[r], logit, 1.0f);
            }
        };
        const int nc = J >> 3;
        // The streamed rows' loads go out first and land under the resident work:
        // warp w takes streamed rows v0 + nres + w, + 16, ...
        const int vs = v0 + nres + warp;
        uint4 sw[kLstmChunks];
        if (vs < v1) {
#pragma unroll
            for (int i = 0; i < kLstmChunks; ++i) {
                const int c = lane + 32 * i;
                if (c < nc) sw[i] = lg::ldWeight(p.w_out + static_cast<int64_t>(vs) * J + c * 8);
            }
        }
        // Resident rows from shared memory: warp w takes v0 + w, + 16, ...
        for (int v = v0 + warp; v < v0 + nres; v += kLstmWarps) {
            const T* wrow = wres + static_cast<int64_t>(v - v0) * J;
            float acc[R] = {};
            for (int c = lane; c < nc; c += 32) {
                const uint4 wv = *reinterpret_cast<const uint4*>(wrow + c * 8);
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    if (r < B) {
                        const uint4 xv = *reinterpret_cast<const uint4*>(xa + r * W + c * 8);
#pragma unroll
                        for (int q = 0; q < 4; ++q) {
                            const uint32_t w2 = detail::word(wv, q), x2 = detail::word(xv, q);
                            acc[r] = detail::fmaPair<T, 0, 0>(w2, x2, acc[r]);
                            acc[r] = detail::fmaPair<T, 1, 1>(w2, x2, acc[r]);
                        }
                    }
                }
            }
            lg::warpSum<R>(acc);
            take(v, acc);
        }
        // Streamed rows (one a warp unless a CTA streams more than 16).
        for (int v = vs; v < v1; v += kLstmWarps) {
            if (v != vs) {
#pragma unroll
                for (int i = 0; i < kLstmChunks; ++i) {
                    const int c = lane + 32 * i;
                    if (c < nc) sw[i] = lg::ldWeight(p.w_out + static_cast<int64_t>(v) * J + c * 8);
                }
            }
            float acc[R] = {};
#pragma unroll
            for (int i = 0; i < kLstmChunks; ++i) {
                const int c = lane + 32 * i;
                if (c < nc) {
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        if (r < B) {
                            const uint4 xv = *reinterpret_cast<const uint4*>(xa + r * W + c * 8);
#pragma unroll
                            for (int q = 0; q < 4; ++q) {
                                const uint32_t w2 = detail::word(sw[i], q);
                                const uint32_t x2 = detail::word(xv, q);
                                acc[r] = detail::fmaPair<T, 0, 0>(w2, x2, acc[r]);
                                acc[r] = detail::fmaPair<T, 1, 1>(w2, x2, acc[r]);
                            }
                        }
                    }
                }
            }
            lg::warpSum<R>(acc);
            take(v, acc);
        }
        if (lane == 0) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
                red[(warp * R + r) * 4 + 0] = __uint_as_float(best[r]);
                red[(warp * R + r) * 4 + 1] = lm[r];
                red[(warp * R + r) * 4 + 2] = ls[r];
            }
        }
        __syncthreads();
        if (tid < B && s_t[tid] < s_len[tid]) {
            const int r = tid;
            uint32_t k = 0u;
            float m = -INFINITY, s = 0.0f;
            for (int w = 0; w < kLstmWarps; ++w) {
                const float* c = red + (w * R + r) * 4;
                const uint32_t kw = __float_as_uint(c[0]);
                k = kw > k ? kw : k;
                if constexpr (kTrack) detail::lseCombine(m, s, c[1], c[2]);
            }
            const unsigned long long tagged =
                (static_cast<unsigned long long>(round + 1) << 32) | k;
            atomicMax(&p.best[(round & 1) * B + r], tagged);
            if constexpr (kTrack) {
                float* o = p.lse + ((static_cast<int64_t>(round & 1) * G + cta) * B + r) * 2;
                o[0] = m;
                o[1] = s;
            }
        }
        lg::gridSync(p.bar, target);

        // ---- the decision, the same in every CTA ----
        if (tid == 0) {
            int any = 0;
            for (int r = 0; r < B; ++r) {
                s_emit[r] = 0;
                if (s_t[r] >= s_len[r]) continue;
                const unsigned long long tagged = __ldcg(&p.best[(round & 1) * B + r]);
                const uint32_t key = static_cast<uint32_t>(tagged);
                const int token = gc::keyToken(key);
                if (token != p.blank && s_sym[r] < p.max_sym) {
                    const int cidx = s_cnt[r];
                    if (cta == 0 && cidx < p.cap) {
                        const int64_t o = static_cast<int64_t>(r) * p.cap + cidx;
                        p.tokens[o] = token;
                        p.frames[o] = s_t[r];
                        if constexpr (kTrack) {
                            float m = -INFINITY, s = 0.0f;
                            for (int g = 0; g < G; ++g) {
                                const float* c =
                                    p.lse + ((static_cast<int64_t>(round & 1) * G + g) * B + r) * 2;
                                detail::lseCombine(m, s, __ldcg(c), __ldcg(c + 1));
                            }
                            p.probs[o] = __expf(gc::keyLogit<T>(key) - (m + __logf(s)));
                        }
                    }
                    s_cnt[r] = cidx + 1;
                    s_sym[r] += 1;
                    s_tok[r] = token;
                    s_emit[r] = 1;
                    any = 1;
                } else {
                    s_t[r] += 1;
                    s_sym[r] = 0;
                }
            }
            s_any_emit = any;
        }
        ++round;
        __syncthreads();
        if (!s_any_emit) continue;

        // ---- the LSTM, layer by layer, for the rows that emitted ----
        const int rp = emissions & 1, wp = rp ^ 1;  // read / write parity
        for (int l = 0; l < L; ++l) {
            const int in = l == 0 ? E : H;
            // Layer input (the emitted labels' embeddings, or the layer below's
            // new hidden state) and this layer's previous hidden state, whole.
            for (int idx = tid * kVec; idx < B * in; idx += kLstmThreads * kVec) {
                const int r = idx / in, k = idx - r * in;
                uint4 v;
                if (l == 0) {
                    v = s_emit[r] ? *reinterpret_cast<const uint4*>(
                                        p.emb + static_cast<int64_t>(s_tok[r]) * E + k)
                                  : make_uint4(0, 0, 0, 0);
                } else {
                    const int64_t below = (static_cast<int64_t>(l - 1) * 2 + wp) * B + r;
                    v = lg::ldShared16(p.hx + below * H + k);
                }
                *reinterpret_cast<uint4*>(xa + r * W + k) = v;
            }
            for (int idx = tid * kVec; idx < B * H; idx += kLstmThreads * kVec) {
                const int r = idx / H, k = idx - r * H;
                *reinterpret_cast<uint4*>(xb + r * W + k) =
                    lg::ldShared16(p.hx + ((static_cast<int64_t>(l) * 2 + rp) * B + r) * H + k);
            }
            __syncthreads();
            // Gate rows (unit j, gate g): warp per row, ih and hh dots in one sum.
            for (int q = warp; q < 4 * nu; q += kLstmWarps) {
                const int ju = q >> 2, g = q & 3;
                const int64_t row = static_cast<int64_t>(g) * H + u0 + ju;
                // The ih and hh rows as one pair of in-flight rows (in == H).
                const T* wr[2] = {p.w_ih[l] + row * in, p.w_hh[l] + row * H};
                const T* xr[2] = {xa, xb};
                float acc[2][R] = {};
                lg::laneDots<T, R, 2>(wr, xr, H, W, 2, B, lane, acc);
                lg::warpSum<2, R>(acc);
                if (lane == 0) {
                    const float* b2 = bg + ((l * units_max) * 4 + q) * 2;
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        // One accumulator for both dots, then b_ih, then b_hh,
                        // as the recurrent kernel sums them.
                        float a = acc[0][r] + acc[1][r];
                        if (p.b_ih[l] != nullptr) a += b2[0];
                        if (p.b_hh[l] != nullptr) a += b2[1];
                        gates[q * R + r] = a;
                    }
                }
            }
            __syncthreads();
            // The cell update for the emitting rows; the others keep theirs.
            for (int idx = tid; idx < nu * B; idx += kLstmThreads) {
                const int ju = idx / B, r = idx - ju * B;
                const int si = (l * units_max + ju) * R + r;
                T h = hown[si];
                if (s_emit[r]) {
                    const float* gq = gates + (ju * 4) * R + r;
                    const float ig = fastSigmoid(gq[0]);
                    const float fg = fastSigmoid(gq[R]);
                    const float cg = tanhf(gq[2 * R]);
                    const float og = fastSigmoid(gq[3 * R]);
                    const float c = fg * cells[si] + ig * cg;
                    const T cv = fromFloat<T>(c);
                    h = fromFloat<T>(og * tanhf(c));
                    cells[si] = toFloat<T>(cv);
                    hown[si] = h;
                }
                __stcg(p.hx + ((static_cast<int64_t>(l) * 2 + wp) * B + r) * H + u0 + ju, h);
            }
            lg::gridSync(p.bar, target);
        }
        // ---- decoder projection of the new top hidden state ----
        for (int idx = tid * kVec; idx < B * H; idx += kLstmThreads * kVec) {
            const int r = idx / H, k = idx - r * H;
            *reinterpret_cast<uint4*>(xa + r * W + k) =
                lg::ldShared16(p.hx + ((static_cast<int64_t>(L - 1) * 2 + wp) * B + r) * H + k);
        }
        __syncthreads();
        for (int n = d0 + warp; n < d1; n += kLstmWarps) {
            const T* wr[1] = {p.w_dp + static_cast<int64_t>(n) * H};
            const T* xr[1] = {xa};
            float acc[1][R] = {};
            lg::laneDots<T, R, 1>(wr, xr, H, W, 1, B, lane, acc);
            lg::warpSum<1, R>(acc);
            if (lane == 0) {
                const float bias = bd[n - d0];
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    if (r < B && s_emit[r]) {
                        __stcg(p.dpx + static_cast<int64_t>(r) * J + n,
                               fromFloat<T>(detail::roundTo<T>(acc[0][r] + bias)));
                    }
                }
            }
        }
        lg::gridSync(p.bar, target);
        for (int idx = tid; idx < B * J; idx += kLstmThreads) {
            dps[idx] = toFloat<T>(__ldcg(p.dpx + idx));
        }
        ++emissions;
    }

    // ---- the carried state back: own units' hidden and cell, the projection ----
    for (int idx = tid; idx < L * nu * B; idx += kLstmThreads) {
        const int l = idx / (nu * B), rem = idx - l * nu * B;
        const int ju = rem / B, r = rem - ju * B;
        const int64_t dst = (static_cast<int64_t>(l) * B + r) * H + u0 + ju;
        const int si = (l * units_max + ju) * R + r;
        p.h_out[dst] = hown[si];
        p.c_out[dst] = fromFloat<T>(cells[si]);
    }
    if (cta == 0 && tid < B) p.counts[tid] = s_cnt[tid];
    for (int idx = tid; idx < (d1 - d0) * B; idx += kLstmThreads) {
        const int dn = idx / B, r = idx - dn * B;
        p.dec_proj_out[static_cast<int64_t>(r) * J + d0 + dn] =
            __ldcg(p.dpx + static_cast<int64_t>(r) * J + d0 + dn);
    }
}

//! CTAs a launch uses: one per SM, capped by the work (every CTA must own at
//! least one vocabulary row) and by `max_ctas` when positive.
inline int lstmGreedyGrid(int V, int max_ctas) {
    int dev = 0, sms = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    int g = sms;
    if (max_ctas > 0 && max_ctas < g) g = max_ctas;
    if (V < g) g = V;
    return g;
}

template <typename T, int R, bool kTrack>
cudaError_t launchLstmGreedy(LstmGreedyParams<T> p, int G, cudaStream_t stream) {
    auto kernel = lstmGreedyKernel<T, R, kTrack>;
    const int units = (p.H + G - 1) / G, outs = (p.J + G - 1) / G, vocab = (p.V + G - 1) / G;
    // Every byte of shared memory the buffers leave goes to resident head rows:
    // at batch 1 on Nemotron that is ~72 of a CTA's ~77, so a step streams ~5.
    cudaFuncAttributes attr;
    int dev = 0, optin = 0;
    cudaError_t st = cudaFuncGetAttributes(&attr, kernel);
    if (st == cudaSuccess) st = cudaGetDevice(&dev);
    if (st == cudaSuccess) {
        st = cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    }
    if (st != cudaSuccess) return st;
    const size_t base = lstmGreedySmem<T>(R, p.E, p.H, p.L, p.J, units, outs, vocab, 0).total;
    // The driver keeps 1 KB of every block's shared memory for itself.
    const int64_t room = static_cast<int64_t>(optin) - static_cast<int64_t>(attr.sharedSizeBytes) -
                         static_cast<int64_t>(base) - 1024;
    if (room < 0) return cudaErrorInvalidConfiguration;
    const int64_t fit = room / (static_cast<int64_t>(p.J) * sizeof(T));
    p.res_rows = static_cast<int>(fit < vocab ? fit : vocab);
    const size_t smem =
        lstmGreedySmem<T>(R, p.E, p.H, p.L, p.J, units, outs, vocab, p.res_rows).total;
    st = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                              static_cast<int>(smem));
    if (st != cudaSuccess) {
        cudaGetLastError();
        return st;
    }
    // Every CTA must be resident at once: the grid barrier waits for all of them.
    int per_sm = 0, sms = 0;
    st = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, kLstmThreads, smem);
    if (st == cudaSuccess) st = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    if (st != cudaSuccess) return st;
    if (per_sm * sms < G) return cudaErrorCooperativeLaunchTooLarge;
    // The argmax keys and the barrier counter: contiguous, at the workspace's end.
    st = cudaMemsetAsync(p.best, 0,
                         reinterpret_cast<char*>(p.bar) + 256 - reinterpret_cast<char*>(p.best),
                         stream);
    if (st != cudaSuccess) return st;
    void* args[] = {&p};
    return cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(kernel), dim3(G),
                                       dim3(kLstmThreads), args, smem, stream);
}

//! Launch the fused LSTM-predictor greedy decode over `G` CTAs (one resident per
//! SM).  Half precision; E, H and J multiples of 8; V <= 65535; B <=
//! kLstmMaxRows.  `workspace` holds lstmGreedyWorkspace(B, H, L, J, G).total
//! bytes, 256-byte aligned; nothing in it need be initialised.
template <typename T>
cudaError_t LstmGreedyDecode(LstmGreedyParams<T> p, void* workspace, int G,
                             cudaStream_t stream) {
    if (p.B == 0) return cudaSuccess;
    const LstmGreedyWorkspace ws = lstmGreedyWorkspace<T>(p.B, p.H, p.L, p.J, G);
    char* base = static_cast<char*>(workspace);
    p.hx = reinterpret_cast<T*>(base + ws.hx);
    p.dpx = reinterpret_cast<T*>(base + ws.dpx);
    p.lse = reinterpret_cast<float*>(base + ws.lse);
    p.best = reinterpret_cast<unsigned long long*>(base + ws.best);
    p.bar = reinterpret_cast<unsigned*>(base + ws.bar);
    if (p.B < 0 || p.B > kLstmMaxRows || p.L < 1 || p.L > kLstmMaxLayers || p.E != p.H ||
        p.H % 8 != 0 || p.J % 8 != 0 || p.H > kLstmMaxK || p.J > kLstmMaxK || p.V < 1 ||
        p.V > kLstmMaxVocab || p.num_frames < 0 || G < 1) {
        return cudaErrorInvalidValue;
    }
    const bool track = p.probs != nullptr;
    switch (p.B) {
        case 1:
            return track ? launchLstmGreedy<T, 1, true>(p, G, stream)
                         : launchLstmGreedy<T, 1, false>(p, G, stream);
        case 2:
            return track ? launchLstmGreedy<T, 2, true>(p, G, stream)
                         : launchLstmGreedy<T, 2, false>(p, G, stream);
        default:
            return track ? launchLstmGreedy<T, 4, true>(p, G, stream)
                         : launchLstmGreedy<T, 4, false>(p, G, stream);
    }
}

}  // namespace transducer
}  // namespace oasr
