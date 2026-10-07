// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Fused modified beam search (at most one symbol per frame) for a
// stateless-predictor transducer: one launch runs every frame of a chunk.
//
// Each utterance's `k` hypotheses are held by one CTA -- or a cluster pair of
// CTAs, see below -- which for every frame inside the utterance runs, on the
// device:
//
//   1. the joiner over all k hypotheses -- `act(enc[t] + dec_proj[j])` and the
//      output GEMV, which reads the joiner head once for the k rows;
//   2. the selection -- log-softmax, score add, top-k over the k * V
//      candidates, the (parent, label) split -- with beam_topk.cuh's results,
//      recording the frame's back-pointers;
//   3. the predictor for the new hypotheses that took a label (label-window
//      shift, embedding, grouped convolution, ReLU, decoder-projection GEMV).
//      A hypothesis that took blank keeps its parent's window, so its decoder
//      projection is copied rather than recomputed.
//
// and after the last frame walks the back-pointers from every final slot, so
// the host reads each hypothesis's tokens rather than the history.
//
// What it replaces: the per-frame op-by-op step (predictor and decoder
// projection over every hypothesis, joiner GEMM, the fused selection), about
// ten kernels a frame even replayed from CUDA graphs, each a few microseconds
// at decode shapes, with the decoder projection recomputed for every
// hypothesis whether its window moved or not; and a host walk of the history
// that cost a quarter of the decode.
//
// The GEMVs.  A weight is read K-major (`(K, ld)`, `ld` a multiple of
// kGemvTile).  The 16 warps are two output groups of eight; a warp takes a
// slice of K and a lane OUT consecutive outputs (one 8- or 16-byte load per K
// row), and every weight element read is used for all R rows.  The weights
// stream through registers double-buffered -- the next stage's loads are in
// flight while this stage's multiply-adds run -- because at R rows a stage is
// R times the arithmetic of the greedy kernel's and no longer hides a load's
// latency on its own.  The eight slice partials of an output are summed in
// slice order through shared memory: a fixed order, so a result never depends
// on the batch or on timing.  The multiply-adds are the greedy kernel's
// (`fmaPair`): sm_100+'s mixed-precision FMA, bit-identical to
// `fmaf(float(w), float(x), acc)`.
//
// Cluster pairs.  A frame is bounded by how fast one SM pulls the two weights
// (512 KB each for the icefall transducer) out of L2.  While the batch leaves
// an SM for each, two CTAs in a thread-block cluster (sm_90+) serve an
// utterance: they take alternate 256-wide passes of every GEMV, with a lane
// owning 4 outputs instead of 8 so all 16 warps of both work, and push their
// logits, row maxima and decoder projections into each other's shared memory
// before a cluster barrier.  Both then run the selection on the same data, so
// both hold the same beam, and the lead CTA writes it.  An output's arithmetic
// is the same chain whichever CTA computes it and however wide a lane's
// outputs are -- the K slices are fixed -- so a cluster pair and a lone CTA
// produce the same bits, and a row's result does not depend on its batch.
//
// The selection runs on every warp, W = 16 / R of them per hypothesis row.
// The row maximum comes out of the joiner's epilogue (exact, so any reduction
// order gives it); one warp per row then exponentiates and sums in torch's
// order -- lane l adds elements l, l + 32, ... in sequence, then xor
// butterflies -- so the normaliser is `softmax_warp_forward`'s bit for bit, as
// beam_topk.cuh's is.  Each warp then extracts its own candidates' top k as
// packed 64-bit keys (`rankKey`: the total's bits made monotone, then the
// complemented flat index, so one integer compare is beam_topk.cuh's order),
// and warp 0 merges the 16 sorted lists.  Candidates are ranked by total, then
// by lower flat index `j * V + v`: a total order, so the selection does not
// depend on how the candidates were split.
//
// Numerical contract: the greedy kernel's (see greedy_decode.cuh) for
// everything up to the logits -- every rounding point of the op-by-op path is
// kept, the GEMV accumulation order is this kernel's own (one fp32 chain per
// output; the library GEMMs it replaces also round split-K partials to the
// activation dtype at some shapes) -- and beam_topk.cuh's for the selection,
// which is bit-identical to torch's given the same logits.  So a logit can land
// an ulp or so from the op-by-op path's, and on data where every sum is exact
// the two agree bit for bit.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstdint>
#include <cooperative_groups.h>
#include <type_traits>
#include <oasr/common/math.h>
#include <oasr/common/vec_dtypes.h>
#include <oasr/transducer/beam_topk.cuh>
#include <oasr/transducer/greedy_decode.cuh>

namespace oasr {
namespace transducer {

//! Threads per CTA, and the warps that split each output group's K.
constexpr int kBeamThreads = 512;
constexpr int kBeamWarps = kBeamThreads / 32;
constexpr int kBeamSlices = 8;
//! Outputs a lane owns: 8 when one CTA searches an utterance (one 512-wide
//! pass per 512 outputs), 4 in a cluster pair (alternate 256-wide passes, so
//! every warp of both CTAs works).  A pass is two groups x 32 lanes x OUT.
template <int OUT>
constexpr int beamTile() {
    return 64 * OUT;
}
//! Widest beam: the k rows' GEMV accumulators live in registers.
constexpr int kBeamDecodeMaxBeam = 8;
//! Largest vocabulary: torch's warp log-softmax, whose normaliser the
//! selection reproduces, covers 1024.
constexpr int kBeamDecodeMaxVocab = 1024;

template <typename T>
struct StatelessBeamParams {
    // ---- per-utterance inputs ----
    const T* enc_proj;  // (B, frames, J), last dim contiguous: encoder in joiner space
    int64_t enc_stride_b;
    int64_t enc_stride_t;
    const int64_t* lengths;     // (B,) valid frames per utterance
    const int64_t* context_in;  // (B, k, ctx) predictor label windows
    const float* scores_in;     // (B, k)
    // ---- weights (the greedy kernel's layouts) ----
    const T* w_out_t;  // (J, ld_out) joiner output projection, K-major
    const T* b_out;    // (>= V,) or nullptr
    const T* emb;      // (V, D) predictor embedding
    const T* conv_w;   // group > 1: (D, 1, ctx, group); group == 1: (ctx, 1, D); nullptr if ctx == 1
    const T* w_dp_t;   // (D, ld_dp) decoder projection, K-major
    const T* b_dp;     // (J,) or nullptr
    // ---- outputs ----
    int64_t* context_out;  // (B, k, ctx), may alias context_in
    float* scores_out;     // (B, k), may alias scores_in
    int64_t* parents;      // (frames, B, k) slot each new slot extended
    int64_t* labels;       // (frames, B, k) label it took (blank = none)
    // Optional: the chunk walked back from every final slot, packed int32 --
    // roots (B * k), counts (B * k), tokens (B * k * frames); nullptr skips it.
    int32_t* walk;
    // ---- sizes / constants ----
    int B;
    int frames;
    int k;
    int J;       // joiner dim (multiple of kStageK)
    int D;       // decoder dim (multiple of kStageK)
    int V;       // vocabulary
    int ld_out;  // multiple of kGemvTile, >= V
    int ld_dp;   // multiple of kGemvTile, >= J
    int ctx;
    int group;
    int blank;
    int activation;
};

namespace beam_decode_detail {

//! K values per GEMV stage.  A lane holds R x OUT accumulators and two
//! staged sets of KS x OUT weights (registers, double-buffered): as deep as
//! keeps that within 80 of its 128 registers.
template <int R, int OUT>
constexpr int stageK() {
    return (8 + R) * OUT <= 80 ? 8 : ((4 + R) * OUT <= 80 ? 4 : 2);
}

//! Rows whose slice partials share the reduction buffer at once: all of them
//! up to four rows (one barrier pair per pass), else the most that keep the
//! buffer at 32 KB.
template <int R, int OUT>
constexpr int rowGroup() {
    return R <= 4 ? R : 16 / OUT;
}

__host__ __device__ constexpr int roundUp8(int x) { return (x + 7) / 8 * 8; }

//! A lane's OUT consecutive half-precision weights of one K row.
template <int OUT>
using LaneVec = typename std::conditional<OUT == 8, uint4, uint2>::type;

template <typename T, int OUT, int KS>
__device__ __forceinline__ void loadStage(LaneVec<OUT> (&w)[KS], const T* base, int kk, int ldw) {
#pragma unroll
    for (int u = 0; u < KS; ++u) {
        w[u] = *reinterpret_cast<const LaneVec<OUT>*>(base + static_cast<int64_t>(kk + u) * ldw);
    }
}

__device__ __forceinline__ uint32_t laneWord(const uint4& v, int i) { return detail::word(v, i); }
__device__ __forceinline__ uint32_t laneWord(const uint2& v, int i) { return i == 0 ? v.x : v.y; }

// acc[r][i] += sum over K in [kk, kk + KS) of xs[r][K] * w[K - kk][i].
template <typename T, int R, int OUT, int KS>
__device__ __forceinline__ void fmaStage(float (&acc)[R][OUT], const LaneVec<OUT> (&w)[KS],
                                         const T* xs, int ldx, int kk) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
        uint32_t xw[KS / 2];
        if constexpr (KS == 8) {
            const uint4 v = *reinterpret_cast<const uint4*>(xs + r * ldx + kk);
            xw[0] = v.x;
            xw[1] = v.y;
            xw[2] = v.z;
            xw[3] = v.w;
        } else if constexpr (KS == 4) {
            const uint2 v = *reinterpret_cast<const uint2*>(xs + r * ldx + kk);
            xw[0] = v.x;
            xw[1] = v.y;
        } else {
            static_assert(KS == 2, "stage of 2, 4 or 8");
            xw[0] = *reinterpret_cast<const uint32_t*>(xs + r * ldx + kk);
        }
#pragma unroll
        for (int u = 0; u < KS; ++u) {
            const uint32_t x2 = xw[u >> 1];
#pragma unroll
            for (int i = 0; i < OUT / 2; ++i) {
                const uint32_t w2 = laneWord(w[u], i);
                float& lo = acc[r][2 * i];
                float& hi = acc[r][2 * i + 1];
                if (u & 1) {
                    lo = detail::fmaPair<T, 0, 1>(w2, x2, lo);
                    hi = detail::fmaPair<T, 1, 1>(w2, x2, hi);
                } else {
                    lo = detail::fmaPair<T, 0, 0>(w2, x2, lo);
                    hi = detail::fmaPair<T, 1, 0>(w2, x2, hi);
                }
            }
        }
    }
}

// acc[r][i] = sum over K in [kb, ke) of xs[r][K] * wt[K][col + i]: the OUT
// outputs a lane owns, for every row, K in ascending order -- the same chain
// for an output whatever OUT is.  The weights are double-buffered through
// registers.  kb and ke are multiples of KS.
template <typename T, int R, int OUT>
__device__ __forceinline__ void rowsSliceFma(float (&acc)[R][OUT], const T* __restrict__ wt,
                                             int ldw, int col, int kb, int ke, const T* xs,
                                             int ldx) {
    constexpr int KS = stageK<R, OUT>();
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int i = 0; i < OUT; ++i) acc[r][i] = 0.0f;
    }
    if (kb >= ke) return;
    const T* base = wt + col;
    LaneVec<OUT> wa[KS], wb[KS];
    loadStage<T, OUT, KS>(wa, base, kb, ldw);
    int kk = kb;
    while (true) {
        const bool more = kk + KS < ke;
        if (more) loadStage<T, OUT, KS>(wb, base, kk + KS, ldw);
        fmaStage<T, R, OUT, KS>(acc, wa, xs, ldx, kk);
        kk += KS;
        if (!more) break;
        const bool more2 = kk + KS < ke;
        if (more2) loadStage<T, OUT, KS>(wa, base, kk + KS, ldw);
        fmaStage<T, R, OUT, KS>(acc, wb, xs, ldx, kk);
        kk += KS;
        if (!more2) break;
    }
}

// One GEMV pass over every row: out(r, n0 + n) for the pass's beamTile<OUT>()
// outputs = sum over the K slices, in slice order, handed to `epi(r, n0 + n,
// sum)`.  A warp's lanes hand over 32 consecutive outputs of one row.  Ends
// with a barrier, so `part` and the epilogue's writes are settled when it
// returns.  An output's arithmetic is the same whichever CTA of a cluster, and
// whichever pass, computes it.
template <typename T, int R, int OUT, typename Epilogue>
__device__ __forceinline__ void gemvPass(const T* __restrict__ wt, int ldw, int n0, int kb, int ke,
                                         const T* xs, int ldx, float* part, int slice, int col,
                                         int tid, Epilogue epi) {
    constexpr int G = rowGroup<R, OUT>();
    constexpr int kTile = beamTile<OUT>();
    float acc[R][OUT];
    rowsSliceFma<T, R, OUT>(acc, wt, ldw, n0 + col, kb, ke, xs, ldx);
#pragma unroll
    for (int g0 = 0; g0 < R; g0 += G) {
#pragma unroll
        for (int gi = 0; gi < G; ++gi) {
            float* dst = part + (gi * kBeamSlices + slice) * kTile + col;
#pragma unroll
            for (int i = 0; i < OUT; i += 4) {
                *reinterpret_cast<float4*>(dst + i) =
                    make_float4(acc[g0 + gi][i], acc[g0 + gi][i + 1], acc[g0 + gi][i + 2],
                                acc[g0 + gi][i + 3]);
            }
        }
        __syncthreads();
#pragma unroll
        for (int q = 0; q < (G * kTile + kBeamThreads - 1) / kBeamThreads; ++q) {
            const int idx = q * kBeamThreads + tid;
            if (G * kTile % kBeamThreads == 0 || idx < G * kTile) {  // warp-uniform
                const int gi = idx / kTile;
                const int n = idx - gi * kTile;
                float v = 0.0f;
#pragma unroll
                for (int sl = 0; sl < kBeamSlices; ++sl) {
                    v += part[(gi * kBeamSlices + sl) * kTile + n];
                }
                epi(g0 + gi, n0 + n, v);
            }
        }
        __syncthreads();
    }
}

// The CTAs of a cluster that search one utterance together (sm_90+): this
// CTA's rank, the cluster's size, a peer's view of a shared-memory address,
// and the cluster-wide barrier (release / acquire, so remote writes before it
// are visible after it).  One CTA, and no-ops, below sm_90.
struct BeamCluster {
    int rank = 0;
    int size = 1;
    __device__ __forceinline__ void init() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        namespace cg = cooperative_groups;
        rank = static_cast<int>(cg::this_cluster().block_rank());
        size = static_cast<int>(cg::this_cluster().num_blocks());
#endif
    }
    template <typename P>
    __device__ __forceinline__ P* peer(P* local) const {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        if (size > 1) {
            return cooperative_groups::this_cluster().map_shared_rank(local, rank ^ 1);
        }
#endif
        return nullptr;
    }
    __device__ __forceinline__ void sync() const {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        if (size > 1) {
            cooperative_groups::this_cluster().sync();
            return;
        }
#endif
        __syncthreads();
    }
};

// `dst[i] = src[i]` for GS consecutive elements, as one vector load when GS
// half-precision values make a whole one (the callers' offsets are multiples
// of GS elements from 16-byte-aligned rows).
template <typename T, int GS>
__device__ __forceinline__ void loadGroup(T (&dst)[GS], const T* src) {
    if constexpr (GS * sizeof(T) == 16) {
        *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
    } else if constexpr (GS * sizeof(T) == 8) {
        *reinterpret_cast<uint2*>(dst) = *reinterpret_cast<const uint2*>(src);
    } else if constexpr (GS * sizeof(T) == 4) {
        *reinterpret_cast<uint32_t*>(dst) = *reinterpret_cast<const uint32_t*>(src);
    } else {
#pragma unroll
        for (int i = 0; i < GS; ++i) dst[i] = src[i];
    }
}

// The predictor hidden state of channel `d` for every row: relu(round(conv
// over the window's embeddings)), each row accumulating in the op-by-op
// path's fmaf order (window position, then group member).  GS is the conv
// group size when it is a compile-time case, else 0 (run-time `gs`).
template <typename T, int R, int GS>
__device__ __forceinline__ void groupedHidden(T* hs, int D, int d, const T* emb, const T* conv_w,
                                              const int (*win)[kMaxContext], int ctx, int gs) {
    const int g = GS > 0 ? GS : gs;
    const int c0 = (d / g) * g;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;
    for (int s = 0; s < ctx; ++s) {
        const T* w = conv_w + (static_cast<int64_t>(d) * ctx + s) * g;
        if constexpr (GS > 0) {
            alignas(16) T wv[GS];
            loadGroup<T, GS>(wv, w);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                alignas(16) T ev[GS];
                loadGroup<T, GS>(ev, emb + static_cast<int64_t>(win[r][s]) * D + c0);
#pragma unroll
                for (int i = 0; i < GS; ++i) acc[r] = fmaf(toFloat<T>(ev[i]), toFloat<T>(wv[i]), acc[r]);
            }
        } else {
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const T* e = emb + static_cast<int64_t>(win[r][s]) * D + c0;
                for (int i = 0; i < g; ++i) acc[r] = fmaf(toFloat<T>(e[i]), toFloat<T>(w[i]), acc[r]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) hs[r * D + d] = relu<T>(fromFloat<T>(acc[r]));
}

// torch's `Add`, reduced by xor butterflies of a run-time width.
__device__ __forceinline__ float warpSumLikeTorch(float v, int width) {
    for (int off = width / 2; off > 0; off /= 2) v = v + __shfl_xor_sync(0xffffffffu, v, off, width);
    return v;
}

// A candidate as one integer whose unsigned order is the selection order:
// higher total first, equal totals by lower flat index (beam_topk.cuh's
// `ranksAbove`).  The high word is the total's bits made monotone (-0 ranks
// as +0, since `==` holds between them), the low word the complemented index.
// Comparing two is two instructions rather than the float/int chain's seven.
// No candidate maps to 0, which marks an empty slot.
__device__ __forceinline__ uint64_t rankKey(float v, int idx) {
    uint32_t u = __float_as_uint(v);
    if (u == 0x80000000u) u = 0u;
    const uint32_t ord = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return (static_cast<uint64_t>(ord) << 32) | static_cast<uint32_t>(~static_cast<uint32_t>(idx));
}

__device__ __forceinline__ float keyValue(uint64_t key) {
    const uint32_t ord = static_cast<uint32_t>(key >> 32);
    return __uint_as_float((ord & 0x80000000u) ? (ord & 0x7fffffffu) : ~ord);
}

__device__ __forceinline__ int keyIndex(uint64_t key) {
    return static_cast<int>(~static_cast<uint32_t>(key));
}

__device__ __forceinline__ uint64_t warpMaxKey(uint64_t key) {
#pragma unroll
    for (int off = 16; off > 0; off /= 2) {
        const uint64_t o = __shfl_xor_sync(0xffffffffu, key, off);
        key = o > key ? o : key;
    }
    return key;
}

}  // namespace beam_decode_detail

//! Dynamic shared memory of one CTA: the slice partials (float), then the
//! encoder frame double buffer,
//! joiner input, predictor hidden, decoder projection double buffer, logits
//! (double-buffered by frame, which is what lets a cluster peer write the next
//! frame's while this CTA still reads this one's) and the two biases (all T).
template <typename T, int R, int OUT>
__host__ __device__ constexpr size_t statelessBeamSmemBytes(int J, int D, int V) {
    return sizeof(float) * static_cast<size_t>(beam_decode_detail::rowGroup<R, OUT>()) *
               kBeamSlices * beamTile<OUT>() +
           sizeof(T) * (static_cast<size_t>(2) * J + static_cast<size_t>(R) * J +
                        static_cast<size_t>(R) * D + static_cast<size_t>(2) * R * J +
                        static_cast<size_t>(2 * R + 1) * beam_decode_detail::roundUp8(V) +
                        static_cast<size_t>(J));
}

// One CTA per utterance; one CTA per SM (see greedy_decode.cuh's note on the
// register budget).
template <typename T, int R, int OUT>
__global__ void __launch_bounds__(kBeamThreads, 1)
    statelessBeamKernel(const StatelessBeamParams<T> p) {
    namespace bd = beam_decode_detail;
    constexpr int G = bd::rowGroup<R, OUT>();
    constexpr int kTile = beamTile<OUT>();
    //! Selection warps per hypothesis row, and the candidates a lane holds.
    constexpr int W = kBeamWarps / R;
    constexpr int E = kBeamDecodeMaxVocab / (32 * W);
    extern __shared__ __align__(16) unsigned char smem_raw[];
    const int J = p.J, D = p.D, V = p.V, k = p.k, ctx = p.ctx;
    const int ldl = bd::roundUp8(V);
    float* part = reinterpret_cast<float*>(smem_raw);  // G x slices x tile
    T* es = reinterpret_cast<T*>(part + G * kBeamSlices * kTile);  // 2 x J encoder frames
    T* xs = es + 2 * J;   // R x J joiner input (after activation)
    T* hs = xs + R * J;   // R x D predictor hidden (after ReLU)
    T* dps = hs + R * D;  // 2 x R x J decoder projection (current / next beam)
    T* lgs = dps + 2 * R * J;         // 2 x R x ldl logits (by frame parity)
    T* bias_out = lgs + 2 * R * ldl;  // ldl joiner output bias
    T* bias_dp = bias_out + ldl;  // J decoder projection bias

    __shared__ uint64_t wcand[kBeamWarps * R];
    __shared__ float pmaxs[2][R][kBeamWarps];
    __shared__ float row_lsum[R];
    __shared__ int s_ctx[2][R][kMaxContext];
    __shared__ float s_score[2][R];
    __shared__ int s_par[R], s_emit[R];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int slice = warp % kBeamSlices;
    const int col = (warp / kBeamSlices) * (kTile / 2) + lane * OUT;
    bd::BeamCluster cluster;
    cluster.init();
    const int b = blockIdx.x / cluster.size;
    //! The cluster's lead CTA writes the outputs.  The CTAs of a cluster take
    //! alternate GEMV passes.
    const bool lead = cluster.rank == 0;
    const int pass0 = cluster.rank * kTile;
    const int pass_step = cluster.size * kTile;
    const int64_t len64 = p.lengths[b];
    const int len = len64 < 0 ? 0 : (len64 > p.frames ? p.frames : static_cast<int>(len64));
    constexpr int kVec = 16 / sizeof(T);

    // torch's warp log-softmax geometry for a row of V (softmax_warp_forward).
    int pow2 = 1;
    while (pow2 < V) pow2 *= 2;
    const int width = pow2 < 32 ? pow2 : 32;
    const int iters = pow2 / width;

    if (tid < R) {
        const int r = tid;
        for (int c = 0; c < ctx; ++c) {
            s_ctx[0][r][c] =
                r < k ? static_cast<int>(p.context_in[(static_cast<int64_t>(b) * k + r) * ctx + c])
                      : p.blank;
            // Rows past the beam are never written by a frame, but the
            // predictor reads every row's window: keep theirs valid.
            s_ctx[1][r][c] = s_ctx[0][r][c];
        }
        s_score[0][r] = r < k ? p.scores_in[static_cast<int64_t>(b) * k + r] : -INFINITY;
        s_emit[r] = r < k;  // the first predictor pass covers every hypothesis
    }
    for (int v = tid; v < ldl; v += kBeamThreads) {
        bias_out[v] = (p.b_out != nullptr && v < V) ? p.b_out[v] : fromFloat<T>(0.0f);
    }
    for (int n = tid; n < J; n += kBeamThreads) {
        bias_dp[n] = p.b_dp != nullptr ? p.b_dp[n] : fromFloat<T>(0.0f);
    }
    cluster.sync();  // a peer's shared memory exists before anyone writes it

    // K-slices of the two GEMVs, in multiples of kStageK whatever this
    // instantiation's stage, so a row's accumulation order -- and so its
    // logits -- never depends on how many rows the CTA holds or how wide a
    // lane's outputs are.
    static_assert(kStageK % bd::stageK<R, OUT>() == 0, "a stage divides kStageK");
    const int kj = ((J + kBeamSlices - 1) / kBeamSlices + kStageK - 1) / kStageK * kStageK;
    const int kd = ((D + kBeamSlices - 1) / kBeamSlices + kStageK - 1) / kStageK * kStageK;
    const int jb = min(J, slice * kj), je = min(J, jb + kj);
    const int db = min(D, slice * kd), de = min(D, db + kd);

    // The predictor for the windows `win` into `dst`, for the rows `s_emit`
    // marks.  The hidden state is computed for every row -- the rows that did
    // not take a label hold valid windows, and branch-free rows keep the
    // embedding loads in flight together -- and only marked rows are stored.
    auto predictor = [&](const int (*win)[kMaxContext], T* dst) {
        for (int d = tid; d < D; d += kBeamThreads) {
            if (ctx == 1) {
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    hs[r * D + d] = relu<T>(p.emb[static_cast<int64_t>(win[r][0]) * D + d]);
                }
            } else if (p.group == 1) {
                float acc[R];
#pragma unroll
                for (int r = 0; r < R; ++r) acc[r] = 0.0f;
                for (int s = 0; s < ctx; ++s) {
                    const float w = toFloat<T>(p.conv_w[static_cast<int64_t>(s) * D + d]);
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        acc[r] = fmaf(toFloat<T>(p.emb[static_cast<int64_t>(win[r][s]) * D + d]),
                                      w, acc[r]);
                    }
                }
#pragma unroll
                for (int r = 0; r < R; ++r) hs[r * D + d] = relu<T>(fromFloat<T>(acc[r]));
            } else if (p.group == 4) {
                bd::groupedHidden<T, R, 4>(hs, D, d, p.emb, p.conv_w, win, ctx, 4);
            } else if (p.group == 2) {
                bd::groupedHidden<T, R, 2>(hs, D, d, p.emb, p.conv_w, win, ctx, 2);
            } else if (p.group == 8) {
                bd::groupedHidden<T, R, 8>(hs, D, d, p.emb, p.conv_w, win, ctx, 8);
            } else {
                bd::groupedHidden<T, R, 0>(hs, D, d, p.emb, p.conv_w, win, ctx, p.group);
            }
        }
        __syncthreads();
        for (int n0 = pass0; n0 < J; n0 += pass_step) {
            bd::gemvPass<T, R, OUT>(p.w_dp_t, p.ld_dp, n0, db, de, hs, D, part, slice, col, tid,
                               [&](int r, int n, float sum) {
                                   if (r < k && s_emit[r] && n < J) {
                                       const T v = fromFloat<T>(sum + toFloat<T>(bias_dp[n]));
                                       dst[r * J + n] = v;
                                       if (cluster.size > 1) *cluster.peer(dst + r * J + n) = v;
                                   }
                               });
        }
        if (cluster.size > 1) cluster.sync();
    };

    int cur = 0;
    if (len > 0) {
        // Frame 0 into buffer 0, under the first predictor pass.
        for (int idx = tid * kVec; idx < J; idx += kBeamThreads * kVec) {
            detail::cpAsync16(es + idx, p.enc_proj + static_cast<int64_t>(b) * p.enc_stride_b + idx);
        }
        predictor(s_ctx[0], dps);
        detail::cpAsyncWaitAll();
        __syncthreads();
    }

    // The selection's row for this warp, and how many candidates a lane holds.
    const int srow = warp / W;
    const int wi = warp - srow * W;
    const int ecount = (V + 32 * W - 1) / (32 * W);
    //! Row-maximum chunks the joiner's epilogue writes (32 outputs each).
    const int nchunks = min(kBeamWarps, (V + 31) / 32);

    int eb = 0;
    for (int t = 0; t < len; ++t) {
        const int nxt = cur ^ 1;
        T* dcur = dps + cur * R * J;
        T* dnxt = dps + nxt * R * J;

        // ---- joiner input: act(round(enc[t] + dec[j])), rounded ----
        for (int j = tid; j < J; j += kBeamThreads) {
            const float e = toFloat<T>(es[eb * J + j]);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                float a = 0.0f;
                if (r < k) {
                    const float s = detail::roundTo<T>(e + toFloat<T>(dcur[r * J + j]));
                    a = p.activation == kJoinerTanh ? tanhf(s) : relu<float>(s);
                }
                xs[r * J + j] = fromFloat<T>(a);
            }
        }
        __syncthreads();

        // ---- prefetch frame t+1, under the joiner GEMV ----
        if (t + 1 < len) {
            for (int idx = tid * kVec; idx < J; idx += kBeamThreads * kVec) {
                detail::cpAsync16(es + (eb ^ 1) * J + idx,
                                  p.enc_proj + static_cast<int64_t>(b) * p.enc_stride_b +
                                      static_cast<int64_t>(t + 1) * p.enc_stride_t + idx);
            }
        }

        // ---- joiner output GEMV: logits, rounded ----
        T* lg = lgs + (t & 1) * R * ldl;
        float(*pmax)[kBeamWarps] = pmaxs[t & 1];
        for (int n0 = pass0; n0 < V; n0 += pass_step) {
            // Each warp also keeps its 32 outputs' maximum per row -- chunk
            // (v mod 512) / 32 of a running maximum over 512-wide spans -- so
            // the selection starts from the row maxima (the maximum is exact,
            // whatever order it is taken in).
            bd::gemvPass<T, R, OUT>(
                p.w_out_t, p.ld_out, n0, jb, je, xs, J, part, slice, col, tid,
                [&](int r, int v, float sum) {
                    if (r >= k) return;  // uniform across the warp
                    float m = -INFINITY;
                    if (v < V) {
                        const T logit = fromFloat<T>(sum + toFloat<T>(bias_out[v]));
                        lg[r * ldl + v] = logit;
                        if (cluster.size > 1) *cluster.peer(lg + r * ldl + v) = logit;
                        m = toFloat<T>(logit);
                    }
#pragma unroll
                    for (int off = 16; off > 0; off /= 2) {
                        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
                    }
                    if (lane == 0) {
                        const int chunk = (v & 511) >> 5;
                        m = v < 512 ? m : fmaxf(pmax[r][chunk], m);
                        pmax[r][chunk] = m;
                        if (cluster.size > 1) *cluster.peer(&pmax[r][chunk]) = m;
                    }
                });
        }
        if (cluster.size > 1) cluster.sync();  // both halves' logits, everywhere

        // ---- selection: the row's normaliser and every warp's top k, then
        // warp 0 merges.  A lane's candidates are v = wi * 32 + lane + e * 32 * W,
        // e < EC: the smaller of two counts when the vocabulary fits it ----
        const bool live = srow < k;
        auto select = [&](auto ec) {
            constexpr int EC = decltype(ec)::value;
            float mx = -INFINITY;
#pragma unroll
            for (int w = 0; w < kBeamWarps; ++w) {
                if (w < nchunks) mx = fmaxf(mx, live ? pmax[srow][w] : 0.0f);
            }
            // The normaliser, by the row's first warp in torch's layout: lane l
            // exponentiates its elements l, l + width, ... and adds them in
            // sequence, then xor butterflies.  The padding past V would add
            // exact zeros, so it is skipped.
            if (live && wi == 0) {
                float sum = 0.0f;
                if (lane < width) {
#pragma unroll 4
                    for (int it = 0; it < iters; ++it) {
                        const int v = lane + it * width;
                        if (v < V) sum += __nv_expf(toFloat<T>(lg[srow * ldl + v]) - mx);
                    }
                }
                sum = bd::warpSumLikeTorch(sum, width);
                if (lane == 0) row_lsum[srow] = __nv_logf(sum);
            }
            float x[EC];
#pragma unroll
            for (int e = 0; e < EC; ++e) {
                const int v = wi * 32 + lane + e * 32 * W;
                x[e] = (live && v < V) ? toFloat<T>(lg[srow * ldl + v]) : 0.0f;
            }
            __syncthreads();
            const float score = live ? s_score[cur][srow] : 0.0f;
            const float lsum = live ? row_lsum[srow] : 0.0f;
            uint64_t key[EC];
#pragma unroll
            for (int e = 0; e < EC; ++e) {
                const int v = wi * 32 + lane + e * 32 * W;
                key[e] = (live && v < V) ? bd::rankKey(score + ((x[e] - mx) - lsum), srow * V + v)
                                         : 0ull;
            }
            for (int rd = 0; rd < k; ++rd) {
                uint64_t best = key[0];
#pragma unroll
                for (int e = 1; e < EC; ++e) best = key[e] > best ? key[e] : best;
                best = bd::warpMaxKey(best);
#pragma unroll
                for (int e = 0; e < EC; ++e) key[e] = key[e] == best ? 0ull : key[e];
                if (lane == 0) wcand[warp * R + rd] = best;
            }
        };
        if (ecount <= E / 2) {
            select(std::integral_constant<int, E / 2>{});
        } else {
            select(std::integral_constant<int, E>{});
        }
        __syncthreads();
        int emit = 0;
        if (warp == 0) {
            // Lane w walks warp w's sorted list.  Keys are unique, so the lane
            // whose head won is the one that advances.
            int head = 0;
            uint64_t sel = 0ull;
            for (int rd = 0; rd < k; ++rd) {
                const uint64_t mine =
                    (lane < kBeamWarps && head < k) ? wcand[lane * R + head] : 0ull;
                const uint64_t best = bd::warpMaxKey(mine);
                if (lane < kBeamWarps && head < k && mine == best) ++head;
                if (lane == rd) sel = best;
            }
            // ---- the new beam: windows, scores, back-pointers ----
            if (lane < k) {
                const int i = lane;
                const int flat = bd::keyIndex(sel);
                const int par = flat / V;
                const int lab = flat - par * V;
                emit = lab != p.blank;
                s_par[i] = par;
                s_emit[i] = emit;
                s_score[nxt][i] = bd::keyValue(sel);
                for (int c = 0; c < ctx; ++c) {
                    s_ctx[nxt][i][c] =
                        emit ? (c + 1 < ctx ? s_ctx[cur][par][c + 1] : lab) : s_ctx[cur][par][c];
                }
                if (lead) {
                    const int64_t o = (static_cast<int64_t>(t) * p.B + b) * k + i;
                    p.parents[o] = par;
                    p.labels[o] = lab;
                }
            }
        }
        const int any_emit = __syncthreads_or(emit);

        // ---- decoder projection of the new beam: a hypothesis that took blank
        // kept its parent's window, so it keeps its parent's projection ----
        for (int j = tid * kVec; j < J; j += kBeamThreads * kVec) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
                if (r < k && !s_emit[r]) {
                    *reinterpret_cast<uint4*>(dnxt + r * J + j) =
                        *reinterpret_cast<const uint4*>(dcur + s_par[r] * J + j);
                }
            }
        }
        if (any_emit) predictor(s_ctx[nxt], dnxt);
        detail::cpAsyncWaitAll();
        __syncthreads();
        cur = nxt;
        eb ^= 1;
    }

    // Every CTA of the cluster holds the same beam; the lead one writes it.
    // Its peer leaves once no write can still reach it.
    if (cluster.size > 1) cluster.sync();
    if (!lead) return;

    // Frames past the utterance: every slot its own parent, emitting blank --
    // what the walk back through the chunk expects.
    for (int e = tid; e < (p.frames - len) * k; e += kBeamThreads) {
        const int t = len + e / k;
        const int i = e - (e / k) * k;
        const int64_t o = (static_cast<int64_t>(t) * p.B + b) * k + i;
        p.parents[o] = i;
        p.labels[o] = p.blank;
    }
    if (tid < k) {
        for (int c = 0; c < ctx; ++c) {
            p.context_out[(static_cast<int64_t>(b) * k + tid) * ctx + c] = s_ctx[cur][tid][c];
        }
        p.scores_out[static_cast<int64_t>(b) * k + tid] = s_score[cur][tid];
    }

    // ---- the walk: each final slot back to the chunk's first frame, on the
    // device, so the host reads every hypothesis's tokens rather than the
    // history ----
    if (p.walk != nullptr) {
        __syncthreads();  // the history this CTA wrote is visible to all of it
        const int64_t H = static_cast<int64_t>(p.B) * k;
        if (warp < k) {
            const int64_t h = static_cast<int64_t>(b) * k + warp;
            int32_t* path = p.walk + 2 * H + h * p.frames;
            if (lane == 0) {
                int slot = warp;
                for (int t = len - 1; t >= 0; --t) {
                    const int64_t o = (static_cast<int64_t>(t) * p.B + b) * k + slot;
                    path[t] = static_cast<int32_t>(p.labels[o]);
                    slot = static_cast<int>(p.parents[o]);
                }
                p.walk[h] = slot;  // the slot it held when the chunk began
            }
            __syncwarp();
            // Keep the labels, in place: a chunk's writes land at or before
            // the entries the warp has already read.
            int count = 0;
            for (int t0 = 0; t0 < len; t0 += 32) {
                const int t = t0 + lane;
                const int lab = t < len ? path[t] : p.blank;
                const bool keep = t < len && lab != p.blank;
                const unsigned mask = __ballot_sync(0xffffffffu, keep);
                __syncwarp();
                if (keep) path[count + __popc(mask & ((1u << lane) - 1u))] = lab;
                count += __popc(mask);
                __syncwarp();
            }
            if (lane == 0) p.walk[H + h] = count;
        }
    }
}

//! Rows the kernel instantiates for a beam: the next of 2, 4, 8.
inline int statelessBeamRows(int k) { return k <= 2 ? 2 : (k <= 4 ? 4 : 8); }

// Whether one CTA's working set -- the dynamic shared memory above plus the
// kernel's static shared memory, returned in `static_bytes` -- fits the
// current device's opt-in limit.
template <typename T, int R, int OUT>
cudaError_t statelessBeamFitsRows(int J, int D, int V, bool* fits, size_t* static_bytes = nullptr) {
    cudaFuncAttributes attr;
    cudaError_t st = cudaFuncGetAttributes(&attr, statelessBeamKernel<T, R, OUT>);
    if (st != cudaSuccess) return st;
    int dev = 0, optin = 0;
    st = cudaGetDevice(&dev);
    if (st != cudaSuccess) return st;
    st = cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    if (st != cudaSuccess) return st;
    *fits = statelessBeamSmemBytes<T, R, OUT>(J, D, V) + attr.sharedSizeBytes <=
            static_cast<size_t>(optin);
    if (static_bytes != nullptr) *static_bytes = attr.sharedSizeBytes;
    return cudaSuccess;
}

//! CTAs per utterance: a pair -- each streaming half of every GEMV's
//! weights, the per-SM L2 bandwidth that bounds a frame -- when the device
//! launches clusters and the batch leaves an SM for each, else one.
inline int statelessBeamClusterSize(int batch) {
    int dev = 0, sms = 0, clusters = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&clusters, cudaDevAttrClusterLaunch, dev) != cudaSuccess) {
        cudaGetLastError();
        return 1;
    }
    return clusters != 0 && 2 * batch <= sms ? 2 : 1;
}

template <typename T, int R, int OUT>
cudaError_t launchStatelessBeam(const StatelessBeamParams<T>& p, int cluster,
                                cudaStream_t stream) {
    bool fits = false;
    size_t static_bytes = 0;
    cudaError_t st = statelessBeamFitsRows<T, R, OUT>(p.J, p.D, p.V, &fits, &static_bytes);
    if (st != cudaSuccess) return st;
    if (!fits) return cudaErrorInvalidConfiguration;
    const size_t smem = statelessBeamSmemBytes<T, R, OUT>(p.J, p.D, p.V);
    auto kernel = statelessBeamKernel<T, R, OUT>;
    // Past 48 KB in all -- static included -- a block needs the opt-in.
    if (smem + static_bytes > 48 * 1024) {
        st = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(smem));
        if (st != cudaSuccess) {
            cudaGetLastError();  // not sticky: do not hand it to the next launch
            return st;
        }
    }
    if (cluster <= 1) {
        kernel<<<p.B, kBeamThreads, smem, stream>>>(p);
        return cudaGetLastError();
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(static_cast<unsigned>(p.B * cluster));
    cfg.blockDim = dim3(kBeamThreads);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = static_cast<unsigned>(cluster);
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    return cudaLaunchKernelEx(&cfg, kernel, p);
}

//! Whether a beam of `k` over these dims fits the device whatever the
//! cluster size (see statelessBeamFitsRows); false for a beam the kernel does
//! not instantiate.
template <typename T>
cudaError_t StatelessBeamFits(int k, int J, int D, int V, bool* fits) {
    *fits = false;
    if (k < 1 || k > kBeamDecodeMaxBeam) return cudaSuccess;
    bool one = false, pair = false;
    cudaError_t st = cudaSuccess;
    switch (statelessBeamRows(k)) {
        case 2:
            st = statelessBeamFitsRows<T, 2, 8>(J, D, V, &one);
            if (st == cudaSuccess) st = statelessBeamFitsRows<T, 2, 4>(J, D, V, &pair);
            break;
        case 4:
            st = statelessBeamFitsRows<T, 4, 8>(J, D, V, &one);
            if (st == cudaSuccess) st = statelessBeamFitsRows<T, 4, 4>(J, D, V, &pair);
            break;
        default:
            st = statelessBeamFitsRows<T, 8, 8>(J, D, V, &one);
            if (st == cudaSuccess) st = statelessBeamFitsRows<T, 8, 4>(J, D, V, &pair);
            break;
    }
    *fits = one && pair;
    return st;
}

//! Launch the fused beam search over a chunk.  Half precision only; the
//! weights in the greedy kernel's layouts; 1 <= k <= kBeamDecodeMaxBeam,
//! k <= V <= kBeamDecodeMaxVocab.  `cluster` is the CTAs per utterance, 1 or
//! 2; 0 picks statelessBeamClusterSize.  The result does not depend on it.
template <typename T>
cudaError_t StatelessBeamDecode(const StatelessBeamParams<T>& p, int cluster,
                                cudaStream_t stream) {
    if (p.B == 0) return cudaSuccess;
    if (cluster < 0 || cluster > 2) return cudaErrorInvalidValue;
    if (cluster == 0) cluster = statelessBeamClusterSize(p.B);
    if (p.k < 1 || p.k > kBeamDecodeMaxBeam || p.V < p.k || p.V > kBeamDecodeMaxVocab ||
        p.ctx < 1 || p.ctx > kMaxContext || p.J % kStageK != 0 || p.D % kStageK != 0 ||
        p.ld_out % kGemvTile != 0 || p.ld_dp % kGemvTile != 0 || p.ld_out < p.V ||
        p.ld_dp < p.J || p.frames < 0) {
        return cudaErrorInvalidValue;
    }
    const bool pair = cluster == 2;
    switch (statelessBeamRows(p.k)) {
        case 2:
            return pair ? launchStatelessBeam<T, 2, 4>(p, 2, stream)
                        : launchStatelessBeam<T, 2, 8>(p, 1, stream);
        case 4:
            return pair ? launchStatelessBeam<T, 4, 4>(p, 2, stream)
                        : launchStatelessBeam<T, 4, 8>(p, 1, stream);
        default:
            return pair ? launchStatelessBeam<T, 8, 4>(p, 2, stream)
                        : launchStatelessBeam<T, 8, 8>(p, 1, stream);
    }
}

}  // namespace transducer
}  // namespace oasr
