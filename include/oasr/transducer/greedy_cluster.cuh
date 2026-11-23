// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Fused greedy decode for a stateless-predictor transducer, one utterance per
// thread-block cluster (sm_90+).
//
// greedy_decode.cuh decodes a row on one CTA, and a step there is bound by how
// fast one SM pulls the joiner head and the decoder projection out of L2 --
// 512 KB each on the icefall Zipformer transducer, at ~78 GB/s per SM on an
// RTX 5090, ~8 us a step whatever the batch.  At batch 1 that kernel is 70 %
// of a request's GPU time, on one SM of 170.
//
// Here a cluster of C CTAs (up to 16) decodes each row.  Both GEMVs' outputs
// are cut into blocks of 32 -- one output per lane -- dealt round-robin to the
// cluster's CTAs, so a CTA reads 1/C of each weight a step, and when its share
// fits it is copied into shared memory once and read from there.  A step:
//
//   1. every CTA builds the whole joiner input `act(round(enc[t] + dec))` from
//      its own copy of the decoder projection, and the logits of its blocks of
//      the vocabulary;
//   2. its best (logit, token) -- and its logsumexp when tracking -- goes into
//      every CTA's shared memory with `st.async`, which completes bytes on the
//      receiver's mbarrier;
//   3. every CTA waits on its own mbarrier, reduces the same C candidates the
//      same way, and so takes the same emit / advance decision with no further
//      exchange; the cluster's first CTA records the emission;
//   4. on an emission every CTA computes the whole predictor hidden state --
//      window shift, embedding, grouped convolution, ReLU: a few thousand
//      multiply-adds over rows it keeps in shared memory -- and its blocks of the
//      decoder projection, and sends those into every CTA's copy the same way.
//
// Why mbarriers and not a cluster barrier.  The cluster barrier that orders
// distributed-shared-memory stores (`barrier.cluster.arrive.release`, and what
// cooperative groups' `cluster.sync()` emits) compiles to a GPU-scope
// `MEMBAR.ALL.GPU` on sm_120: every arrival waits for the CTA's outstanding
// global traffic -- here the frame prefetches and the token writes -- and a
// round costs 0.58 us with one store in flight, against 0.22 us for an
// `st.async` exchange over 16 CTAs.  The barrier is used only at the start
// (peers' mbarriers exist) and the end (no CTA leaves while addressed).
//
// Exactness.  An output's arithmetic is greedy_decode.cuh's chain: the K axis
// is cut into the same 16 slices (warp w takes slice w), a slice is one fp32
// `fmaf` chain in K order, and the 16 slice partials are summed in slice order
// starting from zero, before the same rounding points.  So tokens, frames and
// the final predictor state do not depend on which kernel -- or cluster size --
// decoded a row, and a row's result does not depend on its batch.  The tracked
// posterior's logsumexp is combined in another order (per CTA, then over the
// cluster), so a posterior can differ from the one-CTA kernel's in the last
// bits.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cooperative_groups.h>
#include <cstdint>
#include <map>
#include <mutex>
#include <tuple>
#include <type_traits>
#include <oasr/common/math.h>
#include <oasr/common/vec_dtypes.h>
#include <oasr/transducer/greedy_decode.cuh>

namespace oasr {
namespace transducer {

//! Largest cluster: sm_90+ allows 16 CTAs with the non-portable opt-in.
constexpr int kGreedyClusterMax = 16;
//! Largest vocabulary: a candidate's token is the low half of a 32-bit key.
constexpr int kGreedyClusterMaxVocab = 65535;
//! Outputs per column block: one per lane of a warp.
constexpr int kColBlock = 32;
//! Encoder frames in flight: a ring refilled one frame ahead of the oldest as
//! the row advances, so a frame's load is never on a step's critical path (a
//! cluster step is shorter than a global load).
constexpr int kFrameRing = 4;

namespace gc {

//! Column blocks of 32 a GEMV with `n` outputs has.
__host__ __device__ __forceinline__ int numBlocks(int n) { return (n + kColBlock - 1) / kColBlock; }

//! Blocks the CTA of rank `r` owns when `nb` blocks are dealt round-robin to `C` CTAs.
__host__ __device__ __forceinline__ int ownedBlocks(int nb, int r, int C) {
    return r < nb ? (nb - r + C - 1) / C : 0;
}

//! Outputs below `n` among the blocks rank `r` owns.
__host__ __device__ __forceinline__ int ownedOutputs(int n, int r, int C) {
    int count = 0;
    for (int blk = r; blk < numBlocks(n); blk += C) {
        const int rest = n - blk * kColBlock;
        count += rest < kColBlock ? rest : kColBlock;
    }
    return count;
}

//! Row stride (elements) of a resident share, output-major: K plus a pad that
//! makes the byte stride an odd multiple of 16, so the eight 16-byte loads of a
//! quarter warp (eight outputs, the same K) hit eight different bank groups.
__host__ __device__ __forceinline__ int residentStride(int K) {
    return K + ((K / 8) % 2 == 0 ? 8 : 16);
}

//! K slice of warp `w` -- greedy_decode.cuh's: ceil(K / kWarps) rounded up to
//! whole stages, clipped to K.
__device__ __forceinline__ void kSlice(int K, int w, int* kb, int* ke) {
    const int kw = ((K + kWarps - 1) / kWarps + kStageK - 1) / kStageK * kStageK;
    *kb = min(K, w * kw);
    *ke = min(K, *kb + kw);
}

__device__ __forceinline__ uint32_t smemAddr(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

template <int N>
__device__ __forceinline__ void cpAsyncWaitGroup() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

__device__ __forceinline__ void cpAsync8(void* smem_dst, const void* gmem_src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(smemAddr(smem_dst)),
                 "l"(gmem_src)
                 : "memory");
}

__device__ __forceinline__ void cpAsyncCommit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

// The cluster a row is decoded by, and the distributed-shared-memory exchange
// it runs on (sm_90+; the launcher never asks for a cluster below that, and
// these compile to nothing there).
struct Cluster {
    int rank = 0;
    int size = 1;
    __device__ __forceinline__ void init() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        rank = static_cast<int>(cooperative_groups::this_cluster().block_rank());
        size = static_cast<int>(cooperative_groups::this_cluster().num_blocks());
#endif
    }
    //! The full cluster barrier: only at the start and the end.
    __device__ __forceinline__ void sync() const {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        cooperative_groups::this_cluster().sync();
#else
        __syncthreads();
#endif
    }
};

__device__ __forceinline__ void mbarInit(uint64_t* bar, int count) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smemAddr(bar)), "r"(count)
                 : "memory");
#endif
}

//! Make the mbarrier initialisations visible to the cluster's st.async.
__device__ __forceinline__ void mbarInitFence() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#endif
}

//! This CTA's one arrival for the phase, expecting `bytes` from st.async.
//! Peers' bytes may land before it: the transaction count goes negative and
//! the phase still cannot complete without this arrival.
__device__ __forceinline__ void mbarExpect(uint64_t* bar, uint32_t bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::"r"(
                     smemAddr(bar)),
                 "r"(bytes)
                 : "memory");
#endif
}

//! Wait for the phase of `parity` to complete; the st.async bytes it counted
//! are visible to this thread afterwards.  CTA-scope acquire, as for any async
//! copy into this CTA's shared memory: a cluster-scope one adds an L1
//! invalidate (CCTL.IVALL) to every completed wait, for global data this
//! exchange never carries.
__device__ __forceinline__ void mbarWait(uint64_t* bar, uint32_t parity) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    const uint32_t addr = smemAddr(bar);
    uint32_t done = 0;
    do {
        asm volatile(
            "{ .reg .pred p;\n"
            "  mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
            "  selp.u32 %0, 1, 0, p; }\n"
            : "=r"(done)
            : "r"(addr), "r"(parity)
            : "memory");
    } while (!done);
#endif
}

//! 16 bytes into CTA `rank`'s copy of `local`, completing them on its copy of `bar`.
__device__ __forceinline__ void sendV4(float* local, const uint64_t* bar, int rank, float4 v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    uint32_t dst, mbar;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(dst)
                 : "r"(smemAddr(local)), "r"(rank));
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(mbar)
                 : "r"(smemAddr(bar)), "r"(rank));
    asm volatile(
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, "
        "[%5];" ::"r"(dst),
        "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(mbar)
        : "memory");
#endif
}

// Slice partials of this CTA's blocks of `w x`:
//   part[(warp * nbmax + i) * 32 + lane] = sum over k in the warp's slice of w[k][col] x[k],
// col = 32 (rank + i C) + lane, one fmaf chain in k order from zero.  Resident:
// the share in shared memory, output-major (`ws`, block i at ws + i 32 ldr,
// row stride ldr), eight K per 16-byte load.  Streamed: the K-major global
// weight (`wg`, row stride `ldg`), a lane's column every K.
template <typename T, bool kResident>
__device__ __forceinline__ void slicePartials(const T* wg, int ldg, const T* ws, int ldr, int K,
                                              int nown, int rank, int C, const T* xs, float* part,
                                              int nbmax, int warp, int lane) {
    int kb, ke;
    kSlice(K, warp, &kb, &ke);
    for (int i = 0; i < nown; ++i) {
        float acc = 0.0f;
        if constexpr (kResident) {
            // Four stages' loads issued before the first FMA: the slice is a few
            // stages long (four at K = 512), and a stage-at-a-time loop left every
            // load's latency in front of its FMAs.  (Eight spilled.)
            const T* w = ws + (static_cast<int64_t>(i) * kColBlock + lane) * ldr + kb;
            const T* x = xs + kb;
            const int ns = (ke - kb) / kStageK;
            for (int s0 = 0; s0 < ns; s0 += 4) {
                uint4 wv[4], xv[4];
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    if (s0 + u < ns) {
                        wv[u] = *reinterpret_cast<const uint4*>(w + (s0 + u) * kStageK);
                        xv[u] = *reinterpret_cast<const uint4*>(x + (s0 + u) * kStageK);
                    }
                }
                // fmaPair: sm_100+'s mixed-precision FMA on a half of each packed
                // pair, bit-identical to fmaf(float(w), float(x), acc).
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    if (s0 + u < ns) {
#pragma unroll
                        for (int q = 0; q < kStageK / 2; ++q) {
                            const uint32_t w2 = detail::word(wv[u], q), x2 = detail::word(xv[u], q);
                            acc = detail::fmaPair<T, 0, 0>(w2, x2, acc);
                            acc = detail::fmaPair<T, 1, 1>(w2, x2, acc);
                        }
                    }
                }
            }
        } else {
            const T* w = wg + (rank + i * C) * kColBlock + lane;
#pragma unroll 2
            for (int k = kb; k < ke; k += kStageK) {
                T wv[kStageK];
#pragma unroll
                for (int u = 0; u < kStageK; ++u) wv[u] = w[static_cast<int64_t>(k + u) * ldg];
                const uint4 xv = *reinterpret_cast<const uint4*>(xs + k);
                const T* xh = reinterpret_cast<const T*>(&xv);
#pragma unroll
                for (int u = 0; u < kStageK; ++u) {
                    acc = fmaf(toFloat<T>(wv[u]), toFloat<T>(xh[u]), acc);
                }
            }
        }
        part[(warp * nbmax + i) * kColBlock + lane] = acc;
    }
}

// greedy_decode.cuh's crossWarpSum for one output: the slice partials in slice order.
__device__ __forceinline__ float slicesSum(const float* part, int nbmax, int i, int lane) {
    float v = 0.0f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) v += part[(w * nbmax + i) * kColBlock + lane];
    return v;
}

template <typename T>
__device__ __forceinline__ uint16_t bitsOf(T v) {
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __bfloat16_as_ushort(v);
    } else {
        return __half_as_ushort(v);
    }
}

template <typename T>
__device__ __forceinline__ T fromBits(uint16_t u) {
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __ushort_as_bfloat16(u);
    } else {
        return __ushort_as_half(u);
    }
}

// A candidate as one unsigned key: the logit's own bits (it is rounded to T),
// made monotone, in the high half -- NaN above everything, as torch.argmax
// propagates it, and -0 equal to +0 -- and the complemented token in the low
// half.  One unsigned max is then argmaxBetter's order: the larger logit, then
// the lower token.  Needs V <= 65535; 0 is "no candidate" (every real one is
// larger: -inf's high half is 0x03ff for fp16, 0x007f for bf16).
template <typename T>
__device__ __forceinline__ uint32_t packKey(float logit, int v) {
    const uint32_t u =
        isnan(logit) ? 0x7fffu : (logit == 0.0f ? 0u : bitsOf<T>(fromFloat<T>(logit)));
    const uint32_t k = (u & 0x8000u) ? (~u & 0xffffu) : (u | 0x8000u);
    return (k << 16) | (0xffffu - static_cast<uint32_t>(v));
}

template <typename T>
__device__ __forceinline__ float keyLogit(uint32_t key) {
    const uint32_t k = key >> 16;
    const uint32_t u = (k & 0x8000u) ? (k & 0x7fffu) : (~k & 0xffffu);
    return toFloat<T>(fromBits<T>(static_cast<uint16_t>(u)));
}

__device__ __forceinline__ int keyToken(uint32_t key) {
    return static_cast<int>(0xffffu - (key & 0xffffu));
}

// A candidate: its key and, tracking, the logsumexp (max, sum) of the logits
// it stands for.
struct Cand {
    uint32_t key;
    float m, s;
};

// Every lane gets the warp's best key (one `redux.sync`) and, tracking, the
// logsumexp of the lanes in its group of kWidth (an xor tree, so every group
// computes it in the same order).
template <bool kTrack, int kWidth>
__device__ __forceinline__ void candWarpReduce(Cand& a) {
    a.key = __reduce_max_sync(0xffffffffu, a.key);
    if constexpr (kTrack) {
#pragma unroll
        for (int off = kWidth / 2; off > 0; off >>= 1) {
            const float om = __shfl_xor_sync(0xffffffffu, a.m, off);
            const float os = __shfl_xor_sync(0xffffffffu, a.s, off);
            detail::lseCombine(a.m, a.s, om, os);
        }
    }
}

}  // namespace gc

//! Where everything lives in one CTA's dynamic shared memory (byte offsets,
//! each 16-byte aligned).  The same function on the host (sizing, fit) and the
//! device (carving), so the two cannot disagree.
struct GreedyClusterLayout {
    size_t wo, wd, ring, xs, hs, erow, convw, dps, bo, bd, part, cand, red, bars, total;
    int nbo0, nbd0, nbmax, ldo, ldd;
};

//! Elements of a predictor convolution weight: none at context 1, (ctx, 1, D)
//! depthwise, (D, 1, ctx, group) grouped.
__host__ __device__ __forceinline__ int64_t convElems(int D, int ctx, int group) {
    return ctx == 1 ? 0
                    : (group == 1 ? static_cast<int64_t>(ctx) * D
                                  : static_cast<int64_t>(D) * ctx * group);
}

template <typename T>
__host__ __device__ inline GreedyClusterLayout greedyClusterLayout(int J, int D, int V, int C,
                                                                   int ctx, int group,
                                                                   bool resident) {
    GreedyClusterLayout L{};
    L.nbo0 = gc::ownedBlocks(gc::numBlocks(V), 0, C);
    L.nbd0 = gc::ownedBlocks(gc::numBlocks(J), 0, C);
    L.nbmax = L.nbo0 > L.nbd0 ? L.nbo0 : L.nbd0;
    L.ldo = gc::residentStride(J);
    L.ldd = gc::residentStride(D);
    size_t o = 0;
    auto take = [&o](size_t bytes) {
        const size_t at = o;
        o += (bytes + 15) / 16 * 16;
        return at;
    };
    const size_t t = sizeof(T), f = sizeof(float), cb = kColBlock;
    L.wo = take(resident ? t * cb * L.nbo0 * L.ldo : 0);  // head share, output-major
    L.wd = take(resident ? t * cb * L.nbd0 * L.ldd : 0);  // decoder-projection share
    L.ring = take(t * kFrameRing * J);                    // encoder frames
    L.xs = take(t * J);                                   // joiner input
    L.hs = take(t * D);                                   // predictor hidden
    L.erow = take(t * ctx * D);                           // the window's embedding rows
    L.convw = take(t * static_cast<size_t>(convElems(D, ctx, group)));
    L.dps = take(f * J);                           // decoder projection, whole
    L.bo = take(f * cb * L.nbo0);                  // head bias share
    L.bd = take(f * cb * L.nbd0);                  // decoder-projection bias share
    L.part = take(f * kWarps * L.nbmax * cb);      // K-slice partials
    L.cand = take(f * 2 * kGreedyClusterMax * 4);  // two rounds of cluster candidates
    L.red = take(f * kWarps * 4);                  // per-warp candidates
    L.bars = take(sizeof(uint64_t) * 3);           // candidate rounds 0 / 1, projection
    L.total = o;
    return L;
}

template <typename T, bool kTrack, bool kResident>
__global__ void __launch_bounds__(kThreads, 1)
    statelessGreedyClusterKernel(const StatelessGreedyParams<T> p) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    gc::Cluster cl;
    cl.init();
    const int C = cl.size, rank = cl.rank;
    const int J = p.J, D = p.D, ctx = p.ctx;
    const int b = blockIdx.x / C;  // the row this cluster decodes
    const int nbo = gc::ownedBlocks(gc::numBlocks(p.V), rank, C);
    const int nbd = gc::ownedBlocks(gc::numBlocks(J), rank, C);
    const GreedyClusterLayout L = greedyClusterLayout<T>(J, D, p.V, C, ctx, p.group, kResident);
    const int nbmax = L.nbmax;

    T* wo_s = reinterpret_cast<T*>(smem_raw + L.wo);
    T* wd_s = reinterpret_cast<T*>(smem_raw + L.wd);
    T* es = reinterpret_cast<T*>(smem_raw + L.ring);
    T* xs = reinterpret_cast<T*>(smem_raw + L.xs);
    T* hs = reinterpret_cast<T*>(smem_raw + L.hs);
    T* erow = reinterpret_cast<T*>(smem_raw + L.erow);
    T* convw = reinterpret_cast<T*>(smem_raw + L.convw);
    float* dps = reinterpret_cast<float*>(smem_raw + L.dps);
    float* bo = reinterpret_cast<float*>(smem_raw + L.bo);
    float* bd = reinterpret_cast<float*>(smem_raw + L.bd);
    float* part = reinterpret_cast<float*>(smem_raw + L.part);
    float* cand = reinterpret_cast<float*>(smem_raw + L.cand);
    float* red = reinterpret_cast<float*>(smem_raw + L.red);
    uint64_t* bars = reinterpret_cast<uint64_t*>(smem_raw + L.bars);

    __shared__ int64_t s_win[kMaxContext];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    constexpr int kVec = 16 / sizeof(T);

    // The loop state lives in every thread's registers: every thread sees the
    // same candidates and takes the same decision, so it needs no broadcast.
    int t = 0, sym = 0, cnt = 0, ehead = 0;
    const int len = [&] {
        const int64_t l = p.lengths[b];
        return l < 0 ? 0 : (l > p.num_frames ? p.num_frames : static_cast<int>(l));
    }();

    if (tid < ctx) s_win[tid] = p.window_in[static_cast<int64_t>(b) * ctx + tid];
    if (tid == 0) {
        gc::mbarInit(&bars[0], 1);
        gc::mbarInit(&bars[1], 1);
        gc::mbarInit(&bars[2], 1);
        gc::mbarInitFence();
    }
    // Everything a step reads that is not a weight share or a frame is staged
    // here once, so no step waits on a global load it could have done up front.
    for (int j = tid; j < J; j += kThreads) {
        dps[j] = toFloat<T>(p.dec_proj_in[static_cast<int64_t>(b) * J + j]);
    }
    for (int o = tid; o < L.nbo0 * kColBlock; o += kThreads) {
        const int v = (rank + (o / kColBlock) * C) * kColBlock + (o % kColBlock);
        bo[o] = (o < nbo * kColBlock && v < p.V && p.b_out != nullptr) ? toFloat<T>(p.b_out[v])
                                                                        : 0.0f;
    }
    for (int o = tid; o < L.nbd0 * kColBlock; o += kThreads) {
        const int n = (rank + (o / kColBlock) * C) * kColBlock + (o % kColBlock);
        bd[o] = (o < nbd * kColBlock && n < J && p.b_dp != nullptr) ? toFloat<T>(p.b_dp[n]) : 0.0f;
    }
    const int64_t nconv = convElems(D, ctx, p.group);
    for (int64_t i = tid; i < nconv; i += kThreads) convw[i] = p.conv_w[i];
    if constexpr (kResident) {
        // This CTA's blocks of both weights, transposed to output-major: element
        // (k, col) of a K-major (K, ld) weight, col = 32 (rank + i C) + c, goes
        // to row i 32 + c, column k.  Eight columns per 16-byte global load.
        for (int idx = tid; idx < nbo * J * (kColBlock / kVec); idx += kThreads) {
            const int i = idx / (J * (kColBlock / kVec));
            const int rem = idx - i * J * (kColBlock / kVec);
            const int k = rem / (kColBlock / kVec), c = (rem % (kColBlock / kVec)) * kVec;
            const uint4 v = *reinterpret_cast<const uint4*>(
                p.w_out_t + static_cast<int64_t>(k) * p.ld_out + (rank + i * C) * kColBlock + c);
            const T* vh = reinterpret_cast<const T*>(&v);
#pragma unroll
            for (int u = 0; u < kVec; ++u) {
                wo_s[(static_cast<int64_t>(i) * kColBlock + c + u) * L.ldo + k] = vh[u];
            }
        }
        for (int idx = tid; idx < nbd * D * (kColBlock / kVec); idx += kThreads) {
            const int i = idx / (D * (kColBlock / kVec));
            const int rem = idx - i * D * (kColBlock / kVec);
            const int k = rem / (kColBlock / kVec), c = (rem % (kColBlock / kVec)) * kVec;
            const uint4 v = *reinterpret_cast<const uint4*>(
                p.w_dp_t + static_cast<int64_t>(k) * p.ld_dp + (rank + i * C) * kColBlock + c);
            const T* vh = reinterpret_cast<const T*>(&v);
#pragma unroll
            for (int u = 0; u < kVec; ++u) {
                wd_s[(static_cast<int64_t>(i) * kColBlock + c + u) * L.ldd + k] = vh[u];
            }
        }
    }
    __syncthreads();  // s_win
    // The window's embedding rows, slot (ehead + s) % ctx for position s; a
    // negative label (before the utterance start) embeds to zero.
    for (int idx = tid; idx < ctx * D; idx += kThreads) {
        const int s = idx / D, d = idx - s * D;
        const int64_t lab = s_win[s];
        erow[idx] = fromFloat<T>(detail::embedAt(detail::embedRow(p.emb, lab, D), lab, d));
    }
    // Frames: thread j < J / kFv owns elements [j kFv, +kFv) of every frame --
    // it loads them (cp.async, frame f into ring slot f % kFrameRing, one commit
    // group per frame) and is the only thread that reads them, so a frame needs
    // no barrier.  The first kFrameRing frames now, frame t + kFrameRing when
    // the row leaves t: "frame t is in" is "all but the newest kFrameRing - 1
    // of this thread's groups are done".  Eight bytes a thread, so the joiner
    // input below is spread over J / 4 threads.
    constexpr int kFv = 8 / sizeof(T);
    const T* enc_row = p.enc_proj + static_cast<int64_t>(b) * p.enc_stride_b;
    const bool frame_owner = tid * kFv < J;
    auto issue_frame = [&](int f) {
        if (f < len) {
            gc::cpAsync8(es + (f % kFrameRing) * J + tid * kFv,
                         enc_row + static_cast<int64_t>(f) * p.enc_stride_t + tid * kFv);
        }
        gc::cpAsyncCommit();
    };
    if (frame_owner) {
        for (int f = 0; f < kFrameRing; ++f) issue_frame(f);
    }
    // Peers' mbarriers and shared memory exist before anyone sends.
    cl.sync();

    int round = 0;  // candidate rounds: buffer / mbarrier round & 1, phase round >> 1
    int emissions = 0;
    const uint32_t cand_bytes = static_cast<uint32_t>(C * 16);
    const uint32_t dps_bytes = static_cast<uint32_t>(J * 4);
    while (t < len) {
        // ---- joiner input: act(round(enc[t] + dec)), rounded, by the frame owners ----
        if (frame_owner) {
            gc::cpAsyncWaitGroup<kFrameRing - 1>();
            const uint2 ev = *reinterpret_cast<const uint2*>(es + (t % kFrameRing) * J + tid * kFv);
            const float4 dv = *reinterpret_cast<const float4*>(dps + tid * kFv);
            const T* eh = reinterpret_cast<const T*>(&ev);
            const float dd[4] = {dv.x, dv.y, dv.z, dv.w};
            uint2 xv;
            T* xh = reinterpret_cast<T*>(&xv);
#pragma unroll
            for (int u = 0; u < kFv; ++u) {
                const float s = detail::roundTo<T>(toFloat<T>(eh[u]) + dd[u]);
                xh[u] = fromFloat<T>(p.activation == kJoinerTanh ? tanhf(s) : relu<float>(s));
            }
            *reinterpret_cast<uint2*>(xs + tid * kFv) = xv;
        }
        // Armed by the last thread, which owns no frame elements and is idle here.
        if (tid == kThreads - 1) gc::mbarExpect(&bars[round & 1], cand_bytes);
        __syncthreads();
        // ---- this CTA's logits ----
        gc::slicePartials<T, kResident>(p.w_out_t, p.ld_out, wo_s, L.ldo, J, nbo, rank, C, xs,
                                        part, nbmax, warp, lane);
        __syncthreads();
        gc::Cand c{0u, -INFINITY, 0.0f};
        for (int o = tid; o < nbo * kColBlock; o += kThreads) {
            const int i = o / kColBlock;
            const int v = (rank + i * C) * kColBlock + lane;
            if (v < p.V) {
                const float logit = detail::roundTo<T>(gc::slicesSum(part, nbmax, i, lane) + bo[o]);
                const uint32_t key = gc::packKey<T>(logit, v);
                c.key = key > c.key ? key : c.key;
                if constexpr (kTrack) detail::lseCombine(c.m, c.s, logit, 1.0f);
            }
        }
        if (nbo * kColBlock <= kColBlock) {
            // One block (the common case): warp 0 holds every output.
            if (warp == 0) gc::candWarpReduce<kTrack, 32>(c);
        } else {
            gc::candWarpReduce<kTrack, 32>(c);
            if (lane == 0) {
                red[warp * 4 + 0] = __uint_as_float(c.key);
                red[warp * 4 + 1] = c.m;
                red[warp * 4 + 2] = c.s;
            }
            __syncthreads();
            if (warp == 0) {
                c = gc::Cand{0u, -INFINITY, 0.0f};
                if (lane < kWarps) {
                    c = gc::Cand{__float_as_uint(red[lane * 4 + 0]), red[lane * 4 + 1],
                                 red[lane * 4 + 2]};
                }
                gc::candWarpReduce<kTrack, 32>(c);
            }
        }
        // Warp 0: the CTA's candidate (every lane holds it) into slot `rank` of
        // every CTA's round buffer.
        if (warp == 0 && lane < C) {
            gc::sendV4(cand + ((round & 1) * kGreedyClusterMax + rank) * 4, &bars[round & 1], lane,
                       make_float4(__uint_as_float(c.key), c.m, c.s, 0.0f));
        }
        gc::mbarWait(&bars[round & 1], static_cast<uint32_t>((round >> 1) & 1));

        // ---- the decision: the same C candidates reduced the same way in every thread ----
        {
            const float* cr = cand + ((round & 1) * kGreedyClusterMax + (lane & 15)) * 4;
            gc::Cand d{0u, -INFINITY, 0.0f};
            if ((lane & 15) < C) d = gc::Cand{__float_as_uint(cr[0]), cr[1], cr[2]};
            gc::candWarpReduce<kTrack, kGreedyClusterMax>(d);
            ++round;
            const int token = gc::keyToken(d.key);
            const bool emit = token != p.blank && sym < p.max_sym;
            if (!emit) {
                // Leave frame t: its ring slot takes frame t + kFrameRing.
                if (frame_owner) issue_frame(t + kFrameRing);
                ++t;
                sym = 0;
                continue;
            }
            if (rank == 0 && tid == 0 && cnt < p.cap) {
                const int64_t o = static_cast<int64_t>(b) * p.cap + cnt;
                p.tokens[o] = token;
                p.frames[o] = t;
                if constexpr (kTrack) {
                    p.probs[o] = __expf(gc::keyLogit<T>(d.key) - (d.m + __logf(d.s)));
                }
            }
            if (tid == 0) {
                for (int i = 0; i + 1 < ctx; ++i) s_win[i] = s_win[i + 1];
                s_win[ctx - 1] = token;
            }
            ++cnt;
            ++sym;
            // ---- the new label's embedding row replaces the window's oldest ----
            const T* src = p.emb + static_cast<int64_t>(token) * D;
            for (int idx = tid * kVec; idx < D; idx += kThreads * kVec) {
                *reinterpret_cast<uint4*>(erow + ehead * D + idx) =
                    *reinterpret_cast<const uint4*>(src + idx);
            }
            ehead = ehead + 1 == ctx ? 0 : ehead + 1;  // the slot of window position 0
        }
        if (tid == kThreads - 1) gc::mbarExpect(&bars[2], dps_bytes);
        __syncthreads();
        // ---- predictor hidden: relu(round(conv(window rows))), all of it, here ----
        for (int d = tid; d < D; d += kThreads) {
            T h;
            if (ctx == 1) {
                h = erow[d];
            } else if (p.group == 1) {
                float acc = 0.0f;
                for (int s = 0, slot = ehead; s < ctx; ++s, slot = slot + 1 == ctx ? 0 : slot + 1) {
                    acc = fmaf(toFloat<T>(erow[slot * D + d]), toFloat<T>(convw[s * D + d]), acc);
                }
                h = fromFloat<T>(acc);
            } else if (p.group == 4) {
                // The icefall group size: four channels in one 8-byte load each.
                const int c0 = d & ~3;
                float acc = 0.0f;
                for (int s = 0, slot = ehead; s < ctx; ++s, slot = slot + 1 == ctx ? 0 : slot + 1) {
                    const uint2 ev = *reinterpret_cast<const uint2*>(erow + slot * D + c0);
                    const uint2 wv = *reinterpret_cast<const uint2*>(
                        convw + (static_cast<int64_t>(d) * ctx + s) * 4);
                    const T* e = reinterpret_cast<const T*>(&ev);
                    const T* w = reinterpret_cast<const T*>(&wv);
#pragma unroll
                    for (int i = 0; i < 4; ++i) acc = fmaf(toFloat<T>(e[i]), toFloat<T>(w[i]), acc);
                }
                h = fromFloat<T>(acc);
            } else {
                const int gs = p.group;
                const int c0 = (d / gs) * gs;
                float acc = 0.0f;
                for (int s = 0, slot = ehead; s < ctx; ++s, slot = slot + 1 == ctx ? 0 : slot + 1) {
                    const T* e = erow + slot * D + c0;
                    const T* w = convw + (static_cast<int64_t>(d) * ctx + s) * gs;
                    for (int i = 0; i < gs; ++i) {
                        acc = fmaf(toFloat<T>(e[i]), toFloat<T>(w[i]), acc);
                    }
                }
                h = fromFloat<T>(acc);
            }
            hs[d] = relu<T>(h);
        }
        __syncthreads();
        // ---- this CTA's blocks of the decoder projection, into every CTA ----
        gc::slicePartials<T, kResident>(p.w_dp_t, p.ld_dp, wd_s, L.ldd, D, nbd, rank, C, hs, part,
                                        nbmax, warp, lane);
        __syncthreads();
        // Warp w finishes block w (and w + 16, ...) and sends it to every CTA,
        // this one included: lane l carries outputs [4 (l % 8), +4) of the block
        // to CTAs l / 8, l / 8 + 4, ..., so the 8 C sends are spread over the warp.
        for (int i = warp; i < nbd; i += kWarps) {
            const int n0 = (rank + i * C) * kColBlock;
            float v = 0.0f;
            if (n0 + lane < J) {
                v = detail::roundTo<T>(gc::slicesSum(part, nbmax, i, lane) +
                                       bd[i * kColBlock + lane]);
            }
            const int q = lane & 7;
            const float4 v4 = make_float4(__shfl_sync(0xffffffffu, v, 4 * q),
                                          __shfl_sync(0xffffffffu, v, 4 * q + 1),
                                          __shfl_sync(0xffffffffu, v, 4 * q + 2),
                                          __shfl_sync(0xffffffffu, v, 4 * q + 3));
            if (n0 + 4 * q < J) {
                for (int r = lane >> 3; r < C; r += 4) {
                    gc::sendV4(dps + n0 + 4 * q, &bars[2], r, v4);
                }
            }
        }
        gc::mbarWait(&bars[2], static_cast<uint32_t>(emissions & 1));
        ++emissions;
    }

    // ---- write the carried state back (the cluster's first CTA) ----
    __syncthreads();
    if (rank == 0) {
        if (tid == 0) p.counts[b] = cnt;
        if (tid < ctx) p.window_out[static_cast<int64_t>(b) * ctx + tid] = s_win[tid];
        for (int j = tid; j < J; j += kThreads) {
            p.dec_proj_out[static_cast<int64_t>(b) * J + j] = fromFloat<T>(dps[j]);
        }
    }
    // Nothing is in flight to this CTA (every phase it armed completed), but a
    // peer must not leave while it could still be addressed.
    cl.sync();
}

//! Whether the cluster kernel can run here at all: sm_90+ with cluster launch.
inline bool statelessGreedyClustersAvailable() {
    int dev = 0, major = 0, clusters = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&clusters, cudaDevAttrClusterLaunch, dev) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return major >= 9 && clusters != 0;
}

template <typename T, bool kTrack, bool kResident>
cudaError_t launchStatelessGreedyCluster(const StatelessGreedyParams<T>& p, int C,
                                         cudaStream_t stream, int* max_clusters) {
    auto kernel = statelessGreedyClusterKernel<T, kTrack, kResident>;
    cudaFuncAttributes attr;
    cudaError_t st = cudaFuncGetAttributes(&attr, kernel);
    if (st != cudaSuccess) return st;
    int dev = 0, optin = 0;
    st = cudaGetDevice(&dev);
    if (st != cudaSuccess) return st;
    st = cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    if (st != cudaSuccess) return st;
    const size_t smem =
        greedyClusterLayout<T>(p.J, p.D, p.V, C, p.ctx, p.group, kResident).total;
    if (smem + attr.sharedSizeBytes > static_cast<size_t>(optin)) {
        return cudaErrorInvalidConfiguration;
    }
    st = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                              static_cast<int>(smem));
    if (st == cudaSuccess && C > 8) {
        st = cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    }
    if (st != cudaSuccess) {
        cudaGetLastError();  // not sticky: do not hand it to the next launch
        return st;
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(static_cast<unsigned>(p.B * C));
    cfg.blockDim = dim3(kThreads);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cudaLaunchAttribute la[1];
    la[0].id = cudaLaunchAttributeClusterDimension;
    la[0].val.clusterDim.x = static_cast<unsigned>(C);
    la[0].val.clusterDim.y = 1;
    la[0].val.clusterDim.z = 1;
    cfg.attrs = la;
    cfg.numAttrs = 1;
    if (max_clusters != nullptr) {
        st = cudaOccupancyMaxActiveClusters(max_clusters, kernel, &cfg);
        if (st != cudaSuccess) cudaGetLastError();
        return st;
    }
    return cudaLaunchKernelEx(&cfg, kernel, p);
}

//! Resident when this cluster size's weight shares fit next to the buffers.
template <typename T>
bool statelessGreedyClusterResident(const StatelessGreedyParams<T>& p, int C) {
    int dev = 0, optin = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) !=
            cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    // 512 B of headroom for the kernel's static shared memory.
    return greedyClusterLayout<T>(p.J, p.D, p.V, C, p.ctx, p.group, true).total + 512 <=
           static_cast<size_t>(optin);
}

template <typename T>
cudaError_t dispatchStatelessGreedyCluster(const StatelessGreedyParams<T>& p, int C,
                                           cudaStream_t stream, int* max_clusters = nullptr) {
    const bool track = p.probs != nullptr;
    if (statelessGreedyClusterResident(p, C)) {
        return track ? launchStatelessGreedyCluster<T, true, true>(p, C, stream, max_clusters)
                     : launchStatelessGreedyCluster<T, false, true>(p, C, stream, max_clusters);
    }
    return track ? launchStatelessGreedyCluster<T, true, false>(p, C, stream, max_clusters)
                 : launchStatelessGreedyCluster<T, false, false>(p, C, stream, max_clusters);
}

//! Launch the cluster greedy decode with `C` CTAs per row (2, 4, 8 or 16).
//! The same contract as StatelessGreedyDecode; sm_90+ only.
template <typename T>
cudaError_t StatelessGreedyClusterDecode(const StatelessGreedyParams<T>& p, int C,
                                         cudaStream_t stream) {
    if (p.B == 0) return cudaSuccess;
    if ((C != 2 && C != 4 && C != 8 && C != 16) || p.ctx < 1 || p.ctx > kMaxContext ||
        p.J % kStageK != 0 || p.D % kStageK != 0 || p.ld_out % kGemvTile != 0 ||
        p.ld_dp % kGemvTile != 0 || p.ld_out < p.V || p.ld_dp < p.J || p.num_frames < 0 ||
        p.V > kGreedyClusterMaxVocab) {
        return cudaErrorInvalidValue;
    }
    if (!statelessGreedyClustersAvailable()) return cudaErrorNotSupported;
    return dispatchStatelessGreedyCluster(p, C, stream);
}

//! How many clusters of `C` CTAs the device keeps resident at once for this
//! geometry (0 when it cannot run them).
template <typename T>
int statelessGreedyMaxClusters(const StatelessGreedyParams<T>& p, int C) {
    if (!statelessGreedyClustersAvailable()) return 0;
    int n = 0;
    StatelessGreedyParams<T> q = p;
    q.B = 1;
    if (dispatchStatelessGreedyCluster(q, C, nullptr, &n) != cudaSuccess) return 0;
    return n;
}

//! The cluster kernel serves batches up to this; past it the launcher keeps
//! greedy_decode.cuh's one CTA per row.  A large batch is decoded on a side
//! stream beside the next batch's encoder, and the one-CTA kernel leaves that
//! encoder most of the GPU -- 64 SMs of 170 at batch 64, where the cluster
//! kernel is only 1.2x faster on its own.
constexpr int kGreedyClusterMaxBatch = 32;
//! CTAs the launcher spends on a cluster decode at most.
constexpr int kGreedyClusterMaxCtas = 128;

//! Clusters of `C` the device keeps resident for this geometry, cached per
//! device and shape (an occupancy query is a few microseconds of host time).
template <typename T>
int statelessGreedyMaxClustersCached(const StatelessGreedyParams<T>& p, int C) {
    struct Key {
        int dev, J, D, V, ctx, group, C;
        bool track;
        bool operator<(const Key& o) const {
            return std::tie(dev, J, D, V, ctx, group, C, track) <
                   std::tie(o.dev, o.J, o.D, o.V, o.ctx, o.group, o.C, o.track);
        }
    };
    static std::mutex mu;
    static std::map<Key, int> cache;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    const Key key{dev, p.J, p.D, p.V, p.ctx, p.group, C, p.probs != nullptr};
    {
        std::lock_guard<std::mutex> lock(mu);
        const auto it = cache.find(key);
        if (it != cache.end()) return it->second;
    }
    const int n = statelessGreedyMaxClusters(p, C);
    std::lock_guard<std::mutex> lock(mu);
    cache[key] = n;
    return n;
}

//! CTAs per row when the launcher picks: the largest cluster that keeps the
//! whole batch in one resident wave within kGreedyClusterMaxCtas, for a batch
//! up to kGreedyClusterMaxBatch; else 1 (greedy_decode.cuh's kernel).
//! Measured on the icefall Zipformer transducer (RTX 5090, real LibriSpeech
//! decode inputs, the decode kernel alone): 16 CTAs 5.7x at batch 1-8, 8 CTAs
//! 3.1x at 16, 4 CTAs 2.0x at 32.
template <typename T>
int statelessGreedyAutoCluster(const StatelessGreedyParams<T>& p) {
    if (p.B < 1 || p.B > kGreedyClusterMaxBatch || p.V > kGreedyClusterMaxVocab ||
        !statelessGreedyClustersAvailable()) {
        return 1;
    }
    for (const int C : {16, 8, 4, 2}) {
        if (p.B * C > kGreedyClusterMaxCtas) continue;
        if (statelessGreedyMaxClustersCached(p, C) >= p.B) return C;
    }
    return 1;
}

}  // namespace transducer
}  // namespace oasr
