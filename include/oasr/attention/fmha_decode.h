// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// The C++ lane's single-query decode step: CUDA cores and a cluster merge.
//
// A decode step attends one query row per sequence over its cache.  The MMA
// mainloop (`fmha_mainloop_sm80.h`) serves it with a 64-row tile, and the tile
// is not what costs: on an RTX 5090 (ncu, base clocks, B8 H6 T_k 1500) that
// kernel ran one 4-warp CTA per SM -- 57 KB of shared memory, 168 registers per
// thread, 8 % of warp slots active -- and reached 42-52 % of DRAM bandwidth on a
// step that is nothing but a read of K and V.  Its split pass then paid a second
// launch for the combine.
//
// This kernel is the read, nothing else:
//
//   * **One CTA per (split, query head, batch row).**  A group of `kGroup`
//     threads holds one key row -- one 128-bit load each -- and the CTA's
//     `kThreads / kGroup` groups walk the split's keys `kGroups` apart, so a
//     warp's loads are whole cache lines.
//   * **Loads in flight.**  Each group takes `kUnroll` keys per step, and two
//     register buffers alternate (the next step's K and V are issued before
//     this step's dot products), so every thread has 2-4 x `kUnroll` 128-bit
//     loads outstanding.  A paged split's page ids are read into shared memory
//     once, so no K/V load waits on a page-table load.
//   * **An online softmax per group**, in fp32, on base-2 exponents with the
//     scale folded into `q` -- the dot product is already `log2(e) * scale *
//     q.k`.  The groups merge once, through shared memory.
//   * **Splits merge through a thread-block cluster.**  The splits of one
//     (head, batch row) are one cluster; each leaves its unnormalised row and
//     (max, sum) in shared memory, and every rank reads all ranks' statistics
//     in one round of distributed-shared-memory reads, then writes its own
//     slice of the output row.  No workspace, no second launch: measured, the
//     separate combine kernel cost 1.2-1.6 us per call at the batch sizes that
//     split, and a rank-0-only merge that read the ranks one at a time cost
//     about the same as the kernel it replaced.  Clusters need sm_90, so this
//     file serves sm_90 and newer; the MMA path keeps the older parts.
//
// Configuration (256 threads, 4 keys in flight per group, the register
// ping-pong) and the split rule (`oasr.jit.fmha.decode_num_splits`) were
// chosen by measurement: of ten configurations over 23 served shapes, this one
// had the least regret against the best per shape (1.0 % L2-rotated, 2.7 %
// L2-warm), and the rule picks within 0.6 % of the best measured split over 284
// shapes (`.artifacts` is not cited here; see `oasr/jit/fmha.py`).
//
// **The split count must be a function of shape alone** (`AGENTS.md` rule 11):
// it decides the merge order of the fp32 partials, so a capture-dependent
// answer would make a replayed graph differ from eager.  The caller computes it
// from shape and SM count and passes it in; nothing here reads capture state.
//
// Output contract, the same as the MMA path: a row with no live key (its key
// window `[start, len)` is empty) is exactly zero, never NaN.

#pragma once

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>

#include <cstdint>
#include <type_traits>

namespace oasr {
namespace attention {

//! CTA width of the decode kernel.
inline constexpr int kFmhaDecodeThreads = 256;
//! Keys each thread group takes per step, per register buffer.
inline constexpr int kFmhaDecodeUnroll = 4;
//! Page ids a split caches in shared memory; a longer split reads the table.
inline constexpr int kFmhaDecodePageCache = 512;
//! Most splits: the largest *portable* cluster.  Non-portable clusters (up to
//! 16 on sm_120) measured slower than 8 everywhere they were tried -- a
//! cluster's CTAs must be co-resident on one GPC.
inline constexpr int kFmhaDecodeMaxSplits = 8;
//! Widest padded head dim: a key row is read by at most one warp.
inline constexpr int kFmhaDecodeMaxHeadDim = 256;

/*! \brief Arguments of one decode step.
 *
 * Strides are in elements.  Dense K/V are `(B, H_kv, T_k, D)` and `k_outer`
 * steps the batch; paged K/V are `(num_pages, page_size, H_kv, D)` and
 * `k_outer` steps the page.  `seqlens` / `seqstarts` bound each row's key
 * window `[start, len)`; without `seqlens` a row is `t_k` keys long.
 */
struct FmhaDecodeParams {
    void const* q;
    int64_t q_batch_stride, q_head_stride;
    void const* k;
    void const* v;
    int64_t k_row_stride, k_head_stride, k_outer_stride;
    int64_t v_row_stride, v_head_stride, v_outer_stride;
    int const* block_table;
    int64_t table_stride;
    int page_size;
    int const* seqlens;
    int const* seqstarts;
    int t_k;
    void* o;
    int64_t o_batch_stride, o_head_stride;
    int num_splits;
    int batch, heads, kv_heads, head_dim;
    float scale_log2;
};

namespace decode_detail {

template <bool kBf16>
__device__ __forceinline__ void to_float8(uint4 const& r, float (&f)[8]) {
    if constexpr (kBf16) {
        __nv_bfloat162 const* h = reinterpret_cast<__nv_bfloat162 const*>(&r);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            float2 const x = __bfloat1622float2(h[i]);
            f[2 * i] = x.x;
            f[2 * i + 1] = x.y;
        }
    } else {
        __half2 const* h = reinterpret_cast<__half2 const*>(&r);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            float2 const x = __half22float2(h[i]);
            f[2 * i] = x.x;
            f[2 * i + 1] = x.y;
        }
    }
}

template <bool kBf16>
__device__ __forceinline__ uint16_t from_float(float x) {
    if constexpr (kBf16) {
        __nv_bfloat16 const h = __float2bfloat16_rn(x);
        return *reinterpret_cast<uint16_t const*>(&h);
    } else {
        __half const h = __float2half_rn(x);
        return *reinterpret_cast<uint16_t const*>(&h);
    }
}

__device__ __forceinline__ uint4 ldg128(uint16_t const* p) {
    return __ldg(reinterpret_cast<uint4 const*>(p));
}

constexpr int pow2AtLeast(int x) {
    int p = 1;
    while (p < x) {
        p <<= 1;
    }
    return p;
}

}  // namespace decode_detail

/*! \brief One decode step; see the file comment.
 *
 * \tparam Element   `cutlass::half_t` or `cutlass::bfloat16_t`
 * \tparam kHeadDim  **padded** head dim; `p.head_dim` is the real one
 * \tparam kPaged    K/V are a page pool read through `block_table`
 * \tparam kCluster  launched as clusters of `num_splits` CTAs along x
 */
template <class Element, int kHeadDim, bool kPaged, bool kCluster>
__global__ void __launch_bounds__(kFmhaDecodeThreads)
    fmha_decode_kernel(FmhaDecodeParams const p) {
    using namespace decode_detail;
    constexpr bool kBf16 = std::is_same<Element, cutlass::bfloat16_t>::value;
    constexpr int kThreads = kFmhaDecodeThreads;
    constexpr int kUnroll = kFmhaDecodeUnroll;
    constexpr int kVec = 8;
    constexpr int kVecs = kHeadDim / kVec;
    constexpr int kGroup = pow2AtLeast(kVecs);
    static_assert(kHeadDim % kVec == 0 && kGroup <= 32, "a key row is read by at most one warp");
    constexpr int kGroups = kThreads / kGroup;
    constexpr int kStep = kGroups * kUnroll;

    int const split = int(blockIdx.x);
    int const h = int(blockIdx.y);
    int const b = int(blockIdx.z);
    int const tid = int(threadIdx.x);
    int const g = tid / kGroup;
    int const lane = tid % kGroup;
    int const hk = h / (p.heads / p.kv_heads);
    // Lanes past the real head dim -- a padded head dim, or the spare lanes of a
    // group rounded up to a power of two -- hold zeros and load nothing.
    bool const active = lane * kVec < p.head_dim;

    // This split's share of the row's key window.  Cut from the row's own
    // window, so a short row's later splits are empty and merge as nothing.
    int const len = p.seqlens != nullptr ? p.seqlens[b] : p.t_k;
    int const start = p.seqstarts != nullptr ? p.seqstarts[b] : 0;
    int const n = len - start;
    int const chunk = n > 0 ? (n + p.num_splits - 1) / p.num_splits : 0;
    int const lo = start + split * chunk;
    int const hi = min(lo + chunk, len);

    float qf[8];
    if (active) {
        uint16_t const* qp = reinterpret_cast<uint16_t const*>(p.q) + b * p.q_batch_stride +
                             h * p.q_head_stride + lane * kVec;
        to_float8<kBf16>(ldg128(qp), qf);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            qf[i] *= p.scale_log2;
        }
    } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            qf[i] = 0.f;
        }
    }

    uint16_t const* kb =
        reinterpret_cast<uint16_t const*>(p.k) + hk * p.k_head_stride + lane * kVec;
    uint16_t const* vb =
        reinterpret_cast<uint16_t const*>(p.v) + hk * p.v_head_stride + lane * kVec;
    if constexpr (!kPaged) {
        kb += b * p.k_outer_stride;
        vb += b * p.v_outer_stride;
    }
    int const* table = kPaged ? p.block_table + b * p.table_stride : nullptr;

    __shared__ int s_pages[kPaged ? kFmhaDecodePageCache : 1];
    int page0 = 0;
    bool cached = false;
    if constexpr (kPaged) {
        if (hi > lo) {
            page0 = lo / p.page_size;
            int const npages = (hi - 1) / p.page_size - page0 + 1;
            cached = npages <= kFmhaDecodePageCache;
            if (cached) {
                for (int i = tid; i < npages; i += kThreads) {
                    s_pages[i] = __ldg(table + page0 + i);
                }
            }
        }
        __syncthreads();
    }

    float m = -INFINITY;
    float l = 0.f;
    float acc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        acc[i] = 0.f;
    }

    auto load = [&](int base, uint4(&kr)[kUnroll], uint4(&vr)[kUnroll]) {
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            int const key = base + u * kGroups + g;
            if (key < hi && active) {
                int64_t ko, vo;
                if constexpr (kPaged) {
                    int const pi = key / p.page_size;
                    int const page = cached ? s_pages[pi - page0] : __ldg(table + pi);
                    int const row = key - pi * p.page_size;
                    ko = page * p.k_outer_stride + row * p.k_row_stride;
                    vo = page * p.v_outer_stride + row * p.v_row_stride;
                } else {
                    ko = key * p.k_row_stride;
                    vo = key * p.v_row_stride;
                }
                kr[u] = ldg128(kb + ko);
                vr[u] = ldg128(vb + vo);
            } else {
                kr[u] = make_uint4(0, 0, 0, 0);
                vr[u] = make_uint4(0, 0, 0, 0);
            }
        }
    };
    auto consume = [&](int base, uint4 const(&kr)[kUnroll], uint4 const(&vr)[kUnroll]) {
        float s[kUnroll];
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            float kf[8];
            to_float8<kBf16>(kr[u], kf);
            float d = 0.f;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                d = fmaf(qf[i], kf[i], d);
            }
            // The group's lanes are aligned and a power of two, so an xor
            // offset below `kGroup` never leaves the group.
#pragma unroll
            for (int o = kGroup / 2; o > 0; o >>= 1) {
                d += __shfl_xor_sync(0xffffffffu, d, o);
            }
            s[u] = (base + u * kGroups + g < hi) ? d : -INFINITY;
        }
        float mx = m;
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            mx = fmaxf(mx, s[u]);
        }
        // A step with no live key for this group leaves its state untouched; an
        // empty state is never carried as a max (cf. AGENTS.md rule 10).
        if (mx != -INFINITY) {
            float const sc = exp2f(m - mx);
            l *= sc;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                acc[i] *= sc;
            }
#pragma unroll
            for (int u = 0; u < kUnroll; ++u) {
                float const pu = exp2f(s[u] - mx);
                l += pu;
                float vf[8];
                to_float8<kBf16>(vr[u], vf);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    acc[i] = fmaf(pu, vf[i], acc[i]);
                }
            }
            m = mx;
        }
    };

    // Two register buffers in ping-pong.  The loop bounds are uniform across the
    // CTA, so every lane reaches every shuffle.
    {
        uint4 kA[kUnroll], vA[kUnroll], kB[kUnroll], vB[kUnroll];
        int base = lo;
        if (base < hi) {
            load(base, kA, vA);
        }
        while (base < hi) {
            int const next = base + kStep;
            if (next < hi) {
                load(next, kB, vB);
            }
            consume(base, kA, vA);
            base = next;
            if (base >= hi) {
                break;
            }
            int const next2 = base + kStep;
            if (next2 < hi) {
                load(next2, kA, vA);
            }
            consume(base, kB, vB);
            base = next2;
        }
    }

    // Merge the groups: weights `exp2(m_g - M)`, in group order.
    __shared__ float s_m[kGroups];
    __shared__ float s_l[kGroups];
    __shared__ float s_w[kGroups];
    __shared__ float s_acc[kThreads * kVec];
    __shared__ float s_stat[2];
    if (lane == 0) {
        s_m[g] = m;
        s_l[g] = l;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        s_acc[g * (kGroup * kVec) + lane * kVec + i] = acc[i];
    }
    __syncthreads();
    if (tid == 0) {
        float M = -INFINITY;
        for (int j = 0; j < kGroups; ++j) {
            M = fmaxf(M, s_m[j]);
        }
        float L = 0.f;
        for (int j = 0; j < kGroups; ++j) {
            float const w = s_m[j] == -INFINITY ? 0.f : exp2f(s_m[j] - M);
            s_w[j] = w;
            L += s_l[j] * w;
        }
        s_stat[0] = M;
        s_stat[1] = L;
    }
    __syncthreads();

    uint16_t* const op =
        reinterpret_cast<uint16_t*>(p.o) + b * p.o_batch_stride + h * p.o_head_stride;
    if constexpr (!kCluster) {
        float const M = s_stat[0];
        float const L = s_stat[1];
        float const inv = (M != -INFINITY && L > 0.f) ? 1.f / L : 0.f;
        for (int d = tid; d < p.head_dim; d += kThreads) {
            float o = 0.f;
#pragma unroll
            for (int j = 0; j < kGroups; ++j) {
                o = fmaf(s_acc[j * (kGroup * kVec) + d], s_w[j], o);
            }
            op[d] = from_float<kBf16>(o * inv);
        }
    } else {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        namespace cg = cooperative_groups;
        __shared__ float s_o[kHeadDim];
        __shared__ float s_rm[kFmhaDecodeMaxSplits];
        __shared__ float s_rl[kFmhaDecodeMaxSplits];
        for (int d = tid; d < p.head_dim; d += kThreads) {
            float o = 0.f;
#pragma unroll
            for (int j = 0; j < kGroups; ++j) {
                o = fmaf(s_acc[j * (kGroup * kVec) + d], s_w[j], o);
            }
            s_o[d] = o;
        }
        cg::cluster_group cl = cg::this_cluster();
        int const nr = int(cl.num_blocks());
        int const rank = int(cl.block_rank());
        cl.sync();
        // Every rank's (max, sum), in one parallel round of remote reads.
        if (tid < nr) {
            float const* st = cl.map_shared_rank(s_stat, tid);
            s_rm[tid] = st[0];
            s_rl[tid] = st[1];
        }
        __syncthreads();
        float MM = -INFINITY;
#pragma unroll
        for (int r = 0; r < kFmhaDecodeMaxSplits; ++r) {
            MM = r < nr ? fmaxf(MM, s_rm[r]) : MM;
        }
        float wr[kFmhaDecodeMaxSplits];
        float LL = 0.f;
#pragma unroll
        for (int r = 0; r < kFmhaDecodeMaxSplits; ++r) {
            wr[r] = (r < nr && s_rm[r] != -INFINITY) ? exp2f(s_rm[r] - MM) : 0.f;
            LL += r < nr ? s_rl[r] * wr[r] : 0.f;
        }
        float const inv = (MM != -INFINITY && LL > 0.f) ? 1.f / LL : 0.f;
        // Rank r writes its contiguous slice of the row, each element's remote
        // reads issued together and summed in rank order.
        int const per = (p.head_dim + nr - 1) / nr;
        int const d0 = rank * per;
        int const d1 = min(d0 + per, p.head_dim);
        for (int d = d0 + tid; d < d1; d += kThreads) {
            float vals[kFmhaDecodeMaxSplits];
#pragma unroll
            for (int r = 0; r < kFmhaDecodeMaxSplits; ++r) {
                vals[r] = r < nr ? cl.map_shared_rank(s_o, r)[d] : 0.f;
            }
            float o = 0.f;
#pragma unroll
            for (int r = 0; r < kFmhaDecodeMaxSplits; ++r) {
                o = fmaf(vals[r], wr[r], o);
            }
            op[d] = from_float<kBf16>(o * inv);
        }
        // No rank leaves -- releasing its shared memory -- while another reads it.
        cl.sync();
#endif
    }
}

/*! \brief Launch one decode step: `num_splits` CTAs per (head, batch row).
 *
 * One split is an ordinary launch; more are one cluster per (head, batch row),
 * merged in the kernel.  `num_splits` is the caller's, from shape and SM count.
 */
template <int Arch, class Element, int kHeadDim, bool kPaged>
cudaError_t run_fmha_decode(FmhaDecodeParams const& p, cudaStream_t stream) {
    static_assert(Arch >= 90, "the decode kernel merges its splits through a cluster (sm_90+)");
    static_assert(kHeadDim <= kFmhaDecodeMaxHeadDim, "a key row is read by at most one warp");
    if (p.batch <= 0 || p.heads <= 0) {
        return cudaSuccess;
    }
    if (p.num_splits < 1 || p.num_splits > kFmhaDecodeMaxSplits || p.kv_heads <= 0 ||
        p.heads % p.kv_heads != 0) {
        return cudaErrorInvalidValue;
    }
    dim3 const grid(uint32_t(p.num_splits), uint32_t(p.heads), uint32_t(p.batch));
    if (p.num_splits == 1) {
        fmha_decode_kernel<Element, kHeadDim, kPaged, false>
            <<<grid, kFmhaDecodeThreads, 0, stream>>>(p);
        return cudaGetLastError();
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = dim3(kFmhaDecodeThreads);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = uint32_t(p.num_splits);
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    return cudaLaunchKernelEx(&cfg, fmha_decode_kernel<Element, kHeadDim, kPaged, true>, p);
}

}  // namespace attention
}  // namespace oasr
