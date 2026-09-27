// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// The Ampere-class FMHA mainloop: a cp.async ring over K and V, an
// m16n8k16 QK gemm, the bias add, the mask, the online softmax, and the PV
// gemm.
//
// Structurally modelled on FlashAttention's `hopper/mainloop_fwd_sm80.hpp`.
// The collective/shell decomposition and the cp.async fence accounting are
// FA's and are worth copying exactly; the softmax numerics, the causal
// alignment and the bias are not (see `fmha_softmax.h`, `fmha_mask.h`,
// `fmha_bias.h`).
//
// ---------------------------------------------------------------------------
// The K loop is split four ways
// ---------------------------------------------------------------------------
//
//   (a) the first (highest) tile   -- straddles `seqlen_k`; also `Is_first`
//   (b) the causal / local right edge
//   (c) the interior               -- **no mask at all**
//   (d) the left edge              -- sliding window and/or `seqstart_k`
//
// The CuTeDSL backend applies the full element-wise predicate to every tile.
// At 64x64 that is 4096 compares and selects per tile that (c) does not need,
// and (c) is most of the loop on any long sequence.
//
// Loop (d) is instantiated unconditionally even when `Is_local` is false,
// because it is where the *runtime* `seqstart_k` predicate lives -- that is
// what lets left padding stay out of the compile-time variant space.  It runs
// zero iterations when neither a window nor a start applies, so the cost is a
// compare.
//
// ---------------------------------------------------------------------------
// Ring and drain order
// ---------------------------------------------------------------------------
//
// The prologue commits Q, then `kStages` K tiles interleaved with `kStages - 1`
// V tiles, with each `cp_async_fence()` **outside** the `if` that guards its
// copy so the number of committed groups is fixed regardless of how many tiles
// actually exist.  `cp_async_wait<2*kStages - 1>` then drains exactly Q, and
// `cp_async_wait<2*kStages - 2>` inside the loop drains the oldest K just
// before its QK gemm consumes it.  Getting a fence inside the `if` is the
// classic way to make this deadlock on a short sequence.
//
// K and V loads are **predicated against `seqlen_k`** on the first tile, using
// the ZFILL cp.async atom.  Rows past the end arrive as zeros rather than stale
// memory, which is what retires the caller-side precondition that `V` be finite
// up to the K-*tile* boundary above the length: a NaN there used to enter
// through `P @ V`, where `0 * NaN` is NaN and no mask can intercept it, and it
// did so non-deterministically -- repeated identical runs disagreed.
//
// Every load but one -- Q, every deeper K/V tile, the paged gather -- folds
// its predicates into the ZFILL atom's own `src_size` (`cute_sm80::copy_zfill`)
// rather than into control flow.  The branching form, `if (pred) copy else
// clear`, makes ptxas order a synchronous `STS` against the asynchronous
// `LDGSTS` to the same address, and it paid for that with a `BSSY`/`BSYNC`
// pair and three dead `LDS` per copy: 144-168 dead instructions per variant,
// on every K tile.  Zero-filling is correct for every skipped element here --
// K rows past the length are masked to `-inf`, V rows must be zero (above), Q
// rows past `seqlen_q` are never stored, and the head-dim residue must be zero
// because the QK gemm runs over the padded extent.  The one exception is the
// dense first tile, issued once per CTA; `copy_rows_or_clear` records why it
// keeps the branch.

#pragma once

#include "cutlass_fmha_configs.h"
#include "fmha_bias.h"
#include "fmha_block.h"
#include "fmha_mask.h"
#include "fmha_paged_kv.h"
#include "fmha_params.h"
#include "fmha_seqlen.h"
#include "fmha_softmax.h"
#include "fmha_utils.h"

namespace oasr {
namespace attention {

using namespace cute;

template <int kNWarps_, int kStages_, bool Q_in_regs_, class TileShape_MNK_, int kHeadDimV_,
          class Element_, class ElementAccum_, int SmVersion_, bool Is_causal_,
          bool Is_local_, bool Has_bias_, bool PagedKV_, bool Split_ = false>
struct CollectiveFmhaMainloopSm80 {
    using TileShape_MNK = TileShape_MNK_;
    using Element = Element_;
    using ElementAccum = ElementAccum_;
    using ArchTraits = FmhaArch<SmVersion_>;
    using ArchTag = typename ArchTraits::Tag;

    static constexpr int kSmVersion = SmVersion_;
    static constexpr int kStages = kStages_;
    static constexpr int kNWarps = kNWarps_;
    static constexpr bool Q_in_regs = Q_in_regs_;
    static constexpr bool Is_causal = Is_causal_;
    static constexpr bool Is_local = Is_local_;
    static constexpr bool Has_bias = Has_bias_;
    static constexpr bool PagedKV = PagedKV_;
    static constexpr bool Split = Split_;
    static_assert(kStages > 0);
    static_assert(!(Is_causal && Is_local), "causal and local are exclusive");
    static_assert(!ArchTraits::kIsWarpSpecialized,
                  "this collective is the Ampere-class one; a warp-specialized arch wants "
                  "its own");

    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    static constexpr int kHeadDim = get<2>(TileShape_MNK{});
    static constexpr int kHeadDimV = kHeadDimV_;

    using SeqlenInfo_t = SeqlenInfoQK;
    using BlockMN_t = BlockMN<kBlockM, kBlockN, Is_causal, Is_local>;

    using TiledMma = TiledMMA<typename ArchTraits::template MmaAtom<Element>,
                              Layout<Shape<Int<kNWarps>, _1, _1>>,
                              Tile<Int<16 * kNWarps>, _16, _16>>;
    static constexpr int NumMmaThreads = CUTE_STATIC_V(size(TiledMma{}));

    // --- shared memory ----------------------------------------------------
    using SmemLayoutAtomQKV = typename cute_sm80::SmemLayoutAtomSwizzled<Element, kHeadDim>::type;
    using SmemLayoutQ =
        decltype(tile_to_shape(SmemLayoutAtomQKV{}, select<0, 2>(TileShape_MNK{})));
    using SmemLayoutK = decltype(tile_to_shape(
        SmemLayoutAtomQKV{},
        make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));
    using SmemLayoutV = SmemLayoutK;
    //! A *view* of V with its two leading modes swapped -- not a transpose, no
    //! data movement.  `ldmatrix.trans` does the rest when it is read.
    using SmemLayoutVt = decltype(composition(
        SmemLayoutV{},
        make_ordered_layout(make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{}),
                                       Int<kStages>{}),
                            Step<_2, _1, _3>{})));

    //! With `Q_in_regs`, Q is read to registers in the prologue and the same
    //! bytes then carry the V ring.  Saves `kStages * N * D * sizeof(Element)`.
    static constexpr bool Share_QV_Smem = Q_in_regs;

    struct TensorStorageSharedQV : cute::aligned_struct<128> {
        union {
            cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> smem_v;
            cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
        };
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k;
    };
    struct TensorStorageSeparateQV : cute::aligned_struct<128> {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> smem_v;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
    };
    using TensorStorage =
        std::conditional_t<Share_QV_Smem, TensorStorageSharedQV, TensorStorageSeparateQV>;

    // --- copy atoms -------------------------------------------------------
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0,
                  "padded head dim must be a multiple of the 128-bit load width");
    static constexpr int kBlockKGmem =
        cute_sm80::SmemLayoutAtomSwizzled<Element, kHeadDim>::kRowWidth;
    static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerLoad;
    static_assert(NumMmaThreads % kGmemThreadsPerRow == 0);
    using GmemLayoutAtom =
        Layout<Shape<Int<NumMmaThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
               Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopyQKV =
        decltype(make_tiled_copy(typename ArchTraits::template GmemCopyAtom<Element>{},
                                 GmemLayoutAtom{},
                                 Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));
    static_assert(kBlockM % CUTE_STATIC_V(shape<0>(GmemLayoutAtom{})) == 0,
                  "so the Q load never overshoots kBlockM");

    using SmemCopyAtom = typename ArchTraits::template SmemCopyAtom<Element>;
    using SmemCopyAtomTransposed = typename ArchTraits::template SmemCopyAtomTransposed<Element>;

    using Params = FmhaParams<Element>;

    /*! \brief Run the K loop for one Q tile.
     *
     * \return false when this tile has no keys to attend to; the caller then
     *   writes zeros.  The CuTeDSL backend instead clamps the range so at least
     *   one block always runs, purely to keep the prologue on one code path --
     *   returning early is strictly less work for the same answer, and it is
     *   what retires the `seqlen_k >= 1` precondition.
     */
    template <class FrgTensorO, class SoftmaxT, class SharedStorage>
    CUTLASS_DEVICE bool mma(Params const& params, FrgTensorO& tOrO, SoftmaxT& softmax,
                            int const thread_idx, SeqlenInfo_t const& info,
                            cute::tuple<int32_t, int32_t, int32_t> const& block_coord,
                            SharedStorage& shared_storage, int const split_idx = 0) {
        static_assert(is_rmem<FrgTensorO>::value, "O accumulates in registers");
        int const m_block = get<0>(block_coord);
        int const bidh = get<1>(block_coord);
        int const bidb = get<2>(block_coord);
        int const bidh_kv = params.qhead_per_khead_divmod.divide(bidh);

        auto const n_block_min_max = BlockMN_t::get_n_block_min_max(
            info, m_block, params.window_size_left, params.window_size_right);
        // Plain locals, not a structured binding: the lambdas below capture
        // these, and capturing a binding is a C++20 feature.
        int n_block_min_ = cute::get<0>(n_block_min_max);
        int n_block_max_ = cute::get<1>(n_block_min_max);
        if constexpr (Split) {
            // Each CTA walks one contiguous chunk of the range.  Everything
            // downstream -- the four-way mask split, the prologue, the ring --
            // reads `n_block_min` / `n_block_max` and needs no other change:
            // the mask predicates a narrowed range conservatively (a tile that
            // is interior to the *whole* range is also interior to a chunk of
            // it), so narrowing here is the entire mainloop-side cost.
            FmhaBlockRange const r =
                fmhaSplitRange(n_block_min_, n_block_max_, split_idx, params.num_splits);
            n_block_min_ = r.lo;
            n_block_max_ = r.hi;
        }
        int const n_block_min = n_block_min_;
        int const n_block_max = n_block_max_;
        if (n_block_max <= n_block_min) {
            return false;
        }

        Tensor sQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()),
                                SmemLayoutQ{});
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()),
                                SmemLayoutK{});
        Tensor sV = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()),
                                SmemLayoutV{});
        Tensor sVt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()),
                                 SmemLayoutVt{});

        bool const is_varlen_q = params.cu_seqlens_q != nullptr;
        bool const is_varlen_k = params.cu_seqlens_k != nullptr;

        Tensor mQ = make_tensor(
            make_gmem_ptr(params.ptr_q + info.offset_q * get<0>(params.stride_q)),
            params.shape_q, params.stride_q)(_, _, bidh, !is_varlen_q ? bidb : 0);
        Tensor gQ = local_tile(mQ, select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{}));

        // In paged mode the fourth mode is the *page*, not the batch, so the
        // pool stays un-sliced and each row picks its own page below.
        Tensor mKpool = make_tensor(
            make_gmem_ptr(params.ptr_k + (PagedKV ? 0 : info.offset_k * get<0>(params.stride_k))),
            params.shape_k, params.stride_k);
        Tensor mVpool = make_tensor(
            make_gmem_ptr(params.ptr_v + (PagedKV ? 0 : info.offset_k * get<0>(params.stride_v))),
            params.shape_k, params.stride_v);
        Tensor mK = mKpool(_, _, bidh_kv, !is_varlen_k ? bidb : 0);
        Tensor gK = local_tile(mK, select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));
        Tensor mV = mVpool(_, _, bidh_kv, !is_varlen_k ? bidb : 0);
        Tensor gV = local_tile(mV, select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));

        GmemTiledCopyQKV gmem_tiled_copy_QKV;
        auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(thread_idx);
        auto gmem_thr0_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(_0{});

        Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);
        Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
        Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);
        Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

        TiledMma tiled_mma;
        auto thr_mma = tiled_mma.get_slice(thread_idx);
        Tensor tSrQ = thr_mma.partition_fragment_A(sQ);

        auto smem_tiled_copy_Q = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
        auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(thread_idx);
        auto smem_tiled_copy_K = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
        auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(thread_idx);
        auto smem_tiled_copy_V = make_tiled_copy_B(SmemCopyAtomTransposed{}, tiled_mma);
        auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(thread_idx);
        Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
        Tensor tSsK = smem_thr_copy_K.partition_S(sK);
        Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

        // The head-dim residue: `head_dim` need only be a multiple of the
        // 128-bit load width, while the layouts are built on the padded dim.
        Tensor cKV = cute::make_identity_tensor(select<1, 2>(TileShape_MNK{}));
        Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);
        Tensor t0KVcKV = gmem_thr0_copy_QKV.partition_S(cKV);
        Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < size(tKVpKV); ++k) {
            tKVpKV(k) = get<1>(tKVcKV(_0{}, _0{}, k)) < get<1>(params.shape_k);
        }

        int const seqlen_k = info.seqlen_k;
        int n_block = n_block_max - 1;

        // --- K/V loads -----------------------------------------------------
        // `Seqlenk_mask` is only ever true on the first (highest) tile: every
        // deeper tile lies wholly inside [0, seqlen_k) by construction.
        int const max_pages = PagedKV ? int(get<1>(params.shape_pagetable)) : 0;

        // One loader for both K and V, dense or paged: every skipped row and
        // column arrives as zeros through the ZFILL atom.  For V that is
        // required, not merely tidy -- a stale NaN past the length would reach
        // the output through `P @ V`, where the softmax weight is 0 but
        // `0 * NaN` is NaN.  For K it is harmless: those columns are masked.
        auto kv_col_ok = [&](int k) { return bool(tKVpKV(k)); };
        auto load_KV = [&](auto const& mPool, auto&& sDst, auto const& tSrc, auto&& tDst,
                           int const n_blk, auto seqlenk_mask_type) {
            static constexpr bool Seqlenk_mask = decltype(seqlenk_mask_type)::value;
            if constexpr (PagedKV) {
                paged_gather_tile<kBlockN, kHeadDim, Seqlenk_mask>(
                    mPool, sDst, params.ptr_pagetable, get<0>(params.stride_pagetable), bidb,
                    bidh_kv, n_blk, seqlen_k, max_pages, params.page_size_divmod,
                    gmem_tiled_copy_QKV, gmem_thr_copy_QKV, tKVcKV, tKVpKV);
            } else if constexpr (Seqlenk_mask) {
                // The first tile, once per CTA: the branching form, on purpose.
                // See `copy_rows_or_clear` for the measurement.
                int const row_limit =
                    seqlen_k - n_blk * kBlockN - int(get<0>(tKVcKV(_0{}, _0{}, _0{})));
                copy_rows_or_clear(
                    typename ArchTraits::template GmemCopyAtom<Element>{}, tSrc, tDst,
                    [&](int m) { return int(get<0>(t0KVcKV(_0{}, m, _0{}))) < row_limit; },
                    kv_col_ok);
            } else {
                // Every deeper tile lies wholly inside [0, seqlen_k).
                cute_sm80::copy_zfill(gmem_tiled_copy_QKV, tSrc, tDst, cute_sm80::AlwaysTrue{},
                                      kv_col_ok);
            }
        };
        auto load_K = [&](int const n_blk, int const stage, auto seqlenk_mask_type) {
            load_KV(mKpool, sK(_, _, stage), tKgK(_, _, _, n_blk), tKsK(_, _, _, stage), n_blk,
                    seqlenk_mask_type);
        };
        auto load_V = [&](int const n_blk, int const stage, auto seqlenk_mask_type) {
            load_KV(mVpool, sV(_, _, stage), tVgV(_, _, _, n_blk), tVsV(_, _, _, stage), n_blk,
                    seqlenk_mask_type);
        };

        auto preprocess_Q = [&] {
            cute::cp_async_wait<Share_QV_Smem ? 1 : kStages * 2 - 1>();
            if constexpr (Q_in_regs) {
                __syncthreads();
                Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
                Tensor tSsQ_copy_view = smem_thr_copy_Q.partition_S(sQ);
                cute::copy(smem_tiled_copy_Q, tSsQ_copy_view, tSrQ_copy_view);
            }
        };

        // --- prologue ------------------------------------------------------
        if constexpr (Share_QV_Smem) {
            __syncthreads();
        }
        {
            Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
            Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
            Tensor cQ = cute::make_identity_tensor(select<0, 2>(TileShape_MNK{}));
            Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);
            Tensor t0QcQ = gmem_thr0_copy_QKV.partition_S(cQ);
            Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
            CUTLASS_PRAGMA_UNROLL
            for (int k = 0; k < size(tQpQ); ++k) {
                tQpQ(k) = get<1>(tQcQ(_0{}, _0{}, k)) < get<1>(params.shape_q);
            }
            // Q rows past `seqlen_q` arrive as zeros.  Their scores are inert
            // either way -- per-row state never crosses rows and the epilogue
            // does not store them.  Thread 0's row coordinates are
            // compile-time constants; this thread's offset is folded into the
            // limit instead.
            int const q_row_limit =
                info.seqlen_q - m_block * kBlockM - int(get<0>(tQcQ(_0{}, _0{}, _0{})));
            cute_sm80::copy_zfill(
                gmem_tiled_copy_QKV, tQgQ, tQsQ,
                [&](int m) { return int(get<0>(t0QcQ(_0{}, m, _0{}))) < q_row_limit; },
                [&](int k) { return bool(tQpQ(k)); });
        }
        cute::cp_async_fence();

        if constexpr (Share_QV_Smem) {
            // Q, then one K stage, then read Q to registers, and only then may
            // V overwrite sQ.  Without that barrier the V cp.async races the
            // ldmatrix of Q -- a cross-warp WAR the CuTeDSL backend's
            // equivalent path still has latent.
            load_K(n_block, 0, cute::true_type{});
            cute::cp_async_fence();
            preprocess_Q();
            __syncthreads();
        } else {
            __syncthreads();
        }

        // The bias plane for this (batch, head).  Its extents -- not the
        // sequence lengths -- are what the load predicates against, because it
        // is the allocation that faults.
        bool const bias_packed = Has_bias && params.bias_offsets != nullptr;
        auto bias_plane = [&] {
            if constexpr (Has_bias) {
                auto shp = bias_packed ? make_shape(info.seqlen_q, info.seqlen_k)
                                       : make_shape(int(get<0>(params.shape_bias)),
                                                    int(get<1>(params.shape_bias)));
                // A packed block-diagonal bias has **per-segment** strides:
                // segment `s`'s block is `(H, T_q_s, T_k_s)` row-major at
                // `bias_offsets[s]`, so the row stride is `seqlen_k` and the
                // head stride `seqlen_q * seqlen_k`.  Both are functions of the
                // segment, so neither can come out of `params.stride_bias` --
                // the host has one number and the kernel needs one per CTA.
                auto strd = bias_packed
                                ? make_stride(int64_t(info.seqlen_k), get<1>(params.stride_bias))
                                : make_stride(get<0>(params.stride_bias),
                                              get<1>(params.stride_bias));
                Element const* base = params.ptr_bias + info.bias_offset;
                if (!bias_packed) {
                    base += int64_t(bidh) * get<2>(params.stride_bias) +
                            int64_t(bidb) * get<3>(params.stride_bias);
                } else {
                    base += int64_t(bidh) * int64_t(info.seqlen_q) * int64_t(info.seqlen_k);
                }
                return make_tensor(make_gmem_ptr(base), shp, strd);
            } else {
                // Never dereferenced; keeps the type well-formed.
                return make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)),
                                   make_shape(0, 0), make_stride(int64_t(0), cute::_1{}));
            }
        }();

        // Vectorising the bias read needs every *row start* 4-byte aligned.
        // Under a packed bias that depends on this segment's own offset and
        // length, so the host's CTA-uniform answer does not apply and the
        // condition is re-derived here.  Note `bias_offset` is an element
        // count, not a byte count.
        bool const bias_vectorizable =
            bias_packed ? ((info.seqlen_k % 2) == 0 && (info.bias_offset % 2) == 0)
                        : params.bias_vectorizable;

        cute::for_each(cute::make_int_sequence<kStages>{}, [&](auto stage) {
            static constexpr bool Is_first_stage = CUTE_STATIC_V(stage) == 0;
            static constexpr bool Is_last_stage = CUTE_STATIC_V(stage) == kStages - 1;
            if constexpr (!Share_QV_Smem || !Is_first_stage) {
                if (Is_first_stage || n_block - stage >= n_block_min) {
                    load_K(n_block - stage, stage, cute::bool_constant<Is_first_stage>{});
                }
                // Fence outside the `if`: the committed group count must not
                // depend on how many tiles exist, or the waits below are wrong.
                cute::cp_async_fence();
            }
            if constexpr (!Is_last_stage) {
                if (Is_first_stage || n_block - stage >= n_block_min) {
                    load_V(n_block - stage, stage, cute::bool_constant<Is_first_stage>{});
                }
                cute::cp_async_fence();
            }
        });

        if constexpr (!Share_QV_Smem) {
            preprocess_Q();
        }

        Mask<kBlockM, kBlockN, TiledMma> mask(thread_idx, seqlen_k, info.seqstart_k,
                                              params.window_size_left,
                                              params.window_size_right);

        int smem_pipe_read = 0;
        int smem_pipe_write = kStages - 1;

        auto sync = [&] {
            cute::cp_async_wait<kStages * 2 - 2>();
            __syncthreads();
        };
        auto load_K_next = [&] {
            if (n_block - kStages >= n_block_min) {
                load_K(n_block - kStages, kStages > 1 ? smem_pipe_write : 0,
                       cute::false_type{});
            }
            cute::cp_async_fence();
        };

        cute::clear(tOrO);

        auto kv_step = [&](int const n_blk, auto mask_fn, auto is_first_type) {
            static constexpr bool Is_first = decltype(is_first_type)::value;
            Tensor tSrS = partition_fragment_C(tiled_mma, select<0, 1>(TileShape_MNK{}));
            cute::clear(tSrS);
            sync();
            auto load_V_next = [&] {
                if (n_blk - kStages + 1 >= n_block_min) {
                    load_V(n_blk - kStages + 1, kStages > 1 ? smem_pipe_write : 0,
                           cute::bool_constant<Is_first && kStages == 1>{});
                }
                cute::cp_async_fence();
            };
            Tensor tSrQ_cur = cute::conditional_return<Q_in_regs>(
                tSrQ, thr_mma.partition_fragment_A(sQ));
            Tensor tSrK = thr_mma.partition_fragment_B(sK(_, _, _0{}));
            // The next V copy is issued from *inside* the gemm, after the first
            // k-tile's ldmatrix, so it overlaps the whole QK rather than
            // sitting in front of it.
            cute_sm80::gemm_sm80<Q_in_regs>(
                tSrS, tSrQ_cur, tSrK, tSsQ, tSsK(_, _, _, kStages > 1 ? smem_pipe_read : 0),
                tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K,
                load_V_next);
            smem_pipe_write = smem_pipe_write < kStages - 1 ? smem_pipe_write + 1 : 0;
            if constexpr (kStages == 1) {
                sync();
                load_K_next();
            }
            if constexpr (Has_bias) {
                // Read after the QK gemm, deliberately not a tile ahead: see
                // `fmha_bias.h` for the prefetch that was measured and why it
                // is not here.
                auto rBias = make_bias_fragment<kBlockM, kBlockN, Element>(tiled_mma, thread_idx);
                load_bias_tile<kBlockM, kBlockN>(rBias, bias_plane, tiled_mma, thread_idx,
                                                 m_block, n_blk, bias_vectorizable);
                apply_bias_tile(tSrS, rBias, params.inv_softmax_scale);
            }
            // Bias first, then mask: a score modification applied *after* the
            // mask would turn a -inf back into a finite number.
            mask_fn(tSrS, n_blk);
            auto scores_scale = softmax.template max_get_scale<Is_first, /*Check_inf=*/true>(
                tSrS);
            softmax.template online_softmax<Is_first>(tSrS, scores_scale);
            Tensor tOrP_acc = make_tensor(
                tSrS.data(), cute_sm80::convert_layout_acc_Aregs<TiledMma>(tSrS.layout()));
            Tensor tOrP = cute_sm80::convert_type<Element>(tOrP_acc);
            if constexpr (!Is_first) {
                softmax.rescale_o(tOrO, scores_scale);
            }
            if constexpr (kStages > 1) {
                sync();
            }
            Tensor tOrV = thr_mma.partition_fragment_B(sVt(_, _, _0{}));
            cute_sm80::gemm_rs_sm80(tOrO, tOrP, tOrV,
                                    tOsVt(_, _, _, kStages > 1 ? smem_pipe_read : 0), tiled_mma,
                                    smem_tiled_copy_V, smem_thr_copy_V);
            if constexpr (kStages > 1) {
                load_K_next();
            }
            smem_pipe_read = smem_pipe_read < kStages - 1 ? smem_pipe_read + 1 : 0;
        };

        // --- (a) the first tile: bounded above by seqlen_k -----------------
        {
            auto mask_fn = [&](auto& tSrS, int nb) {
                mask.template apply</*Seqlenk_mask=*/true, Is_causal, Is_local,
                                    /*Seqstart_mask=*/true>(tSrS, m_block, nb);
            };
            kv_step(n_block, mask_fn, cute::true_type{});
        }
        --n_block;

        // --- (b) the causal / local right edge -----------------------------
        int const n_block_right_masked =
            BlockMN_t::get_n_block_min_right_masked(m_block, n_block_min,
                                                    params.window_size_right);
        if constexpr (Is_causal || Is_local) {
            auto mask_fn = [&](auto& tSrS, int nb) {
                mask.template apply</*Seqlenk_mask=*/false, Is_causal, Is_local,
                                    /*Seqstart_mask=*/false>(tSrS, m_block, nb);
            };
            CUTLASS_PRAGMA_NO_UNROLL
            for (; n_block >= n_block_right_masked; --n_block) {
                kv_step(n_block, mask_fn, cute::false_type{});
            }
        }

        // --- (c) the interior: no predicate at all -------------------------
        int const n_block_left_unmasked = BlockMN_t::get_n_block_min_left_unmasked(
            info, m_block, n_block_min, params.window_size_left);
        {
            auto mask_fn = [](auto&, int) {};
            CUTLASS_PRAGMA_NO_UNROLL
            for (; n_block >= n_block_left_unmasked; --n_block) {
                kv_step(n_block, mask_fn, cute::false_type{});
            }
        }

        // --- (d) the left edge: sliding window and/or seqstart_k -----------
        // Instantiated unconditionally: this is where the *runtime*
        // `seqstart_k` predicate lives, which is what keeps left padding out of
        // the compile-time variant space.  Zero iterations when neither applies.
        {
            auto mask_fn = [&](auto& tSrS, int nb) {
                mask.template apply</*Seqlenk_mask=*/false, /*Causal_mask=*/false, Is_local,
                                    /*Seqstart_mask=*/true>(tSrS, m_block, nb);
            };
            CUTLASS_PRAGMA_NO_UNROLL
            for (; n_block >= n_block_min; --n_block) {
                kv_step(n_block, mask_fn, cute::false_type{});
            }
        }

        softmax.rescale_o(tOrO, softmax.finalize());
        return true;
    }
};

}  // namespace attention
}  // namespace oasr
