// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// The FMHA epilogue: fp32 accumulator -> served dtype -> global memory.
//
// Structurally modelled on FlashAttention's `hopper/epilogue_fwd.hpp`, minus
// its split-KV, PackGQA and FP8 paths, and with **one correction that the
// original would get wrong on consumer Blackwell**: FA selects its TMA store
// and its smem layout with `ArchTag::kMinComputeCapability >= 90`
// (`epilogue_fwd.hpp:37,80`).  `cutlass::arch::Sm120::kMinComputeCapability` is
// 120, so that test is *true* on a part with no TMA at all.  This epilogue asks
// `FmhaArch<SM>::kHasTma` instead, which says what it means.
//
// The accumulator goes out through shared memory rather than straight to gmem.
// The MMA-C partition scatters a thread's values across the row in a pattern
// that makes a direct global store a handful of narrow, uncoalesced
// transactions; staging through smem and re-reading it with the gmem tiled copy
// turns that into one 128-bit access per thread.  `sO` deliberately aliases the
// mainloop's `sV + sK` region -- the mainloop is finished with both by the time
// the epilogue runs -- and never `sQ`, which is what keeps it sound when `sQ`
// is itself aliased onto `sV`.

#pragma once

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include "cutlass_fmha_configs.h"
#include "fmha_params.h"
#include "fmha_seqlen.h"
#include "fmha_utils.h"

namespace oasr {
namespace attention {

using namespace cute;

/*! \brief Store `O` for one Q tile.
 *
 * \tparam TileShape_MNK_PV `(kBlockM, kHeadDimV, kBlockN)`
 * \tparam NumEpilogueThreads the mainloop's MMA thread count
 */
template <class TileShape_MNK_PV_, class Element_, class ArchTag_, int NumEpilogueThreads_>
struct CollectiveEpilogue {
    using TileShape_MNK_PV = TileShape_MNK_PV_;
    using Element = Element_;
    using ArchTag = ArchTag_;
    static constexpr int NumEpilogueThreads = NumEpilogueThreads_;

    static constexpr int kBlockM = get<0>(TileShape_MNK_PV{});
    static constexpr int kHeadDimV = get<1>(TileShape_MNK_PV{});

    using SmemLayoutAtomO = typename FmhaSmemLayoutAtom<Element, kHeadDimV>::type;
    using SmemLayoutO =
        decltype(tile_to_shape(SmemLayoutAtomO{}, select<0, 1>(TileShape_MNK_PV{})));

    struct TensorStorage : cute::aligned_struct<128> {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutO>> smem_o;
    };

    static constexpr int kGmemElemsPerStore = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDimV % kGmemElemsPerStore == 0,
                  "head_dim must be a multiple of the 128-bit store width");
    static constexpr int kBlockKGmem = FmhaSmemLayoutAtom<Element, kHeadDimV>::kBlockKGmem;
    static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerStore;
    static_assert(NumEpilogueThreads % kGmemThreadsPerRow == 0);
    using GmemLayoutAtomO =
        Layout<Shape<Int<NumEpilogueThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
               Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopyO = decltype(make_tiled_copy(
        Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{}, GmemLayoutAtomO{},
        Layout<Shape<_1, Int<kGmemElemsPerStore>>>{}));
    using SmemCopyAtomO = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>;

    struct Arguments {
        Element* ptr_o;
        ShapeQKV shape_o;  //!< (T_q, D_v, H, B)
        StrideQKV stride_o;
        int32_t const* cu_seqlens_q;
        //! Split-KV partials; null unless the kernel was launched split.
        float* ptr_o_partial = nullptr;    //!< (num_splits, B, H, T_q, D) fp32
        float* ptr_lse_partial = nullptr;  //!< (num_splits, B, H, T_q) fp32
    };
    using Params = Arguments;

    static Params to_underlying_arguments(Arguments const& args) { return args; }

    /*! \brief Write one Q tile's output.
     *
     * `tOrO` must already be finalized (divided by the row sum) by the caller.
     */
    template <class FrgTensorO, class TiledMma, class SharedStorage>
    CUTLASS_DEVICE void store(Params const& params, FrgTensorO& tOrO,
                              SharedStorage& shared_storage, TiledMma tiled_mma,
                              int const thread_idx, SeqlenInfoQK const& info,
                              cute::tuple<int32_t, int32_t, int32_t> const& block_coord) {
        auto [m_block, bidh, bidb] = block_coord;

        Tensor sO = make_tensor(make_smem_ptr(shared_storage.tensors.epilogue.smem_o.data()),
                                SmemLayoutO{});
        Tensor tOrO_out = convert_type<Element>(tOrO);

        // rmem -> smem, through the MMA's own C partition.
        auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
        auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(thread_idx);
        Tensor taccOrO = smem_thr_copy_O.retile_S(tOrO_out);
        Tensor taccOsO = smem_thr_copy_O.partition_D(sO);
        // Every warp must be done reading sV/sK before we write over them.
        __syncthreads();
        cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);
        __syncthreads();

        // smem -> rmem -> gmem, now coalesced.
        bool const is_varlen = params.cu_seqlens_q != nullptr;
        Tensor mO = make_tensor(make_gmem_ptr(params.ptr_o + info.offset_q *
                                                                 get<0>(params.stride_o)),
                                params.shape_o, params.stride_o)(_, _, bidh,
                                                                 !is_varlen ? bidb : 0);
        Tensor gO = local_tile(mO, select<0, 1>(TileShape_MNK_PV{}), make_coord(m_block, _0{}));

        GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(thread_idx);
        Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO_store = make_fragment_like(tOsO);
        cute::copy(gmem_tiled_copy_O, tOsO, tOrO_store);

        Tensor cO = cute::make_identity_tensor(select<0, 1>(TileShape_MNK_PV{}));
        Tensor tOcO = gmem_thr_copy_O.partition_S(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < size(tOpO); ++k) {
            tOpO(k) = get<1>(tOcO(_0{}, _0{}, k)) < get<1>(params.shape_o);
        }
        // Rows past this stream's query length are never written: under varlen
        // they belong to the *next* segment, so zeroing them would corrupt a
        // neighbour rather than merely waste a store.
        copy_predicated</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false,
                        /*Clear_OOB_K=*/false>(gmem_tiled_copy_O, tOrO_store, tOgO, tOcO, tOpO,
                                               info.seqlen_q - m_block * kBlockM);
    }

    /*! \brief Write one split's unnormalised-but-rescaled `O` and its LSE.
     *
     * Straight from registers to global memory, with no shared-memory staging.
     * The staging in :func:`store` exists to turn the MMA-C scatter into one
     * 128-bit access per thread, and it works because `sO` can overlay the
     * mainloop's `sV + sK`.  An fp32 `O` is twice as wide, and at a wide head
     * dim on a narrow K tile the overlay no longer fits inside what the
     * mainloop already reserved -- it would have to grow the shared-memory
     * budget for every variant to speed up the one that decodes.  The partials
     * are also small by construction (split-KV only fires when `T_q` is tiny),
     * so the uncoalesced store is charged against a few hundred kilobytes.
     *
     * `tOrO` must already be finalized; `lse` comes from
     * `Softmax::log_sum_exp()` and is **base 2**.
     */
    template <class FrgTensorO, class TiledMma, class LseT>
    CUTLASS_DEVICE void store_partial(
        Params const& params, FrgTensorO const& tOrO, LseT const& lse, TiledMma tiled_mma,
        int const thread_idx, SeqlenInfoQK const& info,
        cute::tuple<int32_t, int32_t, int32_t> const& block_coord, int const split_idx) {
        auto [m_block, bidh, bidb] = block_coord;
        int const T_q = int(get<0>(params.shape_o));
        int const D = int(get<1>(params.shape_o));
        int const H = int(get<2>(params.shape_o));
        int const B = int(get<3>(params.shape_o));

        auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
        Tensor cO = cute::make_identity_tensor(select<0, 1>(TileShape_MNK_PV{}));
        Tensor tOcO = thr_mma.partition_C(cO);
        Tensor tOcO_rc = make_tensor(tOcO.data(), convert_layout_acc_rowcol(tOcO.layout()));
        Tensor tOrO_rc = make_tensor(tOrO.data(), convert_layout_acc_rowcol(tOrO.layout()));

        int64_t const plane = int64_t(T_q) * D;
        float* o_base = params.ptr_o_partial +
                        ((int64_t(split_idx) * B + bidb) * H + bidh) * plane;
        float* lse_base = params.ptr_lse_partial +
                          ((int64_t(split_idx) * B + bidb) * H + bidh) * T_q;

        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < size<0>(tOrO_rc); ++m) {
            int const row = int(get<0>(tOcO_rc(m, _0{}))) + m_block * kBlockM;
            if (row >= info.seqlen_q) {
                continue;
            }
            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < size<1>(tOrO_rc); ++n) {
                int const col = int(get<1>(tOcO_rc(m, n)));
                if (col < D) {
                    o_base[int64_t(row) * D + col] = tOrO_rc(m, n);
                }
            }
            // One lane per row owns column 0; let it write the LSE so the
            // other three do not race to store the same value.
            if (int(get<1>(tOcO_rc(m, _0{}))) == 0) {
                lse_base[row] = lse(m);
            }
        }
    }

    /*! \brief Write zeros for a Q tile with no keys to attend to.
     *
     * A fully masked query row is defined to come back **zero**, not NaN: a NaN
     * pad row is not inert, because in the next layer a masked key still
     * contributes `0 * NaN` and poisons the rows that are real.
     */
    CUTLASS_DEVICE void store_zero(Params const& params, int const thread_idx,
                                   SeqlenInfoQK const& info,
                                   cute::tuple<int32_t, int32_t, int32_t> const& block_coord) {
        auto [m_block, bidh, bidb] = block_coord;
        bool const is_varlen = params.cu_seqlens_q != nullptr;
        Tensor mO = make_tensor(make_gmem_ptr(params.ptr_o + info.offset_q *
                                                                 get<0>(params.stride_o)),
                                params.shape_o, params.stride_o)(_, _, bidh,
                                                                 !is_varlen ? bidb : 0);
        Tensor gO = local_tile(mO, select<0, 1>(TileShape_MNK_PV{}), make_coord(m_block, _0{}));

        GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(thread_idx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_fragment_like(tOgO);
        cute::clear(tOrO);

        Tensor cO = cute::make_identity_tensor(select<0, 1>(TileShape_MNK_PV{}));
        Tensor tOcO = gmem_thr_copy_O.partition_S(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < size(tOpO); ++k) {
            tOpO(k) = get<1>(tOcO(_0{}, _0{}, k)) < get<1>(params.shape_o);
        }
        copy_predicated</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false,
                        /*Clear_OOB_K=*/false>(gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO,
                                               info.seqlen_q - m_block * kBlockM);
    }
};

}  // namespace attention
}  // namespace oasr
