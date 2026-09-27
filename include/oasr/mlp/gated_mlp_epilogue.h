// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// The gated-MLP epilogue: two fp32 accumulators -> bias -> activation -> gate
// multiply -> served dtype -> global memory.
//
// Everything downstream of the MMA happens in fp32 with **one** rounding at
// the very end.  That is a real numerical difference from the two-GEMM path it
// replaces, which rounds the activated gate and the up projection separately
// before multiplying them, and it is why
// `tests/kernels/test_gated_mlp.py::test_fused_matches_two_gemm_path` is a
// tolerance test rather than a bit-exact one.
//
// The result goes out through shared memory rather than straight to gmem.  The
// MMA-C partition gives a thread two adjacent columns per row, which on its own
// would write the tile in 4-byte pieces; staging through smem and re-reading it
// with the gmem tiled copy turns that into one 128-bit access per thread.  The
// buffer is a union with the mainloop's ring (see `gated_mlp_kernel.h`), which
// the mainloop has finished with and drained before this runs.

#pragma once

#include "cutlass_gated_mlp_configs.h"
#include "gated_mlp_params.h"

namespace oasr {
namespace mlp {

using namespace cute;

/*! \brief Finish and store one `(m_block, n_block)` output tile.
 *
 * \tparam TileShape_MN_  `(kBlockM, kBlockN)`
 * \tparam Activation_    an `oasr::ActivationType` value, applied to the
 *                        **gate** half only
 * \tparam Has_bias_      compiled in; with `false` neither bias pointer is
 *                        ever dereferenced, so a caller with no bias need not
 *                        stream a zero vector once per CTA
 */
template <class TileShape_MN_, class Element_, class ArchTag_, int NumEpilogueThreads_,
          int Activation_, bool Has_bias_>
struct CollectiveGatedMlpEpilogue {
    using TileShape_MN = TileShape_MN_;
    using Element = Element_;
    using ArchTag = ArchTag_;
    using Activation = GatedMlpActivation<Activation_>;

    static constexpr int NumEpilogueThreads = NumEpilogueThreads_;
    static constexpr bool Has_bias = Has_bias_;
    static constexpr int kBlockM = get<0>(TileShape_MN{});
    static constexpr int kBlockN = get<1>(TileShape_MN{});

    //! Sized on N, the output's contiguous axis -- a different atom from the
    //! mainloop's, which is sized on K.
    using SmemLayoutAtomO = typename cute_sm80::SmemLayoutAtomSwizzled<Element, kBlockN>::type;
    using SmemLayoutO = decltype(tile_to_shape(SmemLayoutAtomO{}, TileShape_MN{}));

    struct TensorStorage : cute::aligned_struct<128> {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutO>> smem_o;
    };

    static constexpr int kGmemElemsPerStore = sizeof(cute::uint128_t) / sizeof(Element);
    static constexpr int kSmemRowWidth =
        cute_sm80::SmemLayoutAtomSwizzled<Element, kBlockN>::kRowWidth;
    static constexpr int kGmemThreadsPerRow = kSmemRowWidth / kGmemElemsPerStore;
    static_assert(kBlockN % kSmemRowWidth == 0);
    static_assert(NumEpilogueThreads % kGmemThreadsPerRow == 0);
    static constexpr int kGmemRowsPerPass = NumEpilogueThreads / kGmemThreadsPerRow;
    static_assert(kBlockM % kGmemRowsPerPass == 0, "the store would overshoot kBlockM");

    using GmemLayoutAtomO = Layout<Shape<Int<kGmemRowsPerPass>, Int<kGmemThreadsPerRow>>,
                                   Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopyO = decltype(make_tiled_copy(
        Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{}, GmemLayoutAtomO{},
        Layout<Shape<_1, Int<kGmemElemsPerStore>>>{}));
    using SmemCopyAtomO = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>;

    using Params = GatedMlpParams<Element>;

    template <class FrgTensorC, class TiledMma, class SharedStorage>
    CUTLASS_DEVICE void store(Params const& params, FrgTensorC& acc_g, FrgTensorC const& acc_u,
                              SharedStorage& shared_storage, TiledMma tiled_mma,
                              int const thread_idx, int const m_block, int const n_block) {
        auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
        Tensor cAcc = cute::make_identity_tensor(TileShape_MN{});
        Tensor tAcc_c = thr_mma.partition_C(cAcc);

        // The `(row, col)` view exists for one reason: the bias is per column,
        // and in this view a thread's *distinct* columns are `size<1>`.  Read
        // per element instead and the same few values are fetched once per row.
        Tensor acc_g_rc =
            make_tensor(acc_g.data(), cute_sm80::convert_layout_acc_rowcol(acc_g.layout()));
        Tensor acc_u_rc =
            make_tensor(acc_u.data(), cute_sm80::convert_layout_acc_rowcol(acc_u.layout()));
        Tensor tAcc_c_rc =
            make_tensor(tAcc_c.data(), cute_sm80::convert_layout_acc_rowcol(tAcc_c.layout()));

        int const n0 = n_block * kBlockN;
        // Declared unconditionally so the loop below can name them; with
        // `Has_bias` false nothing ever reads them and they cost no registers.
        Tensor bias_g = make_tensor<float>(make_shape(size<1>(acc_g_rc)));
        Tensor bias_u = make_tensor<float>(make_shape(size<1>(acc_u_rc)));
        if constexpr (Has_bias) {
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < size<1>(acc_g_rc); ++j) {
                int const col = n0 + int(get<1>(tAcc_c_rc(_0{}, j)));
                bool const in_range = col < params.N;
                bias_g(j) = in_range ? float(params.ptr_bg[col]) : 0.f;
                bias_u(j) = in_range ? float(params.ptr_bu[col]) : 0.f;
            }
        }

        // In place into `acc_g`: the gated result needs no fragment of its own.
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size<0>(acc_g_rc); ++i) {
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < size<1>(acc_g_rc); ++j) {
                float g = acc_g_rc(i, j);
                float u = acc_u_rc(i, j);
                if constexpr (Has_bias) {
                    g += bias_g(j);
                    u += bias_u(j);
                }
                acc_g_rc(i, j) = Activation::apply(g) * u;
            }
        }
        Tensor rO = cute_sm80::convert_type<Element>(acc_g);

        // rmem -> smem, through the MMA's own C partition.  The mainloop has
        // already drained its ring and barriered, which is what makes writing
        // over it sound; see `CollectiveGatedMlpMainloopSm80::mma`.
        Tensor sO = make_tensor(make_smem_ptr(shared_storage.tensors.epilogue.smem_o.data()),
                                SmemLayoutO{});
        auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
        auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(thread_idx);
        cute::copy(smem_tiled_copy_O, smem_thr_copy_O.retile_S(rO),
                   smem_thr_copy_O.partition_D(sO));
        __syncthreads();

        // smem -> rmem -> gmem, now coalesced.
        Tensor mO = make_tensor(make_gmem_ptr(params.ptr_o),
                                make_shape(params.M, params.N),
                                make_stride(params.stride_o, _1{}));
        Tensor gO = local_tile(mO, TileShape_MN{}, make_coord(m_block, n_block));

        GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(thread_idx);
        auto gmem_thr0_copy_O = gmem_tiled_copy_O.get_thread_slice(_0{});
        Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_fragment_like(tOsO);
        cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

        Tensor cO = cute::make_identity_tensor(TileShape_MN{});
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor t0OcO = gmem_thr0_copy_O.partition_D(cO);
        int const row_limit =
            params.M - m_block * kBlockM - int(get<0>(tOcO(_0{}, _0{}, _0{})));
        int const col_limit = params.N - n0 - int(get<1>(tOcO(_0{}, _0{}, _0{})));
        Tensor tOpO = make_tensor<bool>(make_shape(size<1>(tOgO)));
        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < size(tOpO); ++m) {
            tOpO(m) = int(get<0>(t0OcO(_0{}, m, _0{}))) < row_limit;
        }
        // Column predication is per 128-bit *vector*, not per element: `N` and
        // `kBlockN` are both multiples of the 8-element store width, so a
        // vector is either wholly in range or wholly out and there is no
        // straddling case to split.
        cute_sm80::copy_if(
            gmem_tiled_copy_O, tOrO, tOgO, [&](int m) { return bool(tOpO(m)); },
            [&](int k) { return int(get<1>(t0OcO(_0{}, _0{}, k))) < col_limit; });
    }
};

}  // namespace mlp
}  // namespace oasr
