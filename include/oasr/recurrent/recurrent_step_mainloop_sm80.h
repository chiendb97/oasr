// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// The Ampere-class recurrent-step mainloop: a cp.async ring carrying the
// `previous_h` tile and the `weight_hh` tile, and one m16n8k16 chain
// accumulating the recurrent affine in FP32.
//
// Structurally the FlashAttention SM80 collective
// (`hopper/mainloop_fwd_sm80.hpp`) with the QK/PV pair replaced by a single
// gemm.  The collective / shell decomposition and the cp.async fence
// accounting are FA's and are worth copying exactly.
//
// ---------------------------------------------------------------------------
// Ring and drain order
// ---------------------------------------------------------------------------
//
// The prologue commits `kStages - 1` groups, one per K tile, with each
// `cp_async_fence()` **outside** the `if` that guards its copy so the number
// of committed groups does not depend on how many tiles exist.  Group index
// then equals K-tile index, and `cp_async_wait<kStages - 2>` at the top of
// iteration `t` drains exactly tile `t`.  Getting a fence inside the `if` is
// the classic way to make this deadlock on a short K -- and a recurrent step's
// K is the hidden width, which at 256 and a 64-wide K tile is only four tiles
// against a five-stage ring.
//
// The refill for tile `t + kStages - 1` is issued from *inside* the gemm --
// after the first k-step's `ldmatrix`, through `gemm_sm80`'s hook -- so the
// copy overlaps the tensor-core work instead of sitting in front of it.  It
// writes the stage that iteration `t - 1` read, which the top-of-loop
// `__syncthreads()` has already retired; no second barrier is needed, and the
// CuTeDSL lane's extra one in the same position is doing nothing.
//
// ---------------------------------------------------------------------------
// Predication
// ---------------------------------------------------------------------------
//
// `M` (the cohort) and `N` (the gate-interleaved width) are runtime extents
// and need not be tile multiples, so the row axis of both operands is
// predicated -- once, into registers, before the loop, because the bound does
// not change across K.
//
// `K` is predicated too, which the CuTeDSL lane does not do at all: it loops
// `ceil_div(K, k_block)` and predicates only the row axis, so a hidden width
// that is not a whole number of K tiles reads past the end of both operands.
// With the ZFILL cp.async atom a predicated-off load writes **zeros**, and a
// zero is the identity for the dot product being accumulated -- so a K residue
// is exactly right rather than merely safe, and it costs one compare per copy.

#pragma once

#include "cutlass_recurrent_step_configs.h"
#include "recurrent_step_params.h"

namespace oasr {
namespace recurrent {

using namespace cute;

template <class TileShape_MNK_, int kStages_, int kWarpsM_, int kWarpsN_, class Element_,
          class ElementAccum_, int SmVersion_>
struct CollectiveRecurrentStepMainloopSm80 {
    using TileShape_MNK = TileShape_MNK_;
    using Element = Element_;
    using ElementAccum = ElementAccum_;
    using ArchTraits = RecurrentStepArch<SmVersion_>;
    using ArchTag = typename ArchTraits::Tag;

    static constexpr int kSmVersion = SmVersion_;
    static constexpr int kStages = kStages_;
    static constexpr int kWarpsM = kWarpsM_;
    static constexpr int kWarpsN = kWarpsN_;
    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    static constexpr int kBlockK = get<2>(TileShape_MNK{});

    static_assert(kStages >= 2,
                  "the refill writes the stage the previous iteration read, so one stage "
                  "would have the ring overwrite itself");
    static_assert(!ArchTraits::kIsWarpSpecialized,
                  "this collective is the Ampere-class one; a warp-specialized arch wants "
                  "its own");

    //! The MMA is tiled `(kWarpsM, kWarpsN)` over a 16x16 permutation of the
    //! m16n8k16 atom.  Warps tile M *and* N: a recurrent step has a small M
    //! (the cohort) and a large N (gates * hidden), so spending every warp on
    //! M -- as an attention kernel does -- would force `kBlockM` up to
    //! `warps * 16` and leave the N axis to a single warp.  `kWarpsN` is what
    //! lets a 16-row cohort still use four warps.
    using TiledMma = TiledMMA<typename ArchTraits::template MmaAtom<Element>,
                              Layout<Shape<Int<kWarpsM>, Int<kWarpsN>, _1>>,
                              Tile<Int<16 * kWarpsM>, Int<16 * kWarpsN>, _16>>;
    static constexpr int NumMmaThreads = CUTE_STATIC_V(size(TiledMma{}));

    // --- shared memory ----------------------------------------------------
    //! Chosen from the K tile: K is the contiguous axis of both operands.
    using SmemLayoutAtom = typename cute_sm80::SmemLayoutAtomSwizzled<Element, kBlockK>::type;
    using SmemLayoutA = decltype(tile_to_shape(
        SmemLayoutAtom{}, make_shape(Int<kBlockM>{}, Int<kBlockK>{}, Int<kStages>{})));
    using SmemLayoutB = decltype(tile_to_shape(
        SmemLayoutAtom{}, make_shape(Int<kBlockN>{}, Int<kBlockK>{}, Int<kStages>{})));

    struct TensorStorage : cute::aligned_struct<128> {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutA>> smem_a;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutB>> smem_b;
    };

    // --- copy atoms -------------------------------------------------------
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static constexpr int kSmemRowWidth = cute_sm80::SmemLayoutAtomSwizzled<Element, kBlockK>::kRowWidth;
    static constexpr int kGmemThreadsPerRow = kSmemRowWidth / kGmemElemsPerLoad;
    static_assert(kBlockK % kSmemRowWidth == 0);
    static_assert(NumMmaThreads % kGmemThreadsPerRow == 0);
    static constexpr int kGmemRowsPerPass = NumMmaThreads / kGmemThreadsPerRow;
    // A gmem->smem pass wider than the tile puts the *partition* out of range,
    // which is an illegal access no predicate can intercept.
    static_assert(kBlockM % kGmemRowsPerPass == 0, "previous_h's load would overshoot kBlockM");
    static_assert(kBlockN % kGmemRowsPerPass == 0, "weight_hh's load would overshoot kBlockN");

    using GmemLayoutAtom = Layout<Shape<Int<kGmemRowsPerPass>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopy =
        decltype(make_tiled_copy(typename ArchTraits::template GmemCopyAtom<Element>{},
                                 GmemLayoutAtom{}, Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));
    using SmemCopyAtom = typename ArchTraits::template SmemCopyAtom<Element>;

    using Params = RecurrentStepParams<Element>;

    /*! \brief Run the whole K loop for one `(m_block, n_block)` gate tile.
     *
     * `acc` comes in cleared and goes out holding the raw recurrent affine
     * `previous_h @ weight_hh^T`.  The input projection, the nonlinearities
     * and the cell update are the epilogue's, in FP32, with one rounding at
     * the end.
     */
    template <class FrgTensorC, class SharedStorage>
    CUTLASS_DEVICE void mma(Params const& params, FrgTensorC& acc, int const thread_idx,
                            int const m_block, int const n_block, SharedStorage& shared_storage) {
        static_assert(is_rmem<FrgTensorC>::value, "the affine accumulates in registers");

        Tensor mA = make_tensor(make_gmem_ptr(params.ptr_prev_h), make_shape(params.M, params.K),
                                make_stride(params.stride_prev_h, _1{}));
        Tensor mB = make_tensor(make_gmem_ptr(params.ptr_weight), make_shape(params.N, params.K),
                                make_stride(params.stride_weight, _1{}));

        Tensor gA = local_tile(mA, Shape<Int<kBlockM>, Int<kBlockK>>{}, make_coord(m_block, _));
        Tensor gB = local_tile(mB, Shape<Int<kBlockN>, Int<kBlockK>>{}, make_coord(n_block, _));

        Tensor sA = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_a.data()),
                                SmemLayoutA{});
        Tensor sB = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_b.data()),
                                SmemLayoutB{});

        GmemTiledCopy gmem_tiled_copy;
        auto gmem_thr_copy = gmem_tiled_copy.get_thread_slice(thread_idx);
        auto gmem_thr0_copy = gmem_tiled_copy.get_thread_slice(_0{});

        Tensor tAgA = gmem_thr_copy.partition_S(gA);  // (CPY, CPY_M, CPY_K, k_tiles)
        Tensor tAsA = gmem_thr_copy.partition_D(sA);  // (CPY, CPY_M, CPY_K, kStages)
        Tensor tBgB = gmem_thr_copy.partition_S(gB);
        Tensor tBsB = gmem_thr_copy.partition_D(sB);

        // --- row predicates, computed once -------------------------------
        Tensor cA = cute::make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockK>>{});
        Tensor cB = cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kBlockK>>{});
        Tensor tAcA = gmem_thr_copy.partition_S(cA);
        Tensor tBcB = gmem_thr_copy.partition_S(cB);
        Tensor t0AcA = gmem_thr0_copy.partition_S(cA);
        Tensor t0BcB = gmem_thr0_copy.partition_S(cB);

        int const a_row_limit = params.M - m_block * kBlockM - int(get<0>(tAcA(_0{}, _0{}, _0{})));
        int const b_row_limit = params.N - n_block * kBlockN - int(get<0>(tBcB(_0{}, _0{}, _0{})));
        int const k_col_offset = int(get<1>(tAcA(_0{}, _0{}, _0{})));

        Tensor tApA = make_tensor<bool>(make_shape(size<1>(tAsA)));
        Tensor tBpB = make_tensor<bool>(make_shape(size<1>(tBsB)));
        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < size(tApA); ++m) {
            tApA(m) = int(get<0>(t0AcA(_0{}, m, _0{}))) < a_row_limit;
        }
        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < size(tBpB); ++m) {
            tBpB(m) = int(get<0>(t0BcB(_0{}, m, _0{}))) < b_row_limit;
        }

        // --- MMA fragments ------------------------------------------------
        TiledMma tiled_mma;
        auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
        Tensor tCrA = thr_mma.partition_fragment_A(sA(_, _, _0{}));
        Tensor tCrB = thr_mma.partition_fragment_B(sB(_, _, _0{}));

        auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
        auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(thread_idx);
        auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
        auto smem_thr_copy_B = smem_tiled_copy_B.get_thread_slice(thread_idx);
        Tensor tCsA = smem_thr_copy_A.partition_S(sA);
        Tensor tCsB = smem_thr_copy_B.partition_S(sB);

        // `int(kBlockK)`, not `kBlockK`: `cute::ceil_div` takes its arguments by
        // const reference, which ODR-uses the class-scope `static constexpr` and
        // leaves nvcc looking for a device-side definition that a host-side inline
        // variable does not have ("identifier ... is undefined in device code").
        int const k_tiles = cute::ceil_div(params.K, int(kBlockK));

        // One `load_tile` for both operands: they share the tiled copy, the
        // row-pass geometry and the K bound, and one `cp_async_fence()` covers
        // the pair, so group index stays equal to K-tile index.
        auto load_tile = [&](int const k_tile, int const stage) {
            int const max_k = params.K - k_tile * kBlockK - k_col_offset;
            // The K predicate compares thread 0's compile-time column against a
            // limit with this thread's own offset already folded in.
            auto k_ok = [&](int k) { return int(get<1>(t0AcA(_0{}, _0{}, k))) < max_k; };
            cute_sm80::copy_zfill(gmem_tiled_copy, tAgA(_, _, _, k_tile), tAsA(_, _, _, stage),
                                  [&](int m) { return bool(tApA(m)); }, k_ok);
            cute_sm80::copy_zfill(gmem_tiled_copy, tBgB(_, _, _, k_tile), tBsB(_, _, _, stage),
                                  [&](int m) { return bool(tBpB(m)); }, k_ok);
        };

        // --- prologue ------------------------------------------------------
        CUTLASS_PRAGMA_UNROLL
        for (int stage = 0; stage < kStages - 1; ++stage) {
            if (stage < k_tiles) {
                load_tile(stage, stage);
            }
            // Outside the `if`: the committed group count must not depend on
            // how many K tiles exist, or the waits below are wrong.
            cute::cp_async_fence();
        }

        // --- mainloop -------------------------------------------------------
        int smem_pipe_read = 0;
        int smem_pipe_write = kStages - 1;

        CUTLASS_PRAGMA_NO_UNROLL
        for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
            cute::cp_async_wait<kStages - 2>();
            __syncthreads();

            int const next_tile = k_tile + kStages - 1;
            int const write_stage = smem_pipe_write;
            cute_sm80::gemm_sm80(acc, tCrA, tCrB, tCsA(_, _, _, smem_pipe_read),
                                 tCsB(_, _, _, smem_pipe_read), tiled_mma, smem_tiled_copy_A,
                                 smem_tiled_copy_B, smem_thr_copy_A, smem_thr_copy_B, [&] {
                                     if (next_tile < k_tiles) {
                                         load_tile(next_tile, write_stage);
                                     }
                                     cute::cp_async_fence();
                                 });

            smem_pipe_read = smem_pipe_read + 1 < kStages ? smem_pipe_read + 1 : 0;
            smem_pipe_write = smem_pipe_write + 1 < kStages ? smem_pipe_write + 1 : 0;
        }

        // The epilogue lays its staging buffer over this ring, so every
        // outstanding copy has to have landed and every warp has to be done
        // reading before it writes.  Both are once per CTA.
        cute::cp_async_wait<0>();
        __syncthreads();
    }
};

}  // namespace recurrent
}  // namespace oasr
