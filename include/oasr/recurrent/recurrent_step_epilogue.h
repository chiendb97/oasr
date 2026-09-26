// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// The recurrent-step epilogue: FP32 affine -> input projection -> gate
// nonlinearities -> cell update -> served dtype -> global memory.
//
// Everything downstream of the MMA happens in FP32 with **one** rounding at
// the very end, which is what makes a 640-term dot product safe in half
// precision.
//
// ---------------------------------------------------------------------------
// Why the accumulator goes back through shared memory
// ---------------------------------------------------------------------------
//
// An LSTM cell consumes four *adjacent* gate columns, and the m16n8k16
// thread-value layout gives one thread only two of them.  Staging the
// accumulator and re-reading it by coordinate makes the gather independent of
// that layout instead of encoding it -- which is also what keeps this epilogue
// correct if a future architecture's MMA atom partitions C differently.  A
// `__shfl_xor_sync(.., 1)` would exchange the missing pair on SM80 in fewer
// instructions, and it is deliberately not used: it is a fact about *this*
// atom's lane assignment, and the whole point of the collective split is that
// the epilogue survives a new mainloop.
//
// ---------------------------------------------------------------------------
// Two things this does that the CuTeDSL lane does not
// ---------------------------------------------------------------------------
//
// 1. **The global loads are issued before the staging barrier.**  The
//    transition needs `input_gates` and `previous_c` from global memory and
//    the affine from shared.  Issuing the two gmem loads *first* puts their
//    latency behind the accumulator store and the `__syncthreads()` that
//    follows it, instead of in front of the arithmetic.  They cost registers
//    (at most `kSlotsPerThread * (kGates + 1)` halves -- 20 registers on the
//    widest tile) and nothing else; the CuTeDSL lane issues them after the
//    barrier and pays the latency exposed.
//
// 2. **The staging buffer is padded by 8 floats per row, not 1.**  Both
//    numbers avoid the write conflict the unpadded layout has; only this one
//    also keeps every row 16-byte aligned, so a cell's four gates come back in
//    one `LDS.128` instead of four `LDS.32`.  The arithmetic, for a warp
//    storing the MMA-C fragment (8 rows x 4 column-pairs, issued as two
//    16-lane phases of `STS.64`):
//
//      stride % 32 == 8  ->  rows 0..3 start at banks 0, 8, 16, 24 and each
//      covers 8 banks with its four column-pairs, so a phase covers all 32
//      banks exactly once.  Rows 4..7 repeat it in the second phase.
//
//    and for the read (four-lane-group `LDS.128`, consecutive threads taking
//    consecutive hidden units) the same stride leaves each 8-lane phase on 32
//    distinct banks.  `+1` is conflict-free for the store and *misaligns* the
//    load; `+8` is conflict-free for both.

#pragma once

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include <cute/tensor.hpp>

#include "cutlass_recurrent_step_configs.h"
#include "recurrent_step_params.h"
#include "recurrent_step_utils.h"

namespace oasr {
namespace recurrent {

using namespace cute;

/*! \brief Finish and store one `(m_block, n_block)` gate tile.
 *
 * \tparam TileShape_MN_  `(kBlockM, kBlockN)` in *gate-interleaved* columns
 * \tparam Kind_          which recurrence; fixes the gate count and whether
 *                        the cell pointers are live
 */
template <class TileShape_MN_, class Element_, class ArchTag_, int NumEpilogueThreads_,
          RecurrentKind Kind_>
struct CollectiveRecurrentStepEpilogue {
    using TileShape_MN = TileShape_MN_;
    using Element = Element_;
    using ArchTag = ArchTag_;
    using Transition = RecurrentTransition<Kind_>;

    static constexpr RecurrentKind kKind = Kind_;
    static constexpr int NumEpilogueThreads = NumEpilogueThreads_;
    static constexpr int kBlockM = get<0>(TileShape_MN{});
    static constexpr int kBlockN = get<1>(TileShape_MN{});
    static constexpr int kGates = Transition::kGates;
    static constexpr bool kHasCell = Transition::kHasCell;

    //! Hidden units this tile owns.  The tile holds *whole* units by
    //! construction (`recurrentStepTileValid` refuses any other `block_n`), so
    //! the transition never straddles a tile edge and no cross-CTA reduction
    //! is needed.
    static constexpr int kUnits = kBlockN / kGates;
    static_assert(kUnits >= 1 && kBlockN % kGates == 0,
                  "a CTA tile must hold a whole number of hidden units");

    //! `(row, unit)` slots per thread.  Exact, not a ceiling: the tile table
    //! only admits tiles where it divides (`recurrentStepTileValid`), which is
    //! what lets the loop below carry no bound check on `slot` itself.
    static constexpr int kSlots = kBlockM * kUnits;
    static_assert(kSlots % NumEpilogueThreads == 0,
                  "the epilogue's slot count must divide the CTA; the tile table refuses "
                  "anything else");
    static constexpr int kSlotsPerThread = kSlots / NumEpilogueThreads;

    // -----------------------------------------------------------------------
    // Known: the widest one-gate variant spills
    // -----------------------------------------------------------------------
    //
    // `cuobjdump -res-usage` on the built cells, sm_120, fp16:
    //
    //     every LSTM variant          REG 54-68,  STACK 0   (2-8 slots/thread)
    //     LSTM  tile 7 (128x128)      REG 120,    STACK 0   (8 slots/thread)
    //     RNN   tile 6 (128x64)       REG 118,    STACK 0   (16 slots/thread)
    //     RNN   tile 7 (128x128)      REG 128,    STACK 32  (32 slots/thread)
    //
    // A 512-thread CTA gets 128 registers, so that last row is against the
    // ceiling and spills into the inner loop.  Two plausible causes were
    // tested and **both refuted by measurement**, which is why they are
    // recorded rather than left for someone to re-try:
    //
    //   1. carrying the slot indices across the barrier instead of
    //      recomputing them -- recomputing changed the register count by
    //      exactly zero; ptxas was already rematerialising them;
    //   2. hoisting every slot's global load above the barrier -- capping the
    //      hoist at 16 slots also changed it by exactly zero.
    //
    // The remaining suspect is the fully-unrolled transition loop itself: at
    // 32 slots `CUTLASS_PRAGMA_UNROLL` keeps that many iterations' addresses
    // and results live at once.  Bounding the unroll is the next thing to try,
    // and it needs measuring against the LSTM variants, which are the ones on
    // a routed path and are all comfortably inside the budget.
    //
    // Not fixed here because nothing reaches it: a vanilla RNN is never routed
    // under `auto` (`oasr/jit/recurrent_cute.py` refuses `gate_count != 4` --
    // there is no matched single-step reference to route it against), tile 7
    // needs hidden > 1536 *and* batch > 128, and no shipped model uses a
    // vanilla RNN at all.  Declared rather than silently carried
    // (`AGENTS.md` rule 3).

    //! FP32 accumulator staging, padded by 8 floats per row.  See the header
    //! comment: `+8` is the pad that is conflict-free for the MMA-C store
    //! *and* 16-byte aligned for the four-gate load.
    static constexpr int kAccStride = kBlockN + 8;
    static_assert(kAccStride % 4 == 0, "a staged row must stay 16-byte aligned");
    static_assert(kAccStride % 32 == 8 || kBlockN % 32 != 0,
                  "the store's conflict-free pad is stride % 32 == 8");
    using SmemLayoutAcc =
        decltype(make_layout(TileShape_MN{}, make_stride(Int<kAccStride>{}, _1{})));

    struct TensorStorage : cute::aligned_struct<128> {
        cute::array_aligned<float, cute::cosize_v<SmemLayoutAcc>> smem_acc;
    };

    //! The MMA-C fragment's inner mode is a *pair of adjacent columns*, so the
    //! store vectorises to 8 bytes.  128-bit is not reachable here and asking
    //! for it would only make the atom fall back.
    using SmemCopyAtomAcc = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<64>, float>;

    //! One hidden unit's gates, as they sit in `input_gates`.
    using GateVec = cutlass::AlignedArray<Element, kGates>;

    using Params = RecurrentStepParams<Element>;

    template <class FrgTensorC, class TiledMma, class SharedStorage>
    CUTLASS_DEVICE void store(Params const& params, FrgTensorC const& acc,
                              SharedStorage& shared_storage, TiledMma tiled_mma,
                              int const thread_idx, int const m_block, int const n_block) {
        int const row0 = m_block * kBlockM;
        int const unit0 = n_block * kUnits;

        // One hidden unit's gates, as an aligned vector.  8 bytes for an LSTM,
        // which the shape contract's `stride % 8 == 0` guarantees is addressable.
        auto load_gates = [&](int m, int hid) {
            return *reinterpret_cast<GateVec const*>(params.ptr_gates + m * params.stride_gates +
                                                     int64_t(hid) * kGates);
        };
        auto load_prev_c = [&](int m, int hid) -> Element {
            if constexpr (kHasCell) {
                return params.ptr_prev_c[m * params.stride_prev_c + hid];
            } else {
                (void)m;
                (void)hid;
                return Element(0);
            }
        };

        // --- 1. issue the global loads first --------------------------------
        // Nothing below depends on them until after the barrier, so their
        // latency hides behind the accumulator store and the barrier itself.
        // Only the loaded *values* cross the barrier; the slot indices are
        // recomputed below, which costs nothing -- carrying them instead was
        // measured at exactly the same register count, because ptxas
        // rematerialises them either way.
        GateVec gates_reg[kSlotsPerThread];
        Element prev_c_reg[kSlotsPerThread];

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kSlotsPerThread; ++i) {
            int const slot = thread_idx + i * NumEpilogueThreads;
            int const r = slot / kUnits;
            int const m = row0 + r;
            int const hid = unit0 + (slot - r * kUnits);
            if (m < params.M && hid < params.H) {
                gates_reg[i] = load_gates(m, hid);
                prev_c_reg[i] = load_prev_c(m, hid);
            }
        }

        // --- 2. stage the accumulator ---------------------------------------
        // The mainloop has drained its ring and barriered, which is what makes
        // writing over it sound; see `CollectiveRecurrentStepMainloopSm80::mma`.
        Tensor sAcc = make_tensor(make_smem_ptr(shared_storage.tensors.epilogue.smem_acc.data()),
                                  SmemLayoutAcc{});
        auto smem_tiled_copy_acc = make_tiled_copy_C(SmemCopyAtomAcc{}, tiled_mma);
        auto smem_thr_copy_acc = smem_tiled_copy_acc.get_thread_slice(thread_idx);
        cute::copy(smem_tiled_copy_acc, smem_thr_copy_acc.retile_S(acc),
                   smem_thr_copy_acc.partition_D(sAcc));
        __syncthreads();

        // --- 3. transition and store ----------------------------------------
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kSlotsPerThread; ++i) {
            int const slot = thread_idx + i * NumEpilogueThreads;
            int const r = slot / kUnits;
            int const u = slot - r * kUnits;
            int const m = row0 + r;
            int const hid = unit0 + u;
            if (m >= params.M || hid >= params.H) {
                continue;
            }

            float g[kGates];
            float const* acc_row = &sAcc(r, u * kGates);
            if constexpr (kGates == 4) {
                // One LDS.128; the `+8` pad is what makes this address
                // 16-byte aligned on every row.
                //
                // The raw cast is the one place this file leaves CuTe, so it
                // was checked rather than assumed: `cuobjdump -sass` on the
                // built cell shows 24 `LDS.128` and no generic `LD`, i.e.
                // ptxas recovers the shared address space through the cast
                // and emits the wide *shared* load.  If that ever stops being
                // true the fix is `recast<float4>(sAcc)`, which keeps the
                // `smem_ptr` -- `kAccStride % 4 == 0` above is what makes that
                // legal.  Do not "simplify" this to four scalar reads.
                float4 const v = *reinterpret_cast<float4 const*>(acc_row);
                g[0] = v.x;
                g[1] = v.y;
                g[2] = v.z;
                g[3] = v.w;
            } else {
                CUTLASS_PRAGMA_UNROLL
                for (int j = 0; j < kGates; ++j) {
                    g[j] = acc_row[j];
                }
            }
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < kGates; ++j) {
                g[j] += float(gates_reg[i][j]);
            }

            float const prev_c = kHasCell ? float(prev_c_reg[i]) : 0.f;
            float out_h = 0.f;
            float out_c = 0.f;
            Transition::apply(g, prev_c, out_h, out_c);

            params.ptr_h[m * params.stride_h + hid] = Element(out_h);
            if constexpr (kHasCell) {
                params.ptr_c[m * params.stride_c + hid] = Element(out_c);
            }
        }
    }
};

}  // namespace recurrent
}  // namespace oasr
