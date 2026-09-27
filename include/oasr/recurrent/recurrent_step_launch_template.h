// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// The one place in this family that names an architecture.
//
// `run_recurrent_step<Arch, TileIdx, ...>` reads the tile out of
// `kRecurrentStepTiles`, assembles a (mainloop, epilogue) pair and launches
// it.  Adding a Hopper or Blackwell lane later is a second
// `RecurrentStepArch<SM>` specialization, a second collective, and one more
// arm in the `conditional_t` described below -- the kernel shell, the
// arguments, the tile table, the FFI signature and the whole Python side are
// arch-free and do not move.
//
// This mirrors FlashAttention's `hopper/flash_fwd_launch_template.h:36-55`.
// One deliberate difference: FA dispatches on `Arch >= 90`, which would be
// *true* for sm_120.  Here the dispatch is on
// `RecurrentStepArch<SM>::kIsWarpSpecialized`, a bool that says what it means,
// so consumer Blackwell cannot fall into a Hopper path.

#pragma once

#include <cuda_runtime.h>

#include <oasr/common/cute_sm80.h>

#include "cutlass_recurrent_step_configs.h"
#include "recurrent_step_epilogue.h"
#include "recurrent_step_kernel.h"
#include "recurrent_step_mainloop_sm80.h"
#include "recurrent_step_params.h"
#include "recurrent_step_tiles.h"

namespace oasr {
namespace recurrent {

/*! \brief Build and launch one fully-specialised fused recurrent step.
 *
 * \tparam Arch     compute capability as `major*10 + minor`
 * \tparam TileIdx  index into `kRecurrentStepTiles`
 * \tparam Element  `cutlass::half_t` or `cutlass::bfloat16_t`
 * \tparam Kind     which recurrence; fixes the gate count and the cell state
 *
 * Everything about the configuration collapses at compile time, so there is
 * nothing here for a caller -- Python included -- to get out of step with.
 * The `static_assert` on `SharedStorageSize` closes the last gap: the tile
 * table's arithmetic and the real `sizeof(SharedStorage)` are non-obviously
 * equal once the union and its alignment are involved, and the Python routing
 * sizes occupancy from the former.
 */
template <int Arch, int TileIdx, class Element, RecurrentKind Kind>
cudaError_t run_recurrent_step(RecurrentStepParams<Element> const& params, cudaStream_t stream) {
    using ArchTraits = RecurrentStepArch<Arch>;

    static_assert(TileIdx >= 0 && TileIdx < kRecurrentStepTileCount, "no such tile");
    static constexpr RecurrentStepTile kTile = kRecurrentStepTiles[TileIdx];
    static constexpr int kGates = recurrentGateCount(Kind);
    static_assert(recurrentStepTileValid(kTile, Arch, int(sizeof(Element)), kGates),
                  "this tile does not fit this architecture -- the Python side must refuse "
                  "the config before it gets here");

    static constexpr int kWarpsN = kTile.warps_n;
    static constexpr int kWarpsM = kTile.threads / 32 / kTile.warps_n;

    using TileShape_MNK =
        cute::Shape<cute::Int<kTile.block_m>, cute::Int<kTile.block_n>, cute::Int<kTile.block_k>>;

    // The extensibility seam.  Today every `RecurrentStepArch` specialization
    // is Ampere-class, so this resolves unconditionally; a Hopper or Blackwell
    // lane becomes
    //
    //     using CollectiveMainloop = std::conditional_t<
    //         ArchTraits::kIsWarpSpecialized,
    //         CollectiveRecurrentStepMainloopSm90<...>,
    //         CollectiveRecurrentStepMainloopSm80<...>>;
    //
    // and nothing below this line changes.  Asserting it here rather than
    // writing a `conditional_t` whose two arms are identical keeps the
    // invariant checkable instead of decorative.
    static_assert(!ArchTraits::kIsWarpSpecialized,
                  "a warp-specialized arch needs its own collective; add the conditional_t "
                  "arm described above");
    using CollectiveMainloop =
        CollectiveRecurrentStepMainloopSm80<TileShape_MNK, kTile.stages, kWarpsM, kWarpsN, Element,
                                            float, Arch>;
    using CollectiveEpilogue = CollectiveRecurrentStepEpilogue<
        cute::Shape<cute::Int<kTile.block_m>, cute::Int<kTile.block_n>>, Element,
        typename ArchTraits::Tag, CollectiveMainloop::NumMmaThreads, Kind>;

    using Kernel = RecurrentStepKernel<CollectiveMainloop, CollectiveEpilogue>;

    static_assert(int(CollectiveMainloop::NumMmaThreads) == kTile.threads,
                  "the tile's thread count and the MMA's disagree");
    static_assert(Kernel::SharedStorageSize == recurrentStepSmemBytes(kTile, int(sizeof(Element))),
                  "the tile table and sizeof(SharedStorage) disagree; the routing sizes "
                  "occupancy from the table");
    static_assert(Kernel::SharedStorageSize <= ArchTraits::kSmemBudgetBytes,
                  "this tile overflows the architecture's shared memory");

    // The table says what the architecture offers; the launcher asks what
    // *this* device grants before opting in -- a five-stage ring once failed
    // at launch with an empty message by skipping exactly that.
    return cute_sm80::launch_kernel<Kernel>(params, stream);
}

}  // namespace recurrent
}  // namespace oasr
