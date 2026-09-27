// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// The one place in this family that names an architecture.
//
// `run_gated_mlp<Arch, TileIdx, ...>` reads the tile out of `kGatedMlpTiles`,
// assembles a (mainloop, epilogue) pair and launches it.  Adding a Hopper or
// Blackwell lane later is a second `GatedMlpArch<SM>` specialization, a second
// collective, and one more arm in the `conditional_t` described below -- the
// kernel shell, the epilogue, the arguments, the FFI signature and the whole
// Python side are arch-free and do not move.
//
// This mirrors FlashAttention's `hopper/flash_fwd_launch_template.h:36-55`.
// One deliberate difference: FA dispatches on `Arch >= 90`, which would be
// *true* for sm_120.  Here the dispatch is on
// `GatedMlpArch<SM>::kIsWarpSpecialized`, a bool that says what it means, so
// consumer Blackwell cannot fall into a Hopper path.

#pragma once

#include <cuda_runtime.h>

#include <oasr/common/cute_sm80.h>

#include "cutlass_gated_mlp_configs.h"
#include "gated_mlp_epilogue.h"
#include "gated_mlp_kernel.h"
#include "gated_mlp_mainloop_sm80.h"
#include "gated_mlp_params.h"

namespace oasr {
namespace mlp {

/*! \brief Build and launch one fully-specialised gated-MLP kernel.
 *
 * \tparam Arch        compute capability as `major*10 + minor`
 * \tparam TileIdx     index into `kGatedMlpTiles`
 * \tparam Element     `cutlass::half_t` or `cutlass::bfloat16_t`
 * \tparam Activation  an `oasr::ActivationType` value
 * \tparam Has_bias    whether the bias pointers are live
 *
 * Everything about the configuration collapses at compile time, so there is
 * nothing here for a caller -- Python included -- to get out of step with.
 * The `static_assert` on `SharedStorageSize` closes the last gap: the tile
 * table's arithmetic and the real `sizeof(SharedStorage)` are non-obviously
 * equal once the union and its alignment are involved, and the Python routing
 * sizes occupancy from the former.
 */
template <int Arch, int TileIdx, class Element, int Activation, bool Has_bias>
cudaError_t run_gated_mlp(GatedMlpParams<Element> const& params, cudaStream_t stream) {
    using ArchTraits = GatedMlpArch<Arch>;

    static_assert(TileIdx >= 0 && TileIdx < kGatedMlpTileCount, "no such tile");
    static constexpr GatedMlpTile kTile = kGatedMlpTiles[TileIdx];
    static_assert(gatedMlpTileValid(kTile, Arch, int(sizeof(Element))),
                  "this tile does not fit this architecture -- the Python side must refuse "
                  "the config before it gets here");

    static constexpr int kWarpsN = kTile.warps_n;
    static constexpr int kWarpsM = kTile.threads / 32 / kTile.warps_n;

    using TileShape_MNK =
        cute::Shape<cute::Int<kTile.block_m>, cute::Int<kTile.block_n>,
                    cute::Int<kTile.block_k>>;

    // The extensibility seam.  Today every `GatedMlpArch` specialization is
    // Ampere-class, so this resolves unconditionally; a Hopper or Blackwell
    // lane becomes
    //
    //     using CollectiveMainloop = std::conditional_t<
    //         ArchTraits::kIsWarpSpecialized,
    //         CollectiveGatedMlpMainloopSm90<...>,
    //         CollectiveGatedMlpMainloopSm80<...>>;
    //
    // and nothing below this line changes.  Asserting it here rather than
    // writing a `conditional_t` whose two arms are identical keeps the
    // invariant checkable instead of decorative.
    static_assert(!ArchTraits::kIsWarpSpecialized,
                  "a warp-specialized arch needs its own collective; add the conditional_t "
                  "arm described above");
    using CollectiveMainloop =
        CollectiveGatedMlpMainloopSm80<TileShape_MNK, kTile.stages, kWarpsM, kWarpsN, Element,
                                       float, Arch>;
    using CollectiveEpilogue = CollectiveGatedMlpEpilogue<
        cute::Shape<cute::Int<kTile.block_m>, cute::Int<kTile.block_n>>, Element,
        typename ArchTraits::Tag, CollectiveMainloop::NumMmaThreads, Activation, Has_bias>;

    using Kernel = GatedMlpKernel<CollectiveMainloop, CollectiveEpilogue>;

    static_assert(int(CollectiveMainloop::NumMmaThreads) == kTile.threads,
                  "the tile's thread count and the MMA's disagree");
    static_assert(Kernel::SharedStorageSize ==
                      gatedMlpSmemBytes(kTile, int(sizeof(Element))),
                  "the tile table and sizeof(SharedStorage) disagree; the routing sizes "
                  "occupancy from the table");
    static_assert(Kernel::SharedStorageSize <= ArchTraits::kSmemBudgetBytes,
                  "this tile overflows the architecture's shared memory");

    // The table says what the architecture offers; the launcher asks what
    // *this* device grants before opting in.
    return cute_sm80::launch_kernel<Kernel>(params, stream);
}

}  // namespace mlp
}  // namespace oasr
