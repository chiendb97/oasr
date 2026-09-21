// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// The one place in this family that names an architecture.
//
// `run_fmha<Arch, ...>` resolves the tile from `fmhaResolveTile`, assembles
// a (mainloop, epilogue, scheduler) triple, and launches it.  Adding a Hopper
// or Blackwell lane later is a second `FmhaArch<SM>` specialization, a second
// collective, and one more arm in the `conditional_t` below -- the kernel
// shell, the arguments, the FFI signature and the whole Python side are
// arch-free and do not move.
//
// This mirrors FlashAttention's `hopper/flash_fwd_launch_template.h:36-55`,
// which selects `ArchTag` and the collective the same way.  One deliberate
// difference: FA dispatches on `Arch >= 90`, which would be *true* for sm_120.
// Here the dispatch is on `FmhaArch<SM>::kIsWarpSpecialized`, a bool that says
// what it means, so consumer Blackwell cannot fall into a Hopper path.

#pragma once

#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/device_kernel.h>

#include <oasr/common/arch_dispatch.h>

#include "cutlass_fmha_configs.h"
#include "epilogue.h"
#include "fmha_combine.h"
#include "fmha_params.h"
#include "mainloop_sm80.h"
#include "tile_scheduler.h"
// `fmha_kernel_sm80.h` must follow the scheduler: it names
// `TileScheduler::SharedStorage` in its own union.
#include "fmha_kernel_sm80.h"

namespace oasr {
namespace attention {

/*! \brief Build and launch one fully-specialised FMHA kernel.
 *
 * \tparam Arch        compute capability as `major*10 + minor`
 * \tparam kHeadDim    **padded** head dim
 * \tparam Element     `cutlass::half_t` or `cutlass::bfloat16_t`
 *
 * The tile is *derived*, never passed in: `fmhaResolveTile` is `constexpr`, so
 * the whole config collapses at compile time and there is nothing for a caller
 * -- Python included -- to get out of step with.  The `static_assert` below
 * closes the last gap, catching any drift between the resolver's arithmetic and
 * the real `sizeof(SharedStorage)` (the union and its padding make the two
 * non-obviously equal).
 */
template <int Arch, int kHeadDim, int kHeadDimV, class Element, bool Is_causal, bool Is_local,
          bool Has_bias, bool PagedKV, bool Split = false>
cudaError_t run_fmha(FmhaParams<Element> const& params, cudaStream_t stream) {
    using ArchTraits = FmhaArch<Arch>;

    static constexpr FmhaTile kTile =
        fmhaResolveTile(Arch, kHeadDim, kHeadDimV, int(sizeof(Element)));
    static_assert(kTile.valid,
                  "no (K tile, ring depth) fits this architecture's shared memory at this "
                  "head dim -- the Python side must refuse the config before it gets here");

    using TileShape_MNK =
        cute::Shape<cute::Int<kTile.block_m>, cute::Int<kTile.block_n>, cute::Int<kHeadDim>>;

    // The extensibility seam.  Today every `FmhaArch` specialization is
    // Ampere-class, so this resolves unconditionally; a Hopper or Blackwell
    // lane becomes
    //
    //     using CollectiveMainloop = std::conditional_t<
    //         ArchTraits::kIsWarpSpecialized,
    //         CollectiveMainloopSm90<...>,
    //         CollectiveMainloopSm80<...>>;
    //
    // and nothing below this line changes.  Asserting it here rather than
    // writing a `conditional_t` whose two arms are identical keeps the
    // invariant checkable instead of decorative.
    static_assert(!ArchTraits::kIsWarpSpecialized,
                  "a warp-specialized arch needs its own collective; add the conditional_t "
                  "arm described above");
    using CollectiveMainloop =
        CollectiveMainloopSm80<kTile.num_warps, kTile.num_stages, kTile.q_in_regs,
                                  TileShape_MNK, kHeadDimV, Element, float, Arch, Is_causal,
                                  Is_local, Has_bias, PagedKV, Split>;

    using CollectiveEpilogue = CollectiveEpilogue<
        cute::Shape<cute::Int<kTile.block_m>, cute::Int<kHeadDimV>, cute::Int<kTile.block_n>>,
        Element, typename ArchTraits::Tag, CollectiveMainloop::NumMmaThreads>;

    using Scheduler = SingleTileScheduler;
    using AttnKernel = FmhaKernelSm80<CollectiveMainloop, CollectiveEpilogue, Scheduler>;

    static_assert(AttnKernel::SharedStorageSize <= ArchTraits::kSmemBudgetBytes,
                  "fmhaResolveTile and sizeof(SharedStorage) disagree");

    typename AttnKernel::Params kernel_params;
    kernel_params.mainloop = params;
    // `shape_q`'s head-dim mode carries the **real** head dim, not the padded
    // one, and that is exactly what the epilogue's store predicate needs: it
    // bounds how many columns of each row are written.  Substituting
    // `kHeadDimV` here writes the *padded* width into a tensor that is only
    // `head_dim` wide -- at head_dim 16 (padded 32) that is 16 columns of
    // overrun per row, landing in the next row and corrupting output that the
    // kernel itself computed correctly.  Data-dependent, and invisible at every
    // head dim that happens to be a multiple of 32.
    //
    // When a separate value width lands (D_v != D_q), it arrives as its own
    // runtime extent on the arguments, not as the compile-time padded constant.
    kernel_params.epilogue = typename CollectiveEpilogue::Params{
        params.ptr_o,          params.shape_q,        params.stride_o, params.cu_seqlens_q,
        params.ptr_o_partial, params.ptr_lse_partial};

    // For packed input the caller sets `shape_q` to
    // `(max_seqlen_q, D, H, num_segments)` with a zero batch stride, so the
    // grid is derived the same way in both modes and the mainloop's early exit
    // handles the segments shorter than `max_seqlen_q`.
    int const num_blocks_m = cute::ceil_div(int(cute::get<0>(params.shape_q)), kTile.block_m);
    int const num_splits = Split ? (params.num_splits > 0 ? params.num_splits : 1) : 1;
    kernel_params.scheduler = Scheduler::to_underlying_arguments(
        {num_blocks_m, int(cute::get<2>(params.shape_q)), int(cute::get<3>(params.shape_q)),
         num_splits});

    dim3 const grid = AttnKernel::get_grid_shape(kernel_params);
    dim3 const block = AttnKernel::get_block_shape();
    int const smem_size = AttnKernel::SharedStorageSize;

    auto kernel = cutlass::device_kernel<AttnKernel>;
    // The table above says what the architecture offers; this asks what *this*
    // device grants.  They agree on every part we ship, and when they do not
    // the failure should name both numbers rather than surface as a driver
    // error from inside `cudaFuncSetAttribute`.
    if (smem_size > oasr::getDeviceMaxSharedMemoryOptin()) {
        return cudaErrorInvalidValue;
    }
    cudaError_t status = oasr::optInSharedMemory(kernel, size_t(smem_size));
    if (status != cudaSuccess) {
        return status;
    }
    kernel<<<grid, block, smem_size, stream>>>(kernel_params);
    return cudaGetLastError();
}

/*! \brief The combine pass for a split launch.
 *
 * Separate from `run_fmha` rather than chained onto it because the caller
 * owns the workspace and therefore owns the decision to split at all; a
 * launcher that decided for itself would need the SM count and the shape, and
 * that is exactly the decision rule 11 says must be visible and pure.
 *
 * `Arch` is not used: the combine is arch-free, and it is deliberately *not*
 * tiled by `fmhaResolveTile`'s M tile -- see `fmha_combine.h` for the
 * measurement that made tiling it that way a 2.6x tax on the kernel it
 * reduces. The parameter stays so the call site reads like `run_fmha`'s.
 */
template <int Arch, int kHeadDim, class Element>
cudaError_t run_fmha_combine_for(FmhaParams<Element> const& params, cudaStream_t stream) {
    return oasr::attention::run_fmha_combine<Element, kHeadDim>(
        params.ptr_o, params.ptr_o_partial, params.ptr_lse_partial, params.num_splits,
        int(cute::get<0>(params.shape_q)), int(cute::get<1>(params.shape_q)),
        int(cute::get<2>(params.shape_q)), int(cute::get<3>(params.shape_q)),
        cute::get<0>(params.stride_o), cute::get<2>(params.stride_o),
        cute::get<3>(params.stride_o), stream);
}

}  // namespace attention
}  // namespace oasr
