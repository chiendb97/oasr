// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// The kernel shell: mainloop + epilogue + scheduler, and the shared-memory
// union that lets the epilogue reuse the mainloop's K/V region.
//
// Structurally FlashAttention's `hopper/flash_fwd_kernel_sm80.h`.  This is the
// piece that makes the family arch-extensible: it names no architecture and
// touches no layout, so an SM90 warp-specialized collective is a different
// `CollectiveMainloop_` and nothing here changes.  (Named without an `Sm80`
// suffix for that reason, like `GatedMlpKernel` and `RecurrentStepKernel`.)

#pragma once

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>

#include "fmha_seqlen.h"
#include "fmha_softmax.h"
#include "fmha_utils.h"

namespace oasr {
namespace attention {

using namespace cute;

template <class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_>
struct FmhaKernel {
    using CollectiveMainloop = CollectiveMainloop_;
    using CollectiveEpilogue = CollectiveEpilogue_;
    using TileScheduler = TileScheduler_;

    using TileShape_MNK = typename CollectiveMainloop::TileShape_MNK;
    using TiledMma = typename CollectiveMainloop::TiledMma;
    using ArchTag = typename CollectiveMainloop::ArchTag;
    using Element = typename CollectiveMainloop::Element;
    using MainloopParams = typename CollectiveMainloop::Params;
    using EpilogueParams = typename CollectiveEpilogue::Params;
    using SchedulerParams = typename TileScheduler::Params;
    using SeqlenInfo_t = typename CollectiveMainloop::SeqlenInfo_t;

    static constexpr int kSmVersion = CollectiveMainloop::kSmVersion;
    static constexpr bool Split = CollectiveMainloop::Split;
    static constexpr uint32_t NumThreads = CUTE_STATIC_V(size(TiledMma{}));
    static constexpr uint32_t MaxThreadsPerBlock = NumThreads;
    static constexpr uint32_t MinBlocksPerMultiprocessor = NumThreads == 128 ? 2 : 1;
    static constexpr int kBlockM = get<0>(TileShape_MNK{});

    // The epilogue's `smem_o` is laid over the mainloop's `smem_v + smem_k` and
    // never over `smem_q` -- which is what keeps the union sound when `smem_q`
    // is itself aliased onto `smem_v` under `Q_in_regs`.  Pad the mainloop side
    // when `sO` is the larger of the two so the overlay still starts at
    // `smem_v`.
    static constexpr int mainloop_smem_padding_ =
        int(sizeof(typename CollectiveEpilogue::TensorStorage)) -
        int(sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_v))) -
        int(sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_k)));
    static constexpr int mainloop_smem_padding =
        mainloop_smem_padding_ < 0 ? 0 : mainloop_smem_padding_;

    struct SharedStorage {
        struct TensorStorage : cute::aligned_struct<128> {
            union {
                struct {
                    cute::array<uint32_t, mainloop_smem_padding / sizeof(uint32_t)> padding_;
                    typename CollectiveMainloop::TensorStorage mainloop;
                };
                typename CollectiveEpilogue::TensorStorage epilogue;
            };
        } tensors;
        alignas(16) typename TileScheduler::SharedStorage smem_scheduler;
    };

    static constexpr int SharedStorageSize = sizeof(SharedStorage);

    struct Params {
        MainloopParams mainloop;
        EpilogueParams epilogue;
        SchedulerParams scheduler;
    };

    static dim3 get_grid_shape(Params const& params) {
        return TileScheduler::get_grid_shape(params.scheduler);
    }
    static dim3 get_block_shape() { return dim3(MaxThreadsPerBlock, 1, 1); }

    CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
        SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

        CollectiveMainloop mainloop;
        CollectiveEpilogue epilogue;
        TileScheduler scheduler(
            reinterpret_cast<typename TileScheduler::SharedStorage*>(
                &shared_storage.smem_scheduler));
        TiledMma tiled_mma;
        scheduler.init_consumer();

        CUTLASS_PRAGMA_NO_UNROLL
        for (auto work_tile_info = scheduler.get_initial_work(params.scheduler);
             work_tile_info.is_valid(params.scheduler);
             work_tile_info = scheduler.get_next_work(params.scheduler, work_tile_info)) {
            Tensor tOrO = partition_fragment_C(tiled_mma, select<0, 2>(TileShape_MNK{}));
            auto block_coord = work_tile_info.get_block_coord(params.scheduler);
            int const bidb = get<2>(block_coord);

            // One row of the MMA-C accumulator per (row-pair, MMA_M).
            Softmax<2 * (2 * kBlockM / int(NumThreads))> softmax(
                params.mainloop.softmax_scale_log2);
            softmax.init();

            SeqlenInfo_t info{bidb,
                              int(get<0>(params.mainloop.shape_q)),
                              int(get<0>(params.mainloop.shape_k)),
                              params.mainloop.cu_seqlens_q,
                              params.mainloop.cu_seqlens_k,
                              params.mainloop.seqused_k,
                              params.mainloop.seqstarts_k,
                              params.mainloop.bias_offsets};

            int const split_idx = work_tile_info.get_split(params.scheduler);
            bool const tile_valid =
                mainloop.mma(params.mainloop, tOrO, softmax, threadIdx.x, info, block_coord,
                             shared_storage, split_idx);
            scheduler.prefetch_next_work(params.scheduler, work_tile_info);
            if constexpr (Split) {
                // A split with no K tiles still stores: its LSE is -inf, which
                // the combine weights to exactly zero.  Skipping the store
                // instead would leave whatever the workspace held last time,
                // and the workspace is uninitialised by design -- allocating
                // it zeroed would cost a full memset of
                // `num_splits * B * H * T_q * D` floats per call.
                if (!tile_valid) {
                    cute::clear(tOrO);
                }
                // `log_sum_exp` reads the post-`finalize` row sum, which `mma`
                // has already all-reduced across the quad; on the invalid path
                // `row_max` is still -inf and it short-circuits to -inf.
                epilogue.store_partial(params.epilogue, tOrO, softmax.log_sum_exp(), tiled_mma,
                                       threadIdx.x, info, block_coord, split_idx);
            } else if (tile_valid) {
                epilogue.store(params.epilogue, tOrO, shared_storage, tiled_mma, threadIdx.x,
                               info, block_coord);
            } else {
                epilogue.store_zero(params.epilogue, threadIdx.x, info, block_coord);
            }
        }
    }
};

}  // namespace attention
}  // namespace oasr
