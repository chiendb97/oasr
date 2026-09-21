// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Work assignment for the FMHA kernel.
//
// One CTA, one Q tile: grid `(m_blocks, num_heads, batch)`.  That is what the
// CuTeDSL backend does and it is the right default -- for a rectangular
// attention every tile costs the same, so there is nothing for a scheduler to
// balance.
//
// The seam exists because two cases will want more.  A *causal* tile's cost
// falls linearly with `m_block`, so the last wave of CTAs runs far shorter than
// the first; and split-KV wants a third grid axis.  Both are served by a
// persistent scheduler over a flattened work index, which is why this is a
// class with `get_initial_work` / `get_next_work` rather than a bare
// `blockIdx` read -- adding one later changes this file and nothing else.

#pragma once

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/fast_math.h>

namespace oasr {
namespace attention {

struct FmhaTileSchedulerArguments {
    int num_blocks_m;
    int num_head;
    int batch;
    //! K-range splits per Q tile (flash-decoding).  1 disables splitting, and
    //! the grid is then exactly what it was before this axis existed.
    int num_splits = 1;
};

struct FmhaTileSchedulerParams {
    int num_blocks_m;
    int num_head;
    int batch;
    int num_splits;
    //! Divides `blockIdx.z` into `(batch, split)`.  A divmod rather than a
    //! fourth grid dimension because CUDA has only three.
    cutlass::FastDivmod split_divmod;
};

/*! \brief One tile per CTA, straight off `blockIdx`. */
struct SingleTileScheduler {
    struct SharedStorage {};

    using Arguments = FmhaTileSchedulerArguments;
    using Params = FmhaTileSchedulerParams;

    static Params to_underlying_arguments(Arguments const& args) {
        int const splits = args.num_splits > 0 ? args.num_splits : 1;
        return Params{args.num_blocks_m, args.num_head, args.batch, splits,
                      cutlass::FastDivmod(splits)};
    }

    static dim3 get_grid_shape(Params const& params) {
        return dim3(uint32_t(params.num_blocks_m), uint32_t(params.num_head),
                    uint32_t(params.batch) * uint32_t(params.num_splits));
    }

    struct WorkTileInfo {
        int m_block;
        int bidh;
        int bidb;
        int split_idx;
        bool valid_;

        CUTLASS_DEVICE bool is_valid(Params const&) const { return valid_; }
        CUTLASS_DEVICE cute::tuple<int32_t, int32_t, int32_t> get_block_coord(
            Params const&) const {
            return {m_block, bidh, bidb};
        }
        CUTLASS_DEVICE int get_split(Params const&) const { return split_idx; }
    };

    CUTLASS_DEVICE SingleTileScheduler(SharedStorage*) {}
    CUTLASS_DEVICE void init_consumer() const {}
    CUTLASS_DEVICE void prefetch_next_work(Params const&, WorkTileInfo&) const {}

    CUTLASS_DEVICE WorkTileInfo get_initial_work(Params const& params) const {
        int bidb = int(blockIdx.z);
        int split = 0;
        if (params.num_splits > 1) {
            // `z = batch * num_splits + split`, so adjacent splits of one
            // stream land on adjacent CTAs and share their K pages in L2.
            // `divmod(rem, x)` *returns* the quotient and writes the
            // remainder, so the batch is the return value, not the argument.
            bidb = params.split_divmod.divmod(split, int(blockIdx.z));
        }
        return {int(blockIdx.x), int(blockIdx.y), bidb, split, true};
    }

    CUTLASS_DEVICE WorkTileInfo get_next_work(Params const&, WorkTileInfo const&) const {
        return {0, 0, 0, 0, false};
    }
};

}  // namespace attention
}  // namespace oasr
