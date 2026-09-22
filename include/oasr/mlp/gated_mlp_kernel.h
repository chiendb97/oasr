// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// The kernel shell: mainloop + epilogue, and the shared-memory union that lets
// the epilogue reuse the mainloop's ring.
//
// Structurally FlashAttention's `hopper/flash_fwd_kernel_sm80.h`.  This is the
// piece that makes the family arch-extensible: it names no architecture and
// touches no layout, so an SM90 warp-specialized collective is a different
// `CollectiveMainloop_` and nothing here changes.

#pragma once

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>

#include <type_traits>

#include "gated_mlp_params.h"

namespace oasr {
namespace mlp {

using namespace cute;

template <class CollectiveMainloop_, class CollectiveEpilogue_>
struct GatedMlpKernel {
    using CollectiveMainloop = CollectiveMainloop_;
    using CollectiveEpilogue = CollectiveEpilogue_;

    using TileShape_MNK = typename CollectiveMainloop::TileShape_MNK;
    using TiledMma = typename CollectiveMainloop::TiledMma;
    using ArchTag = typename CollectiveMainloop::ArchTag;
    using Element = typename CollectiveMainloop::Element;
    using Params = typename CollectiveMainloop::Params;

    static constexpr int kSmVersion = CollectiveMainloop::kSmVersion;
    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    static constexpr uint32_t NumThreads = CUTE_STATIC_V(size(TiledMma{}));
    static constexpr uint32_t MaxThreadsPerBlock = NumThreads;
    //! Deliberately 1.  `__launch_bounds__`'s second argument clamps the
    //! register budget so that many blocks fit, and this kernel carries *two*
    //! accumulator chains -- asking for two resident blocks would buy occupancy
    //! that shared memory does not grant anyway (the deep-ring tiles are one
    //! CTA per SM by construction) and pay for it in spills.
    static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

    static_assert(std::is_same_v<Params, typename CollectiveEpilogue::Params>,
                  "mainloop and epilogue read one argument struct");

    // A **union**, not an overlay: the mainloop drains its ring and barriers
    // before the epilogue writes, so the two never coexist, and the union
    // sizes itself to the larger without either side having to know the
    // other's bytes.  `gatedMlpSmemBytes` states the same `max` and the
    // launcher static_asserts the two equal.
    //
    // No `cute::aligned_struct<128>` base here, deliberately: both arms
    // already carry that base themselves, and repeating it on the enclosing
    // struct defeats the empty-base optimisation -- a base class that also
    // appears as a base of the member at offset 0 must keep a distinct
    // address, so the struct grows by a whole 128-byte alignment unit.  The
    // union's alignment is the max of its members' either way.  Measured:
    // +128 B on every tile, which the `==` assert below is what caught.
    union TensorStorage {
        typename CollectiveMainloop::TensorStorage mainloop;
        typename CollectiveEpilogue::TensorStorage epilogue;
    };

    struct SharedStorage {
        TensorStorage tensors;
    };

    static constexpr int SharedStorageSize = int(sizeof(SharedStorage));

    /*! \brief One CTA per `(n_block, m_block)`.
     *
     * N on `x` so that consecutive CTAs share `x`'s rows: the weights are read
     * once each either way, but the activation tile stays hot in L2 across the
     * N sweep.
     */
    static dim3 get_grid_shape(Params const& params) {
        return dim3(uint32_t((params.N + kBlockN - 1) / kBlockN),
                    uint32_t((params.M + kBlockM - 1) / kBlockM), 1);
    }

    static dim3 get_block_shape() { return dim3(MaxThreadsPerBlock, 1, 1); }

    CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
        SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

        int const n_block = int(blockIdx.x);
        int const m_block = int(blockIdx.y);

        TiledMma tiled_mma;
        Tensor acc_g = partition_fragment_C(tiled_mma, select<0, 1>(TileShape_MNK{}));
        Tensor acc_u = partition_fragment_C(tiled_mma, select<0, 1>(TileShape_MNK{}));
        cute::clear(acc_g);
        cute::clear(acc_u);

        CollectiveMainloop mainloop;
        CollectiveEpilogue epilogue;
        mainloop.mma(params, acc_g, acc_u, int(threadIdx.x), m_block, n_block, shared_storage);
        epilogue.store(params, acc_g, acc_u, shared_storage, tiled_mma, int(threadIdx.x),
                       m_block, n_block);
    }
};

}  // namespace mlp
}  // namespace oasr
