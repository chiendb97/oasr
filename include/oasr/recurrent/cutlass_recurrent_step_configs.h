// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Per-architecture traits for the fused recurrent step.  This is the
// `cutlass_*_configs.h` third of the three-header CUTLASS pattern (config /
// template / dispatch) that GEMM, BMM, Conv2D, attention and the gated MLP all
// follow -- see `include/oasr/mlp/cutlass_gated_mlp_configs.h`.
//
// One thing lives here: `RecurrentStepArch<SM>` -- what an architecture *is*,
// for this kernel family: its shared-memory budget, its warp slots, its MMA
// and copy atoms, and explicit capability bools.
//
// It is `constexpr` over an `sm` **argument**, never over `__CUDA_ARCH__` and
// never over a live device query, for the reasons `oasr/common/arch_facts.h`
// sets out.  The CTA tiles themselves and the ladder that picks one live next
// door in `recurrent_step_tiles.h`, which needs no CuTe.

#pragma once

// clang-format off
//
// This block is **order-sensitive and must not be sorted**.  `cute/tensor.hpp`
// is CuTe's entry point and has to come before any individual `cute/atom/*` or
// `cute/arch/*` header -- the atoms' free functions are declared against
// declarations it pulls in, and including them bare is a parse error.
//
// The repo's `.clang-format` sets `IncludeBlocks: Regroup` with
// `SortIncludes: true`, which merges the groups below and sorts
// `cute/arch/copy_sm75.hpp` *above* `cute/tensor.hpp` -- exactly the parse
// error this comment describes.  `AGENTS.md`'s formatting command covers
// `csrc/` and not `include/`, which is why the sibling families' hand-ordered
// headers survive; the guard is here so this one survives a wider invocation
// too.
#include <cute/tensor.hpp>

#include <oasr/common/arch_facts.h>

#include "recurrent_step_tiles.h"

#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma_sm80.h>
#include <cutlass/numeric_types.h>

#include <type_traits>
// clang-format on

namespace oasr {
namespace recurrent {

// ---------------------------------------------------------------------------
// Per-architecture traits
// ---------------------------------------------------------------------------

/*! \brief What one architecture is, for this kernel family.
 *
 * `Tag` selects *instructions*; `kSmemBudgetBytes` and `kMaxThreadsPerSm`
 * select *tuning*.  Keeping those two jobs apart is why sm_86, sm_89 and
 * sm_120 all route through `cutlass::arch::Sm80` -- the tag they share is the
 * tag whose instructions they run -- while still budgeting their own 99 KB and
 * their own 1536 warp slots.
 *
 * \warning Never branch on `Tag::kMinComputeCapability >= 90`.  `Sm120`'s is
 *   120, so that test is *true* on consumer Blackwell, which has neither TMA
 *   nor warp specialization.  Branch on `kHasTma` / `kIsWarpSpecialized`,
 *   which say what they mean.
 */
template <int SmVersion>
struct RecurrentStepArch;

namespace detail {

/*! \brief The Ampere-class (mma.sync + cp.async) traits every sm_8x/sm_12x shares.
 *
 * This composition -- a cp.async multistage ring, swizzled shared memory,
 * `ldmatrix.x4`, warp-level `mma.sync` m16n8k16, FP32 accumulate -- is what
 * SM80 through SM120 all actually run for FP16/BF16.  GeForce Blackwell has no
 * FP16 tcgen05 path, so its own CUTLASS GEMM uses the same warp-level atom.
 */
template <int SmVersion>
struct RecurrentStepArchAmpere {
    using Tag = cutlass::arch::Sm80;

    static constexpr int kSmVersion = SmVersion;
    static constexpr int kSmemCapacityBytes = smemCapacityForSm(SmVersion);
    static constexpr int kSmemBudgetBytes = smemBudgetForSm(SmVersion);
    static constexpr int kMaxThreadsPerSm = maxThreadsPerSmForSm(SmVersion);

    //! Register file and L2 differ enough on the consumer parts to move the
    //! tile choice; they do not change which instructions are legal.
    static constexpr bool kIsSm86Or89 = (SmVersion == 86 || SmVersion == 89);

    static constexpr bool kHasCpAsync = true;
    static constexpr bool kHasTma = false;
    static constexpr bool kIsWarpSpecialized = false;

    static constexpr int kMaxThreadsPerBlock = 1024;

    template <class Element>
    using MmaAtom = std::conditional_t<std::is_same_v<Element, cutlass::half_t>,
                                       cute::MMA_Atom<cute::SM80_16x8x16_F32F16F16F32_TN>,
                                       cute::MMA_Atom<cute::SM80_16x8x16_F32BF16BF16F32_TN>>;

    //! gmem -> smem for `previous_h` and `weight_hh`.  ZFILL, not the plain
    //! cp.async: a predicated-off copy writes **zeros** rather than leaving
    //! stale shared memory, and a zero is the identity for the dot product
    //! this kernel is accumulating.  That is what lets the K residue be
    //! handled by predication instead of by refusing every hidden width that
    //! is not a whole number of K tiles -- which is the contract the CuTeDSL
    //! lane silently imposes (it loops `ceil_div(K, k_block)` and predicates
    //! only the row axis).
    template <class Element>
    using GmemCopyAtom =
        cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>, Element>;

    //! `previous_h` is `(M, K)` row-major and `weight_hh` is `(N, K)` -- the
    //! gate-interleaved `(out, in)` layout the packer produces -- so both the
    //! A and the B operand of the TN `mma.sync` read through the *same*
    //! non-transposed `ldmatrix`.
    template <class Element>
    using SmemCopyAtom = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>;
};

}  // namespace detail

template <>
struct RecurrentStepArch<80> : detail::RecurrentStepArchAmpere<80> {};
template <>
struct RecurrentStepArch<86> : detail::RecurrentStepArchAmpere<86> {};
template <>
struct RecurrentStepArch<89> : detail::RecurrentStepArchAmpere<89> {};
template <>
struct RecurrentStepArch<120> : detail::RecurrentStepArchAmpere<120> {};

// sm_90 and sm_100 deliberately have no specialization yet.  The Ampere-class
// mainloop would *run* there (mma.sync and cp.async are both available), but
// routing to it would be a performance claim with no measurement behind it,
// and the right answer on those parts is a wgmma / tcgen05 mainloop of their
// own -- each of which needs its own epilogue as well, because neither
// partitions the accumulator the way `partition_fragment_C` does here.
// `oasr/jit/recurrent_step.py` declares the omission and counts it rather than
// hiding it (`AGENTS.md` rule 3).
//
// Adding one later is exactly three things, and none of them is in this file's
// consumers:
//
//   1. a `RecurrentStepArch<90>` here, with `kIsWarpSpecialized = true` and
//      `kHasTma = true`;
//   2. a `CollectiveRecurrentStepMainloopSm90` beside the Sm80 one, plus its
//      epilogue if the accumulator partition differs;
//   3. one more arm in the `conditional_t` that
//      `recurrent_step_launch_template.h` describes and `static_assert`s
//      against today.
//
// The kernel shell, the arguments, the tile table, the FFI signature and the
// whole Python side are arch-free and do not move.

static_assert(!RecurrentStepArch<120>::kHasTma && !RecurrentStepArch<120>::kIsWarpSpecialized,
              "consumer Blackwell has neither; a kMinComputeCapability >= 90 test would "
              "claim both");

}  // namespace recurrent
}  // namespace oasr
