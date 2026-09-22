// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Per-architecture traits and the CTA-tile table for the fused gated MLP.
// This is the `cutlass_*_configs.h` third of the three-header CUTLASS pattern
// (config / template / dispatch) that GEMM, BMM, Conv2D and attention already
// follow -- see `include/oasr/attention/cutlass_fmha_configs.h`.
//
// One thing lives here: `GatedMlpArch<SM>` -- what an architecture *is*, for
// this kernel family: its shared-memory budget, its warp slots, its MMA and
// copy atoms, and explicit capability bools.
//
// It is `constexpr` over an `sm` **argument**, never over `__CUDA_ARCH__` and
// never over a live device query, for the reasons `oasr/common/arch_facts.h`
// sets out.  The CTA tiles themselves and the arithmetic that picks one live
// next door in `gated_mlp_tiles.h`, which needs no CuTe.

#pragma once

// `cute/tensor.hpp` is CuTe's entry point and must come before any individual
// `cute/atom/*` header -- the atoms' free functions are declared against
// declarations it pulls in, and including them bare is a parse error.
#include <cute/tensor.hpp>

#include <oasr/common/arch_facts.h>

#include "gated_mlp_tiles.h"

#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma_sm80.h>
#include <cutlass/numeric_types.h>

#include <type_traits>

namespace oasr {
namespace mlp {

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
struct GatedMlpArch;

namespace detail {

/*! \brief The Ampere-class (mma.sync + cp.async) traits every sm_8x/sm_12x shares. */
template <int SmVersion>
struct GatedMlpArchAmpere {
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

    //! gmem -> smem for X and both weights.  ZFILL, not the plain cp.async: a
    //! predicated-off copy writes **zeros** rather than leaving stale shared
    //! memory, and a zero is the identity for the dot product this kernel is
    //! accumulating.  That is what lets the K residue be handled by predication
    //! instead of by refusing every `K` that is not a whole number of K tiles
    //! -- which is the contract the CuTeDSL lane has to impose.
    template <class Element>
    using GmemCopyAtom =
        cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>, Element>;

    template <class Element>
    using GmemCopyAtomO =
        cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>, Element>;

    //! X and both weights are K-major -- `nn.Linear`'s own `(out, in)` layout
    //! for the weights -- so the A and B operands of the TN `mma.sync` read
    //! through the *same* non-transposed `ldmatrix`.
    template <class Element>
    using SmemCopyAtom = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>;
};

}  // namespace detail

template <>
struct GatedMlpArch<80> : detail::GatedMlpArchAmpere<80> {};
template <>
struct GatedMlpArch<86> : detail::GatedMlpArchAmpere<86> {};
template <>
struct GatedMlpArch<89> : detail::GatedMlpArchAmpere<89> {};
template <>
struct GatedMlpArch<120> : detail::GatedMlpArchAmpere<120> {};

// sm_90 and sm_100 deliberately have no specialization yet.  The Ampere-class
// mainloop would *run* there (mma.sync and cp.async are both available), but
// routing to it would be a performance claim with no measurement behind it,
// and the right answer on those parts is a wgmma/TMA mainloop of their own.
// `oasr/jit/gated_mlp.py` declares the omission and counts it rather than
// hiding it.  Adding one later is: a `GatedMlpArch<90>` here, a collective,
// and one `conditional_t` arm in `gated_mlp_launch_template.h`.  The kernel
// shell, the epilogue, the arguments, the FFI signature and the whole Python
// side are arch-free and do not move.

static_assert(!GatedMlpArch<120>::kHasTma && !GatedMlpArch<120>::kIsWarpSpecialized,
              "consumer Blackwell has neither; a kMinComputeCapability >= 90 test would "
              "claim both");

}  // namespace mlp
}  // namespace oasr
