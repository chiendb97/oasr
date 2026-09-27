// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// The Ampere-class CuTe toolkit the C++ CUTLASS/CuTe kernel families share:
// attention (`include/oasr/attention/`), the gated MLP (`include/oasr/mlp/`)
// and the recurrent step (`include/oasr/recurrent/recurrent_step_*.h`).
//
// "Ampere-class" means the instruction set, not the part: `mma.sync`
// m16n8k16 with FP32 accumulate, `cp.async` into a swizzled shared-memory
// ring, and `ldmatrix`.  That is what sm_80, sm_86, sm_89 *and* sm_120 all run
// for FP16/BF16 -- GeForce Blackwell has no FP16 tcgen05 path, so its own
// CUTLASS GEMM uses the same warp-level atom -- which is why one
// `cutlass::arch::Sm80` tag serves all four.
//
// ---------------------------------------------------------------------------
// What lives here, and what does not
// ---------------------------------------------------------------------------
//
// Here: facts about those instructions and the idioms that follow from them --
// the per-architecture traits every family derives from, the swizzled
// shared-memory atom, the accumulator re-views, the packed down-conversion,
// the predicated tiled copies and the two warp-level gemm shapes.  Each family
// used to carry its own copy of these, and the copies had already drifted in
// both directions that matter: the gated MLP found that a branching
// `if (pred) copy else clear` around an asynchronous copy costs ptxas three
// dead `LDS` and a `BSSY`/`BSYNC` per copy (0.89x -> 1.02x once removed), and
// that the attention family's `convert_type` returned a view of a dead local
// -- and neither fix had reached the family it was not found in.  One copy is
// what makes a fix land everywhere.
//
// Not here: anything that is a property of one *algorithm* -- the online
// softmax, the masks, the bias read, the paged gather, the dual-B gemm, the
// recurrent state transition.  Those stay in their family directories, which
// remain the unit of kernel independence for everything that is actually
// theirs.
//
// The CuTe-free half of this contract -- the swizzle-row rule, the pass
// geometry, the occupancy estimate -- is `tile_rules.h`, so the capability
// oracles can build without CuTe.

#pragma once

// clang-format off
//
// Order-sensitive; must not be sorted.  `cute/tensor.hpp` is CuTe's entry
// point and has to come before any individual `cute/atom/*` or `cute/arch/*`
// header -- the atoms' free functions are declared against declarations it
// pulls in, and including them bare is a parse error.  `.clang-format` sets
// `IncludeBlocks: Regroup` with `SortIncludes: true`, which would sort
// `cute/arch/copy_sm75.hpp` above it.
#include <cute/tensor.hpp>

#include <oasr/common/arch_dispatch.h>
#include <oasr/common/arch_facts.h>
#include <oasr/common/tile_rules.h>

#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma_sm80.h>
#include <cutlass/cutlass.h>
#include <cutlass/device_kernel.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include <cuda_runtime.h>

#include <type_traits>
// clang-format on

namespace oasr {
namespace cute_sm80 {

using namespace cute;

// ---------------------------------------------------------------------------
// Per-architecture traits
// ---------------------------------------------------------------------------

/*! \brief What an Ampere-class architecture is, for every CuTe family.
 *
 * A family declares the architectures it serves by specialising its own
 * `XxxArch<SM>` from this -- adding only its tuning ceilings -- so the set of
 * served parts stays a per-family decision while the instruction selection is
 * stated once.
 *
 * `Tag` selects *instructions*; `kSmemBudgetBytes`, `kMaxThreadsPerSm` and
 * `kIsSm86Or89` select *tuning*.  Keeping those two jobs apart is why sm_86,
 * sm_89 and sm_120 route through `cutlass::arch::Sm80` -- the tag they share
 * is the tag whose instructions they run -- while still budgeting their own
 * 99 KB and their own 1536 warp slots.  FlashAttention makes the same split
 * (`hopper/flash_fwd_launch_template.h:36`).
 *
 * \warning Never branch on `Tag::kMinComputeCapability >= 90`.  `Sm120`'s is
 *   120, so that test is *true* on consumer Blackwell, which has neither TMA
 *   nor warp specialization -- FlashAttention's epilogue selects its TMA store
 *   exactly that way (`hopper/epilogue_fwd.hpp:37`).  Branch on `kHasTma` /
 *   `kIsWarpSpecialized`, which say what they mean.
 */
template <int SmVersion>
struct ArchAmpere {
    using Tag = cutlass::arch::Sm80;

    static constexpr int kSmVersion = SmVersion;
    static constexpr int kSmemCapacityBytes = smemCapacityForSm(SmVersion);
    static constexpr int kSmemBudgetBytes = smemBudgetForSm(SmVersion);
    static constexpr int kMaxThreadsPerSm = maxThreadsPerSmForSm(SmVersion);

    //! Register file and L2 differ enough on the consumer parts to move a
    //! tile choice; they do not change which instructions are legal.
    static constexpr bool kIsSm86Or89 = (SmVersion == 86 || SmVersion == 89);

    static constexpr bool kHasCpAsync = true;
    static constexpr bool kHasTma = false;
    static constexpr bool kIsWarpSpecialized = false;

    template <class Element>
    using MmaAtom = std::conditional_t<std::is_same_v<Element, cutlass::half_t>,
                                       cute::MMA_Atom<cute::SM80_16x8x16_F32F16F16F32_TN>,
                                       cute::MMA_Atom<cute::SM80_16x8x16_F32BF16BF16F32_TN>>;

    //! gmem -> smem for every mainloop operand.  ZFILL, not the plain
    //! cp.async: a predicated-off copy writes **zeros** rather than leaving
    //! stale shared memory.  Zero is the identity for a dot product, so a
    //! residue -- rows past an extent, a K tail -- is *correct* rather than
    //! merely safe, and a length-bounded load needs no caller-side
    //! precondition on what lies past the length.
    template <class Element>
    using GmemCopyAtom =
        cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>, Element>;

    //! smem -> gmem for the output, one 128-bit vector per thread.
    template <class Element>
    using GmemCopyAtomO =
        cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>, Element>;

    //! smem -> registers for a K-major operand.  Both A and B of the TN
    //! `mma.sync` are K-major in every family here, so they share it.
    template <class Element>
    using SmemCopyAtom = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>;

    //! smem -> registers for an operand consumed transposed; `ldmatrix.trans`
    //! does the transpose for free.
    template <class Element>
    using SmemCopyAtomTransposed = cute::Copy_Atom<cute::SM75_U16x8_LDSM_T, Element>;
};

static_assert(!ArchAmpere<120>::kHasTma && !ArchAmpere<120>::kIsWarpSpecialized,
              "consumer Blackwell has neither; a kMinComputeCapability >= 90 test would "
              "claim both");

// ---------------------------------------------------------------------------
// Shared-memory layout atom
// ---------------------------------------------------------------------------

/*! \brief The swizzled `(8, row)` atom a tile with contiguous extent \p kExtent
 *  is built from.
 *
 * The swizzle makes the `ldmatrix` that follows bank-conflict free; the row
 * width is `smemSwizzleRowWidth` (`tile_rules.h`).  A family that aliases two
 * buffers over the same bytes must build both from the same atom, or the
 * aliasing is unsound.
 */
template <class Element, int kExtent>
struct SmemLayoutAtomSwizzled {
    static constexpr int kRowWidth = smemSwizzleRowWidth(kExtent, int(sizeof(Element)));
    static constexpr int kSwizzle = smemSwizzleBits(kRowWidth);
    using type =
        decltype(composition(Swizzle<kSwizzle, 3, 3>{},
                             Layout<Shape<_8, Int<kRowWidth>>, Stride<Int<kRowWidth>, _1>>{}));
};

// ---------------------------------------------------------------------------
// Accumulator re-views
// ---------------------------------------------------------------------------

/*! \brief Re-view an m16n8k16 MMA-C accumulator as `(row, col)`.
 *
 * `acc` comes out of `partition_fragment_C` shaped `((2, 2), MMA_M, MMA_N)`:
 * each thread holds a 2x2 block whose inner mode (stride 1) is a *column*
 * pair and whose outer mode is a *row* pair.  The row axis is therefore
 * `(2, MMA_M)` and everything else is column.  Every per-row quantity (a
 * softmax statistic, a mask row, a store predicate) and every per-column one
 * (a bias) indexes through this view.
 */
template <class Layout>
CUTLASS_DEVICE auto convert_layout_acc_rowcol(Layout acc_layout) {
    static_assert(decltype(rank(acc_layout))::value == 3);
    // FA3's version also carries a `V` mode and asserts rank<0> == 3; that is
    // the Hopper fragment, not this one.
    static_assert(decltype(rank<0>(acc_layout))::value == 2,
                  "SM80's MMA-C fragment is ((2, 2), MMA_M, MMA_N)");
    return make_layout(make_layout(get<0, 1>(acc_layout), get<1>(acc_layout)),
                       make_layout(get<0, 0>(acc_layout), get<2>(acc_layout)));
}

/*! \brief Re-view an m16n8k16 MMA-C accumulator as an MMA-A fragment.
 *
 * A gemm's output consumed as the next gemm's A operand without going through
 * shared memory (attention's `P`).  `((2, 2), MMA_M, MMA_N)` ->
 * `(((2, 2), 2), MMA_M, MMA_N / 2)`: the A operand of m16n8k16 is twice as
 * wide in k as one C tile is in n, so two neighbouring n tiles fold into one k
 * tile.  A pure re-view.
 */
template <class MMA, class Layout>
CUTLASS_DEVICE auto convert_layout_acc_Aregs(Layout acc_layout) {
    using X = Underscore;
    static_assert(decltype(rank(acc_layout))::value == 3);
    static_assert(decltype(rank<0>(acc_layout))::value == 2);
    auto l = logical_divide(acc_layout, Shape<X, X, _2>{});
    return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
}

/*! \brief Round-to-nearest convert an FP32 fragment down to the served dtype.
 *
 * Returns an **owning** register tensor.  FlashAttention's version returns a
 * view over a local `cutlass::Array` that has already gone out of scope; it
 * survives because everything inlines and the array stays in registers, but
 * it is a dangling reference on paper.
 *
 * The packed `NumericArrayConverter` rather than a per-element `static_cast`:
 * it emits `cvt.rn.f16x2.f32`, halving the instruction count for the same
 * rounding.
 */
template <typename To_type, typename Engine, typename Layout>
CUTLASS_DEVICE auto convert_type(Tensor<Engine, Layout> const& tensor) {
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    Tensor out = make_tensor<To_type>(tensor.layout());
    cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    *reinterpret_cast<cutlass::Array<To_type, numel>*>(out.data()) =
        convert_op(*reinterpret_cast<cutlass::Array<From_type, numel> const*>(tensor.data()));
    return out;
}

// ---------------------------------------------------------------------------
// Predicated tiled copies
// ---------------------------------------------------------------------------
//
// Both take a `(CPY, CPY_M, CPY_K)` partition pair and two predicates, one per
// partition row `m` and one per partition column `k`, as callables.  A family
// passes whichever form it has -- a precomputed `bool` per row, or a limit
// compared against a **thread-0** identity tensor, whose entries are
// compile-time constants, so the comparison carries no per-thread arithmetic
// once this thread's own offset has been folded into the limit.  That trick is
// FlashAttention's.  Everything inlines, so a callable costs nothing over a
// hand-written loop.

/*! \brief Branch-free gmem->smem copy that **zero-fills** what it skips.
 *
 * Both predicates are folded into the ZFILL cp.async's own `src_size` operand
 * (`Copy_Atom::with(bool)`) instead of into control flow.  A predicated-off
 * copy writes zeros, which is the identity for the dot product the tile feeds,
 * so a residue on either axis is correct rather than merely safe.
 *
 * Worth a helper of its own because the branching form is expensive in a way
 * that is invisible in the source.  Written `if (pred) copy(...) else
 * clear(...)`, ptxas has to order a synchronous `STS` against an asynchronous
 * `LDGSTS` to the same shared address, and it does that by bracketing every
 * copy in `BSSY`/`BSYNC` and padding it with three dead `@!PT LDS RZ, [RZ]`.
 * Measured on the gated MLP's 64x64x32 tile: the K loop's load section went
 * from ~60 instructions to ~12, the LSU pipe from 2.34M to 1.2M instructions,
 * and the kernel from 0.89x of its CuTeDSL twin to ahead of it.  Nothing about
 * the C++ says any of that; the SASS does.
 *
 * The source address of a predicated-off copy is still formed, so it must be
 * *computable* (no out-of-range page-table read to produce it), but it is
 * never dereferenced: `src_size == 0` reads nothing.
 */
template <class CopyAtom, class TV, class Tiler, class TensorS, class TensorD, class RowPred,
          class ColPred>
CUTLASS_DEVICE void copy_zfill(TiledCopy<CopyAtom, TV, Tiler> const& tiled_copy,
                               TensorS const& S, TensorD&& D, RowPred const& row_ok,
                               ColPred const& col_ok) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));
    auto copy_atom = static_cast<CopyAtom const&>(tiled_copy);
    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < size<1>(S); ++m) {
        bool const r = row_ok(m);
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < size<2>(S); ++k) {
            bool const ok = r && col_ok(k);
            cute::copy(copy_atom.with(ok), S(_, m, k), D(_, m, k));
        }
    }
}

/*! \brief Tiled copy that **skips** what it may not write.
 *
 * For stores.  Rows past an extent belong to no output row -- under varlen
 * they belong to the *next* segment -- and columns past one belong to the next
 * row of a row-major buffer, so writing either would corrupt real output
 * rather than merely waste a store.  There is deliberately no "clear the
 * skipped part" mode: a load that needs zeros gets them from `copy_zfill`, not
 * from an `STS` that ptxas would have to order against the asynchronous copies
 * around it.
 */
template <class CopyAtom, class TV, class Tiler, class TensorS, class TensorD, class RowPred,
          class ColPred>
CUTLASS_DEVICE void copy_if(TiledCopy<CopyAtom, TV, Tiler> const& tiled_copy, TensorS const& S,
                            TensorD&& D, RowPred const& row_ok, ColPred const& col_ok) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));
    auto copy_atom = static_cast<CopyAtom const&>(tiled_copy);
    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < size<1>(S); ++m) {
        if (row_ok(m)) {
            CUTLASS_PRAGMA_UNROLL
            for (int k = 0; k < size<2>(S); ++k) {
                if (col_ok(k)) {
                    cute::copy(copy_atom, S(_, m, k), D(_, m, k));
                }
            }
        }
    }
}

//! A predicate that is always true, for an axis a caller knows is in range.
struct AlwaysTrue {
    CUTLASS_DEVICE bool operator()(int) const { return true; }
};

// ---------------------------------------------------------------------------
// Warp-level gemm shapes
// ---------------------------------------------------------------------------

/*! \brief `acc += A @ B` over one K tile, both operands read from shared memory.
 *
 * The `ldmatrix` for k-step 0 is issued before the loop and k-step `i + 1`
 * before step `i`'s MMA, so the shared-memory read for the next step overlaps
 * this step's tensor-core work.  The prefetch is guarded by `i < size - 1`
 * rather than wrapping with `(i + 1) % K`, so the last k-step does not issue a
 * whole extra round of `ldmatrix` whose result is discarded (the CuTeDSL
 * lanes' `gemm_with_smem_prefetch` does).
 *
 * \tparam A_in_regs  A was read into registers already (attention's
 *   `Q_in_regs`); skip its per-step `ldmatrix` entirely.
 * \param fn fires once, after the first k-step's `ldmatrix` has been issued.
 *   Mainloops use it to launch the next stage's cp.async, so that copy
 *   overlaps the MMA chain instead of sitting in front of it.
 */
template <bool A_in_regs = false, typename Tensor0, typename Tensor1, typename Tensor2,
          typename Tensor3, typename Tensor4, typename TiledMma, typename TiledCopyA,
          typename TiledCopyB, typename ThrCopyA, typename ThrCopyB, typename Hook>
CUTLASS_DEVICE void gemm_sm80(Tensor0& acc, Tensor1& tCrA, Tensor2& tCrB, Tensor3 const& tCsA,
                              Tensor4 const& tCsB, TiledMma tiled_mma,
                              TiledCopyA const& smem_tiled_copy_A,
                              TiledCopyB const& smem_tiled_copy_B,
                              ThrCopyA const& smem_thr_copy_A,
                              ThrCopyB const& smem_thr_copy_B, Hook fn) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));
    if constexpr (!A_in_regs) {
        cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{}));
    }
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            if constexpr (!A_in_regs) {
                cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1));
            }
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
        }
        if (i == 0) {
            fn();
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

/*! \brief `acc += A @ B` with A already in registers and B from shared memory.
 *
 * Attention's PV gemm: `A` is `P`, which the softmax has just produced in
 * registers, so only `B` (the transposed V tile) is read through `ldmatrix`.
 */
template <typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
          typename TiledMma, typename TiledCopy, typename ThrCopy>
CUTLASS_DEVICE void gemm_rs_sm80(Tensor0& acc, Tensor1 const& tCrA, Tensor2& tCrB,
                                 Tensor3 const& tCsB, TiledMma tiled_mma,
                                 TiledCopy const& smem_tiled_copy_B,
                                 ThrCopy const& smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

// ---------------------------------------------------------------------------
// Launch
// ---------------------------------------------------------------------------

/*! \brief Launch one kernel shell through `cutlass::device_kernel`.
 *
 * \tparam Kernel a shell exposing `Params`, `SharedStorageSize`,
 *   `get_grid_shape(Params)` and `get_block_shape()` -- the contract every
 *   family's `*_kernel.h` meets.
 *
 * The shared-memory size is the kernel's own `sizeof(SharedStorage)`, which
 * each launcher has already `static_assert`ed against its architecture's
 * table.  The table says what the architecture offers; the check below asks
 * what *this* device grants, so a part that grants less refuses here instead
 * of failing inside `cudaFuncSetAttribute` with an empty message.
 */
template <class Kernel>
cudaError_t launch_kernel(typename Kernel::Params const& params, cudaStream_t stream) {
    dim3 const grid = Kernel::get_grid_shape(params);
    dim3 const block = Kernel::get_block_shape();
    int const smem_size = Kernel::SharedStorageSize;

    auto kernel = cutlass::device_kernel<Kernel>;
    if (smem_size > oasr::getDeviceMaxSharedMemoryOptin()) {
        return cudaErrorInvalidValue;
    }
    cudaError_t status = oasr::optInSharedMemory(kernel, size_t(smem_size));
    if (status != cudaSuccess) {
        return status;
    }
    kernel<<<grid, block, smem_size, stream>>>(params);
    return cudaGetLastError();
}

}  // namespace cute_sm80
}  // namespace oasr
