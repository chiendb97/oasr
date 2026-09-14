// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Cutlass GEMM configuration templates
// Cutlass 2.x SM75/SM80/SM86/SM89 dispatch.
// Cutlass 3.x SM90+/SM100+/SM120+ dispatch.
//
// SM90/SM120 scheduling follows Quack's GemmConfig convention:
//   kPingpong=true  → KernelTmaWarpSpecializedPingpong
//   kPingpong=false → KernelTmaWarpSpecializedCooperative
//
// SM100 scheduling uses kSMs (1 or 2) via SMTypeAdapter:
//   kSMs=1 → KernelTmaWarpSpecialized1SmSm100
//   kSMs=2 → KernelTmaWarpSpecialized2SmSm100
//
#pragma once

#include <cassert>

#ifdef __GNUC__
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wstrict-aliasing"
    #pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma.h>
#include <cutlass/gemm/gemm.h>

#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/fusion/operations.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>

#ifdef __GNUC__
    #pragma GCC diagnostic pop
#endif

namespace oasr {
namespace gemm {

template <int SmVersion>
struct CutlassArch;

template <>
struct CutlassArch<75> {
    using Type = cutlass::arch::Sm75;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
};
template <>
struct CutlassArch<80> {
    using Type = cutlass::arch::Sm80;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
};
// Ampere consumer / workstation (A10, A10G, A16, A40, RTX 30-series).
template <>
struct CutlassArch<86> {
    using Type = cutlass::arch::Sm80;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
};
// Ada Lovelace (L4, L40S, RTX 4090, RTX Ada).
template <>
struct CutlassArch<89> {
    using Type = cutlass::arch::Sm80;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
};

template <>
struct CutlassArch<90> {
    using Type = cutlass::arch::Sm90;
};

template <>
struct CutlassArch<100> {
    using Type = cutlass::arch::Sm100;
};

// GeForce Blackwell (RTX 50 series, RTX PRO 6000).  SM120 is here for a second
// reason on top of the one above: its CUTLASS 3.x CollectiveBuilder for
// OpClassTensorOp is restricted to F8/F6/F4 MMA and does not accept FP16/BF16
// at all, so SM120 is routed down the 2.x lane in the first place — see
// `_get_sm120_configs` and `default_config_for_sm` in `oasr/jit/gemm.py`.
template <>
struct CutlassArch<120> {
    using Type = cutlass::arch::Sm80;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
};

template <int BM, int BN, int BK, int WM, int WN, int WK, int kStages, int kSmVersion>
struct CutlassGemmConfig {
    using ThreadblockShape = cutlass::gemm::GemmShape<BM, BN, BK>;
    using WarpShape = cutlass::gemm::GemmShape<WM, WN, WK>;

    using SmArch = typename CutlassArch<kSmVersion>::Type;
    using InstructionShape = typename CutlassArch<kSmVersion>::InstructionShape;

    static constexpr int Stages = kStages;
};

//==============================================================================
// SM90 / SM120 schedule adapter — Quack-style pingpong / cooperative
//==============================================================================

/// Select cooperative or ping-pong mainloop and epilogue schedules.
template <bool kPingpong>
struct Sm90ScheduleAdapter {
    // kPingpong=false: cooperative schedule
    using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
    using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
};

template <>
struct Sm90ScheduleAdapter<true> {
    // kPingpong=true: pingpong schedule
    // Note: pingpong uses TmaWarpSpecialized (base) for the epilogue, not a
    // dedicated pingpong epilogue type — this matches CUTLASS 3.x examples.
    using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;
    using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
};

//==============================================================================
// SM100 schedule adapter — 1-SM / 2-SM co-operative
//==============================================================================

template <int kSMs>
struct SMTypeAdapter;

template <>
struct SMTypeAdapter<1> {
    static constexpr int Scale = 1;
    using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized1Sm;
    using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized1SmSm100;
};

template <>
struct SMTypeAdapter<2> {
    static constexpr int Scale = 2;
    using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized2Sm;
    using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized2SmSm100;
};

//==============================================================================
// Grouped-GEMM (ptr-array) schedule adapter
//
// A grouped GEMM takes an *array* of problems, so it needs the ptr-array
// schedules rather than the dense ones above.  These are a parallel family, not
// a variation: `KernelPtrArrayTmaWarpSpecializedCooperative` has no SM100
// specialization at all, and asking the SM100 builder for it fails at
// `CollectiveBuilder ... has no member "CollectiveOp"` rather than degrading.
//==============================================================================

template <int kSMs>
struct GroupSMTypeAdapter;

template <>
struct GroupSMTypeAdapter<1> {
    using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
    using MainloopSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmSm100;
};

template <>
struct GroupSMTypeAdapter<2> {
    using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm;
    using MainloopSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmSm100;
};

// Select the schedule adapter for the compiled SM family.

template <int kSmVersion, int kSMs, bool kPingpong>
struct GemmScheduleSelector {
    // Default: SM90 and SM120 — pingpong or cooperative
    using EpilogueSchedule = typename Sm90ScheduleAdapter<kPingpong>::EpilogueSchedule;
    using MainloopSchedule = typename Sm90ScheduleAdapter<kPingpong>::MainloopSchedule;

    // SM90 / SM120 grouped GEMM: one cooperative ptr-array schedule, no 1-/2-SM
    // split to make.  Ping-pong has no ptr-array form, so a pingpong config
    // still groups cooperatively.
    using GroupEpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
    using GroupMainloopSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
};

template <int kSMs, bool kPingpong>
struct GemmScheduleSelector<100, kSMs, kPingpong> {
    // SM100: 1-SM or 2-SM co-operative scheduling
    using EpilogueSchedule = typename SMTypeAdapter<kSMs>::EpilogueSchedule;
    using MainloopSchedule = typename SMTypeAdapter<kSMs>::MainloopSchedule;

    using GroupEpilogueSchedule = typename GroupSMTypeAdapter<kSMs>::EpilogueSchedule;
    using GroupMainloopSchedule = typename GroupSMTypeAdapter<kSMs>::MainloopSchedule;
};

//==============================================================================
// CutlassGemmConfigSm90 — CUTLASS 3.x GEMM configuration for SM90+
//
// Template parameters (Quack-aligned):
//   BM, BN, BK         — tile dimensions
//   CM, CN             — cluster dimensions (CK is always 1)
//   kSMs               — number of co-operative SMs (1 or 2, SM100 only)
//   kStages            — pipeline stages
//   kSmVersion         — SM version: 90, 100, or 120
//   kPingpong          — True → Pingpong schedule (SM90/SM120 only)
//==============================================================================

template <int BM, int BN, int BK, int CM, int CN, int kSMs, int kStages, int kSmVersion,
          bool kPingpong = false>
struct CutlassGemmConfigSm90 {
    // BM is the MMA tile M as the builder sees it, never scaled by kSMs.
    //
    // It used to be `BM * kSMs`, on the belief that a 2-SM SM100 atom needs the
    // tile doubled so both SMs together cover BM*2 rows.  They do cover 2x the
    // rows, but CUTLASS wants the *combined* extent: its own SM100 dense-GEMM
    // tests pass `MmaTileShape = Shape<_256,_128,_64>` alongside
    // `ClusterShape = Shape<_2,_1,_1>`, and the doubling turned OASR's 256-row
    // configs into 512 and tripped
    // `static_assert(M == 128 || M == 256, "Invalid TileShape_M.")`
    // (`gemm/collective/builders/sm100_common.inl:375`).  That is 13 of the 37
    // emitted sm_100 variants, and one unbuildable variant fails the whole JIT
    // module -- so no GEMM, BMM or grouped GEMM built on B200 at all.
    //
    // kSMs stays a parameter: unlike the conv path, which lets
    // `KernelScheduleAuto` pick the atom from the cluster, the GEMM path still
    // selects `KernelTmaWarpSpecialized{1,2}SmSm100` through `SMTypeAdapter`.
    // It selects the *schedule*; it does not scale the *tile*.
    //
    // SM90 and SM120 always pass kSMs=1, so this is an SM100-only correction and
    // their generated kernels are byte-identical either way.
    using TileShape = cute::Shape<cute::Int<BM>, cute::Int<BN>, cute::Int<BK>>;
    using ClusterShape = cute::Shape<cute::Int<CM>, cute::Int<CN>, cute::Int<1>>;
    using SmArch = typename CutlassArch<kSmVersion>::Type;

    using _ScheduleSelector = GemmScheduleSelector<kSmVersion, kSMs, kPingpong>;
    using EpilogueSchedule = typename _ScheduleSelector::EpilogueSchedule;
    using MainloopSchedule = typename _ScheduleSelector::MainloopSchedule;

    // The grouped (ptr-array) counterparts, for CutlassGroupGemmKernelSm90.
    using GroupEpilogueSchedule = typename _ScheduleSelector::GroupEpilogueSchedule;
    using GroupMainloopSchedule = typename _ScheduleSelector::GroupMainloopSchedule;

    static constexpr int Stages = kStages;
};

}  // namespace gemm
}  // namespace oasr
