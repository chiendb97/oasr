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
template <>
struct CutlassArch<86> {
    using Type = cutlass::arch::Sm86;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
};
template <>
struct CutlassArch<89> {
    using Type = cutlass::arch::Sm89;
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

// SM120 (GeForce Blackwell / RTX 50 series) note:
//   The CUTLASS 3.x SM120 CollectiveBuilder for OpClassTensorOp is restricted to
//   F8/F6/F4 MMA only — it does NOT support FP16/BF16 GEMM.  For FP16/BF16 we
//   instead drive SM120 through the CUTLASS 2.x path, whose SM80 tensor-op
//   specialisations (mma.sync.aligned.m16n8k16) are forward-compatible with
//   SM120 hardware.  So CutlassArch<120> intentionally aliases to Sm80.
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

/// Select SM90/SM120 epilogue and mainloop schedules based on kPingpong.
///
/// kPingpong=false → Cooperative warp-specialised WGMMA
///   - Mainloop: KernelTmaWarpSpecializedCooperative
///   - Epilogue: TmaWarpSpecializedCooperative
///
/// kPingpong=true → Pingpong warp-specialised WGMMA
///   - Mainloop: KernelTmaWarpSpecializedPingpong
///   - Epilogue: TmaWarpSpecialized  (pingpong uses the base TMA epilogue)
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
// GemmScheduleSelector — routes to the correct adapter per SM architecture
//
//   SM90  / SM120 → Sm90ScheduleAdapter<kPingpong>
//   SM100         → SMTypeAdapter<kSMs>
//==============================================================================

template <int kSmVersion, int kSMs, bool kPingpong>
struct GemmScheduleSelector {
    // Default: SM90 and SM120 — pingpong or cooperative
    using EpilogueSchedule = typename Sm90ScheduleAdapter<kPingpong>::EpilogueSchedule;
    using MainloopSchedule = typename Sm90ScheduleAdapter<kPingpong>::MainloopSchedule;
};

template <int kSMs, bool kPingpong>
struct GemmScheduleSelector<100, kSMs, kPingpong> {
    // SM100: 1-SM or 2-SM co-operative scheduling
    using EpilogueSchedule = typename SMTypeAdapter<kSMs>::EpilogueSchedule;
    using MainloopSchedule = typename SMTypeAdapter<kSMs>::MainloopSchedule;
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
    // TileShape: for SM100 with kSMs=2 the tile M dimension is doubled so that
    // both SMs together cover BM*2 rows.  For SM90/SM120 kSMs=1 so no scaling.
    using TileShape = cute::Shape<cute::Int<BM * kSMs>, cute::Int<BN>, cute::Int<BK>>;
    using ClusterShape = cute::Shape<cute::Int<CM>, cute::Int<CN>, cute::Int<1>>;
    using SmArch = typename CutlassArch<kSmVersion>::Type;

    using _ScheduleSelector = GemmScheduleSelector<kSmVersion, kSMs, kPingpong>;
    using EpilogueSchedule = typename _ScheduleSelector::EpilogueSchedule;
    using MainloopSchedule = typename _ScheduleSelector::MainloopSchedule;

    static constexpr int Stages = kStages;
};

}  // namespace gemm
}  // namespace oasr
