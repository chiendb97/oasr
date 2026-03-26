// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// FlashInfer-style GEMM configuration system.
//
// Provides compile-time template configs (GemmConfig, SmMMATraits, DefaultGemmConfig)
// and runtime config structs (CutlassGemmConfig) for CUTLASS 3.x SM90+/SM100+ dispatch.

#pragma once

#include <cassert>
#include <cstdint>
#include <ostream>
#include <sstream>
#include <string>

#ifdef __GNUC__
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wstrict-aliasing"
    #pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma.h>
#include <cutlass/gemm/gemm.h>

#ifdef __GNUC__
    #pragma GCC diagnostic pop
#endif

namespace oasr {
namespace gemm {

//==============================================================================
// GemmStatus -- operation status codes
//==============================================================================

enum class GemmStatus {
    SUCCESS = 0,
    INVALID_ARGUMENT,
    WORKSPACE_TOO_SMALL,
    NOT_SUPPORTED,
    INTERNAL_ERROR,
    CUDA_ERROR,
    CUTLASS_ERROR,
};

inline const char* getGemmStatusString(GemmStatus status) {
    switch (status) {
        case GemmStatus::SUCCESS: return "SUCCESS";
        case GemmStatus::INVALID_ARGUMENT: return "INVALID_ARGUMENT";
        case GemmStatus::WORKSPACE_TOO_SMALL: return "WORKSPACE_TOO_SMALL";
        case GemmStatus::NOT_SUPPORTED: return "NOT_SUPPORTED";
        case GemmStatus::INTERNAL_ERROR: return "INTERNAL_ERROR";
        case GemmStatus::CUDA_ERROR: return "CUDA_ERROR";
        case GemmStatus::CUTLASS_ERROR: return "CUTLASS_ERROR";
        default: return "UNKNOWN";
    }
}

//==============================================================================
// GemmConfig -- compile-time tile configuration (FlashInfer-style)
//==============================================================================

template <int TileM_, int TileN_, int TileK_,
          int WarpM_, int WarpN_, int WarpK_,
          int Stages_>
struct GemmConfig {
    static constexpr int TileM = TileM_;
    static constexpr int TileN = TileN_;
    static constexpr int TileK = TileK_;
    static constexpr int WarpM = WarpM_;
    static constexpr int WarpN = WarpN_;
    static constexpr int WarpK = WarpK_;
    static constexpr int Stages = Stages_;

    using ThreadBlock = cutlass::gemm::GemmShape<TileM_, TileN_, TileK_>;
    using Warps = cutlass::gemm::GemmShape<WarpM_, WarpN_, WarpK_>;
    static constexpr int NumStages = Stages_;
};

//==============================================================================
// SmMMATraits -- per-SM MMA traits (hardware-determined)
//==============================================================================

template <int SmVersion>
struct SmMMATraits;

template <>
struct SmMMATraits<70> {
    using MMAShape = cutlass::gemm::GemmShape<8, 8, 4>;
    using MMAOp = cutlass::arch::OpClassTensorOp;
    using SmArch = cutlass::arch::Sm70;
};

template <>
struct SmMMATraits<75> {
    using MMAShape = cutlass::gemm::GemmShape<16, 8, 8>;
    using MMAOp = cutlass::arch::OpClassTensorOp;
    using SmArch = cutlass::arch::Sm75;
};

template <>
struct SmMMATraits<80> {
    using MMAShape = cutlass::gemm::GemmShape<16, 8, 16>;
    using MMAOp = cutlass::arch::OpClassTensorOp;
    using SmArch = cutlass::arch::Sm80;
};

template <>
struct SmMMATraits<86> {
    using MMAShape = cutlass::gemm::GemmShape<16, 8, 16>;
    using MMAOp = cutlass::arch::OpClassTensorOp;
    using SmArch = cutlass::arch::Sm80;
};

template <>
struct SmMMATraits<89> {
    using MMAShape = cutlass::gemm::GemmShape<16, 8, 16>;
    using MMAOp = cutlass::arch::OpClassTensorOp;
    using SmArch = cutlass::arch::Sm80;
};

template <>
struct SmMMATraits<90> {
    using MMAShape = cutlass::gemm::GemmShape<16, 8, 16>;
    using MMAOp = cutlass::arch::OpClassTensorOp;
    using SmArch = cutlass::arch::Sm80;
};

template <>
struct SmMMATraits<100> {
    using MMAShape = cutlass::gemm::GemmShape<16, 8, 16>;
    using MMAOp = cutlass::arch::OpClassTensorOp;
    using SmArch = cutlass::arch::Sm80;
};

template <>
struct SmMMATraits<103> {
    using MMAShape = cutlass::gemm::GemmShape<16, 8, 16>;
    using MMAOp = cutlass::arch::OpClassTensorOp;
    using SmArch = cutlass::arch::Sm80;
};

template <>
struct SmMMATraits<120> {
    using MMAShape = cutlass::gemm::GemmShape<16, 8, 16>;
    using MMAOp = cutlass::arch::OpClassTensorOp;
    using SmArch = cutlass::arch::Sm80;
};

//==============================================================================
// Named config aliases
//==============================================================================

// SM70/75 -- fewer pipeline stages
using GemmConfig_128x128x32_s2 = GemmConfig<128, 128, 32, 64, 64, 32, 2>;

// SM80+ -- deeper pipelines
using GemmConfig_128x128x64_s3 = GemmConfig<128, 128, 64, 64, 64, 64, 3>;
using GemmConfig_128x128x64_s4 = GemmConfig<128, 128, 64, 64, 64, 64, 4>;
using GemmConfig_256x128x64_s3 = GemmConfig<256, 128, 64, 64, 64, 64, 3>;
using GemmConfig_64x64x64_s4   = GemmConfig<64, 64, 64, 32, 32, 64, 4>;

//==============================================================================
// DefaultGemmConfig -- per-SM default tile config
//==============================================================================

template <int SmVersion>
struct DefaultGemmConfig;

template <> struct DefaultGemmConfig<70>  { using type = GemmConfig_128x128x32_s2; };
template <> struct DefaultGemmConfig<75>  { using type = GemmConfig_128x128x32_s2; };
template <> struct DefaultGemmConfig<80>  { using type = GemmConfig_128x128x64_s3; };
template <> struct DefaultGemmConfig<86>  { using type = GemmConfig_128x128x64_s3; };
template <> struct DefaultGemmConfig<89>  { using type = GemmConfig_128x128x64_s3; };
template <> struct DefaultGemmConfig<90>  { using type = GemmConfig_128x128x64_s4; };
template <> struct DefaultGemmConfig<100> { using type = GemmConfig_128x128x64_s4; };
template <> struct DefaultGemmConfig<103> { using type = GemmConfig_128x128x64_s4; };
template <> struct DefaultGemmConfig<120> { using type = GemmConfig_128x128x64_s4; };

//==============================================================================
// JIT compile-time config selection via -D flags
//==============================================================================

#ifdef OASR_GEMM_TILE_M
using JitGemmConfig = GemmConfig<OASR_GEMM_TILE_M, OASR_GEMM_TILE_N, OASR_GEMM_TILE_K,
                                  OASR_GEMM_WARP_M, OASR_GEMM_WARP_N, OASR_GEMM_WARP_K,
                                  OASR_GEMM_STAGES>;
#endif

//==============================================================================
// Runtime config enums -- for CUTLASS 3.x SM90+/SM100+ dispatch (future use)
//==============================================================================

// Note: The shapes are in the format MxNxK. The K shape of the runtime config MUST match the K
// shape in the kernel layout details when doing weight only quantization.
enum class CutlassTileConfig {
    // Signals that we should run heuristics do choose a config
    Undefined,
  
    // Signals that we should run heuristics do choose a config
    ChooseWithHeuristic,
  
    // SiMT config
    CtaShape128x128x8_WarpShape64x64x8,
  
    // TensorCore configs CTA_N = 128, CTA_K = 64
    // Warp configs for M=16
    CtaShape16x128x64_WarpShape16x32x64,
    // Warp configs for M=32
    CtaShape32x128x64_WarpShape32x32x64,
  
    // Warp configs for M=64
    CtaShape64x128x64_WarpShape32x64x64,
    CtaShape64x64x128_WarpShape32x64x64,
    CtaShape64x128x64_WarpShape64x32x64,
  
    // Warp configs for M=128
    CtaShape128x64x64_WarpShape64x32x64,
    CtaShape128x128x64_WarpShape64x32x64,
    CtaShape128x128x64_WarpShape64x64x64,
    CtaShape128x128x64_WarpShape128x32x64,
    CtaShape128x256x64_WarpShape64x64x64,
  
    // Warp configs for M=256
    CtaShape256x128x64_WarpShape64x64x64,
  
    // TensorCore config CTA_N = 64, CTA_K = 128
    CtaShape128x64x128_WarpShape64x32x128,
  
    // TensorCore config CTA_N = 256, CTA_K = 64
    CtaShape16x256x64_WarpShape16x64x64,
  
    // TensorCore config CTA_N = 256, CTA_K = 128
    CtaShape16x256x128_WarpShape16x64x128
  
  };
  
  enum class SplitKStyle {
    NO_SPLIT_K,
    SPLIT_K_SERIAL,
    STREAM_K,  // Sm80+
               // SPLIT_K_PARALLEL // Not supported yet
  };
  
  enum class CutlassTileConfigSM90 {
    // Signals that we should run heuristics do choose a config
    Undefined,
  
    // Signals that we should run heuristics do choose a config
    ChooseWithHeuristic,
  
    // CTA configs for M=64
    CtaShape64x16x128B,
    CtaShape64x32x128B,
    CtaShape64x64x128B,
    CtaShape64x128x128B,
    CtaShape64x256x128B,
  
    // CTA configs for M=128
    CtaShape128x16x128B,
    CtaShape128x32x128B,
    CtaShape128x64x128B,
    CtaShape128x128x128B,
    CtaShape128x256x128B,
  
    // CTA configs for M=128
    CtaShape256x128x128B,
  };
  
  enum class CutlassTileConfigSM100 {
    // Signals that we should run heuristics do choose a config
    Undefined,
  
    // Signals that we should run heuristics do choose a config
    ChooseWithHeuristic,
  
    /*
     * Grouped GEMM
     */
    // M=64
    CtaShape64x32x128B,
    CtaShape64x64x128B,
    CtaShape64x128x128B,
    CtaShape64x256x128B,
  
    // M=128
    CtaShape128x8x256B,
    CtaShape128x16x128B,
    CtaShape128x32x128B,
    CtaShape128x64x128B,
    CtaShape128x128x128B,
    CtaShape128x256x128B,
    CtaShape128x128x256B,
    CtaShape128x256x256B,
  
    // M=256
    CtaShape256x64x128B,
    CtaShape256x128x128B,
    CtaShape256x256x128B,
  
    // SM103
    CtaShape128x128x768B,
    CtaShape128x192x768B,
    CtaShape128x256x768B,
  };
  
  enum class CutlassTileConfigSM120 {
    // Signals that we should run heuristics do choose a config
    Undefined,
  
    // Signals that we should run heuristics do choose a config
    ChooseWithHeuristic,
  
    CtaShape128x128x128B,
    CtaShape128x128x64B,
    CtaShape256x128x64B,
    CtaShape128x256x64B,
    CtaShape128x128x256B,
    CtaShape256x128x128B,
  };
  
  enum class MainloopScheduleType {
    AUTO,  // Automatically selects between pingpong and cooperative schedules on Hopper. On older
           // architectures, this defaults to the "legacy" main loop schedule.
    PINGPONG,
    COOPERATIVE,
    WARPSPECIALIZED
  };
  
  static auto get_mainloop_schedule_name(MainloopScheduleType schedule) {
    if (schedule == MainloopScheduleType::AUTO) {
      return "auto";
    } else if (schedule == MainloopScheduleType::PINGPONG) {
      return "pingpong";
    } else if (schedule == MainloopScheduleType::COOPERATIVE) {
      return "cooperative";
    } else if (schedule == MainloopScheduleType::WARPSPECIALIZED) {
      return "warpspecialized";
    }
    return "unknown schedule";
  }
  
  enum class EpilogueScheduleType {
    AUTO,  // Automatically chooses an epilogue schedule compatible with the selected main loop
           // schedule for Hopper. For architectures older than hopper, the epilogue is always
           // performed by the same thread block as the main loop.
  };
  
  enum class TileShape {
    TileShape_64x16x128,
    TileShape_64x32x128,
    TileShape_64x64x128,
    TileShape_64x128x128,
    TileShape_64x256x128,
    TileShape_64x512x128,
    TileShape_128x16x128,
    TileShape_128x32x128,
    TileShape_128x64x128,
    TileShape_128x128x128,
    TileShape_128x256x128,
    // SM103
    TileShape_128x128x768,
    TileShape_128x192x768,
    TileShape_128x256x768
  };
  
  template <TileShape Shape_MNK>
  constexpr auto get_tile_shape() {
    using namespace cute;
    if constexpr (Shape_MNK == TileShape::TileShape_64x16x128) {
      return cute::Shape<_64, _16, _128>{};
    } else if constexpr (Shape_MNK == TileShape::TileShape_64x32x128) {
      return cute::Shape<_64, _32, _128>{};
    } else if constexpr (Shape_MNK == TileShape::TileShape_64x64x128) {
      return cute::Shape<_64, _64, _128>{};
    } else if constexpr (Shape_MNK == TileShape::TileShape_64x128x128) {
      return cute::Shape<_64, _128, _128>{};
    } else if constexpr (Shape_MNK == TileShape::TileShape_64x256x128) {
      return cute::Shape<_64, _256, _128>{};
    } else if constexpr (Shape_MNK == TileShape::TileShape_64x512x128) {
      return cute::Shape<_64, _512, _128>{};
    } else if constexpr (Shape_MNK == TileShape::TileShape_128x16x128) {
      return cute::Shape<_128, _16, _128>{};
    } else if constexpr (Shape_MNK == TileShape::TileShape_128x32x128) {
      return cute::Shape<_128, _32, _128>{};
    } else if constexpr (Shape_MNK == TileShape::TileShape_128x64x128) {
      return cute::Shape<_128, _64, _128>{};
    } else if constexpr (Shape_MNK == TileShape::TileShape_128x128x128) {
      return cute::Shape<_128, _128, _128>{};
    } else if constexpr (Shape_MNK == TileShape::TileShape_128x256x128) {
      return cute::Shape<_128, _256, _128>{};
    } else if constexpr (Shape_MNK == TileShape::TileShape_128x128x768) {  // SM103
      return cute::Shape<_128, _128, _768>{};
    } else if constexpr (Shape_MNK == TileShape::TileShape_128x192x768) {  // SM103
      return cute::Shape<_128, _192, _768>{};
    } else if constexpr (Shape_MNK == TileShape::TileShape_128x256x768) {  // SM103
      return cute::Shape<_128, _256, _768>{};
    }
  }
  
  static auto get_tile_shape_name(TileShape Shape_MNK) {
    if (Shape_MNK == TileShape::TileShape_64x16x128) {
      return "64x16x128";
    } else if (Shape_MNK == TileShape::TileShape_64x32x128) {
      return "64x32x128";
    } else if (Shape_MNK == TileShape::TileShape_64x64x128) {
      return "64x64x128";
    } else if (Shape_MNK == TileShape::TileShape_64x128x128) {
      return "64x128x128";
    } else if (Shape_MNK == TileShape::TileShape_64x256x128) {
      return "64x256x128";
    } else if (Shape_MNK == TileShape::TileShape_64x512x128) {
      return "64x512x128";
    } else if (Shape_MNK == TileShape::TileShape_128x16x128) {
      return "128x16x128";
    } else if (Shape_MNK == TileShape::TileShape_128x32x128) {
      return "128x32x128";
    } else if (Shape_MNK == TileShape::TileShape_128x64x128) {
      return "128x64x128";
    } else if (Shape_MNK == TileShape::TileShape_128x128x128) {
      return "128x128x128";
    } else if (Shape_MNK == TileShape::TileShape_128x256x128) {
      return "128x256x128";
    } else if (Shape_MNK == TileShape::TileShape_128x128x768) {  // SM103
      return "128x128x768";
    } else if (Shape_MNK == TileShape::TileShape_128x192x768) {  // SM103
      return "128x192x768";
    } else if (Shape_MNK == TileShape::TileShape_128x256x768) {  // SM103
      return "128x256x768";
    }
    return "Unknown shape";
  }
  
  enum class ClusterShape {
    ClusterShape_1x1x1,
    ClusterShape_2x1x1,
    ClusterShape_1x2x1,
    ClusterShape_2x2x1,
    ClusterShape_1x4x1,
    ClusterShape_4x2x1,
    ClusterShape_2x4x1,
    ClusterShape_4x4x1,
    ClusterShape_1x8x1,
    ClusterShape_8x1x1,
    ClusterShape_4x1x1
  };
  
  static auto get_cluster_shape_name(ClusterShape Shape_MNK) {
    if (Shape_MNK == ClusterShape::ClusterShape_1x1x1) {
      return "1x1x1";
    } else if (Shape_MNK == ClusterShape::ClusterShape_2x1x1) {
      return "2x1x1";
    } else if (Shape_MNK == ClusterShape::ClusterShape_1x2x1) {
      return "1x2x1";
    } else if (Shape_MNK == ClusterShape::ClusterShape_2x2x1) {
      return "2x2x1";
    } else if (Shape_MNK == ClusterShape::ClusterShape_1x8x1) {
      return "1x8x1";
    } else if (Shape_MNK == ClusterShape::ClusterShape_8x1x1) {
      return "8x1x1";
    } else if (Shape_MNK == ClusterShape::ClusterShape_4x1x1) {
      return "4x1x1";
    }
    return "Unknown shape";
  }
  
  template <ClusterShape Shape_MNK>
  constexpr auto get_cluster_shape() {
    using namespace cute;
    if constexpr (Shape_MNK == ClusterShape::ClusterShape_1x1x1) {
      return cute::Shape<_1, _1, _1>{};
    } else if constexpr (Shape_MNK == ClusterShape::ClusterShape_2x1x1) {
      return cute::Shape<_2, _1, _1>{};
    } else if constexpr (Shape_MNK == ClusterShape::ClusterShape_1x2x1) {
      return cute::Shape<_1, _2, _1>{};
    } else if constexpr (Shape_MNK == ClusterShape::ClusterShape_2x2x1) {
      return cute::Shape<_2, _2, _1>{};
    } else if constexpr (Shape_MNK == ClusterShape::ClusterShape_1x8x1) {
      return cute::Shape<_1, _8, _1>{};
    } else if constexpr (Shape_MNK == ClusterShape::ClusterShape_8x1x1) {
      return cute::Shape<_8, _1, _1>{};
    } else if constexpr (Shape_MNK == ClusterShape::ClusterShape_4x1x1) {
      return cute::Shape<_4, _1, _1>{};
    }
  }
  
  struct CutlassGemmConfig {
    enum CandidateConfigTypeParam : int {
      NONE = 0,
      WEIGHT_ONLY = 1u << 0,
      SIMT_ONLY = 1u << 1,
      INT8_ONLY = 1u << 2,
      HOPPER = 1u << 3,
      BLACKWELL = 1u << 4,
      GROUPED_GEMM = 1u << 5,
      FP8_ONLY = 1u << 6,
      FP4_ONLY = 1u << 7
    };
  
    CutlassTileConfig tile_config_sm80 = CutlassTileConfig::ChooseWithHeuristic;
    SplitKStyle split_k_style = SplitKStyle::NO_SPLIT_K;
    int split_k_factor = -1;
    int stages = -1;
  
    // config options for sm90
    CutlassTileConfigSM90 tile_config_sm90 = CutlassTileConfigSM90::ChooseWithHeuristic;
    CutlassTileConfigSM100 tile_config_sm100 = CutlassTileConfigSM100::ChooseWithHeuristic;
    CutlassTileConfigSM120 tile_config_sm120 = CutlassTileConfigSM120::ChooseWithHeuristic;
    MainloopScheduleType mainloop_schedule = MainloopScheduleType::AUTO;
    EpilogueScheduleType epilogue_schedule = EpilogueScheduleType::AUTO;
    ClusterShape cluster_shape = ClusterShape::ClusterShape_1x1x1;
    bool enableCudaKernel = false;
    int sm_version = 80;  // Use 80 as a catch all for <90
    bool is_tma_warp_specialized = false;
    bool use_stream_k =
        false;  // SM120/SM121: false = DP scheduler (default), true = StreamK scheduler
  
    CutlassGemmConfig() = default;
  
    CutlassGemmConfig(CutlassTileConfig tile_config, SplitKStyle split_k_style, int split_k_factor,
                      int stages)
        : tile_config_sm80(tile_config),
          split_k_style(split_k_style),
          split_k_factor(split_k_factor),
          stages(stages),
          sm_version(80) {}
  
    CutlassGemmConfig(CutlassTileConfigSM90 tile_config_sm90, MainloopScheduleType mainloop_schedule,
                      EpilogueScheduleType epilogue_schedule, ClusterShape cluster_shape)
        : tile_config_sm90(tile_config_sm90),
          mainloop_schedule(mainloop_schedule),
          epilogue_schedule(epilogue_schedule),
          cluster_shape(cluster_shape),
          sm_version(90),
          is_tma_warp_specialized(true) {}
  
    CutlassGemmConfig(CutlassTileConfigSM100 tile_config_sm100,
                      MainloopScheduleType mainloop_schedule, EpilogueScheduleType epilogue_schedule,
                      ClusterShape cluster_shape)
        : tile_config_sm100(tile_config_sm100),
          mainloop_schedule(mainloop_schedule),
          epilogue_schedule(epilogue_schedule),
          cluster_shape(cluster_shape),
          sm_version(100),
          is_tma_warp_specialized(true) {}
  
    // SM120/SM121 constructor with optional StreamK scheduler
    // use_stream_k: false = DP scheduler (default), true = StreamK scheduler (auto heuristic)
    CutlassGemmConfig(CutlassTileConfigSM120 tile_config_sm120,
                      MainloopScheduleType mainloop_schedule, EpilogueScheduleType epilogue_schedule,
                      ClusterShape cluster_shape, bool use_stream_k = false)
        : tile_config_sm120(tile_config_sm120),
          mainloop_schedule(mainloop_schedule),
          epilogue_schedule(epilogue_schedule),
          cluster_shape(cluster_shape),
          sm_version(120),
          is_tma_warp_specialized(true),
          use_stream_k(use_stream_k) {}
  
    int getTileConfigAsInt() const {
      if (sm_version == 120 || sm_version == 121) return (int)tile_config_sm120;
      if (sm_version == 110) return (int)tile_config_sm100;
      if (sm_version >= 100) return (int)tile_config_sm100;
      if (sm_version == 90) return (int)tile_config_sm90;
      if (sm_version < 90) return (int)tile_config_sm80;
      assert(false && "Invalid SM version");
      return -1;
    }
  
    std::string toString() const {
      std::stringstream tactic;
      tactic << "Cutlass GEMM Tactic";
      if (is_tma_warp_specialized) {
        assert(sm_version >= 90 && "Invalid cutlass GEMM config");
        tactic << "\n\tstyle=TMA Warp Specialized"
               << "\n\tsm: " << sm_version << "\n\ttile shape ID: " << getTileConfigAsInt()
               << "\n\tcluster shape ID: " << (int)cluster_shape
               << "\n\tmainloop sched: " << (int)mainloop_schedule
               << "\n\tepi sched: " << (int)epilogue_schedule
               << "\n\tenable cuda kernel: " << (enableCudaKernel ? "true" : "false");
        // SM120/SM121 specific: StreamK scheduler option
        if (sm_version == 120 || sm_version == 121) {
          tactic << "\n\tscheduler: " << (use_stream_k ? "StreamK (auto heuristic)" : "DP (default)");
        }
      } else if (tile_config_sm80 != CutlassTileConfig::ChooseWithHeuristic) {
        assert(sm_version < 90 && "Invalid cutlass GEMM config");
        tactic << "\n\tstyle=compatible"
               << "\n\ttile shape ID: " << (int)tile_config_sm80 << "\n\tstages: " << (int)stages
               << "\n\tsplit k: " << (int)split_k_factor
               << "\n\tenable cuda kernel: " << (enableCudaKernel ? "true" : "false");
      } else if (enableCudaKernel) {
        tactic << "\n\tenable cuda kernel: " << (enableCudaKernel ? "true" : "false");
      } else {
        tactic << "\n\tundefined";
      }
      tactic << "\n";
      return tactic.str();
    }
  };
  
  inline std::ostream& operator<<(std::ostream& out, CutlassGemmConfig const& config) {
    // clang-format off
      if (config.is_tma_warp_specialized)
      {
          out << "tile_config_sm90_enum: " << config.getTileConfigAsInt()
              << ", mainloop_schedule_enum: " << int(config.mainloop_schedule)
              << ", epilogue_schedule_enum: " << int(config.epilogue_schedule)
              << ", cluster_shape_enum: " << int(config.cluster_shape)
              << ", enable_cuda_kernel: " << (config.enableCudaKernel ? "true" : "false");
      }
      else
      {
          out << "tile_config_enum: " << config.getTileConfigAsInt()
              << ", split_k_style_enum: " << int(config.split_k_style)
              << ", split_k_factor: " << config.split_k_factor
              << ", stages: " << config.stages
              << ", enable_cuda_kernel: " << (config.enableCudaKernel ? "true" : "false");
      }
    // clang-format on
    return out;
  }

}  // namespace gemm
}  // namespace oasr
