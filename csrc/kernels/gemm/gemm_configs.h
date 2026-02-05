// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// CUTLASS GEMM configuration types for optimized matrix operations
// Reference: https://github.com/flashinfer-ai/flashinfer/tree/main/include/flashinfer/gemm

#pragma once

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace oasr {
namespace kernels {
namespace gemm {

//==============================================================================
// Tile Configuration Enums
//==============================================================================

/**
 * @brief CTA tile configurations for SM80 (Ampere)
 * Format: CtaShape<M>x<N>x<K>_WarpShape<M>x<N>x<K>
 */
enum class CutlassTileConfigSM80 {
    Undefined,
    ChooseWithHeuristic,
    
    // SiMT configuration
    CtaShape128x128x8_WarpShape64x64x8,
    
    // TensorCore configurations - various M sizes
    // M=16
    CtaShape16x128x64_WarpShape16x32x64,
    // M=32
    CtaShape32x128x64_WarpShape32x32x64,
    // M=64
    CtaShape64x128x64_WarpShape32x64x64,
    CtaShape64x64x128_WarpShape32x64x64,
    CtaShape64x128x64_WarpShape64x32x64,
    // M=128
    CtaShape128x64x64_WarpShape64x32x64,
    CtaShape128x128x64_WarpShape64x32x64,
    CtaShape128x128x64_WarpShape64x64x64,
    CtaShape128x128x64_WarpShape128x32x64,
    CtaShape128x256x64_WarpShape64x64x64,
    // M=256
    CtaShape256x128x64_WarpShape64x64x64,
};

/**
 * @brief CTA tile configurations for SM90 (Hopper)
 * Uses TMA (Tensor Memory Accelerator) for efficient data movement
 */
enum class CutlassTileConfigSM90 {
    Undefined,
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
    
    // CTA configs for M=256
    CtaShape256x128x128B,
};

//==============================================================================
// Scheduling and Execution Configuration
//==============================================================================

/**
 * @brief Split-K parallelization strategies
 */
enum class SplitKStyle {
    NO_SPLIT_K,
    SPLIT_K_SERIAL,      // Serial reduction across split-K tiles
    STREAM_K,            // Stream-K scheduling (SM80+)
};

/**
 * @brief Mainloop scheduling strategies (SM90+)
 */
enum class MainloopScheduleType {
    AUTO,                // Automatically select between pingpong and cooperative
    PINGPONG,            // Ping-pong double buffering
    COOPERATIVE,         // Cooperative scheduling across warps
    WARPSPECIALIZED,     // Warp-specialized scheduling
};

/**
 * @brief Epilogue scheduling strategies (SM90+)
 */
enum class EpilogueScheduleType {
    AUTO,                // Automatically select compatible epilogue
};

//==============================================================================
// Cluster Configuration (SM90+)
//==============================================================================

/**
 * @brief Thread block cluster shapes for SM90+
 * Format: ClusterShape_<M>x<N>x<K>
 */
enum class ClusterShape {
    ClusterShape_1x1x1,
    ClusterShape_2x1x1,
    ClusterShape_1x2x1,
    ClusterShape_2x2x1,
    ClusterShape_1x4x1,
    ClusterShape_4x1x1,
    ClusterShape_4x2x1,
    ClusterShape_2x4x1,
    ClusterShape_4x4x1,
    ClusterShape_1x8x1,
    ClusterShape_8x1x1,
};

inline const char* getClusterShapeName(ClusterShape shape) {
    switch (shape) {
        case ClusterShape::ClusterShape_1x1x1: return "1x1x1";
        case ClusterShape::ClusterShape_2x1x1: return "2x1x1";
        case ClusterShape::ClusterShape_1x2x1: return "1x2x1";
        case ClusterShape::ClusterShape_2x2x1: return "2x2x1";
        case ClusterShape::ClusterShape_1x4x1: return "1x4x1";
        case ClusterShape::ClusterShape_4x1x1: return "4x1x1";
        case ClusterShape::ClusterShape_4x2x1: return "4x2x1";
        case ClusterShape::ClusterShape_2x4x1: return "2x4x1";
        case ClusterShape::ClusterShape_4x4x1: return "4x4x1";
        case ClusterShape::ClusterShape_1x8x1: return "1x8x1";
        case ClusterShape::ClusterShape_8x1x1: return "8x1x1";
        default: return "unknown";
    }
}

inline const char* getMainloopScheduleName(MainloopScheduleType schedule) {
    switch (schedule) {
        case MainloopScheduleType::AUTO: return "auto";
        case MainloopScheduleType::PINGPONG: return "pingpong";
        case MainloopScheduleType::COOPERATIVE: return "cooperative";
        case MainloopScheduleType::WARPSPECIALIZED: return "warpspecialized";
        default: return "unknown";
    }
}

//==============================================================================
// Layout and Transpose Configuration
//==============================================================================

/**
 * @brief Matrix layout types
 */
enum class MatrixLayout {
    RowMajor,
    ColumnMajor,
};

/**
 * @brief Transpose operation types
 */
enum class TransposeOp {
    NoTranspose,       // N
    Transpose,         // T
    ConjugateTranspose // C (for complex types)
};

//==============================================================================
// Epilogue Configuration
//==============================================================================

/**
 * @brief Epilogue fusion options
 */
enum class EpilogueFusion {
    NONE,              // No fusion: D = alpha * A @ B + beta * C
    BIAS,              // D = alpha * A @ B + bias
    BIAS_RELU,         // D = ReLU(alpha * A @ B + bias)
    BIAS_GELU,         // D = GELU(alpha * A @ B + bias)
    BIAS_SWISH,        // D = Swish(alpha * A @ B + bias)
    GATED,             // D = activation(A @ B1) * (A @ B2)
};

//==============================================================================
// GEMM Configuration Structure
//==============================================================================

/**
 * @brief Unified GEMM configuration for all supported architectures
 * 
 * Supports SM80 (Ampere) and SM90 (Hopper) with automatic configuration
 * selection based on problem size and hardware capabilities.
 */
struct GemmConfig {
    // Configuration type flags
    enum ConfigTypeFlags : int {
        NONE         = 0,
        WEIGHT_ONLY  = 1 << 0,
        SIMT_ONLY    = 1 << 1,
        INT8_ONLY    = 1 << 2,
        HOPPER       = 1 << 3,
        GROUPED_GEMM = 1 << 4,
        FP8_ONLY     = 1 << 5,
    };
    
    // SM80 configuration
    CutlassTileConfigSM80 tile_config_sm80 = CutlassTileConfigSM80::ChooseWithHeuristic;
    SplitKStyle split_k_style = SplitKStyle::NO_SPLIT_K;
    int split_k_factor = 1;
    int stages = -1;  // -1 for auto
    
    // SM90 configuration  
    CutlassTileConfigSM90 tile_config_sm90 = CutlassTileConfigSM90::ChooseWithHeuristic;
    MainloopScheduleType mainloop_schedule = MainloopScheduleType::AUTO;
    EpilogueScheduleType epilogue_schedule = EpilogueScheduleType::AUTO;
    ClusterShape cluster_shape = ClusterShape::ClusterShape_1x1x1;
    
    // General configuration
    int sm_version = 80;          // Target SM version
    bool is_tma_warp_specialized = false;  // Use TMA warp specialization (SM90+)
    bool enable_cuda_graph = false;        // Enable CUDA graph capture
    
    // Default constructor
    GemmConfig() = default;
    
    // SM80 constructor
    GemmConfig(CutlassTileConfigSM80 tile_config, SplitKStyle split_k,
               int split_k_factor, int stages)
        : tile_config_sm80(tile_config)
        , split_k_style(split_k)
        , split_k_factor(split_k_factor)
        , stages(stages)
        , sm_version(80) {}
    
    // SM90 constructor
    GemmConfig(CutlassTileConfigSM90 tile_config_sm90,
               MainloopScheduleType mainloop,
               EpilogueScheduleType epilogue,
               ClusterShape cluster)
        : tile_config_sm90(tile_config_sm90)
        , mainloop_schedule(mainloop)
        , epilogue_schedule(epilogue)
        , cluster_shape(cluster)
        , sm_version(90)
        , is_tma_warp_specialized(true) {}
    
    /**
     * @brief Get tile configuration as integer for comparison
     */
    int getTileConfigAsInt() const {
        if (sm_version >= 90) {
            return static_cast<int>(tile_config_sm90);
        }
        return static_cast<int>(tile_config_sm80);
    }
    
    /**
     * @brief Convert configuration to string for debugging
     */
    std::string toString() const {
        std::ostringstream oss;
        oss << "GemmConfig{";
        if (is_tma_warp_specialized) {
            oss << "sm=" << sm_version
                << ", tile_id=" << getTileConfigAsInt()
                << ", cluster=" << getClusterShapeName(cluster_shape)
                << ", mainloop=" << getMainloopScheduleName(mainloop_schedule);
        } else {
            oss << "sm=" << sm_version
                << ", tile_id=" << getTileConfigAsInt()
                << ", stages=" << stages
                << ", split_k=" << split_k_factor;
        }
        oss << "}";
        return oss.str();
    }
};

//==============================================================================
// Default Configuration Generators
//==============================================================================

/**
 * @brief Get candidate configurations for SM80 auto-tuning
 */
inline std::vector<GemmConfig> getDefaultSM80Configs() {
    std::vector<GemmConfig> configs;
    
    std::vector<CutlassTileConfigSM80> tiles = {
        CutlassTileConfigSM80::CtaShape128x128x64_WarpShape64x64x64,
        CutlassTileConfigSM80::CtaShape128x256x64_WarpShape64x64x64,
        CutlassTileConfigSM80::CtaShape256x128x64_WarpShape64x64x64,
        CutlassTileConfigSM80::CtaShape64x128x64_WarpShape32x64x64,
    };
    
    for (auto tile : tiles) {
        for (int stages : {2, 3, 4}) {
            configs.emplace_back(tile, SplitKStyle::NO_SPLIT_K, 1, stages);
        }
    }
    
    return configs;
}

/**
 * @brief Get candidate configurations for SM90 auto-tuning
 */
inline std::vector<GemmConfig> getDefaultSM90Configs() {
    std::vector<GemmConfig> configs;
    
    std::vector<CutlassTileConfigSM90> tiles = {
        CutlassTileConfigSM90::CtaShape64x64x128B,
        CutlassTileConfigSM90::CtaShape64x128x128B,
        CutlassTileConfigSM90::CtaShape128x64x128B,
        CutlassTileConfigSM90::CtaShape128x128x128B,
        CutlassTileConfigSM90::CtaShape128x256x128B,
    };
    
    std::vector<ClusterShape> clusters = {
        ClusterShape::ClusterShape_1x1x1,
        ClusterShape::ClusterShape_2x1x1,
        ClusterShape::ClusterShape_1x2x1,
        ClusterShape::ClusterShape_2x2x1,
    };
    
    for (auto tile : tiles) {
        for (auto cluster : clusters) {
            configs.emplace_back(tile,
                                 MainloopScheduleType::AUTO,
                                 EpilogueScheduleType::AUTO,
                                 cluster);
        }
    }
    
    return configs;
}

} // namespace gemm
} // namespace kernels
} // namespace oasr
