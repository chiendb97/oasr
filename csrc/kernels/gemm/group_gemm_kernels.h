// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Grouped GEMM kernel interfaces using NVIDIA CUTLASS
// Supports variable-sized problems with BF16 and FP16 precision

#pragma once

#include "group_gemm_params.h"
#include "gemm_configs.h"
#include "gemm_utils.h"
#include "common/types.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <vector>
#include <memory>

namespace oasr {
namespace kernels {
namespace gemm {

//==============================================================================
// Grouped GEMM Interface
//==============================================================================

/**
 * @brief Execute grouped GEMM operation
 * 
 * Computes GEMM for multiple problems with potentially different sizes:
 * D[i] = alpha * A[i] @ B[i] + beta * C[i]
 * 
 * This is particularly useful for:
 * - Variable-length sequence processing (e.g., in ASR)
 * - LoRA fine-tuning with different ranks
 * - Mixture of Experts (MoE) models
 * 
 * @param params Grouped GEMM parameters
 * @return Status code
 */
GemmStatus invokeGroupGemm(const GroupGemmParams& params);

/**
 * @brief Grouped GEMM with typed interface
 */
template <typename DTypeIn, typename DTypeOut = DTypeIn>
GemmStatus invokeGroupGemmTyped(
    void* workspace_float, size_t workspace_float_size,
    void* workspace_int, size_t workspace_int_size,
    const GemmProblemDesc* problems, int num_problems,
    const DTypeIn* const* A_array, const DTypeIn* const* B_array,
    DTypeOut* const* D_array,
    const int64_t* lda_array, const int64_t* ldb_array, const int64_t* ldd_array,
    bool weight_column_major,
    cudaStream_t stream);

/**
 * @brief Query required workspace sizes for grouped GEMM
 */
void queryGroupGemmWorkspaceSize(
    int num_problems,
    const GemmProblemDesc* problems,  // Can be null for conservative estimate
    DataType dtype,
    size_t& float_workspace_size,
    size_t& int_workspace_size,
    const GemmConfig& config = GemmConfig());

//==============================================================================
// Grouped GEMM Runner Class
//==============================================================================

/**
 * @brief Grouped GEMM runner class
 */
class GroupGemmRunner {
public:
    /**
     * @brief Construct grouped GEMM runner
     * @param dtype Data type (FP16 or BF16)
     * @param sm_version SM version (default: auto-detect)
     */
    explicit GroupGemmRunner(DataType dtype, int sm_version = -1);
    ~GroupGemmRunner();
    
    GroupGemmRunner(const GroupGemmRunner&) = delete;
    GroupGemmRunner& operator=(const GroupGemmRunner&) = delete;
    
    /**
     * @brief Get required workspace sizes
     */
    void getWorkspaceSize(int num_problems,
                          size_t& float_workspace_size,
                          size_t& int_workspace_size) const;
    
    /**
     * @brief Run grouped GEMM
     */
    GemmStatus run(const GroupGemmParams& params);
    
    /**
     * @brief Run grouped GEMM with explicit arrays
     */
    GemmStatus run(
        const GemmProblemDesc* problems, int num_problems,
        const void* const* A_array, const void* const* B_array,
        void* const* D_array,
        const int64_t* lda_array, const int64_t* ldb_array, const int64_t* ldd_array,
        void* workspace_float, size_t workspace_float_size,
        void* workspace_int, size_t workspace_int_size,
        bool weight_column_major,
        cudaStream_t stream);
    
    /**
     * @brief Get available configurations
     */
    std::vector<GemmConfig> getConfigs() const;
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

//==============================================================================
// Segment GEMM (for contiguous segments)
//==============================================================================

/**
 * @brief Parameters for segment GEMM
 * 
 * For processing contiguous segments in a single buffer.
 * Useful when data is packed into a single tensor with segment offsets.
 */
struct SegmentGemmParams {
    const void* A;              // [total_tokens, K] packed input
    const void* B;              // [K, N] shared weights (or per-segment)
    void* D;                    // [total_tokens, N] packed output
    
    const int* segment_offsets; // [num_segments + 1] cumsum of segment lengths
    int num_segments;
    int K;                      // Input dimension
    int N;                      // Output dimension
    
    DataType dtype = DataType::FP16;
    bool weight_column_major = false;
    
    void* workspace = nullptr;
    size_t workspace_size = 0;
    
    cudaStream_t stream = nullptr;
};

/**
 * @brief Execute segment GEMM
 * 
 * Efficiently handles variable-length segments packed into contiguous memory.
 */
GemmStatus invokeSegmentGemm(const SegmentGemmParams& params);

} // namespace gemm
} // namespace kernels
} // namespace oasr
