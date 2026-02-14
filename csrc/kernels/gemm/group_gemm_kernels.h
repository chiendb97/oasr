// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Grouped GEMM kernel interfaces using NVIDIA CUTLASS
// Supports variable-sized problems with BF16 and FP16 precision

#pragma once

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
// Grouped GEMM: Problem descriptor
//==============================================================================

/**
 * @brief Problem specification for a single GEMM in a group (M, N, K)
 */
struct GemmProblemDesc {
    int M;
    int N;
    int K;

    GemmProblemDesc() : M(0), N(0), K(0) {}
    GemmProblemDesc(int m, int n, int k) : M(m), N(n), K(k) {}
};

//==============================================================================
// Grouped GEMM Interface
//==============================================================================

/**
 * @brief Execute grouped GEMM operation
 *
 * Computes: D[i] = alpha * A[i] @ B[i] + beta * D[i] for variable-sized problems
 *
 * @param problems [num_problems] problem dimensions on device (GemmProblemDesc*)
 * @param num_problems Number of problems
 * @param A_array [num_problems] array of A pointers
 * @param B_array [num_problems] array of B pointers
 * @param D_array [num_problems] array of D pointers
 * @param lda_array [num_problems] leading dimensions for A
 * @param ldb_array [num_problems] leading dimensions for B
 * @param ldd_array [num_problems] leading dimensions for D
 * @param dtype Data type (FP16 or BF16)
 * @param workspace_float Workspace buffer
 * @param workspace_float_size Size of float workspace
 * @param stream CUDA stream
 * @return Status code
 */
GemmStatus invokeGroupGemm(const GemmProblemDesc* problems, int num_problems,
                           const void* const* A_array, const void* const* B_array,
                           void* const* D_array,
                           const int64_t* lda_array, const int64_t* ldb_array,
                           const int64_t* ldd_array,
                           DataType dtype,
                           void* workspace_float, size_t workspace_float_size,
                           cudaStream_t stream = nullptr);

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
    GemmStatus run(const GemmProblemDesc* problems, int num_problems,
                   const void* const* A_array, const void* const* B_array,
                   void* const* D_array,
                   const int64_t* lda_array, const int64_t* ldb_array,
                   const int64_t* ldd_array,
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
 * @brief Execute segment GEMM (stub: not implemented)
 *
 * For processing contiguous segments in a single buffer.
 */
GemmStatus invokeSegmentGemm(const void* A, const void* B, void* D,
                             const int* segment_offsets, int num_segments,
                             int K, int N, DataType dtype,
                             bool weight_column_major,
                             void* workspace, size_t workspace_size,
                             cudaStream_t stream = nullptr);

} // namespace gemm
} // namespace kernels
} // namespace oasr
