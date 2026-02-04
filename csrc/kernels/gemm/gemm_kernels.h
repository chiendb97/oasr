// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Standard GEMM kernel interfaces using NVIDIA CUTLASS
// Supports BF16 and FP16 precision
// Reference: https://github.com/flashinfer-ai/flashinfer/tree/main/include/flashinfer/gemm

#pragma once

#include "gemm_params.h"
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
// Standard GEMM Interface
//==============================================================================

/**
 * @brief Execute GEMM operation
 * 
 * Computes: D = alpha * op(A) @ op(B) + beta * C
 * 
 * @param params GEMM parameters
 * @return Status code
 */
GemmStatus invokeGemm(const GemmParams& params);

/**
 * @brief Execute GEMM with specific data type
 * 
 * Template specializations for FP16 and BF16
 */
template <typename T>
GemmStatus invokeGemmTyped(const GemmParams& params);

/**
 * @brief Query required workspace size for GEMM
 */
size_t queryGemmWorkspaceSize(int M, int N, int K, DataType dtype,
                              const GemmConfig& config = GemmConfig());

//==============================================================================
// GEMM with Fused Operations
//==============================================================================

/**
 * @brief GEMM with fused bias and activation
 * 
 * Computes: D = activation(A @ B + bias)
 */
GemmStatus invokeGemmBiasActivation(
    const void* A, const void* B, const void* bias, void* D,
    int M, int N, int K,
    ActivationType activation,
    DataType dtype,
    cudaStream_t stream);

//==============================================================================
// GEMM Runner Class
//==============================================================================

/**
 * @brief Abstract interface for GEMM runners
 */
class GemmRunnerInterface {
public:
    virtual ~GemmRunnerInterface() = default;
    
    /**
     * @brief Execute GEMM operation
     */
    virtual void gemm(const void* A, const void* B, void* D,
                      int M, int N, int K,
                      const GemmConfig& config,
                      void* workspace, size_t workspace_size,
                      cudaStream_t stream) = 0;
    
    /**
     * @brief Get required workspace size
     */
    virtual size_t getWorkspaceSize(int M, int N, int K) = 0;
    
    /**
     * @brief Get available configurations for auto-tuning
     */
    virtual std::vector<GemmConfig> getConfigs() const = 0;
};

/**
 * @brief FP16 GEMM runner using CUTLASS
 */
class Fp16GemmRunner : public GemmRunnerInterface {
public:
    explicit Fp16GemmRunner(int sm_version = 80);
    ~Fp16GemmRunner() override;
    
    void gemm(const void* A, const void* B, void* D,
              int M, int N, int K,
              const GemmConfig& config,
              void* workspace, size_t workspace_size,
              cudaStream_t stream) override;
    
    size_t getWorkspaceSize(int M, int N, int K) override;
    std::vector<GemmConfig> getConfigs() const override;
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief BF16 GEMM runner using CUTLASS
 */
class Bf16GemmRunner : public GemmRunnerInterface {
public:
    explicit Bf16GemmRunner(int sm_version = 80);
    ~Bf16GemmRunner() override;
    
    void gemm(const void* A, const void* B, void* D,
              int M, int N, int K,
              const GemmConfig& config,
              void* workspace, size_t workspace_size,
              cudaStream_t stream) override;
    
    size_t getWorkspaceSize(int M, int N, int K) override;
    std::vector<GemmConfig> getConfigs() const override;
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

//==============================================================================
// Auto-Tuning
//==============================================================================

/**
 * @brief Profile and select best GEMM configuration
 * 
 * @param M, N, K Problem dimensions
 * @param dtype Data type
 * @param num_warmup Number of warmup iterations
 * @param num_iter Number of timing iterations
 * @return Best configuration
 */
GemmConfig autoTuneGemm(int M, int N, int K, DataType dtype,
                        int num_warmup = 5, int num_iter = 10,
                        cudaStream_t stream = nullptr);

} // namespace gemm
} // namespace kernels
} // namespace oasr
