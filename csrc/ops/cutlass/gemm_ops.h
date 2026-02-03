// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/types.h"
#include <cuda_runtime.h>

namespace oasr {
namespace ops {

/**
 * @brief GEMM operation configuration
 * 
 * Configures CUTLASS GEMM operations for optimal performance.
 */
struct GemmConfig {
    // Matrix dimensions: C = alpha * A @ B + beta * C
    // A: [M, K], B: [K, N], C: [M, N]
    int M;
    int N;
    int K;
    
    // Scaling factors
    float alpha;
    float beta;
    
    // Data types
    DataType dtype_a;           // Type of matrix A
    DataType dtype_b;           // Type of matrix B
    DataType dtype_c;           // Type of matrix C
    DataType dtype_compute;     // Compute type (accumulator)
    
    // Matrix layout (row-major or column-major)
    bool trans_a;               // Transpose A
    bool trans_b;               // Transpose B
    
    // Leading dimensions
    int lda;                    // Leading dimension of A
    int ldb;                    // Leading dimension of B
    int ldc;                    // Leading dimension of C
    
    // Batched GEMM
    int batch_count;            // Number of matrices in batch
    int64_t stride_a;           // Stride between A matrices
    int64_t stride_b;           // Stride between B matrices
    int64_t stride_c;           // Stride between C matrices
    
    // Epilogue configuration
    enum class Epilogue {
        NONE,                   // No epilogue
        BIAS,                   // Add bias
        RELU,                   // ReLU activation
        GELU,                   // GELU activation
        SWISH,                  // Swish activation
        BIAS_RELU,              // Bias + ReLU
        BIAS_GELU,              // Bias + GELU
        BIAS_SWISH,             // Bias + Swish
    };
    Epilogue epilogue;
    const void* bias;           // Bias vector [N] for epilogue
    
    GemmConfig()
        : M(0), N(0), K(0)
        , alpha(1.0f), beta(0.0f)
        , dtype_a(DataType::FP16), dtype_b(DataType::FP16)
        , dtype_c(DataType::FP16), dtype_compute(DataType::FP32)
        , trans_a(false), trans_b(false)
        , lda(0), ldb(0), ldc(0)
        , batch_count(1), stride_a(0), stride_b(0), stride_c(0)
        , epilogue(Epilogue::NONE), bias(nullptr)
    {}
};

/**
 * @brief Abstract interface for GEMM operations
 * 
 * Provides a unified interface for different GEMM implementations:
 * - cuBLAS
 * - CUTLASS
 * - Custom CUDA kernels
 */
class GemmOp {
public:
    virtual ~GemmOp() = default;
    
    /**
     * @brief Execute GEMM operation
     * 
     * @param a Pointer to matrix A
     * @param b Pointer to matrix B
     * @param c Pointer to matrix C (input/output)
     * @param config GEMM configuration
     * @param stream CUDA stream
     */
    virtual void execute(const void* a, const void* b, void* c,
                         const GemmConfig& config, cudaStream_t stream) = 0;
    
    /**
     * @brief Get workspace size required for this operation
     */
    virtual size_t getWorkspaceSize(const GemmConfig& config) const = 0;
    
    /**
     * @brief Set workspace buffer
     */
    virtual void setWorkspace(void* workspace, size_t size) = 0;
};

/**
 * @brief CUTLASS-based GEMM implementation
 */
class CutlassGemm : public GemmOp {
public:
    CutlassGemm();
    ~CutlassGemm() override;
    
    void execute(const void* a, const void* b, void* c,
                 const GemmConfig& config, cudaStream_t stream) override;
    
    size_t getWorkspaceSize(const GemmConfig& config) const override;
    void setWorkspace(void* workspace, size_t size) override;
    
    /**
     * @brief Select optimal kernel configuration based on problem size
     */
    void autoTune(const GemmConfig& config, cudaStream_t stream);
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief cuBLAS-based GEMM implementation
 */
class CublasGemm : public GemmOp {
public:
    CublasGemm();
    ~CublasGemm() override;
    
    void execute(const void* a, const void* b, void* c,
                 const GemmConfig& config, cudaStream_t stream) override;
    
    size_t getWorkspaceSize(const GemmConfig& config) const override;
    void setWorkspace(void* workspace, size_t size) override;
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief GEMM with automatic backend selection
 * 
 * Selects the optimal GEMM implementation based on:
 * - Problem size
 * - Data types
 * - Available hardware
 */
void gemm(const void* a, const void* b, void* c,
          const GemmConfig& config, cudaStream_t stream);

/**
 * @brief Batched GEMM
 * 
 * C[i] = alpha * A[i] @ B[i] + beta * C[i] for i in [0, batch_count)
 */
void batchedGemm(const void* a, const void* b, void* c,
                 const GemmConfig& config, cudaStream_t stream);

/**
 * @brief Grouped GEMM (different sizes per batch)
 * 
 * For variable-length sequence processing.
 * 
 * @param a Array of A matrix pointers
 * @param b Array of B matrix pointers
 * @param c Array of C matrix pointers
 * @param configs Array of GEMM configs (one per group)
 * @param group_count Number of groups
 * @param stream CUDA stream
 */
void groupedGemm(const void* const* a, const void* const* b, void** c,
                 const GemmConfig* configs, int group_count,
                 cudaStream_t stream);

/**
 * @brief Get optimal GEMM implementation for given configuration
 */
std::unique_ptr<GemmOp> createOptimalGemm(const GemmConfig& config);

} // namespace ops
} // namespace oasr
