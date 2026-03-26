// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// SM100+ (Blackwell) GEMM template stub.
//
// Placeholder for future CUTLASS 3.x TMA-based GEMM implementation.
// Currently aliases the CUTLASS 2.x template. When CUTLASS 3.x support
// is added, this file will contain the SM100-specific kernel using
// TMA (Tensor Memory Accelerator) and wgmma instructions.

#pragma once

#include <oasr/gemm/gemm_cutlass_template.h>

namespace oasr {
namespace gemm {

// For now, SM100+ uses the same CUTLASS 2.x template as earlier architectures.
// Future: Replace with CUTLASS 3.x CollectiveMainloop + TMA pipeline.
template <typename Config, typename MMATraits, typename ElementA, typename ElementB,
          typename ElementCD, typename LayoutA, typename LayoutB, typename LayoutCD,
          template <int, typename, typename> class EpilogueFunctor>
using CutlassGemmKernelSm100 =
    CutlassGemmKernel<Config, MMATraits, ElementA, ElementB, ElementCD, LayoutA, LayoutB, LayoutCD,
                       EpilogueFunctor>;

}  // namespace gemm
}  // namespace oasr
