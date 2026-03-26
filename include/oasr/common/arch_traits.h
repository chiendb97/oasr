// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Legacy ArchTraits header -- kept for backward compatibility.
//
// GEMM configs: see gemm/cutlass_gemm_configs.h (SmMMATraits, GemmConfig)
// Conv2D configs: see conv/cutlass_conv2d_configs.h (SmConv2dTraits, Conv2dConfig)
//
// This file is intentionally minimal. New code should use the per-family
// config headers directly.

#pragma once

#ifdef __GNUC__
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wstrict-aliasing"
    #pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma.h>

#ifdef __GNUC__
    #pragma GCC diagnostic pop
#endif
