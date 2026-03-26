// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Standard GEMM kernel -- facade header.
//
// Includes the FlashInfer-style layered headers:
//   - cutlass_gemm_configs.h : GemmConfig, SmMMATraits, DefaultGemmConfig
//   - gemm_cutlass_template.h : CutlassGemmKernel template implementation
//   - gemm_cutlass.h : Public Gemm() and GemmActivation() dispatch interface
//   - gemm_utils.h : GemmStatus codes and dispatch macros

#pragma once

#include <oasr/gemm/cutlass_gemm_configs.h>
#include <oasr/gemm/gemm_cutlass.h>
#include <oasr/gemm/gemm_cutlass_template.h>
#include <oasr/gemm/gemm_utils.h>
