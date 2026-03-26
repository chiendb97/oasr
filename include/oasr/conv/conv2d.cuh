// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Conv2D kernel -- facade header.
//
// Includes the FlashInfer-style layered headers:
//   - cutlass_conv2d_configs.h : Conv2dConfig, SmConv2dTraits, DefaultConv2dConfig
//   - conv2d_cutlass_template.h : CutlassConv2dFpropKernel template
//   - conv2d_cutlass.h : Public Conv2D() and Conv2DActivation() dispatch

#pragma once

#include <oasr/conv/cutlass_conv2d_configs.h>
#include <oasr/conv/conv2d_cutlass.h>
#include <oasr/conv/conv2d_cutlass_template.h>
