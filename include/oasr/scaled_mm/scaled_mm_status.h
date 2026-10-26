// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// The result of a scaled-GEMM launch, shared by both kernel generations.

#pragma once

namespace oasr {
namespace scaled_mm {

enum class ScaledMmStatus { SUCCESS = 0, CANNOT_IMPLEMENT, INITIALIZE_FAILED, RUN_FAILED };

}  // namespace scaled_mm
}  // namespace oasr
