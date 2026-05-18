# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""CuteDSL-based fused attention kernels.

Per-arch forward-pass classes:
    FmhaBase          -- abstract spec.
    FmhaSm80          -- Ampere kernel (sm_80 / sm_86 / sm_89). Thin shim.
    FmhaSm120         -- consumer Blackwell (RTX 50xx). Thin shim with
                         the 99 KB smem cap.
"""

from .base import FmhaBase
from .fmha_sm80 import FmhaSm80
from .fmha_sm120 import FmhaSm120

__all__ = ["FmhaBase", "FmhaSm80", "FmhaSm120"]
