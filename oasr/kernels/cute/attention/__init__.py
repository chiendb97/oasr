# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""CuteDSL-based fused attention kernels.

Per-arch forward-pass classes:
    FmhaBase          -- abstract spec.
    FmhaForwardSm80   -- the SM80 / SM120 mainloop. Composes the helper
                         modules under ``oasr.kernels.cute``.
    FmhaSm80          -- Ampere kernel (sm_80 / sm_86 / sm_89). Thin shim.
    FmhaSm120         -- consumer Blackwell (RTX 50xx). Thin shim with
                         the 99 KB smem cap.
"""

from .base import FmhaBase
from .fmha_fwd_sm80 import FmhaForwardSm80
from .fmha_sm80 import FmhaSm80
from .fmha_sm120 import FmhaSm120

__all__ = ["FmhaBase", "FmhaForwardSm80", "FmhaSm80", "FmhaSm120"]
