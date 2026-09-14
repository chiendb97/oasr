# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""CuteDSL-based fused attention kernels.

Per-arch forward-pass classes.  One per supported SM, even where the kernel
body is shared: the class also answers the shared-memory budget, so an arch
served by a class that names a different one is budgeted wrong (see
``FmhaSm80._smem_arch_str`` and ``base.pick_arch_cls``).

    FmhaBase          -- abstract spec.
    FmhaSm80          -- Ampere data-center kernel (A100, A30). Carries the
                         mainloop; 163 KB smem cap.
    FmhaSm86          -- Ampere consumer / workstation (A10, A40, RTX 30xx).
                         Thin shim with the 99 KB smem cap.
    FmhaSm89          -- Ada Lovelace (L4, L40S, RTX 4090). Thin shim with
                         the 99 KB smem cap.
    FmhaSm120         -- consumer Blackwell (RTX 50xx). Thin shim with
                         the 99 KB smem cap.
"""

from .base import FmhaBase
from .fmha_sm80 import FmhaSm80
from .fmha_sm86 import FmhaSm86
from .fmha_sm89 import FmhaSm89
from .fmha_sm120 import FmhaSm120

__all__ = ["FmhaBase", "FmhaSm80", "FmhaSm86", "FmhaSm89", "FmhaSm120"]
