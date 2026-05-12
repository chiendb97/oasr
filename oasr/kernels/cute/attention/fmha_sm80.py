# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""SM80 fused multi-head attention (CuteDSL).

Thin shim over :class:`FmhaForwardSm80` -- the mainloop, helpers, and
feasibility logic live in :mod:`fmha_fwd_sm80`. SM80-specific knob: the
``_smem_arch_str`` used by ``cutlass_utils.get_smem_capacity_in_bytes``
(163 KB on SM80 / sm_86 / sm_89).
"""

from .fmha_fwd_sm80 import FmhaForwardSm80


class FmhaSm80(FmhaForwardSm80):
    """Ampere / Ada (sm_80 / sm_86 / sm_89) FMHA forward."""

    arch = 80
    _smem_arch_str = "sm_80"
