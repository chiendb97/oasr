# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""SM120 fused multi-head attention (CuteDSL).

Backend for consumer Blackwell (RTX 50xx, sm_120). SM120 reuses the SM80
m16n8k16 / cp.async path.
"""

from .fmha_sm80 import FmhaSm80


class FmhaSm120(FmhaSm80):
    """SM120 FMHA kernel -- SM80 kernel body, SM120-sized smem budget.
    """

    arch = 120

    _smem_arch_str = "sm_120"
