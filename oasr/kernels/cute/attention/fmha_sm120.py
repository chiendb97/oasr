# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""SM120 fused multi-head attention (CuteDSL).

Backend for consumer Blackwell (RTX 50xx, RTX PRO 6000, sm_120). Reuses the
SM80 mainloop verbatim; the only difference is the 99 KB smem cap, which
sm_120 shares with sm_86 and sm_89.
"""

from .fmha_sm80 import FmhaSm80


class FmhaSm120(FmhaSm80):
    """SM120 FMHA -- SM80 kernel body, SM120-sized smem budget."""

    arch = 120
