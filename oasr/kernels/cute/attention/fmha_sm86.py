# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""SM86 fused multi-head attention (CuteDSL).

Backend for Ampere consumer / workstation parts (A10, A10G, A16, A40,
RTX 30-series, sm_86). Reuses the SM80 mainloop verbatim; the only difference
is the 99 KB smem cap, which sm_86 shares with Ada and consumer Blackwell and
*not* with the A100 the mainloop is named after.

Its absence is what made ``pick_arch_cls`` hand sm_86 the A100-budgeted
:class:`FmhaSm80`; see :meth:`FmhaSm80._smem_arch_str`.
"""

from .fmha_sm80 import FmhaSm80


class FmhaSm86(FmhaSm80):
    """SM86 FMHA -- SM80 kernel body, SM86-sized smem budget."""

    arch = 86
