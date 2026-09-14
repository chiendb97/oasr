# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""SM89 fused multi-head attention (CuteDSL).

Backend for Ada Lovelace (L4, L40S, RTX 4090, RTX Ada, sm_89). Reuses the SM80
mainloop verbatim; the only difference is the 99 KB smem cap, which sm_89
shares with sm_86 and consumer Blackwell and *not* with the A100 the mainloop
is named after.

Its absence is what made ``pick_arch_cls`` hand sm_89 the A100-budgeted
:class:`FmhaSm80`; see :meth:`FmhaSm80._smem_arch_str`.
"""

from .fmha_sm80 import FmhaSm80


class FmhaSm89(FmhaSm80):
    """SM89 FMHA -- SM80 kernel body, SM89-sized smem budget."""

    arch = 89
