# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""One dtype tolerance ladder.

The pair ``(1e-4, 1e-4) if dtype is float32 else (1e-2, 1e-2)`` was written out
by hand in nine test modules and near-copied in three more, which meant a
kernel that needed a looser bound got one file relaxed and the rest left to
drift.  Ask for the ladder, and pass ``half=`` / ``rtol=`` / ``atol=`` when a
particular kernel genuinely needs a different bound -- an override at the call
site is visible; a fork of the table is not.
"""

from __future__ import annotations

import torch

#: Default (rtol, atol) per dtype.  fp32 is the reference path, so it is held an
#: order tighter than the two 10-bit-mantissa formats.
_LADDER = {
    torch.float32: (1e-4, 1e-4),
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (1e-2, 1e-2),
}


def tol(dtype: torch.dtype, *, half: tuple[float, float] | None = None) -> dict[str, float]:
    """``{"rtol": ..., "atol": ...}`` for ``dtype``, ready to splat into assert_close.

    ``half`` overrides the fp16/bf16 row only, which is the override every
    caller that needed one actually wanted.
    """
    if half is not None and dtype in (torch.float16, torch.bfloat16):
        rtol, atol = half
    else:
        rtol, atol = _LADDER.get(dtype, (1e-3, 1e-3))
    return {"rtol": rtol, "atol": atol}
