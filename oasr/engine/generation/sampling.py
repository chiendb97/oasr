# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Per-row next-token selection for the incremental AR strategies.

Greedy is the production default and stays a single batched ``argmax`` — the
sampling path activates only for rows whose :class:`DecodingOptions` set
``temperature > 0``, and those rows are drawn one by one (sampling requests
are rare, batches are small, and the per-row filters differ anyway).

Semantics match HuggingFace ``generate``: logits are divided by the
temperature, then ``top_k`` keeps the k highest logits, then ``top_p``
(nucleus) keeps the smallest descending-probability prefix whose cumulative
mass reaches ``top_p`` (always at least one token), and one token is drawn
from the renormalised distribution.  Draws use the process-global torch
generator — seed with ``torch.manual_seed`` for reproducibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Sequence

import torch

if TYPE_CHECKING:
    from ..request import DecodingOptions


def _sample_row(logits: torch.Tensor, opts: "DecodingOptions") -> int:
    """Draw one token from a single ``(V,)`` logits row per ``opts``."""
    row = logits / opts.temperature
    if opts.top_k > 0 and opts.top_k < row.numel():
        kth = torch.topk(row, opts.top_k).values[-1]
        row = row.masked_fill(row < kth, float("-inf"))
    if opts.top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(row, descending=True)
        cum = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        # Keep every token up to (and including) the one that crosses top_p.
        cut = cum >= opts.top_p
        cut[..., 1:] = cut[..., :-1].clone()
        cut[..., 0] = False
        row = row.masked_fill(cut.scatter(0, sorted_idx, cut), float("-inf"))
    probs = torch.softmax(row, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def select_next_tokens(
    logits: torch.Tensor,
    opts_rows: Sequence[Optional["DecodingOptions"]],
) -> torch.Tensor:
    """Pick the next token per row of ``logits (B, V)``.

    ``opts_rows[i]`` is row ``i``'s :class:`DecodingOptions` (or ``None``).
    Rows without sampling take the batched argmax (the unchanged greedy fast
    path — no per-row work at all when nobody samples).
    """
    tokens = logits.argmax(dim=-1)
    sampled: List[int] = [i for i, o in enumerate(opts_rows) if o is not None and o.sampling]
    for i in sampled:
        tokens[i] = _sample_row(logits[i], opts_rows[i])
    return tokens
