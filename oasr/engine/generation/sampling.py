# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Per-row next-token selection for the incremental AR strategies.

Greedy is the production default and stays a single batched ``argmax`` — the
sampling path activates only for rows whose :class:`DecodingOptions` set
``temperature > 0``, and those rows are filtered and drawn one by one (sampling
requests are rare, batches are small, and the per-row filters differ anyway).

Nothing here waits for the device.  Each row's draw stays a device tensor and
all of them land in the token vector with one ``index_copy_``, so the only sync
of a decode step is the caller's own read of the tokens.  A sampled row used to
cost three: ``.item()`` on the draw, the ``tokens[i] = int`` write, and a
``cut[..., 0] = False`` write -- a Python scalar into a CUDA tensor is a
blocking copy, basic indexing included.  The rows are still filtered and drawn
one at a time, and not batched, because a batched ``multinomial`` consumes the
generator differently (a seeded run would draw other tokens) and a batched
``cumsum`` differs from a 1-D one in the last bits, which can move a ``top_p``
cut.  Per row, the tokens are the same as before, draw for draw.

Semantics match HuggingFace ``generate``: logits are divided by the
temperature, then ``top_k`` keeps the k highest logits, then ``top_p``
(nucleus) keeps the smallest descending-probability prefix whose cumulative
mass reaches ``top_p`` (always at least one token), and one token is drawn
from the renormalised distribution.  Draws use the process-global torch
generator — seed with ``torch.manual_seed`` for reproducibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import torch

from oasr.utils.staging import to_device

if TYPE_CHECKING:
    from ..request import DecodingOptions


def _sample_row(logits: torch.Tensor, opts: "DecodingOptions") -> torch.Tensor:
    """Draw one token from a single ``(V,)`` logits row per ``opts``: a ``(1,)`` tensor.

    Stays on the device: no ``.item()``, and no Python scalar written into a
    device tensor.
    """
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
        cut[..., 0].fill_(False)
        row = row.masked_fill(cut.scatter(0, sorted_idx, cut), float("-inf"))
    probs = torch.softmax(row, dim=-1)
    return torch.multinomial(probs, num_samples=1)


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
    sampled = [(i, o) for i, o in enumerate(opts_rows) if o is not None and o.sampling]
    if not sampled:
        return tokens
    draws = torch.cat([_sample_row(logits[i], o) for i, o in sampled])
    rows = to_device([i for i, _ in sampled], dtype=torch.long, device=tokens.device)
    return tokens.index_copy_(0, rows, draws.to(tokens.dtype))
