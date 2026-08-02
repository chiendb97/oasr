# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Token / position embeddings — the waist's gather entry point.

There is no OASR embedding kernel today: a gather is bandwidth-bound and
``F.embedding`` is already the right implementation.  The module exists anyway
because the waist is a *structural* commitment, not only a performance one —
the vocabulary axis is where tensor parallelism (``VocabParallelEmbedding``)
and weight quantization attach, and a model that reaches for ``nn.Embedding``
directly has to be rewritten when either lands.  Keeping it here means the
conformance test in ``tests/test_layer_waist.py`` can hold the line.

Parameter layout is ``nn.Embedding``'s (``weight (num_embeddings, dim)``), so
a checkpoint loads 1:1 and tied LM heads (``x @ embed.weight.t()``) still work.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class Embedding(nn.Module):
    """``ids -> weight[ids]`` (drop-in for ``nn.Embedding``, inference paths).

    Only the inference-relevant subset of ``nn.Embedding``'s options is
    carried: ``padding_idx`` (kept because icefall's stateless transducer
    predictor declares it, and it must survive a state-dict round trip) and
    the ``device`` / ``dtype`` factory kwargs.  Training-only knobs
    (``max_norm``, ``scale_grad_by_freq``, ``sparse``) are deliberately absent
    rather than silently ignored.
    """

    #: Under tensor parallelism the vocabulary axis is the shard axis.
    tp_dim = 0

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        nn.init.normal_(self.weight)
        if padding_idx is not None:
            with torch.no_grad():
                self.weight[padding_idx].fill_(0)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """``ids (*)`` int64 → ``(*, embedding_dim)``."""
        return F.embedding(ids, self.weight, self.padding_idx)

    def extra_repr(self) -> str:
        s = f"{self.num_embeddings}, {self.embedding_dim}"
        if self.padding_idx is not None:
            s += f", padding_idx={self.padding_idx}"
        return s


#: Name the vocabulary-parallel role explicitly at the definition site, the way
#: :class:`~oasr.layers.linear.ColumnParallelLinear` does for projections.
VocabParallelEmbedding = Embedding


__all__ = ["Embedding", "VocabParallelEmbedding"]
