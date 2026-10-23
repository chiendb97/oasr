# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""What a model's structure allows: the data an architecture declares.

A model returns a :class:`QuantLayout` from ``quantization_layout()``:

* :class:`SmoothGroup` s — GEMMs that see one shared input, the producer of that
  input, and how a SmoothQuant per-channel scale ``s`` is absorbed (§6.5 of the
  design).  Only declared groups are smoothed: inferring them from module names
  is fragile on fused QKV, ``PointwiseConv1d`` and a cache ``cat`` between a norm
  and its GEMM.
* ``keep_full_precision`` — globs the model says must not be quantized
  (subsampling front-ends, positional projections, output heads).
* :class:`GemmChain` s — a GEMM whose (activated) output is read by exactly one
  other GEMM and nothing else (the FFN's ``w_1`` -> ``w_2``), so the producer's
  epilogue may quantize straight for the consumer.
* ``batch_invariant`` — full-precision GEMMs to pin to a batch-invariant kernel
  when the model is quantized (``Linear.batch_invariant``): an fp layer whose
  rounding depends on how many rows share the launch feeds int8 layers that
  amplify it.

Pure data: a model file imports these dataclasses and nothing else from
:mod:`oasr.quantization`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

#: How a group's smoothing vector ``s`` is absorbed (``x' = x / s``, ``W' = W·s``).
#:
#: * ``norm_affine``   — the producer is a LayerNorm / RMSNorm: ``γ /= s``, ``β /= s``.
#: * ``v_to_o``        — the producer is the V projection -- the V rows of a fused
#:   QKV projection (``v_rows``), or a separate ``v_proj`` whole -- whose output
#:   feeds the consumer through attention (linear in V per channel).
#: * ``producer_rows`` — the producer is a GEMM whose output reaches the consumer
#:   through a per-channel positively homogeneous map: a ReLU, or the gated MLP's
#:   ``act(gate) * up`` (linear in ``up``).  Its output rows and bias absorb ``1/s``.
#: * ``runtime``       — nothing can absorb it: the consumer stores
#:   ``smooth_inv = 1/s`` and the activation-quantize kernel applies it.
FOLD_KINDS = ("norm_affine", "v_to_o", "producer_rows", "runtime")


@dataclass(frozen=True)
class SmoothGroup:
    """GEMMs sharing one input activation, and how its smoothing is absorbed."""

    producer: Optional[str]
    consumers: Tuple[str, ...]
    fold: str
    #: The producer norm's output is read by the consumers and nothing else, so
    #: when they all take a pre-quantized per-token input the norm may emit it
    #: (``LayerNorm.quantize_output``).  Only for a norm producer.
    fuse_output_quant: bool = False
    #: ``v_to_o``: the slice of the producer's output rows that is V
    #: (``(start, stop)``; ``(0, out_features)`` for a separate ``v_proj``).
    v_rows: Optional[Tuple[int, int]] = None
    #: ``v_to_o``: ``(n_head, n_kv_head, head_dim)``.  Under GQA a V channel feeds
    #: ``n_head // n_kv_head`` output channels, so ``s`` is tied across them.
    heads: Optional[Tuple[int, int, int]] = None

    def __post_init__(self) -> None:
        if self.fold not in FOLD_KINDS:
            raise ValueError(f"SmoothGroup.fold must be one of {FOLD_KINDS}, got {self.fold!r}")
        if not self.consumers:
            raise ValueError("a SmoothGroup needs at least one consumer")
        if self.fold == "runtime" and self.producer is not None:
            raise ValueError("a runtime-smoothed group has no producer")
        if self.fold != "runtime" and self.producer is None:
            raise ValueError(f"fold {self.fold!r} needs a producer")
        if self.fold == "v_to_o" and self.v_rows is None:
            raise ValueError("a v_to_o group needs v_rows (the V slice of the producer)")
        if self.fuse_output_quant and self.fold != "norm_affine":
            raise ValueError("fuse_output_quant needs a norm producer (fold 'norm_affine')")


@dataclass(frozen=True)
class GemmChain:
    """``consumer(producer(x))`` with nothing else reading the producer's output.

    The producer's fused activation, if any, is part of "its output".  When the
    consumer carries a calibrated static input scale, the producer's epilogue
    quantizes for it and the pair exchanges int8 (``scaled_mm_quant``) -- no fp
    round trip, no quantize launch.  A per-token consumer cannot be served this
    way: its scale needs the whole row, which no single output tile sees.
    """

    producer: str
    consumer: str


@dataclass(frozen=True)
class QuantLayout:
    """Everything an architecture declares about quantizing itself."""

    groups: Tuple[SmoothGroup, ...] = ()
    keep_full_precision: Tuple[str, ...] = ()
    chains: Tuple[GemmChain, ...] = ()
    batch_invariant: Tuple[str, ...] = ()


__all__ = ["FOLD_KINDS", "GemmChain", "QuantLayout", "SmoothGroup"]
