# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""SmoothQuant (Xiao et al., ICML 2023): migrate activation outliers into weights.

For a group of GEMMs sharing one input ``X`` (``[*, K]``), a per-channel vector
``s`` rewrites ``X @ Wᵀ`` as ``(X / s) @ (W · s)ᵀ`` — exact in real arithmetic —
with

    s_j = clamp(max|X_j|^α / max_k max_n |W_k[n, j]|^(1-α), s_lo_j, s_hi)

so the activations' outlier channels shrink and the weights absorb them, where
int8 per-channel weights have range to spare.  ``X / s`` is never computed: the
model's declared :class:`~oasr.quantization.layout.SmoothGroup` says where it is
absorbed (the producer norm's affine, V through attention, a ReLU producer's
rows, or a runtime ``smooth_inv`` the quantize kernel applies).

Two rules keep it exact on a real model:

* every scale is computed from the *original* weights and statistics before any
  fold is applied — a layer can be a consumer (its input columns) in one group
  and a producer (its output rows) in another, and the folds commute;
* ``s_lo`` keeps every folded parameter inside the served dtype: ``γ / s`` of a
  norm overflows fp16 when ``s`` is tiny, and the accuracy gate serves fp16.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import AbstractSet, Dict, List, Optional, Tuple, cast

import torch
from torch import nn

from oasr.layers._quantizable import QuantizableGemm

from ..calibration.observers import AbsMaxPerChannel
from ..layout import FOLD_KINDS, QuantLayout, SmoothGroup
from ..spec import QuantizationSpec
from .base import ForwardLoop, QuantAlgorithm, register_quant_algorithm

#: The largest finite fp16 value: folded parameters must stay below
#: ``fp16_headroom`` of it.
_FP16_MAX = 65504.0


@dataclass(frozen=True)
class SmoothQuantOptions:
    """SmoothQuant knobs."""

    #: Migration strength: 0 leaves activations alone, 1 moves everything to W.
    alpha: float = 0.5
    #: Upper clamp on ``s``.
    s_max: float = 1e4
    #: Lower clamp on ``s`` (llm-compressor's ``MINIMUM_SMOOTHING_SCALE``).
    s_floor: float = 1e-5
    #: Folded producer parameters must stay within this fraction of fp16's range.
    fp16_headroom: float = 0.5
    #: Comma-separated fold kinds to apply (a subset of ``FOLD_KINDS``).
    folds: str = ",".join(FOLD_KINDS)


@register_quant_algorithm("smoothquant")
class SmoothQuant(QuantAlgorithm):
    """Fold per-channel smoothing scales into the model's declared groups."""

    options_cls = SmoothQuantOptions
    needs_calibration = True

    def apply(
        self,
        model: nn.Module,
        layout: QuantLayout,
        spec: QuantizationSpec,
        forward_loop: Optional[ForwardLoop],
    ) -> QuantizationSpec:
        opts: SmoothQuantOptions = self.options or SmoothQuantOptions()
        if not 0.0 <= opts.alpha <= 1.0:
            raise ValueError(f"SmoothQuant alpha must be in [0, 1], got {opts.alpha}")
        if forward_loop is None:
            raise ValueError("SmoothQuant needs calibration audio (a forward_loop)")
        if not layout.groups:
            raise ValueError(
                f"{type(model).__name__} declares no smoothing groups "
                "(quantization_layout()); SmoothQuant cannot run on it"
            )
        folds = {f.strip() for f in opts.folds.split(",") if f.strip()}
        unknown = folds - set(FOLD_KINDS)
        if unknown:
            raise ValueError(f"unknown fold kind(s) {sorted(unknown)}; known: {FOLD_KINDS}")

        modules = dict(model.named_modules())
        groups = [g for g in layout.groups if g.fold in folds and self._selected(g, spec)]
        for group in groups:
            _validate_group(group, modules, spec)

        # Every consumer of a group reads the same input, so each one is observed
        # and the statistic is their maximum: a group whose consumers span modules
        # (an encoder output feeding every decoder layer's cross-attention) is
        # measured wherever the calibration loop happens to reach it.
        observers = {id(g): AbsMaxPerChannel() for g in groups}
        handles = [
            modules[c].register_forward_pre_hook(_observe(observers[id(g)]))
            for g in groups
            for c in g.consumers
        ]
        try:
            forward_loop(model)
        finally:
            for handle in handles:
                handle.remove()

        unobserved = [g for g in groups if observers[id(g)].rows == 0]
        if unobserved:
            raise RuntimeError(
                f"the calibration loop never reached {len(unobserved)} smoothing group(s) "
                f"(first: {unobserved[0].consumers[0]}); a decoder the loop does not drive "
                "cannot be smoothed -- see BaseAsrModel.calibration_forward"
            )
        plans: List[Tuple[SmoothGroup, torch.Tensor]] = []
        for group in groups:
            act_amax = observers[id(group)].result()
            plans.append((group, _smoothing_vector(group, act_amax, modules, opts)))
        # A quantized producer's weight is stored as 8-bit values with fp32 per-row
        # scales, so only a full-precision one must stay inside the served range.
        full_precision = frozenset(
            g.producer
            for g, _ in plans
            if g.producer is not None
            and isinstance(modules[g.producer], QuantizableGemm)
            and spec.resolve(g.producer) is None
        )
        with torch.no_grad():
            for group, s in plans:
                _fold(group, s, modules, opts, full_precision=full_precision)

        runtime = tuple(c for g, _ in plans if g.fold == "runtime" for c in g.consumers)
        producer = dict(spec.producer)
        producer.update(
            {
                "algorithm": "smoothquant",
                "alpha": opts.alpha,
                "folds": sorted(folds),
                "smoothed_groups": len(plans),
            }
        )
        return dataclasses.replace(
            spec,
            runtime_smoothed=tuple(spec.runtime_smoothed) + runtime,
            producer=producer,
        )

    @staticmethod
    def _selected(group: SmoothGroup, spec: QuantizationSpec) -> bool:
        """Smooth a group only if its consumers are quantized (all, for ``runtime``)."""
        quantized = [spec.resolve(c) is not None for c in group.consumers]
        if group.fold == "runtime":
            return all(quantized)
        return any(quantized)


def _observe(observer: AbsMaxPerChannel):
    def hook(module: nn.Module, args):  # noqa: ARG001 - torch hook signature
        observer.update(args[0])

    return hook


def _validate_group(
    group: SmoothGroup, modules: Dict[str, nn.Module], spec: QuantizationSpec
) -> None:
    from oasr.layers.norm import LayerNorm, RMSNorm

    for path in group.consumers + ((group.producer,) if group.producer else ()):
        if path not in modules:
            raise ValueError(f"smoothing group names {path!r}, which the model does not have")
    for path in group.consumers:
        if not isinstance(modules[path], QuantizableGemm):
            raise ValueError(f"smoothing consumer {path} is not a QuantizableGemm")
    widths = {modules[c].in_features for c in group.consumers}  # type: ignore[union-attr]
    if len(widths) != 1:
        raise ValueError(f"smoothing group consumers disagree on input width: {group.consumers}")
    producer = modules[group.producer] if group.producer else None
    if group.fold == "norm_affine" and not isinstance(producer, (LayerNorm, RMSNorm)):
        raise ValueError(f"norm_affine producer {group.producer} is not a LayerNorm/RMSNorm")
    if group.fold in ("v_to_o", "producer_rows") and not isinstance(producer, QuantizableGemm):
        raise ValueError(f"{group.fold} producer {group.producer} is not a QuantizableGemm")
    width = widths.pop()
    if group.fold == "norm_affine":
        norm_weight = _param(cast(nn.Module, producer), "weight")
        if norm_weight is None or norm_weight.numel() != width:
            raise ValueError(
                f"norm_affine producer {group.producer} has no {width}-channel affine weight "
                f"to absorb the smoothing of {group.consumers}"
            )
    if group.fold == "producer_rows":
        rows = cast(QuantizableGemm, producer).out_features
        if rows != width:
            raise ValueError(
                f"producer_rows producer {group.producer} has {rows} output rows, its "
                f"consumers read {width} channels"
            )
    if group.fold == "v_to_o":
        assert group.v_rows is not None
        r0, r1 = group.v_rows
        rows = cast(QuantizableGemm, producer).out_features
        if not 0 <= r0 < r1 <= rows:
            raise ValueError(f"v_rows {group.v_rows} is outside {group.producer}'s {rows} rows")
        heads = group.heads
        v_width = width if heads is None else heads[1] * heads[2]
        if heads is not None and (heads[0] * heads[2] != width or heads[0] % heads[1]):
            raise ValueError(
                f"v_to_o heads {heads} do not describe the {width}-channel attention output "
                f"read by {group.consumers}"
            )
        if r1 - r0 != v_width:
            raise ValueError(
                f"v_rows {group.v_rows} ({r1 - r0} rows) is not the {v_width} V channels "
                f"feeding {group.consumers}"
            )


def _gemm(modules: Dict[str, nn.Module], path: Optional[str]) -> QuantizableGemm:
    assert path is not None
    return cast(QuantizableGemm, modules[path])


def _param(module: nn.Module, name: str) -> Optional[torch.Tensor]:
    value = getattr(module, name, None)
    return value if isinstance(value, torch.Tensor) else None


def _consumer_weights(group: SmoothGroup, modules: Dict[str, nn.Module]) -> List[torch.Tensor]:
    return [_gemm(modules, c).gemm_weight().detach().float() for c in group.consumers]


def _tie_heads(x: torch.Tensor, heads: Tuple[int, int, int]) -> torch.Tensor:
    """Max over the query heads sharing a KV head: ``[h * d] -> [h_kv * d]``."""
    h, h_kv, d = heads
    return x.reshape(h_kv, h // h_kv, d).amax(dim=1).reshape(h_kv * d)


def _untie_heads(x: torch.Tensor, heads: Tuple[int, int, int]) -> torch.Tensor:
    """``[h_kv * d] -> [h * d]``: each query head takes its KV head's value."""
    h, h_kv, d = heads
    return x.reshape(h_kv, 1, d).expand(h_kv, h // h_kv, d).reshape(h * d)


def _smoothing_vector(
    group: SmoothGroup,
    act_amax: torch.Tensor,
    modules: Dict[str, nn.Module],
    opts: SmoothQuantOptions,
) -> torch.Tensor:
    """``s`` in the consumers' input-channel space (V-channel space for GQA ``v_to_o``)."""
    weights = _consumer_weights(group, modules)
    device = weights[0].device
    w_amax = torch.stack([w.abs().amax(dim=0) for w in weights]).amax(dim=0)
    act = act_amax.float().to(device)
    gqa = group.fold == "v_to_o" and group.heads is not None and group.heads[0] != group.heads[1]
    if gqa:
        assert group.heads is not None
        act, w_amax = _tie_heads(act, group.heads), _tie_heads(w_amax, group.heads)
    s = act.clamp_min(1e-12).pow(opts.alpha) / w_amax.clamp_min(1e-12).pow(1.0 - opts.alpha)
    # A channel with no weight, or one the calibration audio never saw active (a
    # ReLU that stayed off), carries nothing to balance: leave it alone.  The
    # second matters -- the formula would send ``s`` to its floor, and the first
    # out-of-domain input that does fire the channel is then amplified ~1e5 and
    # owns its token's quantization scale (Paraformer's FFNs, calibrated on
    # English, went to noise on Chinese).
    s = torch.where((w_amax > 0) & (act > 0), s, torch.ones_like(s))
    lo = torch.full_like(s, opts.s_floor)
    bound = _producer_bound(group, modules, opts)
    if bound is not None:
        lo = torch.maximum(lo, bound.to(device))
    return torch.minimum(torch.maximum(s, lo), torch.full_like(s, opts.s_max))


def _producer_bound(
    group: SmoothGroup, modules: Dict[str, nn.Module], opts: SmoothQuantOptions
) -> Optional[torch.Tensor]:
    """The smallest ``s`` that keeps the producer's folded parameters in fp16 range.

    With a 1e-6 relative margin, so ``param / s`` at the bound rounds inside the
    limit rather than onto it.
    """
    limit = opts.fp16_headroom * _FP16_MAX / (1.0 + 1e-6)
    if group.fold == "norm_affine":
        assert group.producer is not None
        norm = modules[group.producer]
        weight = _param(norm, "weight")
        assert weight is not None
        mags = weight.detach().float().abs()
        bias = _param(norm, "bias")
        if bias is not None:
            mags = torch.maximum(mags, bias.detach().float().abs())
        return mags / limit
    if group.fold in ("v_to_o", "producer_rows"):
        producer = _gemm(modules, group.producer)
        w = producer.gemm_weight().detach().float()
        b = _param(cast(nn.Module, producer), "bias")
        if group.fold == "v_to_o":
            assert group.v_rows is not None
            r0, r1 = group.v_rows
            w = w[r0:r1]
            b = None if b is None else b[r0:r1]
        mags = w.abs().amax(dim=1)
        if b is not None:
            mags = torch.maximum(mags, b.detach().float().abs())
        return mags / limit
    return None


def _scale_columns(layer: QuantizableGemm, s: torch.Tensor) -> None:
    """``W[:, j] *= s_j`` on a consumer (in place, through its GEMM view)."""
    w = layer.gemm_weight().data
    w.mul_(s.to(w.dtype).reshape(1, -1))


def _divide(t: Optional[torch.Tensor], s: torch.Tensor) -> None:
    if t is not None:
        t.data.div_(s.to(t.dtype))


def _fold(
    group: SmoothGroup,
    s: torch.Tensor,
    modules: Dict[str, nn.Module],
    opts: SmoothQuantOptions,
    *,
    full_precision: Optional[AbstractSet[str]] = None,
) -> None:
    """Apply *group*'s fold with scale *s*, refusing a served tensor past fp16's range.

    What is checked is what is *stored* in the served dtype: a norm's affine, a GEMM
    producer's bias, and a GEMM producer's weight only when it stays full precision
    (*full_precision*; ``None`` checks every producer weight).  A quantized
    producer's weight becomes 8-bit values with fp32 per-row scales -- per-row
    quantization is blind to the row scaling this fold applies -- and a layer that
    is a consumer of one group and the producer of another has its weight scaled
    both ways (``W * s1 / s2``), which no single group's bound can see.
    """
    limit = opts.fp16_headroom * _FP16_MAX

    def weight_if_stored(path: Optional[str], w: torch.Tensor) -> List[Optional[torch.Tensor]]:
        return [w] if full_precision is None or path in full_precision else []

    consumers = [_gemm(modules, c) for c in group.consumers]
    if group.fold == "norm_affine":
        assert group.producer is not None
        norm = modules[group.producer]
        _divide(_param(norm, "weight"), s)
        _divide(_param(norm, "bias"), s)
        _check_range(group.producer, [_param(norm, "weight"), _param(norm, "bias")], limit)
        for layer in consumers:
            _scale_columns(layer, s)
    elif group.fold == "v_to_o":
        assert group.v_rows is not None
        r0, r1 = group.v_rows
        producer = _gemm(modules, group.producer)
        w = producer.gemm_weight().data
        w[r0:r1].div_(s.to(w.dtype).reshape(-1, 1))
        bias = _param(cast(nn.Module, producer), "bias")
        if bias is not None:
            bias.data[r0:r1].div_(s.to(bias.dtype))
        _check_range(
            group.producer,
            weight_if_stored(group.producer, w[r0:r1]) + [None if bias is None else bias[r0:r1]],
            limit,
        )
        s_out = s
        if group.heads is not None and group.heads[0] != group.heads[1]:
            s_out = _untie_heads(s, group.heads)
        for layer in consumers:
            _scale_columns(layer, s_out)
    elif group.fold == "producer_rows":
        producer = _gemm(modules, group.producer)
        w = producer.gemm_weight().data
        w.div_(s.to(w.dtype).reshape(-1, 1))
        bias = _param(cast(nn.Module, producer), "bias")
        _divide(bias, s)
        _check_range(group.producer, weight_if_stored(group.producer, w) + [bias], limit)
        for layer in consumers:
            _scale_columns(layer, s)
    elif group.fold == "runtime":
        inv = s.reciprocal().float()
        for layer in consumers:
            module = cast(nn.Module, layer)
            if "smooth_inv" in module._buffers:
                raise RuntimeError(f"{group.consumers} is smoothed twice")
            module.register_buffer("smooth_inv", inv.clone())
            _scale_columns(layer, s)
    else:  # pragma: no cover - validated in SmoothGroup
        raise ValueError(group.fold)


def _check_range(path: Optional[str], tensors: List[Optional[torch.Tensor]], limit: float) -> None:
    for t in tensors:
        if t is not None and float(t.detach().float().abs().max()) > limit:
            raise RuntimeError(
                f"SmoothQuant folded {path} past fp16's range; raise s_floor or lower alpha"
            )
