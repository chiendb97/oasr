# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Position-wise feed-forward blocks — the waist's MLP entry point.

Six models each carried their own two-linears-and-an-activation class
(``fc1``/``fc2``, ``w_1``/``w_2``, ``gate``/``up``/``down``), differing only in
the projection *names* and the activation.  What is worth sharing is the
fusion decision — whether the activation folds into the GEMM epilogue — and
that is what these classes own.

Projection names stay configurable rather than canonicalised.  Renaming them
would rewrite every migrated checkpoint's key layout for cosmetic gain, and
"state-dict keys mirror upstream 1:1" is a property this project has paid to
keep (see ``docs/checkpoints.md``).

For the same reason these classes are used only where the upstream layout
*already* nests the FFN under a name (WeNet/FunASR ``feed_forward.w_1``, HF
``mlp.gate_proj``).  HF Whisper and the Qwen2-Audio tower put ``fc1``/``fc2``
flat on the layer, so composing a ``FeedForward`` there would insert a level
into every checkpoint key to save one line of ``F.gelu``; those two keep a pair
of :class:`~oasr.layers.linear.Linear`s.  That is the intended outcome, not an
unfinished migration.

Activation names follow :mod:`oasr.layers.linear`: ``gelu`` is the **exact
erf** form and is never fused, ``gelu_tanh`` is the approximation the CUDA
epilogue implements.  Asking for the wrong one is a silent accuracy change on
an erf-trained model, so the two are separate names rather than one flag.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .linear import ColumnParallelLinear, LinearActivation, RowParallelLinear
from .norm import LayerNorm

#: Activations that fold into the GEMM epilogue (see ``LinearActivation``).
_FUSABLE = frozenset({"relu", "swish", "silu", "gelu_tanh"})

_UNFUSED_ACTIVATION: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "relu": F.relu,
    "swish": F.silu,
    "silu": F.silu,
    "gelu": F.gelu,
    "gelu_tanh": lambda x: F.gelu(x, approximate="tanh"),
    "identity": lambda x: x,
}


class FeedForward(nn.Module):
    """``w_2(activation(w_1(x)))``, optionally with a norm between them.

    Parameters
    ----------
    d_model, hidden : int
        Outer and inner widths.
    activation : str
        One of :data:`_UNFUSED_ACTIVATION`'s keys.  A fusable one is folded
        into the first GEMM's epilogue when no ``inner_norm_eps`` is set.
    bias, out_bias : bool
        Bias on the first / second projection.  ``out_bias`` defaults to
        ``bias``; FunASR's decoder FFN has a bias-free ``w_2``.
    names : tuple[str, str]
        Attribute (and therefore checkpoint-key) names of the two projections.
    inner_norm_eps : float, optional
        When set, a :class:`~oasr.layers.norm.LayerNorm` named ``norm`` is
        applied between the activation and ``w_2`` — FunASR's
        ``PositionwiseFeedForwardDecoderSANM``.  Its presence rules out
        epilogue fusion, since the norm has to see the activated values.
    """

    def __init__(
        self,
        d_model: int,
        hidden: int,
        *,
        activation: str = "relu",
        bias: bool = True,
        out_bias: Optional[bool] = None,
        names: Tuple[str, str] = ("w_1", "w_2"),
        inner_norm_eps: Optional[float] = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if activation not in _UNFUSED_ACTIVATION:
            raise ValueError(
                f"activation={activation!r} not known; expected one of "
                f"{sorted(_UNFUSED_ACTIVATION)}"
            )
        self.activation = activation
        self._names = names
        self.fused = activation in _FUSABLE and inner_norm_eps is None

        first_cls = LinearActivation if self.fused else ColumnParallelLinear
        first_kwargs = {"activation_type": activation} if self.fused else {}
        self.add_module(
            names[0],
            first_cls(d_model, hidden, bias=bias, device=device, dtype=dtype, **first_kwargs),
        )
        self.norm = (
            LayerNorm(hidden, eps=inner_norm_eps, device=device, dtype=dtype)
            if inner_norm_eps is not None
            else None
        )
        self.add_module(
            names[1],
            RowParallelLinear(
                hidden,
                d_model,
                bias=bias if out_bias is None else out_bias,
                device=device,
                dtype=dtype,
            ),
        )

    def _proj(self, which: int) -> nn.Module:
        """Resolve one of the two projections by its configured name.

        They are registered under the checkpoint's names rather than fixed
        attributes, so the lookup goes through ``_modules``.
        """
        return self._modules[self._names[which]]  # type: ignore[return-value]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h: torch.Tensor = self._proj(0)(x)
        if not self.fused:
            h = _UNFUSED_ACTIVATION[self.activation](h)
        if self.norm is not None:
            h = self.norm(h)
        out: torch.Tensor = self._proj(1)(h)
        return out

    def extra_repr(self) -> str:
        return f"activation={self.activation}, fused={self.fused}"


class GatedMLP(nn.Module):
    """``down(activation(gate(x)) * up(x))`` — the SwiGLU/GeGLU LLM block.

    Gate and up stay two separate projections rather than one merged
    column-parallel GEMM: HF checkpoints ship them apart, and fusing them would
    put a name-mapping step back into ``load_weights`` for a single GEMM launch
    saved.  ``MergedColumnParallelLinear`` is the natural home for that
    optimization when a converter can pay for the concatenation at load time.
    """

    def __init__(
        self,
        d_model: int,
        hidden: int,
        *,
        activation: str = "silu",
        bias: bool = False,
        names: Tuple[str, str, str] = ("gate_proj", "up_proj", "down_proj"),
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if activation not in _UNFUSED_ACTIVATION:
            raise ValueError(
                f"activation={activation!r} not known; expected one of "
                f"{sorted(_UNFUSED_ACTIVATION)}"
            )
        self.activation = activation
        self._names = names
        self.fused = activation in _FUSABLE

        gate_cls = LinearActivation if self.fused else ColumnParallelLinear
        gate_kwargs = {"activation_type": activation} if self.fused else {}
        self.add_module(
            names[0],
            gate_cls(d_model, hidden, bias=bias, device=device, dtype=dtype, **gate_kwargs),
        )
        self.add_module(
            names[1], ColumnParallelLinear(d_model, hidden, bias=bias, device=device, dtype=dtype)
        )
        self.add_module(
            names[2], RowParallelLinear(hidden, d_model, bias=bias, device=device, dtype=dtype)
        )

    def _proj(self, which: int) -> nn.Module:
        """Resolve gate / up / down by its configured name (see
        :meth:`FeedForward._proj`)."""
        return self._modules[self._names[which]]  # type: ignore[return-value]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate: torch.Tensor = self._proj(0)(x)
        if not self.fused:
            gate = _UNFUSED_ACTIVATION[self.activation](gate)
        out: torch.Tensor = self._proj(2)(gate * self._proj(1)(x))
        return out

    def extra_repr(self) -> str:
        return f"activation={self.activation}, fused={self.fused}"


__all__ = ["FeedForward", "GatedMLP"]
