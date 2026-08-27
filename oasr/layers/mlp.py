# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Shared feed-forward blocks with configurable checkpoint key names.

Names remain configurable to preserve source state-dict layouts. ``gelu`` and
``gelu_tanh`` stay distinct because they select numerically different epilogues.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple, cast

import torch
import torch.nn.functional as F
from torch import nn

import oasr

from ._backend import use_gated_mlp_kernel
from .linear import ColumnParallelLinear, LinearActivation, RowParallelLinear
from .norm import LayerNorm

#: Activations that fold into the GEMM epilogue (see ``LinearActivation``).
_FUSABLE = frozenset({"relu", "swish", "silu", "gelu", "gelu_tanh"})

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

    Gate and up stay two separate **parameters** rather than one merged
    column-parallel GEMM: HF checkpoints ship them apart, and fusing the
    *storage* would put a name-mapping step back into ``load_weights``.
    ``MergedColumnParallelLinear`` is the natural home for that when a converter
    can pay for the concatenation at load time.

    They do not stay two separate **kernels**.  With ``fuse_gate_up`` (the
    default) both projections, the gate activation and the multiply run as one
    ``oasr.gated_mlp`` launch that never materializes either ``(M, hidden)``
    intermediate — a dual-B tensor-core GEMM sharing one A tile, which needs
    nothing from the checkpoint because it reads ``(out, in)`` weights where
    they lie.  Whether it is taken is a measured, shape-only decision
    (:mod:`oasr.jit.mlp`): it is ahead from one row up to ~128 and
    behind above that, where the block stops being bandwidth bound and a library
    GEMM's tiling freedom wins.  Outside the band the two-GEMM path below runs,
    and that path is itself fully kernel-backed — declining the fusion is a
    performance route, never a kernel gap.
    """

    def __init__(
        self,
        d_model: int,
        hidden: int,
        *,
        activation: str = "silu",
        bias: bool = False,
        names: Tuple[str, str, str] = ("gate_proj", "up_proj", "down_proj"),
        fuse_gate_up: bool = True,
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
        #: Offer the whole gate/up/act/multiply to one kernel.  Per *call*, the
        #: shape still decides; this only says whether to ask.
        self.fuse_gate_up = bool(fuse_gate_up)

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
        gate_proj, up_proj = self._proj(0), self._proj(1)
        # ``_proj`` resolves through ``_modules``, so the parameter types are
        # only known to be ``Tensor | Module``; the projections are always
        # ``Linear``-shaped by construction.
        w_gate = cast(torch.Tensor, gate_proj.weight)
        w_up = cast(torch.Tensor, up_proj.weight)
        b_gate = cast(Optional[torch.Tensor], gate_proj.bias)
        b_up = cast(Optional[torch.Tensor], up_proj.bias)
        if self.fuse_gate_up and use_gated_mlp_kernel(
            x, w_gate, activation=self.activation, has_bias=b_gate is not None
        ):
            h = oasr.gated_mlp(x, w_gate, w_up, b_gate, b_up, activation=self.activation)
        else:
            gate: torch.Tensor = gate_proj(x)
            if not self.fused:
                gate = _UNFUSED_ACTIVATION[self.activation](gate)
            h = gate * up_proj(x)
        out: torch.Tensor = self._proj(2)(h)
        return out

    def extra_repr(self) -> str:
        return (
            f"activation={self.activation}, fused={self.fused}, "
            f"fuse_gate_up={self.fuse_gate_up}"
        )


__all__ = ["FeedForward", "GatedMLP"]
