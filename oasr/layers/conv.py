# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Convolution kernel wrappers (PyTorch-style interface).

These classes live under ``oasr.layers`` to mirror structures like
`vllm.model_executor.layers` while providing a thin, torch.nn-like API.

Like every other member of the waist, each class here owns **two** paths and
picks per call via :func:`oasr.layers._backend.use_conv_kernel`: the OASR kernel
on CUDA fp16/bf16, ``torch.nn.functional`` otherwise.  The torch path is what
lets a convolutional front-end run under the fp32 CPU parity oracles — the
evidence every model in this repo is verified with — and it is why the layouts
below are translated rather than assumed:

* :class:`Conv2d` is **NHWC** with a KRSC weight (what the CUTLASS implicit GEMM
  wants), so the fallback permutes both into ``F.conv2d``'s NCHW/KCRS and back;
* :class:`Conv1d` is ``(B, T, C)`` with an ``(out, K, in)`` KSC weight, so the
  kernel path stays in the residual stream's layout and the fallback translates
  to ``F.conv1d``;
* :class:`DepthwiseConv1d` is ``(B, T, C)`` with a ``(K, 1, C)`` weight, so the
  fallback permutes into ``F.conv1d``'s ``(B, C, T)`` / ``(C, 1, K)``;
* :class:`PointwiseConv1d` is a 1x1 convolution, i.e. a GEMM over the channel
  axis, so the fallback is ``F.linear`` on the squeezed weight.

Grouped and depthwise :class:`Conv2d` calls use the direct NHWC kernel; dense
1x1 calls use the layout-equivalent GEMM fast path.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

import oasr

from ._backend import use_conv_kernel

#: Torch equivalent of each fused-epilogue activation id.  ``gelu`` maps to the
#: **tanh** approximation because that is what the CUDA epilogue implements
#: (``include/oasr/common/math.h``); using ``F.gelu``'s exact-erf default here
#: would make the two paths of :class:`Conv2dActivation` disagree, which is the
#: one thing this fallback must not do.
_TORCH_CONV_ACTIVATION = {
    "relu": F.relu,
    "swish": F.silu,
    "silu": F.silu,
    "gelu": lambda x: F.gelu(x, approximate="tanh"),
}

_TORCH_CONV1D_ACTIVATION: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "relu": F.relu,
    "swish": F.silu,
    "silu": F.silu,
    "gelu_tanh": lambda x: F.gelu(x, approximate="tanh"),
}

_FUSED_CONV1D_ACTIVATION = {
    "relu": "relu",
    "swish": "swish",
    "silu": "silu",
    "gelu_tanh": "gelu",
}


class Conv1d(nn.Module):
    """Dense cross-channel convolution over packed ``(B, T, C)`` tensors.

    The kernel stores weights as KSC ``(out_channels, kernel_size,
    in_channels)`` so both activations and filters are directly consumable by
    the height-one CUTLASS implicit-GEMM convolution.  Standard PyTorch
    ``(out, in, kernel)`` checkpoint tensors are transposed once by the load
    hook, never once per request.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if in_channels <= 0 or out_channels <= 0 or kernel_size <= 0:
            raise ValueError(
                "in_channels, out_channels, and kernel_size must be positive, got "
                f"{in_channels}, {out_channels}, {kernel_size}"
            )
        if padding < 0 or stride <= 0 or dilation <= 0:
            raise ValueError(
                f"padding must be non-negative and stride/dilation positive, got "
                f"{padding=}, {stride=}, {dilation=}"
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation

        self.weight = nn.Parameter(
            torch.empty(out_channels, kernel_size, in_channels, device=device, dtype=dtype)
        )
        nn.init.kaiming_uniform_(self.weight.view(out_channels, -1), a=math.sqrt(5))
        if bias:
            fan_in = in_channels * kernel_size
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            self.bias = nn.Parameter(torch.empty(out_channels, device=device, dtype=dtype))
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

    def _torch_forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = F.conv1d(
            x.transpose(1, 2),
            self.weight.permute(0, 2, 1),
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        return out.transpose(1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x: (B, T, in_channels) -> (B, T_out, out_channels)``."""
        if not use_conv_kernel(x):
            return self._torch_forward(x)
        out: torch.Tensor = oasr.conv1d(
            x,
            self.weight,
            self.bias,
            padding=self.padding,
            stride=self.stride,
            dilation=self.dilation,
        )
        return out

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Load standard ``nn.Conv1d`` OIK weights into native KSC layout."""
        weight_key = prefix + "weight"
        if weight_key in state_dict:
            w = state_dict[weight_key]
            if (
                isinstance(w, torch.Tensor)
                and w.ndim == 3
                and w.shape == (self.out_channels, self.in_channels, self.kernel_size)
                and w.shape != self.weight.shape
            ):
                state_dict[weight_key] = w.permute(0, 2, 1).contiguous()

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def extra_repr(self) -> str:
        return (
            f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self.padding}, dilation={self.dilation}, "
            f"bias={self.bias is not None}"
        )


class Conv1dActivation(Conv1d):
    """Dense BTC Conv1D with a fused activation epilogue.

    Exact-erf ``gelu`` is deliberately not accepted.  The shared OASR GELU
    epilogue is the tanh approximation, so callers must opt into it explicitly
    as ``gelu_tanh``; Whisper and Qwen2-Audio keep their exact ``F.gelu`` as a
    separate operation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        activation_type: str = "swish",
        device=None,
        dtype=None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        if activation_type not in _FUSED_CONV1D_ACTIVATION:
            raise ValueError(
                f"activation_type={activation_type!r} is not fusable; "
                f"expected one of {sorted(_FUSED_CONV1D_ACTIVATION)}"
            )
        self.activation_name = activation_type
        self.activation = oasr.get_activation_type_id(_FUSED_CONV1D_ACTIVATION[activation_type])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not use_conv_kernel(x):
            return _TORCH_CONV1D_ACTIVATION[self.activation_name](self._torch_forward(x))
        out: torch.Tensor = oasr.conv1d_activation(
            x,
            self.weight,
            self.bias,
            self.activation,
            padding=self.padding,
            stride=self.stride,
            dilation=self.dilation,
        )
        return out

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, activation={self.activation_name}"


class DepthwiseConv1d(nn.Module):
    """BTC depthwise convolution with asymmetric padding and fused FSMN masking."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        padding: int | tuple[int, int] = 0,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if channels <= 0 or kernel_size <= 0:
            raise ValueError(
                f"channels and kernel_size must be positive, got {channels=}, {kernel_size=}"
            )
        padding_pair = (padding, padding) if isinstance(padding, int) else tuple(padding)
        if len(padding_pair) != 2 or any(
            not isinstance(pad, int) or pad < 0 for pad in padding_pair
        ):
            raise ValueError(f"padding must be an int or non-negative (left, right), got {padding}")
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = padding_pair

        self.weight = nn.Parameter(
            torch.empty(kernel_size, 1, channels, device=device, dtype=dtype)
        )
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in = kernel_size
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            self.bias = nn.Parameter(torch.empty(channels, device=device, dtype=dtype))
            torch.nn.init.uniform_(self.bias, -bound, bound)
        else:
            # ``register_parameter``, not a plain ``None`` attribute:
            # ``load_state_dict`` reports an unexpected key either way, but only
            # the registered form keeps ``named_parameters()`` honest (pinned by
            # ``tests/test_layer_waist.py::test_bias_free_layers_register_bias_as_none``).
            self.register_parameter("bias", None)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        add_input: bool = False,
    ) -> torch.Tensor:
        """Apply depthwise convolution, optionally fusing the FSMN masked residual.

        With ``mask`` and ``add_input=True`` this computes exactly
        ``(conv(x * mask) + x * mask) * mask`` in one CUDA kernel.
        """
        if use_conv_kernel(x):
            return oasr.depthwise_conv1d(
                x,
                self.weight,
                self.bias,
                self.padding,
                mask=mask,
                add_input=add_input,
            )
        masked = x if mask is None else x * mask
        # (K, 1, C) -> (C, 1, K); (B, T, C) -> (B, C, T) and back.
        out: torch.Tensor = F.conv1d(
            F.pad(masked.transpose(1, 2), self.padding),
            self.weight.permute(2, 1, 0),
            self.bias,
            groups=self.channels,
        )
        out = out.transpose(1, 2).contiguous()
        if add_input:
            if out.shape != masked.shape:
                raise ValueError("add_input requires a length-preserving depthwise convolution")
            out = out + masked
        if mask is not None:
            if out.shape[:2] != mask.shape[:2]:
                raise ValueError("masking requires a length-preserving depthwise convolution")
            out = out * mask
        return out

    def extra_repr(self) -> str:
        return (
            f"{self.channels}, kernel_size={self.kernel_size}, padding={self.padding}, "
            f"bias={self.bias is not None}"
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Support loading WeNet Conv1d depthwise weights.

        WeNet depthwise conv uses PyTorch Conv1d layout [C, 1, K].
        OASR depthwise kernel expects [K, 1, C], so transpose on load.
        """
        weight_key = prefix + "weight"
        if weight_key in state_dict:
            w = state_dict[weight_key]
            # Detect WeNet-style layout and convert to OASR layout.
            if (
                isinstance(w, torch.Tensor)
                and w.ndim == 3
                and w.shape[1] == 1
                and w.shape[0] == self.channels
                and w.shape[2] == self.kernel_size
                and w.shape != self.weight.shape
            ):
                state_dict[weight_key] = w.permute(2, 1, 0).contiguous()

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class PointwiseConv1d(nn.Module):
    """Wrapper for pointwise (1x1) convolution kernel."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation_type: str | None = None,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation_name = activation_type
        self.activation = (
            None if activation_type is None else oasr.get_activation_type_id(activation_type)
        )

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, 1, device=device, dtype=dtype)
        )
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in = in_channels
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            self.bias = nn.Parameter(torch.empty(out_channels, device=device, dtype=dtype))
            torch.nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # weight: [out_channels, in_channels, 1] -> [out_channels, in_channels]
        # oasr.gemm accepts batched A (e.g. [batch, seq_len, in_channels]) and
        # preserves the leading dimensions in the output, so no manual reshape
        # is needed here.  This also picks up GEMM autotuning automatically.
        weight = self.weight.squeeze(-1)
        if not use_conv_kernel(x):
            dense: torch.Tensor = F.linear(x, weight, self.bias)
            if self.activation is None:
                return dense
            return _TORCH_CONV_ACTIVATION[str(self.activation_name)](dense)
        fused: torch.Tensor = (
            oasr.gemm_activation(x, weight, self.bias, self.activation)
            if self.activation is not None
            else oasr.gemm(x, weight, self.bias)
        )
        return fused


class Conv2d(nn.Module):
    """2D convolution backed by CUTLASS Ampere Tensor Core Implicit GEMM.

    Tensors use NHWC layout throughout:
      - input  [N, H, W, in_channels]
      - weight [out_channels, kernel_h, kernel_w, in_channels / groups]  (KRSC)
      - output [N, P, Q, out_channels]

    Alignment requirement (CUTLASS 128-bit loads):
      ``in_channels % 8 == 0``  and  ``out_channels % 8 == 0``.

    Supports FP16 and BF16 dtypes on the kernel path; anything else takes
    ``F.conv2d`` (see the module docstring).

    Grouped convolutions use a direct NHWC CUDA kernel.  Dense 1x1 convolutions
    are GEMMs over the channel axis and dispatch through the GEMM family.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        stride: int | tuple[int, int] = 1,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        kernel_h, kernel_w = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        )
        pad_h, pad_w = (padding, padding) if isinstance(padding, int) else tuple(padding)
        stride_h, stride_w = (stride, stride) if isinstance(stride, int) else tuple(stride)
        dilation_h, dilation_w = (
            (dilation, dilation) if isinstance(dilation, int) else tuple(dilation)
        )
        if groups < 1 or in_channels % groups or out_channels % groups:
            raise ValueError(
                f"groups={groups} must divide in_channels={in_channels} and "
                f"out_channels={out_channels}"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_h, kernel_w)
        self.padding = (pad_h, pad_w)
        self.stride = (stride_h, stride_w)
        self.dilation = (dilation_h, dilation_w)
        self.groups = groups

        # Weight stored as [K, R, S, IC/groups] (KRSC) for NHWC implicit GEMM.
        self.weight = nn.Parameter(
            torch.empty(
                out_channels,
                kernel_h,
                kernel_w,
                in_channels // groups,
                device=device,
                dtype=dtype,
            )
        )
        # Kaiming uniform with fan_in = IC/groups * R * S.
        # Viewing as [K, IC*R*S, 1] gives the correct fan_in to nn.init.
        nn.init.kaiming_uniform_(self.weight.view(out_channels, -1, 1), a=math.sqrt(5))

        if bias:
            fan_in = (in_channels // groups) * kernel_h * kernel_w
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            self.bias = nn.Parameter(torch.empty(out_channels, device=device, dtype=dtype))
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

    def _use_kernel(self, x: torch.Tensor) -> bool:
        return use_conv_kernel(x)

    def _torch_forward(
        self, x: torch.Tensor, padding: tuple[int, int] | None = None
    ) -> torch.Tensor:
        """NHWC/KRSC → ``F.conv2d``'s NCHW/KCRS and back."""
        out: torch.Tensor = F.conv2d(
            x.permute(0, 3, 1, 2),
            self.weight.permute(0, 3, 1, 2),
            self.bias,
            stride=self.stride,
            padding=self.padding if padding is None else padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return out.permute(0, 2, 3, 1).contiguous()

    def forward(self, x: torch.Tensor, *, padding: tuple[int, int] | None = None) -> torch.Tensor:
        """x: [N, H, W, in_channels] -> [N, P, Q, out_channels]."""
        if not self._use_kernel(x):
            return self._torch_forward(x, padding)
        pad_h, pad_w = self.padding if padding is None else padding
        out: torch.Tensor = oasr.conv2d(
            x,
            self.weight,
            self.bias,
            pad_h,
            pad_w,
            self.stride[0],
            self.stride[1],
            self.dilation[0],
            self.dilation[1],
            groups=self.groups,
        )
        return out

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Support loading standard PyTorch Conv2d weights.

        PyTorch nn.Conv2d stores weights as [K, IC/groups, R, S] (NCHW).
        OASR Conv2d expects [K, R, S, IC/groups] (NHWC), so permute on load.
        """
        weight_key = prefix + "weight"
        if weight_key in state_dict:
            w = state_dict[weight_key]
            if (
                isinstance(w, torch.Tensor)
                and w.ndim == 4
                and w.shape[0] == self.out_channels
                and w.shape[1] == self.in_channels // self.groups
                and w.shape[2:] == self.kernel_size
            ):
                # [K, IC, R, S] -> [K, R, S, IC]
                state_dict[weight_key] = w.permute(0, 2, 3, 1).contiguous()

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def extra_repr(self) -> str:
        return (
            f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self.padding}, groups={self.groups}, "
            f"bias={self.bias is not None}"
        )


class Conv2dActivation(Conv2d):
    """2D convolution with fused activation backed by CUTLASS Ampere Tensor Core.

    Computes ``output = activation(conv2d(input, weight) + bias)``.
    Same NHWC layout, alignment requirements, grouped kernel and torch
    fallback as :class:`Conv2d`, which it subclasses — the two used to carry
    identical ``__init__`` and ``_load_from_state_dict`` bodies, and the layout
    translation the fallback needs is worth having in exactly one place.

    Supported activations: ``"relu"``, ``"gelu"``, ``"swish"`` / ``"silu"``.
    Note ``"gelu"`` is the **tanh** approximation on both paths, matching the
    CUDA epilogue.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        stride: int | tuple[int, int] = 1,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        activation_type: str = "swish",
        device=None,
        dtype=None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        if activation_type not in _TORCH_CONV_ACTIVATION:
            raise ValueError(
                f"activation_type={activation_type!r} has no torch counterpart; "
                f"expected one of {sorted(_TORCH_CONV_ACTIVATION)}"
            )
        self.activation_name = activation_type
        self.activation = oasr.get_activation_type_id(activation_type)

    def forward(self, x: torch.Tensor, *, padding: tuple[int, int] | None = None) -> torch.Tensor:
        """x: [N, H, W, in_channels] -> [N, P, Q, out_channels]."""
        if not self._use_kernel(x):
            return _TORCH_CONV_ACTIVATION[self.activation_name](self._torch_forward(x, padding))
        pad_h, pad_w = self.padding if padding is None else padding
        out: torch.Tensor = oasr.conv2d_activation(
            x,
            self.weight,
            self.bias,
            self.activation,
            pad_h,
            pad_w,
            self.stride[0],
            self.stride[1],
            self.dilation[0],
            self.dilation[1],
            groups=self.groups,
        )
        return out

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, activation={self.activation_name}"


class Glu(nn.Module):
    """Gated linear unit over the last dimension: ``x[..., :C] * sigmoid(x[..., C:])``.

    Carries no parameters — it exists so a model can reach the ``oasr.glu``
    kernel *and* still run on CPU/fp32 without the call site re-deriving the
    backend decision.  ``F.glu(x, dim=-1)`` is the same function; the Conformer
    convolution module calls ``oasr.glu`` directly and predates this.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if use_conv_kernel(x):
            gated: torch.Tensor = oasr.glu(x)
            return gated
        return F.glu(x, dim=-1)


__all__ = [
    "Conv1d",
    "Conv1dActivation",
    "Conv2d",
    "Conv2dActivation",
    "DepthwiseConv1d",
    "Glu",
    "PointwiseConv1d",
]
