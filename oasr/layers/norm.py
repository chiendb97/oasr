# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Normalization — the waist's norm entry point.

Each module owns a CUDA-kernel path and a torch reference path and picks
between them in :mod:`oasr.layers._backend` (kernels are CUDA fp32/fp16/bf16;
CPU and everything else falls through to ``torch``).  Parameter names and
shapes are ``nn.*``-compatible so a migrated model keeps its checkpoint keys.

**Epsilon is an ecosystem property, not a default.**  Getting it wrong is a
silent accuracy bug rather than an error, and it has already cost this project
a debugging session (Paraformer parity breaks at PyTorch's 1e-5), so the
conventions are named here rather than rediscovered per model:

===================  =========  ===================================================
Constant             Value      Used by
===================  =========  ===================================================
``TORCH_EPS``        ``1e-5``   ``nn.LayerNorm``'s default; HF Whisper, Qwen2-Audio
                                tower, WeNet/ESPnet *transformer* blocks
``ESPNET_EPS``       ``1e-12``  ESPnet/FunASR ``LayerNorm`` — Paraformer's SANM
                                encoder, NAR decoder and CIF path
``QWEN2_RMS_EPS``    ``1e-6``   Qwen2 ``rms_norm_eps`` (the checkpoint's value
                                still wins; this is the family default)
===================  =========  ===================================================
"""

from __future__ import annotations

from typing import Optional, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

import oasr

from ._backend import use_norm_kernel

#: ``nn.LayerNorm``'s default epsilon.
TORCH_EPS = 1e-5
#: ESPnet / FunASR ``LayerNorm`` epsilon.  Paraformer needs this one.
ESPNET_EPS = 1e-12
#: Qwen2 ``rms_norm_eps`` family default.
QWEN2_RMS_EPS = 1e-6


def kernel_activation(x: torch.Tensor, name: str) -> torch.Tensor:
    """Torch spelling of an OASR fused-epilogue activation.

    ``gelu`` is the **tanh approximation** here, matching
    ``include/oasr/common/math.h``, *not* ``F.gelu``'s exact erf default.  Any
    reference path standing in for a fused kernel has to agree with the kernel,
    so the mapping lives in one place.
    """
    if name in ("swish", "silu"):
        return F.silu(x)
    if name == "relu":
        return F.relu(x)
    if name == "gelu":
        return F.gelu(x, approximate="tanh")
    raise ValueError(f"no torch equivalent registered for activation {name!r}")


def _batch_norm_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Inference BatchNorm over a channels-last input (the kernel's layout)."""
    out = (x - running_mean) * torch.rsqrt(running_var + eps) * weight
    return out if bias is None else out + bias


def _rms_norm_ref(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], eps: float
) -> torch.Tensor:
    """Torch spelling of OASR RMSNorm's fp32 accumulation and affine order."""
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * weight.float()
    if bias is not None:
        out = out + bias.float()
    return out.to(x.dtype)


def _add_layer_norm_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    eps: float,
    alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Torch spelling of fused add + LayerNorm with an fp32 sum."""
    summed = residual.float() + alpha * x.float()
    bias_float = None if bias is None else bias.float()
    normalized = F.layer_norm(summed, (x.size(-1),), weight.float(), bias_float, eps).to(x.dtype)
    return normalized, summed.to(x.dtype)


def _add_rms_norm_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    eps: float,
    alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Torch spelling of fused add + RMSNorm with an fp32 sum."""
    summed = residual.float() + alpha * x.float()
    normalized = _rms_norm_ref(summed, weight, bias, eps).to(x.dtype)
    return normalized, summed.to(x.dtype)


class LayerNorm(nn.Module):
    """LayerNorm over the last dimension.

    ``eps`` has no default on purpose-adjacent grounds: it *does* default to
    :data:`TORCH_EPS` for drop-in compatibility with ``nn.LayerNorm``, but see
    the module docstring before accepting it for a non-PyTorch ecosystem.
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = TORCH_EPS,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape, device=device, dtype=dtype))

        if bias:
            self.bias = nn.Parameter(torch.zeros(normalized_shape, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if use_norm_kernel(x):
            return oasr.layer_norm(x, self.weight, self.bias, self.eps)
        return F.layer_norm(x, (self.normalized_shape,), self.weight, self.bias, self.eps)

    def forward_add(
        self, x: torch.Tensor, residual: torch.Tensor, alpha: float = 1.0
    ) -> torch.Tensor:
        """Normalize ``residual + alpha * x`` with an fp32 fused sum."""
        if use_norm_kernel(x):
            return cast(
                torch.Tensor,
                oasr.add_layer_norm(x, residual, self.weight, self.bias, self.eps, alpha=alpha),
            )
        normalized, _ = _add_layer_norm_ref(x, residual, self.weight, self.bias, self.eps, alpha)
        return normalized

    def forward_add_residual(
        self, x: torch.Tensor, residual: torch.Tensor, alpha: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(LayerNorm(s), s)`` for ``s = residual + alpha * x``."""
        if use_norm_kernel(x):
            return cast(
                Tuple[torch.Tensor, torch.Tensor],
                oasr.add_layer_norm_residual(x, residual, self.weight, self.bias, self.eps, alpha),
            )
        return _add_layer_norm_ref(x, residual, self.weight, self.bias, self.eps, alpha)

    def extra_repr(self) -> str:
        return f"{self.normalized_shape}, eps={self.eps}, bias={self.bias is not None}"


class RMSNorm(nn.Module):
    """RMS normalization over the last dimension.

    ``out = x * rsqrt(mean(x²) + eps) * weight`` accumulated in **fp32** and
    rounded once, on the store.  HF's ``Qwen2RMSNorm`` instead rounds the
    normalized activation back to the input dtype *before* multiplying by
    ``weight``, so under bf16 the two differ by one rounding step (OASR's order
    is the more accurate one, and the one vLLM uses).  The torch path below
    mirrors the kernel, not HF, so switching backends never changes the answer;
    ``tests/test_speech_llm.py`` is the gate that the difference does not move
    a real checkpoint's tokens.
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = TORCH_EPS,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape, device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.zeros(normalized_shape, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if use_norm_kernel(x):
            return oasr.rms_norm(x, self.weight, self.bias, self.eps)
        return _rms_norm_ref(x, self.weight, self.bias, self.eps)

    def forward_add(
        self, x: torch.Tensor, residual: torch.Tensor, alpha: float = 1.0
    ) -> torch.Tensor:
        """Normalize ``residual + alpha * x`` with an fp32 fused sum."""
        if use_norm_kernel(x):
            return cast(
                torch.Tensor,
                oasr.add_rms_norm(x, residual, self.weight, self.bias, self.eps, alpha),
            )
        normalized, _ = _add_rms_norm_ref(x, residual, self.weight, self.bias, self.eps, alpha)
        return normalized

    def forward_add_residual(
        self, x: torch.Tensor, residual: torch.Tensor, alpha: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(RMSNorm(s), s)`` for ``s = residual + alpha * x``."""
        if use_norm_kernel(x):
            return cast(
                Tuple[torch.Tensor, torch.Tensor],
                oasr.add_rms_norm_residual(x, residual, self.weight, self.bias, self.eps, alpha),
            )
        return _add_rms_norm_ref(x, residual, self.weight, self.bias, self.eps, alpha)

    def extra_repr(self) -> str:
        return f"{self.normalized_shape}, eps={self.eps}, bias={self.bias is not None}"


class BiasNorm(nn.Module):
    """Zipformer BiasNorm: a cheap LayerNorm replacement.

    ``scales = mean((x - bias)**2, dim=channel_dim, keepdim=True)**-0.5 *
    exp(log_scale)``; ``output = x * scales``.  No eps term — matches icefall's
    inference-time BiasNorm.  Parameter names (``bias``, ``log_scale``) mirror
    icefall so a checkpoint loads 1:1.

    ``channel_dim`` exists because icefall's constructor takes it; only the
    last-dim case (the one the Zipformer encoder uses) has a kernel, and any
    other value takes the torch path.
    """

    def __init__(
        self,
        num_channels: int,
        channel_dim: int = -1,
        log_scale: float = 1.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.channel_dim = channel_dim
        self.log_scale = nn.Parameter(torch.tensor(log_scale, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(num_channels, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel_dim = self.channel_dim
        if channel_dim < 0:
            channel_dim += x.ndim
        if channel_dim == x.ndim - 1 and use_norm_kernel(x):
            return oasr.bias_norm(x, self.bias, self.log_scale)
        bias: torch.Tensor = self.bias
        for _ in range(channel_dim + 1, x.ndim):
            bias = bias.unsqueeze(-1)
        scales = (
            torch.mean((x - bias) ** 2, dim=channel_dim, keepdim=True) ** -0.5
        ) * self.log_scale.exp()
        return x * scales

    def extra_repr(self) -> str:
        return f"{self.num_channels}, channel_dim={self.channel_dim}"


class GroupNorm(nn.Module):
    """Wrapper for group normalization kernel."""

    def __init__(
        self,
        num_channels: int,
        num_groups: int,
        eps: float = 1e-5,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.num_groups = num_groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels, device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.zeros(num_channels, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if use_norm_kernel(x):
            return oasr.group_norm(x, self.weight, self.bias, self.num_groups, self.eps)
        # Channels are last here (the kernel's layout); ``F.group_norm`` wants
        # (N, C, *), so normalize over the grouped channel axis directly.
        shape = x.shape
        grouped = x.reshape(-1, self.num_groups, self.num_channels // self.num_groups)
        mean = grouped.mean(dim=-1, keepdim=True)
        var = grouped.var(dim=-1, unbiased=False, keepdim=True)
        out = ((grouped - mean) * torch.rsqrt(var + self.eps)).reshape(shape)
        out = out * self.weight
        return out if self.bias is None else out + self.bias


class BatchNorm1d(nn.Module):
    """Wrapper for 1D batch normalization kernel (inference).

    running_mean and running_var are registered as buffers (non-trainable).
    """

    running_mean: torch.Tensor
    running_var: torch.Tensor

    def __init__(
        self, num_channels: int, eps: float = 1e-5, bias: bool = True, device=None, dtype=None
    ):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels, device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.zeros(num_channels, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)
        self.register_buffer("running_mean", torch.zeros(num_channels, device=device, dtype=dtype))
        self.register_buffer("running_var", torch.ones(num_channels, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if use_norm_kernel(x):
            return oasr.batch_norm_1d(
                x, self.weight, self.bias, self.running_mean, self.running_var, self.eps
            )
        return _batch_norm_ref(
            x, self.weight, self.bias, self.running_mean, self.running_var, self.eps
        )


class BatchNormSwish(nn.Module):
    """Wrapper for fused BatchNorm + Swish kernel."""

    running_mean: torch.Tensor
    running_var: torch.Tensor

    def __init__(
        self, num_channels: int, eps: float = 1e-5, bias: bool = True, device=None, dtype=None
    ):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(num_channels, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(num_channels, device=device, dtype=dtype))
        self.register_buffer("running_mean", torch.zeros(num_channels, device=device, dtype=dtype))
        self.register_buffer("running_var", torch.ones(num_channels, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if use_norm_kernel(x):
            return oasr.batch_norm_swish(
                x, self.weight, self.bias, self.running_mean, self.running_var, self.eps
            )
        return F.silu(
            _batch_norm_ref(
                x, self.weight, self.bias, self.running_mean, self.running_var, self.eps
            )
        )


class AddLayerNorm(nn.Module):
    """Wrapper for fused add + layer norm: output = LayerNorm(x + residual)."""

    def __init__(
        self, normalized_shape: int, eps: float = 1e-5, bias: bool = True, device=None, dtype=None
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape, device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.zeros(normalized_shape, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        if use_norm_kernel(x):
            return cast(
                torch.Tensor,
                oasr.add_layer_norm(x, residual, self.weight, self.bias, self.eps, alpha=alpha),
            )
        normalized, _ = _add_layer_norm_ref(x, residual, self.weight, self.bias, self.eps, alpha)
        return normalized

    def forward_residual(
        self, x: torch.Tensor, residual: torch.Tensor, alpha: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if use_norm_kernel(x):
            return cast(
                Tuple[torch.Tensor, torch.Tensor],
                oasr.add_layer_norm_residual(x, residual, self.weight, self.bias, self.eps, alpha),
            )
        return _add_layer_norm_ref(x, residual, self.weight, self.bias, self.eps, alpha)


class AddRMSNorm(nn.Module):
    """Wrapper for fused add + RMSNorm, with optional residual passthrough."""

    def __init__(
        self, normalized_shape: int, eps: float = 1e-5, bias: bool = True, device=None, dtype=None
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape, device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.zeros(normalized_shape, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        if use_norm_kernel(x):
            return cast(
                torch.Tensor,
                oasr.add_rms_norm(x, residual, self.weight, self.bias, self.eps, alpha),
            )
        normalized, _ = _add_rms_norm_ref(x, residual, self.weight, self.bias, self.eps, alpha)
        return normalized

    def forward_residual(
        self, x: torch.Tensor, residual: torch.Tensor, alpha: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if use_norm_kernel(x):
            return cast(
                Tuple[torch.Tensor, torch.Tensor],
                oasr.add_rms_norm_residual(x, residual, self.weight, self.bias, self.eps, alpha),
            )
        return _add_rms_norm_ref(x, residual, self.weight, self.bias, self.eps, alpha)


class LayerNormActivation(nn.Module):
    """Fused LayerNorm + Activation: output = activation(LayerNorm(x))."""

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        bias: bool = True,
        activation: str = "swish",
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.activation_name = activation
        self.activation_type = oasr.get_activation_type_id(activation)
        self.weight = nn.Parameter(torch.ones(normalized_shape, device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.zeros(normalized_shape, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if use_norm_kernel(x):
            return oasr.layer_norm_activation(
                x, self.weight, self.bias, self.eps, self.activation_type
            )
        normed = F.layer_norm(x, (self.normalized_shape,), self.weight, self.bias, self.eps)
        return kernel_activation(normed, self.activation_name)


class RMSNormActivation(nn.Module):
    """Fused RMSNorm + Activation: output = activation(RMSNorm(x))."""

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        bias: bool = True,
        activation: str = "swish",
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.activation_name = activation
        self.activation_type = oasr.get_activation_type_id(activation)
        self.weight = nn.Parameter(torch.ones(normalized_shape, device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.zeros(normalized_shape, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if use_norm_kernel(x):
            return oasr.rms_norm_activation(
                x, self.weight, self.bias, self.eps, self.activation_type
            )
        return kernel_activation(
            _rms_norm_ref(x, self.weight, self.bias, self.eps), self.activation_name
        )


class BatchNormActivation(nn.Module):
    """Fused BatchNorm + Activation: output = activation(BatchNorm(x))."""

    running_mean: torch.Tensor
    running_var: torch.Tensor

    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-5,
        activation: str = "swish",
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.activation_name = activation
        self.activation_type = oasr.get_activation_type_id(activation)
        self.weight = nn.Parameter(torch.ones(num_channels, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(num_channels, device=device, dtype=dtype))
        self.register_buffer("running_mean", torch.zeros(num_channels, device=device, dtype=dtype))
        self.register_buffer("running_var", torch.ones(num_channels, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if use_norm_kernel(x):
            return oasr.batch_norm_activation(
                x,
                self.weight,
                self.bias,
                self.running_mean,
                self.running_var,
                self.eps,
                self.activation_type,
            )
        normed = _batch_norm_ref(
            x, self.weight, self.bias, self.running_mean, self.running_var, self.eps
        )
        return kernel_activation(normed, self.activation_name)


class GlobalCMVN(nn.Module):
    """Global cepstral mean and variance normalization.

    Stores pre-computed mean and inverse std-dev as buffers
    and applies ``oasr.cmvn(x, mean, istd)`` to input features.
    """

    mean: torch.Tensor
    istd: torch.Tensor

    def __init__(self, mean: torch.Tensor, istd: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("istd", istd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if use_norm_kernel(x):
            return oasr.cmvn(x, self.mean, self.istd)
        return (x - self.mean) * self.istd


__all__ = [
    "ESPNET_EPS",
    "QWEN2_RMS_EPS",
    "TORCH_EPS",
    "kernel_activation",
    "LayerNorm",
    "RMSNorm",
    "GroupNorm",
    "BiasNorm",
    "BatchNorm1d",
    "BatchNormSwish",
    "AddLayerNorm",
    "AddRMSNorm",
    "LayerNormActivation",
    "RMSNormActivation",
    "BatchNormActivation",
    "GlobalCMVN",
]
