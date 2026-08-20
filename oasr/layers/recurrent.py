# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""LSTM and vanilla RNN layers built on OASR's fused recurrent kernels.

The parameter names, shapes, gate order, state convention, and forward return
types match PyTorch's unidirectional ``nn.LSTM`` / ``nn.RNN``.  The modules are
implemented from first principles: CPU/fp32 evaluates the equations with
``F.linear`` and CUDA FP16/BF16 uses one fused affine+activation+state kernel
per layer and timestep.  No framework recurrent module is wrapped internally.
"""

from __future__ import annotations

import math
import numbers
import warnings
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, cast

import torch
import torch.nn.functional as F
from torch import nn

import oasr

from ._backend import use_recurrent_kernel

# Scalar fused GEMV wins decode-sized cohorts because it avoids packing and
# intermediate gates.  At 16+ rows tensor cores amortize those costs and avoid
# re-reading every recurrent weight once per batch row.
_TENSOR_CORE_BATCH_FLOOR = 16


@lru_cache(maxsize=None)
def _supports_recurrent_tensor_core(device_index: int) -> bool:
    """Should this device's LSTM/RNN reach for the tensor-core path *by default*?

    The kernels themselves go back to SM75 (`recurrent_cutlass.cuh` maps Turing
    onto its own two-stage `m16n8k8` composition), but the crossover this gate
    encodes was measured on Ampere and later. Turing's narrower MMA moves that
    crossover by an unmeasured amount, so it is not selected automatically —
    `oasr.lstm_gemm_layer` / `oasr.rnn_gemm_layer` and the autotuner still reach
    it there.
    """
    major, _ = torch.cuda.get_device_capability(device_index)
    return major >= 8


def _validate_common(
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    bidirectional: bool,
    proj_size: int,
) -> None:
    if input_size <= 0:
        raise ValueError(f"input_size must be greater than zero, got {input_size}")
    if hidden_size <= 0:
        raise ValueError(f"hidden_size must be greater than zero, got {hidden_size}")
    if num_layers <= 0:
        raise ValueError(f"num_layers must be greater than zero, got {num_layers}")
    if not isinstance(dropout, numbers.Number) or not 0 <= float(dropout) <= 1:
        raise ValueError(f"dropout must be a number in [0, 1], got {dropout!r}")
    if dropout > 0 and num_layers == 1:
        warnings.warn(
            "dropout adds dropout after all but the last recurrent layer, so non-zero "
            "dropout expects num_layers greater than 1",
            stacklevel=3,
        )
    if bidirectional:
        raise ValueError("bidirectional recurrent layers are not implemented")
    if proj_size:
        raise ValueError("projected LSTM state is not implemented; proj_size must be 0")


class _RecurrentBase(nn.Module):
    _gate_count: int

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        bias: bool,
        batch_first: bool,
        dropout: float,
        bidirectional: bool,
        proj_size: int,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        _validate_common(input_size, hidden_size, num_layers, dropout, bidirectional, proj_size)
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.bias = bool(bias)
        self.batch_first = bool(batch_first)
        self.dropout = float(dropout)
        self.bidirectional = False
        self.proj_size = 0
        self._packed_parameter_cache: Dict[
            Tuple[int, Tuple[Tuple[int, int], ...]],
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
        ] = {}

        factory_kwargs = {"device": device, "dtype": dtype}
        self._flat_weights_names: List[str] = []
        self._all_weights: List[List[str]] = []
        for layer in range(self.num_layers):
            layer_input_size = self.input_size if layer == 0 else self.hidden_size
            names = [f"weight_ih_l{layer}", f"weight_hh_l{layer}"]
            setattr(
                self,
                names[0],
                nn.Parameter(
                    torch.empty(
                        self._gate_count * self.hidden_size,
                        layer_input_size,
                        **factory_kwargs,
                    )
                ),
            )
            setattr(
                self,
                names[1],
                nn.Parameter(
                    torch.empty(
                        self._gate_count * self.hidden_size,
                        self.hidden_size,
                        **factory_kwargs,
                    )
                ),
            )
            if self.bias:
                names.extend([f"bias_ih_l{layer}", f"bias_hh_l{layer}"])
                setattr(
                    self,
                    names[2],
                    nn.Parameter(
                        torch.empty(self._gate_count * self.hidden_size, **factory_kwargs)
                    ),
                )
                setattr(
                    self,
                    names[3],
                    nn.Parameter(
                        torch.empty(self._gate_count * self.hidden_size, **factory_kwargs)
                    ),
                )
            self._all_weights.append(names)
            self._flat_weights_names.extend(names)
        self.reset_parameters()

    @property
    def all_weights(self) -> List[List[torch.Tensor]]:
        return [[getattr(self, name) for name in names] for names in self._all_weights]

    @property
    def _flat_weights(self) -> List[torch.Tensor]:
        return [getattr(self, name) for name in self._flat_weights_names]

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)
        self._packed_parameter_cache.clear()

    def flatten_parameters(self) -> None:
        """Compatibility no-op; OASR consumes the checkpoint layout directly."""

    def _apply(self, fn, recurse: bool = True):
        # Packed tensors are derived, not parameters or buffers, so Module._apply
        # cannot migrate them. Rebuild lazily after .to()/.cuda() instead.
        self._packed_parameter_cache.clear()
        return super()._apply(fn, recurse)

    def _biases(self, layer: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.bias:
            return None, None
        return getattr(self, f"bias_ih_l{layer}"), getattr(self, f"bias_hh_l{layer}")

    @staticmethod
    def _parameter_signature(*parameters: Optional[torch.Tensor]) -> Tuple[Tuple[int, int], ...]:
        return tuple(
            (-1, -1) if parameter is None else (parameter.data_ptr(), parameter._version)
            for parameter in parameters
        )

    def _packed_lstm_parameters(
        self, layer: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        weight_ih = getattr(self, f"weight_ih_l{layer}")
        weight_hh = getattr(self, f"weight_hh_l{layer}")
        bias_ih, bias_hh = self._biases(layer)
        signature = self._parameter_signature(weight_ih, weight_hh, bias_ih, bias_hh)
        key = (layer, signature)
        cached = self._packed_parameter_cache.get(key)
        if cached is not None:
            return cached

        hidden_size = self.hidden_size

        def pack_weight(weight: torch.Tensor) -> torch.Tensor:
            return (
                weight.detach()
                .reshape(4, hidden_size, weight.shape[1])
                .permute(1, 0, 2)
                .reshape(4 * hidden_size, weight.shape[1])
                .contiguous()
            )

        combined_bias = None
        if bias_ih is not None:
            combined_bias = (
                (bias_ih.detach() + cast(torch.Tensor, bias_hh).detach())
                .reshape(4, hidden_size)
                .transpose(0, 1)
                .contiguous()
                .reshape(-1)
            )
        packed = (pack_weight(weight_ih), pack_weight(weight_hh), combined_bias)
        # A parameter update changes the signature.  Retain only the current
        # version for this layer so repeated optimizer steps cannot grow VRAM.
        self._packed_parameter_cache = {
            cache_key: value
            for cache_key, value in self._packed_parameter_cache.items()
            if cache_key[0] != layer
        }
        self._packed_parameter_cache[key] = packed
        return packed

    def _combined_rnn_bias(self, layer: int) -> Optional[torch.Tensor]:
        bias_ih, bias_hh = self._biases(layer)
        if bias_ih is None:
            return None
        signature = self._parameter_signature(bias_ih, bias_hh)
        key = (layer, signature)
        cached = self._packed_parameter_cache.get(key)
        if cached is not None:
            return cached[2]
        combined = bias_ih.detach() + cast(torch.Tensor, bias_hh).detach()
        packed = (combined, combined, combined)
        self._packed_parameter_cache = {
            cache_key: value
            for cache_key, value in self._packed_parameter_cache.items()
            if cache_key[0] != layer
        }
        self._packed_parameter_cache[key] = packed
        return combined

    def _normalize_input(self, input: torch.Tensor) -> Tuple[torch.Tensor, int, bool, bool]:
        if input.dim() not in (2, 3):
            raise ValueError(f"recurrent input must be 2-D or 3-D, got {input.dim()}-D tensor")
        unbatched = input.dim() == 2
        if input.shape[-1] != self.input_size:
            raise RuntimeError(
                f"input.size(-1) must equal input_size. Expected {self.input_size}, "
                f"got {input.shape[-1]}"
            )
        # The launcher takes strided batch/time rows but requires the tensor
        # itself to be contiguous, while nn.LSTM accepts any view.  Materialize
        # here so a BTC view of a TBC tensor behaves the same as it does in torch.
        input = input.contiguous()
        if unbatched:
            # Unbatched input is always (T, C); batch_first has no effect.
            return input.unsqueeze(1), 1, True, False
        batch_size = input.shape[0 if self.batch_first else 1]
        return input, batch_size, False, self.batch_first

    def _normalize_h(
        self,
        state: Optional[torch.Tensor],
        input: torch.Tensor,
        batch_size: int,
        unbatched: bool,
        name: str,
    ) -> torch.Tensor:
        expected = (
            (self.num_layers, self.hidden_size)
            if unbatched
            else (self.num_layers, batch_size, self.hidden_size)
        )
        if state is None:
            return input.new_zeros(self.num_layers, batch_size, self.hidden_size)
        if tuple(state.shape) != expected:
            raise RuntimeError(f"Expected {name} size {expected}, got {tuple(state.shape)}")
        state = state.unsqueeze(1) if unbatched else state
        # Each per-layer slice reaches the launcher directly, so the stack has to
        # be contiguous for `state[layer]` to be.  A state threaded through
        # `unstack_states` is a non-contiguous view of a wider batch.
        return state.contiguous()

    def _drop_between_layers(self, output: torch.Tensor, layer: int) -> torch.Tensor:
        if self.dropout and self.training and layer + 1 < self.num_layers:
            return F.dropout(output, p=self.dropout, training=True)
        return output

    def extra_repr(self) -> str:
        fields = [str(self.input_size), str(self.hidden_size)]
        if self.num_layers != 1:
            fields.append(f"num_layers={self.num_layers}")
        if not self.bias:
            fields.append("bias=False")
        if self.batch_first:
            fields.append("batch_first=True")
        if self.dropout:
            fields.append(f"dropout={self.dropout}")
        return ", ".join(fields)


class LSTM(_RecurrentBase):
    """A unidirectional multi-layer LSTM with PyTorch-compatible parameters."""

    _gate_count = 4

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        proj_size: int = 0,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout,
            bidirectional,
            proj_size,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def _torch_layer(
        input: torch.Tensor,
        initial_h: torch.Tensor,
        initial_c: torch.Tensor,
        weight_ih: torch.Tensor,
        weight_hh: torch.Tensor,
        bias_ih: Optional[torch.Tensor],
        bias_hh: Optional[torch.Tensor],
        batch_first: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence = input.transpose(0, 1) if batch_first else input
        hidden, cell = initial_h, initial_c
        outputs = []
        for timestep in range(sequence.shape[0]):
            gates = F.linear(sequence[timestep], weight_ih, bias_ih)
            gates = gates + F.linear(hidden, weight_hh, bias_hh)
            input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, dim=-1)
            input_gate = torch.sigmoid(input_gate)
            forget_gate = torch.sigmoid(forget_gate)
            cell_gate = torch.tanh(cell_gate)
            output_gate = torch.sigmoid(output_gate)
            cell = forget_gate * cell + input_gate * cell_gate
            hidden = output_gate * torch.tanh(cell)
            outputs.append(hidden)
        output = torch.stack(outputs)
        if batch_first:
            output = output.transpose(0, 1)
        return output, hidden, cell

    def forward(
        self,
        input: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        input, batch_size, unbatched, kernel_batch_first = self._normalize_input(input)
        if hx is None:
            initial_h = self._normalize_h(None, input, batch_size, unbatched, "hidden[0]")
            initial_c = self._normalize_h(None, input, batch_size, unbatched, "hidden[1]")
        else:
            if len(hx) != 2:
                raise RuntimeError("hx for LSTM must be a (hidden, cell) pair")
            initial_h = self._normalize_h(hx[0], input, batch_size, unbatched, "hidden[0]")
            initial_c = self._normalize_h(hx[1], input, batch_size, unbatched, "hidden[1]")

        output = input
        kernel = use_recurrent_kernel(input)
        tensor_core = (
            kernel
            and _supports_recurrent_tensor_core(input.device.index or 0)
            and batch_size >= _TENSOR_CORE_BATCH_FLOOR
            and self.hidden_size >= 1024
            and self.input_size % 8 == 0
            and self.hidden_size % 8 == 0
        )
        current_batch_first = kernel_batch_first
        final_h: List[torch.Tensor] = []
        final_c: List[torch.Tensor] = []
        kernel_h = (
            input.new_empty(self.num_layers, batch_size, self.hidden_size) if kernel else None
        )
        kernel_c = (
            input.new_empty(self.num_layers, batch_size, self.hidden_size) if kernel else None
        )
        for layer in range(self.num_layers):
            weight_ih = getattr(self, f"weight_ih_l{layer}")
            weight_hh = getattr(self, f"weight_hh_l{layer}")
            bias_ih, bias_hh = self._biases(layer)
            if tensor_core:
                tensor_h = cast(torch.Tensor, kernel_h)
                tensor_c = cast(torch.Tensor, kernel_c)
                output, hidden, cell = oasr.lstm_gemm_layer(
                    output,
                    initial_h[layer],
                    initial_c[layer],
                    weight_ih,
                    weight_hh,
                    bias_ih,
                    bias_hh,
                    batch_first=current_batch_first,
                    final_h=tensor_h[layer],
                    final_c=tensor_c[layer],
                    _packed_parameters=self._packed_lstm_parameters(layer),
                )
                current_batch_first = False
            elif kernel:
                tensor_h = cast(torch.Tensor, kernel_h)
                tensor_c = cast(torch.Tensor, kernel_c)
                output, hidden, cell = oasr.lstm_layer(
                    output,
                    initial_h[layer],
                    initial_c[layer],
                    weight_ih,
                    weight_hh,
                    bias_ih,
                    bias_hh,
                    batch_first=kernel_batch_first,
                    final_h=tensor_h[layer],
                    final_c=tensor_c[layer],
                )
            else:
                output, hidden, cell = self._torch_layer(
                    output,
                    initial_h[layer],
                    initial_c[layer],
                    weight_ih,
                    weight_hh,
                    bias_ih,
                    bias_hh,
                    kernel_batch_first,
                )
            if not kernel:
                final_h.append(hidden)
                final_c.append(cell)
            output = self._drop_between_layers(output, layer)

        h_n = cast(torch.Tensor, kernel_h) if kernel else torch.stack(final_h)
        c_n = cast(torch.Tensor, kernel_c) if kernel else torch.stack(final_c)
        if tensor_core and self.batch_first:
            output = output.transpose(0, 1)
        if unbatched:
            return output.squeeze(1), (h_n.squeeze(1), c_n.squeeze(1))
        return output, (h_n, c_n)


class RNN(_RecurrentBase):
    """A unidirectional multi-layer Elman RNN (tanh or ReLU)."""

    _gate_count = 1

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: str = "tanh",
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        if nonlinearity not in ("tanh", "relu"):
            raise ValueError(
                f"Unknown nonlinearity {nonlinearity!r}. Select from 'tanh' or 'relu'."
            )
        super().__init__(
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout,
            bidirectional,
            0,
            device=device,
            dtype=dtype,
        )
        self.nonlinearity = nonlinearity

    def _torch_layer(
        self,
        input: torch.Tensor,
        initial_h: torch.Tensor,
        weight_ih: torch.Tensor,
        weight_hh: torch.Tensor,
        bias_ih: Optional[torch.Tensor],
        bias_hh: Optional[torch.Tensor],
        batch_first: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = input.transpose(0, 1) if batch_first else input
        hidden = initial_h
        outputs = []
        activation = torch.tanh if self.nonlinearity == "tanh" else torch.relu
        for timestep in range(sequence.shape[0]):
            hidden = activation(
                F.linear(sequence[timestep], weight_ih, bias_ih)
                + F.linear(hidden, weight_hh, bias_hh)
            )
            outputs.append(hidden)
        output = torch.stack(outputs)
        if batch_first:
            output = output.transpose(0, 1)
        return output, hidden

    def forward(
        self, input: torch.Tensor, hx: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input, batch_size, unbatched, kernel_batch_first = self._normalize_input(input)
        initial_h = self._normalize_h(hx, input, batch_size, unbatched, "hidden")
        output = input
        kernel = use_recurrent_kernel(input)
        tensor_core = (
            kernel
            and _supports_recurrent_tensor_core(input.device.index or 0)
            and batch_size >= _TENSOR_CORE_BATCH_FLOOR
            and self.hidden_size >= 1024
            and self.input_size % 8 == 0
            and self.hidden_size % 8 == 0
        )
        current_batch_first = kernel_batch_first
        final_h: List[torch.Tensor] = []
        kernel_h = (
            input.new_empty(self.num_layers, batch_size, self.hidden_size) if kernel else None
        )
        for layer in range(self.num_layers):
            weight_ih = getattr(self, f"weight_ih_l{layer}")
            weight_hh = getattr(self, f"weight_hh_l{layer}")
            bias_ih, bias_hh = self._biases(layer)
            if tensor_core:
                tensor_h = cast(torch.Tensor, kernel_h)
                output, hidden = oasr.rnn_gemm_layer(
                    output,
                    initial_h[layer],
                    weight_ih,
                    weight_hh,
                    bias_ih,
                    bias_hh,
                    nonlinearity=self.nonlinearity,
                    batch_first=current_batch_first,
                    final_h=tensor_h[layer],
                    _combined_input_bias=self._combined_rnn_bias(layer),
                )
                current_batch_first = False
            elif kernel:
                tensor_h = cast(torch.Tensor, kernel_h)
                output, hidden = oasr.rnn_layer(
                    output,
                    initial_h[layer],
                    weight_ih,
                    weight_hh,
                    bias_ih,
                    bias_hh,
                    nonlinearity=self.nonlinearity,
                    batch_first=kernel_batch_first,
                    final_h=tensor_h[layer],
                )
            else:
                output, hidden = self._torch_layer(
                    output,
                    initial_h[layer],
                    weight_ih,
                    weight_hh,
                    bias_ih,
                    bias_hh,
                    kernel_batch_first,
                )
            if not kernel:
                final_h.append(hidden)
            output = self._drop_between_layers(output, layer)

        h_n = cast(torch.Tensor, kernel_h) if kernel else torch.stack(final_h)
        if tensor_core and self.batch_first:
            output = output.transpose(0, 1)
        if unbatched:
            return output.squeeze(1), h_n.squeeze(1)
        return output, h_n

    def extra_repr(self) -> str:
        base = super().extra_repr()
        if self.nonlinearity == "relu":
            base += ", nonlinearity='relu'"
        return base


__all__ = ["LSTM", "RNN"]
