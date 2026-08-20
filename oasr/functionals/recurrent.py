# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Functional API for fused recurrent layers."""

from __future__ import annotations

import functools
from typing import Optional, Tuple

import torch

from oasr.api_logging import oasr_api
from oasr.tune import OpKey, get_tuner, is_tuning_enabled

# Private Python/CUDA launcher contract.  The ordinary path uses a cheap
# shape heuristic; OASR's existing autotuner can replace it per measured shape.
_FUSED_16X64 = 0
_FUSED_32X64 = 1
_FUSED_64X64 = 2


@functools.cache
def _get_recurrent_module():
    from oasr.jit.recurrent import gen_recurrent_module

    return gen_recurrent_module().build_and_load()


def _check_layer_inputs(
    input: torch.Tensor,
    initial_h: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: Optional[torch.Tensor],
    bias_hh: Optional[torch.Tensor],
    gate_count: int,
    batch_first: bool,
) -> Tuple[int, int, int]:
    if input.dim() != 3:
        raise ValueError(f"recurrent layer expects a 3-D input, got shape {tuple(input.shape)}")
    sequence_length = input.shape[1 if batch_first else 0]
    batch_size = input.shape[0 if batch_first else 1]
    input_size = input.shape[2]
    if min(sequence_length, batch_size, input_size) <= 0:
        raise ValueError(
            "recurrent input dimensions must be positive, got "
            f"sequence={sequence_length}, batch={batch_size}, input_size={input_size}"
        )
    if weight_hh.dim() != 2:
        raise ValueError(f"weight_hh must be 2-D, got shape {tuple(weight_hh.shape)}")
    hidden_size = weight_hh.shape[1]
    expected_ih = (gate_count * hidden_size, input_size)
    expected_hh = (gate_count * hidden_size, hidden_size)
    if tuple(weight_ih.shape) != expected_ih:
        raise ValueError(f"weight_ih must have shape {expected_ih}, got {tuple(weight_ih.shape)}")
    if tuple(weight_hh.shape) != expected_hh:
        raise ValueError(f"weight_hh must have shape {expected_hh}, got {tuple(weight_hh.shape)}")
    if tuple(initial_h.shape) != (batch_size, hidden_size):
        raise ValueError(
            f"initial_h must have shape {(batch_size, hidden_size)}, "
            f"got {tuple(initial_h.shape)}"
        )
    for name, bias in (("bias_ih", bias_ih), ("bias_hh", bias_hh)):
        if bias is not None and tuple(bias.shape) != (gate_count * hidden_size,):
            raise ValueError(
                f"{name} must have shape {(gate_count * hidden_size,)}, got {tuple(bias.shape)}"
            )
    return sequence_length, batch_size, hidden_size


def _output_shape(input: torch.Tensor, hidden_size: int) -> Tuple[int, int, int]:
    return (input.shape[0], input.shape[1], hidden_size)


def _cell_ring(
    input: torch.Tensor,
    sequence_length: int,
    batch_size: int,
    hidden_size: int,
    final_c: torch.Tensor,
) -> torch.Tensor:
    """Scratch cell history: ``(slices, batch, hidden)``, time-major always.

    Only ``cell[t-1]`` is ever read, so two slices suffice however long the
    sequence is.  A single-timestep sequence needs one, and the RNNT hot path is
    exactly that: reuse ``final_c`` as that slice, avoiding an otherwise
    needless allocator round-trip.
    """
    if sequence_length == 1:
        return final_c.unsqueeze(0)
    return input.new_empty(2, batch_size, hidden_size)


def _combined_bias(
    bias_ih: Optional[torch.Tensor], bias_hh: Optional[torch.Tensor]
) -> Optional[torch.Tensor]:
    if bias_ih is None:
        return bias_hh
    if bias_hh is None:
        return bias_ih
    return bias_ih + bias_hh


def _pack_lstm_parameters(
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: Optional[torch.Tensor],
    bias_hh: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Convert PyTorch gate-major parameters to ``[hidden, gate]`` rows."""
    hidden_size = weight_hh.shape[1]

    def pack_weight(weight: torch.Tensor) -> torch.Tensor:
        return (
            weight.reshape(4, hidden_size, weight.shape[1])
            .permute(1, 0, 2)
            .reshape(4 * hidden_size, weight.shape[1])
            .contiguous()
        )

    bias = _combined_bias(bias_ih, bias_hh)
    if bias is not None:
        bias = bias.reshape(4, hidden_size).transpose(0, 1).contiguous().reshape(-1)
    return pack_weight(weight_ih), pack_weight(weight_hh), bias


def _default_recurrent_tactic(batch_size: int) -> int:
    if batch_size <= 16:
        return _FUSED_16X64
    if batch_size <= 32:
        return _FUSED_32X64
    return _FUSED_64X64


def _dispatch_lstm_gemm(
    runner_args: tuple,
    sequence_length: int,
    batch_size: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
    tactic: Optional[Tuple[int, int]] = None,
) -> None:
    if tactic is not None:
        _get_recurrent_module().lstm_gemm_layer(*runner_args, *tactic)
        return
    if is_tuning_enabled():
        get_tuner().dispatch(
            OpKey("recurrent", "lstm"),
            (sequence_length, batch_size, hidden_size, int(bool(runner_args[-1]))),
            dtype,
            device,
            runner_args,
        )
        return
    _get_recurrent_module().lstm_gemm_layer(*runner_args, _default_recurrent_tactic(batch_size), 1)


def _dispatch_rnn_gemm(
    runner_args: tuple,
    nonlinearity: str,
    sequence_length: int,
    batch_size: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
    tactic: Optional[Tuple[int, int]] = None,
) -> None:
    if tactic is not None:
        _get_recurrent_module().rnn_gemm_layer(*runner_args, *tactic)
        return
    if is_tuning_enabled():
        get_tuner().dispatch(
            OpKey("recurrent", f"rnn_{nonlinearity}"),
            (sequence_length, batch_size, hidden_size, int(bool(runner_args[-1]))),
            dtype,
            device,
            runner_args,
        )
        return
    _get_recurrent_module().rnn_gemm_layer(*runner_args, _default_recurrent_tactic(batch_size), 1)


@oasr_api
def lstm_layer(
    input: torch.Tensor,
    initial_h: torch.Tensor,
    initial_c: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: Optional[torch.Tensor] = None,
    bias_hh: Optional[torch.Tensor] = None,
    *,
    batch_first: bool = False,
    out: Optional[torch.Tensor] = None,
    final_h: Optional[torch.Tensor] = None,
    final_c: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one unidirectional LSTM layer with fused affine/gate/state steps.

    Parameter layout and gate order match :class:`torch.nn.LSTM`.  ``input`` is
    TBC by default or BTC with ``batch_first=True``; initial and final states
    are always ``(batch, hidden)``.  CUDA FP16/BF16 is the intended serving
    scope.  The higher-level :class:`oasr.layers.LSTM` owns CPU/fp32 fallback.
    """
    sequence_length, batch_size, hidden_size = _check_layer_inputs(
        input, initial_h, weight_ih, weight_hh, bias_ih, bias_hh, 4, batch_first
    )
    if tuple(initial_c.shape) != (batch_size, hidden_size):
        raise ValueError(
            f"initial_c must have shape {(batch_size, hidden_size)}, "
            f"got {tuple(initial_c.shape)}"
        )
    shape = _output_shape(input, hidden_size)
    if out is None:
        out = input.new_empty(shape)
    if final_h is None:
        final_h = input.new_empty(batch_size, hidden_size)
    if final_c is None:
        final_c = input.new_empty(batch_size, hidden_size)
    cells = _cell_ring(input, sequence_length, batch_size, hidden_size, final_c)
    _get_recurrent_module().lstm_layer(
        out,
        final_h,
        final_c,
        cells,
        input,
        initial_h,
        initial_c,
        weight_ih,
        weight_hh,
        bias_ih,
        bias_hh,
        bool(batch_first),
    )
    return out, final_h, final_c


@oasr_api
def rnn_layer(
    input: torch.Tensor,
    initial_h: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: Optional[torch.Tensor] = None,
    bias_hh: Optional[torch.Tensor] = None,
    *,
    nonlinearity: str = "tanh",
    batch_first: bool = False,
    out: Optional[torch.Tensor] = None,
    final_h: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run one unidirectional vanilla RNN layer.

    ``nonlinearity`` is ``"tanh"`` or ``"relu"`` and parameter semantics
    match :class:`torch.nn.RNN`.
    """
    if nonlinearity not in ("tanh", "relu"):
        raise ValueError(f"nonlinearity must be 'tanh' or 'relu', got {nonlinearity!r}")
    _, batch_size, hidden_size = _check_layer_inputs(
        input, initial_h, weight_ih, weight_hh, bias_ih, bias_hh, 1, batch_first
    )
    shape = _output_shape(input, hidden_size)
    if out is None:
        out = input.new_empty(shape)
    if final_h is None:
        final_h = input.new_empty(batch_size, hidden_size)
    _get_recurrent_module().rnn_layer(
        out,
        final_h,
        input,
        initial_h,
        weight_ih,
        weight_hh,
        bias_ih,
        bias_hh,
        0 if nonlinearity == "tanh" else 1,
        bool(batch_first),
    )
    return out, final_h


@oasr_api
def lstm_gemm_layer(
    input: torch.Tensor,
    initial_h: torch.Tensor,
    initial_c: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: Optional[torch.Tensor] = None,
    bias_hh: Optional[torch.Tensor] = None,
    *,
    batch_first: bool = False,
    final_h: Optional[torch.Tensor] = None,
    final_c: Optional[torch.Tensor] = None,
    _packed_parameters: Optional[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]] = None,
    _tactic: Optional[Tuple[int, int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tensor-core LSTM layer for large cohorts and wide hidden states.

    The input affine is evaluated once across the complete sequence.  Parameters
    are packed so each cell's four gates are adjacent.  CUTLASS evaluates each
    dependent recurrent affine, and its custom epilogue writes hidden/cell
    state directly.  Output is deliberately TBC so every preceding timestep
    is a contiguous GEMM operand.
    """
    sequence_length, batch_size, hidden_size = _check_layer_inputs(
        input, initial_h, weight_ih, weight_hh, bias_ih, bias_hh, 4, batch_first
    )
    if tuple(initial_c.shape) != (batch_size, hidden_size):
        raise ValueError(
            f"initial_c must have shape {(batch_size, hidden_size)}, "
            f"got {tuple(initial_c.shape)}"
        )
    from oasr.functionals.gemm import gemm

    if _packed_parameters is None:
        packed_weight_ih, packed_weight_hh, combined_bias = _pack_lstm_parameters(
            weight_ih, weight_hh, bias_ih, bias_hh
        )
    else:
        packed_weight_ih, packed_weight_hh, combined_bias = _packed_parameters
    input_gates = gemm(input, packed_weight_ih, combined_bias)
    output = input.new_empty(sequence_length, batch_size, hidden_size)
    if final_h is None:
        final_h = input.new_empty(batch_size, hidden_size)
    if final_c is None:
        final_c = input.new_empty(batch_size, hidden_size)
    cells = _cell_ring(input, sequence_length, batch_size, hidden_size, final_c)
    workspace = input.new_empty(batch_size, 4 * hidden_size)
    runner_args = (
        output,
        final_h,
        final_c,
        cells,
        workspace,
        input_gates,
        initial_h,
        initial_c,
        packed_weight_hh,
        None,
        bool(batch_first),
    )
    _dispatch_lstm_gemm(
        runner_args,
        sequence_length,
        batch_size,
        hidden_size,
        input.dtype,
        input.device,
        _tactic,
    )
    return output, final_h, final_c


@oasr_api
def rnn_gemm_layer(
    input: torch.Tensor,
    initial_h: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: Optional[torch.Tensor] = None,
    bias_hh: Optional[torch.Tensor] = None,
    *,
    nonlinearity: str = "tanh",
    batch_first: bool = False,
    final_h: Optional[torch.Tensor] = None,
    _combined_input_bias: Optional[torch.Tensor] = None,
    _tactic: Optional[Tuple[int, int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tensor-core vanilla-RNN layer; output is time-major and contiguous."""
    if nonlinearity not in ("tanh", "relu"):
        raise ValueError(f"nonlinearity must be 'tanh' or 'relu', got {nonlinearity!r}")
    sequence_length, batch_size, hidden_size = _check_layer_inputs(
        input, initial_h, weight_ih, weight_hh, bias_ih, bias_hh, 1, batch_first
    )
    from oasr.functionals.gemm import gemm

    combined_bias = (
        _combined_bias(bias_ih, bias_hh) if _combined_input_bias is None else _combined_input_bias
    )
    input_gates = gemm(input, weight_ih, combined_bias)
    output = input.new_empty(sequence_length, batch_size, hidden_size)
    if final_h is None:
        final_h = input.new_empty(batch_size, hidden_size)
    runner_args = (
        output,
        final_h,
        input_gates,
        initial_h,
        weight_hh,
        None,
        0 if nonlinearity == "tanh" else 1,
        bool(batch_first),
    )
    _dispatch_rnn_gemm(
        runner_args,
        nonlinearity,
        sequence_length,
        batch_size,
        hidden_size,
        input.dtype,
        input.device,
        _tactic,
    )
    return output, final_h


__all__ = ["lstm_gemm_layer", "lstm_layer", "rnn_gemm_layer", "rnn_layer"]
