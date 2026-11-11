# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Functional API for fused recurrent layers."""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Callable, Optional, Tuple, cast

import torch

from oasr.api_logging import oasr_api
from oasr.jit.recurrent_select import tuned_choice as _tuned_choice
from oasr.tune import OpKey, get_tuner, is_tuning_enabled

if TYPE_CHECKING:
    from oasr.jit.recurrent_select import RecChoice

logger = logging.getLogger("oasr.functionals.recurrent")

#: One warning per process from :func:`_pack_lstm_parameters`; a caller that
#: repacks per call needs telling once, not per timestep.
_PACK_WARNED = False

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
    # A T=1 step runs this once per layer, so a transducer predictor pays it per
    # emitted label.  ``torch.Size`` is already a tuple and compares equal to
    # one, so the comparisons below use it directly; wrapping each in ``tuple()``
    # copied four shapes per call for nothing.  ``tuple()`` stays in the error
    # messages, which are cold.
    shape = input.shape
    if len(shape) != 3:
        raise ValueError(f"recurrent layer expects a 3-D input, got shape {tuple(shape)}")
    sequence_length = shape[1] if batch_first else shape[0]
    batch_size = shape[0] if batch_first else shape[1]
    input_size = shape[2]
    if sequence_length <= 0 or batch_size <= 0 or input_size <= 0:
        raise ValueError(
            "recurrent input dimensions must be positive, got "
            f"sequence={sequence_length}, batch={batch_size}, input_size={input_size}"
        )
    hh_shape = weight_hh.shape
    if len(hh_shape) != 2:
        raise ValueError(f"weight_hh must be 2-D, got shape {tuple(hh_shape)}")
    hidden_size = hh_shape[1]
    gated = gate_count * hidden_size
    if weight_ih.shape != (gated, input_size):
        raise ValueError(
            f"weight_ih must have shape {(gated, input_size)}, got {tuple(weight_ih.shape)}"
        )
    if hh_shape != (gated, hidden_size):
        raise ValueError(f"weight_hh must have shape {(gated, hidden_size)}, got {tuple(hh_shape)}")
    if initial_h.shape != (batch_size, hidden_size):
        raise ValueError(
            f"initial_h must have shape {(batch_size, hidden_size)}, "
            f"got {tuple(initial_h.shape)}"
        )
    if bias_ih is not None and bias_ih.shape != (gated,):
        raise ValueError(f"bias_ih must have shape {(gated,)}, got {tuple(bias_ih.shape)}")
    if bias_hh is not None and bias_hh.shape != (gated,):
        raise ValueError(f"bias_hh must have shape {(gated,)}, got {tuple(bias_hh.shape)}")
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
    """Convert PyTorch gate-major parameters to ``[hidden, gate]`` rows.

    Two permute-copies of the whole weight matrix and one of the bias — **40.9 us
    at LSTM(640, 640)**, which is more than the timestep it feeds.  It is meant to
    run once per set of weights, not once per call.

    :class:`oasr.layers.LSTM` caches it per layer and threads the result in as
    ``_packed_parameters``; a caller reaching for ``oasr.lstm_gemm_layer``
    directly gets no such cache and nothing used to say so.  Hence the warning
    below: it fires once per process, names the parameter to pass, and stays out
    of the way of the layer, which never reaches it.
    """
    global _PACK_WARNED
    if not _PACK_WARNED:
        _PACK_WARNED = True
        logger.warning(
            "_pack_lstm_parameters ran inside a recurrent layer call: this repacks "
            "the whole weight matrix (~41 us at hidden=640, more than the timestep "
            "it feeds) and is meant to run once per weight set. Pass the result as "
            "`_packed_parameters=`, or use oasr.layers.LSTM / RNN, which cache it."
        )
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


def _dtype_str(x: torch.Tensor) -> str:
    return "float16" if x.dtype is torch.float16 else "bfloat16"


def _tuned(
    op: str, x: torch.Tensor, seq_len: int, batch: int, hidden: int, input_size: int
) -> Optional["RecChoice"]:
    """The measured recurrent path for this call, or ``None`` (``oasr.jit.recurrent_select``)."""
    if x.dtype not in (torch.float16, torch.bfloat16) or not x.is_cuda:
        return None
    return _tuned_choice(op, _dtype_str(x), hidden, input_size, seq_len, batch)


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

    ``out``, ``final_h`` and ``final_c`` must not share storage with an input:
    at a single timestep one launch reads ``initial_h`` while it stores
    ``final_h``, so the launcher refuses an in-place state update rather than
    return half-updated state.
    """
    sequence_length, batch_size, hidden_size = _check_layer_inputs(
        input, initial_h, weight_ih, weight_hh, bias_ih, bias_hh, 4, batch_first
    )
    if initial_c.shape != (batch_size, hidden_size):
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
    match :class:`torch.nn.RNN`.  ``out`` and ``final_h`` must not share storage
    with an input, as in :func:`lstm_layer`.
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


def _state_spans_overlap(
    outputs: Tuple[torch.Tensor, ...], inputs: Tuple[torch.Tensor, ...]
) -> bool:
    """Whether any output's storage span meets an input's, or the other output's.

    Spans, not elements, so two interleaved views of one buffer count as
    overlapping: it only decides whether to leave the CuTeDSL lane, and the
    launchers then apply the exact rule (``oasr::StorageOverlaps``).
    """

    def span(t: torch.Tensor) -> Tuple[int, int]:
        if t.numel() == 0:
            return (0, 0)
        reach = sum((n - 1) * s for n, s in zip(t.shape, t.stride()))
        start = t.data_ptr()
        return (start, start + (reach + 1) * t.element_size())

    out_spans = [span(t) for t in outputs]
    others = out_spans[1:] + [span(t) for t in inputs]
    for i, (a0, a1) in enumerate(out_spans):
        for b0, b1 in others[i:]:
            if a0 < b1 and b0 < a1 and a1 > a0 and b1 > b0:
                return True
    return False


def _lstm_step_fused(
    input_gates: torch.Tensor,
    previous_h: torch.Tensor,
    previous_c: torch.Tensor,
    packed_weight_hh: torch.Tensor,
    out_h: torch.Tensor,
    out_c: torch.Tensor,
    outputs_are_fresh: bool = False,
    route: Optional[Tuple[str, object]] = None,
) -> bool:
    """Run one timestep on the fused recurrent step.  ``False`` if it declined.

    ``route`` is a ``(lane, callable)`` the caller already chose -- the measured
    selection's lane and tile (``recurrent_cute.step_for_tile``); ``None`` asks
    the hand-timed band (``recurrent_cute.routed_step``).

    ``outputs_are_fresh`` says ``out_h`` / ``out_c`` were just allocated, so they
    cannot overlap the inputs and the CuTeDSL lane need not test them.

    ``oasr.jit.recurrent_cute`` arbitrates between the two lanes and hands back
    ``(backend, callable)``; the two have different signatures, so the name
    decides how to call it.  The C++ lane takes its **outputs first**
    (``AGENTS.md`` rule 4) and reads the stream from the FFI environment; the
    CuTeDSL one takes them last plus a stream handle, which must be resolved
    *inside* any CUDA-graph capture (see :mod:`oasr.jit.cute_runtime`).

    Declining rather than raising is deliberate: the caller has a working path
    either way, and a build failure on one shape should cost that shape its
    speedup, not the request.  A *wrong answer* would still raise -- nothing here
    swallows anything but construction and compilation.
    """
    from oasr.jit import recurrent_cute

    routed = route
    if routed is None:
        dtype_str = "float16" if input_gates.dtype is torch.float16 else "bfloat16"
        routed = recurrent_cute.routed_step(
            dtype_str=dtype_str,
            gate_count=4,
            activation="lstm",
            hidden=out_h.shape[1],
            batch=out_h.shape[0],
        )
    if routed is None:
        return False
    backend, step = routed
    step = cast(Callable[..., None], step)
    if backend == "cxx":
        # The launcher refuses outputs that share storage with its inputs.
        step(out_h, out_c, previous_h, packed_weight_hh, input_gates, previous_c)
    else:
        # The CuTeDSL callable checks nothing on the host, and the step is
        # wrong when the new state overlaps the old (several CTAs read each
        # previous_h row while others store h).  Decline, so the caller's
        # decomposed path -- whose launcher refuses it by name -- answers.
        # Only for buffers the caller supplied: the test costs ~3-5 us in
        # Python, on a lane that is the default fused step where the C++ one is
        # not compiled (sm_90, sm_100).
        if not outputs_are_fresh and _state_spans_overlap((out_h, out_c), (previous_h, previous_c)):
            return False
        step(
            previous_h,
            packed_weight_hh,
            input_gates,
            previous_c,
            out_h,
            out_c,
            recurrent_cute.current_stream(),
        )
    return True


#: The name this was called before there were two lanes.  Kept because it is
#: what the recorded A/B commands and the benchmark harness reach for.
_lstm_step_cute = _lstm_step_fused


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
    _outputs_are_fresh: bool = False,
    _step_route: Optional[Tuple[str, object]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tensor-core LSTM layer for large cohorts and wide hidden states.

    The input affine is evaluated once across the complete sequence.  Parameters
    are packed so each cell's four gates are adjacent.  CUTLASS evaluates each
    dependent recurrent affine, and its custom epilogue writes hidden/cell
    state directly.  Output is deliberately TBC so every preceding timestep
    is a contiguous GEMM operand.  ``final_h`` and ``final_c`` must not share
    storage with an input, as in :func:`lstm_layer`.
    """
    sequence_length, batch_size, hidden_size = _check_layer_inputs(
        input, initial_h, weight_ih, weight_hh, bias_ih, bias_hh, 4, batch_first
    )
    if initial_c.shape != (batch_size, hidden_size):
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
    # The measured selection (``oasr.jit.recurrent_select``): the fused step's
    # lane and tile, or the decomposed path and its tactic.  ``None`` -- no
    # ``recurrent`` file covers the shape, or it names ``scalar`` / ``auto`` --
    # leaves the hand-timed rules in charge.  A caller's ``_tactic`` wins.
    tuned_route = _step_route
    skip_fused = False
    if _tactic is None and _step_route is None:
        choice = _tuned("lstm", input, sequence_length, batch_size, hidden_size, weight_ih.shape[1])
        if choice is not None and choice.path in ("cxx", "cute"):
            from oasr.jit import recurrent_cute

            tuned_route = recurrent_cute.step_for_tile(
                lane=choice.path,
                dtype_str=_dtype_str(input),
                gate_count=4,
                activation="lstm",
                tile=choice.tile,
            )
            skip_fused = tuned_route is None
        elif choice is not None and choice.path == "gemm":
            skip_fused = True
            if choice.tactic >= 0:
                _tactic = (choice.tactic, choice.split)
    # Buffers this call allocates cannot overlap its inputs; ``oasr.layers.LSTM``
    # says the same of the ones it passes (``_outputs_are_fresh``).
    fresh = _outputs_are_fresh or (final_h is None and final_c is None)
    if final_h is None:
        final_h = input.new_empty(batch_size, hidden_size)
    if final_c is None:
        final_c = input.new_empty(batch_size, hidden_size)

    # A single timestep is exactly the shape the fused step owns: it does the
    # recurrent GEMM and the state transition in one launch, where this path
    # otherwise materializes a gate tile and pays a second kernel for the
    # transition.  Only inside the measured band, and on whichever of the two
    # lanes wins -- see oasr/jit/recurrent_cute.py, the arbiter.
    #
    # Taken *before* the tensors below are allocated: `output`, `cells` and
    # `workspace` exist only for the decomposed path, and at these sizes three
    # unused allocations plus the cell ring cost about as much host time as the
    # kernel costs GPU time.
    if sequence_length == 1 and _tactic is None and not skip_fused:
        # ``_lstm_step_fused`` owns the whole decision now -- band, lane, arch
        # probe and compile all memoise behind one dict lookup in
        # ``recurrent_cute.routed_step``, so asking ``should_use`` here first only
        # paid for the table scan twice.
        #
        # The projection keeps the caller's layout, so the one timestep is
        # ``[:, 0]`` batch-first and ``[0]`` time-major.  ``[0]`` of a batch-first
        # ``(B, 1, 4H)`` is the first *batch row*: the C++ lane refuses that view,
        # and inside the band every batch-first caller -- a transducer predictor
        # at 64 concurrent streams -- failed its cohort.
        if _lstm_step_fused(
            input_gates[:, 0] if batch_first else input_gates[0],
            initial_h,
            initial_c,
            packed_weight_hh,
            final_h,
            final_c,
            outputs_are_fresh=fresh,
            route=tuned_route,
        ):
            # h was written straight into final_h, so the (1, B, H) layer output
            # is a view of it rather than a copy.
            return final_h.unsqueeze(0), final_h, final_c

    output = input.new_empty(sequence_length, batch_size, hidden_size)
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
    """Tensor-core vanilla-RNN layer; output is time-major and contiguous.

    ``final_h`` must not share storage with an input, as in :func:`lstm_layer`.
    """
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
    if _tactic is None:
        # The measured selection's tactic (``oasr.jit.recurrent_select``).
        choice = _tuned(
            f"rnn_{nonlinearity}",
            input,
            sequence_length,
            batch_size,
            hidden_size,
            weight_ih.shape[1],
        )
        if choice is not None and choice.path == "gemm" and choice.tactic >= 0:
            _tactic = (choice.tactic, choice.split)
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


def _check_slot_step(
    input: torch.Tensor,
    state_h: torch.Tensor,
    state_slots: torch.Tensor,
    read_parity: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    gate_count: int,
) -> Tuple[int, int, int, int]:
    if input.dim() != 2:
        raise ValueError(f"slot-step input must be 2-D (batch, input_size), got {input.dim()}-D")
    if state_h.dim() != 3 or state_h.shape[0] != 2:
        raise ValueError(
            "state_h must be the (2, slots, hidden) ring -- one slice is read while the "
            f"other is written, got {tuple(state_h.shape)}"
        )
    batch_size, input_size = input.shape
    slot_count, hidden_size = state_h.shape[1], state_h.shape[2]
    if tuple(weight_ih.shape) != (gate_count * hidden_size, input_size):
        raise ValueError(
            f"weight_ih must have shape {(gate_count * hidden_size, input_size)}, "
            f"got {tuple(weight_ih.shape)}"
        )
    if tuple(weight_hh.shape) != (gate_count * hidden_size, hidden_size):
        raise ValueError(
            f"weight_hh must have shape {(gate_count * hidden_size, hidden_size)}, "
            f"got {tuple(weight_hh.shape)}"
        )
    if state_slots.dim() != 1 or state_slots.shape[0] != batch_size:
        raise ValueError(
            f"state_slots must be a 1-D int64 tensor of {batch_size} slot ids, "
            f"got shape {tuple(state_slots.shape)}"
        )
    if state_slots.dtype != torch.int64:
        raise ValueError(f"state_slots must be int64, got {state_slots.dtype}")
    if read_parity.dim() != 1 or read_parity.shape[0] != batch_size:
        raise ValueError(
            f"read_parity must be a 1-D int32 tensor of {batch_size} entries, "
            f"got shape {tuple(read_parity.shape)}"
        )
    if read_parity.dtype != torch.int32:
        raise ValueError(f"read_parity must be int32, got {read_parity.dtype}")
    return batch_size, input_size, slot_count, hidden_size


@oasr_api
def lstm_slot_step(
    input: torch.Tensor,
    state_h: torch.Tensor,
    state_c: torch.Tensor,
    state_slots: torch.Tensor,
    read_parity: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: Optional[torch.Tensor] = None,
    bias_hh: Optional[torch.Tensor] = None,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """One LSTM timestep over slot-addressed state, for continuous batching.

    Row ``i`` of ``input`` belongs to stream slot ``state_slots[i]``, so the rows
    of one call may be at completely different timesteps of different sequences
    and the membership may change every call.  A recurrence carries no history
    beyond ``(h, c)``, which is what makes that free here.

    ``state_h`` is a ``(2, slots, hidden)`` ring: row ``i`` is read from slice
    ``read_parity[i]`` and written to the other one, because a CTA owning hidden
    unit ``j`` reads its row's whole ``h`` vector but writes only element ``j``,
    so an in-place update would race.  The parity is *per row* rather than per
    call so that a stream which sits out a tick keeps its current ``h`` where it
    last wrote it; one parity for the whole batch would read a stale slice for
    every idle stream.  ``state_c`` is ``(slots, hidden)`` and is updated in
    place, because each cell element is read and written by its owning thread.

    Returns the ``(batch, hidden)`` dense hidden rows for this timestep.  The
    caller flips ``read_parity`` for the rows it stepped -- which is what
    :class:`oasr.cache.RecurrentStateCache` does.
    """
    batch_size, _, _, hidden_size = _check_slot_step(
        input, state_h, state_slots, read_parity, weight_ih, weight_hh, 4
    )
    if state_c.dim() != 2 or state_c.shape[1] != hidden_size:
        raise ValueError(
            f"state_c must have shape (slots, {hidden_size}), got {tuple(state_c.shape)}"
        )
    if out is None:
        out = input.new_empty(batch_size, hidden_size)
    _get_recurrent_module().lstm_slot_step(
        out,
        state_h,
        state_c,
        input,
        state_slots,
        read_parity,
        weight_ih,
        weight_hh,
        bias_ih,
        bias_hh,
    )
    return out


@oasr_api
def rnn_slot_step(
    input: torch.Tensor,
    state_h: torch.Tensor,
    state_slots: torch.Tensor,
    read_parity: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: Optional[torch.Tensor] = None,
    bias_hh: Optional[torch.Tensor] = None,
    *,
    nonlinearity: str = "tanh",
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """One vanilla-RNN timestep over slot-addressed state.  See :func:`lstm_slot_step`."""
    if nonlinearity not in ("tanh", "relu"):
        raise ValueError(f"nonlinearity must be 'tanh' or 'relu', got {nonlinearity!r}")
    batch_size, _, _, hidden_size = _check_slot_step(
        input, state_h, state_slots, read_parity, weight_ih, weight_hh, 1
    )
    if out is None:
        out = input.new_empty(batch_size, hidden_size)
    _get_recurrent_module().rnn_slot_step(
        out,
        state_h,
        input,
        state_slots,
        read_parity,
        weight_ih,
        weight_hh,
        bias_ih,
        bias_hh,
        0 if nonlinearity == "tanh" else 1,
    )
    return out


__all__ = [
    "lstm_gemm_layer",
    "lstm_layer",
    "lstm_slot_step",
    "rnn_gemm_layer",
    "rnn_layer",
    "rnn_slot_step",
]
