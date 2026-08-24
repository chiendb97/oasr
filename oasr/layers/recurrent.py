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
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, cast

import torch
import torch.nn.functional as F
from torch import nn

import oasr

from ._backend import use_recurrent_kernel


def _version_of(tensor: torch.Tensor) -> int:
    """``tensor._version``, or ``-1`` when the counter is not tracked.

    A module constructed or moved to the device inside ``torch.inference_mode()``
    holds inference tensors, and reading ``_version`` on one raises
    ``RuntimeError: Inference tensors do not track version counter`` -- which
    used to take down the first forward rather than the construction that caused
    it.  An inference tensor cannot be mutated in place, so a constant stands in
    for its counter.
    """
    try:
        return tensor._version
    except RuntimeError:
        return -1


def _versions_of(params: tuple) -> Optional[tuple]:
    """Version counters for a ``(w_ih, w_hh, b_ih, b_hh)`` tuple, or ``None``.

    Spelled out rather than as a generator expression: this runs once per layer
    per timestep, and the generator's frame cost more than the four reads.
    """
    a, b, c, d = params
    try:
        return (
            a._version,
            b._version,
            -1 if c is None else c._version,
            -1 if d is None else d._version,
        )
    except RuntimeError:
        return None


if TYPE_CHECKING:  # pragma: no cover
    from oasr.cache.recurrent_state import RecurrentStateCache

# Where the scalar path stops winning.
#
# The two paths differ in *what work they repeat*, not only in whether they use
# MMA.  The scalar path fuses one timestep into a single launch, so it re-reads
# every weight and recomputes the whole input projection once per timestep.  The
# tensor-core path evaluates the input projection once for the entire sequence as
# one fat GEMM and leaves only the dependent recurrent affine per timestep.  That
# makes the crossover depend on the sequence length as much as on the batch:
#
#   * At T == 1 there is no repetition to amortize, and the scalar kernel's one
#     launch beats the tensor-core path's GEMM + step + finalizer until the step
#     itself is big enough to fill the device.  Crossover is at a large B*H.
#   * At T > 1 the scalar path pays the input projection T times at roughly
#     10 TFLOP/s while the tensor-core path pays it once at over 100, so the
#     crossover collapses to a much smaller B*H.
#
# The thresholds below are the measured crossovers on SM120 (RTX 5090, FP16 and
# BF16, GPU-only time via CUDA-graph replay) over B in [1, 512] and H in
# [256, 2048].  They replace a `B >= 16 and H >= 1024` guess that sent every
# H < 1024 shape to the scalar path at every batch size: at B=512, H=640 --
# Nemotron's predictor width -- that guess cost 10.5x (7.34 ms against 0.70 ms).
#
# The RNN crossover sits at a larger batch than the LSTM one because a single
# gate is a quarter of the arithmetic per timestep, so the tensor-core path's
# extra launches amortize later.
#
# A single `B * hidden` product does not fit the measurements: the crossover
# batch falls faster than 1/H, because the scalar path's cost grows with both the
# width it re-reads and the batch it folds over.  These are therefore tables of
# (inclusive hidden-width bound, smallest batch that prefers tensor cores),
# ascending in width, taken directly from the measured grid.
_WIDTH_BUCKET_SENTINEL = 1 << 30

_TENSOR_CORE_MIN_BATCH_STEP = {  # T == 1
    4: ((256, 128), (768, 64), (_WIDTH_BUCKET_SENTINEL, 32)),
    1: ((512, 128), (_WIDTH_BUCKET_SENTINEL, 64)),
}
_TENSOR_CORE_MIN_BATCH_SEQ = {  # T > 1
    # At H >= 1536 even a single row is better off on tensor cores; below 768 the
    # scalar cohort kernel holds on until 8-32 rows.
    #
    # Collapsing every T > 1 into one table costs one cell: the advantage grows
    # with T, because a longer sequence amortizes the one input-projection GEMM
    # further.  At H=640, B=8 the scalar path is 11% ahead at T=8 and 22% behind
    # at T=32, and this table picks tensor cores for both.  Splitting T into bands
    # would recover that cell at the price of boundaries measured on one GPU.
    4: ((256, 32), (512, 16), (768, 8), (1536, 2), (_WIDTH_BUCKET_SENTINEL, 1)),
    1: ((256, 32), (_WIDTH_BUCKET_SENTINEL, 16)),
}

# `LstmLayerImpl` / `RnnLayerImpl` select the shared-weight cohort kernel at or
# above this batch; below it they fall back to the one-CTA-per-output GEMV, whose
# cost grows steeply with hidden width.  Two rows is 64 threads per CTA and
# measured slower than the GEMV, so the floor is four.  Kept in sync with
# `use_cohort` in `include/oasr/recurrent/recurrent.cuh`.
_COHORT_BATCH_FLOOR = 4


def _prefer_tensor_core(sequence_length: int, batch: int, hidden: int, gates: int) -> bool:
    """Should this shape take the tensor-core path rather than the scalar one?"""
    table = (_TENSOR_CORE_MIN_BATCH_STEP if sequence_length == 1 else _TENSOR_CORE_MIN_BATCH_SEQ)[
        gates
    ]
    for width, min_batch in table:
        if hidden <= width:
            return batch >= min_batch
    raise AssertionError("width bucket table must end in a sentinel")


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
        #: layer -> (parameter objects, their version counters or None, packed form).
        #: The hot-path shortcut past ``_packed_parameter_cache``; see
        #: :meth:`_packed_lstm_parameters`.
        self._packed_fast_slot: Dict[int, Tuple[tuple, Optional[tuple], tuple]] = {}

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
        self._packed_fast_slot.clear()

    def flatten_parameters(self) -> None:
        """Compatibility no-op; OASR consumes the checkpoint layout directly."""

    def _apply(self, fn, recurse: bool = True):
        # Packed tensors are derived, not parameters or buffers, so Module._apply
        # cannot migrate them. Rebuild lazily after .to()/.cuda() instead.
        self._packed_parameter_cache.clear()
        self._packed_fast_slot.clear()
        return super()._apply(fn, recurse)

    def _biases(self, layer: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.bias:
            return None, None
        return getattr(self, f"bias_ih_l{layer}"), getattr(self, f"bias_hh_l{layer}")

    @staticmethod
    def _parameter_signature(*parameters: Optional[torch.Tensor]) -> Tuple[Tuple[int, int], ...]:
        return tuple(
            (-1, -1) if parameter is None else (parameter.data_ptr(), _version_of(parameter))
            for parameter in parameters
        )

    def _packed_lstm_parameters(
        self, layer: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # Hot path: a T=1 step calls this once per layer, so a transducer
        # predictor pays it per emitted label.  Building the signature the long
        # way cost 3.19 us of the 100 us that ``LSTM.forward`` spends on a
        # 2-layer T=1 step -- four ``getattr`` calls through
        # ``nn.Module.__getattr__`` (parameters live in ``_parameters``, so every
        # one is a miss and a walk), four ``data_ptr()``, four ``_version``, a
        # generator expression and a tuple build, all to hit a dict.
        #
        # The fast slot keeps the parameter *objects* alongside their packed
        # form, so a hit reads no attributes at all and only re-checks the
        # version counters -- which is the half that can change without
        # ``_apply`` running.
        fast = self._packed_fast_slot.get(layer)
        if fast is not None:
            params, versions, packed = fast
            if versions is None or versions == _versions_of(params):
                return packed
        weight_ih = getattr(self, f"weight_ih_l{layer}")
        weight_hh = getattr(self, f"weight_hh_l{layer}")
        bias_ih, bias_hh = self._biases(layer)
        signature = self._parameter_signature(weight_ih, weight_hh, bias_ih, bias_hh)
        key = (layer, signature)
        cached = self._packed_parameter_cache.get(key)
        if cached is not None:
            self._store_fast_slot(layer, (weight_ih, weight_hh, bias_ih, bias_hh), cached)
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
        self._store_fast_slot(layer, (weight_ih, weight_hh, bias_ih, bias_hh), packed)
        return packed

    def _store_fast_slot(self, layer: int, params: tuple, packed: tuple) -> None:
        """Record the per-layer fast slot, or decline to if versions are untracked.

        ``Tensor._version`` raises on an inference tensor (a module constructed or
        moved to the device inside ``torch.inference_mode()``), which is also why
        the slow path reads it through :func:`_version_of`.  With no counter to
        watch there is nothing that could invalidate the slot, and an inference
        tensor cannot be mutated in place anyway, so ``None`` means "trust it".
        """
        self._packed_fast_slot[layer] = (params, _versions_of(params), packed)

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

    def _check_step_state(self, frames: torch.Tensor, cache) -> None:
        if frames.dim() != 2 or frames.shape[1] != self.input_size:
            raise ValueError(
                f"step frames must be (rows, {self.input_size}), got {tuple(frames.shape)}"
            )
        if cache.num_layers != self.num_layers or cache.hidden_size != self.hidden_size:
            raise ValueError(
                f"cache geometry {(cache.num_layers, cache.hidden_size)} does not match this "
                f"module's {(self.num_layers, self.hidden_size)}"
            )
        if not use_recurrent_kernel(frames):
            raise NotImplementedError(
                "the slot-addressed step is a CUDA FP16/BF16 kernel; there is no torch "
                "fallback for it, because a fallback would hide the missing kernel rather "
                "than report it. Use forward() for CPU or fp32."
            )

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
            and self.input_size % 8 == 0
            and self.hidden_size % 8 == 0
            and _prefer_tensor_core(
                input.shape[1 if kernel_batch_first else 0],
                batch_size,
                self.hidden_size,
                self._gate_count,
            )
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

    def step(
        self,
        frames: torch.Tensor,
        cache: "RecurrentStateCache",
        slot_ids: torch.Tensor,
        read_parity: torch.Tensor,
    ) -> torch.Tensor:
        """Advance ``frames``' rows one timestep against slot-addressed state.

        Row ``i`` of ``frames`` belongs to stream slot ``slot_ids[i]``, so one call
        may mix rows that are at completely different timesteps of different
        sequences -- which is what lets
        :class:`~oasr.cache.RecurrentContinuousBatcher` keep the batch full instead
        of running a cohort at the length of its longest member.

        The caller is responsible for :meth:`RecurrentStateCache.commit_step` once
        every layer has been issued: all layers of one tick share a parity.

        Returns
        -------
        Tensor
            ``(rows, hidden_size)`` output of the last layer.
        """
        self._check_step_state(frames, cache)
        if not cache.has_cell:
            raise ValueError("LSTM.step needs a cache built with cell state (cell=True)")
        output = frames.contiguous()
        for layer in range(self.num_layers):
            bias_ih, bias_hh = self._biases(layer)
            output = oasr.lstm_slot_step(
                output,
                cache.hidden(layer),
                cache.cell(layer),
                slot_ids,
                read_parity,
                getattr(self, f"weight_ih_l{layer}"),
                getattr(self, f"weight_hh_l{layer}"),
                bias_ih,
                bias_hh,
            )
            output = self._drop_between_layers(output, layer)
        return output


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
            and self.input_size % 8 == 0
            and self.hidden_size % 8 == 0
            and _prefer_tensor_core(
                input.shape[1 if kernel_batch_first else 0],
                batch_size,
                self.hidden_size,
                self._gate_count,
            )
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

    def step(
        self,
        frames: torch.Tensor,
        cache: "RecurrentStateCache",
        slot_ids: torch.Tensor,
        read_parity: torch.Tensor,
    ) -> torch.Tensor:
        """Advance ``frames``' rows one timestep.  See :meth:`LSTM.step`."""
        self._check_step_state(frames, cache)
        output = frames.contiguous()
        for layer in range(self.num_layers):
            bias_ih, bias_hh = self._biases(layer)
            output = oasr.rnn_slot_step(
                output,
                cache.hidden(layer),
                slot_ids,
                read_parity,
                getattr(self, f"weight_ih_l{layer}"),
                getattr(self, f"weight_hh_l{layer}"),
                bias_ih,
                bias_hh,
                nonlinearity=self.nonlinearity,
            )
            output = self._drop_between_layers(output, layer)
        return output


__all__ = ["LSTM", "RNN"]
