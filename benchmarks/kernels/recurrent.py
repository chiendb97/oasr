# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""LSTM and vanilla RNN -- ``oasr/functionals/recurrent.py``.

Many arms: the OASR layer, the hand-written non-MMA step, five CUTLASS tactics,
and cuDNN.  The baseline is cuDNN rather than a generic ``torch`` arm, because
``torch.nn.LSTM`` *is* cuDNN here -- the previous harness carried both names for
one callable and relabelled ``torch`` to ``cudnn`` when writing the row, so
``--backends torch`` produced a row that said ``cudnn``.
"""

from __future__ import annotations

import argparse
from typing import Any, Callable, Dict

import torch
from torch import nn

import oasr
from benchmarks.core.driver import Work, params_of
from oasr.layers import LSTM, RNN

SUBROUTINES = ["lstm", "rnn_tanh", "rnn_relu", "lstm_slot_step", "lstm_step_cute"]

#: A recurrence is a dependent chain; measuring one arm in a contiguous block
#: lets the allocator warm for the next one.
INTERLEAVE = True

#: torch.nn.LSTM dispatches to cuDNN on CUDA, so that is what the row should say.
REF_BACKEND = "cudnn"

# The first three rows are the Nemotron prediction-network operating point as
# decode cohorts fill.  The rest span recurrent sequence workloads without
# building an uninformative full Cartesian product.
_SHAPES = [
    (1, 1, 640, 2),
    (8, 1, 640, 2),
    (32, 1, 640, 2),
    (1, 16, 256, 2),
    (8, 32, 512, 2),
    (32, 128, 256, 2),
    (16, 64, 1024, 1),
]

# The slot step is one timestep by construction, so it sweeps the batch instead
# of the sequence: it exists for continuous batching, where the interesting axis
# is how many concurrent streams a tick carries.
_SLOT_SHAPES = [
    (1, 1, 640, 1),
    (8, 1, 640, 1),
    (32, 1, 640, 1),
    (128, 1, 640, 1),
    (32, 1, 256, 1),
    (32, 1, 1024, 1),
]

DEFAULT_CONFIGS: Dict[str, list] = {
    sub: [
        {"batch": b, "seq": t, "input_size": h, "hidden_size": h, "num_layers": layers}
        for b, t, h, layers in (
            _SLOT_SHAPES if sub in ("lstm_slot_step", "lstm_step_cute") else _SHAPES
        )
    ]
    for sub in SUBROUTINES
}


def parse_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--seq", type=int, default=None)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=2)


def resolve_configs(args: argparse.Namespace, subroutine: str) -> list:
    dims = (args.batch, args.seq, args.input_size, args.hidden_size)
    if all(v is not None for v in dims):
        return [
            {
                "batch": dims[0],
                "seq": dims[1],
                "input_size": dims[2],
                "hidden_size": dims[3],
                "num_layers": args.num_layers,
            }
        ]
    return DEFAULT_CONFIGS[subroutine]


def _setup(
    subroutine: str, config: dict[str, Any], dtype: torch.dtype
) -> tuple[dict[str, Callable], Callable]:
    batch = config["batch"]
    sequence = config["seq"]
    input_size = config["input_size"]
    hidden_size = config["hidden_size"]
    num_layers = config["num_layers"]
    x = torch.randn(batch, sequence, input_size, device="cuda", dtype=dtype)
    h = torch.randn(num_layers, batch, hidden_size, device="cuda", dtype=dtype)

    if subroutine == "lstm_step_cute":
        # The CuTeDSL fused step against the tensor-core GEMM it replaces.  Both
        # consume a precomputed input projection, so this is the recurrent step
        # alone -- which is the only place the fusion shows up undiluted.
        from oasr.jit import recurrent_cute

        n = 4 * hidden_size
        prev_h = torch.randn(batch, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(n, hidden_size, device="cuda", dtype=dtype) * hidden_size**-0.5
        in_gates = torch.randn(batch, n, device="cuda", dtype=dtype)
        prev_c = torch.randn(batch, hidden_size, device="cuda", dtype=dtype)
        out_h = torch.empty(batch, hidden_size, device="cuda", dtype=dtype)
        out_c = torch.empty(batch, hidden_size, device="cuda", dtype=dtype)
        gate_buf = torch.empty(batch, n, device="cuda", dtype=dtype)
        dtype_str = "float16" if dtype is torch.float16 else "bfloat16"
        try:
            step = recurrent_cute.get_compiled_step(
                dtype_str=dtype_str,
                gate_count=4,
                activation="lstm",
                hidden=hidden_size,
                batch=batch,
            )
        except Exception as exc:  # no CuTeDSL, or no tile for this shape
            print(f"  [WARNING] CuTeDSL step unavailable: {exc}")
            step = None

        @torch.no_grad()
        def cute_fn():
            step(
                prev_h,
                weight,
                in_gates,
                prev_c,
                out_h,
                out_c,
                recurrent_cute.current_stream(),
            )
            return out_h

        @torch.no_grad()
        def gemm_fn():
            """Lower bound on the decomposed path: its GEMM, without the epilogue."""
            return torch.mm(prev_h, weight.t(), out=gate_buf)

        # Named "cublas", not "torch": the routine relabels a "torch" backend as
        # "cudnn" on output, and this arm is neither.
        fns = {"cublas": gemm_fn}
        if step is not None:
            fns["cute"] = cute_fn
        # No meaningful benchmark reference: the arms compute different things (the
        # GEMM alone is a lower bound, not an equivalent).  Correctness is checked
        # against FP32 equations in tests/test_recurrent_cute.py, which is a
        # stronger oracle than a refcheck against another kernel would be.
        return fns, (cute_fn if step is not None else gemm_fn)

    if subroutine == "lstm_slot_step":
        # Slots deliberately exceed the row count so the gather is a real
        # scattered read, as it is when streams retire and are replaced.
        slots = max(2 * batch, batch + 1)
        ours = LSTM(input_size, hidden_size, num_layers=1, device="cuda", dtype=dtype).eval()
        weight_ih = ours.weight_ih_l0
        weight_hh = ours.weight_hh_l0
        bias_ih, bias_hh = ours._biases(0)
        frames = torch.randn(batch, input_size, device="cuda", dtype=dtype)
        slot_ids = torch.randperm(slots, device="cuda")[:batch].to(torch.int64)
        # Mixed parity is the realistic case: rows admitted at different ticks
        # have taken different numbers of steps.
        parity = torch.randint(0, 2, (batch,), device="cuda", dtype=torch.int32)
        base_ring = torch.randn(2, slots, hidden_size, device="cuda", dtype=dtype) * 0.2
        base_cells = torch.randn(slots, hidden_size, device="cuda", dtype=dtype) * 0.2
        # Each arm gets its own copy of identical state.  Both mutate it in place
        # and compute the same function, so they stay in step across iterations
        # while neither can perturb the other's reference.
        ring, cells = base_ring.clone(), base_cells.clone()
        ring_ref, cells_ref = base_ring.clone(), base_cells.clone()
        long_parity = parity.long()

        # Both arms are read-modify-write on cell state, so a call is not pure and
        # repeated calls would drift.  Each restores from the pristine copy first,
        # which costs both arms the same two small copies and keeps the reference
        # comparison valid however many times the harness invokes either one.
        @torch.no_grad()
        def slot_fn():
            ring.copy_(base_ring)
            cells.copy_(base_cells)
            return oasr.lstm_slot_step(
                frames, ring, cells, slot_ids, parity, weight_ih, weight_hh, bias_ih, bias_hh
            )

        @torch.no_grad()
        def gather_fn():
            """What the same tick costs without a slot-addressed kernel."""
            ring_ref.copy_(base_ring)
            cells_ref.copy_(base_cells)
            h = ring_ref[long_parity, slot_ids].contiguous()
            c = cells_ref.index_select(0, slot_ids).contiguous()
            out, final_h, final_c = oasr.lstm_layer(
                frames.unsqueeze(0), h, c, weight_ih, weight_hh, bias_ih, bias_hh
            )
            ring_ref[1 - long_parity, slot_ids] = final_h
            cells_ref.index_copy_(0, slot_ids, final_c)
            return out[0]

        return {"oasr": slot_fn, "gather": gather_fn}, gather_fn

    if subroutine == "lstm":
        ours = LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            device="cuda",
            dtype=dtype,
        ).eval()
        baseline = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            device="cuda",
            dtype=dtype,
        ).eval()
        baseline.load_state_dict(ours.state_dict())
        baseline.flatten_parameters()
        c = torch.randn_like(h)

        @torch.no_grad()
        def oasr_fn():
            return ours(x, (h, c))

        @torch.no_grad()
        def cudnn_fn():
            return baseline(x, (h, c))

        @torch.no_grad()
        def native_fn():
            output = x
            final_h = []
            final_c = []
            for layer in range(num_layers):
                bias_ih, bias_hh = ours._biases(layer)
                output, hidden, cell = oasr.lstm_layer(
                    output,
                    h[layer],
                    c[layer],
                    getattr(ours, f"weight_ih_l{layer}"),
                    getattr(ours, f"weight_hh_l{layer}"),
                    bias_ih,
                    bias_hh,
                    batch_first=True,
                )
                final_h.append(hidden)
                final_c.append(cell)
            return output, (torch.stack(final_h), torch.stack(final_c))

        def make_cutlass_fn(tactic: tuple[int, int]):
            @torch.no_grad()
            def cutlass_fn():
                output = x
                final_h = []
                final_c = []
                current_batch_first = True
                for layer in range(num_layers):
                    bias_ih, bias_hh = ours._biases(layer)
                    output, hidden, cell = oasr.lstm_gemm_layer(
                        output,
                        h[layer],
                        c[layer],
                        getattr(ours, f"weight_ih_l{layer}"),
                        getattr(ours, f"weight_hh_l{layer}"),
                        bias_ih,
                        bias_hh,
                        batch_first=current_batch_first,
                        _packed_parameters=ours._packed_lstm_parameters(layer),
                        _tactic=tactic,
                    )
                    current_batch_first = False
                    final_h.append(hidden)
                    final_c.append(cell)
                return output.transpose(0, 1), (torch.stack(final_h), torch.stack(final_c))

            return cutlass_fn

    else:
        nonlinearity = "relu" if subroutine == "rnn_relu" else "tanh"
        ours = RNN(
            input_size,
            hidden_size,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            batch_first=True,
            device="cuda",
            dtype=dtype,
        ).eval()
        baseline = nn.RNN(
            input_size,
            hidden_size,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            batch_first=True,
            device="cuda",
            dtype=dtype,
        ).eval()
        baseline.load_state_dict(ours.state_dict())
        baseline.flatten_parameters()

        @torch.no_grad()
        def oasr_fn():
            return ours(x, h)

        @torch.no_grad()
        def cudnn_fn():
            return baseline(x, h)

        @torch.no_grad()
        def native_fn():
            output = x
            final_h = []
            for layer in range(num_layers):
                bias_ih, bias_hh = ours._biases(layer)
                output, hidden = oasr.rnn_layer(
                    output,
                    h[layer],
                    getattr(ours, f"weight_ih_l{layer}"),
                    getattr(ours, f"weight_hh_l{layer}"),
                    bias_ih,
                    bias_hh,
                    nonlinearity=nonlinearity,
                    batch_first=True,
                )
                final_h.append(hidden)
            return output, torch.stack(final_h)

        def make_cutlass_fn(tactic: tuple[int, int]):
            @torch.no_grad()
            def cutlass_fn():
                output = x
                final_h = []
                current_batch_first = True
                for layer in range(num_layers):
                    bias_ih, bias_hh = ours._biases(layer)
                    output, hidden = oasr.rnn_gemm_layer(
                        output,
                        h[layer],
                        getattr(ours, f"weight_ih_l{layer}"),
                        getattr(ours, f"weight_hh_l{layer}"),
                        bias_ih,
                        bias_hh,
                        nonlinearity=nonlinearity,
                        batch_first=current_batch_first,
                        _combined_input_bias=ours._combined_rnn_bias(layer),
                        _tactic=tactic,
                    )
                    current_batch_first = False
                    final_h.append(hidden)
                return output.transpose(0, 1), torch.stack(final_h)

            return cutlass_fn

    functions = {
        "oasr": oasr_fn,
        "native": native_fn,
        "cutlass16": make_cutlass_fn((0, 1)),
        "cutlass32": make_cutlass_fn((1, 1)),
        "cutlass64": make_cutlass_fn((2, 1)),
        "streamk": make_cutlass_fn((3, 1)),
        "splitk": make_cutlass_fn((4, 4)),
        "cudnn": cudnn_fn,
    }
    if subroutine == "lstm":
        functions["serial_splitk"] = make_cutlass_fn((5, 4))
    return functions, cudnn_fn


def _assert_close(subroutine: str, actual, expected) -> float:
    if subroutine == "lstm":
        actual_tensors = (actual[0], *actual[1])
        expected_tensors = (expected[0], *expected[1])
    elif subroutine in ("lstm_slot_step", "lstm_step_cute"):
        # One dense tensor, not a state tuple.
        actual_tensors, expected_tensors = (actual,), (expected,)
    else:
        actual_tensors, expected_tensors = actual, expected
    max_diff = 0.0
    for got, ref in zip(actual_tensors, expected_tensors):
        max_diff = max(max_diff, (got.float() - ref.float()).abs().max().item())
        torch.testing.assert_close(got, ref, rtol=3e-2, atol=3e-2)
    return max_diff


def _flops(subroutine: str, config: dict[str, Any]) -> int:
    gates = 4 if subroutine.startswith("lstm") else 1
    batch = config["batch"]
    sequence = config["seq"]
    hidden = config["hidden_size"]
    total = 0
    layer_input = config["input_size"]
    for _ in range(config["num_layers"]):
        total += 2 * gates * batch * sequence * hidden * (layer_input + hidden)
        layer_input = hidden
    return total


def build_fns(
    subroutine: str, cfg: dict, dtype: torch.dtype, args: argparse.Namespace
) -> Dict[str, Callable[[], Any]]:
    functions, _reference = _setup(subroutine, cfg, dtype)
    return functions


def describe(subroutine: str, cfg: dict, dtype: torch.dtype) -> Work:
    return Work(
        shape=(
            f"B={cfg['batch']},T={cfg['seq']},I={cfg['input_size']},"
            f"H={cfg['hidden_size']},L={cfg['num_layers']}"
        ),
        params=params_of(cfg),
        flops=_flops(subroutine, cfg),
    )
