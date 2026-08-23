"""LSTM and vanilla-RNN benchmarks against PyTorch/cuDNN."""

from __future__ import annotations

import argparse
import statistics
from typing import Any, Callable

import torch
from torch import nn

import oasr
from benchmarks.routines.bench_utils import (
    BenchResult,
    OutputWriter,
    bench_fn,
    parse_dtype,
)
from oasr.layers import LSTM, RNN

SUBROUTINES = ["lstm", "rnn_tanh", "rnn_relu", "lstm_slot_step", "lstm_step_cute"]

# The first three rows are the Nemotron prediction-network operating point as
# decode cohorts fill.  The remaining rows span recurrent sequence workloads
# without constructing an uninformative full Cartesian product.
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

DEFAULT_CONFIGS: dict[str, list[dict[str, Any]]] = {
    subroutine: [
        {
            "batch": batch,
            "seq": sequence,
            "input_size": hidden,
            "hidden_size": hidden,
            "num_layers": layers,
        }
        for batch, sequence, hidden, layers in (
            _SLOT_SHAPES if subroutine in ("lstm_slot_step", "lstm_step_cute") else _SHAPES
        )
    ]
    for subroutine in SUBROUTINES
}


def get_default_configs() -> dict[str, list[dict[str, Any]]]:
    return DEFAULT_CONFIGS


def parse_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--seq", type=int, default=None)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=2)


def _resolve_configs(args: argparse.Namespace, subroutine: str) -> list[dict[str, Any]]:
    values = (
        getattr(args, "batch", None),
        getattr(args, "seq", None),
        getattr(args, "input_size", None),
        getattr(args, "hidden_size", None),
    )
    if all(value is not None for value in values):
        return [
            {
                "batch": values[0],
                "seq": values[1],
                "input_size": values[2],
                "hidden_size": values[3],
                "num_layers": getattr(args, "num_layers", 2),
            }
        ]
    return DEFAULT_CONFIGS[subroutine]


def _shape_str(config: dict[str, Any]) -> str:
    return (
        f"B={config['batch']},T={config['seq']},I={config['input_size']},"
        f"H={config['hidden_size']},L={config['num_layers']}"
    )


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
        "torch": cudnn_fn,
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


def _bench_interleaved(
    functions: dict[str, Callable],
    dry_run_iters: int,
    num_iters: int,
    use_cuda_events: bool,
) -> dict[str, tuple[float, float]]:
    """Alternate backend order across rounds to neutralize allocator warmth."""
    names = list(functions)
    rounds = min(5, max(1, num_iters // 4))
    iterations = max(1, num_iters // rounds)
    samples: dict[str, list[float]] = {name: [] for name in names}
    for round_index in range(rounds):
        offset = round_index % len(names)
        order = names[offset:] + names[:offset]
        for name in order:
            median_ms, _ = bench_fn(
                functions[name],
                dry_run_iters=dry_run_iters if round_index == 0 else 0,
                num_iters=iterations,
                use_cuda_events=use_cuda_events,
            )
            samples[name].append(median_ms)
    return {
        name: (
            statistics.median(values),
            statistics.stdev(values) if len(values) > 1 else 0.0,
        )
        for name, values in samples.items()
    }


def run_test(args: argparse.Namespace, output: OutputWriter) -> None:
    subroutine = getattr(args, "subroutine", None) or "lstm"
    if subroutine not in SUBROUTINES:
        raise ValueError(f"Unknown recurrent subroutine: {subroutine}")
    dtype_str = getattr(args, "dtype", "float16")
    dtype = parse_dtype(dtype_str)
    backends = getattr(args, "backends", None) or ["oasr", "cudnn"]

    for config in _resolve_configs(args, subroutine):
        functions, cudnn_fn = _setup(subroutine, config, dtype)
        shape = _shape_str(config)
        if getattr(args, "refcheck", False):
            try:
                for backend in backends:
                    if backend not in functions:
                        continue
                    max_diff = _assert_close(subroutine, functions[backend](), cudnn_fn())
                    output.write_verbose(f"{shape},backend={backend}: max_abs_diff={max_diff:.6f}")
            except AssertionError as error:
                print(f"  [ERROR] Output mismatch for {shape}: {error}")
                if not getattr(args, "allow_output_mismatch", False):
                    continue

        selected = {backend: functions[backend] for backend in backends if backend in functions}
        for backend in backends:
            if backend not in functions:
                print(f"  [WARNING] Unknown backend {backend!r}, skipping")
        timings = _bench_interleaved(
            selected,
            dry_run_iters=getattr(args, "dry_run_iters", 5),
            num_iters=getattr(args, "num_iters", 30),
            use_cuda_events=getattr(args, "use_cuda_events", False),
        )
        for backend, (median_ms, std_ms) in timings.items():
            tflops = _flops(subroutine, config) / (median_ms * 1e-3) / 1e12
            output.write_result(
                BenchResult(
                    routine="recurrent",
                    subroutine=subroutine,
                    backend="cudnn" if backend == "torch" else backend,
                    shape=shape,
                    dtype=dtype_str,
                    median_ms=median_ms,
                    std_ms=std_ms,
                    tflops=tflops,
                )
            )


def run_standalone(variant: str = "lstm") -> None:
    if variant not in SUBROUTINES:
        raise ValueError(f"Unknown recurrent variant: {variant}")
    raise SystemExit(
        "Use: python benchmarks/oasr_benchmark.py --routine recurrent "
        f"--subroutine {variant} --backends oasr cudnn --refcheck"
    )
