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

SUBROUTINES = ["lstm", "rnn_tanh", "rnn_relu"]

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

DEFAULT_CONFIGS: dict[str, list[dict[str, Any]]] = {
    subroutine: [
        {
            "batch": batch,
            "seq": sequence,
            "input_size": hidden,
            "hidden_size": hidden,
            "num_layers": layers,
        }
        for batch, sequence, hidden, layers in _SHAPES
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
    actual_tensors = (actual[0], *actual[1]) if subroutine == "lstm" else actual
    expected_tensors = (expected[0], *expected[1]) if subroutine == "lstm" else expected
    max_diff = 0.0
    for got, ref in zip(actual_tensors, expected_tensors):
        max_diff = max(max_diff, (got.float() - ref.float()).abs().max().item())
        torch.testing.assert_close(got, ref, rtol=3e-2, atol=3e-2)
    return max_diff


def _flops(subroutine: str, config: dict[str, Any]) -> int:
    gates = 4 if subroutine == "lstm" else 1
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
