#!/usr/bin/env python3
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""The OASR benchmark CLI -- kernels, decoders, engine, service and accuracy.

    python benchmarks/run.py --list
    python benchmarks/run.py --family gemm --subroutine bmm --backends cutlass torch \\
        --batch-count 256 --M 200 --N 200 --K 64 --dtype float16 --refcheck
    python benchmarks/run.py --testlist benchmarks/testlists/all_kernels.txt --output out.csv
    python benchmarks/run.py --family engine --ckpt-dir "$CKPT_DIR" --audio-dir "$AUDIO_DIR"

Kernel families emit the kernel schema; engine / service / accuracy / decoder
families emit the workload schema.  ``--print-schema`` lists both.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from benchmarks.core import registry  # noqa: E402
from benchmarks.core.driver import run_sweep  # noqa: E402
from benchmarks.core.report import Reporter, print_schema  # noqa: E402
from benchmarks.core.schema import CATEGORY_KERNEL  # noqa: E402
from benchmarks.core.timing import TIMER_CUDA_EVENTS, TIMER_TRITON  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="OASR benchmark CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # `--routine` is the name the testlists and the older docs use.
    p.add_argument(
        "--family", "--routine", dest="family", default=None, help="Benchmark family (see --list)"
    )
    p.add_argument("--subroutine", default=None, help="Specific benchmark within the family")
    p.add_argument(
        "--backends",
        nargs="+",
        default=None,
        help="Backends to measure (default: all the family offers)",
    )
    p.add_argument(
        "--ref-backend",
        "--ref_backend",
        dest="ref_backend",
        default=None,
        help="Backend that --refcheck and speedup_vs_ref compare against",
    )
    p.add_argument("--dtype", default="float16", help="float16, bfloat16 or float32")

    p.add_argument(
        "--iters",
        "--num_iters",
        dest="iters",
        type=int,
        default=30,
        help="Measured iterations (milliseconds when --timer triton)",
    )
    p.add_argument(
        "--warmup-iters",
        "--dry_run_iters",
        dest="warmup_iters",
        type=int,
        default=5,
        help="Warmup iterations",
    )
    p.add_argument(
        "--timer",
        choices=[TIMER_CUDA_EVENTS, TIMER_TRITON],
        default=TIMER_CUDA_EVENTS,
        help="Timing loop. cuda_events counts iterations; triton budgets milliseconds",
    )
    p.add_argument(
        "--use_cuda_events",
        action="store_true",
        help="Deprecated: cuda_events is the default timer",
    )
    p.add_argument(
        "--min-measure-ms",
        type=float,
        default=20.0,
        help="Wall-time floor per measurement; raises the iteration count for fast kernels (0 disables)",
    )
    p.add_argument(
        "--no-flush",
        action="store_true",
        help="Do not evict L2 between iterations (measures a warm cache)",
    )
    interleave = p.add_mutually_exclusive_group()
    interleave.add_argument(
        "--interleave",
        dest="interleave",
        action="store_true",
        default=None,
        help="Rotate backend order between rounds",
    )
    interleave.add_argument(
        "--no-interleave",
        dest="interleave",
        action="store_false",
        help="Measure each backend in one contiguous block",
    )

    p.add_argument("--refcheck", action="store_true", help="Compare backends against --ref-backend")
    p.add_argument(
        "--allow-output-mismatch",
        "--allow_output_mismatch",
        dest="allow_output_mismatch",
        action="store_true",
        help="Keep benchmarking after a refcheck failure",
    )
    p.add_argument(
        "--autotune",
        action="store_true",
        help="Profile and cache the best kernel config for each shape",
    )
    p.add_argument("--cache", default=None, help="Autotune cache JSON")
    p.add_argument(
        "--no-tune",
        action="store_true",
        help="With --autotune, load the cache only and skip profiling",
    )
    p.add_argument(
        "--profile",
        action="store_true",
        help="One NVTX-wrapped iteration per backend, for ncu / nsys",
    )

    p.add_argument(
        "--output",
        "--output-path",
        "--output_path",
        dest="output",
        default=None,
        help="CSV path; a .meta.json run manifest is written beside it",
    )
    p.add_argument(
        "--case-tag",
        "--case_tag",
        dest="case_tag",
        default=None,
        help="Label for the rows, for A/B comparisons",
    )
    p.add_argument("--testlist", default=None, help="File of CLI invocations, one per line")
    p.add_argument(
        "--generate_repro_command",
        action="store_true",
        help="Deprecated: every row carries its own repro_command",
    )
    p.add_argument("-v", action="count", default=0, dest="verbosity", help="-v, -vv")
    p.add_argument("--list", action="store_true", help="List families and subroutines")
    p.add_argument("--print-schema", action="store_true", help="Print both output schemas")
    return p


#: Flags that belong to the run, not to one testlist line.  Every one of these
#: is inherited by each line -- the previous batch mode inherited six and
#: silently dropped --dtype, --iters, --warmup-iters and --backends.
_GLOBAL_ONLY = (
    "output",
    "case_tag",
    "verbosity",
    "testlist",
    "list",
    "print_schema",
    "generate_repro_command",
)
_INHERITED = (
    "backends",
    "ref_backend",
    "dtype",
    "iters",
    "warmup_iters",
    "timer",
    "no_flush",
    "interleave",
    "refcheck",
    "allow_output_mismatch",
    "profile",
    "use_cuda_events",
    "autotune",
    "cache",
    "no_tune",
    "min_measure_ms",
)


def _repro_command(args: argparse.Namespace, extra: list) -> str:
    parts = ["python benchmarks/run.py", f"--family {args.family}"]
    if args.subroutine:
        parts.append(f"--subroutine {args.subroutine}")
    if args.backends:
        parts.append("--backends " + " ".join(args.backends))
    parts += [
        f"--dtype {args.dtype}",
        f"--iters {args.iters}",
        f"--warmup-iters {args.warmup_iters}",
        f"--timer {args.timer}",
    ]
    if args.refcheck:
        parts.append("--refcheck")
    parts += extra
    return " ".join(parts)


def run_one(args: argparse.Namespace, extra: list, reporter: Reporter) -> None:
    """Parse the family's own flags, then hand off to the sweep or the harness."""
    module = registry.get_family(args.family)

    sub_parser = argparse.ArgumentParser(prog=f"--family {args.family}", add_help=False)
    module.parse_args(sub_parser)
    # Strict, not parse_known_args: an unrecognised flag used to be discarded,
    # after which resolve_configs saw None, fell back to DEFAULT_CONFIGS, and the
    # line ran a full default sweep at the wrong shapes -- reported as a pass.
    family_args = sub_parser.parse_args(extra)
    merged = argparse.Namespace(**{**vars(args), **vars(family_args)})

    if merged.subroutine is None:
        merged.subroutine = module.SUBROUTINES[0]
    if merged.subroutine not in module.SUBROUTINES:
        raise SystemExit(
            f"[ERROR] Unknown subroutine '{merged.subroutine}' for family "
            f"'{args.family}'. Available: {module.SUBROUTINES}"
        )

    repro = _repro_command(merged, extra)
    reporter.repro(repro)

    if hasattr(module, "run"):  # workload harness
        module.run(merged, reporter, repro_command=repro)
    else:
        run_sweep(registry.resolve(args.family), module, merged, reporter, repro_command=repro)


def main() -> int:
    parser = build_parser()
    args, extra = parser.parse_known_args()

    if args.print_schema:
        print_schema()
        return 0
    if args.list:
        print("Available families and subroutines:\n")
        for family, subs in sorted(registry.list_subroutines().items()):
            kind = "kernel" if family in registry.KERNEL_FAMILIES else "workload"
            print(f"  {family}  [{kind}]")
            for sub in subs:
                print(f"    - {sub}")
        return 0
    if args.use_cuda_events:
        args.timer = TIMER_CUDA_EVENTS
    # Whether --dtype was typed decides if a family's own default applies.
    args.dtype_explicit = any(a == "--dtype" or a.startswith("--dtype=") for a in sys.argv[1:])

    if not args.family and not args.testlist:
        parser.print_help()
        return 0

    category = (
        CATEGORY_KERNEL
        if args.testlist and not args.family
        else registry.category_of(args.family or "gemm")
    )
    reporter = Reporter(
        output=args.output,
        category=category,
        case_tag=args.case_tag or "",
        verbosity=args.verbosity,
    )
    if args.verbosity >= 2:
        from benchmarks.core import env

        info = env.device_info()
        print(f"[VVERBOSE] gpu_name = '{info['name']}'")
        print(f"[VVERBOSE] sm = {info['sm']}, memory = {info['memory_gb']} GB")

    try:
        if args.testlist:
            lines = _read_testlist(args.testlist)
            print(f"[INFO] Running {len(lines)} tests from {args.testlist}")
            for i, tokens in enumerate(lines, 1):
                line_args, line_extra = parser.parse_known_args(tokens)
                for name in _GLOBAL_ONLY:
                    setattr(line_args, name, getattr(args, name))
                line_args.dtype_explicit = (
                    any(t == "--dtype" or t.startswith("--dtype=") for t in tokens)
                    or args.dtype_explicit
                )
                for name in _INHERITED:
                    if getattr(line_args, name) == parser.get_default(name):
                        setattr(line_args, name, getattr(args, name))
                if not line_args.family:
                    reporter.warn(f"testlist line {i} has no --family/--routine, skipping")
                    continue
                reporter.header(
                    f"[{i}/{len(lines)}] {line_args.family}/{line_args.subroutine or 'default'}"
                )
                run_one(line_args, line_extra, reporter)
        else:
            reporter.header(f"{args.family}/{args.subroutine or 'default'}")
            run_one(args, extra, reporter)
    finally:
        reporter.finalize()
    return 0


def _read_testlist(path: str) -> list:
    lines = []
    for n, raw in enumerate(Path(path).read_text().splitlines(), 1):
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        try:
            lines.append(shlex.split(text))
        except ValueError as exc:
            raise SystemExit(f"[ERROR] {path}:{n}: {exc}") from None
    return lines


if __name__ == "__main__":
    sys.exit(main())
