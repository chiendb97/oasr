# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""torch.compile (Inductor) as a fusion reference -- a benchmark, never a runner.

``oasr tune compare-inductor`` asks one question per block an encoder repeats
(a layer, and the modules inside it): *does Inductor find a fusion, or a GEMM
tile, that OASR's kernels miss?*  It compiles the block's torch path
(``OASR_LAYERS_BACKEND=torch``) and times it against the OASR path with the
protocol every tuner here uses (:func:`oasr.tune.bench.measure`), then writes a
report whose to-do list names the blocks where Inductor is ahead and the
kernels it launched to get there.  Nothing here writes a tuning file or reaches
dispatch: an Inductor win is kernel work, not a choice to serve.

How a block is found and fed:

* **Blocks** are the encoder's non-leaf modules whose class repeats (the layer
  and its feed-forward / attention / convolution modules), one per class and
  weight shape -- so a Zipformer stack of another width is its own block.  A
  leaf is one kernel, and the GEMM and conv tables already own those.
* **Arguments** are recorded, not synthesised: one offline encoder forward on
  the CPU (fp32, real weights) with a pre-hook on each block, stopped once every
  block has been reached.  Masks, positional embeddings and keyword arguments
  are what the model really passes, for any architecture.

The arms, each called on the block in the served dtype on the current GPU:

* ``oasr`` -- what serves: the OASR kernels.
* ``torch`` -- the layers' torch path, eager: what Inductor starts from.
* ``inductor`` -- ``torch.compile`` of the torch path, default mode: fusion of
  the pointwise and reduction glue, GEMMs left to ATen (cuBLAS).
* ``inductor-max`` -- ``max-autotune-no-cudagraphs`` (coordinate-descent
  tuning) with the requested GEMM backends (default ATEN, TRITON, CUTLASS,
  CUTEDSL): Triton GEMM templates with fused epilogues, and whatever the CUTLASS
  / CuTeDSL generators offer on this architecture.  Which backends offered a
  candidate and which won is recorded per autotuned op, so a backend that
  cannot generate for this GPU shows up as offering nothing.

The timed graph is the harness's own (``bench.measure`` captures each arm), so
Inductor's CUDA graphs stay off and every arm is compared on GPU time with the
same L2 treatment: the block's weights are rotated through
:func:`~oasr.tune.bench.l2_copies` copies sized by the model's weight bytes, and
one compiled graph serves every copy (parameters are graph inputs).  A call
Inductor cannot trace -- an OASR kernel the torch path still reaches, such as
the fused attention -- is a graph break: it runs as itself inside the compiled
arm, the break is listed in the report, and Inductor's number is the
conservative one (it cannot fuse across a break).

Numerics: every arm is compared with the block in fp32 (torch path) and gated
with :func:`~oasr.tune.bench.numerics_ok` against the eager torch arm's own
error, so a win bought with different arithmetic is reported, not listed.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import functools
import json
import logging
import math
import re
import time
import warnings
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
from torch import nn

from . import bench

__all__ = [
    "ARMS",
    "COMPILED_ARMS",
    "DEFAULT_BACKENDS",
    "DEFAULT_BATCHES",
    "DEFAULT_SECONDS",
    "GAP",
    "ArmResult",
    "BlockResult",
    "discover_blocks",
    "encoder_inputs",
    "capture_calls",
    "measure_block",
    "shape_ladder",
    "failed_block",
    "autotune_subproc",
    "compare",
    "available_backends",
    "fused_ops",
    "compute_ops",
    "compact_kernel_name",
    "kernel_inventory",
    "summarize",
    "render_markdown",
    "save_report",
]

#: Every arm, in report order.
ARMS: Tuple[str, ...] = ("oasr", "torch", "inductor", "inductor-max")
#: The arms ``torch.compile`` builds.
COMPILED_ARMS: Tuple[str, ...] = ("inductor", "inductor-max")
#: Inductor's GEMM backends the ``inductor-max`` arm asks for.
DEFAULT_BACKENDS: Tuple[str, ...] = ("ATEN", "TRITON", "CUTLASS", "CUTEDSL")
DEFAULT_BATCHES: Tuple[int, ...] = (1, 8, 32)
DEFAULT_SECONDS: Tuple[float, ...] = (4.0, 16.0)
#: An arm is ahead when it is this much faster (and the difference clears the noise).
GAP = 0.05
#: Upper bound on the bytes of rotated block copies (this box shares its GPUs).
DEFAULT_MAX_COPY_BYTES = 256 << 20

# =============================================================================
# Discovery and argument capture
# =============================================================================


def _param_signature(module: nn.Module) -> Tuple[Tuple[str, Tuple[int, ...]], ...]:
    return tuple((n, tuple(p.shape)) for n, p in module.named_parameters())


def _is_stack(module: nn.Module) -> bool:
    """Whether *module* holds a list of repeated layers (a Zipformer stack): its
    layers are blocks of their own, so timing the stack only times them again."""
    for sub in module.modules():
        if isinstance(sub, nn.ModuleList) and len(sub) >= 2:
            kinds = Counter(type(m) for m in sub if next(m.children(), None) is not None)
            if kinds and max(kinds.values()) >= 2:
                return True
    return False


def discover_blocks(
    encoder: nn.Module, select: Optional[Sequence[str]] = None
) -> List[Tuple[str, nn.Module]]:
    """The encoder's repeated non-leaf modules, one per (class, weight shapes).

    A stack of layers is not a block (:func:`_is_stack`); its layers are.
    ``select`` keeps only blocks whose class name, or a part of whose module
    path, matches one of its entries.  Returned in module order (a layer before
    the modules inside it).
    """
    counts: Counter = Counter(type(m) for _, m in encoder.named_modules())
    seen: set = set()
    out: List[Tuple[str, nn.Module]] = []
    for path, mod in encoder.named_modules():
        if not path or counts[type(mod)] < 2:
            continue
        if isinstance(mod, (nn.ModuleList, nn.ModuleDict)):
            continue
        if next(mod.children(), None) is None or next(mod.parameters(), None) is None:
            continue
        key = (type(mod), _param_signature(mod))
        if key in seen or _is_stack(mod):
            continue
        if select and not any(s == type(mod).__name__ or s in path for s in select):
            continue
        seen.add(key)
        out.append((path, mod))
    return out


def encoder_inputs(
    feature_spec: Any, batch: int, seconds: float, *, seed: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Random features shaped like *seconds* of audio for the encoder: ``(xs, lens, T)``.

    The width is the spec's feature dim times its low-frame-rate stack; the
    frame count follows the frame shift and LFR stride, except for a
    fixed-window frontend (Whisper, Qwen2-Audio), which always feeds its window.
    """
    dim = int(getattr(feature_spec, "feature_dim", 80)) * int(getattr(feature_spec, "lfr_m", 1))
    shift_ms = float(getattr(feature_spec, "frame_shift_ms", 10.0))
    lfr_n = int(getattr(feature_spec, "lfr_n", 1))
    frames = max(1, int(round(seconds * 1000.0 / shift_ms / lfr_n)))
    fixed = None
    if feature_spec is not None and hasattr(feature_spec, "to_feature_config"):
        try:
            fixed = feature_spec.to_feature_config().fixed_window_frames
        except Exception:  # noqa: BLE001 -- a spec with no config mapping: free length
            fixed = None
    if fixed:
        frames = int(fixed)
    g = torch.Generator().manual_seed(seed)
    xs = torch.randn(batch, frames, dim, generator=g)
    lens = torch.full((batch,), frames, dtype=torch.int32)
    return xs, lens, frames


class _Reached(BaseException):
    """Every block has been reached: stop the capture forward (not an ``Exception``)."""


def _map_tensors(v: Any, fn: Callable[[torch.Tensor], torch.Tensor]) -> Any:
    """*v* with *fn* applied to every tensor in it, containers rebuilt as their own
    type: a NamedTuple stays one (Nemotron's ``AttentionBlocks`` is read by
    attribute), and so does a dataclass."""
    if isinstance(v, torch.Tensor):
        return fn(v)
    if isinstance(v, tuple):
        items = [_map_tensors(x, fn) for x in v]
        return type(v)(*items) if hasattr(v, "_fields") else tuple(items)
    if isinstance(v, list):
        return [_map_tensors(x, fn) for x in v]
    if isinstance(v, dict):
        return {k: _map_tensors(x, fn) for k, x in v.items()}
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        changes = {
            f.name: _map_tensors(getattr(v, f.name), fn)
            for f in dataclasses.fields(v)
            if f.init and hasattr(v, f.name)
        }
        return dataclasses.replace(v, **changes)
    return v


def _clone(v: Any) -> Any:
    return _map_tensors(v, lambda t: t.detach().clone())


def capture_calls(
    encoder: nn.Module,
    blocks: Sequence[Tuple[str, nn.Module]],
    xs: torch.Tensor,
    lens: torch.Tensor,
) -> Dict[str, Tuple[tuple, dict]]:
    """Run *encoder* offline on *xs* (CPU, torch path) and record each block's first call.

    Returns ``{path: (args, kwargs)}`` with tensors cloned, for the blocks the
    forward reached.
    """
    from oasr.layers._backend import layers_backend_override

    wanted = {id(m): p for p, m in blocks}
    calls: Dict[str, Tuple[tuple, dict]] = {}

    def hook(mod: nn.Module, args: tuple, kwargs: dict) -> None:
        path = wanted.get(id(mod))
        if path is None or path in calls:
            return
        calls[path] = (_clone(args), _clone(kwargs))
        if len(calls) == len(wanted):
            raise _Reached

    handles = [m.register_forward_pre_hook(hook, with_kwargs=True) for _, m in blocks]
    try:
        with torch.no_grad(), layers_backend_override("torch"):
            encoder(xs, lens)
    except _Reached:
        pass
    finally:
        for h in handles:
            h.remove()
    return calls


# =============================================================================
# Inductor bookkeeping
# =============================================================================


def _cutlass_dir() -> Optional[str]:
    """The repo's CUTLASS checkout when it carries the Python generator Inductor needs."""
    root = Path(__file__).resolve().parents[2] / "3rdparty" / "cutlass"
    return str(root) if (root / "python" / "cutlass_library").is_dir() else None


#: ``None`` until :func:`_preload_cutlass` ran; then ``""`` (importable) or why not.
_CUTLASS_IMPORT: Optional[str] = None


def _preload_cutlass(cutlass_dir: str) -> str:
    """Import the CUTLASS Python package Inductor's CUTLASS backend uses, from a
    scratch directory; ``""`` when it imports, else the import error.

    ``cutlass_cppgen`` opens ``compiled_cache.db`` relative to the working
    directory when it is imported -- even when the import then fails -- so
    Inductor's first attempt would leave one wherever the command ran.  Tried
    once, here, the file lands in a directory that is then removed.
    """
    global _CUTLASS_IMPORT
    if _CUTLASS_IMPORT is not None:
        return _CUTLASS_IMPORT
    import os
    import tempfile

    from torch._inductor import config as inductor_config

    status = ""
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="oasr-cutlass-") as scratch:
        os.chdir(scratch)
        try:
            with inductor_config.patch({"cutlass.cutlass_dir": cutlass_dir}):
                from torch._inductor.codegen.cutlass.utils import try_import_cutlass

                if not try_import_cutlass():
                    status = "Inductor could not import cutlass_library"
                    try:  # Inductor logs the reason at DEBUG only: get it ourselves
                        import cutlass_cppgen  # type: ignore[import-not-found]  # noqa: F401
                    except Exception as exc:  # noqa: BLE001
                        status = f"{type(exc).__name__}: {exc}"[:200]
        except Exception as exc:  # noqa: BLE001
            status = f"{type(exc).__name__}: {exc}"[:200]
        finally:
            os.chdir(cwd)
    _CUTLASS_IMPORT = status
    return status


def available_backends(requested: Sequence[str]) -> Dict[str, str]:
    """Per requested Inductor GEMM backend: ``"ok"`` or why it cannot offer anything here.

    Informational: an unavailable backend is still passed to Inductor (which
    skips it), and the autotune record shows what each one actually offered.
    """
    import importlib.util

    out: Dict[str, str] = {}
    for b in (x.strip().upper() for x in requested):
        if b in ("ATEN", "TRITON"):
            out[b] = "ok"
        elif b == "CUTLASS":
            cdir = _cutlass_dir()
            if cdir is None:
                out[b] = "no CUTLASS checkout with python/cutlass_library"
            else:
                err = _preload_cutlass(cdir)
                out[b] = f"ok ({cdir})" if not err else f"Inductor cannot import it: {err}"
        elif b == "CUTEDSL":
            out[b] = (
                "ok (Inductor uses it for grouped GEMMs on datacenter Blackwell only)"
                if importlib.util.find_spec("cutlass") is not None
                else "CuTeDSL is not installed"
            )
        elif b == "NVGEMM":
            out[b] = (
                "ok" if importlib.util.find_spec("cutlass_api") else "cutlass_api not installed"
            )
        else:
            out[b] = "unknown to this harness (passed through)"
    return out


def _backend_of(choice: Any) -> str:
    cls = type(choice).__name__
    if "Extern" in cls:
        return "ATEN"
    if "Triton" in cls:
        return "TRITON"
    if "CuteDSL" in cls or "CuTeDSL" in cls:
        return "CUTEDSL"
    if "NVUniversal" in cls:
        return "NVGEMM"
    if "CUTLASS" in cls or "Cutlass" in cls:
        return "CUTLASS"
    if "Subgraph" in cls:
        return "SUBGRAPH"
    return cls


#: Where the feedback saver appends while a compiled arm compiles (``None``: idle).
_SINK: Optional[List[Dict[str, Any]]] = None
_SAVER_INSTALLED = False


def _node_size(node: Any) -> List[Any]:
    try:
        return [int(s) if str(s).lstrip("-").isdigit() else str(s) for s in node.get_size()]
    except Exception:  # noqa: BLE001 -- informational only
        return []


def _feedback(timings: Dict[Any, float], name: str, input_nodes: Any, *_rest: Any) -> None:
    if _SINK is None or not timings:
        return
    offered: Counter = Counter()
    best: Dict[str, float] = {}
    for choice, ms in timings.items():
        b = _backend_of(choice)
        offered[b] += 1
        if math.isfinite(ms) and ms < best.get(b, math.inf):
            best[b] = float(ms)
    finite = [(ms, c) for c, ms in timings.items() if math.isfinite(ms)]
    if not finite:
        return
    win_ms, winner = min(finite, key=lambda t: t[0])
    _SINK.append(
        {
            "op": str(name),
            "inputs": [_node_size(n) for n in (input_nodes or [])],
            "offered": dict(offered),
            "best_ms": best,
            "winner": str(getattr(winner, "name", type(winner).__name__)),
            "winner_backend": _backend_of(winner),
            "winner_ms": float(win_ms),
        }
    )


def _install_saver() -> None:
    global _SAVER_INSTALLED
    if _SAVER_INSTALLED:
        return
    from torch._inductor import select_algorithm

    select_algorithm.add_feedback_saver(_feedback)
    _SAVER_INSTALLED = True


def _compile_options(arm: str, backends: Sequence[str], subproc: bool = False) -> Dict[str, Any]:
    """Inductor options for *arm*: no Inductor CUDA graphs, no caches (so every
    compile is real, timed, and reports its autotune choices).

    *subproc* benchmarks max-autotune's candidates in Inductor's tuning
    subprocess.  In-process (Inductor's default) every candidate's module stays
    loaded, and a layer's few hundred of them outgrew the ~2 GiB a shared GPU
    left (OOM at the first layer); in the subprocess, Nemotron's layer on sm120
    left a misaligned-address fault in this process.  :func:`autotune_subproc`
    picks by free memory.
    """
    opts: Dict[str, Any] = {
        "triton.cudagraphs": False,
        "fx_graph_cache": False,
        "autotune_local_cache": False,
        "autotune_remote_cache": False,
    }
    if arm == "inductor-max":
        import torch._inductor as ind

        opts.update(ind.list_mode_options("max-autotune-no-cudagraphs"))
        opts["max_autotune_gemm_backends"] = ",".join(b.strip().upper() for b in backends)
        opts["max_autotune_report_choices_stats"] = False
        opts["autotune_num_choices_displayed"] = 0
        opts["autotune_in_subproc"] = bool(subproc)
        cdir = _cutlass_dir()
        if cdir:
            opts["cutlass.cutlass_dir"] = cdir
    return opts


#: Below this much free device memory, candidates are benchmarked in a subprocess.
SUBPROC_BELOW_FREE_BYTES = 8 << 30


def autotune_subproc(mode: str = "auto") -> bool:
    """``on`` / ``off``, or ``auto``: the subprocess when the device has less than
    :data:`SUBPROC_BELOW_FREE_BYTES` free (see :func:`_compile_options`)."""
    if mode not in ("auto", "on", "off"):
        raise ValueError(f"autotune subprocess mode must be auto, on or off, not {mode!r}")
    if mode != "auto":
        return mode == "on"
    free = _free_device_bytes()
    if free is None:
        free, _ = torch.cuda.mem_get_info()
    return free < SUBPROC_BELOW_FREE_BYTES


def _free_device_bytes() -> Optional[int]:
    """Free memory of the first visible GPU through NVML, without creating a CUDA
    context in this process (an isolating parent leaves the GPU to its children)."""
    import os

    try:
        import pynvml

        pynvml.nvmlInit()
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        entry = visible.split(",")[0].strip() if visible else "0"
        handle = (
            pynvml.nvmlDeviceGetHandleByUUID(entry)
            if entry.startswith("GPU-")
            else pynvml.nvmlDeviceGetHandleByIndex(int(entry))
        )
        return int(pynvml.nvmlDeviceGetMemoryInfo(handle).free)
    except Exception:  # noqa: BLE001 -- no NVML: the caller asks CUDA instead
        return None


@contextlib.contextmanager
def _quiet_inductor(verbose: bool) -> Iterator[None]:
    """Hush Inductor's per-candidate errors (a Triton config over this GPU's shared
    memory is logged at ERROR and skipped) unless *verbose*."""
    if verbose:
        yield
        return
    from torch._inductor import select_algorithm

    names = ("torch._inductor", "torch._inductor.select_algorithm", "torch._dynamo")
    loggers = [logging.getLogger(n) for n in names]
    levels = [lg.level for lg in loggers]
    printing = select_algorithm.PRINT_AUTOTUNE
    for lg in loggers:
        lg.setLevel(logging.CRITICAL)
    select_algorithm.PRINT_AUTOTUNE = False  # the per-op candidate tables on stderr
    try:
        with warnings.catch_warnings():
            # "Failed to benchmark choice" (a Triton config over this GPU's shared
            # memory) and Dynamo tracing through OASR's lru_cache'd lane lookups.
            warnings.filterwarnings(
                "ignore", category=UserWarning, module=r"torch\._(inductor|dynamo)"
            )
            yield
    finally:
        select_algorithm.PRINT_AUTOTUNE = printing
        for lg, lv in zip(loggers, levels):
            lg.setLevel(lv)


def _break_reason(key: str) -> str:
    lines = [ln.strip() for ln in str(key).splitlines() if ln.strip()]
    head = lines[0] if lines else str(key)
    ctx = next((ln for ln in lines if ln.startswith("Developer debug context")), "")
    ctx = ctx.replace("Developer debug context:", "").strip()
    ctx = re.sub(r"LazyVariableTracker\(unrealized: <class '([^']+)'>\)", r"\1", ctx)
    return (head + (f" -- {ctx}" if ctx else ""))[:200]


def _dynamo_snapshot() -> Tuple[int, Counter]:
    from torch._dynamo.utils import counters

    return int(counters["stats"]["unique_graphs"]), Counter(counters["graph_break"])


# =============================================================================
# Measurement
# =============================================================================


@dataclass
class ArmResult:
    arm: str
    status: str = "ok"
    error: str = ""
    #: Per-call GPU time (median over rounds) and 1.4826 x MAD, in us.
    us: float = math.inf
    sigma_us: float = 0.0
    #: ``max |out - fp32 reference|`` and the gate against the eager torch arm.
    err: Optional[float] = None
    numerics_ok: Optional[bool] = None
    #: Kernels one call launches, by time (see :func:`kernel_inventory`).
    kernels: List[Dict[str, Any]] = field(default_factory=list)
    launches: int = 0
    #: Compiled arms only.
    compile_s: Optional[float] = None
    graphs: Optional[int] = None
    breaks: List[str] = field(default_factory=list)
    autotune: List[Dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        d = asdict(self)
        d["us"] = None if not math.isfinite(self.us) else self.us
        return d

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "ArmResult":
        d = dict(d)
        d["us"] = math.inf if d.get("us") is None else float(d["us"])
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class BlockResult:
    block: str
    kind: str
    batch: int
    seconds: float
    frames: int
    inputs: List[List[int]]
    copies: int
    arms: Dict[str, ArmResult]
    conditions: Dict[str, Any] = field(default_factory=dict)

    def best_compiled(self) -> Optional[str]:
        """The fastest compiled arm whose numerics pass."""
        ok = [
            a
            for a in COMPILED_ARMS
            if a in self.arms
            and self.arms[a].status == "ok"
            and self.arms[a].numerics_ok is not False
            and math.isfinite(self.arms[a].us)
        ]
        return min(ok, key=lambda a: self.arms[a].us) if ok else None

    def speedup(self) -> Optional[float]:
        """OASR time / best compiled time (> 1: Inductor ahead)."""
        best = self.best_compiled()
        o = self.arms.get("oasr")
        if best is None or o is None or o.status != "ok" or not math.isfinite(o.us):
            return None
        return o.us / self.arms[best].us

    def verdict(self, gap: float = GAP) -> str:
        """``inductor-ahead`` / ``even`` / ``oasr-ahead`` / ``numerics`` / ``error``."""
        o = self.arms.get("oasr")
        if o is None or o.status != "ok" or not math.isfinite(o.us):
            return "error"
        best = self.best_compiled()
        if best is None:
            failed = [
                a
                for a in COMPILED_ARMS
                if a in self.arms
                and self.arms[a].status == "ok"
                and self.arms[a].numerics_ok is False
            ]
            return "numerics" if failed else "error"
        b = self.arms[best]
        noise = 3.0 * math.hypot(o.sigma_us, b.sigma_us)
        if o.us >= (1.0 + gap) * b.us and o.us - b.us > noise:
            return "inductor-ahead"
        if b.us >= (1.0 + gap) * o.us and b.us - o.us > noise:
            return "oasr-ahead"
        return "even"

    def to_json(self) -> Dict[str, Any]:
        d = asdict(self)
        d["arms"] = {k: v.to_json() for k, v in self.arms.items()}
        return d

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "BlockResult":
        d = dict(d)
        d["arms"] = {k: ArmResult.from_json(v) for k, v in d["arms"].items()}
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


def _to_device(v: Any, device: torch.device, dtype: torch.dtype) -> Any:
    return _map_tensors(
        v, lambda t: t.to(device, dtype=dtype) if t.is_floating_point() else t.to(device)
    )


def _first_tensor(out: Any) -> torch.Tensor:
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)):
        for x in out:
            if isinstance(x, torch.Tensor) and x.numel():
                return x
    raise TypeError(f"block returned no tensor ({type(out).__name__})")


def _shapes(args: tuple, kwargs: dict) -> List[List[int]]:
    vals = list(args) + list(kwargs.values())
    return [list(v.shape) for v in vals if isinstance(v, torch.Tensor)]


def _block_copy(module: nn.Module, device: torch.device, dtype: torch.dtype) -> nn.Module:
    m = copy.deepcopy(module).to(device=device, dtype=dtype).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _bind(mode: str, fn: Callable[..., Any], args: tuple, kwargs: dict) -> Callable[[], Any]:
    from oasr.layers._backend import layers_backend_override

    def call() -> Any:
        with torch.no_grad(), layers_backend_override(mode):
            return fn(*args, **kwargs)

    return call


def _demangle(names: Sequence[str]) -> Dict[str, str]:
    """``{mangled: demangled}`` through ``c++filt`` (``{}`` without it)."""
    import shutil
    import subprocess

    mangled = sorted({n for n in names if n.startswith("_Z")})
    tool = shutil.which("c++filt")
    if not mangled or tool is None:
        return {}
    try:
        res = subprocess.run(
            [tool], input="\n".join(mangled), capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    lines = res.stdout.splitlines()
    return dict(zip(mangled, lines)) if len(lines) == len(mangled) else {}


def _mangled_hint(name: str, keep: int = 8) -> str:
    """A readable label for a mangled name ``c++filt`` could not read (CUPTI
    truncates long template names): its first identifiers, and the first
    ``GemmShape`` it names.  ``_ZN7cutlass6KernelINS_4gemm...`` reads
    ``~cutlass::Kernel::gemm::kernel::Gemm::...``."""
    shape = re.search(r"GemmShapeILi(\d+)ELi(\d+)ELi(\d+)E", name)
    name = re.sub(r"L[a-z]\d+E", "", name)  # integer template arguments (Li8E)
    ids: List[str] = []
    i = 2 if name.startswith("_Z") else 0
    while i < len(name) and len(ids) < keep:
        if not name[i].isdigit():
            i += 1
            continue
        j = i
        while j < len(name) and name[j].isdigit():
            j += 1
        n = int(name[i:j])
        ident = name[j : j + n]
        if n >= 2 and len(ident) == n and re.fullmatch(r"[A-Za-z]\w*", ident):
            if ident not in ids:
                ids.append(ident)
            i = j + n
        else:
            i = j
    tail = f" GemmShape<{shape.group(1)}, {shape.group(2)}, {shape.group(3)}>" if shape else ""
    return "~" + "::".join(ids) + tail


def compact_kernel_name(name: str) -> str:
    """A kernel name a reader can scan: no ``void``, no parameter list, and
    template arguments beyond the second level elided (``A<B<...>, C>``)."""
    n = name[5:] if name.startswith("void ") else name
    kept: List[str] = []
    parens = 0
    for ch in n:  # parameter lists, and a lambda's enclosing signature
        if ch == "(":
            parens += 1
        elif ch == ")":
            parens = max(0, parens - 1)
        elif parens == 0:
            kept.append(ch)
    n = "".join(kept)
    out: List[str] = []
    depth = 0
    for ch in n:
        if ch == "<":
            depth += 1
            if depth == 1:
                out.append("<")
            elif depth == 2:
                out.append("<...")
            continue
        if ch == ">":
            if depth in (1, 2):
                out.append(">")
            depth = max(0, depth - 1)
            continue
        if depth <= 1:
            out.append(ch)
    return "".join(out).strip()


def _pick(calls: Sequence[Callable[[], Any]], i: int) -> Callable[[], Any]:
    """``Arm.make_call``: operand copy *i* of the rotation."""
    return calls[i % len(calls)]


def kernel_inventory(call: Callable[[], Any]) -> Tuple[List[Dict[str, Any]], int]:
    """The GPU kernels (and copies / memsets) one eager *call* launches, by time.

    Each row is ``{"name", "label", "op", "calls", "us"}``: ``op`` is the aten
    op that launched the kernel (``""`` for an OASR launcher or an Inductor
    kernel, which no aten op wraps) and ``label`` the demangled, compacted name.
    """
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Profiler clears events.*")
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            call()
            torch.cuda.synchronize()
    device: Dict[str, List[float]] = {}
    by_op: Dict[Tuple[str, str], List[float]] = {}
    for e in prof.events():
        if e.device_type == DeviceType.CUDA:
            row = device.setdefault(e.name, [0, 0.0])
            row[0] += 1
            row[1] += float(e.time_range.elapsed_us())
        elif e.name.startswith("aten::") and e.kernels:
            for k in e.kernels:
                row = by_op.setdefault((e.name, k.name), [0, 0.0])
                row[0] += 1
                row[1] += float(k.duration)
    attributed: Counter = Counter()
    rows: List[Dict[str, Any]] = []
    for (op, name), (c, t) in by_op.items():
        if name not in device:
            continue
        attributed[name] += int(c)
        rows.append({"name": name, "op": op, "calls": int(c), "us": t})
    for name, (c, t) in device.items():
        left = int(c) - attributed[name]
        if left > 0:
            rows.append({"name": name, "op": "", "calls": left, "us": t * left / c})
    pretty = {k: v for k, v in _demangle([r["name"] for r in rows]).items() if k != v}
    for r in rows:
        if r["name"] in pretty:
            r["label"] = compact_kernel_name(pretty[r["name"]])
        elif r["name"].startswith("_Z"):
            r["label"] = _mangled_hint(r["name"])
        else:
            r["label"] = compact_kernel_name(r["name"])
        r["us"] = round(r["us"], 2)
    rows.sort(key=lambda r: -r["us"])
    return rows, sum(r["calls"] for r in rows)


def measure_block(
    path: str,
    module: nn.Module,
    args: tuple,
    kwargs: dict,
    *,
    batch: int,
    seconds: float,
    frames: int,
    arms: Sequence[str] = ARMS,
    dtype: torch.dtype = torch.float16,
    backends: Sequence[str] = DEFAULT_BACKENDS,
    policy: Optional[bench.BenchPolicy] = None,
    working_set_bytes: Optional[int] = None,
    max_copy_bytes: int = DEFAULT_MAX_COPY_BYTES,
    inventory: bool = True,
    verbose: bool = False,
    subproc: bool = False,
    log: Callable[[str], None] = lambda s: None,
) -> BlockResult:
    """Measure one block at one shape in every arm (see the module docstring)."""
    global _SINK
    bad = [a for a in arms if a not in ARMS]
    if bad:
        raise ValueError(f"unknown arms {bad}; choose from {ARMS}")
    policy = policy or bench.BenchPolicy(solo=False)
    device = torch.device("cuda", torch.cuda.current_device())
    torch._dynamo.reset()

    block_bytes = (
        sum(p.numel() for p in module.parameters()) * torch.empty(0, dtype=dtype).element_size()
    )
    copies = bench.l2_copies(
        block_bytes, working_set_bytes=working_set_bytes, max_total_bytes=max_copy_bytes
    )
    mods = [_block_copy(module, device, dtype) for _ in range(copies)]
    a16 = _to_device(args, device, dtype)
    k16 = _to_device(kwargs, device, dtype)

    ref = _block_copy(module, device, torch.float32)
    want = _first_tensor(
        _bind(
            "torch",
            ref,
            _to_device(args, device, torch.float32),
            _to_device(kwargs, device, torch.float32),
        )()
    ).float()
    del ref

    results: Dict[str, ArmResult] = {a: ArmResult(a) for a in arms}
    makers: Dict[str, Callable[[int], Callable[[], Any]]] = {}

    for arm in arms:
        res = results[arm]
        try:
            if arm in COMPILED_ARMS:
                _install_saver()
                opts = _compile_options(arm, backends, subproc)
                if "cutlass.cutlass_dir" in opts and "CUTLASS" in opts.get(
                    "max_autotune_gemm_backends", ""
                ):
                    _preload_cutlass(opts["cutlass.cutlass_dir"])
                compiled = [torch.compile(m, dynamic=False, options=opts) for m in mods]
                graphs0, breaks0 = _dynamo_snapshot()
                sink: List[Dict[str, Any]] = []
                _SINK = sink
                t0 = time.perf_counter()
                try:
                    with _quiet_inductor(verbose):
                        out = _bind("torch", compiled[0], a16, k16)()
                        torch.cuda.synchronize()
                finally:
                    _SINK = None
                res.compile_s = round(time.perf_counter() - t0, 2)
                graphs1, breaks1 = _dynamo_snapshot()
                res.graphs = graphs1 - graphs0
                res.breaks = sorted({_break_reason(k) for k in breaks1 - breaks0})
                res.autotune = sink
                # Every copy once, so no guard is first evaluated inside a capture.
                for c in compiled[1:]:
                    _bind("torch", c, a16, k16)()
                calls = [_bind("torch", c, a16, k16) for c in compiled]
            else:
                calls = [_bind(arm, m, a16, k16) for m in mods]
                out = calls[0]()
            res.err = bench.numerics_error(_first_tensor(out), want)
            if inventory:
                res.kernels, res.launches = kernel_inventory(calls[0])
            makers[arm] = functools.partial(_pick, calls)
        except Exception as exc:  # noqa: BLE001 -- one arm failing must not lose the others
            torch.cuda.synchronize()
            res.status = "error"
            res.error = f"{type(exc).__name__}: {exc}"[:500]
            log(f"  {arm}: {res.error}")

    ref_err = None
    for name in ("torch", "oasr"):
        r = results.get(name)
        if r is not None and r.status == "ok" and r.err is not None:
            ref_err = r.err
            break
    for r in results.values():
        if r.status == "ok" and r.err is not None:
            r.numerics_ok = bench.numerics_ok(r.err, ref_err, dtype)

    timed = [bench.Arm(a, makers[a], forced=True) for a in arms if a in makers]
    meas, cond = bench.measure(timed, policy, copies=copies)
    for name, m in meas.items():
        r = results[name]
        if m.status != "ok":
            r.status, r.error = "error", m.error
            continue
        r.us = m.median_ms * 1e3
        r.sigma_us = m.sigma_ms * 1e3

    del makers, timed, mods
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    return BlockResult(
        block=path,
        kind=type(module).__name__,
        batch=batch,
        seconds=seconds,
        frames=frames,
        inputs=_shapes(args, kwargs),
        copies=copies,
        arms=results,
        conditions=cond.to_json(),
    )


def shape_ladder(
    feature_spec: Any, batches: Sequence[int], seconds: Sequence[float]
) -> List[Tuple[int, float, int]]:
    """``(batch, seconds, frames)`` per distinct shape: a fixed-window frontend
    gives every length the same frame count, so only one is kept."""
    out: List[Tuple[int, float, int]] = []
    seen: set = set()
    for b in batches:
        for s in seconds:
            frames = encoder_inputs(feature_spec, 1, s)[2]
            if (b, frames) not in seen:
                seen.add((b, frames))
                out.append((int(b), float(s), frames))
    return out


def failed_block(
    path: str,
    kind: str,
    batch: int,
    seconds: float,
    frames: int,
    arms: Sequence[str],
    reason: str,
) -> BlockResult:
    """The record of a block whose measurement could not finish (every arm errored)."""
    return BlockResult(
        block=path,
        kind=kind,
        batch=batch,
        seconds=seconds,
        frames=frames,
        inputs=[],
        copies=0,
        arms={a: ArmResult(a, status="error", error=reason[:500]) for a in arms},
    )


def compare(
    encoder: nn.Module,
    feature_spec: Any,
    *,
    batches: Sequence[int] = DEFAULT_BATCHES,
    seconds: Sequence[float] = DEFAULT_SECONDS,
    select: Optional[Sequence[str]] = None,
    paths: Optional[Sequence[str]] = None,
    arms: Sequence[str] = ARMS,
    dtype: torch.dtype = torch.float16,
    backends: Sequence[str] = DEFAULT_BACKENDS,
    policy: Optional[bench.BenchPolicy] = None,
    working_set_bytes: Optional[int] = None,
    max_copy_bytes: int = DEFAULT_MAX_COPY_BYTES,
    inventory: bool = True,
    verbose: bool = False,
    subproc: bool = False,
    log: Callable[[str], None] = print,
    on_start: Optional[Callable[[str], None]] = None,
    on_result: Optional[Callable[[BlockResult], None]] = None,
    on_skip: Optional[Callable[[str, str], None]] = None,
) -> List[BlockResult]:
    """Every discovered block of *encoder* (a CPU fp32 module) at every shape.

    ``paths`` keeps only blocks at exactly these module paths (``select`` matches
    loosely).  The callbacks report progress block by block -- what an isolating
    caller (``oasr tune compare-inductor``) needs to know which block a process
    died in.
    """
    blocks = discover_blocks(encoder, select)
    if paths is not None:
        wanted = set(paths)
        blocks = [(p, m) for p, m in blocks if p in wanted]
    if not blocks:
        wanted_by = list(select or paths or [])
        raise ValueError(
            "no repeated block found" + (f" matching {wanted_by}" if wanted_by else "")
        )
    if working_set_bytes is None:
        elt = torch.empty(0, dtype=dtype).element_size()
        working_set_bytes = sum(p.numel() for p in encoder.parameters()) * elt
    out: List[BlockResult] = []
    for b, s, frames in shape_ladder(feature_spec, batches, seconds):
        xs, lens, _ = encoder_inputs(feature_spec, b, s)
        calls = capture_calls(encoder, blocks, xs, lens)
        for path, mod in blocks:
            if path not in calls:
                log(f"[inductor] {path}: not reached by the offline forward")
                if on_skip:
                    on_skip(path, "not reached by the offline forward")
                continue
            args, kwargs = calls[path]
            log(f"[inductor] {path} ({type(mod).__name__}) B={b} {s:g}s T={frames}")
            if on_start:
                on_start(path)
            try:
                r = measure_block(
                    path,
                    mod,
                    args,
                    kwargs,
                    batch=b,
                    seconds=s,
                    frames=frames,
                    arms=arms,
                    dtype=dtype,
                    backends=backends,
                    policy=policy,
                    working_set_bytes=working_set_bytes,
                    max_copy_bytes=max_copy_bytes,
                    inventory=inventory,
                    verbose=verbose,
                    subproc=subproc,
                    log=log,
                )
            except bench.StickyDeviceError:
                raise
            except Exception as exc:  # noqa: BLE001 -- e.g. the fp32 reference OOMs
                reason = f"{type(exc).__name__}: {str(exc)[:300]}"
                log(f"  skipped: {reason}")
                if on_skip:
                    on_skip(path, reason)
                torch.cuda.empty_cache()
                continue
            line = "  " + "  ".join(
                f"{a} {r.arms[a].us:.1f}us" if r.arms[a].status == "ok" else f"{a} error"
                for a in arms
            )
            sp = r.speedup()
            log(line + (f"  -> {r.verdict()} ({sp:.2f}x)" if sp else f"  -> {r.verdict()}"))
            if on_result:
                on_result(r)
            out.append(r)
    return out


# =============================================================================
# Report
# =============================================================================


_FUSED = re.compile(r"^triton_(?:poi|red|per|tem|for|mm)_fused_(.+?)_\d+$")
#: Op-name pieces that are two words in an aten name.
_OP_WORDS = (
    "_scaled_dot_product_efficient_attention",
    "_scaled_dot_product_flash_attention",
    "_scaled_dot_product_cudnn_attention",
    "_unsafe_view",
    "logical_not",
    "logical_and",
    "index_put",
    "slice_scatter",
    "select_scatter",
    "bitwise_not",
    "bitwise_and",
    "native_layer_norm",
    "native_group_norm",
    "native_batch_norm",
    "layer_norm",
    "group_norm",
    "batch_norm",
    "rms_norm",
    "scaled_dot_product",
    "masked_fill",
    "constant_pad_nd",
    "index_select",
    "_to_copy",
    "addmm",
    "convolution",
)


#: Ops that only move or re-view data: not a fusion worth naming.
_LAYOUT_OPS = frozenset(
    (
        "view",
        "_unsafe_view",
        "t",
        "transpose",
        "permute",
        "clone",
        "expand",
        "unsqueeze",
        "squeeze",
        "reshape",
        "slice",
        "select",
        "split",
        "cat",
        "as_strided",
        "unfold",
        "_to_copy",
        "copy",
        "unbind",
    )
)


def fused_ops(kernel: str) -> List[str]:
    """The aten ops a Triton kernel name says it fused (``[]`` for other kernels).

    Inductor names a kernel ``triton_<kind>_fused_<op>_<op>..._<n>``, ops joined
    by one underscore (an op that itself starts with one, ``_to_copy``, gives
    two).
    """
    m = _FUSED.match(kernel)
    if not m:
        return []
    rest = m.group(1)
    ops: List[str] = []
    first = True
    while rest:
        if not first:
            if not rest.startswith("_"):
                break
            rest = rest[1:]
        first = False
        hit = next(
            (w for w in _OP_WORDS if rest.startswith(w) and rest[len(w) : len(w) + 1] in ("", "_")),
            None,
        )
        if hit is None:
            hit = "_" + rest[1:].split("_", 1)[0] if rest.startswith("_") else rest.split("_", 1)[0]
        if not hit or hit == "_":
            break
        ops.append(hit)
        rest = rest[len(hit) :]
    return ops


def compute_ops(kernel: str) -> List[str]:
    """:func:`fused_ops` without the layout ops, deduplicated in order."""
    seen: List[str] = []
    for op in fused_ops(kernel):
        if op not in _LAYOUT_OPS and op not in seen:
            seen.append(op)
    return seen


def summarize(results: Sequence[BlockResult], gap: float = GAP) -> List[Dict[str, Any]]:
    """Per block: shapes, geomean speedup, where Inductor is ahead, the worst case."""
    by: Dict[str, List[BlockResult]] = {}
    for r in results:
        by.setdefault(r.block, []).append(r)
    rows: List[Dict[str, Any]] = []
    for block, rs in by.items():
        sps = [s for s in (r.speedup() for r in rs) if s]
        geo = math.exp(sum(math.log(s) for s in sps) / len(sps)) if sps else None
        ahead = [r for r in rs if r.verdict(gap) == "inductor-ahead"]
        worst = max(ahead, key=lambda r: r.speedup() or 0.0) if ahead else None
        rows.append(
            {
                "block": block,
                "kind": rs[0].kind,
                "shapes": len(rs),
                "geomean": geo,
                "verdicts": dict(Counter(r.verdict(gap) for r in rs)),
                "ahead_at": [(r.batch, r.frames) for r in ahead],
                "worst": worst,
            }
        )
    rows.sort(key=lambda d: -(d["worst"].speedup() if d["worst"] else 0.0))
    return rows


def _fmt_us(a: Optional[ArmResult]) -> str:
    if a is None:
        return "--"
    if a.status != "ok" or not math.isfinite(a.us):
        return "error"
    flag = "" if a.numerics_ok is not False else " (numerics)"
    return f"{a.us:.1f}{flag}"


def _short(name: str, width: int = 90) -> str:
    return name if len(name) <= width else name[: width - 3] + "..."


def _kernel_lines(a: ArmResult, top: int = 8) -> List[str]:
    lines = []
    for k in a.kernels[:top]:
        fused = compute_ops(k["name"])
        tag = f" -- fuses {' + '.join(fused)}" if len(fused) > 1 else ""
        op = f" ({k['op']})" if k.get("op") else ""
        label = k.get("label") or k["name"]
        if label.startswith("_Z"):  # a report written before the name was read
            label = _mangled_hint(label)
        lines.append(f"  - `{_short(label)}`{op} x{k['calls']}, {k['us']:.1f} us{tag}")
    if len(a.kernels) > top:
        rest = sum(k["us"] for k in a.kernels[top:])
        lines.append(f"  - ... {len(a.kernels) - top} more kernels, {rest:.1f} us")
    return lines


def render_markdown(results: Sequence[BlockResult], meta: Dict[str, Any], gap: float = GAP) -> str:
    """The report: summary, per-shape table, kernel to-do list, Inductor's autotune record."""
    arms = [a for a in ARMS if any(a in r.arms for r in results)]
    out: List[str] = []
    out.append(f"# Inductor fusion reference: {meta.get('model', '?')}")
    out.append("")
    out.append(
        f"{meta.get('architecture', '?')}, {meta.get('dtype', '?')}, {meta.get('gpu', '?')} "
        f"(sm{meta.get('sm', '?')}); torch {meta.get('torch', '?')}, triton "
        f"{meta.get('triton', '?')}, CuTeDSL {meta.get('cutedsl', '?')}; {meta.get('date', '')}."
    )
    out.append("")
    out.append(
        "Arms: `oasr` = the OASR kernels (what serves); `torch` = the layers' torch path, "
        "eager; `inductor` = `torch.compile` of the torch path (default mode); "
        "`inductor-max` = `max-autotune-no-cudagraphs`, GEMM backends "
        f"{', '.join(meta.get('backends', []))}. Every arm timed by `oasr.tune.bench` "
        "(graph-captured loops, rotated weight copies, interleaved rounds); us per call. "
        f"Ahead = at least {gap:.0%} faster and beyond 3 sigma. Benchmark only: nothing "
        "here is served."
    )
    av = meta.get("backend_status") or {}
    if av:
        out.append("")
        out.append(
            "Backends: " + "; ".join(f"{k}: {v}" for k, v in av.items()) + ". "
            "What each one offered and won is under *Inductor autotune*."
        )
    out.append("")
    out.append("## Summary by block")
    out.append("")
    out.append("| block | class | shapes | OASR / best Inductor (geomean) | verdicts | worst |")
    out.append("|---|---|---|---|---|---|")
    summary = summarize(results, gap)
    for row in summary:
        geo = f"{row['geomean']:.2f}x" if row["geomean"] else "--"
        v = ", ".join(f"{k} {n}" for k, n in sorted(row["verdicts"].items()))
        w = row["worst"]
        worst = f"{w.speedup():.2f}x at B{w.batch} T{w.frames}" if w else "--"
        out.append(
            f"| `{row['block']}` | {row['kind']} | {row['shapes']} | {geo} | {v} | {worst} |"
        )
    out.append("")
    out.append("OASR / best Inductor above 1.00x means Inductor is faster.")
    out.append("")
    out.append("## Per shape")
    out.append("")
    head = "| block | B | T (frames) | " + " | ".join(f"{a} us" for a in arms)
    head += " | OASR/best | kernels " + "/".join(arms) + " | verdict |"
    out.append(head)
    out.append("|" + "---|" * (len(arms) + 6))
    for r in results:
        sp = r.speedup()
        kern = "/".join(
            str(r.arms[a].launches) if a in r.arms and r.arms[a].status == "ok" else "-"
            for a in arms
        )
        out.append(
            f"| `{r.block}` | {r.batch} | {r.frames} | "
            + " | ".join(_fmt_us(r.arms.get(a)) for a in arms)
            + f" | {f'{sp:.2f}x' if sp else '--'} | {kern} | {r.verdict(gap)} |"
        )
    out.append("")
    out.append("## Kernel to-do (Inductor ahead)")
    out.append("")
    todo = [row for row in summary if row["worst"] is not None]
    if not todo:
        out.append(f"None: Inductor is not {gap:.0%} ahead of OASR on any block and shape.")
    for row in todo:
        case: BlockResult = row["worst"]
        best = case.best_compiled()
        assert best is not None
        o, b = case.arms["oasr"], case.arms[best]
        shapes = ", ".join(f"B{bb} T{t}" for bb, t in row["ahead_at"])
        out.append(
            f"### `{case.block}` ({case.kind}): {case.speedup():.2f}x at B{case.batch} T{case.frames}, "
            f"{o.us - b.us:.1f} us per call"
        )
        out.append("")
        out.append(f"Ahead at {shapes}. Worst case in detail:")
        out.append("")
        out.append(f"- `oasr`: {o.us:.1f} us, {o.launches} launches, error {o.err:.3g}")
        out.extend(_kernel_lines(o))
        glue = [k for k in o.kernels if str(k.get("op", "")).startswith("aten::")]
        if glue:
            ops = sorted({k["op"] for k in glue})
            out.append(
                f"- torch kernels still on the OASR path: {sum(k['calls'] for k in glue)} "
                f"launches, {sum(k['us'] for k in glue):.1f} us ({', '.join(ops)})"
            )
        out.append(f"- `{best}`: {b.us:.1f} us, {b.launches} launches, error {b.err:.3g}")
        out.extend(_kernel_lines(b))
        if b.breaks:
            out.append(
                f"- graph breaks in `{best}` (OASR calls it could not trace or fuse across): "
                + "; ".join(f"`{x}`" for x in b.breaks[:4])
            )
        out.append("")
    out.append("## Inductor autotune")
    out.append("")
    out.append("| block | B | T | op | inputs | offered | winner (backend) |")
    out.append("|---|---|---|---|---|---|---|")
    any_tune = False
    for r in results:
        a = r.arms.get("inductor-max")
        if a is None:
            continue
        for t in a.autotune:
            any_tune = True
            offered = ", ".join(f"{k} {n}" for k, n in sorted(t["offered"].items()))
            ins = " x ".join("x".join(str(s) for s in dims) for dims in t["inputs"])
            out.append(
                f"| `{r.block}` | {r.batch} | {r.frames} | {t['op']} | {ins} | {offered} | "
                f"`{_short(t['winner'], 40)}` ({t['winner_backend']}) |"
            )
    if not any_tune:
        out.append("| -- | | | | | | no autotuned op recorded |")
    out.append("")
    return "\n".join(out)


def save_report(
    results: Sequence[BlockResult],
    meta: Dict[str, Any],
    out_dir: Path,
    stem: str,
    gap: float = GAP,
) -> Tuple[Path, Path]:
    """Write ``<stem>.json`` (every number) and ``<stem>.md`` (the report)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    jpath = out_dir / f"{stem}.json"
    mpath = out_dir / f"{stem}.md"
    jpath.write_text(
        json.dumps({"meta": meta, "gap": gap, "results": [r.to_json() for r in results]}, indent=1)
    )
    mpath.write_text(render_markdown(results, meta, gap))
    return jpath, mpath


def load_report(path: Path) -> Tuple[Dict[str, Any], List[BlockResult]]:
    """Read a ``<stem>.json`` written by :func:`save_report`."""
    d = json.loads(Path(path).read_text())
    return d.get("meta", {}), [BlockResult.from_json(r) for r in d.get("results", [])]
