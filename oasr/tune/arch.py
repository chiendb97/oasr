# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""``ArchProfile``: what a candidate generator and a cost model may know about a GPU.

Three kinds of fact, from three places, on purpose:

* **Static ISA/template facts** -- which MMA lane a family runs, which pipeline
  depths its templates can be instantiated at.  Compile-time properties, kept
  as tables so the test suite can parametrise over every family on one box.
* **Device-queried facts** -- SM count, opt-in shared memory, L2, threads and
  registers per SM.  Read from the device, never copied into a table as the
  runtime answer (sm86/sm89 were once budgeted with A100's 163 KB).
* **Micro-benchmarked facts** -- DRAM bandwidth, achieved tensor throughput,
  the eager launch floor and the per-node cost inside a replayed graph.
  Measured once per SKU and cached, never derived from clocks: the clock
  formula under-reports GDDR7 by more than 2x (``oasr/jit/measured.py``).
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

__all__ = ["ArchProfile", "LANE_BY_FAMILY", "STAGE_DOMAIN", "current", "synthetic"]

#: MMA lane of each compiled SM family's GEMM/conv kernels.
LANE_BY_FAMILY: Dict[int, str] = {
    75: "sm80_mma",
    80: "sm80_mma",
    86: "sm80_mma",
    89: "sm80_mma",
    90: "sm90_wgmma",
    100: "sm100_tcgen05",
    120: "sm80_mma",
}

#: Pipeline depths the CUTLASS 2.x templates can be instantiated at, per family.
#: Turing's ``kernel::DefaultGemm`` tensor-op specialisation exists at two stages
#: and no other (a three-stage TU fails to compile); the multistage Sm80 path
#: starts at three.  Which of these *fit* is the device's shared memory's call.
STAGE_DOMAIN: Dict[int, Tuple[int, ...]] = {
    75: (2,),
    80: (3, 4),
    86: (3, 4),
    89: (3, 4),
    120: (3, 4),
    # The mma.sync half of the mixed GEMM spaces (oasr.jit.gemm._MIXED_LANE_SMS).
    90: (3, 4),
    100: (3, 4),
}


@dataclass(frozen=True)
class ArchProfile:
    # --- static ---
    family: int
    cc: Tuple[int, int]
    lane: str
    stage_domain: Tuple[int, ...]
    # --- queried ---
    name: str
    num_sms: int
    smem_optin: int
    smem_per_sm: int
    regs_per_sm: int
    max_threads_per_sm: int
    l2_bytes: int
    # --- micro-benchmarked (None until measured) ---
    dram_gbps: Optional[float] = None
    tensor_tflops: Optional[float] = None
    launch_us_eager: Optional[float] = None
    launch_us_graph: Optional[float] = None
    #: Where the numbers came from: "device", "synthetic", "cache".
    source: str = "device"

    @property
    def sku(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "sms": self.num_sms,
            "l2": self.l2_bytes,
            "smem_optin": self.smem_optin,
        }

    def smem_budget(self) -> int:
        """Shared memory a launch may request (opt-in minus the driver reserve)."""
        return max(0, self.smem_optin - 1024)

    def to_json(self) -> Dict[str, Any]:
        d = asdict(self)
        d["cc"] = list(self.cc)
        d["stage_domain"] = list(self.stage_domain)
        return d


def synthetic(family: int) -> ArchProfile:
    """A table-built profile for *family* -- for tests and cross-arch codegen.

    Uses the per-family shared-memory and thread tables that the C++ capability
    code mirrors (``oasr.jit.arch_facts``) and representative SM counts; the
    micro-benchmarked fields stay ``None``.
    """
    from oasr.jit import arch_facts

    sms = {75: 40, 80: 108, 86: 84, 89: 142, 90: 132, 100: 148, 120: 170}[family]
    l2 = {75: 4, 80: 40, 86: 6, 89: 96, 90: 50, 100: 126, 120: 96}[family] * 2**20
    cap = arch_facts.SMEM_CAPACITY.get(family, 64 * 1024 if family == 75 else 101376)
    return ArchProfile(
        family=family,
        cc=(family // 10, family % 10),
        lane=LANE_BY_FAMILY[family],
        stage_domain=STAGE_DOMAIN.get(family, ()),
        name=f"synthetic-sm{family}",
        num_sms=sms,
        smem_optin=cap,
        smem_per_sm=cap + 1024,
        regs_per_sm=65536,
        max_threads_per_sm=arch_facts.MAX_THREADS_PER_SM.get(family, 2048),
        l2_bytes=l2,
        source="synthetic",
    )


def _cache_path(name: str, sms: int) -> Optional[Path]:
    from oasr.tune.database import user_root

    root = user_root()
    if root is None:
        return None
    safe = "".join(ch if ch.isalnum() else "_" for ch in name)
    return root / "arch" / f"{safe}_{sms}.json"


_CURRENT: Optional[ArchProfile] = None


def current(measure: bool = False) -> ArchProfile:
    """This device's profile; ``measure=True`` fills (and caches) the benchmarked fields."""
    global _CURRENT
    import torch

    from oasr.jit.core import _get_target_sm

    if _CURRENT is not None and (not measure or _CURRENT.dram_gbps is not None):
        return _CURRENT
    family = _get_target_sm()
    p = torch.cuda.get_device_properties(torch.cuda.current_device())
    base = ArchProfile(
        family=family,
        cc=(p.major, p.minor),
        lane=LANE_BY_FAMILY[family],
        stage_domain=STAGE_DOMAIN.get(family, ()),
        name=p.name,
        num_sms=int(p.multi_processor_count),
        smem_optin=int(getattr(p, "shared_memory_per_block_optin", 0) or 0),
        smem_per_sm=int(getattr(p, "shared_memory_per_multiprocessor", 0) or 0),
        regs_per_sm=int(getattr(p, "regs_per_multiprocessor", 65536) or 65536),
        max_threads_per_sm=int(getattr(p, "max_threads_per_multi_processor", 2048) or 2048),
        l2_bytes=int(getattr(p, "L2_cache_size", 0) or 0),
    )
    cached = _cache_path(base.name, base.num_sms)
    if cached is not None and cached.is_file():
        with contextlib.suppress(Exception):
            d = json.loads(cached.read_text())
            base = ArchProfile(
                **{
                    **base.to_json(),
                    "cc": base.cc,
                    "stage_domain": base.stage_domain,
                    **{
                        k: d.get(k)
                        for k in (
                            "dram_gbps",
                            "tensor_tflops",
                            "launch_us_eager",
                            "launch_us_graph",
                        )
                    },
                    "source": "cache",
                }
            )
    if measure and base.dram_gbps is None:
        base = _microbench(base)
        if cached is not None:
            with contextlib.suppress(OSError):
                cached.parent.mkdir(parents=True, exist_ok=True)
                cached.write_text(json.dumps(base.to_json(), indent=1))
    _CURRENT = base
    return base


def _graph_ms(name: str, call, calls_per_graph: int) -> float:
    from oasr.tune import bench

    res, _ = bench.measure(
        [bench.Arm(name, lambda i: call)],
        bench.BenchPolicy(calls_per_graph=calls_per_graph, stage_rounds=(5,), solo=False),
    )
    return float(res[name].median_ms)


def _dram_gbps(l2_bytes: int) -> Optional[float]:
    """A graph-timed copy of a buffer far larger than L2."""
    import torch

    n = max(256 * 2**20, 8 * l2_bytes) // 2
    src = torch.empty(n, device="cuda", dtype=torch.float16)
    dst = torch.empty_like(src)
    t = _graph_ms("copy", lambda: dst.copy_(src), 4) / 1e3
    return (2 * n * 2) / t / 1e9 if t > 0 else None


def _tensor_tflops() -> Optional[float]:
    """A large cuBLAS GEMM."""
    import torch

    s = 8192
    a = torch.randn(s, s, device="cuda", dtype=torch.float16)
    b = torch.randn(s, s, device="cuda", dtype=torch.float16)
    c = torch.empty(s, s, device="cuda", dtype=torch.float16)
    t = _graph_ms("mm", lambda: torch.mm(a, b, out=c), 2) / 1e3
    return (2 * s**3) / t / 1e12 if t > 0 else None


def _launch_us() -> Tuple[float, float]:
    """A trivial kernel's eager issue cost and its per-node cost inside a graph."""
    import torch

    from oasr.tune import bench

    x = torch.zeros(8, device="cuda")
    eager = bench.measure_issue_us(lambda: x.add_(1.0), reps=256)
    return eager, _graph_ms("tiny", lambda: x.add_(1.0), 64) * 1e3


def _microbench(base: ArchProfile) -> ArchProfile:
    """DRAM bandwidth, tensor throughput, eager and in-graph launch costs.

    One helper per measurement, so each one's buffers are freed before the next
    allocates (the DRAM copy alone is >= 512 MiB).
    """
    import torch

    dram = _dram_gbps(base.l2_bytes)
    tflops = _tensor_tflops()
    eager, graph_us = _launch_us()
    torch.cuda.empty_cache()
    return ArchProfile(
        **{
            **base.to_json(),
            "cc": base.cc,
            "stage_domain": base.stage_domain,
            "dram_gbps": dram,
            "tensor_tflops": tflops,
            "launch_us_eager": eager,
            "launch_us_graph": graph_us,
            "source": "device",
        }
    )
