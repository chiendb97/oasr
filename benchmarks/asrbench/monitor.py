# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Resource telemetry, sampled from a separate process.

Why a process and not a thread: the in-process engine's step loop is
interpreter-bound, and a sampler thread in the same interpreter takes the GIL
every interval -- the measurement would perturb what it measures.  A child
process samples the target PID through ``psutil`` and the GPU through NVML,
writes one JSON line per sample (flushed, so a killed sampler loses at most
one), and costs the benchmark nothing but a scheduler slot.

Samples carry ``time.monotonic()``, which is system-wide on Linux, so the
parent attributes them to each benchmark's measured window afterwards; nothing
is aggregated while measuring.

Everything is optional.  No ``psutil`` -> no CPU/RSS; no NVML (a non-NVIDIA box,
a container without the driver library) -> no GPU fields.  What was available
is recorded alongside the samples, so an absent metric reads "not measured",
never "zero".

Run directly: ``python -m benchmarks.asrbench.monitor --pid N --interval 0.2 --out f.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from benchmarks.asrbench.metrics import summarize

#: Shortest window over which energy is reported at all.
MIN_ENERGY_SPAN_S = 1.0

# ---------------------------------------------------------------------------
# Probes (child side)
# ---------------------------------------------------------------------------


class _ProcProbe:
    def __init__(self, pid: int) -> None:
        import psutil

        self._psutil = psutil
        self.proc = psutil.Process(pid)
        self.proc.cpu_percent(None)  # prime: the first call always returns 0.0
        psutil.cpu_percent(None)
        self._children: Dict[int, Any] = {}

    def sample(self) -> Dict[str, Any]:
        ps = self._psutil
        cpu = self.proc.cpu_percent(None)
        rss = self.proc.memory_info().rss
        # A server may run inference in worker processes; count the tree.
        try:
            kids = self.proc.children(recursive=True)
        except ps.Error:
            kids = []
        for k in kids:
            if k.pid not in self._children:
                self._children[k.pid] = k
                k.cpu_percent(None)
                continue
            try:
                cpu += self._children[k.pid].cpu_percent(None)
                rss += self._children[k.pid].memory_info().rss
            except ps.Error:
                self._children.pop(k.pid, None)
        vm = ps.virtual_memory()
        return {
            "cpu_pct": cpu,
            "rss_mb": rss / 2**20,
            "sys_cpu_pct": ps.cpu_percent(None),
            "sys_mem_mb": (vm.total - vm.available) / 2**20,
        }


class _NvmlProbe:
    def __init__(self, indices: Optional[Sequence[int]] = None) -> None:
        import pynvml

        self.nv = pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        self.handles = [
            pynvml.nvmlDeviceGetHandleByIndex(i) for i in (indices or range(count)) if i < count
        ]
        self.names = [_s(pynvml.nvmlDeviceGetName(h)) for h in self.handles]

    def sample(self) -> Dict[str, Any]:
        nv = self.nv
        util, mem, power, energy = [], [], [], []
        for h in self.handles:
            try:
                util.append(float(nv.nvmlDeviceGetUtilizationRates(h).gpu))
                mem.append(nv.nvmlDeviceGetMemoryInfo(h).used / 2**20)
            except nv.NVMLError:
                continue
            try:
                power.append(nv.nvmlDeviceGetPowerUsage(h) / 1000.0)
            except nv.NVMLError:
                pass
            try:
                energy.append(float(nv.nvmlDeviceGetTotalEnergyConsumption(h)))
            except nv.NVMLError:
                pass
        out: Dict[str, Any] = {}
        if util:
            out["gpu_util_pct"] = sum(util) / len(util)
            out["gpu_mem_mb"] = sum(mem)
        if power:
            out["gpu_power_w"] = sum(power)
        if energy and len(energy) == len(self.handles):
            out["gpu_energy_mj"] = sum(energy)
        return out


def _s(x: Any) -> str:
    return x.decode() if isinstance(x, bytes) else str(x)


def visible_gpu_indices() -> Optional[List[int]]:
    """Physical NVML indices named by ``CUDA_VISIBLE_DEVICES``, if it uses indices."""
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is None or env.strip() == "":
        return None
    try:
        return [int(x) for x in env.split(",") if x.strip()]
    except ValueError:
        return None  # UUIDs: fall back to every device NVML sees


def sampler_main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="asrbench resource sampler")
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--interval", type=float, default=0.2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-gpu", action="store_true")
    args = ap.parse_args(argv)

    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("flag", True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("flag", True))

    available: Dict[str, Any] = {"psutil": False, "nvml": False}
    proc: Optional[_ProcProbe] = None
    gpu: Optional[_NvmlProbe] = None
    try:
        proc = _ProcProbe(args.pid)
        available["psutil"] = True
    except Exception as exc:  # noqa: BLE001 -- absent or the PID is gone
        available["psutil_error"] = f"{type(exc).__name__}: {exc}"
    if not args.no_gpu:
        try:
            gpu = _NvmlProbe(visible_gpu_indices())
            available["nvml"] = True
            available["gpus"] = gpu.names
        except Exception as exc:  # noqa: BLE001 -- no driver / not NVIDIA
            available["nvml_error"] = f"{type(exc).__name__}: {exc}"

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(json.dumps({"header": True, "interval_s": args.interval, **available}) + "\n")
        f.flush()
        next_t = time.monotonic()
        while not stop["flag"]:
            sample: Dict[str, Any] = {"t": time.monotonic()}
            try:
                if proc is not None:
                    sample.update(proc.sample())
            except Exception:  # noqa: BLE001 -- target exited
                proc = None
            if gpu is not None:
                sample.update(gpu.sample())
            f.write(json.dumps(sample) + "\n")
            f.flush()
            next_t += args.interval
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.monotonic()
    return 0


# ---------------------------------------------------------------------------
# Parent side
# ---------------------------------------------------------------------------


class ResourceMonitor:
    """Start/stop the sampler and attribute its samples to time windows."""

    def __init__(self, out_path: Path, interval_s: float = 0.2, gpu: bool = True) -> None:
        self.out_path = Path(out_path)
        self.interval_s = float(interval_s)
        self.gpu = gpu
        self._proc: Optional[subprocess.Popen] = None
        self.pid: Optional[int] = None

    def start(self, pid: int) -> None:
        self.stop()
        self.pid = pid
        repo_root = str(Path(__file__).resolve().parents[2])
        cmd = [
            sys.executable,
            "-m",
            "benchmarks.asrbench.monitor",
            "--pid",
            str(pid),
            "--interval",
            str(self.interval_s),
            "--out",
            str(self.out_path),
        ]
        if not self.gpu:
            cmd.append("--no-gpu")
        env = dict(os.environ)
        env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
        # Appending would interleave two samplers' lines; the header names the PID.
        if self.out_path.exists():
            self.out_path = self.out_path.with_name(
                f"{self.out_path.stem}.{pid}{self.out_path.suffix}"
            )
            cmd[cmd.index("--out") + 1] = str(self.out_path)
        self._proc = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL)

    def stop(self) -> None:
        if self._proc is None:
            return
        self._proc.terminate()
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        self._proc = None

    def header(self) -> Dict[str, Any]:
        for row in read_samples(self.out_path, header=True):
            return row
        return {}

    def samples(self) -> List[Dict[str, Any]]:
        return read_samples(self.out_path)


def read_samples(path: Path, header: bool = False) -> List[Dict[str, Any]]:
    out = []
    if not Path(path).is_file():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn final line from a killed sampler
            if bool(row.get("header")) == header:
                out.append(row)
    return out


def summarize_window(
    samples: Sequence[Dict[str, Any]],
    t0: float,
    t1: float,
    audio_s: Optional[float] = None,
) -> Dict[str, float]:
    """Resource metrics over samples with ``t0 <= t <= t1`` (monotonic seconds).

    A window shorter than the sampling period holds at most one sample; it is
    then widened by one period on each side so memory peaks are still
    reported.  ``<metric>.n`` says how many samples a figure rests on.
    """
    win = [s for s in samples if t0 <= s.get("t", -1.0) <= t1]
    if len(win) < 2 and len(samples) >= 2:
        ts = [s["t"] for s in samples if "t" in s]
        period = (ts[-1] - ts[0]) / max(1, len(ts) - 1)
        win = [s for s in samples if t0 - period <= s.get("t", -1.0) <= t1 + period]
    out: Dict[str, float] = {}
    if not win:
        return out
    for key in ("cpu_pct", "rss_mb", "sys_cpu_pct", "sys_mem_mb", "gpu_util_pct", "gpu_mem_mb"):
        vals = [s[key] for s in win if key in s]
        if vals:
            stats = summarize(vals, key)
            for stat in ("mean", "max", "n"):
                if f"{key}.{stat}" in stats:
                    out[f"{key}.{stat}"] = stats[f"{key}.{stat}"]
    power = [(s["t"], s["gpu_power_w"]) for s in win if "gpu_power_w" in s]
    if power:
        out["gpu_power_w.mean"] = sum(p for _, p in power) / len(power)
        out["gpu_power_w.max"] = max(p for _, p in power)
    energy = [(s["t"], s["gpu_energy_mj"]) for s in win if "gpu_energy_mj" in s]
    joules: Optional[float] = None
    span = win[-1]["t"] - win[0]["t"]
    if span < MIN_ENERGY_SPAN_S:
        # The driver's energy counter and the power sensor both update on their
        # own period; over a sub-second window they read as zero or noise.
        energy, power = [], []
    if len(energy) >= 2:
        # The driver's cumulative counter beats integrating sampled power.
        joules = (energy[-1][1] - energy[0][1]) / 1000.0
    elif len(power) >= 2:
        joules = sum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(power, power[1:]))
    if joules is not None:
        out["energy_j"] = joules
        if audio_s:
            out["energy_j_per_audio_h"] = joules / (audio_s / 3600.0)
    return out


if __name__ == "__main__":
    sys.exit(sampler_main())
