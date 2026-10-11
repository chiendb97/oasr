# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Hardware and software metadata for a run -- enough to reproduce or disqualify it.

Collected without importing torch or touching CUDA: when the backend is a
server in another process, creating a CUDA context in the client would put a
few hundred MiB on the device the server is being measured on.  GPU identity
comes from NVML instead; an in-process backend adds the torch view itself
(:meth:`Backend.info`).

The ``OASR_*`` switches and the JIT cache hashes are recorded for the same
reason ``benchmarks/core/env.py`` records them: they select kernels, so a
number measured under one setting is not comparable with one under another.
"""

from __future__ import annotations

import os
import platform
import socket
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
_JIT_CACHE = Path.home() / ".cache" / "oasr" / "jit"

#: Distributions whose versions are recorded when installed.
PACKAGES = (
    "oasr",
    "torch",
    "torchaudio",
    "numpy",
    "transformers",
    "httpx",
    "websockets",
    "nvidia-cutlass-dsl",
    "apache-tvm-ffi",
    "psutil",
    "nvidia-ml-py",
    "faster-whisper",
    "sherpa-onnx",
    "vllm",
)


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return ""


def git_info() -> Dict[str, Any]:
    return {
        "commit": _git("rev-parse", "HEAD"),
        "short": _git("rev-parse", "--short", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
    }


def cpu_info() -> Dict[str, Any]:
    model = platform.processor()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    out: Dict[str, Any] = {"model": model, "logical_cores": os.cpu_count()}
    try:
        out["affinity_cores"] = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    try:
        import psutil

        out["physical_cores"] = psutil.cpu_count(logical=False)
        out["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        pass
    return out


def gpu_info() -> List[Dict[str, Any]]:
    """Every NVML-visible GPU; empty on a box without NVIDIA's driver library."""
    try:
        import pynvml

        pynvml.nvmlInit()
    except Exception:  # noqa: BLE001 -- not NVIDIA, or no driver in this container
        return []
    out = []
    try:
        driver = _s(pynvml.nvmlSystemGetDriverVersion())
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            dev: Dict[str, Any] = {
                "index": i,
                "name": _s(pynvml.nvmlDeviceGetName(h)),
                "uuid": _s(pynvml.nvmlDeviceGetUUID(h)),
                "memory_gb": round(pynvml.nvmlDeviceGetMemoryInfo(h).total / 1e9, 1),
                "driver": driver,
            }
            for key, fn in (
                ("compute_capability", pynvml.nvmlDeviceGetCudaComputeCapability),
                ("power_limit_w", pynvml.nvmlDeviceGetEnforcedPowerLimit),
                ("max_sm_clock_mhz", lambda h: pynvml.nvmlDeviceGetMaxClockInfo(h, 1)),
            ):
                try:
                    v = fn(h)
                    dev[key] = (
                        ".".join(map(str, v))
                        if isinstance(v, tuple)
                        else (v / 1000.0 if key == "power_limit_w" else v)
                    )
                except Exception:  # noqa: BLE001
                    pass
            out.append(dev)
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass
    return out


def _s(x: Any) -> str:
    return x.decode() if isinstance(x, bytes) else str(x)


def packages() -> Dict[str, str]:
    out = {}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return out


def jit_cache() -> Dict[str, Any]:
    if not _JIT_CACHE.is_dir():
        return {"path": str(_JIT_CACHE), "present": False, "hashes": []}
    return {
        "path": str(_JIT_CACHE),
        "present": True,
        "hashes": sorted(p.name for p in _JIT_CACHE.iterdir() if p.is_dir()),
    }


def collect() -> Dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "argv": list(sys.argv),
        "cwd": os.getcwd(),
        "git": git_info(),
        "cpu": cpu_info(),
        "gpus": gpu_info(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "packages": packages(),
        "oasr_env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("OASR_")},
        "jit_cache": jit_cache(),
    }
