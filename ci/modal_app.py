#!/usr/bin/env python3
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Run the GPU test suites on rented accelerators, one image across every arch.

Usage::

    modal run ci/modal_app.py::fetch_assets                     # once, ~20 GiB
    modal run ci/modal_app.py::main --gpus H100 --dry-run       # the plan + worst-case bill
    modal run ci/modal_app.py::main --gpus L40S,H100,B200
    modal run ci/modal_app.py::main --gpus H100 --suites engine,models
    modal run ci/modal_app.py::main --gpus B200 --no-assets \\
        --paths tests/kernels/test_fmha.py,tests/models/test_layer_waist.py

Where the money goes, and what keeps it down:

* **nvcc, not the tests, dominates a cold run, and it needs no GPU.**  Every
  architecture's kernels are compiled first on a CPU-only container
  (``precompile``: ``oasr tune prebuild --all`` with ``OASR_CUDA_ARCH_LIST``)
  into the per-arch JIT prefix the GPU run reads, and an architecture whose
  kernels do not compile is reported without renting its GPU.  Cold, the
  kernels family took 20 min on an A100 and was cut at 38 % after 90 min on a
  B200, nearly all of it nvcc.  Each run reports what it still had to compile.
* **A hard ceiling per run** (``--timeout-s``, 30 min by default) and a refusal
  to start when the worst case -- every run hitting it -- exceeds
  ``--max-usd``.  Each run prints its own estimate at list prices; ``modal
  billing report --for today`` is the authority.
* **Narrow runs**: ``--paths`` takes test files or node ids instead of whole
  families, and ``--no-assets`` skips the checkpoint-gated tests.

The GEMM tuning matrix is its own app, ``ci/modal_tune.py``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import modal

REPO_REMOTE = "/repo"

# Container imports lose the checkout-relative package layout, so prefer the
# copied repository path when it exists and use ``__file__`` locally.
REPO_ROOT = (
    Path(REPO_REMOTE)
    if (Path(REPO_REMOTE) / "ci" / "gpu_suites.py").is_file()
    else Path(__file__).resolve().parent.parent
)
CI_DIR = REPO_ROOT / "ci"
sys.path.insert(0, str(CI_DIR))
sys.path.insert(0, str(REPO_ROOT / "tests"))
import assets as test_assets  # noqa: E402  — both need the sys.path lines above
import modal_assets  # noqa: E402
from gpu_suites import DEFAULT_MARKER_EXPR, SUITES, paths_for  # noqa: E402

APP_NAME = "oasr-gpu-ci"
ASSETS_MOUNT = "/assets"
JIT_MOUNT = "/jit"
CUTLASS_HOME = "/opt/cutlass"

#: Modal accelerator -> the compute capability it reports, for the plan
#: printout and for ``--gpus all``.  Modal is the authority at run time; this
#: table only has to be right about *which* architectures the sweep covers.
GPU_ARCH: dict[str, str] = {
    "T4": "sm_75",
    "A10G": "sm_86",
    "L4": "sm_89",
    "A100-40GB": "sm_80",
    "A100-80GB": "sm_80",
    "L40S": "sm_89",
    "H100": "sm_90",
    "H200": "sm_90",
    "B200": "sm_100",
    "RTX-PRO-6000": "sm_120",
}

#: One per distinct architecture, newest first.  This is what ``--gpus all``
#: runs: five architectures, not ten accelerators, because two L40S-class cards
#: exercise the same kernels.
#:
#: Expect red on some of these today, and that is the point of having them here
#: rather than a shorter list that stays green.  As of the first sweep only
#: sm_80 and sm_120 — the two architectures anyone develops OASR on — pass the
#: kernels family; sm_90 and sm_100 fail to *compile* the CUTLASS 3.x conv path
#: (``CutlassConv2dFpropKernelSm90`` asks ``CollectiveBuilder`` for a collective
#: CUTLASS has no specialization for), and sm_89 fails a large fraction of the
#: GEMM tests.  Removing an accelerator from this tuple would hide that, so
#: don't; fix the kernel, or record the gap where gaps are recorded.
ARCH_SWEEP = ("B200", "RTX-PRO-6000", "H100", "L40S", "A100-40GB")

DEFAULT_GPU = os.environ.get("OASR_MODAL_GPU") or "RTX-PRO-6000"

#: Modal list prices, USD per second (modal.com/pricing, read 2026-11-16).  They
#: price the plan's worst case and each run's estimate, nothing else.
GPU_USD_PER_S: dict[str, float] = {
    "T4": 0.000164,
    "L4": 0.000222,
    "A10G": 0.000306,
    "L40S": 0.000542,
    "A100-40GB": 0.000583,
    "A100-80GB": 0.000694,
    "RTX-PRO-6000": 0.000842,
    "H100": 0.001097,
    "H200": 0.001261,
    "B200": 0.001736,
}
CPU_USD_PER_CORE_S = 0.0000131
MEM_USD_PER_GIB_S = 0.00000222

#: Modal bills CPU and memory at whichever is higher, the request or the use, and
#: lets a container burst past its request to a limit.  So every container here
#: *requests* little and *limits* high: a stage that is busy gets the cores, and
#: one that idles -- a test waiting on the device, a compile down to its last
#: translation unit -- pays for what it uses rather than for what it may need.
#:
#: GPU test container, ``(request, limit)``: one pytest process waiting on the
#: device, plus whatever ``precompile`` cannot reach (a module a test generates
#: for itself, the CuTeDSL kernels, which compile in-process).
GPU_CPU = (2.0, 8.0)
GPU_MEMORY_MIB = (8 * 1024, 48 * 1024)
#: Hard ceiling on one (accelerator, suite) run.  It used to be 3 h, sized for a
#: cold JIT cache compiling on the GPU's CPU share -- at B200 prices one such run
#: was ~$19.  With the kernels compiled ahead, a run is its tests.
GPU_TIMEOUT_S = 1800

#: The compile container per CUTLASS lane: limits ``(cores, memory MiB)``, cores =
#: nvcc jobs.  Memory is a hard limit (thrashing, then a kill), and ``prebuild``
#: runs every module at once with at least one job each, so the compilers in
#: flight outnumber the jobs: ``prebuild --all -j20`` for sm_89 (36 modules, the
#: C++ CuTe cells among them) peaked at 34 GiB over 66 compiler processes, and
#: 32 jobs under a 48 GiB limit thrashed until Modal killed it.  A 3.x (wgmma /
#: tcgen05) TU peaks at 4-5.4 GiB.  The request is ``COMPILE_REQUEST``: on sm_90
#: 20 jobs were busy for the first ~6 min of a 15 min compile and one or two for
#: the rest, a tail that now costs its own cores rather than the whole limit.
COMPILE: dict[str, tuple[int, int]] = {"2x": (24, 80 * 1024), "3x": (24, 128 * 1024)}
COMPILE_REQUEST = (4.0, 8 * 1024)
#: One TU sets the floor: ``topk``'s two compile serially at one job each, 570 s
#: for sm_100, 910 s for sm_90 and 1378 s for sm_89 on the reference box.
COMPILE_TIMEOUT_S = 3600

#: Every SM the ``_C.so`` build is configured for.  It compiles no CUDA
#: (``OASR_SOURCES`` is pure C++), so a wide list costs nothing and one image
#: serves the whole sweep — which is the point: an image per arch would mean an
#: image build per arch, and the kernels that actually differ are JIT-compiled
#: on the target anyway.
CUDA_ARCHITECTURES = os.environ.get("OASR_MODAL_CUDA_ARCH", "80;86;89;90;100;120")


def _torch_arch_list(cmake_archs: str) -> str:
    """``"80;90;100"`` -> ``"8.0;9.0;10.0"``: the CMake list in torch's spelling."""
    return ";".join(f"{a[:-1]}.{a[-1]}" for a in cmake_archs.split(";") if a)


#: The toolkit the JIT compiles with, and the torch built against the same one.
#: CUDA 13, not 12.8: ``oasr.jit.core._GENCODE_TARGET`` builds sm_100 for the
#: ``sm_100f`` family target, which nvcc 12.8 rejects ("Unsupported gpu
#: architecture 'compute_100f'"), so a 12.8 image compiled nothing for a B200.
#: cu130, not cu132 like the tuning image: cu132 has no torchaudio, which the
#: models family needs.  The three wheels are pinned as a set -- torchaudio and
#: torchcodec are built against one torch release each.
CUDA_IMAGE = os.environ.get("OASR_MODAL_CUDA_IMAGE", "nvidia/cuda:13.0.2-devel-ubuntu24.04")
TORCH_INDEX = os.environ.get("OASR_MODAL_TORCH_INDEX", "https://download.pytorch.org/whl/cu130")
TORCH_WHEELS = ("torch==2.11.0", "torchaudio==2.11.0", "torchcodec==0.11.0")


def _cutlass_ref() -> str:
    """The exact CUTLASS commit ``3rdparty/cutlass`` is pinned to.

    Read from the *index*, not the working tree, so it is right whether or not
    the submodule is checked out — which matters because the whole point is that
    CI no longer checks it out.  ``OASR_MODAL_CUTLASS_REF`` overrides; the
    literal is the v4.6.1 tag the JIT cache key stamps
    (``oasr/jit/attention.py::MIN_CUTEDSL_VERSION`` and the pinned
    ``nvidia-cutlass-dsl`` wheel are versioned with it).
    """
    override = os.environ.get("OASR_MODAL_CUTLASS_REF")
    if override:
        return override
    try:
        out = subprocess.check_output(
            ["git", "ls-tree", "HEAD", "3rdparty/cutlass"],
            cwd=str(REPO_ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        ).split()
        if len(out) >= 3:
            return out[2]
    except Exception:
        pass
    return "e05f953a5b3d38adc240df2ff928e0421c2abba3"  # v4.6.1


CUTLASS_REF = _cutlass_ref()

#: Assets ``fetch_assets`` cannot supply, so ``--strict-assets`` names them
#: instead of failing the suite over them.  Derived from the source table rather
#: than written out again here: an asset that gains a public source drops off
#: this list by itself, which is the failure mode the flag exists to prevent
#: (an allow-list that quietly outlives the gap it documented).
ALLOW_MISSING: tuple[str, ...] = tuple(
    sorted(set(modal_assets.NO_PUBLIC_SOURCE) | {"SPEECH_LLM_TINY_REF", "OASR_PARAFORMER_REF"})
)

#: Container asset paths derive from the central declarations. Assets without a
#: relative slot cannot live in the volume.
ASSET_LAYOUT: dict[str, str] = {
    asset.env: asset.relpath for asset in test_assets.ASSETS.values() if asset.relpath
}

app = modal.App(APP_NAME)

assets_vol = modal.Volume.from_name("oasr-ci-assets", create_if_missing=True)
# Kernels JIT-compile on first *call*; without a warm cache every run pays the
# full compile (~683 MiB of artifacts on the reference box).
JIT_VOLUME = "oasr-ci-jit-cache"
jit_vol = modal.Volume.from_name(JIT_VOLUME, create_if_missing=True)

image = (
    # -devel, not -runtime: the JIT shells out to nvcc at run time.
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.12")
    .apt_install(
        "git",
        "curl",
        "build-essential",
        "cmake",
        "ninja-build",
        "protobuf-compiler",  # tonic's build.rs compiles rust/proto/*.proto
        # openssl-sys is the one crate in the oasr-core tree that links a system
        # library, and it needs both: the headers, and pkg-config to locate them
        # ("Could not find directory of OpenSSL installation").  It arrives via
        # oasr-serve -> metrics-exporter-prometheus -> hyper-tls -> native-tls,
        # so nothing in this repo names it and a base image without it fails
        # only at the very end, in the Rust half of `pip install -e .`.
        "pkg-config",
        "libssl-dev",
    )
    # CUTLASS, from the submodule's pinned commit rather than from the caller's
    # disk.  The tarball is the source tree without the history: ~200 MiB over
    # the wire against 2.0 GiB in the image context, and only the two include
    # roots ``oasr/jit/env.py`` collects are kept.  Its own layer, keyed on the
    # sha, so a code change does not re-fetch it.
    .run_commands(
        f"mkdir -p {CUTLASS_HOME}",
        f"curl -sSL https://github.com/NVIDIA/cutlass/archive/{CUTLASS_REF}.tar.gz"
        f" | tar xz --strip-components=1 -C {CUTLASS_HOME}"
        f" cutlass-{CUTLASS_REF}/include cutlass-{CUTLASS_REF}/tools/util/include",
        f"test -f {CUTLASS_HOME}/include/cutlass/version.h",
    )
    # setuptools-rust builds the oasr._core PyO3 extension during pip install.
    .run_commands(
        "curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal",
        "echo 'source $HOME/.cargo/env' >> /root/.bashrc",
    )
    .env({"PATH": "/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"})
    .pip_install(*TORCH_WHEELS, index_url=TORCH_INDEX)
    .pip_install(
        # pyproject [project.dependencies] + the optional extras whose absence
        # would turn real coverage into importorskips.
        "numpy",
        # Pinned to what the JIT is developed and tuned against (ci/modal_tune.py):
        # every kernel module is built and loaded through it.
        "apache-tvm-ffi==0.1.10",
        "jinja2>=3.0",
        "PyYAML>=5.4",
        "safetensors",
        "huggingface_hub",
        "sentencepiece",
        "tokenizers",
        "transformers",
        "soundfile",
        # CuteDSL FMHA; without it oasr.jit.attention falls back to SDPA and the
        # cute-backend tests stop covering the kernel that ships.  Pinned, and
        # kept in step with the 3rdparty/cutlass submodule tag: the two halves
        # of CUTLASS are versioned together upstream, and a CI that compiles
        # kernels against one release while tracing them on another is testing a
        # combination nobody runs.  Floor lives in
        # oasr/jit/attention.py::MIN_CUTEDSL_VERSION.
        "nvidia-cutlass-dsl==4.6.1",
        "pytest==8.1.1",
    )
    # `pip install -e . --no-build-isolation` below needs pyproject's
    # [build-system] requires already present — setup.py must `import torch` to
    # find its cmake prefix, which an isolated build env cannot do.  Keep this
    # list in step with pyproject.toml.
    .pip_install(
        "setuptools>=61.0",
        "wheel",
        "cmake>=3.18",
        "pybind11>=2.10",
        "ninja",
        "setuptools-rust>=1.10",
    )
    # copy=True so the build below is baked into a cached layer.  A code change
    # re-runs only this final layer; the CUDA/Rust/torch layers stay cached.
    # Absolute, so `modal run` works from any cwd.
    .add_local_dir(
        REPO_ROOT,
        REPO_REMOTE,
        copy=True,
        # Tool caches are regenerated by any local test/lint run, so leaving them
        # in makes the final layer's hash change for no reason -- and a cache
        # written *during* the upload aborts it outright ("file modified while
        # reading").
        ignore=[
            "**/.git",
            # 26 069 files, 2.0 GiB, and byte-identical to what the CUTLASS
            # layer above already fetched.  Symlinked back into place below.
            "3rdparty",
            "3rdparty/**",
            "**/target",
            "**/build",
            "**/__pycache__",
            "**/*.so",
            "**/.venv",
            "**/.pytest_cache",
            "**/.ruff_cache",
            "**/.mypy_cache",
            # Gitignored working directories: benchmark output, nsys/ncu reports.
            # Nothing in the suite reads them and they reach hundreds of MiB.
            "**/.artifacts",
            "profiling_results",
            "profiling_results/**",
        ],
    )
    .run_commands(
        # oasr/jit/env.py looks for CUTLASS at ``<repo>/3rdparty/cutlass`` and
        # nowhere else, so put it there.  A symlink rather than a copy: the JIT
        # only ever reads headers out of it, and ``cutlass_version_stamp`` reads
        # ``include/cutlass/version.h`` through the link like any other path.
        f"mkdir -p {REPO_REMOTE}/3rdparty && ln -sfn {CUTLASS_HOME} {REPO_REMOTE}/3rdparty/cutlass",
        # TORCH_CUDA_ARCH_LIST as well: torch's CMake config replaces
        # CMAKE_CUDA_ARCHITECTURES with -gencode flags of its own, and with no
        # GPU in the builder its default list starts at compute_50, which CUDA
        # 13 removed ("nvcc fatal: Unsupported gpu architecture").
        f"cd {REPO_REMOTE} && CUDA_ARCHITECTURES='{CUDA_ARCHITECTURES}'"
        f" TORCH_CUDA_ARCH_LIST='{_torch_arch_list(CUDA_ARCHITECTURES)}'"
        " pip install -e . --no-build-isolation",
    )
    .workdir(REPO_REMOTE)
)


def _asset_env() -> dict[str, str]:
    """Asset env vars as the container sees them.  Evaluated *remotely*."""
    env = {name: f"{ASSETS_MOUNT}/{rel}" for name, rel in ASSET_LAYOUT.items()}
    # AUDIO_PATH is the one asset with no slot of its own: it is "some wav out
    # of WAV_DIR", a thing to pick rather than a thing to seed — which is what
    # tests/assets.py tells a human to do (`ls $WAV_DIR/*.wav | head -1`).  Left
    # unpicked it is simply unset in the container, and --strict-assets turns
    # that into a failed suite over a file the Volume already holds.  Sorted, so
    # every suite in a run agrees on which wav that is.
    wavs = sorted(Path(env["WAV_DIR"]).glob("*.wav")) if "WAV_DIR" in env else []
    if wavs:
        env["AUDIO_PATH"] = str(wavs[0])
    return env


def _jit_dir(cc: str | None = None) -> str:
    """Per-architecture JIT cache prefix -- this device's, or capability *cc*'s.

    *cc* ("9.0") is for a CPU container compiling ahead for an architecture
    it does not have; it must name the same prefix the GPU run derives.

    The cache *key* already separates architectures — ``_default_cuda_cflags``
    puts ``-gencode=...sm_XX`` into the flags ``JitSpec._content_hash`` hashes —
    so this is not a correctness fix.  It is about the Volume: five suites on
    five GPUs all committing under one prefix is five concurrent writers to the
    same paths, and a per-arch prefix makes each sweep's cache its own.
    """
    if cc is not None:
        return f"{JIT_MOUNT}/sm{cc.replace('.', '')}"
    try:
        import torch

        major, minor = torch.cuda.get_device_capability()
        return f"{JIT_MOUNT}/sm{major}{minor}"
    except Exception:
        return f"{JIT_MOUNT}/unknown"


def _cc_of(gpu: str) -> str:
    """``"sm_100"`` -> ``"10.0"``: the ``OASR_CUDA_ARCH_LIST`` spelling of *gpu*'s arch."""
    digits = GPU_ARCH[gpu].split("_")[1]
    return f"{digits[:-1]}.{digits[-1]}"


def _lane_of(cc: str) -> str:
    """CUTLASS lane of a capability: 3.x (TMA/wgmma/tcgen05) on sm_90/sm_100, else 2.x."""
    return "3x" if cc.split(".")[0] in ("9", "10") else "2x"


def _jit_libs(root: str) -> set[str]:
    """Every compiled library under a JIT prefix (``<name>/<hash>/<name>.so``)."""
    base = Path(root)
    return {str(p.relative_to(base)) for p in base.glob("*/*/*.so")} if base.is_dir() else set()


def _committing(vol, every_s: float = 60.0):
    """Start a thread committing *vol* every *every_s* seconds; returns its stop event.

    Modal preempts GPU containers and reschedules the input from the start, and
    a library compiled before that is lost unless it was committed.
    """
    import threading

    stop = threading.Event()

    def _loop():
        while not stop.wait(every_s):
            try:
                vol.commit()
            except Exception as exc:  # noqa: BLE001 -- the next tick retries
                print(f"[modal] volume commit failed: {exc}", flush=True)

    threading.Thread(target=_loop, daemon=True).start()
    return stop


def _usd(seconds: float, gpu: str | None, cpu: float, memory_mib: float) -> float:
    """List-price cost of a container alive for *seconds* at *cpu* cores and *memory_mib*."""
    per_s = GPU_USD_PER_S.get(gpu, 0.0) if gpu else 0.0
    per_s += cpu * CPU_USD_PER_CORE_S + memory_mib / 1024 * MEM_USD_PER_GIB_S
    return seconds * per_s


def _billed_shape(wall_s: float, request: tuple[float, int]) -> tuple[float, float]:
    """Estimated mean ``(cores, MiB)`` billed: the request, or the use above it.

    Use is read from this process's children (the pytest or prebuild tree):
    their CPU time over the wall for cores, and that many concurrent copies of
    the largest one's peak RSS for memory -- an estimate, not Modal's meter.
    """
    import resource

    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    cores = (ru.ru_utime + ru.ru_stime) / max(wall_s, 1.0)
    mib = max(cores, 1.0) * ru.ru_maxrss / 1024
    return max(request[0], cores), max(request[1], mib)


@app.function(
    image=image,
    cpu=(COMPILE_REQUEST[0], COMPILE["2x"][0]),
    memory=(COMPILE_REQUEST[1], COMPILE["2x"][1]),
    timeout=COMPILE_TIMEOUT_S,
    retries=0,
    volumes={JIT_MOUNT: jit_vol},
)
def precompile(cc: str, jobs: int) -> dict:
    """``oasr tune prebuild --all`` for capability *cc* on this CPU container.

    Writes into the arch's JIT prefix, where the GPU run finds every library:
    ``JitSpec``'s cache key is device-free (sources, headers, flags, toolchain
    identity) and this is the image the GPU run uses.  Returns the per-module
    report.
    """
    import json

    report = Path("/tmp/prebuild.json")
    env = {**os.environ, "OASR_CUDA_ARCH_LIST": cc, "OASR_JIT_DIR": _jit_dir(cc)}
    cmd = ["python", "-m", "oasr.cli", "tune", "prebuild", "--all"]
    cmd += ["--jobs", str(jobs), "--json", str(report)]
    print(f"[modal] precompile cc={cc} -j{jobs} jit={env['OASR_JIT_DIR']}", flush=True)
    t0 = time.time()
    stop = _committing(jit_vol)
    try:
        proc = subprocess.run(cmd, cwd=REPO_REMOTE, env=env)
    finally:
        stop.set()
        jit_vol.commit()
    out = json.loads(report.read_text()) if report.is_file() else {"modules": []}
    wall = time.time() - t0
    cores, mib = _billed_shape(wall, COMPILE_REQUEST)
    out.update(
        {
            "cc": cc,
            "returncode": proc.returncode,
            "container_s": round(wall, 1),
            "usd": round(_usd(wall, None, cores, mib), 3),
            "mean_cores": round(cores, 1),
        }
    )
    return out


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    cpu=GPU_CPU,
    memory=GPU_MEMORY_MIB,
    timeout=GPU_TIMEOUT_S,
    retries=0,
    # The container is billed while it idles before scaling down; a run is one
    # input, so there is nothing to wait for.
    scaledown_window=2,
    volumes={ASSETS_MOUNT: assets_vol, JIT_MOUNT: jit_vol},
)
def run_suite(
    name: str,
    strict: bool = True,
    extra: str = "",
    assets: bool = True,
    paths: str = "",
    gpu: str = "",
) -> dict:
    """Run one family from ci/gpu_suites.py, or the comma-separated *paths*.

    Returns the outcome rather than raising, so a failed run still reports its
    time, its cost and what it had to compile.  ``assets=False`` runs with the
    Volume unused: no asset env var is set, so every checkpoint- and
    audio-gated test skips and what is left is the kernel / layer / plumbing
    coverage that needs only a GPU.  ``strict`` stays meaningful there -- the
    allow-list simply widens to every declared asset, so the run still *names*
    what it did not cover instead of turning the gate off.
    """
    import collections

    import torch

    t0 = time.time()
    jit = _jit_dir()
    # MAX_JOBS: ninja's own default counts the host's cores, not this container's
    # limit, and a compile that reaches here would oversubscribe the memory.
    env = {**os.environ, "OASR_JIT_DIR": jit, "MAX_JOBS": str(int(GPU_CPU[1]))}
    if assets:
        env.update(_asset_env())
    device = torch.cuda.get_device_name()
    print(f"[modal] {device} suite={name} strict={strict} assets={assets} jit={jit}", flush=True)

    targets = [p.strip() for p in paths.split(",") if p.strip()] or paths_for(name)
    cmd = ["python", "-m", "pytest", *targets, "-m", DEFAULT_MARKER_EXPR, "-q", "-rfE"]
    cmd += ["-p", "no:cacheprovider", "--durations=15"]
    if strict:
        cmd.append("--strict-assets")
        allowed = ALLOW_MISSING if assets else tuple(test_assets.ASSETS)
        for missing in allowed:
            cmd += ["--allow-missing-asset", missing]
    if extra:
        cmd += extra.split()

    before = _jit_libs(jit)
    tail: collections.deque[str] = collections.deque(maxlen=60)
    failures: list[str] = []
    stop = _committing(jit_vol, every_s=120.0)
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_REMOTE,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            tail.append(line.rstrip())
            if line.startswith(("FAILED ", "ERROR ")):
                failures.append(line.rstrip())
        returncode = proc.wait()
    finally:
        # Persist whatever the JIT compiled so the next run starts warm.
        stop.set()
        jit_vol.commit()
    compiled = sorted({lib.split("/")[0] for lib in _jit_libs(jit) - before})
    wall = time.time() - t0
    summary = next(
        (
            ln
            for ln in reversed(tail)
            if " in " in ln and ("passed" in ln or "failed" in ln or "error" in ln)
        ),
        "",
    )
    return {
        "suite": name,
        "gpu": gpu,
        "device": device,
        "returncode": returncode,
        "summary": summary.strip("= "),
        "wall_s": round(wall, 1),
        "usd": round(_usd(wall, gpu, *_billed_shape(wall, (GPU_CPU[0], GPU_MEMORY_MIB[0]))), 3),
        "compiled_on_gpu": compiled,
        # What to show for a failed run: pytest's own short summary, else (a
        # crash, a collection error) the end of the output.
        "tail": failures[:40] or (list(tail)[-25:] if returncode != 0 else []),
    }


@app.function(
    image=image,
    timeout=10800,
    volumes={ASSETS_MOUNT: assets_vol},
)
def _fetch(names: list[str], force: bool, wav_limit: int) -> list[str]:
    """Download the declared assets into the Volume.  CPU only, no GPU rented."""
    # Commit after every asset, not once at the end: a 20 GiB fetch that dies on
    # the ninth download should keep the eight that landed.
    lines = modal_assets.fetch(
        Path(ASSETS_MOUNT),
        names=names,
        force=force,
        wav_limit=wav_limit,
        relpaths=ASSET_LAYOUT,
        commit=assets_vol.commit,
    )
    assets_vol.commit()
    return lines


@app.local_entrypoint()
def fetch_assets(
    only: str = "",
    skip: str = "",
    force: bool = False,
    wav_limit: int = 600,
    dry_run: bool = False,
):
    """Populate the ``oasr-ci-assets`` Volume from the internet, in the container.

    Run once; it is idempotent, so a later invocation costs one marker check per
    asset.  Roughly 20 GiB in total, three quarters of it Qwen2-Audio-7B —
    ``--skip SPEECH_LLM_CKPT`` leaves that out and keeps everything else.

        modal run ci/modal_app.py::fetch_assets
        modal run ci/modal_app.py::fetch_assets --only WAV_DIR,NEMOTRON_CKPT
        modal run ci/modal_app.py::fetch_assets --skip SPEECH_LLM_CKPT
        modal run ci/modal_app.py::fetch_assets --dry-run
    """
    names = [s.strip() for s in only.split(",") if s.strip()]
    dropped = {s.strip() for s in skip.split(",") if s.strip()}
    wanted = [s.env for s in modal_assets.plan(names) if s.env not in dropped]

    print("[modal] fetching, inside the container:")
    total = 0
    for env_name in wanted:
        src = modal_assets.SOURCES[env_name]
        total += src.approx_mib
        where = src.repo or f"{len(src.urls)} file(s)"
        print(f"  {env_name:<24} ~{src.approx_mib:>6} MiB  {where}")
    print(f"  {'':<24} ~{total:>6} MiB total")
    if modal_assets.NO_PUBLIC_SOURCE:
        print("\n  no public source (upload with seed_assets, or leave missing):")
        for env_name, why in modal_assets.NO_PUBLIC_SOURCE.items():
            print(f"    {env_name:<22} {why.split('.')[0]}.")

    if dry_run:
        print("\n--dry-run: nothing fetched")
        return

    print()
    for line in _fetch.remote(wanted, force, wav_limit):
        print(line)


#: Consecutive polling errors before a stage counts as failed.  A transient
#: "Deadline exceeded" from the control plane once read as a failed compile and
#: abandoned a run whose compile had succeeded (2026-10-02).
_POLL_TOLERANCE = 5


def _poll(call, errors: dict, key: str):
    """``call``'s result if it has one now, else ``None`` (never blocks).

    Raises only after ``_POLL_TOLERANCE`` consecutive errors -- a remote failure
    repeats on every poll, a network blip does not.
    """
    try:
        res = call.get(timeout=0)
    except (TimeoutError, modal.exception.TimeoutError):
        errors[key] = 0
        return None
    except Exception as exc:  # noqa: BLE001 -- counted, re-raised when persistent
        errors[key] = errors.get(key, 0) + 1
        if errors[key] >= _POLL_TOLERANCE:
            raise
        print(f"  [poll] {key}: {type(exc).__name__}: {exc} (retrying)", flush=True)
        return None
    errors[key] = 0
    return res


def _print_run(res: dict) -> None:
    status = "ok  " if res["returncode"] == 0 else "FAIL"
    print(
        f"  {status} {res['gpu']} / {res['suite']}: {res['summary'] or 'no summary'} "
        f"[{res['wall_s']:.0f}s, ~${res['usd']:.2f}]",
        flush=True,
    )
    if res["compiled_on_gpu"]:
        print(f"       compiled on the GPU: {', '.join(res['compiled_on_gpu'])}", flush=True)
    for line in res.get("tail", []):
        print(f"       | {line}", flush=True)


@app.local_entrypoint()
def main(
    gpus: str = "",
    suites: str = "",
    paths: str = "",
    strict: bool = True,
    extra: str = "",
    assets: bool = True,
    precompile_first: bool = True,
    run_on_compile_failure: bool = False,
    compile_only: bool = False,
    timeout_s: int = GPU_TIMEOUT_S,
    max_usd: float = 10.0,
    dry_run: bool = False,
):
    """Run the families on one or more accelerators: compile on CPU, then test on GPU.

    ``--gpus`` is a comma-separated list of Modal accelerators, ``all`` for one
    per distinct architecture (``ARCH_SWEEP``), or empty for the default.
    ``--suites`` is a comma-separated list of names from ci/gpu_suites.py; empty
    means all of them.  ``--paths`` replaces the families with explicit test
    files or node ids, run as one suite.  Every (gpu, suite) pair of an
    architecture starts as soon as that architecture's compile is done.

    ``--assets`` reads the checkpoints and audio out of the ``oasr-ci-assets``
    Volume; ``--no-assets`` is the GPU-only coverage that needs no external
    input.  ``--no-precompile-first`` skips the CPU stage (the cache is known
    warm), ``--compile-only`` runs nothing else (does it build for a B200? for
    the price of CPU time), ``--run-on-compile-failure`` rents the GPU even for
    an architecture some module failed to compile for, and ``--max-usd`` refuses
    a plan whose worst case -- every run hitting ``--timeout-s`` -- costs more.
    """
    path_list = [p.strip() for p in paths.split(",") if p.strip()]
    if path_list:
        names = ["paths"]
        missing = [p for p in path_list if not (REPO_ROOT / p.split("::")[0]).exists()]
        if missing:
            raise SystemExit(f"no such test path(s): {missing}")
    else:
        names = [s.strip() for s in suites.split(",") if s.strip()] or list(SUITES)
        unknown = [n for n in names if n not in SUITES]
        if unknown:
            raise SystemExit(f"unknown suite(s): {unknown}; known: {list(SUITES)}")

    if gpus.strip().lower() == "all":
        targets = list(ARCH_SWEEP)
    else:
        targets = [g.strip() for g in gpus.split(",") if g.strip()] or [DEFAULT_GPU]
    unknown_gpu = [g for g in targets if g not in GPU_ARCH or g not in GPU_USD_PER_S]
    if unknown_gpu:
        raise SystemExit(f"unknown accelerator(s) {unknown_gpu}; known: {sorted(GPU_ARCH)}")

    by_cc: dict[str, list[str]] = {}
    for g in targets:
        by_cc.setdefault(_cc_of(g), []).append(g)

    scope = "with checkpoints" if assets else "no checkpoints (kernels/plumbing only)"
    print(
        f"[modal] {len(targets)} accelerator(s) x {len(names)} suite(s), {scope}\n"
        f"        suites: {', '.join(path_list or names)}"
    )
    if compile_only and not precompile_first:
        raise SystemExit("--compile-only needs the compile stage (--precompile-first)")
    worst = 0.0
    if precompile_first:
        for cc in by_cc:
            cores, mem = COMPILE[_lane_of(cc)]
            cost = _usd(COMPILE_TIMEOUT_S, None, cores, mem)  # at its limits throughout
            worst += cost
            print(
                f"        compile sm_{cc.replace('.', ''):<4} {cores} CPU / {mem // 1024} GiB, "
                f"<= {COMPILE_TIMEOUT_S // 60} min (<= ${cost:.2f})"
            )
    for g in [] if compile_only else targets:
        cost = _usd(timeout_s, g, GPU_CPU[1], GPU_MEMORY_MIB[1]) * len(names)
        worst += cost
        print(
            f"        test    {g:<13} {GPU_ARCH[g]:<7} {len(names)} run(s), each "
            f"<= {timeout_s // 60} min (<= ${cost:.2f})"
        )
    print(f"        worst case ${worst:.2f} at list prices (limit --max-usd {max_usd:.2f})")
    if worst > max_usd:
        raise SystemExit(
            f"worst case ${worst:.2f} exceeds --max-usd {max_usd:.2f}: narrow the run, "
            f"lower --timeout-s, or raise the limit"
        )
    if dry_run:
        print("--dry-run: nothing spawned")
        return

    def _spawn_runs(cc: str) -> list:
        return [
            (
                g,
                name,
                run_suite.with_options(gpu=g, timeout=timeout_s).spawn(
                    name, strict, extra, assets, paths, g
                ),
            )
            for g in by_cc[cc]
            for name in names
        ]

    t0 = time.time()
    pre: dict = {}
    runs: list = []
    if precompile_first:
        for cc in by_cc:
            cores, mem = COMPILE[_lane_of(cc)]
            pre[cc] = precompile.with_options(
                cpu=(COMPILE_REQUEST[0], cores), memory=(COMPILE_REQUEST[1], mem)
            ).spawn(cc, cores)
    else:
        for cc in by_cc:
            runs += _spawn_runs(cc)

    results: list[dict] = []
    failed: list[str] = []
    spent = 0.0
    errors: dict = {}
    while pre or runs:
        for cc in list(pre):
            try:
                rep = _poll(pre[cc], errors, f"compile:{cc}")
            except Exception as exc:  # the container itself failed
                rep = {"returncode": -1, "modules": [], "error": str(exc)}
            if rep is None:
                continue
            del pre[cc]
            cost = rep.get("usd", 0.0)
            spent += cost
            mods = rep.get("modules", [])
            bad = [m["name"] for m in mods if m["status"] == "failed"]
            built = sum(m["status"] == "built" for m in mods)
            print(
                f"  sm_{cc.replace('.', '')}: compile rc={rep.get('returncode')} -- {built} built, "
                f"{len(mods) - built - len(bad)} cached, {len(bad)} failed "
                f"[{rep.get('container_s', 0)}s, {rep.get('mean_cores', '?')} cores on average, "
                f"~${cost:.2f}] at {time.time() - t0:.0f}s",
                flush=True,
            )
            for m in mods:
                if m.get("error"):
                    print(f"    {m['name']}: ...{m['error'][-1200:]}", flush=True)
            if rep.get("error"):
                print(f"    {rep['error'][-1200:]}", flush=True)
            if rep.get("returncode") != 0 or bad:
                failed += [f"sm_{cc.replace('.', '')} (compile)"]
                if not run_on_compile_failure:
                    print(
                        f"    not renting {', '.join(by_cc[cc])}: fix the compile first", flush=True
                    )
                    continue
            if not compile_only:
                runs += _spawn_runs(cc)
        for item in list(runs):
            g, name, call = item
            try:
                res = _poll(call, errors, f"{g}/{name}")
            except Exception as exc:  # one run failing must not hide the others
                runs.remove(item)
                failed.append(f"{g} / {name}")
                print(f"  FAIL {g} / {name}: {type(exc).__name__}: {exc}", flush=True)
                continue
            if res is None:
                continue
            runs.remove(item)
            results.append(res)
            spent += res["usd"]
            _print_run(res)
            if res["returncode"] != 0:
                failed.append(f"{g} / {name}")
        if pre or runs:
            time.sleep(10)

    print(f"\n[modal] done in {time.time() - t0:.0f}s, ~${spent:.2f} at list prices", flush=True)
    for res in results:
        _print_run(res)
    if failed:
        raise SystemExit(f"{len(failed)} run(s) failed: {', '.join(failed)}")


@app.local_entrypoint()
def seed_assets(only: str = "", dry_run: bool = False):
    """Upload the assets that have no public source.  Run once, from a box with them.

    This is the residue of what used to be the whole story.  Everything with a
    URL is fetched by ``fetch_assets`` instead; what is left is a WeNet release
    whose download page now 404s, a generated fixture, and a locally built k2
    graph — see ``modal_assets.NO_PUBLIC_SOURCE`` for why each one is here.
    Source paths come from the same env vars the suite reads::

        set -a; source .env; set +a
        modal run ci/modal_app.py::seed_assets --dry-run
        modal run ci/modal_app.py::seed_assets --only CKPT_DIR,SPEECH_LLM_TINY

    Default is every no-public-source asset that resolves locally.  Re-running
    overwrites in place, so it is safe for a single refreshed asset.
    """
    wanted = [s.strip() for s in only.split(",") if s.strip()] or list(
        modal_assets.NO_PUBLIC_SOURCE
    )
    plan, missing, unslotted = [], [], []
    for name in wanted:
        if name not in ASSET_LAYOUT:
            unslotted.append(name)
            continue
        src = test_assets.resolve(name)
        if src is None:
            missing.append(f"{name} (looked at {test_assets.declared(name) or '<unset>'})")
            continue
        plan.append((name, Path(src), ASSET_LAYOUT[name]))

    for name, src, rel in plan:
        kind = "dir " if src.is_dir() else "file"
        print(f"  {kind} {name:<22} {src}  ->  {ASSETS_MOUNT}/{rel}")
    for m in missing:
        print(f"  SKIP {m}")
    for name in unslotted:
        print(f"  SKIP {name} (no relpath in tests/assets.py — nothing to seed it into)")
    if not plan:
        raise SystemExit("nothing to upload; export the asset env vars first")
    if dry_run:
        print("\n--dry-run: nothing uploaded")
        return

    with assets_vol.batch_upload(force=True) as batch:
        for _, src, rel in plan:
            if src.is_dir():
                batch.put_directory(str(src), f"/{rel}")
            else:
                batch.put_file(str(src), f"/{rel}")
    print(f"\nuploaded {len(plan)} asset(s) to the oasr-ci-assets Volume")
