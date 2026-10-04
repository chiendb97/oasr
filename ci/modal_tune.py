#!/usr/bin/env python3
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Build a GEMM tuning file per GPU architecture on rented accelerators.

Usage::

    oasr tune census --ckpt-dir ... --out .artifacts/tune_matrix/census/x.json
    modal run ci/modal_tune.py --dry-run                      # the plan; spends nothing
    modal run ci/modal_tune.py                                # A100, L40S, H100, B200
    modal run ci/modal_tune.py --gpus H100 --census a.json,b.json

A separate app from ``ci/modal_app.py`` on purpose: ``modal run`` builds the
image of every function in the app it runs, and the test image's last layer
copies the whole tree and rebuilds the Rust core -- so tuning inside that app
would rebuild it on every run after any edit, for an image tuning never uses.
It shares the accelerator table, the CUTLASS pin and the JIT Volume.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent))
if Path("/repo/ci/modal_app.py").is_file():  # inside a container
    sys.path.insert(0, "/repo/ci")
from modal_app import (  # noqa: E402  -- needs the path line above
    ARCH_SWEEP,
    CUTLASS_HOME,
    CUTLASS_REF,
    DEFAULT_GPU,
    GPU_ARCH,
    JIT_MOUNT,
    JIT_VOLUME,
    REPO_REMOTE,
    REPO_ROOT,
    _jit_dir,
)

app = modal.App("oasr-gpu-tune")
jit_vol = modal.Volume.from_name(JIT_VOLUME, create_if_missing=True)
#: Per-run checkpoints and logs, ``/runs/<run_id>/sm<XY>/``: what makes a
#: preempted GPU input -- which Modal reschedules, from the start -- resume.
RUNS_MOUNT = "/runs"
runs_vol = modal.Volume.from_name("oasr-tune-runs", create_if_missing=True)


def _blob_transfers_use_the_proxy() -> None:
    """Route Modal's blob transfers through this machine's HTTPS proxy, if it has one.

    A function output over 2 MiB -- a tuning run's measurement log -- comes back
    through blob storage, over an aiohttp session Modal builds without
    ``trust_env``.  On a machine that reaches the internet only through
    ``https_proxy``, that download fails ("Temporary failure in name resolution")
    *after* the GPU time was spent, and the result is gone.  Local process only,
    and a no-op without a proxy variable.  Applied at import, before ``modal run``
    creates (and caches) the session for its first upload.
    """
    if not modal.is_local() or not (os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")):
        return
    try:
        from modal._utils import http_utils
    except ImportError:  # the private module moved: keep the default client
        return

    def _session(timeout):
        import ssl

        import certifi
        from aiohttp import ClientSession, ClientTimeout, TCPConnector

        ssl_context = ssl.create_default_context(cafile=certifi.where())
        return ClientSession(
            connector=TCPConnector(ssl=ssl_context),
            timeout=ClientTimeout(total=timeout),
            trust_env=True,
        )

    http_utils._http_client_with_tls = _session


_blob_transfers_use_the_proxy()

# Where a tuning run's money goes, and what each piece below does about it:
#
# * **nvcc, not measurement, dominates a cold run -- and needs no GPU.** A GPU
#   container compiles on the small CPU share it comes with, at the GPU's rate,
#   and a template that does not compile for the architecture is discovered
#   only after the GPU was rented.  So each architecture's modules are compiled
#   first on a CPU-only container sized for nvcc (``precompile_tuning``,
#   ``oasr tune prebuild`` with ``OASR_CUDA_ARCH_LIST``), straight into the
#   per-arch JIT prefix the GPU run reads.  ``JitSpec``'s cache key is
#   device-free (sources, headers, flags, nvcc identity), so the GPU container
#   finds every library -- and reports it if it had to compile anything anyway.
# * **The GPU container only measures** (``tune_arch``), and is spawned the
#   moment *its own* architecture's compile lands, not after the slowest one.
#   ``--budget-s`` stops the build below the hard ``timeout``: the build writes
#   what it measured (heaviest signatures first), where a timeout discards it.
# * **An image of its own.**  Tuning imports no compiled extension (``oasr._C``,
#   ``oasr._core``), so it needs neither the Rust toolchain nor
#   ``pip install -e .``, and its sources are *mounted*, not baked into a
#   layer: a code change starts containers without an image rebuild.  CUDA
#   13.2 because the test image's 12.8 cannot build the 100 family at all --
#   its nvcc rejects ``compute_100f`` ("Unsupported gpu architecture", probed
#   2026-09-29), which ``oasr.jit.core._GENCODE_TARGET`` targets for sm_100 --
#   and 13.2 with torch 2.12.0+cu132 is the toolchain the tree was validated
#   on.  (cu132 has no torchaudio, which is why the *test* image cannot simply
#   follow; tuning does not need it.)
# * **Preemptible, and resumable.**  Modal preempts GPU containers and reschedules
#   the input from the start (observed: all four at once, 10 min into a run).
#   Non-preemptible capacity costs more; a checkpoint is cheaper.  The build
#   records each finished signature (``--checkpoint``) on the runs Volume,
#   committed every minute and on the way out, so a rescheduled input -- or
#   ``--run-id`` after the local client died -- re-measures only the signature
#   it was in.
# * **Measure fewer, near-identical points.**  A streaming census keys every
#   cohort width, so one signature can carry 140 Ms a few percent apart; the
#   build measures a log-spaced ``TUNE_MAX_POINTS`` of them (``--thin``), the
#   dropped points' traffic folded into the kept ones.  What that costs in
#   regret is scored offline against a full build on a local card first.
# * **No assets Volume, no retries.**  The census is computed locally -- it
#   depends on the model and the engine configuration, not on the GPU -- and
#   shipped as JSON; a failed arch is reported, never re-run at GPU prices.

#: One accelerator per architecture family the shipped DB lacks.  sm_120 is
#: tuned on a developer card (``oasr/tune/db/README.md``); ``--gpus all`` adds
#: the RTX PRO 6000 for an sm_120 SKU overlay.
TUNE_GPUS = ("A100-40GB", "L40S", "H100", "B200")

TUNE_CUDA = os.environ.get("OASR_MODAL_TUNE_CUDA", "13.2.0")
TUNE_TORCH = os.environ.get("OASR_MODAL_TUNE_TORCH", "torch==2.12.0")
TUNE_TORCH_INDEX = os.environ.get(
    "OASR_MODAL_TUNE_TORCH_INDEX", "https://download.pytorch.org/whl/cu132"
)

#: The user tier both stages point at: empty, and the *same* path, because the
#: production compile set includes every config the tiers reference -- a
#: different tier on one side is a different module hash.
TUNE_USER_DB = "/tmp/oasr_tune_user"

#: Compile container per CUTLASS lane: ``(cores, memory MiB)``, cores = nvcc jobs.
#: Cost is core-seconds plus GiB-seconds, so cores cost the same while each has
#: a TU to compile and only cut the wall time; past the TU count they idle,
#: billed.  Memory is jobs x the lane's peak compiler RSS plus headroom -- the
#: two lanes differ 4x, which is why this is per lane.  Measured 2026-09-29 with
#: ``oasr tune prebuild -j32`` (nvcc 13.2, every tuning module, cold):
#:
#:   sm_80  49 TUs  64 s  peak 1.1 GiB  |  sm_90   21 TUs  80 s  peak 4.1 GiB
#:   sm_89  41 TUs  46 s  peak 1.1 GiB  |  sm_100  33 TUs  72 s  peak 4.3 GiB
TUNE_COMPILE = {"2x": (32, 40 * 1024), "3x": (24, 128 * 1024)}
TUNE_COMPILE_TIMEOUT_S = 1800

#: GPU container: the build's own budget sits below the hard timeout.
TUNE_GPU_TIMEOUT_S = 3000
TUNE_BUDGET_S = 2400
#: Its CPU and memory bill on top of the GPU for the whole run (measured on the
#: 5090, 2026-10-01: one busy core, peak RSS 1.7 GiB).
TUNE_GPU_CPU = 2
TUNE_GPU_MEMORY_MIB = 6 * 1024

#: Points per signature, census points included (``oasr tune build --thin``).
#: The streaming censuses key every cohort width -- up to 142 Ms per signature,
#: 3422 points in all -- and neighbouring widths pick the same config.
TUNE_MAX_POINTS = 16

_TUNE_IGNORE = ["**/__pycache__", "**/*.pyc", "**/*.so", "**/.pytest_cache"]

tune_image = (
    modal.Image.from_registry(f"nvidia/cuda:{TUNE_CUDA}-devel-ubuntu24.04", add_python="3.12")
    .apt_install("curl", "g++")
    # The same pinned-CUTLASS layer as the test image (see there).
    .run_commands(
        f"mkdir -p {CUTLASS_HOME}",
        f"curl -sSL https://github.com/NVIDIA/cutlass/archive/{CUTLASS_REF}.tar.gz"
        f" | tar xz --strip-components=1 -C {CUTLASS_HOME}"
        f" cutlass-{CUTLASS_REF}/include cutlass-{CUTLASS_REF}/tools/util/include",
        f"test -f {CUTLASS_HOME}/include/cutlass/version.h",
        f"mkdir -p {REPO_REMOTE}/3rdparty && ln -sfn {CUTLASS_HOME} {REPO_REMOTE}/3rdparty/cutlass",
    )
    .pip_install(TUNE_TORCH, index_url=TUNE_TORCH_INDEX)
    .pip_install(
        "numpy",
        "apache-tvm-ffi==0.1.10",
        "jinja2>=3.0",
        "PyYAML>=5.4",
        "ninja",
        # pynvml: the bench protocol's contention and clock records.
        "nvidia-ml-py",
        # This file imports tests/assets.py, which imports pytest.
        "pytest==8.1.1",
    )
    .env(
        {
            "PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "PYTHONPATH": REPO_REMOTE,
            "OASR_TUNE_USER_DB": TUNE_USER_DB,
        }
    )
    .workdir(REPO_REMOTE)
    # Mounted at container start (no copy=True): a source edit costs no rebuild.
    .add_local_dir(REPO_ROOT / "oasr", f"{REPO_REMOTE}/oasr", ignore=_TUNE_IGNORE)
    .add_local_dir(REPO_ROOT / "csrc", f"{REPO_REMOTE}/csrc", ignore=_TUNE_IGNORE)
    .add_local_dir(REPO_ROOT / "include", f"{REPO_REMOTE}/include", ignore=_TUNE_IGNORE)
    .add_local_dir(REPO_ROOT / "ci", f"{REPO_REMOTE}/ci", ignore=_TUNE_IGNORE)
    .add_local_file(REPO_ROOT / "tests" / "assets.py", f"{REPO_REMOTE}/tests/assets.py")
)


def _cc_of(gpu: str) -> str:
    """``"sm_100"`` -> ``"10.0"``: the ``OASR_CUDA_ARCH_LIST`` spelling of *gpu*'s arch."""
    digits = GPU_ARCH[gpu].split("_")[1]
    return f"{digits[:-1]}.{digits[-1]}"


def _lane_of(cc: str) -> str:
    """CUTLASS lane of a capability: 3.x (TMA/wgmma/tcgen05) on sm_90/sm_100, else 2.x."""
    return "3x" if cc.split(".")[0] in ("9", "10") else "2x"


def _census_files(spec: str) -> list[Path]:
    """Comma-separated files and/or directories (every ``*.json`` inside)."""
    out: list[Path] = []
    for part in (s.strip() for s in spec.split(",") if s.strip()):
        p = Path(part)
        out.extend(sorted(p.glob("*.json")) if p.is_dir() else [p])
    return out


def _write_censuses(texts: list[str]) -> list[str]:
    work = Path("/tmp/tune")
    work.mkdir(parents=True, exist_ok=True)
    args: list[str] = []
    for i, text in enumerate(texts):
        p = work / f"census_{i}.json"
        p.write_text(text)
        args += ["--census", str(p)]
    return args


def _jit_libs(root: str) -> set[str]:
    """Every compiled library under a JIT prefix (``<name>/<hash>/<name>.so``)."""
    base = Path(root)
    return {str(p.relative_to(base)) for p in base.glob("*/*/*.so")} if base.is_dir() else set()


@app.function(
    image=tune_image,
    cpu=TUNE_COMPILE["2x"][0],
    memory=TUNE_COMPILE["2x"][1],
    timeout=TUNE_COMPILE_TIMEOUT_S,
    retries=0,
    volumes={JIT_MOUNT: jit_vol},
)
def precompile_tuning(cc: str, censuses: list[str], jobs: int) -> dict:
    """``oasr tune prebuild`` for *cc* on this CPU container, into the arch's JIT prefix."""
    import json

    report = Path("/tmp/tune/prebuild.json")
    env = {**os.environ, "OASR_CUDA_ARCH_LIST": cc, "OASR_JIT_DIR": _jit_dir(cc)}
    cmd = ["python", "-m", "oasr.cli", "tune", "prebuild", *_write_censuses(censuses)]
    cmd += ["--jobs", str(jobs), "--json", str(report)]
    print(f"[modal] prebuild cc={cc} -j{jobs} jit={env['OASR_JIT_DIR']}", flush=True)
    try:
        proc = subprocess.run(cmd, cwd=REPO_REMOTE, env=env)
    finally:
        jit_vol.commit()
    out = json.loads(report.read_text()) if report.is_file() else {"modules": []}
    out.update({"cc": cc, "returncode": proc.returncode})
    return out


def _committing(vol, every_s: float = 60.0):
    """Start a thread committing *vol* every *every_s* seconds; returns its stop event."""
    import threading

    stop = threading.Event()

    def _loop():
        while not stop.wait(every_s):
            try:
                vol.commit()
            except Exception as exc:  # noqa: BLE001 -- the next tick retries
                print(f"[modal] runs volume commit failed: {exc}", flush=True)

    threading.Thread(target=_loop, daemon=True).start()
    return stop


@app.function(
    image=tune_image,
    gpu=DEFAULT_GPU,
    # The build is one Python thread waiting on the GPU (user time = wall time on
    # the 5090), peak RSS 1.7 GiB: one core for it, one for the Volume commits.
    cpu=TUNE_GPU_CPU,
    memory=TUNE_GPU_MEMORY_MIB,
    timeout=TUNE_GPU_TIMEOUT_S,
    retries=0,
    volumes={JIT_MOUNT: jit_vol, RUNS_MOUNT: runs_vol},
)
def tune_arch(
    censuses: list[str],
    run_id: str,
    top_k: int = 16,
    min_speedup: float = 1.05,
    max_points: int = TUNE_MAX_POINTS,
    budget_s: float = TUNE_BUDGET_S,
    thin: bool = True,
) -> dict:
    """``oasr tune build`` on this GPU over the censuses (JSON text); measures only.

    Returns the tuning file and the (LZMA-compressed) measurement log.  Nothing
    is written to the Volume unless the JIT had to compile something the
    precompile stage missed -- which the result names, since that compile ran
    at GPU prices.
    """
    import lzma
    import time

    import torch

    jit = _jit_dir()
    env = {**os.environ, "OASR_JIT_DIR": jit}
    major, minor = torch.cuda.get_device_capability()
    work = Path(RUNS_MOUNT) / run_id / f"sm{major}{minor}"
    work.mkdir(parents=True, exist_ok=True)
    out, log, ckpt = work / "gemm.json", work / "measurements.jsonl", work / "checkpoint.jsonl"
    resuming = ckpt.is_file() and ckpt.stat().st_size > 0
    cmd = ["python", "-m", "oasr.cli", "tune", "build", *_write_censuses(censuses)]
    cmd += ["--out", str(out), "--fresh", "--log", str(log), "--top-k", str(top_k)]
    cmd += ["--min-speedup", str(min_speedup), "--max-points", str(max_points)]
    cmd += ["--budget-s", str(budget_s), "--checkpoint", str(ckpt)]
    if thin:
        cmd.append("--thin")
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,power.limit", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    print(f"[modal] {smi} jit={jit} run={work}{' (resuming)' if resuming else ''}", flush=True)
    before = _jit_libs(jit)
    t0 = time.time()
    stop = _committing(runs_vol)
    try:
        proc = subprocess.run(cmd, cwd=REPO_REMOTE, env=env)
    finally:  # also on the SIGINT of a preemption: keep what was measured
        stop.set()
        runs_vol.commit()
    wall = time.time() - t0
    compiled = sorted(_jit_libs(jit) - before)
    if compiled:
        jit_vol.commit()
    return {
        "sm": major * 10 + minor,
        "gpu": torch.cuda.get_device_name(0),
        "nvidia_smi": smi,
        "returncode": proc.returncode,
        "wall_s": round(wall, 1),
        "resumed": resuming,
        "compiled_on_gpu": compiled,
        "tuning_file": out.read_text() if out.is_file() else "",
        "measurements_xz": lzma.compress(log.read_bytes()) if log.is_file() else b"",
    }


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


@app.local_entrypoint()
def main(
    gpus: str = "",
    census: str = ".artifacts/tune_matrix/census",
    out_dir: str = ".artifacts/tune_matrix",
    top_k: int = 16,
    min_speedup: float = 1.05,
    max_points: int = TUNE_MAX_POINTS,
    budget_s: int = TUNE_BUDGET_S,
    thin: bool = True,
    run_id: str = "",
    dry_run: bool = False,
):
    """Build a tuning file per architecture: compile on CPU, then measure on GPU.

    ``--run-id`` names the run's checkpoint directory on the runs Volume.  A new
    one is minted by default; passing a previous run's id resumes it (each arch
    re-measures only what it had not finished), e.g. after the local client died.

        oasr tune census --ckpt-dir ... --out .artifacts/tune_matrix/census/x.json
        modal run ci/modal_tune.py --dry-run                 # the plan; spends nothing
        modal run ci/modal_tune.py                           # A100, L40S, H100, B200
        modal run ci/modal_tune.py --gpus H100 --census a.json,b.json
        oasr tune diff oasr/tune/db/sm90/gemm.json .artifacts/tune_matrix/sm90/gemm.json

    Results land in ``out_dir/sm<family>/`` (tuning file, measurement log, the
    compile and run reports) for review -- promoting one to the shipped tier is
    a copy into ``oasr/tune/db/`` in a reviewed change, never automatic.
    """
    import json
    import lzma
    import time

    from oasr.jit.core import _SM_FAMILY
    from oasr.tune.build import TUNED_OPS, _fill_points, thin_census
    from oasr.tune.census import ShapeSet

    files = _census_files(census)
    if not files:
        raise SystemExit(f"no census under {census!r} (produce one with `oasr tune census`)")
    # Validate locally, before anything is rented: a census that does not load
    # fails here for free instead of in five containers.
    uniq: dict = {}
    for f in files:
        for p in ShapeSet.load(str(f)).points:
            if p.op in TUNED_OPS:
                uniq.setdefault(p.sig, {}).setdefault(p.M, p)
    # What the build will measure: the census Ms (thinned, with --thin) plus fill.
    points = sum(
        len(
            _fill_points(
                [p.M for p in (thin_census(list(d.values()), max_points) if thin else d.values())],
                max_points,
            )
        )
        for d in uniq.values()
    )
    texts = [f.read_text() for f in files]
    names = [g.strip() for g in gpus.split(",") if g.strip()]
    targets = list(ARCH_SWEEP) if gpus.strip().lower() == "all" else (names or list(TUNE_GPUS))
    unknown = [g for g in targets if g not in GPU_ARCH]
    if unknown:
        raise SystemExit(f"unknown accelerator(s) {unknown}; known: {sorted(GPU_ARCH)}")

    run_id = run_id or time.strftime("%Y%m%d-%H%M%S")
    print(
        f"[modal] run {run_id}: {len(files)} census file(s), {len(uniq)} tuned signatures, "
        f"{points} points to measure ({'thinned to ' if thin else 'fill to '}{max_points}/signature)"
    )
    for g in targets:
        cc = _cc_of(g)
        cpu, mem = TUNE_COMPILE[_lane_of(cc)]
        print(
            f"  {g:<13} sm_{cc.replace('.', ''):<4} compile: {cpu} CPU / {mem // 1024} GiB "
            f"(<= {TUNE_COMPILE_TIMEOUT_S // 60} min)  measure: <= {budget_s // 60} min "
            f"(hard stop {TUNE_GPU_TIMEOUT_S // 60} min)"
        )
    if dry_run:
        print("--dry-run: nothing spawned")
        return

    # Stage 1: every architecture's compile at once, on CPU.
    pre = {}
    for g in targets:
        cpu, mem = TUNE_COMPILE[_lane_of(_cc_of(g))]
        pre[g] = precompile_tuning.with_options(cpu=cpu, memory=mem).spawn(_cc_of(g), texts, cpu)
    # Stage 2: each GPU as soon as its own compile is done.
    runs: dict = {}
    reports: dict = {}
    t0 = time.time()
    poll_errors: dict = {}
    while pre or runs:
        for g in list(pre):
            try:
                rep = _poll(pre[g], poll_errors, f"compile:{g}")
            except Exception as exc:  # the container itself failed
                rep = {"returncode": -1, "modules": [], "error": str(exc)}
            if rep is None:
                continue
            del pre[g]
            reports[g] = rep
            failed = [m["name"] for m in rep.get("modules", []) if m["status"] == "failed"]
            took = f"{time.time() - t0:.0f}s"
            if rep.get("returncode") != 0 or failed:
                print(f"  {g}: compile FAILED ({failed or rep.get('error', '?')}) after {took}")
                for m in rep.get("modules", []):
                    if m.get("error"):
                        print(f"    {m['name']}: ...{m['error'][-800:]}")
                continue
            built = sum(m["status"] == "built" for m in rep["modules"])
            print(
                f"  {g}: compiled ({built} built, wall {rep.get('wall_s')}s) at {took}; measuring"
            )
            runs[g] = tune_arch.with_options(gpu=g).spawn(
                texts, run_id, top_k, min_speedup, max_points, float(budget_s), thin
            )
        for g in list(runs):
            try:
                res = _poll(runs[g], poll_errors, f"measure:{g}")
            except Exception as exc:  # one arch failing must not hide the others
                print(f"  {g}: measure FAILED {exc}")
                del runs[g]
                continue
            if res is None:
                continue
            del runs[g]
            family = _SM_FAMILY.get(int(res["sm"]), int(res["sm"]))
            dest = Path(out_dir) / f"sm{family}"
            dest.mkdir(parents=True, exist_ok=True)
            if res["tuning_file"]:
                (dest / "gemm.json").write_text(res["tuning_file"])
            if res["measurements_xz"]:
                (dest / "measurements.jsonl").write_bytes(lzma.decompress(res["measurements_xz"]))
            meta = {k: v for k, v in res.items() if k not in ("tuning_file", "measurements_xz")}
            (dest / "run.json").write_text(json.dumps({**meta, "prebuild": reports[g]}, indent=1))
            extra = f"; COMPILED ON GPU: {res['compiled_on_gpu']}" if res["compiled_on_gpu"] else ""
            resumed = " (resumed)" if res.get("resumed") else ""
            print(
                f"  {g} ({res['gpu']}, sm{res['sm']}): rc={res['returncode']} "
                f"in {res['wall_s']:.0f}s{resumed} -> {dest}{extra}"
            )
        if pre or runs:
            time.sleep(10)
