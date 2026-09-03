#!/usr/bin/env python3
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Run the GPU test suites on rented accelerators, one image across every arch.

    modal run ci/modal_app.py::fetch_assets                     # once, ~20 GiB
    modal run ci/modal_app.py::main --gpus L40S,H100,B200
    modal run ci/modal_app.py::main --gpus H100 --suites engine,models
"""

from __future__ import annotations

import os
import subprocess
import sys
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

#: Every SM the ``_C.so`` build is configured for.  It compiles no CUDA
#: (``OASR_SOURCES`` is pure C++), so a wide list costs nothing and one image
#: serves the whole sweep — which is the point: an image per arch would mean an
#: image build per arch, and the kernels that actually differ are JIT-compiled
#: on the target anyway.
CUDA_ARCHITECTURES = os.environ.get("OASR_MODAL_CUDA_ARCH", "80;86;89;90;100;120")

# cu128: the first index with sm_100 (B200) and sm_120 wheels.
TORCH_INDEX = os.environ.get("OASR_MODAL_TORCH_INDEX", "https://download.pytorch.org/whl/cu128")


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
jit_vol = modal.Volume.from_name("oasr-ci-jit-cache", create_if_missing=True)

image = (
    # -devel, not -runtime: the JIT shells out to nvcc at run time.
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
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
    .pip_install("torch", "torchaudio", "torchcodec", index_url=TORCH_INDEX)
    .pip_install(
        # pyproject [project.dependencies] + the optional extras whose absence
        # would turn real coverage into importorskips.
        "numpy",
        "apache-tvm-ffi>=0.1.0",
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
        f"cd {REPO_REMOTE} && CUDA_ARCHITECTURES='{CUDA_ARCHITECTURES}'"
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


def _jit_dir() -> str:
    """Per-architecture JIT cache prefix.

    The cache *key* already separates architectures — ``_default_cuda_cflags``
    puts ``-gencode=...sm_XX`` into the flags ``JitSpec._content_hash`` hashes —
    so this is not a correctness fix.  It is about the Volume: five suites on
    five GPUs all committing under one prefix is five concurrent writers to the
    same paths, and a per-arch prefix makes each sweep's cache its own.
    """
    try:
        import torch

        major, minor = torch.cuda.get_device_capability()
        return f"{JIT_MOUNT}/sm{major}{minor}"
    except Exception:
        return f"{JIT_MOUNT}/unknown"


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    # Three hours, not one.  The kernels family JIT-compiles every GEMM, BMM,
    # Conv2D and attention module on first *call*, and on an architecture whose
    # cache prefix is empty that is the run.  Measured: sm_100 was cancelled at
    # 38 % against the old 5400 s, which reads exactly like a hang and is not
    # one.  The cost is paid once per architecture — the JIT Volume carries it
    # forward — so the ceiling has to clear a cold cache or the first run on any
    # new accelerator reports a timeout instead of a result.
    timeout=10800,
    volumes={ASSETS_MOUNT: assets_vol, JIT_MOUNT: jit_vol},
)
def run_suite(name: str, strict: bool = True, extra: str = "", assets: bool = True) -> str:
    """Run one family from ci/gpu_suites.py.  Raises on failure.

    ``assets=False`` runs the suite with the Volume unused: no asset env var is
    set, so every checkpoint- and audio-gated test skips and what is left is the
    kernel / layer / plumbing coverage that needs only a GPU.  ``strict`` stays
    meaningful there — the allow-list simply widens to every declared asset, so
    the run still *names* what it did not cover instead of turning the gate off.
    """
    env = {**os.environ, "OASR_JIT_DIR": _jit_dir()}
    if assets:
        env.update(_asset_env())

    subprocess.run(["nvidia-smi"], check=False)
    print(
        f"[modal] suite={name} strict={strict} assets={assets} jit={env['OASR_JIT_DIR']}",
        flush=True,
    )

    cmd = [
        "python",
        "-m",
        "pytest",
        *paths_for(name),
        "-m",
        DEFAULT_MARKER_EXPR,
        "-q",
        "-rs",
        "-rE",
    ]
    if strict:
        cmd.append("--strict-assets")
        allowed = ALLOW_MISSING if assets else tuple(test_assets.ASSETS)
        for missing in allowed:
            cmd += ["--allow-missing-asset", missing]
    if extra:
        cmd += extra.split()

    try:
        subprocess.run(cmd, cwd=REPO_REMOTE, env=env, check=True)
    finally:
        # Persist whatever the JIT compiled so the next run starts warm.
        jit_vol.commit()
    return name


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


@app.local_entrypoint()
def main(
    gpus: str = "",
    suites: str = "",
    strict: bool = True,
    extra: str = "",
    assets: bool = True,
):
    """Run the families on one or more accelerators.

    ``--gpus`` is a comma-separated list of Modal accelerators, ``all`` for one
    per distinct architecture (``ARCH_SWEEP``), or empty for the default.
    ``--suites`` is a comma-separated list of names from ci/gpu_suites.py; empty
    means all of them.  Every (gpu, suite) pair runs concurrently.

    ``--assets`` reads the checkpoints and audio out of the ``oasr-ci-assets``
    Volume.  On by default now that ``fetch_assets`` fills it without anybody
    uploading anything; pass ``--no-assets`` for the GPU-only coverage that
    needs no external input.
    """
    names = [s.strip() for s in suites.split(",") if s.strip()] or list(SUITES)
    unknown = [n for n in names if n not in SUITES]
    if unknown:
        raise SystemExit(f"unknown suite(s): {unknown}; known: {list(SUITES)}")

    if gpus.strip().lower() == "all":
        targets = list(ARCH_SWEEP)
    else:
        targets = [g.strip() for g in gpus.split(",") if g.strip()] or [DEFAULT_GPU]
    unknown_gpu = [g for g in targets if g not in GPU_ARCH]
    if unknown_gpu:
        print(f"[modal] note: {unknown_gpu} not in GPU_ARCH; passing through to Modal as-is")

    scope = "with checkpoints" if assets else "no checkpoints (kernels/plumbing only)"
    shown = ", ".join("{} ({})".format(g, GPU_ARCH.get(g, "?")) for g in targets)
    print(
        f"[modal] {len(targets)} accelerator(s) x {len(names)} suite(s), {scope}\n"
        f"        gpus:   {shown}\n"
        f"        suites: {', '.join(names)}"
    )

    # `spawn`, not `starmap`.  `.with_options(gpu=...)` gives each accelerator
    # its own container pool, but `starmap` returns a *lazy* generator: building
    # one per GPU and consuming them in order submits nothing for the second
    # architecture until the first has finished, so a "parallel" sweep runs
    # strictly serially and takes five times as long.  `spawn` submits on the
    # call and hands back a FunctionCall to collect later, which is what makes
    # every (gpu, suite) pair actually start at once.
    calls = [
        (gpu, name, run_suite.with_options(gpu=gpu).spawn(name, strict, extra, assets))
        for gpu in targets
        for name in names
    ]
    print(f"[modal] spawned {len(calls)} run(s); collecting")

    results: dict[str, dict[str, object]] = {gpu: {} for gpu in targets}
    failed: list[tuple[str, str, Exception]] = []
    for gpu, name, call in calls:
        try:
            call.get()
            results[gpu][name] = None
        except Exception as exc:  # one suite failing must not hide the other 29
            results[gpu][name] = exc
            failed.append((gpu, name, exc))

    print()
    for gpu in targets:
        print(f"  {gpu} ({GPU_ARCH.get(gpu, '?')})")
        for name in names:
            exc = results[gpu][name]
            print(f"    {'FAIL' if exc is not None else 'ok  '}  {name}")

    if failed:
        for gpu, name, exc in failed:
            print(f"\n--- {gpu} / {name} ---\n{exc}", file=sys.stderr)
        raise SystemExit(f"{len(failed)}/{len(targets) * len(names)} suite run(s) failed")


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
