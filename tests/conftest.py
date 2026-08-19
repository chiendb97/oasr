#!/usr/bin/env python3
"""
Pytest configuration and shared fixtures for OASR tests.

Two things beyond the usual fixtures live here, both aimed at making a green
run mean something (see ``tests/assets.py`` and ``docs/ci.md``):

``--strict-assets``
    Turn "skipped because the checkpoint was not on this box" into a failure.
    Assets a runner genuinely cannot have are named with
    ``--allow-missing-asset NAME`` so the gap is visible in the workflow file.

``--min-passed N``
    Fail the session if fewer than *N* tests passed.  A CUDA guard added to the
    wrong scope, or an import that quietly turns a module into skips, otherwise
    shrinks coverage without turning anything red.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, "python")

# `tests/` has no __init__.py, so make the sibling helper importable by name
# both here and from the test modules themselves.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import assets  # noqa: E402  — needs the sys.path line above


def pytest_addoption(parser):
    """Register custom command-line options."""
    parser.addoption(
        "--ckpt-dir",
        action="store",
        default="",
        help="Path to WeNet checkpoint dir (overrides $CKPT_DIR)",
    )
    parser.addoption(
        "--audio-path",
        action="store",
        default="",
        help="Path to a test audio file (overrides $AUDIO_PATH)",
    )
    parser.addoption(
        "--lang-dir",
        action="store",
        default="",
        help="Path to a pre-built language directory (overrides $LANG_DIR)",
    )
    parser.addoption(
        "--wav-dir",
        action="store",
        default="",
        help="Directory of test WAV files (overrides $WAV_DIR)",
    )
    parser.addoption(
        "--strict-assets",
        action="store_true",
        default=os.environ.get("OASR_TEST_STRICT_ASSETS", "") == "1",
        help=(
            "Fail instead of skipping when a declared external asset "
            "(checkpoint / audio / reference graph) is missing"
        ),
    )
    parser.addoption(
        "--allow-missing-asset",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "Under --strict-assets, permit this asset to be absent "
            "(repeatable; NAME is the env var, e.g. WENET_REF_DIR)"
        ),
    )
    parser.addoption(
        "--min-passed",
        action="store",
        type=int,
        default=0,
        help="Fail the session if fewer than N tests passed (coverage floor)",
    )


# CLI option -> the env var tests/assets.py reads.  Writing the option back into
# the environment keeps one source of truth: everything resolves through
# ``assets.resolve`` regardless of how the path was supplied.
_OPTION_ENV = {
    "--ckpt-dir": "CKPT_DIR",
    "--audio-path": "AUDIO_PATH",
    "--lang-dir": "LANG_DIR",
    "--wav-dir": "WAV_DIR",
}


def pytest_configure(config):
    """Register custom markers and wire up the asset gate."""
    config.addinivalue_line("markers", "cuda: mark test as requiring CUDA")
    config.addinivalue_line("markers", "slow: mark test as slow-running")
    config.addinivalue_line(
        "markers",
        "requires_assets(*names): skip (or, under --strict-assets, fail) unless "
        "every named asset from tests/assets.py is present",
    )

    for opt, env in _OPTION_ENV.items():
        value = config.getoption(opt)
        if value:
            os.environ[env] = value

    assets.configure(
        strict=config.getoption("--strict-assets"),
        allow_missing=config.getoption("--allow-missing-asset"),
    )


def pytest_report_header(config):
    """Say up front what the run can and cannot check."""
    del config
    missing = [name for name in assets.ASSETS if not assets.present(name)]
    mode = "strict" if assets.STATE.strict else "skip"
    head = (
        f"oasr assets: {len(assets.ASSETS) - len(missing)} present, "
        f"{len(missing)} missing ({mode} mode)"
    )
    if not missing:
        return [head]
    return [head, "  missing: " + ", ".join(sorted(missing))]


def pytest_runtest_setup(item):
    """Honour ``@pytest.mark.cuda`` and ``@pytest.mark.requires_assets(...)``."""
    # `cuda` used to be registered and then ignored: gating came from whether a
    # test happened to request the `device` fixture, so a test inside a
    # `@pytest.mark.cuda` class that did not take `device` ran anyway.  One such
    # test (`TestCtcStateCacheManager::test_get_unallocated_raises`) reached
    # `GpuStreamingDecoder`, which JIT-compiles, and died on a runner with no
    # nvcc.  The marker now means what it says.
    if not torch.cuda.is_available() and next(item.iter_markers(name="cuda"), None) is not None:
        pytest.skip("CUDA not available")

    for mark in item.iter_markers(name="requires_assets"):
        if mark.args:
            assets.require(*mark.args)


#: Passed-test tally for the ``--min-passed`` floor.  Counted here rather than
#: read off the terminal reporter so the floor also works under ``-p no:terminal``.
_PASSED = {"n": 0}


def pytest_runtest_logreport(report):
    if report.when == "call" and report.passed:
        _PASSED["n"] += 1


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print the asset table and the coverage-floor line."""
    del exitstatus
    write = terminalreporter.write_line
    write("")
    write("external assets:")
    for line in assets.report_lines():
        write(line)
    if assets.STATE.gated and not assets.STATE.strict:
        total = sum(assets.STATE.gated.values())
        write(
            f"  -> {total} test(s) skipped for missing assets. "
            f"A green run does not cover them; use --strict-assets to make this fatal."
        )

    floor = config.getoption("--min-passed")
    if floor:
        ok = _PASSED["n"] >= floor
        write(
            f"coverage floor: {_PASSED['n']} passed, minimum {floor}"
            + ("" if ok else "  <-- BELOW FLOOR"),
            red=not ok,
        )


def pytest_sessionfinish(session, exitstatus):
    """Fail the session when fewer tests passed than ``--min-passed`` demands."""
    floor = session.config.getoption("--min-passed")
    if floor and _PASSED["n"] < floor and exitstatus == pytest.ExitCode.OK:
        print(
            f"\nERROR: only {_PASSED['n']} tests passed, below the --min-passed "
            f"floor of {floor}.  Coverage shrank — find what started skipping "
            f"rather than lowering the floor.",
            file=sys.stderr,
        )
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


@pytest.fixture(scope="session")
def ckpt_dir():
    """WeNet checkpoint dir; skips (or fails under --strict-assets) if absent."""
    return assets.require("CKPT_DIR")


@pytest.fixture(scope="session")
def audio_path():
    """A single test .wav; skips (or fails under --strict-assets) if absent."""
    return assets.require("AUDIO_PATH")


@pytest.fixture(scope="session")
def lang_dir():
    """Pre-built lang dir; skips (or fails under --strict-assets) if absent."""
    return assets.require("LANG_DIR")


@pytest.fixture(scope="session")
def wav_dir():
    """Directory of test .wav files; skips (or fails under --strict-assets)."""
    return assets.require("WAV_DIR")


@pytest.fixture(scope="session")
def device():
    """Return CUDA device if available, otherwise skip."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


@pytest.fixture(scope="session")
def oasr_module():
    """Import and return the oasr module."""
    try:
        import oasr

        return oasr
    except ImportError as e:
        pytest.skip(f"oasr module not available: {e}")


@pytest.fixture(params=[torch.float32, torch.float16])
def dtype(request):
    """Parametrize tests with different dtypes."""
    return request.param


@pytest.fixture(params=[torch.float32, torch.float16, torch.bfloat16])
def dtype_all(request):
    """Parametrize tests with all supported dtypes."""
    if request.param == torch.bfloat16:
        if not torch.cuda.is_bf16_supported():
            pytest.skip("BF16 not supported on this device")
    return request.param


# Common test shapes
@pytest.fixture(
    params=[
        (2, 128, 256),  # Small
        (4, 256, 512),  # Medium
        (8, 512, 768),  # Large
    ]
)
def batch_seq_hidden(request):
    """Common (batch_size, seq_len, hidden_size) shapes."""
    return request.param


def get_rtol_atol(dtype):
    """Get relative and absolute tolerance based on dtype."""
    if dtype == torch.float32:
        return 1e-4, 1e-4
    elif dtype == torch.float16:
        return 1e-2, 1e-2
    elif dtype == torch.bfloat16:
        return 1e-2, 1e-2
    else:
        return 1e-3, 1e-3
