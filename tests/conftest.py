#!/usr/bin/env python3
"""
Pytest configuration and shared fixtures for OASR tests.
"""

import os
import sys
import pytest
import torch

# Add python module to path
sys.path.insert(0, 'python')


def pytest_addoption(parser):
    """Register custom command-line options."""
    parser.addoption(
        "--ckpt-dir",
        action="store",
        default=os.environ.get("CKPT_DIR", ""),
        help="Path to WeNet checkpoint dir for load_wenet_checkpoint tests",
    )
    parser.addoption(
        "--audio-path",
        action="store",
        default=os.environ.get("AUDIO_PATH", ""),
        help="Path to a test audio file for decoder integration tests",
    )
    parser.addoption(
        "--lang-dir",
        action="store",
        default=os.environ.get("LANG_DIR", ""),
        help="Path to a pre-built language directory for WFST beam search tests",
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "cuda: mark test as requiring CUDA"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow-running"
    )


@pytest.fixture(scope="session")
def ckpt_dir(request):
    """Path to WeNet checkpoint dir (from --ckpt-dir or CKPT_DIR env)."""
    return request.config.getoption("--ckpt-dir", default="")


@pytest.fixture(scope="session")
def audio_path(request):
    """Path to a test audio file (from --audio-path or AUDIO_PATH env)."""
    return request.config.getoption("--audio-path", default="")


@pytest.fixture(scope="session")
def lang_dir(request):
    """Path to a pre-built language directory (from --lang-dir or LANG_DIR env)."""
    return request.config.getoption("--lang-dir", default="")


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
@pytest.fixture(params=[
    (2, 128, 256),   # Small
    (4, 256, 512),   # Medium
    (8, 512, 768),   # Large
])
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
