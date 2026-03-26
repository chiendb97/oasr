# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from FlashInfer's flashinfer/jit/cubin_loader.py
# (https://github.com/flashinfer-ai/flashinfer)
#
# Concurrent-safe caching utilities for JIT-compiled shared libraries.
# Provides file locking, atomic writes, and SHA256 verification.

import hashlib
import logging
import os
import uuid
from pathlib import Path
from typing import Optional

import filelock

logger = logging.getLogger("oasr.jit")


def write_if_different(path, content: str) -> bool:
    """Write *content* to *path* only if it differs from the current contents.

    Returns True if the file was (re-)written, False if it was already
    up-to-date.  This avoids unnecessary recompilation when the generated
    source has not changed.
    """
    path = Path(path)
    if path.exists():
        existing = path.read_text()
        if existing == content:
            return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return True


def sha256_file(file_path: str) -> str:
    """Compute the SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_so(so_path: str, expected_sha256: Optional[str] = None) -> bool:
    """Check that a cached ``.so`` file exists and optionally verify its hash.

    Parameters
    ----------
    so_path : str
        Path to the shared library.
    expected_sha256 : str, optional
        If provided, the file's SHA256 is checked against this value.

    Returns
    -------
    bool
        True if the file exists (and hash matches, if checked).
    """
    if not os.path.exists(so_path):
        return False
    if expected_sha256 is not None:
        actual = sha256_file(so_path)
        if actual != expected_sha256:
            logger.warning(
                "SHA256 mismatch for %s (expected %s, got %s)",
                so_path,
                expected_sha256,
                actual,
            )
            return False
    return True


def atomic_write_bytes(destination: str, data: bytes) -> None:
    """Write *data* to *destination* atomically via temp file + rename.

    If the process is interrupted, either the old file remains or the new
    file is completely written — never a partial write.
    """
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    temp_path = f"{destination}.{uuid.uuid4().hex}.tmp"
    try:
        with open(temp_path, "wb") as f:
            f.write(data)
        os.replace(temp_path, destination)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def locked_compile(
    lib_path: str,
    compile_fn,
    lock_timeout: int = 600,
) -> str:
    """Compile a shared library with file-lock protection.

    If another process is already compiling the same library, this
    blocks until the lock is released and then returns the existing
    artifact.

    Parameters
    ----------
    lib_path : str
        Target path for the compiled ``.so``.
    compile_fn : callable
        ``compile_fn(lib_path)`` performs the actual compilation.
        Called only if the file does not already exist.
    lock_timeout : int
        Maximum seconds to wait for the lock (default 600 = 10 min).

    Returns
    -------
    str
        The *lib_path* on success.

    Raises
    ------
    filelock.Timeout
        If the lock cannot be acquired within *lock_timeout*.
    """
    lock_path = f"{lib_path}.lock"
    os.makedirs(os.path.dirname(lib_path), exist_ok=True)
    lock = filelock.FileLock(lock_path, timeout=lock_timeout)

    with lock:
        logger.debug("Acquired lock for %s", lib_path)
        if os.path.exists(lib_path):
            logger.debug("Library already exists (compiled by another process): %s", lib_path)
            return lib_path
        compile_fn(lib_path)

    return lib_path
