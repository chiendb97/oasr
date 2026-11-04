# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from FlashInfer's flashinfer/jit/cubin_loader.py
# (https://github.com/flashinfer-ai/flashinfer)
#
# Concurrent-safe caching utilities for JIT-compiled shared libraries.
# Provides file locking, atomic writes, and SHA256 verification.

import hashlib
import json
import logging
import os
import socket
import struct
import time
import uuid
from pathlib import Path
from typing import Optional

import filelock

from .errors import JitBuildError, JitBuildLockTimeout

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


#: ELF64 header, section header and program header, after ``e_ident``.
_ELF64_EHDR = "HHIQQQIHHHHHH"
_ELF64_SHDR = "IIQQQQIIQQ"
_ELF64_PHDR = "IIQQQQQQ"
_SHT_NOBITS = 8


def is_complete_shared_object(so_path) -> bool:
    """Whether *so_path* holds every byte its ELF headers say it does.

    Every section and every segment the headers describe must lie inside the
    file, so one cut short anywhere -- a link killed mid-write -- fails.  This
    has to be checked **before** loading: ``dlopen`` of a truncated library does
    not raise, it kills the process with ``SIGBUS`` the moment the loader
    touches a page past the end of the file, so a load failure cannot be caught
    and repaired after the fact.  Two small reads; cheap enough for every load.

    A missing, empty, or non-ELF file is not complete.  An ELF class other than
    64-bit, which nothing here builds, is given the benefit of the doubt, so an
    unfamiliar layout cannot turn into a rebuild on every load.  What this cannot
    see is a full-length file whose interior was never written, which a linker
    that pre-sizes its output could leave; :func:`locked_compile` never exposes
    an unfinished link under the final name, so only a cache written before it
    did could hold one.
    """
    try:
        with open(so_path, "rb") as f:
            size = os.fstat(f.fileno()).st_size
            ident = f.read(16)
            if len(ident) < 16 or ident[:4] != b"\x7fELF":
                return False
            if ident[4] != 2:  # not ELFCLASS64
                return True
            end = "<" if ident[5] == 1 else ">"
            raw = f.read(struct.calcsize(end + _ELF64_EHDR))
            if len(raw) < struct.calcsize(end + _ELF64_EHDR):
                return False
            ehdr = struct.unpack(end + _ELF64_EHDR, raw)
            phoff, shoff = ehdr[4], ehdr[5]
            phentsize, phnum, shentsize, shnum = ehdr[8], ehdr[9], ehdr[10], ehdr[11]
            extents = [(shoff, shentsize * shnum), (phoff, phentsize * phnum)]
            if any(off + n > size for off, n in extents):
                return False
            f.seek(shoff)
            table = f.read(shentsize * shnum)
            for i in range(shnum):
                sh = struct.unpack_from(end + _ELF64_SHDR, table, i * shentsize)
                if sh[1] != _SHT_NOBITS:  # sh_type; NOBITS sections take no file space
                    extents.append((sh[4], sh[5]))  # sh_offset, sh_size
            f.seek(phoff)
            table = f.read(phentsize * phnum)
            for i in range(phnum):
                ph = struct.unpack_from(end + _ELF64_PHDR, table, i * phentsize)
                extents.append((ph[2], ph[5]))  # p_offset, p_filesz
    except (OSError, struct.error):
        return False
    return all(off + n <= size for off, n in extents)


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


#: How often a process waiting on another's build says so, in seconds.
WAIT_REPORT_S = 30.0

#: How much of a failed build's message the failure record keeps.
_FAILURE_TAIL = 4000


def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _holder(lock_path: str) -> str:
    """Who holds *lock_path*, as `` (pid P on H, for N s)``, or ``""`` if unknown."""
    owner = _read_json(f"{lock_path}.owner")
    if owner is None:
        return ""
    held = time.time() - float(owner.get("since", time.time()))
    return f" (pid {owner.get('pid')} on {owner.get('host')}, for {held:.0f} s)"


def _acquire(lock: filelock.FileLock, lib_path: str, timeout: float) -> None:
    """Take *lock*, saying so every :data:`WAIT_REPORT_S` while another process builds.

    A bare ``filelock.Timeout`` after ten minutes of silence was what a caller
    saw before: no word of what it waited for or who held it.
    """
    start = time.monotonic()
    deadline = start + timeout
    while True:
        try:
            lock.acquire(timeout=max(0.0, min(WAIT_REPORT_S, deadline - time.monotonic())))
            return
        except filelock.Timeout:
            waited = time.monotonic() - start
            holder = _holder(lock.lock_file)
            if time.monotonic() >= deadline:
                raise JitBuildLockTimeout(
                    f"gave up after {waited:.0f} s waiting for another process to finish "
                    f"building {lib_path}{holder}. A build that long is still running, or "
                    f"the lock {lock.lock_file} is held by a process that is not building."
                ) from None
            logger.warning(
                "waiting %.0f s for another process to finish building %s%s",
                waited,
                lib_path,
                holder,
            )


def locked_compile(
    lib_path: str,
    compile_fn,
    lock_timeout: float = 600,
) -> str:
    """Compile a shared library with file-lock protection, and publish it atomically.

    If another process is already compiling the same library, this
    blocks until the lock is released and then returns the existing
    artifact -- or, if that build failed, raises its failure rather than
    repeating it.

    The library is built under a staging name and moved into place with
    ``os.replace`` once it is complete, so *lib_path* never names a partly
    written file.  Linking straight to it did, twice over (audit #23): a process
    that found it mid-link loaded half a library, and a link killed mid-write
    left a truncated one that every later process loaded.  Both died of
    ``SIGBUS`` inside ``dlopen``, not with an exception.  With the publish
    atomic, existence is proof of completeness, which is what lets
    :meth:`~oasr.jit.core.JitSpec.build_and_load` check it without the lock.

    **A failed build is shared with the processes that waited for it.**  The
    builder records its failure (``<lib>.failed``: an attempt id and the
    message) before it releases the lock, and a waiter that then finds a record
    it had not seen before it started waiting raises :class:`JitBuildError`
    with that message instead of building again.  Without it, *N* processes
    starting together paid for *N* identical failed builds one after another,
    and the later ones ran out of lock wait doing it.  A process that arrives
    after the attempt -- the record was already there when it started waiting
    -- builds again, so a cause fixed in between (disk space, a toolchain) is
    picked up, and the record is removed when a build succeeds.  The one window
    left: a waiter that reads the record in the instant between the builder
    writing it and releasing the lock repeats the build once.

    Parameters
    ----------
    lib_path : str
        Target path for the compiled ``.so``.
    compile_fn : callable
        ``compile_fn(path)`` compiles the library to exactly ``path`` -- the
        staging name, not *lib_path*.  Called only if *lib_path* does not
        already hold a complete library.
    lock_timeout : float
        Maximum seconds to wait for the lock (default 600 = 10 min).  A waiter
        logs every :data:`WAIT_REPORT_S` and names the holder when it gives up.

    Returns
    -------
    str
        The *lib_path* on success.

    Raises
    ------
    JitBuildLockTimeout
        If the lock cannot be acquired within *lock_timeout*.
    JitBuildError
        If the build this process waited for failed in the process holding the
        lock.
    Exception
        Whatever *compile_fn* raised, when this process's own build failed.
    """
    lock_path = f"{lib_path}.lock"
    failure_path = f"{lib_path}.failed"
    os.makedirs(os.path.dirname(lib_path), exist_ok=True)
    seen = _read_json(failure_path)
    lock = filelock.FileLock(lock_path)
    _acquire(lock, lib_path, lock_timeout)
    try:
        logger.debug("Acquired lock for %s", lib_path)
        if is_complete_shared_object(lib_path):
            logger.debug("Library already exists (compiled by another process): %s", lib_path)
            return lib_path
        failure = _read_json(failure_path)
        if failure is not None and failure.get("attempt") != (seen or {}).get("attempt"):
            raise JitBuildError(
                f"the build of {lib_path} that this process waited for failed in pid "
                f"{failure.get('pid')} on {failure.get('host')}; not repeating it:\n"
                f"{failure.get('error', '')}"
            )
        atomic_write_bytes(
            f"{lock_path}.owner",
            json.dumps(
                {"pid": os.getpid(), "host": socket.gethostname(), "since": time.time()}
            ).encode(),
        )
        if os.path.exists(lib_path):
            # Only a cache written before the publish was atomic can hold one.
            logger.warning("%s is truncated (a link was killed mid-write); rebuilding", lib_path)
        staging = f"{lib_path}.partial"
        # Left by a link killed mid-write.  Removed rather than trusted to the
        # build tool's own bookkeeping, which would otherwise be all that stood
        # between it and being called up to date.
        if os.path.exists(staging):
            os.remove(staging)
        try:
            compile_fn(staging)
        except Exception as exc:
            atomic_write_bytes(
                failure_path,
                json.dumps(
                    {
                        "attempt": uuid.uuid4().hex,
                        "pid": os.getpid(),
                        "host": socket.gethostname(),
                        "time": time.time(),
                        "error": str(exc)[-_FAILURE_TAIL:],
                    }
                ).encode(),
            )
            raise
        # On disk before the final name points at it, so a crash right after the
        # rename cannot leave the name on an empty file.
        fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(staging, lib_path)
        if os.path.exists(failure_path):
            os.remove(failure_path)
    finally:
        try:
            os.remove(f"{lock_path}.owner")
        except OSError:
            pass
        lock.release()

    return lib_path
