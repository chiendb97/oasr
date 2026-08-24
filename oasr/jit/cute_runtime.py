# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Per-call runtime glue for the CuTeDSL kernel call sites.

A compiled CuTeDSL callable takes the target stream as a ``CUstream`` handle, so
every call has to produce one.  The obvious spelling is expensive:

    torch.cuda.current_stream()                      3.86 us
    torch.cuda.current_stream().cuda_stream          3.90 us
    cuda_driver.CUstream(...) around it              4.06 us
    torch._C._cuda_getCurrentRawStream(dev)          0.09 us

``torch.cuda.current_stream()`` builds a Python ``Stream`` object — a device
query, a ``_CudaStreamBase`` construction and the ``ExternalStream`` wrapper —
to hand back one integer that the private accessor returns directly.  Against a
6 us recurrent step that was two thirds of the launch; against a 26 us FMHA call
it was 15%.

Caching the ``CUstream`` wrapper against the raw pointer removes the rest: the
wrapper is a ctypes-style struct whose construction costs about as much as the
query it wraps, and a process has a handful of distinct streams, not a stream
per call.

The dict is keyed on the raw pointer rather than on the device, so a side stream
(the engine's H2D staging stream, a capture stream) gets its own entry and a
capture never inherits the default stream's handle.  It is unbounded in
principle; in practice a process creates streams at setup, and an entry is two
machine words.

**Read the handle inside a capture, never outside it.**  A handle cached before
``torch.cuda.graph(...)`` is entered still points at the non-capturing stream,
so every launch goes there, the graph records nothing, and replay measures
nothing — which once produced 0.02 us/step and 98,883 TFLOP/s.
"""

from __future__ import annotations

from typing import Any, Dict

#: raw stream pointer -> ``cuda.bindings.driver.CUstream``
_STREAMS: Dict[int, Any] = {}

try:  # the private accessor has been present since torch 2.0
    import torch as _torch

    _RAW_STREAM = getattr(_torch._C, "_cuda_getCurrentRawStream", None)
except Exception:  # pragma: no cover - torch is a hard dependency in practice
    _RAW_STREAM = None


def current_stream():
    """The active CUDA stream as a cached ``CUstream`` handle."""
    import cuda.bindings.driver as cuda_driver
    import torch

    if _RAW_STREAM is not None:
        raw = _RAW_STREAM(torch.cuda.current_device())
    else:  # pragma: no cover - private symbol gone; correct but ~45x slower
        raw = torch.cuda.current_stream().cuda_stream
    handle = _STREAMS.get(raw)
    if handle is None:
        handle = cuda_driver.CUstream(raw)
        _STREAMS[raw] = handle
    return handle
