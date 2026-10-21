# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""What a served engine holds: its quantization, its weights, its allocator peak.

The three workload columns a quantization A/B needs beside the speed and error
columns -- ``quantization``, ``weight_mib`` and ``peak_alloc_mib``.  Shared by
the harnesses that build an ``ASREngine`` in process (engine, accuracy) and so
can ask it directly; the service harness talks to a server and leaves them
empty.

``quantization`` records what was *served*, from ``ASREngine.quantization_info``,
never the flag: a checkpoint written by ``oasr quantize`` is served quantized
with no flag at all, and a row labelled by the flag would call it full precision.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Union

MIB = float(1 << 20)

#: ``--quantization`` spellings meaning "no online quantization".
_NONE = ("", "none", "off")


def parse_quantization(text: Optional[str]) -> Union[None, str, Dict[str, Any]]:
    """``--quantization`` as ``EngineConfig(quantization=...)`` takes it.

    ``None`` / ``""`` / ``"none"`` / ``"off"`` mean no online quantization; a
    JSON object (``'{"scheme": "int8_w8a8", "ignore": "a.*,b.*"}'``) becomes a
    mapping; anything else is a scheme name.  Neither is validated here --
    ``QuantizationConfig.coerce`` does that at engine construction and names the
    presets, so this module needs no copy of the scheme table.
    """
    if text is None:
        return None
    value = text.strip()
    if value.lower() in _NONE:
        return None
    if value.startswith("{"):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"--quantization {text!r} is not valid JSON: {exc}") from None
        if not isinstance(parsed, dict):
            raise ValueError(f"--quantization {text!r} must be a JSON object")
        return parsed
    return value


def quantization_label(value: Union[None, str, Dict[str, Any]]) -> str:
    """A requested quantization as a short label for a sweep cell; ``""`` for none."""
    if value is None:
        return ""
    if isinstance(value, dict):
        label = str(value.get("scheme") or "checkpoint")
        if value.get("ignore"):
            label += f"[ignore={value['ignore']}]"
        return label
    return str(value)


def served_quantization(engine: Any) -> str:
    """The scheme(s) *engine* serves, ``+``-joined, or ``"none"``.

    ``""`` when the engine cannot say (an engine predating
    ``quantization_info``): unknown is not the same as full precision.
    """
    describe = getattr(engine, "quantization_info", None)
    if describe is None:
        return ""
    info = describe()
    if not info:
        return "none"
    return "+".join(str(s) for s in info.get("schemes") or ()) or "unknown"


def model_weight_mib(model: Any) -> Optional[float]:
    """Bytes held by *model*'s weights, in MiB: its parameters plus each quantized
    layer's storage.

    Not every buffer: a positional-encoding table is a buffer too, and a
    streaming runtime grows it to its geometry (``bound_offsets``), so counting
    it made the column differ between chunk sizes of one checkpoint.  A quantized
    layer's weight scales and runtime smoother are buffers that *are* weights;
    its method names them (``preserved_tensors``).
    """
    if model is None:
        return None
    tensors = list(model.parameters())
    for module in model.modules():
        method = getattr(module, "quant_method", None)
        if method is not None:
            for name in method.preserved_tensors(module):
                slot = getattr(module, name, None)
                if slot is not None:
                    tensors.append(slot)
    seen = set()
    total = 0
    for tensor in tensors:
        if id(tensor) in seen:
            continue
        seen.add(id(tensor))
        total += tensor.numel() * tensor.element_size()
    return total / MIB


def reset_peak_memory() -> None:
    """Start a fresh high-water mark (a no-op without CUDA)."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def peak_alloc_mib() -> Optional[float]:
    """The CUDA allocator's high-water mark since :func:`reset_peak_memory`, in MiB.

    Allocated, not reserved: the caching allocator's reserve depends on its
    fragmentation history, so it would differ between two identical runs.
    """
    import torch

    if not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / MIB
