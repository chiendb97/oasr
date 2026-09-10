# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""What the per-architecture test modules share.

The seven architecture files follow one template -- parity against a
reference, streaming equals offline, the converter detects the directory, the
load report is clean, the native round trip preserves the weights -- and the
last of those was written out five times, with the same eight-line tail:

    arch1, b1 = load_checkpoint_bundle(SRC)
    m1, _, _ = instantiate_from_bundle(arch1, b1)
    sd1, sd2 = m1.state_dict(), m2.state_dict()
    assert set(sd1) == set(sd2)
    for k in sd1:
        assert torch.equal(sd1[k], sd2[k]), k

That tail is the *shared* claim; what differs per architecture is which
metadata survives -- the tokenizer kind, the feature spec, the prompt
sequence. :func:`assert_native_weights_round_trip` owns the shared half so
each file is left with only its own.
"""

from __future__ import annotations

from typing import Any, Tuple

import pytest
import torch


def assert_native_weights_round_trip(src: str, tmp_path, expect_arch: str) -> Tuple[Any, Any]:
    """Convert ``src`` to the native format and prove no weight changed.

    Returns ``(bundle, model)`` for the converted copy, so the caller can go
    on to assert the metadata only *it* cares about.
    """
    pytest.importorskip("safetensors")
    from oasr.checkpoints.convert import convert_to_native
    from oasr.models.registry import instantiate_from_bundle, load_checkpoint_bundle

    out = tmp_path / "native"
    convert_to_native(src, str(out))
    arch, bundle = load_checkpoint_bundle(out)
    assert (arch, bundle.source_format) == (expect_arch, "native")
    converted, _cfg, _report = instantiate_from_bundle(arch, bundle)

    src_arch, src_bundle = load_checkpoint_bundle(src)
    original, _c, _r = instantiate_from_bundle(src_arch, src_bundle)

    before, after = original.state_dict(), converted.state_dict()
    assert set(before) == set(after), "the native round trip changed the tensor set"
    for key in before:
        assert torch.equal(before[key], after[key]), f"{key} changed on the native round trip"
    return bundle, converted
