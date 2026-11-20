# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Ahead-of-traffic compilation of the CuTeDSL attention variants.

A variant compiles on first use (~1.3 s), on whichever live request first needs
it.  ``precompile_fmha_variants`` compiles, at engine construction, every tile
the tuning table can route a signature to, so traffic finds them compiled.
"""

from __future__ import annotations

import pytest
import torch

from oasr.jit import attention as jit_attention
from oasr.jit.attention_select import AttnChoice

_SIG = jit_attention._Signature(
    head_dim=64,
    dtype_str="float16",
    num_heads=4,
    num_kv_heads=4,
    has_bias=True,
    paged=False,
    block_size=0,
    causal=False,
    has_seqstart=False,
)


def _key(**over):
    from oasr.tune.database import sig_key

    fields = {
        "op": "fmha",
        "dt": "half",
        "D": 64,
        "H": 4,
        "Hkv": 4,
        "mask": "none",
        "bias": 1,
        "ba": 0,
        "kv": "dense",
        "ss": 0,
        "q": "L",
        "B": 8,
    }
    fields.update(over)
    return sig_key("fmha", **fields)


class TestTableTiles:
    def test_collects_every_cute_tile_of_the_signature(self, monkeypatch):
        table = {
            _key(): [(256, "a", AttnChoice("cute", 32, 32, 64)), (None, "b", AttnChoice("cxx"))],
            _key(B=32, ba=1): [(None, "c", AttnChoice("cute", 64, 16, 128))],
            # Another signature's tiles are not this one's.
            _key(H=8): [(None, "d", AttnChoice("cute", 16, 16, 32))],
        }
        monkeypatch.setattr("oasr.jit.attention_select._tier_views", lambda sm: [("system", table)])
        tiles = jit_attention._table_tiles(_SIG, 120)
        assert tiles == {(32, 32, 64, False), (64, 16, 128, True)}

    def test_no_table_means_no_extra_tiles(self, monkeypatch):
        monkeypatch.setattr("oasr.jit.attention_select._tier_views", lambda sm: [])
        assert jit_attention._table_tiles(_SIG, 120) == set()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
class TestPrecompile:
    def test_precompiled_variants_are_cache_hits_afterwards(self, monkeypatch):
        monkeypatch.setattr("oasr.jit.attention_select._tier_views", lambda sm: [])
        compiled, failed = jit_attention.precompile_fmha_variants([_SIG])
        assert failed == 0
        # The default tile at both bias alignments, unless an earlier test in
        # this process already compiled one of them.
        assert 0 <= compiled <= 2
        assert jit_attention.precompile_fmha_variants([_SIG]) == (0, 0)
        before = jit_attention._compiled_fmha.cache_info()
        for aligned in (False, True):
            jit_attention.get_compiled_fmha(
                head_dim=64,
                dtype_str="float16",
                num_heads=4,
                num_kv_heads=4,
                has_bias=True,
                paged=False,
                bias_aligned=aligned,
            )
        after = jit_attention._compiled_fmha.cache_info()
        assert after.misses == before.misses
        assert after.hits == before.hits + 2

    def test_a_spent_budget_compiles_nothing(self, monkeypatch):
        monkeypatch.setattr("oasr.jit.attention_select._tier_views", lambda sm: [])
        sig = _SIG._replace(num_heads=3, num_kv_heads=3)
        assert jit_attention.precompile_fmha_variants([sig], budget_s=-1.0) == (0, 0)
