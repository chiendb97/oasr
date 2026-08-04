# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""VRAM-aware capacity sizing (architecture review H4).

``max_num_blocks`` and ``decode_kv_budget_gib`` were hardcoded numbers with no
VRAM introspection anywhere in the package, so the operator hand-computed the
pool from layers x heads x head_dim x dtype and either wasted memory or hit the
crash path.  This pins the derivation.

The arithmetic half runs on CPU — that is the point of keeping
``oasr/engine/memory.py`` free of side effects: the formula, the block-size
accounting and the failure messages are testable without a GPU.  The GPU half
checks the two things arithmetic cannot: that a *derived* pool actually
allocates, and that deriving it changes capacity and not transcripts.
"""

from __future__ import annotations

import glob
import os

import pytest
import torch

from oasr.cache.block_pool import BlockPool
from oasr.cache.types import CacheConfig
from oasr.engine.config import EngineConfig
from oasr.engine.memory import (
    MIN_ACTIVATION_RESERVE_BYTES,
    MIN_BLOCKS_PER_STREAM,
    UNMEASURED_ACTIVATION_FRACTION,
    MemoryProfile,
    bytes_per_kv_block,
    derive_decode_kv_budget,
    derive_pool_blocks,
)

GIB = 1024**3


def _profile(
    total_gib: float = 32.0,
    free_gib: float = 28.0,
    activation_gib: float = 2.0,
    utilization: float = 0.90,
    measured: bool = True,
) -> MemoryProfile:
    return MemoryProfile(
        total_bytes=int(total_gib * GIB),
        free_bytes=int(free_gib * GIB),
        activation_bytes=int(activation_gib * GIB),
        utilization=utilization,
        activation_measured=measured,
    )


# ---------------------------------------------------------------------------
# The profile arithmetic
# ---------------------------------------------------------------------------


class TestMemoryProfile:
    def test_resident_is_read_from_the_driver_not_assumed(self):
        """No separate "weights" term: what is resident is total - free.

        That is what makes the derivation correct when *another* process holds
        memory on the same card — the case a weights-only accounting misses.
        """
        p = _profile(total_gib=32.0, free_gib=20.0)
        assert p.resident_bytes == int(12 * GIB)

    def test_utilization_caps_the_whole_card_weights_included(self):
        p = _profile(total_gib=32.0, free_gib=28.0, utilization=0.5)
        assert p.cap_bytes == int(16 * GIB)
        # 16 GiB cap - 4 GiB resident
        assert p.budget_bytes == int(12 * GIB)

    def test_available_subtracts_the_activation_reserve(self):
        p = _profile(total_gib=32.0, free_gib=28.0, activation_gib=2.0)
        # cap 28.8, resident 4.0 -> budget 24.8; reserve 2.0 * 1.5 = 3.0
        assert p.activation_reserve_bytes == int(3 * GIB)
        assert p.available_bytes == p.budget_bytes - p.activation_reserve_bytes

    def test_reserve_has_a_floor(self):
        """A tiny model measures a tiny peak; the fixed costs above it do not
        shrink with the model."""
        p = _profile(activation_gib=0.001)
        assert p.activation_reserve_bytes == MIN_ACTIVATION_RESERVE_BYTES

    def test_unmeasured_probe_reserves_a_fraction_of_the_budget(self):
        p = _profile(activation_gib=0.0, measured=False)
        expected = int(p.budget_bytes * UNMEASURED_ACTIVATION_FRACTION)
        assert p.activation_reserve_bytes == expected

    def test_over_utilized_card_yields_no_budget_rather_than_a_negative_one(self):
        """Another process already past the cap must not produce a negative
        budget that silently wraps into a huge pool."""
        p = _profile(total_gib=32.0, free_gib=1.0, utilization=0.5)
        assert p.budget_bytes == 0
        assert p.available_bytes == 0

    def test_describe_carries_every_term(self):
        text = _profile().describe()
        for fragment in ("total=", "resident=", "cap=", "utilization=", "available="):
            assert fragment in text, f"{fragment!r} missing from {text!r}"


# ---------------------------------------------------------------------------
# Block accounting
# ---------------------------------------------------------------------------


class TestBytesPerBlock:
    """The formula must track :class:`BlockPool`, or every derivation is wrong
    by a constant factor."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
    def test_matches_a_real_pool_allocation(self, dtype):
        cfg = CacheConfig(
            num_layers=6,
            n_kv_head=4,
            head_dim=32,
            chunk_size=8,
            block_size_frames=8,
            max_num_blocks=16,
            max_batch_size=2,
            device=torch.device("cpu"),
            dtype=dtype,
        )
        pool = BlockPool(cfg)
        # Both pools together, per block.
        actual = (
            sum(t.numel() * t.element_size() for t in (pool._k_pool, pool._v_pool))
            // cfg.max_num_blocks
        )
        assert actual == bytes_per_kv_block(
            num_layers=cfg.num_layers,
            block_size_frames=cfg.block_size_frames,
            n_kv_head=cfg.n_kv_head,
            head_dim=cfg.head_dim,
            dtype=dtype,
        )

    def test_scales_with_every_geometry_term(self):
        base = {
            "num_layers": 12,
            "block_size_frames": 16,
            "n_kv_head": 4,
            "head_dim": 64,
            "dtype": torch.float16,
        }
        one = bytes_per_kv_block(**base)
        for term in ("num_layers", "block_size_frames", "n_kv_head", "head_dim"):
            doubled = dict(base)
            doubled[term] = base[term] * 2
            assert bytes_per_kv_block(**doubled) == 2 * one, term


# ---------------------------------------------------------------------------
# Pool derivation
# ---------------------------------------------------------------------------


class TestDerivePoolBlocks:
    PER_BLOCK = 192 * 1024  # 12 layers x 16 frames x 4 heads x 64 dim x fp16

    def test_fills_the_available_memory(self):
        p = _profile(total_gib=32.0, free_gib=28.0, activation_gib=2.0)
        sizing = derive_pool_blocks(p, self.PER_BLOCK, min_blocks=32, max_blocks=1 << 30)
        assert sizing.limited_by == "vram"
        assert sizing.blocks == p.available_bytes // self.PER_BLOCK
        assert sizing.pool_bytes <= p.available_bytes

    def test_a_bigger_card_derives_a_bigger_pool(self):
        """The whole point: one config across a 24 GB and an 80 GB card."""
        small = derive_pool_blocks(
            _profile(total_gib=24.0, free_gib=20.0),
            self.PER_BLOCK,
            min_blocks=32,
            max_blocks=1 << 30,
        )
        big = derive_pool_blocks(
            _profile(total_gib=80.0, free_gib=76.0),
            self.PER_BLOCK,
            min_blocks=32,
            max_blocks=1 << 30,
        )
        assert big.blocks > small.blocks

    def test_capped_by_what_the_block_table_can_address(self):
        """Blocks past ``max_batch_size * blocks_per_seq`` are allocated and never
        handed out — memory the pool cannot use is worse than memory it does not
        own."""
        p = _profile(total_gib=80.0, free_gib=76.0)
        cap = 4096
        sizing = derive_pool_blocks(p, self.PER_BLOCK, min_blocks=32, max_blocks=cap)
        assert sizing.blocks == cap
        assert sizing.limited_by == "block_table"

    def test_raises_when_the_minimum_does_not_fit(self):
        p = _profile(total_gib=8.0, free_gib=0.5)
        with pytest.raises(ValueError) as exc:
            derive_pool_blocks(p, self.PER_BLOCK, min_blocks=1 << 20, max_blocks=1 << 30)
        msg = str(exc.value)
        # Actionable: the arithmetic, and every lever the operator has.
        for fragment in (
            "max_batch_size",
            "num_left_chunks",
            "gpu_memory_utilization",
            "max_num_blocks",
        ):
            assert fragment in msg, f"{fragment!r} missing from: {msg}"

    def test_rejects_a_floor_above_the_ceiling(self):
        with pytest.raises(ValueError, match="block table"):
            derive_pool_blocks(_profile(), self.PER_BLOCK, min_blocks=100, max_blocks=10)


# ---------------------------------------------------------------------------
# Decoder-KV budget derivation
# ---------------------------------------------------------------------------


class TestDeriveDecodeKvBudget:
    def test_budget_is_the_available_memory(self):
        p = _profile(total_gib=32.0, free_gib=20.0, activation_gib=2.0)
        budget = derive_decode_kv_budget(p)
        assert budget.gib == pytest.approx(p.available_bytes / GIB)
        assert budget.rows is None
        assert not budget.clamped_to_one_row

    def test_reports_the_rows_it_affords(self):
        p = _profile(total_gib=32.0, free_gib=20.0, activation_gib=2.0)
        per_row = 78 * 1024**2  # a Qwen2-Audio-7B row at its position budget
        budget = derive_decode_kv_budget(p, bytes_per_row=per_row)
        assert budget.rows == p.available_bytes // per_row

    def test_clamps_up_so_a_tight_card_still_admits_one_row(self):
        """A budget below one row rejects every request — worse than admitting
        one and letting the allocator be the judge."""
        p = _profile(total_gib=8.0, free_gib=0.5, activation_gib=0.1)
        per_row = 4 * GIB
        budget = derive_decode_kv_budget(p, bytes_per_row=per_row)
        assert budget.clamped_to_one_row
        assert budget.rows == 1
        assert budget.gib == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Who owns a pool at all
# ---------------------------------------------------------------------------


class TestPoolOwnershipIsDeclared:
    """The derivation must only run for a runtime that actually builds a pool.

    A recurrent-state backend (Zipformer) has a ``cache_spec`` and streams
    happily, but allocates its caches per request — deriving a pool size for it
    would mean a probe forward, and possibly a startup failure on a small card,
    over memory nothing will allocate.  Declared on the backend class rather than
    name-checked in the engine, so a new pool-owning runtime opts in by
    declaration.
    """

    def test_only_the_paged_backend_claims_the_pool(self):
        from oasr.engine.streaming_backend import get_streaming_backend_class

        assert get_streaming_backend_class("paged").allocates_paged_pool is True
        assert get_streaming_backend_class("stateful").allocates_paged_pool is False
        assert get_streaming_backend_class("none").allocates_paged_pool is False

    def test_unknown_kind_raises_the_registry_error(self):
        from oasr.engine.streaming_backend import get_streaming_backend_class

        with pytest.raises(NotImplementedError, match="No streaming backend"):
            get_streaming_backend_class("does-not-exist")


# ---------------------------------------------------------------------------
# Config surface
# ---------------------------------------------------------------------------


class TestEngineConfigSurface:
    def test_none_means_derive_and_is_accepted(self):
        cfg = EngineConfig(max_num_blocks=None)
        assert cfg.max_num_blocks is None

    def test_zero_blocks_is_still_a_mistake(self):
        with pytest.raises(ValueError, match="max_num_blocks"):
            EngineConfig(max_num_blocks=0)

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
    def test_utilization_bounds(self, bad):
        with pytest.raises(ValueError, match="gpu_memory_utilization"):
            EngineConfig(gpu_memory_utilization=bad)

    def test_kv_budget_zero_is_off_negative_is_an_error(self):
        assert EngineConfig(decode_kv_budget_gib=0).decode_kv_budget_gib == 0
        with pytest.raises(ValueError, match="decode_kv_budget_gib"):
            EngineConfig(decode_kv_budget_gib=-1.0)

    def test_build_cache_config_refuses_an_unresolved_pool(self):
        """``None`` is a request for a derivation, not a value.  Reaching the
        cache config with it unresolved means nobody derived it."""
        from oasr.models.base import CacheSpec

        spec = CacheSpec(
            num_layers=4, n_kv_head=2, head_dim=32, hidden_dim=128, conv_kernel_size=15
        )
        cfg = EngineConfig(max_num_blocks=None)
        with pytest.raises(ValueError, match="derive from free VRAM"):
            cfg.build_cache_config(spec)

    def test_build_cache_config_passes_a_resolved_pool_through(self):
        from oasr.models.base import CacheSpec

        spec = CacheSpec(
            num_layers=4, n_kv_head=2, head_dim=32, hidden_dim=128, conv_kernel_size=15
        )
        cc = EngineConfig(max_num_blocks=777).build_cache_config(spec)
        assert cc.max_num_blocks == 777


# ---------------------------------------------------------------------------
# On a real device, with a real checkpoint
# ---------------------------------------------------------------------------


def _first_wav(wav_dir: str) -> torch.Tensor:
    import torchaudio

    wavs = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
    wave, _sr = torchaudio.load(wavs[0])
    return wave.squeeze(0)


@pytest.mark.cuda
@pytest.mark.slow
class TestDerivedPoolOnDevice:
    """What the CPU tests structurally cannot check: a pool that really allocates."""

    def _engine(self, ckpt_dir: str, **overrides):
        from oasr.engine import ASREngine, EngineConfig

        kwargs = {
            "ckpt_dir": ckpt_dir,
            "device": "cuda",
            "dtype": torch.float16,
            "chunk_size": 16,
            "num_left_chunks": -1,
            "max_batch_size": 8,
        }
        kwargs.update(overrides)
        return ASREngine(EngineConfig(**kwargs))

    def test_derived_pool_allocates_and_is_recorded(self, ckpt_dir, device):
        engine = self._engine(ckpt_dir, max_num_blocks=None)
        try:
            cfg = engine._config
            assert isinstance(cfg.max_num_blocks, int) and cfg.max_num_blocks > 0
            # The derivation ran and left an auditable profile behind.
            profile = engine._memory_profile
            assert profile is not None
            assert profile.total_bytes > 0
            assert profile.activation_measured
            assert profile.activation_bytes > 0
            # The pool it sized is the pool that got built.
            pool = engine._model_runner._block_pool
            assert pool is not None and pool.num_total_blocks == cfg.max_num_blocks
            # And it fits: allocation already happened, so the card still has
            # room under the utilization cap.
            free, total = torch.cuda.mem_get_info(device)
            assert total - free <= profile.cap_bytes
            # Every stream keeps at least the documented minimum context.
            assert cfg.max_num_blocks >= cfg.max_batch_size * MIN_BLOCKS_PER_STREAM
        finally:
            engine.shutdown()

    def test_deriving_changes_capacity_not_transcripts(self, ckpt_dir, wav_dir, device):
        """A derived pool is bigger, so streams live longer — but a clip that fit
        the old pool must decode to exactly the same text."""
        wave = _first_wav(wav_dir)
        explicit = self._engine(ckpt_dir, max_num_blocks=2048)
        try:
            baseline = explicit.transcribe(wave)
            baseline_blocks = explicit._config.max_num_blocks
        finally:
            explicit.shutdown()
        del explicit
        torch.cuda.empty_cache()

        derived = self._engine(ckpt_dir, max_num_blocks=None)
        try:
            assert derived._config.max_num_blocks >= baseline_blocks
            assert derived.transcribe(wave) == baseline
        finally:
            derived.shutdown()

    def test_with_eviction_the_derivation_is_an_exactness_check(self, ckpt_dir, wav_dir, device):
        """Bounded history is an exact requirement, not a preference.

        With `num_left_chunks >= 0` the pool has no capacity gate — the oldest
        block is recycled — so safety rests entirely on it holding every
        concurrent stream's retained history.  The floor and the ceiling coincide
        there, and the derivation reduces to "check that it fits" and fill in the
        number the operator used to hand-compute.
        """
        left, chunk, block, batch = 16, 16, 16, 8
        engine = self._engine(
            ckpt_dir, max_num_blocks=None, num_left_chunks=left, max_batch_size=batch
        )
        try:
            per_stream = -(-(chunk * left) // block)
            assert engine._config.max_num_blocks == batch * per_stream
            # And the invariant CacheConfig checks is satisfied by construction.
            pool_cfg = engine._model_runner._block_pool.config
            assert pool_cfg.max_num_blocks >= batch * pool_cfg.blocks_per_stream
            assert engine.transcribe(_first_wav(wav_dir)).strip()
        finally:
            engine.shutdown()

    def test_explicit_pool_skips_the_probe_entirely(self, ckpt_dir, device):
        """Nothing is measured — and no probe forward runs — unless a capacity
        was left to derive.  The default config must pay nothing for H4."""
        engine = self._engine(ckpt_dir, max_num_blocks=1024)
        try:
            assert engine._memory_profile is None
            assert engine._config.max_num_blocks == 1024
        finally:
            engine.shutdown()


@pytest.mark.cuda
@pytest.mark.slow
@pytest.mark.requires_assets("WHISPER_CKPT")
class TestDerivedDecodeKvBudget:
    """The AR half: ``decode_kv_budget_gib=None`` derives instead of disabling."""

    def _engine(self, ckpt_dir: str, **overrides):
        from oasr.engine import ASREngine, EngineConfig

        cfg = EngineConfig(
            ckpt_dir=ckpt_dir,
            device="cuda",
            dtype=torch.float16,
            service_mode="offline",
            max_batch_size=4,
            **overrides,
        )
        return ASREngine(cfg)

    def test_budget_is_derived_and_admits_work(self, wav_dir, device):
        from tests import assets

        ckpt = assets.require("WHISPER_CKPT")
        engine = self._engine(ckpt)
        try:
            budget = engine._config.decode_kv_budget_gib
            assert isinstance(budget, float) and budget > 0
            assert engine._memory_profile is not None
            # A derived ceiling must not throttle a batch the card can hold.
            texts = engine.transcribe_offline([_first_wav(wav_dir)])
            assert texts and isinstance(texts[0], str) and texts[0].strip()
        finally:
            engine.shutdown()

    def test_zero_keeps_the_byte_budget_off(self, device):
        from tests import assets

        ckpt = assets.require("WHISPER_CKPT")
        engine = self._engine(ckpt, decode_kv_budget_gib=0)
        try:
            assert engine._config.decode_kv_budget_gib == 0
            assert engine._memory_profile is None  # nothing left to derive
        finally:
            engine.shutdown()
