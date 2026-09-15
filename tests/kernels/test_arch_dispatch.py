#!/usr/bin/env python3
"""``include/oasr/common/arch_dispatch.h`` — the per-device runtime queries.

Two things live here, and both were places a constant stood in for the machine.

**The heterogeneous-node guard: one module serves one SM family.**

``OASR_TARGET_SM`` is baked into a JIT module at compile time and the family is
resolved once, from whichever device happened to be current the first time a
kernel ran.  A process spanning two different GPUs — an A100 and an H100 in the
same box — loads one library for both, and the tensors that arrive on the other
card either fail to launch or run kernels specialised for the wrong
architecture.  Nothing said so.

The guard cannot be *provoked* on a homogeneous box, which is every box anyone
tests on, so these hold the predicate it decides with instead — evaluated inside
a TU carrying the real ``OASR_TARGET_SM`` — plus the structural fact that every
launcher reaches it.
"""

import functools

import pytest
import torch
from helpers import REPO_ROOT

pytestmark = pytest.mark.cuda


@functools.lru_cache(maxsize=1)
def _probe():
    from oasr.jit.core import gen_jit_spec

    src = REPO_ROOT / "tests/kernels/fixtures/arch_probe.cu"
    assert src.exists(), src
    return gen_jit_spec("arch_probe", [src]).build_and_load()


class TestTheBuildTargetIsReal:
    def test_the_module_carries_the_family_it_was_built_for(self):
        from oasr.jit.core import _get_target_sm

        assert _probe().target_sm() == _get_target_sm(), (
            "OASR_TARGET_SM in the compiled module must equal the family the "
            "Python side resolved; they are the two halves of the same decision"
        )

    def test_this_device_is_accepted(self):
        probe = _probe()
        sm = probe.device_sm(torch.cuda.current_device())
        assert sm > 0
        assert probe.arch_matches_build(sm), (
            f"the running device is SM{sm} and the module was built for "
            f"SM{probe.target_sm()} — the JIT resolved the wrong family"
        )
        assert probe.guard_refuses(torch.cuda.current_device()) is False


class TestTheGuardRefusesAnotherArchitecture:
    """What would happen on the second card of a heterogeneous node."""

    def test_a_different_family_is_refused(self):
        probe = _probe()
        built = probe.target_sm()
        others = [sm for sm in (75, 80, 86, 89, 90, 100, 120) if sm != built]
        for sm in others:
            assert not probe.arch_matches_build(
                sm
            ), f"a module built for SM{built} must not accept an SM{sm} device"

    def test_a_capability_below_turing_is_refused(self):
        """``resolveSmVersion`` throws rather than walking down to 75 — a Volta
        card in the box must not be handed Turing kernels."""
        # ``resolveSmVersion`` raises ``std::runtime_error``; tvm_ffi surfaces it
        # as RuntimeError, and the message has to name the capability so the
        # operator knows which card in the box is the problem.
        with pytest.raises(RuntimeError, match="SM70"):
            _probe().arch_matches_build(70)

    def test_a_newer_part_resolves_to_its_family(self):
        """sm_103 is served by the 100 family and sm_121 by 120, so a module built
        for those must accept them — the guard compares *families*, not raw
        capabilities, or a B300 would be refused by its own kernels."""
        probe = _probe()
        built = probe.target_sm()
        if built == 100:
            assert probe.arch_matches_build(103)
        elif built == 120:
            assert probe.arch_matches_build(121)
        else:
            pytest.skip(f"this box builds for sm_{built}; nothing resolves up into it")


class TestTheSharedMemoryBudgetIsTheDevices:
    """48 KiB is what a block gets *without asking*, not what the card has.

    It is the same on every architecture, which is exactly why hardcoding it
    looks portable and is not: it left 52 KiB unused on sm_86/89/120 and 115 KiB
    on sm_80, uniformly. ``getDeviceMaxSharedMemoryOptin`` asks the device.
    """

    def test_it_matches_what_torch_reports(self):
        probe = _probe()
        expected = torch.cuda.get_device_properties(0).shared_memory_per_block_optin
        assert probe.max_smem_optin(0) == expected

    def test_it_never_reports_less_than_the_free_floor(self):
        """A driver that cannot answer must leave the caller on the 48 KiB every
        block gets, not on zero — the value is used as a capacity bound."""
        assert _probe().max_smem_optin(0) >= 48 * 1024

    def test_this_card_has_headroom_over_the_free_floor(self):
        """If this ever stops holding, the constant was fine after all."""
        assert _probe().max_smem_optin(0) > 48 * 1024


class TestTheStaticLimitStaysAt48KiB:
    """The one place the constant is *right*, and must not be "fixed".

    ``topk.cuh`` gates on ``BlockRadixSort::TempStorage``, a ``__shared__``
    member — **static** shared memory, capped at 48 KiB per block on every
    architecture by CUDA itself. ``cudaFuncSetAttribute`` raises the *dynamic*
    ceiling and does nothing for static allocations, so opting in there would
    not compile the kernel, it would just not help. The audit note that prompted
    the recurrent change named this site too; this is the pin that says why it
    was left alone.
    """

    def test_topk_still_gates_on_the_static_limit(self):
        src = (REPO_ROOT / "include/oasr/topk.cuh").read_text()
        assert "48 * 1024" in src, "topk's sort gate is the one 48 KiB constant that is correct"
        decl = src[src.index("kSortFitsShmem") :]
        decl = decl[: decl.index(";") + 1]
        assert "48 * 1024" in decl, decl

    def test_topk_says_why_the_limit_is_not_raised(self):
        """A reader arriving with the audit note in hand will try to widen this;
        the file has to answer before they do."""
        src = (REPO_ROOT / "include/oasr/topk.cuh").read_text()
        assert "static-shmem limit" in src or "static shared" in src

    def test_topk_uses_static_shared_memory_for_the_sort(self):
        """The fact the limit follows from."""
        src = (REPO_ROOT / "include/oasr/topk.cuh").read_text()
        assert "__shared__ typename BlockRadixSort::TempStorage" in src


class TestTheRecurrentGatesAskTheDevice:
    def test_no_hardcoded_48kib_remains_in_the_recurrent_launcher(self):
        """All three sites there size *dynamic* shared memory, so all three can
        and do ask the device."""
        src = (REPO_ROOT / "include/oasr/recurrent/recurrent.cuh").read_text()
        offenders = [ln.strip() for ln in src.splitlines() if "48 * 1024" in ln]
        assert not offenders, offenders

    def test_the_cohort_gate_is_the_device_budget(self):
        src = (REPO_ROOT / "include/oasr/recurrent/recurrent.cuh").read_text()
        assert src.count("getDeviceMaxSharedMemoryOptin()") >= 3

    def test_every_widened_launch_opts_in_first(self):
        """Raising the gate without ``cudaFuncSetAttribute`` turns a declined
        shape into a failed launch."""
        src = (REPO_ROOT / "include/oasr/recurrent/recurrent.cuh").read_text()
        assert src.count("optInSharedMemory(") >= 4


class TestEveryLauncherReachesTheGuard:
    def test_get_stream_calls_it(self):
        """The guard sits in ``get_stream`` because that is the one function every
        launcher calls exactly once with the tensor's device.  A ``CHECK_*`` macro
        would have to be remembered at every call site, and the failure it guards
        against is silent — so forgetting it would not show up."""
        src = (REPO_ROOT / "csrc/tvm_ffi_utils.h").read_text()
        body = src[src.index("inline cudaStream_t get_stream") :]
        body = body[: body.index("\n}")]
        assert "checkDeviceMatchesBuild(device.device_id)" in body, body

    def test_a_real_kernel_still_runs(self):
        """The guard is on the launch path; it must cost correctness nothing."""
        import oasr

        a = torch.randn(64, 256, device="cuda", dtype=torch.float16)
        b = torch.randn(512, 256, device="cuda", dtype=torch.float16)
        out = oasr.gemm(a, b)
        torch.testing.assert_close(out, a @ b.t(), rtol=2e-2, atol=2e-2)
