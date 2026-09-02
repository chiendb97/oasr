# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""``PredictorStepGraphCache`` and the shape of a predictor state.

The cache decides up front whether a state is one a captured step can carry.
That check used to require a *sequence* of CUDA tensors, and icefall's stateless
predictor carries its whole state as a **single** ``(B, context_size)`` label
window — so for that model ``step`` returned ``None`` before ``_capture`` was
ever reached and the graph was never built.  Measured on a 32-utterance batch:
**0 replays against 304 eager fallbacks**, and unblocking it was worth
1.27-1.40x on transducer offline decode.

The second half of the fix is easy to miss: ``detach`` returned a bare tensor
*unchanged*.  That was harmless only while a bare state could never be captured.
The moment it can, the caller is holding graph memory, and a streaming session
that stores it across ticks reads whatever the next replay wrote.
"""

from __future__ import annotations

import pytest
import torch

from oasr.engine.predictor_graph import PredictorStepGraphCache as Cache

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


class TestCapturable:
    @needs_cuda
    def test_a_bare_cuda_tensor_is_capturable(self):
        """The regression: icefall's stateless predictor state is exactly this."""
        assert Cache.capturable(torch.zeros(4, 2, device="cuda"))

    @needs_cuda
    def test_a_sequence_of_cuda_tensors_is_still_capturable(self):
        t = torch.zeros(4, 2, device="cuda")
        assert Cache.capturable((t, t))
        assert Cache.capturable([t, t])

    def test_a_cpu_state_is_not(self):
        assert not Cache.capturable(torch.zeros(4, 2))
        assert not Cache.capturable((torch.zeros(4, 2),))

    def test_non_tensor_states_are_refused(self):
        assert not Cache.capturable(None)
        assert not Cache.capturable(())
        assert not Cache.capturable([])
        assert not Cache.capturable({"h": 1})
        assert not Cache.capturable(((torch.zeros(1),),))  # nested


class TestDetach:
    @needs_cuda
    def test_a_bare_tensor_is_copied_not_aliased(self):
        """Graph memory must not escape; returning the same object let it."""
        src = torch.zeros(4, 2, device="cuda")
        out = Cache.detach(src)
        assert out is not src, "detach aliased the caller's state"
        out.fill_(1.0)
        assert float(src.abs().sum()) == 0.0, "detach returned a view"

    @needs_cuda
    def test_a_sequence_is_copied_elementwise(self):
        src = (torch.zeros(4, 2, device="cuda"), torch.zeros(4, device="cuda"))
        out = Cache.detach(src)
        assert all(a is not b for a, b in zip(src, out))
        out[0].fill_(1.0)
        assert float(src[0].abs().sum()) == 0.0


CONTEXT = 3


class _BareStatePredictor:
    """Minimal stand-in for a stateless predictor: the state is one tensor.

    Shaped like icefall's: ``(B, CONTEXT)`` of label ids, projected to ``dim``.
    """

    def __init__(self, dim=4, device="cuda"):
        # Deterministic and RNG-free on purpose.  A *failed* CUDA-graph capture
        # elsewhere in the process wedges the CUDA generator ("Offset increment
        # outside graph capture"), so a `torch.randn` here would make these tests
        # fail for a reason that has nothing to do with them.
        self.w = (
            torch.linspace(-1.0, 1.0, CONTEXT * dim, device=device)
            .reshape(CONTEXT, dim)
            .contiguous()
        )

    def advance(self, state, tok, emit):
        # Shift the label window left and append the emitted token.
        nxt = torch.roll(state, shifts=-1, dims=1)
        nxt[:, -1] = tok
        return torch.where(emit.unsqueeze(1), nxt, state)

    def predict(self, state):
        return state.float() @ self.w  # (B, CONTEXT) @ (CONTEXT, dim) -> (B, dim)


class _Joiner:
    def decoder_proj(self, x):
        return x * 2.0


@needs_cuda
class TestCapturesAndReplaysABareState:
    def _cache(self):
        return Cache(_BareStatePredictor(), _Joiner(), max_captures=4)

    def test_step_returns_a_replay_not_none(self):
        """What ``0 replays, 304 fallbacks`` looked like before the fix."""
        cache = self._cache()
        state = torch.zeros(4, CONTEXT, dtype=torch.long, device="cuda")
        tok = torch.tensor([1, 2, 3, 4], device="cuda")
        emit = torch.tensor([True, True, False, True], device="cuda")
        out = cache.step(state, tok, emit)
        assert out is not None, "bare-tensor state was refused"
        assert cache.num_captured == 1

    def test_the_returned_state_keeps_the_callers_shape(self):
        """A bare state in must not become a 1-tuple out."""
        cache = self._cache()
        state = torch.zeros(4, CONTEXT, dtype=torch.long, device="cuda")
        tok = torch.tensor([1, 2, 3, 4], device="cuda")
        emit = torch.ones(4, dtype=torch.bool, device="cuda")
        new_state, _proj = cache.step(state, tok, emit)
        assert isinstance(new_state, torch.Tensor), type(new_state)

    def test_replay_matches_the_eager_step(self):
        cache = self._cache()
        pred, join = cache._predictor, cache._joiner
        state = torch.tensor([[0, 1, 2]] * 4, device="cuda")  # (4, CONTEXT)
        tok = torch.tensor([5, 6, 7, 8], device="cuda")
        emit = torch.tensor([True, False, True, False], device="cuda")

        want_state = pred.advance(state, tok, emit)
        want_proj = join.decoder_proj(pred.predict(want_state))
        got_state, got_proj = cache.step(state, tok, emit)

        assert torch.equal(got_state, want_state)
        assert torch.allclose(got_proj, want_proj, atol=0, rtol=0)

    def test_successive_replays_carry_state_forward(self):
        """The graph writes its output back into the buffers it reads."""
        cache = self._cache()
        pred = cache._predictor
        state = torch.zeros(4, CONTEXT, dtype=torch.long, device="cuda")
        emit = torch.ones(4, dtype=torch.bool, device="cuda")
        want = state
        for step in range(5):
            tok = torch.full((4,), step + 1, device="cuda")
            want = pred.advance(want, tok, emit)
            state, _ = cache.step(state, tok, emit)
            assert torch.equal(state, want), f"diverged at step {step}"

    def test_a_second_batch_width_gets_its_own_capture(self):
        cache = self._cache()
        for b in (2, 4):
            cache.step(
                torch.zeros(b, CONTEXT, dtype=torch.long, device="cuda"),
                torch.ones(b, dtype=torch.long, device="cuda"),
                torch.ones(b, dtype=torch.bool, device="cuda"),
            )
        assert cache.num_captured == 2

    def test_a_bare_state_and_a_one_tuple_do_not_share_a_capture(self):
        """They hand the predictor different shapes, so the key must differ."""
        cache = self._cache()
        s = torch.zeros(4, CONTEXT, dtype=torch.long, device="cuda")
        assert cache._key((s,), bare=True) != cache._key((s,), bare=False)
