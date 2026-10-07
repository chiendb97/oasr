# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""``oasr.transducer_beam_topk`` -- one beam-search frame's selection, fused.

The oracle is the torch composition the kernel replaces in
:func:`oasr.engine.decode.transducer_beam.beam_search_step`: ``log_softmax`` of
the float-cast logits, the score add, ``topk`` over the beam's ``k * V``
candidates, the parent/label split, the window reorder and the row masks.

What the kernel promises, and what each test holds it to:

* **Scores bit-identical** to that composition (it reproduces torch's warp
  log-softmax with libdevice's precise exp/log), so ``torch.equal``.
* **The same selected set** in every row -- including a tie at the boundary,
  where both keep the lower index.
* **The same order** wherever the selected scores are distinct.  A tie *inside*
  the selected ``k`` is ranked by lower flat index here, where torch's sort is
  unstable, so those rows are compared as sets.
"""

from __future__ import annotations

import pytest
import torch

import oasr

pytestmark = pytest.mark.cuda

DTYPES = [torch.float16, torch.bfloat16]


def _reference(logits, scores, context, active, blank):
    B, k = scores.shape
    V, ctx = logits.size(1), context.size(2)
    log_probs = torch.log_softmax(logits.float(), dim=-1).view(B, k, V)
    total = scores.unsqueeze(-1) + log_probs
    top_scores, top_idx = total.view(B, k * V).topk(k, dim=-1)
    parent = torch.div(top_idx, V, rounding_mode="floor")
    label = top_idx - parent * V
    gathered = context.gather(1, parent.unsqueeze(-1).expand(B, k, ctx))
    shifted = torch.cat([gathered[:, :, 1:], label.unsqueeze(-1)], dim=2)
    gathered = torch.where((label == blank).unsqueeze(-1), gathered, shifted)
    keep = active.view(B, 1)
    stay = torch.arange(k, device=scores.device).expand(B, k)
    return (
        torch.where(keep.unsqueeze(-1), gathered, context),
        torch.where(keep, top_scores, scores),
        torch.where(keep, parent, stay),
        torch.where(keep, label, torch.full_like(label, blank)),
    )


def _frame(B, k, V, ctx, dtype, *, seed=0, dead=True):
    """A frame with padded (strided) logit rows, some dead slots and inactive rows."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    ld = (V + 4 + 7) // 8 * 8  # the joiner slices a padded projection
    logits = (torch.randn(B * k, ld, device="cuda", generator=gen) * 3).to(dtype)[:, :V]
    scores = torch.randn(B, k, device="cuda", generator=gen) * 5
    if dead and k > 2:
        scores[:, k // 2 + 1 :] = -1.0e30
    context = torch.randint(0, V, (B, k, ctx), device="cuda", generator=gen)
    active = torch.rand(B, device="cuda", generator=gen) > 0.2
    active[0] = True
    return logits, scores, context, active


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("V", [7, 16, 17, 37, 500, 1024])
@pytest.mark.parametrize("k", [1, 4, 8, 32])
@pytest.mark.parametrize("ctx", [1, 2])
def test_matches_the_torch_composition(dtype, V, k, ctx):
    if k > V:
        pytest.skip("a beam wider than the vocabulary has no k candidates")
    B, blank = 9, 0
    logits, scores, context, active = _frame(B, k, V, ctx, dtype, seed=V * 100 + k)
    want = _reference(logits, scores, context, active, blank)
    got = oasr.transducer_beam_topk(logits, scores, context, active, blank)

    assert torch.equal(got[1], want[1]), "scores must be bit-identical"
    got_flat = (got[2] * V + got[3]).tolist()
    want_flat = (want[2] * V + want[3]).tolist()
    for b in range(B):
        assert sorted(got_flat[b]) == sorted(want_flat[b]), f"row {b} selected a different set"
        if len(set(want[1][b].tolist())) == k:  # distinct scores: order and windows too
            assert got_flat[b] == want_flat[b], f"row {b} ordered differently"
            assert torch.equal(got[0][b], want[0][b]), f"row {b} windows differ"


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
def test_ties_rank_by_lower_flat_index(dtype):
    """Equal scores are ordered by ``j * V + v``, lower first -- a total order,
    so the slot a hypothesis lands in never depends on scheduling."""
    B, k, V = 1, 3, 8
    logits = torch.zeros(B * k, V, device="cuda", dtype=dtype)
    logits[:, 5] = 4.0  # every row's best label is 5, by the same margin
    scores = torch.zeros(B, k, device="cuda")  # three identical hypotheses
    context = torch.zeros(B, k, 2, dtype=torch.long, device="cuda")
    active = torch.ones(B, dtype=torch.bool, device="cuda")
    _, new_scores, parent, label = oasr.transducer_beam_topk(logits, scores, context, active, 0)
    assert parent.tolist() == [[0, 1, 2]] and label.tolist() == [[5, 5, 5]]
    assert len(set(new_scores.tolist()[0])) == 1


def test_inactive_rows_keep_their_beam():
    B, k, V, ctx = 4, 4, 37, 2
    logits, scores, context, _ = _frame(B, k, V, ctx, torch.bfloat16, seed=3)
    active = torch.zeros(B, dtype=torch.bool, device="cuda")
    new_context, new_scores, parent, label = oasr.transducer_beam_topk(
        logits, scores, context, active, 0
    )
    assert torch.equal(new_context, context) and torch.equal(new_scores, scores)
    assert parent.tolist() == [list(range(k))] * B and label.tolist() == [[0] * k] * B


def test_destination_passing():
    B, k, V, ctx = 2, 4, 37, 2
    logits, scores, context, active = _frame(B, k, V, ctx, torch.float16, seed=4)
    out = (
        torch.empty(B, k, ctx, dtype=torch.long, device="cuda"),
        torch.empty(B, k, device="cuda"),
        torch.empty(B, k, dtype=torch.long, device="cuda"),
        torch.empty(B, k, dtype=torch.long, device="cuda"),
    )
    got = oasr.transducer_beam_topk(logits, scores, context, active, 0, out=out)
    assert all(g.data_ptr() == o.data_ptr() for g, o in zip(got, out))


@pytest.mark.parametrize(
    "B,k,V,why",
    [(2, 33, 64, "beam"), (2, 4, 1025, "vocabulary"), (2, 8, 5, "vocabulary")],
)
def test_out_of_scope_shapes_are_refused(B, k, V, why):
    """Outside its scope the kernel raises; the beam step routes those frames to a
    declared gap rather than calling it (``transducer_beam_topk_supports``)."""
    from oasr.functionals.transducer import transducer_beam_topk_supports

    logits, scores, context, active = _frame(B, k, V, 2, torch.bfloat16, dead=False)
    assert not transducer_beam_topk_supports(logits, k)
    with pytest.raises(Exception, match=why):
        oasr.transducer_beam_topk(logits, scores, context, active, 0)
