#!/usr/bin/env python3
"""Tuned routing tables declare which GPU their numbers came from.

Two kinds of routing decision live in ``oasr/jit``.  A *derived* one reads the
machine — ``gated_mlp_ctas_per_sm`` takes ``multi_processor_count`` and the real
opt-in shared memory, ``selectBmmTile`` takes the SM count — and follows the card
it runs on.  A *measured* one is a cut-off somebody timed once:
``_LSTM_BANDS``'s ``(hidden, batch)`` bounds, the ``_TILES`` ranking, the
gated-MLP candidate list.

Both are legitimate.  What was missing is the distinction: the measured tables
were applied identically on sm_80, sm_86, sm_89 and sm_120 — an A30 at 56 SMs
and 933 GB/s, an A100 at 108 and 1555, an L40S at 142 and 864, a 5090 at 170 and
1792 — with nothing in the code, the logs or any counter saying which card the
numbers came from.  These tests pin the declaration, not the numbers.

Pure Python plus one device query, so they hold on any box.
"""

import pytest

from oasr.jit.measured import (
    Machine,
    MeasuredOn,
    current_machine,
    extrapolations,
    is_native,
    note_extrapolation,
    reset_extrapolations,
)


class TestMachineIdentity:
    """Identity is what can be queried exactly, and bandwidth cannot be.

    The obvious ``2 * memory_clock_rate * bus_width / 8`` is right for HBM and
    GDDR6 and **wrong for GDDR7**: on an RTX 5090 it returns 896 GB/s against a
    real 1792, because PAM3 signalling puts the effective rate at more than twice
    the reported clock.  A first cut of this module used that formula as an
    identity field, which would have declared an extrapolation *on the very card
    these tables were measured on*.  So ``(sm, sms)`` it is — and that separates
    every card in scope anyway.
    """

    def test_bandwidth_is_not_an_identity_field(self):
        from dataclasses import fields

        names = {f.name for f in fields(Machine)}
        assert names == {"name", "sm", "sms"}, (
            "a field that cannot be read reliably on every memory technology must "
            "not decide whether a table is native here"
        )

    @pytest.mark.parametrize(
        "sm,sms,card",
        [(80, 56, "A30"), (80, 108, "A100"), (89, 142, "L40S"), (120, 170, "RTX 5090")],
    )
    def test_the_cards_in_scope_are_distinguishable(self, sm, sms, card):
        """``(sm, sms)`` has to separate the supported set, or two different
        machines would compare equal and the extrapolation would go unreported."""
        others = {(80, 56), (80, 108), (89, 142), (120, 170)} - {(sm, sms)}
        assert (sm, sms) not in others, card


@pytest.mark.cuda
class TestExtrapolationIsCounted:
    def setup_method(self):
        reset_extrapolations()

    def teardown_method(self):
        reset_extrapolations()

    def test_a_table_measured_here_is_not_an_extrapolation(self):
        here = current_machine()
        if here is None:
            pytest.skip("no CUDA device to identify")
        record = MeasuredOn("t", here, "1 GB/s", "src", "moves")
        assert is_native(record)
        assert note_extrapolation(record) is False
        assert not extrapolations()

    def test_a_table_measured_elsewhere_is_recorded_once(self):
        here = current_machine()
        if here is None:
            pytest.skip("no CUDA device to identify")
        elsewhere = Machine(name="Some Other GPU", sm=here.sm, sms=here.sms + 1)
        record = MeasuredOn("t", elsewhere, "1 GB/s", "src", "moves")
        assert not is_native(record)
        assert note_extrapolation(record) is True
        assert note_extrapolation(record) is True, "still true on the second call"
        assert list(extrapolations()) == ["t"], "one entry per table, not per call"
        assert here.name in extrapolations()["t"]
        assert "Some Other GPU" in extrapolations()["t"]

    def test_identity_is_the_machine_not_the_name(self):
        """Two cards can share a marketing name and differ in clocks, and an
        unidentified device with the same SM count and bandwidth *is*, for a
        bandwidth-bound crossover, the machine that was measured."""
        here = current_machine()
        if here is None:
            pytest.skip("no CUDA device to identify")
        renamed = Machine(name="renamed", sm=here.sm, sms=here.sms)
        assert is_native(MeasuredOn("t", renamed, "1 GB/s", "src", "moves"))


class TestTheShippedRecords:
    """Every fixed cut-off in ``oasr/jit`` names the card it was timed on."""

    def _records(self):
        from oasr.jit import mlp, recurrent_cute

        return [recurrent_cute._MEASURED, mlp._MEASURED]

    def test_each_names_a_table_a_source_and_what_moves_it(self):
        for record in self._records():
            assert record.table, "a record that does not name its table cannot be acted on"
            assert ".artifacts/" in record.source, (
                f"{record.table}: the source must point at the note holding the "
                f"protocol, so a re-measure starts there rather than from scratch"
            )
            assert record.moves_with, f"{record.table}: say what moves the crossover"
            assert record.machine.sms > 0
            assert record.bandwidth, f"{record.table}: record the bandwidth for the re-measure"

    def test_the_two_fixed_cutoff_tables_are_covered(self):
        tables = {r.table for r in self._records()}
        assert any("_LSTM_BANDS" in t for t in tables)
        assert any("_CANDIDATES" in t for t in tables)


class TestTheGatedMlpBandIsDerived:
    """``_BAND_MAX_ROWS`` used to be a literal 64 beside a candidate list whose
    largest ``m_block`` was also 64.

    Agreeing is not the same as being linked: a 128-row candidate added for its
    own sake would have widened what the kernel can do and left the band at 64,
    quietly declining the shapes it was added for.  The band's rule is "one
    m-tile", so the band *is* the largest m-tile that exists.
    """

    def test_the_band_is_the_largest_candidate_tile(self):
        from oasr.jit.mlp import _BAND_MAX_ROWS, _CANDIDATES

        assert _BAND_MAX_ROWS == max(m_max for m_max, _ in _CANDIDATES)

    def test_the_band_is_written_as_a_derivation_not_a_literal(self):
        """The value alone cannot carry this.

        ``_BAND_MAX_ROWS = 64`` and ``max(...) == 64`` are indistinguishable while
        the largest candidate happens to be 64 — which is exactly the state that
        made the coupling invisible.  What has to hold is that the band is
        *spelled* in terms of the list, so adding a candidate moves it.  So this
        reads the source, the way the SM100 epilogue tests do for the same reason.
        """
        from helpers import REPO_ROOT

        src = (REPO_ROOT / "oasr/jit/mlp.py").read_text()
        line = next(ln for ln in src.splitlines() if ln.startswith("_BAND_MAX_ROWS"))
        assert "_CANDIDATES" in line, (
            f"_BAND_MAX_ROWS must be derived from the candidate list, not written "
            f"beside it: {line!r}. A 128-row candidate added on its own would "
            f"widen what the kernel can do and leave the band at 64, declining "
            f"the shapes it was added for."
        )


@pytest.mark.cuda
class TestRoutingIsUnchanged:
    """The declaration must not move a single routing decision.

    An extrapolation that silently narrowed the band would be a performance
    regression wearing an honesty label.
    """

    @pytest.mark.parametrize(
        "hidden,batch,expected",
        [
            (256, 1, True),
            (256, 4096, True),
            (768, 256, True),
            (768, 512, False),
            (1536, 64, True),
            (1536, 128, False),
            (2048, 32, True),
            (2048, 64, False),
        ],
    )
    def test_the_lstm_band_still_says_what_the_table_says(self, hidden, batch, expected):
        from oasr.jit import recurrent_cute as rc

        if rc._probe() is None:
            pytest.skip("the CuTeDSL recurrent step is not available on this device")
        assert rc.should_use(4, hidden, batch) is expected

    def test_the_gated_mlp_band_still_ends_at_one_m_tile(self):
        from oasr.jit.mlp import _BAND_MAX_ROWS, gated_mlp_config_supported, should_use_gated_mlp

        n, k = 11008, 4096
        if not gated_mlp_config_supported(rows=_BAND_MAX_ROWS, n=n, k=k):
            pytest.skip("the CuTeDSL gated MLP is not available on this device")
        assert should_use_gated_mlp(rows=_BAND_MAX_ROWS, n=n, k=k)
        assert not should_use_gated_mlp(rows=_BAND_MAX_ROWS + 1, n=n, k=k)


@pytest.mark.cuda
class TestTheShippedGatesRecordIt:
    """The declaration is only worth anything if the real routing gates take it.

    A record nobody calls is documentation that cannot go stale *and* cannot be
    trusted; these pin the two call sites.
    """

    def setup_method(self):
        reset_extrapolations()

    def teardown_method(self):
        reset_extrapolations()

    def test_the_lstm_band_records_when_it_is_consulted(self):
        from oasr.jit import recurrent_cute as rc

        if rc._probe() is None:
            pytest.skip("the CuTeDSL recurrent step is not available on this device")
        if is_native(rc._MEASURED):
            pytest.skip("this box is the card the band was measured on")
        rc.should_use(4, 256, 8)
        assert rc._MEASURED.table in extrapolations()

    def test_the_gated_mlp_candidate_list_records_when_it_is_scored(self):
        from oasr.jit import mlp

        if is_native(mlp._MEASURED):
            pytest.skip("this box is the card the candidates were measured on")
        mlp.select_gated_mlp_tile(32, 11008)
        assert mlp._MEASURED.table in extrapolations()


@pytest.mark.cuda
class TestTheReportSaysIt:
    def test_an_extrapolated_table_reaches_the_gap_report(self):
        """``format_gap_report`` is the one surface that answers "what did not
        reach a tuned kernel?"; a table tuned on somebody else's GPU belongs
        beside the other four categories it already separates."""
        from oasr.layers._backend import format_gap_report, reset_backend_stats

        here = current_machine()
        if here is None:
            pytest.skip("no CUDA device to identify")
        reset_backend_stats()
        elsewhere = Machine(name="Some Other GPU", sm=here.sm, sms=here.sms + 1)
        note_extrapolation(MeasuredOn("jit.demo._TABLE", elsewhere, "1 GB/s", "src", "moves"))
        report = format_gap_report()
        assert "measured on another GPU" in report, report
        assert "jit.demo._TABLE" in report and "Some Other GPU" in report, report
        reset_backend_stats()
        assert "jit.demo._TABLE" not in format_gap_report()
