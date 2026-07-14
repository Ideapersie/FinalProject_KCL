"""
tests/unit/test_metrics.py — aggregation, ECE, citation P/R, safety envelope.

These metrics turn the per-query JSONL logs into the tables the report needs:
  - aggregate_summary: accuracy / retrieval-rate / mean latency-energy per group,
  - expected_calibration_error: does the gate signal track correctness?
  - citation_precision_recall: are cited sources actually retrieved?
  - safety_envelope: the headline contribution — cheapest policy clearing the
    per-risk accuracy bar at each risk level.

All run on plain dicts (the RunRecord JSONL shape), so no model or data download.
"""

from __future__ import annotations

import pytest

from medrag_adaptive.evaluation.metrics import (
    aggregate_summary,
    expected_calibration_error,
    citation_precision_recall,
    safety_envelope,
    proportion_diff_interval,
    wilson_interval,
    risk_stratified_summary,
    PolicySummary,
)


def _rec(**kw):
    """A minimal RunRecord-shaped dict with sensible defaults."""
    base = dict(
        question_id="q", policy_name="p3_closed_book", risk_level="low",
        is_correct=True, retrieval_triggered=False, gate_signal_value=None,
        f1_score=1.0, latency_ns=1_000_000_000, energy_kwh=0.0,
        citations=[], retrieved_chunk_ids=[],
    )
    base.update(kw)
    return base


# ── aggregate_summary ────────────────────────────────────────────────

class TestAggregate:
    def test_accuracy_and_retrieval_rate(self):
        recs = [
            _rec(is_correct=True, retrieval_triggered=True),
            _rec(is_correct=False, retrieval_triggered=False),
            _rec(is_correct=True, retrieval_triggered=True),
            _rec(is_correct=True, retrieval_triggered=False),
        ]
        s = aggregate_summary(recs)
        assert s.n == 4
        assert s.accuracy == pytest.approx(0.75)
        assert s.retrieval_rate == pytest.approx(0.5)

    def test_mean_latency_seconds(self):
        recs = [_rec(latency_ns=2_000_000_000), _rec(latency_ns=4_000_000_000)]
        s = aggregate_summary(recs)
        assert s.mean_latency_s == pytest.approx(3.0)

    def test_empty_is_safe(self):
        # Accuracy of nothing is NaN, not 0.0 — a missing metric must be loud,
        # not masquerade as a policy that got every question wrong.
        s = aggregate_summary([])
        assert s.n == 0
        assert s.accuracy != s.accuracy      # NaN

    def test_none_is_correct_excluded_from_accuracy(self):
        # The canonical loader sets is_correct=None on open-ended records; those
        # must not be counted as wrong answers.
        recs = [_rec(is_correct=True), _rec(is_correct=None), _rec(is_correct=None)]
        s = aggregate_summary(recs)
        assert s.n == 3                      # all records counted for latency etc.
        assert s.accuracy == pytest.approx(1.0)   # ...but accuracy over the 1 scorable

    def test_all_none_gives_nan_accuracy(self):
        s = aggregate_summary([_rec(is_correct=None), _rec(is_correct=None)])
        assert s.accuracy != s.accuracy      # NaN, not 0.0


# ── ECE ──────────────────────────────────────────────────────────────

class TestECE:
    def test_perfect_calibration_is_zero(self):
        # Confidence exactly equals empirical accuracy in every bin → ECE 0.
        recs = ([_rec(gate_signal_value=0.9, is_correct=True)] * 9
                + [_rec(gate_signal_value=0.9, is_correct=False)] * 1)
        ece = expected_calibration_error(recs, n_bins=10)
        assert ece == pytest.approx(0.0, abs=1e-9)

    def test_overconfidence_is_positive(self):
        # Signal 1.0 but always wrong → max miscalibration.
        recs = [_rec(gate_signal_value=1.0, is_correct=False)] * 10
        assert expected_calibration_error(recs, n_bins=10) == pytest.approx(1.0)

    def test_skips_records_without_signal(self):
        recs = [_rec(gate_signal_value=None, is_correct=True)]
        # No usable signal → defined as 0.0 (nothing to calibrate).
        assert expected_calibration_error(recs) == 0.0


# ── citation precision / recall ──────────────────────────────────────

class TestCitationPR:
    def test_all_cited_were_retrieved(self):
        rec = _rec(citations=[{"chunk_id": "c1"}, {"chunk_id": "c2"}],
                   retrieved_chunk_ids=["c1", "c2", "c3"])
        p, r = citation_precision_recall(rec)
        assert p == pytest.approx(1.0)         # both cited ∈ retrieved
        assert r == pytest.approx(2 / 3)       # 2 of 3 retrieved were cited

    def test_invented_citation_drops_precision(self):
        rec = _rec(citations=[{"chunk_id": "c1"}, {"chunk_id": "ghost"}],
                   retrieved_chunk_ids=["c1"])
        p, r = citation_precision_recall(rec)
        assert p == pytest.approx(0.5)         # only c1 of the 2 cited is real

    def test_no_citations_is_zero(self):
        assert citation_precision_recall(_rec(citations=[])) == (0.0, 0.0)


# ── safety envelope (headline) ───────────────────────────────────────

class TestSafetyEnvelope:
    def test_picks_cheapest_policy_clearing_bar(self):
        # low bar = 0.70. p3 cheap+passes, p1 dear+passes → choose p3.
        recs = (
            [_rec(policy_name="p3_closed_book", risk_level="low",
                  is_correct=True, latency_ns=1_000_000_000)] * 8
            + [_rec(policy_name="p3_closed_book", risk_level="low",
                    is_correct=False, latency_ns=1_000_000_000)] * 2
            + [_rec(policy_name="p1_always_retrieve", risk_level="low",
                    is_correct=True, latency_ns=5_000_000_000)] * 10
        )
        env = safety_envelope(recs, thresholds={"low": 0.70})
        assert env["low"].policy_name == "p3_closed_book"   # cheaper, still ≥0.70

    def test_no_policy_clears_bar_returns_none(self):
        recs = [_rec(policy_name="p3_closed_book", risk_level="high",
                     is_correct=False)] * 10
        env = safety_envelope(recs, thresholds={"high": 0.90})
        assert env["high"] is None

    def test_dearer_policy_chosen_when_only_one_passes(self):
        recs = (
            [_rec(policy_name="p3_closed_book", risk_level="high",
                  is_correct=False, latency_ns=1_000_000_000)] * 10
            + [_rec(policy_name="p1_always_retrieve", risk_level="high",
                    is_correct=True, latency_ns=9_000_000_000)] * 10
        )
        env = safety_envelope(recs, thresholds={"high": 0.90})
        assert env["high"].policy_name == "p1_always_retrieve"


# ── Wilson interval + risk-stratified summary (A5 small-n caveat) ──────

class TestWilson:
    def test_zero_n_is_full_range(self):
        assert wilson_interval(0, 0) == (0.0, 1.0)

    def test_bounds_stay_in_unit_interval(self):
        lo, hi = wilson_interval(6, 6)   # all correct, tiny n
        assert 0.0 <= lo <= hi <= 1.0
        assert lo < 1.0                  # does not collapse to a point at p=1

    def test_wide_interval_for_small_n(self):
        # 6 samples => the interval must be wide (>0.3), the whole point of A5.
        lo, hi = wilson_interval(3, 6)
        assert (hi - lo) > 0.3

    def test_narrows_with_large_n(self):
        lo_s, hi_s = wilson_interval(3, 6)
        lo_l, hi_l = wilson_interval(300, 600)
        assert (hi_l - lo_l) < (hi_s - lo_s)


class TestRiskStratified:
    def test_reports_n_and_ci_per_level(self):
        recs = (
            [_rec(risk_level="low", is_correct=True)] * 20
            + [_rec(risk_level="medium", is_correct=False)] * 10
            + [_rec(risk_level="high", is_correct=True)] * 6
        )
        out = risk_stratified_summary(recs)
        assert out["low"]["n"] == 20
        assert out["high"]["n"] == 6
        assert out["medium"]["accuracy"] == pytest.approx(0.0)
        # high-risk cell keeps a wide interval despite the point estimate of 1.0
        assert (out["high"]["ci_high"] - out["high"]["ci_low"]) > 0.3

    def test_missing_level_is_nan_not_crash(self):
        out = risk_stratified_summary([_rec(risk_level="low", is_correct=True)])
        assert out["high"]["n"] == 0
        assert out["high"]["accuracy"] != out["high"]["accuracy"]  # nan


# ── Newcombe difference interval (the "+9.0 pp" headline claim) ────────

class TestProportionDiff:
    def test_difference_is_correct(self):
        diff, lo, hi = proportion_diff_interval(109, 200, 91, 200)   # P5 vs P1
        assert diff == pytest.approx(0.09, abs=1e-9)
        assert lo < diff < hi

    def test_interval_is_wide_at_n200(self):
        # The honest point of this function: at n=200/arm the interval is wide
        # enough that a +9pp gap is "consistent with an improvement", not proof.
        _, lo, hi = proportion_diff_interval(109, 200, 91, 200)
        assert (hi - lo) > 0.15

    def test_identical_arms_bracket_zero(self):
        diff, lo, hi = proportion_diff_interval(91, 200, 91, 200)
        assert diff == pytest.approx(0.0)
        assert lo < 0.0 < hi

    def test_narrows_with_larger_n(self):
        _, lo_s, hi_s = proportion_diff_interval(109, 200, 91, 200)
        _, lo_l, hi_l = proportion_diff_interval(1090, 2000, 910, 2000)
        assert (hi_l - lo_l) < (hi_s - lo_s)
