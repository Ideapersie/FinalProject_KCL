"""
tests/regression/test_p3_corpus_independence.py — justify reusing the P3 log.

P3 (closed-book) was never re-run against the real MedCorp corpus; the results
tables and figures use `p3_mirage200.jsonl`, which predates it. That is sound —
P3 never retrieves, so no corpus can reach its answers — but "sound" is a claim,
and a claim in a dissertation should be a checked invariant, not a promise.

This test converts the justification into an assertion. It checks that:

  1. P3 evaluated the IDENTICAL question set, in the IDENTICAL order, as the
     MedCorp runs. This makes it a *paired* comparison over the same items —
     which is a stronger basis for comparison than a fresh re-run on a
     re-sampled set would have been.
  2. P3 never retrieved anything on any question.

If either ever becomes false, the reuse is no longer justified and this test
fails, forcing the 3.2-hour re-run rather than letting a silent mismatch through.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "raw_logs"

P3 = RAW / "p3_mirage200.jsonl"
MEDCORP = [RAW / "p1_medcorp_mcq.jsonl",
           RAW / "p4_medcorp_mcq.jsonl",
           RAW / "p5_medcorp_mcq.jsonl"]

pytestmark = pytest.mark.skipif(
    not P3.exists(), reason="run logs not present in this checkout",
)


def _records(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _qids(path: Path) -> list[str]:
    return [r["question_id"] for r in _records(path)]


@pytest.mark.parametrize("other", MEDCORP, ids=lambda p: p.stem)
def test_p3_is_a_paired_run_over_the_same_questions(other):
    """P3 must cover the same question-ids, in the same order, as the MedCorp runs."""
    assert _qids(P3) == _qids(other), (
        f"P3's question set no longer matches {other.name}. The closed-book log "
        f"can only stand in for a MedCorp run while it is a paired comparison "
        f"over the identical questions — otherwise P3 must be re-run."
    )


def test_p3_never_retrieved():
    """P3 is closed-book: no record may show retrieval, and no gate may have run."""
    recs = _records(P3)
    assert recs, "P3 log is empty"

    retrieved = [r["question_id"] for r in recs if r.get("retrieval_triggered")]
    assert not retrieved, (
        f"P3 is supposed to be closed-book but {len(retrieved)} records show "
        f"retrieval_triggered=True; its corpus-independence claim is void."
    )

    cited = [r["question_id"] for r in recs if r.get("retrieved_chunk_ids")]
    assert not cited, "P3 records carry retrieved chunk ids — it retrieved something."


def test_p3_accuracy_is_corpus_independent_reference():
    """Sanity: P3's accuracy is the closed-book reference the report cites."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from medrag_adaptive.evaluation.loading import load_run
    from medrag_adaptive.evaluation.metrics import aggregate_summary

    s = aggregate_summary(load_run(P3))
    assert s.n == 200
    assert s.retrieval_rate == 0.0
    # The closed-book ceiling the whole "distance to safety" argument rests on.
    assert s.accuracy == pytest.approx(0.62, abs=0.005)
