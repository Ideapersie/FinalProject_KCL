"""
tests/unit/test_loading.py — the canonical log loader.

`load_run` is the single code path from a log file to any number in the report,
so its two on-read corrections are load-bearing and tested directly:
  - MCQ records are re-scored with the current extractor,
  - open-ended `is_correct` is voided (it is inconsistent across runs and would
    be meaningless if averaged).

Also asserts the raw file is never modified — the logs are the audit trail.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from medrag_adaptive.evaluation.loading import (
    load_run,
    load_run_with_report,
    is_mcq_record,
)


def _write(tmp_path: Path, records: list[dict]) -> Path:
    p = tmp_path / "run.jsonl"
    with io.open(p, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return p


def _mcq(**kw) -> dict:
    base = dict(question_id="q", correct_answer="B", answer_text="The answer is B.",
                is_correct=True, exact_match=1.0, f1_score=1.0)
    base.update(kw)
    return base


def _open(**kw) -> dict:
    base = dict(question_id="q", correct_answer="mitochondrial dynamics in vivo",
                answer_text="something about mitochondria", is_correct=True,
                exact_match=0.0, f1_score=0.17)
    base.update(kw)
    return base


class TestRecordKind:
    def test_bare_letter_is_mcq(self):
        assert is_mcq_record({"correct_answer": "C"})
        assert is_mcq_record({"correct_answer": " D "})

    def test_prose_is_open_ended(self):
        assert not is_mcq_record({"correct_answer": "the answer is complicated"})

    def test_missing_gold_is_not_mcq(self):
        assert not is_mcq_record({})


class TestRescoring:
    def test_choice_echo_is_rescored_to_correct(self, tmp_path):
        # The real anatomy-022 shape: the model echoes the choice list, then
        # states its answer. The OLD extractor took the leading "A"; the current
        # one takes the stated "B". The log says False; the loader must fix it.
        p = _write(tmp_path, [_mcq(
            correct_answer="B",
            answer_text="A. A B. C C. A D. B  The correct answer is B.",
            is_correct=False,
        )])
        recs, rep = load_run_with_report(p)
        assert recs[0]["is_correct"] is True
        assert rep.rescored == 1
        assert rep.kind == "mcq"

    def test_already_correct_is_not_counted_as_rescored(self, tmp_path):
        p = _write(tmp_path, [_mcq(correct_answer="B", answer_text="The answer is B.",
                                   is_correct=True)])
        _, rep = load_run_with_report(p)
        assert rep.rescored == 0

    def test_wrong_answer_stays_wrong(self, tmp_path):
        p = _write(tmp_path, [_mcq(correct_answer="B", answer_text="The answer is D.",
                                   is_correct=True)])   # log wrongly says True
        recs, rep = load_run_with_report(p)
        assert recs[0]["is_correct"] is False
        assert rep.rescored == 1


class TestOpenEndedVoiding:
    def test_is_correct_is_nulled(self, tmp_path):
        p = _write(tmp_path, [_open(is_correct=True), _open(is_correct=False)])
        recs, rep = load_run_with_report(p)
        assert all(r["is_correct"] is None for r in recs)
        assert rep.kind == "open"
        # Only the non-None one counted as voided.
        assert rep.nulled == 2

    def test_f1_is_preserved(self, tmp_path):
        p = _write(tmp_path, [_open(f1_score=0.42)])
        recs = load_run(p)
        assert recs[0]["f1_score"] == pytest.approx(0.42)


class TestAuditTrail:
    def test_raw_file_is_never_modified(self, tmp_path):
        rec = _mcq(correct_answer="B",
                   answer_text="A. A B. C C. A D. B  The correct answer is B.",
                   is_correct=False)
        p = _write(tmp_path, [rec])
        before = p.read_bytes()
        load_run(p)                      # this flips is_correct in memory
        assert p.read_bytes() == before  # ...but must not touch the file

    def test_empty_file_is_safe(self, tmp_path):
        p = _write(tmp_path, [])
        recs, rep = load_run_with_report(p)
        assert recs == []
        assert rep.kind == "empty"
