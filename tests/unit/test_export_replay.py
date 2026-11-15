"""Tests for scripts/export_replay.py: the static replay demo builder."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("export_replay", ROOT / "scripts" / "export_replay.py")
er = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(er)


def _rec(qid, correct, answer="Answer: A.", retrieved=False, **qvault):
    return {"question_id": qid, "is_correct": correct, "answer_text": answer,
            "retrieval_triggered": retrieved, "gate_decision": "retrieve" if retrieved else "skip",
            "qvault": qvault}


@pytest.mark.parametrize("closed,retrieve,expected", [
    (True, False, "broke"), (False, True, "fixed"), (True, True, "both_right"), (False, False, "both_wrong"),
])
def test_category(closed, retrieve, expected):
    assert er.category(closed, retrieve) == expected


def test_flip_counts_only_uses_shared_ids():
    closed = {"a": _rec("a", True), "b": _rec("b", False), "c": _rec("c", True)}
    retrieve = {"a": _rec("a", False), "b": _rec("b", True), "c": _rec("c", True)}
    gated = {"a": _rec("a", True, retrieved=True), "b": _rec("b", False)}  # "c" missing
    s = er.flip_counts(closed, retrieve, gated)
    assert (s["n"], s["broke"], s["fixed"]) == (2, 1, 1)
    assert s["acc_gated"] == 0.5 and s["gated_retrieval_rate"] == 0.5


def test_build_records_joins_question_text_and_truncates_chunks():
    bench = {"anatomy-000": {"question": "Q?", "options": {"A": "x", "B": "y"}, "answer": "A"}}
    chunks = [{"source": "pubmed", "title": "T", "score": 0.5, "text": "z" * 50}] * 5
    rec = er.build_records(
        closed={"anatomy-000": _rec("anatomy-000", True)},
        retrieve={"anatomy-000": _rec("anatomy-000", False, "The context does not say.", True, retrieved_chunks=chunks)},
        gated={"anatomy-000": _rec("anatomy-000", True)},
        bench=bench, chunk_chars=10, top_k=2,
    )[0]
    assert rec["question"] == "Q?" and rec["gold"] == "A" and rec["category"] == "broke"
    assert rec["retrieve"]["letter"] is None  # model chose no option
    assert len(rec["chunks"]) == 2 and rec["chunks"][0]["text"] == "z" * 10 + "..."


def test_render_escapes_script_close():
    out = er.render("<script>" + er.PLACEHOLDER + "</script>", {"q": "</script><b>"})
    assert out.count("</script>") == 1
    assert json.loads(out[len("<script>"):-len("</script>")]) == {"q": "</script><b>"}


def test_render_requires_placeholder():
    with pytest.raises(ValueError):
        er.render("<html></html>", {})


@pytest.mark.skipif(not (ROOT / "results" / "raw_logs" / "p5_medcorp_mcq.jsonl").exists(), reason="run logs not present")
def test_headline_numbers_match_logs():
    """The dissertation's headline (46 broken, 13 fixed at 3B) must come out of the logs."""
    payload = er.build_payload(er.LOGS, ROOT / "data" / "raw" / "mirage" / "benchmark.json", 400, 3)
    s = payload["summary"]
    assert (s["n"], s["broke"], s["fixed"]) == (200, 46, 13)
    assert len(payload["questions"]) == 200
