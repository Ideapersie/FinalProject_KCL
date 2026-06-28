"""
tests/integration/test_open_ended_path.py — open-ended (paragraph Q&A) path.

Phase 3 moves the harness past MIRAGE's all-MCQ subsets onto genuine free-text
answers. The open-ended source is PubMedQA (qiaojin/PubMedQA, pqa_labeled),
saved by the downloader as JSONL with `question` + `long_answer`. These tests
prove the end-to-end open-ended path on the *real* downloaded data (not mocks):

  - the loader yields open-ended UnifiedQuestions (choices=None);
  - scoring uses token-F1, not letter match;
  - the hallucination probe auto-selects its f1_threshold mode (not letter_match)
    when the question has no choices — the only gate that runs open-ended on any
    tier.

If the PubMedQA file is absent (downloader not run), the data-backed tests skip
rather than fail, so the suite stays green on a fresh checkout.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from medrag_adaptive.data.loaders.ragcare_loader import load_ragcare
from medrag_adaptive.data.schema import UnifiedQuestion
from medrag_adaptive.evaluation.scoring import token_f1
from medrag_adaptive.gating.hallucination_probe_gate import HallucinationProbeGate

from tests.conftest import MockLLMBackend

_PUBMEDQA = Path("data/raw/openqa/pubmedqa_labeled.jsonl")
_needs_data = pytest.mark.skipif(
    not _PUBMEDQA.exists(),
    reason="PubMedQA not downloaded (run scripts/download_datasets.py --only ragcare)",
)


# ── loader: real PubMedQA is open-ended ──────────────────────────────

@_needs_data
def test_pubmedqa_loads_as_open_ended():
    qs = load_ragcare(_PUBMEDQA, max_questions=10)
    assert qs, "no questions loaded"
    for q in qs:
        assert q.choices is None
        assert not q.is_multiple_choice()
        assert q.question_text and q.correct_answer  # both populated


# ── scoring: token-F1 path, not letter match ─────────────────────────

@_needs_data
def test_scoring_uses_token_f1_on_real_gold():
    q = load_ragcare(_PUBMEDQA, max_questions=1)[0]
    # An exact echo of the gold paragraph scores F1 == 1.0; a disjoint answer 0.0.
    assert token_f1(q.correct_answer, q.correct_answer) == pytest.approx(1.0)
    assert token_f1("completely unrelated tokens xyz", q.correct_answer) < 0.2


# ── gate: probe selects f1_threshold mode for open-ended ──────────────

def _open_question() -> UnifiedQuestion:
    return UnifiedQuestion(
        question_id="oq1",
        question_text="Do mitochondria play a role in remodelling lace plant leaves?",
        correct_answer="Yes, mitochondrial dynamics accompany programmed cell death.",
        dataset_source="pubmedqa",
        choices=None,
    )


def test_probe_uses_f1_mode_when_no_choices():
    # Two identical drafts → F1 == 1.0 ≥ threshold → agreement → SKIP.
    llm = MockLLMBackend(draft_text="Mitochondria drive programmed cell death here.")
    gate = HallucinationProbeGate(agreement_mode="letter_match", f1_threshold=0.7)
    d = gate.decide(_open_question(), llm)
    assert d.details["mode"] == "f1_threshold"   # auto-switched, not letter_match
    assert d.retrieve is False                   # identical drafts agree
    assert d.signal_value == pytest.approx(1.0)


def test_probe_retrieves_when_open_drafts_disagree(monkeypatch):
    # Force the two probe framings to return different text → low F1 → RETRIEVE.
    llm = MockLLMBackend()
    texts = iter(["mitochondria yes cell death", "no evidence completely different answer"])
    monkeypatch.setattr(llm, "draft", lambda prompt, max_tokens=48: (next(texts), None))
    gate = HallucinationProbeGate(agreement_mode="letter_match", f1_threshold=0.7)
    d = gate.decide(_open_question(), llm)
    assert d.details["mode"] == "f1_threshold"
    assert d.retrieve is True
