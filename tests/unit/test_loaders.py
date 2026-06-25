"""Unit tests for dataset loaders and the risk tagger."""

from pathlib import Path

import pytest

from medrag_adaptive.data.loaders.mirage_loader import load_mirage
from medrag_adaptive.data.loaders.ragcare_loader import load_ragcare
from medrag_adaptive.data.risk_tagger import classify_risk

FIXTURES = Path(__file__).parent.parent / "fixtures"
MIRAGE = FIXTURES / "mirage_sample.json"
RAGCARE = FIXTURES / "ragcare_sample.jsonl"


# ── MIRAGE loader ──────────────────────────────────────────────────

def test_mirage_loads_all_subsets():
    qs = load_mirage(MIRAGE)
    assert len(qs) == 3                       # 2 mmlu + 1 medqa
    ids = {q.question_id for q in qs}
    assert {"mmlu_0", "mmlu_1", "medqa_0"} == ids


def test_mirage_is_multiple_choice_with_letter_answer():
    qs = load_mirage(MIRAGE)
    q0 = next(q for q in qs if q.question_id == "mmlu_0")
    assert q0.is_multiple_choice()
    assert q0.choices["B"] == "Phrenic nerve"
    assert q0.correct_answer == "B"
    assert q0.dataset_source == "mirage_mmlu"


def test_mirage_subset_filter():
    qs = load_mirage(MIRAGE, subsets=["medqa"])
    assert [q.question_id for q in qs] == ["medqa_0"]


def test_mirage_max_questions_caps():
    qs = load_mirage(MIRAGE, max_questions=2)
    assert len(qs) == 2


def test_mirage_risk_tagged():
    qs = load_mirage(MIRAGE)
    risk = {q.question_id: q.risk_level for q in qs}
    assert risk["mmlu_0"] == "low"            # "innervates" -> anatomy
    assert risk["medqa_0"] == "high"          # "interaction" + "bleeding"


# ── RAGCare loader ─────────────────────────────────────────────────

def test_ragcare_open_ended():
    qs = load_ragcare(RAGCARE)
    assert len(qs) == 2
    assert all(not q.is_multiple_choice() for q in qs)
    assert qs[0].choices is None


def test_ragcare_field_resolution():
    qs = load_ragcare(RAGCARE)
    assert qs[0].question_id == "rc_0"
    assert "cyclooxygenase" in qs[0].correct_answer
    # second record uses query/gold/category aliases + synthesised id
    assert qs[1].question_id == "ragcare_1"
    assert qs[1].specialty == "emergency"
    assert qs[1].dataset_source == "ragcare"


# ── Risk tagger ────────────────────────────────────────────────────

def test_risk_high_on_interaction(high_risk_question):
    assert classify_risk(high_risk_question.question_text,
                         high_risk_question.specialty) == "high"


def test_risk_low_on_anatomy(low_risk_question):
    assert classify_risk(low_risk_question.question_text,
                        low_risk_question.specialty) == "low"


def test_risk_defaults_medium():
    assert classify_risk("Compare the prognosis of two conditions.") == "medium"


@pytest.mark.parametrize("text,expected", [
    ("What is the dose of metformin?", "high"),
    ("What is the anatomy of the heart?", "low"),
    ("What is the first-line treatment for hypertension?", "medium"),
])
def test_risk_keyword_table(text, expected):
    assert classify_risk(text) == expected
