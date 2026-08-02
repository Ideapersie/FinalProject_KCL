"""
tests/unit/test_ui_session.py — the demo's run orchestration.

Runs entirely against the conftest mocks: no GGUF, no index, no network.
"""

from __future__ import annotations

import pytest

from medrag_adaptive.config import ProjectConfig
from medrag_adaptive.ui.session import DemoSession

from tests.conftest import MockLLMBackend, MockRetriever


def _cfg(gate_type: str = "entropy", entropy_threshold: float = 0.7) -> ProjectConfig:
    cfg = ProjectConfig()
    cfg.policy.name = "p5_gated"
    cfg.policy.retrieval_mode = "bm25"
    cfg.gate.type = gate_type
    cfg.gate.entropy_threshold = entropy_threshold
    return cfg


def test_low_confidence_retrieves_and_populates_chunks():
    session = DemoSession(_cfg(), MockLLMBackend(confidence="low"), MockRetriever())
    result = session.answer("Which nerve innervates the diaphragm?", choices=None)
    assert result.verdict == "RETRIEVE"
    assert result.chunks


def test_high_confidence_skips_and_leaves_chunks_empty():
    session = DemoSession(_cfg(), MockLLMBackend(confidence="high"), MockRetriever())
    result = session.answer("Which nerve innervates the diaphragm?", choices=None)
    assert result.verdict == "SKIP"
    assert result.chunks == []


def test_skip_path_flags_identical_answers():
    """On SKIP, P5 and P3 issue the same prompt — the UI must not imply a comparison."""
    session = DemoSession(_cfg(), MockLLMBackend(confidence="high"), MockRetriever())
    result = session.answer("Which nerve innervates the diaphragm?", choices=None)
    assert result.answers_identical is True


def test_entropy_gate_requests_tokens_for_the_ui():
    """MockLLMBackend inherits the default draft_with_tokens, so tokens is None
    and every entropy value is dropped — which is exactly what alignment must
    report rather than silently mislabel."""
    session = DemoSession(_cfg(), MockLLMBackend(confidence="low"), MockRetriever())
    result = session.answer("q", choices=None)
    assert "draft_tokens" in result.gate_details
    assert result.aligned.dropped_entropies > 0


def test_choices_produce_an_mcq_question():
    session = DemoSession(_cfg(), MockLLMBackend(confidence="high"), MockRetriever())
    result = session.answer("q", choices={"A": "one", "B": "two"})
    assert result.is_multiple_choice is True


def test_blank_question_is_refused():
    session = DemoSession(_cfg(), MockLLMBackend(), MockRetriever())
    with pytest.raises(ValueError, match="Enter a question"):
        session.answer("   ", choices=None)


def test_sole_entropy_gate_without_logits_is_refused():
    """An entropy-only gate on a logit-less backend reports a fallback default,
    not a measurement. Refuse rather than render it as a decision."""

    class NoLogitsBackend(MockLLMBackend):
        def draft(self, prompt, max_tokens=48):
            return self._draft_text, None

    session = DemoSession(_cfg(), NoLogitsBackend(), MockRetriever())
    with pytest.raises(ValueError, match="entropy gate cannot run"):
        session.answer("q", choices=None)


def test_top_k_override_is_passed_to_the_retriever():
    session = DemoSession(_cfg(), MockLLMBackend(confidence="low"), MockRetriever())
    result = session.answer("q", choices=None, top_k=2)
    assert len(result.chunks) == 2


def test_threshold_override_flips_the_decision():
    """Same backend, different tau: the slider must actually reach the gate."""
    llm = MockLLMBackend(confidence="low")     # near-uniform logits → high entropy
    session = DemoSession(_cfg(), llm, MockRetriever())
    assert session.answer("q", entropy_threshold=0.1).verdict == "RETRIEVE"
    assert session.answer("q", entropy_threshold=99.0).verdict == "SKIP"


def test_ensemble_records_member_votes():
    session = DemoSession(_cfg(gate_type="ensemble"),
                          MockLLMBackend(confidence="low"), MockRetriever())
    result = session.answer("q", choices=None)
    assert set(result.gate_details["votes"]) == {
        "entropy", "margin", "hallucination_probe"
    }
