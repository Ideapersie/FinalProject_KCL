"""
tests/unit/test_gates.py — Unit tests for the P5 retrieval gates.

Every gate is tested against MockLLMBackend so no real model is needed. The
mock's `confidence` setting drives the synthetic signals:
  - confidence="high" → peaked logits (low entropy, big margin) → SKIP
  - confidence="low"  → uniform logits (high entropy, small margin) → RETRIEVE
"""

from __future__ import annotations

import numpy as np
import pytest

from medrag_adaptive.data.schema import UnifiedQuestion
from medrag_adaptive.gating.base import GateDecision
from medrag_adaptive.gating.entropy_gate import EntropyGate
from medrag_adaptive.gating.margin_gate import MarginGate
from medrag_adaptive.gating.hallucination_probe_gate import HallucinationProbeGate
from medrag_adaptive.gating.ensemble_gate import EnsembleGate

from tests.conftest import MockLLMBackend, VOCAB_SIZE, N_DRAFT_TOKENS


@pytest.fixture
def mcq_question() -> UnifiedQuestion:
    return UnifiedQuestion(
        question_id="q1",
        question_text="Which nerve innervates the diaphragm?",
        correct_answer="A",
        dataset_source="fixture",
        choices={"A": "Phrenic", "B": "Vagus", "C": "Intercostal", "D": "Accessory"},
    )


@pytest.fixture
def open_question() -> UnifiedQuestion:
    return UnifiedQuestion(
        question_id="q2",
        question_text="What is the mechanism of action of metformin?",
        correct_answer="inhibits hepatic gluconeogenesis",
        dataset_source="fixture",
        choices=None,
    )


# ─────────────────────────────────────────────────────────────────
# Entropy gate
# ─────────────────────────────────────────────────────────────────

class TestEntropyGate:
    def test_high_confidence_skips(self, mcq_question):
        gate = EntropyGate(threshold=2.5)
        decision = gate.decide(mcq_question, MockLLMBackend(confidence="high"))
        assert decision.retrieve is False
        assert decision.name == "entropy"

    def test_low_confidence_retrieves(self, mcq_question):
        # Uniform logits over VOCAB_SIZE → entropy ≈ ln(128) ≈ 4.85 > 2.5.
        gate = EntropyGate(threshold=2.5)
        decision = gate.decide(mcq_question, MockLLMBackend(confidence="low"))
        assert decision.retrieve is True

    def test_per_token_entropy_recorded(self, mcq_question):
        gate = EntropyGate(threshold=2.5, draft_max_tokens=N_DRAFT_TOKENS)
        decision = gate.decide(mcq_question, MockLLMBackend(confidence="low"))
        per_token = decision.details["per_token_entropy"]
        assert len(per_token) == N_DRAFT_TOKENS
        # Uniform distribution → each ≈ ln(VOCAB_SIZE).
        assert all(abs(h - np.log(VOCAB_SIZE)) < 0.1 for h in per_token)

    def test_signal_value_is_mean_entropy(self, mcq_question):
        gate = EntropyGate(threshold=2.5)
        decision = gate.decide(mcq_question, MockLLMBackend(confidence="high"))
        assert decision.signal_value is not None
        assert decision.signal_value < 0.5  # peaked → near-zero entropy

    def test_no_logits_abstains(self, mcq_question):
        """When logits are unavailable (low tier), the gate abstains."""
        gate = EntropyGate(threshold=2.5)
        llm = MockLLMBackend(confidence="high")
        # Force draft() to return None logits, as on logits_all=False.
        llm.draft = lambda prompt, max_tokens=48: ("A", None)
        decision = gate.decide(mcq_question, llm)
        assert decision.details["available"] is False


# ─────────────────────────────────────────────────────────────────
# Margin gate
# ─────────────────────────────────────────────────────────────────

class TestMarginGate:
    def test_high_confidence_skips(self, mcq_question):
        # Big top-1/top-2 gap → large margin → SKIP.
        gate = MarginGate(threshold=0.3)
        decision = gate.decide(mcq_question, MockLLMBackend(confidence="high"))
        assert decision.retrieve is False
        assert decision.name == "margin"

    def test_low_confidence_retrieves(self, mcq_question):
        # Tiny gap → small margin → RETRIEVE.
        gate = MarginGate(threshold=0.3)
        decision = gate.decide(mcq_question, MockLLMBackend(confidence="low"))
        assert decision.retrieve is True

    def test_signal_value_recorded(self, mcq_question):
        gate = MarginGate(threshold=0.3)
        decision = gate.decide(mcq_question, MockLLMBackend(confidence="high"))
        assert decision.signal_value is not None
        assert 0.0 <= decision.signal_value <= 1.0


# ─────────────────────────────────────────────────────────────────
# Hallucination probe gate
# ─────────────────────────────────────────────────────────────────

class TestHallucinationProbeGate:
    def test_agreeing_drafts_skip(self, mcq_question):
        # Mock returns same draft for both framings → agree → SKIP.
        gate = HallucinationProbeGate(agreement_mode="letter_match")
        llm = MockLLMBackend(draft_text="A. Phrenic nerve")
        decision = gate.decide(mcq_question, llm)
        assert decision.retrieve is False
        assert decision.name == "hallucination_probe"
        assert decision.details["agreement"] is True

    def test_disagreeing_drafts_retrieve(self, mcq_question):
        # A backend whose two probe calls return different letters.
        gate = HallucinationProbeGate(agreement_mode="letter_match")
        llm = MockLLMBackend()
        answers = iter(["A. Phrenic", "C. Intercostal"])
        llm.draft = lambda prompt, max_tokens=48: (next(answers), None)
        decision = gate.decide(mcq_question, llm)
        assert decision.retrieve is True
        assert decision.details["agreement"] is False

    def test_open_ended_high_f1_skips(self, open_question):
        gate = HallucinationProbeGate(agreement_mode="f1_threshold", f1_threshold=0.7)
        llm = MockLLMBackend()
        same = "metformin inhibits hepatic gluconeogenesis via AMPK"
        llm.draft = lambda prompt, max_tokens=48: (same, None)
        decision = gate.decide(open_question, llm)
        assert decision.retrieve is False

    def test_open_ended_low_f1_retrieves(self, open_question):
        gate = HallucinationProbeGate(agreement_mode="f1_threshold", f1_threshold=0.7)
        llm = MockLLMBackend()
        answers = iter([
            "metformin inhibits hepatic gluconeogenesis",
            "it stimulates pancreatic insulin secretion strongly",
        ])
        llm.draft = lambda prompt, max_tokens=48: (next(answers), None)
        decision = gate.decide(open_question, llm)
        assert decision.retrieve is True

    def test_works_without_logits(self, mcq_question):
        """Probe is text-only, so it works on the low tier (logits_all=False)."""
        gate = HallucinationProbeGate(agreement_mode="letter_match")
        llm = MockLLMBackend(draft_text="A")
        llm.draft = lambda prompt, max_tokens=48: ("A", None)
        decision = gate.decide(mcq_question, llm)
        assert decision.details["available"] is True


# ─────────────────────────────────────────────────────────────────
# Ensemble gate
# ─────────────────────────────────────────────────────────────────

class _StubGate:
    """A gate returning a fixed decision, for ensemble vote tests."""

    def __init__(self, name, retrieve, available=True):
        self.name = name
        self._retrieve = retrieve
        self._available = available

    def decide(self, question, llm):
        return GateDecision(
            name=self.name,
            retrieve=self._retrieve,
            details={"available": self._available},
        )


class TestEnsembleGate:
    def test_two_of_three_retrieve(self, mcq_question):
        gate = EnsembleGate(
            members=[
                _StubGate("entropy", True),
                _StubGate("margin", True),
                _StubGate("hallucination_probe", False),
            ],
            min_votes=2,
        )
        decision = gate.decide(mcq_question, MockLLMBackend())
        assert decision.retrieve is True
        assert decision.name == "ensemble"
        assert decision.details["votes"]["entropy"] == "retrieve"

    def test_one_of_three_skips(self, mcq_question):
        gate = EnsembleGate(
            members=[
                _StubGate("entropy", True),
                _StubGate("margin", False),
                _StubGate("hallucination_probe", False),
            ],
            min_votes=2,
        )
        decision = gate.decide(mcq_question, MockLLMBackend())
        assert decision.retrieve is False

    def test_abstaining_members_excluded(self, mcq_question):
        """An unavailable member's vote does not count; degraded mode noted."""
        gate = EnsembleGate(
            members=[
                _StubGate("entropy", True, available=False),
                _StubGate("margin", True, available=False),
                _StubGate("hallucination_probe", True, available=True),
            ],
            min_votes=2,
        )
        decision = gate.decide(mcq_question, MockLLMBackend())
        # Only the probe is available; with min_votes=2 unreachable, fall back to
        # "retrieve if any available member retrieves".
        assert decision.retrieve is True
        assert decision.details["degraded"] is True
        assert decision.details["members_available"] == 1
