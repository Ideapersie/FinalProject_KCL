"""
tests/unit/test_p5_gated.py — Tests for the gated policy and policy factory.
"""

from __future__ import annotations

import pytest

from medrag_adaptive.data.schema import UnifiedQuestion
from medrag_adaptive.gating.entropy_gate import EntropyGate
from medrag_adaptive.gating.ensemble_gate import EnsembleGate
from medrag_adaptive.policies.p5_gated import GatedPolicy
from medrag_adaptive.policies.p1_always_retrieve import AlwaysRetrievePolicy
from medrag_adaptive.policies.p3_closed_book import ClosedBookPolicy
from medrag_adaptive.policies.factory import build_policy
from medrag_adaptive.config import load_config

from tests.conftest import MockLLMBackend, MockRetriever


@pytest.fixture
def mcq_question() -> UnifiedQuestion:
    return UnifiedQuestion(
        question_id="q1",
        question_text="Which nerve innervates the diaphragm?",
        correct_answer="A",
        dataset_source="fixture",
        choices={"A": "Phrenic", "B": "Vagus", "C": "Intercostal", "D": "Accessory"},
    )


# ─────────────────────────────────────────────────────────────────
# GatedPolicy
# ─────────────────────────────────────────────────────────────────

class TestGatedPolicy:
    def test_high_confidence_does_not_retrieve(self, mcq_question):
        policy = GatedPolicy(
            llm=MockLLMBackend(confidence="high"),
            retriever=MockRetriever(),
            gate=EntropyGate(threshold=2.5),
        )
        result = policy.answer(mcq_question)
        assert result.retrieval_triggered is False
        assert result.retrieved_chunks == []
        assert result.gate_decision == "skip"

    def test_low_confidence_retrieves(self, mcq_question):
        policy = GatedPolicy(
            llm=MockLLMBackend(confidence="low"),
            retriever=MockRetriever(),
            gate=EntropyGate(threshold=2.5),
        )
        result = policy.answer(mcq_question)
        assert result.retrieval_triggered is True
        assert len(result.retrieved_chunks) > 0
        assert result.gate_decision == "retrieve"

    def test_gate_fields_populated(self, mcq_question):
        policy = GatedPolicy(
            llm=MockLLMBackend(confidence="high"),
            retriever=MockRetriever(),
            gate=EntropyGate(threshold=2.5),
        )
        result = policy.answer(mcq_question)
        assert result.gate_name == "entropy"
        assert result.gate_signal_value is not None

    def test_details_exposed_for_logging(self, mcq_question):
        """Gate details (per-token entropy) reach the result for qvault logging."""
        policy = GatedPolicy(
            llm=MockLLMBackend(confidence="low"),
            retriever=MockRetriever(),
            gate=EntropyGate(threshold=2.5),
        )
        result = policy.answer(mcq_question)
        assert "per_token_entropy" in result.gate_details

    def test_requires_retriever(self, mcq_question):
        policy = GatedPolicy(
            llm=MockLLMBackend(confidence="low"),
            retriever=None,
            gate=EntropyGate(threshold=2.5),
        )
        with pytest.raises(ValueError):
            policy.answer(mcq_question)


# ─────────────────────────────────────────────────────────────────
# PolicyFactory
# ─────────────────────────────────────────────────────────────────

class TestPolicyFactory:
    def test_builds_closed_book(self):
        cfg = load_config(
            base="configs/base.yaml",
            policy="configs/policies/p3_closed_book.yaml",
        )
        policy = build_policy(cfg, MockLLMBackend(), retriever=None)
        assert isinstance(policy, ClosedBookPolicy)

    def test_builds_always_retrieve(self):
        cfg = load_config(
            base="configs/base.yaml",
            policy="configs/policies/p1_always_retrieve.yaml",
        )
        policy = build_policy(cfg, MockLLMBackend(), retriever=MockRetriever())
        assert isinstance(policy, AlwaysRetrievePolicy)

    def test_builds_gated_ensemble(self):
        cfg = load_config(
            base="configs/base.yaml",
            policy="configs/policies/p5_gated_entropy.yaml",
        )
        # base.yaml sets gate.type=ensemble with 3 members.
        policy = build_policy(cfg, MockLLMBackend(), retriever=MockRetriever())
        assert isinstance(policy, GatedPolicy)
        assert isinstance(policy.gate, EnsembleGate)
        assert len(policy.gate.members) == 3

    def test_low_tier_is_probe_only(self):
        cfg = load_config(
            base="configs/base.yaml",
            hardware="configs/hardware_low.yaml",
            policy="configs/policies/p5_gated_entropy.yaml",
        )
        # hardware_low overrides gate.type to hallucination_probe.
        policy = build_policy(cfg, MockLLMBackend(), retriever=MockRetriever())
        assert isinstance(policy, GatedPolicy)
        # Single probe gate, not an ensemble.
        assert policy.gate.name == "hallucination_probe"
