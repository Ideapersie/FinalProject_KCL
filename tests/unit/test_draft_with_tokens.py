"""
tests/unit/test_draft_with_tokens.py — token-string plumbing for the demo UI.

The demo's entropy heatmap needs the draft's token STRINGS aligned to the
per-token entropies the gate already computes. These tests pin the two
guarantees that make that safe: a backend that never heard of token capture
still satisfies the interface, and EntropyGate at keep_tokens=False takes the
identical code path it always has.
"""

from __future__ import annotations

from medrag_adaptive.gating.entropy_gate import EntropyGate

from tests.conftest import MockLLMBackend


def test_default_draft_with_tokens_returns_none_tokens():
    """A backend that does not override token capture still satisfies the interface."""
    llm = MockLLMBackend(confidence="high")
    text, logits, tokens = llm.draft_with_tokens("some prompt", max_tokens=8)
    assert isinstance(text, str)
    assert logits is not None
    assert tokens is None


def test_entropy_gate_default_does_not_request_tokens(low_risk_question):
    gate = EntropyGate(threshold=0.7)
    decision = gate.decide(low_risk_question, MockLLMBackend(confidence="high"))
    assert "draft_tokens" not in decision.details


def test_entropy_gate_keep_tokens_adds_key(low_risk_question):
    gate = EntropyGate(threshold=0.7, keep_tokens=True)
    decision = gate.decide(low_risk_question, MockLLMBackend(confidence="high"))
    assert "draft_tokens" in decision.details      # None on a mock, but present


def test_keep_tokens_does_not_change_the_signal(low_risk_question):
    """The flag must be observational only — same H̄, same decision."""
    llm = MockLLMBackend(confidence="low")
    plain = EntropyGate(threshold=0.7).decide(low_risk_question, llm)
    kept = EntropyGate(threshold=0.7, keep_tokens=True).decide(low_risk_question, llm)
    assert plain.signal_value == kept.signal_value
    assert plain.retrieve == kept.retrieve
