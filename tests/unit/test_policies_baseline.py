"""Unit tests for the P1 and P3 baseline policies (mock-backed)."""

import pytest

from medrag_adaptive.policies.p1_always_retrieve import AlwaysRetrievePolicy
from medrag_adaptive.policies.p3_closed_book import ClosedBookPolicy


def test_p3_does_not_retrieve(mock_llm_high, low_risk_question):
    policy = ClosedBookPolicy(llm=mock_llm_high)
    result = policy.answer(low_risk_question)
    assert result.retrieval_triggered is False
    assert result.retrieved_chunks == []
    assert result.gate_name is None
    assert result.policy_name == "p3_closed_book"
    assert result.answer_text  # non-empty


def test_p1_retrieves_and_keeps_chunks(mock_llm_high, mock_retriever, low_risk_question):
    policy = AlwaysRetrievePolicy(llm=mock_llm_high, retriever=mock_retriever)
    result = policy.answer(low_risk_question)
    assert result.retrieval_triggered is True
    assert len(result.retrieved_chunks) > 0
    assert result.policy_name == "p1_always_retrieve"
    assert result.answer_text


def test_p1_without_retriever_raises(mock_llm_high, low_risk_question):
    policy = AlwaysRetrievePolicy(llm=mock_llm_high)
    with pytest.raises(ValueError):
        policy.answer(low_risk_question)
