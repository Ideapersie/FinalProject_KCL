"""
tests/unit/test_retrieval_factory.py — retriever factory + P2/P4 policy tests.

The retriever factory's vector/hybrid branches need a real FAISS index, so those
are marked slow. The bm25 and none branches, and the P2/P4 policies (which use a
MockRetriever), run fast.
"""

from __future__ import annotations

import pytest

from medrag_adaptive.config import load_config
from medrag_adaptive.data.schema import Chunk, UnifiedQuestion
from medrag_adaptive.retrieval.factory import build_retriever
from medrag_adaptive.policies.factory import build_policy
from medrag_adaptive.policies.p2_always_retrieve_cite import (
    AlwaysRetrieveCitePolicy, _parse_citations,
)
from medrag_adaptive.policies.p4_hybrid import HybridRetrievalPolicy

from tests.conftest import MockLLMBackend, MockRetriever


@pytest.fixture
def mcq_question() -> UnifiedQuestion:
    return UnifiedQuestion(
        question_id="q1", question_text="Which nerve innervates the diaphragm?",
        correct_answer="A", dataset_source="fixture",
        choices={"A": "Phrenic", "B": "Vagus", "C": "Intercostal", "D": "Accessory"},
    )


# ─────────────────────────────────────────────────────────────────
# Retriever factory
# ─────────────────────────────────────────────────────────────────

class TestRetrieverFactory:
    def test_none_mode_returns_none(self):
        cfg = load_config(base="configs/base.yaml",
                          policy="configs/policies/p3_closed_book.yaml")
        assert build_retriever(cfg) is None

    def test_bm25_mode_loads_pickle(self):
        cfg = load_config(base="configs/base.yaml",
                          policy="configs/policies/p1_always_retrieve.yaml")
        cfg.policy.retrieval_mode = "bm25"
        r = build_retriever(cfg, bm25_index="indexes/bm25_pilot.pkl")
        assert r.retrieve("phrenic nerve", top_k=2)

    def test_unknown_mode_raises(self):
        cfg = load_config(base="configs/base.yaml",
                          policy="configs/policies/p1_always_retrieve.yaml")
        cfg.policy.retrieval_mode = "telepathy"
        with pytest.raises(ValueError):
            build_retriever(cfg)


# ─────────────────────────────────────────────────────────────────
# P2 citation parsing + policy
# ─────────────────────────────────────────────────────────────────

class TestP2Citations:
    def test_parse_matches_retrieved_titles(self):
        chunks = [
            Chunk(chunk_id="c1", source="bnf", title="Warfarin Interactions", text="..."),
            Chunk(chunk_id="c2", source="statpearls", title="Aspirin", text="..."),
        ]
        answer = "The answer is B.\nSOURCES: Warfarin Interactions"
        cites = _parse_citations(answer, chunks)
        assert len(cites) == 1
        assert cites[0].chunk_id == "c1"

    def test_no_sources_line_yields_no_citations(self):
        chunks = [Chunk(chunk_id="c1", source="x", title="T", text="...")]
        assert _parse_citations("Just an answer, no sources.", chunks) == []

    def test_invented_source_not_counted(self):
        chunks = [Chunk(chunk_id="c1", source="x", title="Real Title", text="...")]
        cites = _parse_citations("SOURCES: Some Made Up Reference", chunks)
        assert cites == []

    def test_policy_populates_citations(self, mcq_question):
        # MockRetriever returns the mock corpus; craft an answer citing one title.
        retr = MockRetriever()
        titles = [c.title for c in retr.retrieve("q")]
        llm = MockLLMBackend()
        llm.answer = lambda prompt, max_tokens=None: f"Answer: A.\nSOURCES: {titles[0]}"
        policy = AlwaysRetrieveCitePolicy(llm=llm, retriever=retr, cite_sources=True)
        result = policy.answer(mcq_question)
        assert len(result.citations) == 1


# ─────────────────────────────────────────────────────────────────
# Policy factory builds P2 / P4
# ─────────────────────────────────────────────────────────────────

class TestPolicyFactoryNewPolicies:
    def test_builds_p2(self):
        cfg = load_config(base="configs/base.yaml",
                          policy="configs/policies/p2_always_retrieve_cite.yaml")
        p = build_policy(cfg, MockLLMBackend(), retriever=MockRetriever())
        assert isinstance(p, AlwaysRetrieveCitePolicy)

    def test_builds_p4(self):
        cfg = load_config(base="configs/base.yaml",
                          policy="configs/policies/p4_hybrid.yaml")
        p = build_policy(cfg, MockLLMBackend(), retriever=MockRetriever())
        assert isinstance(p, HybridRetrievalPolicy)
