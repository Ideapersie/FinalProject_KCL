"""
policies/p1_always_retrieve.py — P1 baseline: always retrieve.

Retrieves evidence for every query and answers with it in context. This is the
accuracy-ceiling baseline against which gated policies (P5) are measured: P5
aims to recover most of P1's accuracy while retrieving on far fewer queries.

P2 (always-retrieve + citation) reuses this class with cite_sources=True; the
citation extraction lives in p2_always_retrieve_cite.py (Week 2).
"""

from __future__ import annotations

from medrag_adaptive.data.schema import PolicyResult, UnifiedQuestion
from medrag_adaptive.models.prompts import build_rag_prompt
from medrag_adaptive.policies.base import Policy


class AlwaysRetrievePolicy(Policy):
    """P1: always retrieve, then answer with retrieved context."""

    name = "p1_always_retrieve"

    def answer(self, question: UnifiedQuestion) -> PolicyResult:
        if self.retriever is None:
            raise ValueError(f"{self.name} requires a retriever")

        chunks = self.retriever.retrieve(question.question_text)
        prompt = build_rag_prompt(
            question.question_text,
            chunks,
            choices=question.choices,
            cite_sources=self.cite_sources,
        )
        text = self.llm.answer(prompt)
        return PolicyResult(
            question_id=question.question_id,
            policy_name=self.name,
            retrieval_triggered=True,
            retrieved_chunks=chunks,
            answer_text=text,
        )
