"""
policies/p5_gated.py — P5: selective/gated retrieval (core novelty).

The gate decides, per query, whether to retrieve. On RETRIEVE the policy behaves
like P1 (retrieve, answer with context); on SKIP it behaves like P3 (answer
closed-book). The aim is to recover most of P1's accuracy while retrieving on
far fewer queries — the project's central claim.

The gate's full decision payload (per-token entropy, per-member votes, the two
probe drafts) is copied onto PolicyResult.gate_details, which the runner writes
into RunRecord.qvault for a fully auditable log.
"""

from __future__ import annotations

from medrag_adaptive.data.schema import PolicyResult, UnifiedQuestion
from medrag_adaptive.models.prompts import build_closed_book_prompt, build_rag_prompt
from medrag_adaptive.policies.base import Policy


class GatedPolicy(Policy):
    """P5: retrieve only when the gate signals the model is uncertain."""

    name = "p5_gated"

    def answer(self, question: UnifiedQuestion) -> PolicyResult:
        if self.gate is None:
            raise ValueError(f"{self.name} requires a gate")

        decision = self.gate.decide(question, self.llm)

        if decision.retrieve:
            if self.retriever is None:
                raise ValueError(f"{self.name} requires a retriever for the RETRIEVE path")
            chunks = self.retriever.retrieve(question.question_text)
            prompt = build_rag_prompt(
                question.question_text,
                chunks,
                choices=question.choices,
                cite_sources=self.cite_sources,
            )
            text = self.llm.answer(prompt)
            retrieved_chunks = chunks
            retrieval_triggered = True
        else:
            prompt = build_closed_book_prompt(question.question_text, question.choices)
            text = self.llm.answer(prompt)
            retrieved_chunks = []
            retrieval_triggered = False

        return PolicyResult(
            question_id=question.question_id,
            policy_name=self.name,
            gate_name=decision.name,
            gate_decision=decision.decision_str,
            gate_signal_value=decision.signal_value,
            gate_details=decision.details,
            retrieval_triggered=retrieval_triggered,
            retrieved_chunks=retrieved_chunks,
            answer_text=text,
        )
