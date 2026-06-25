"""
policies/p3_closed_book.py — P3 baseline: never retrieve.

Answers purely from the model's parametric knowledge. This is the speed/energy
floor baseline: no retrieval, no gate. Used to measure how far parametric-only
accuracy falls short of the always-retrieve ceiling (P1).
"""

from __future__ import annotations

from medrag_adaptive.data.schema import PolicyResult, UnifiedQuestion
from medrag_adaptive.models.prompts import build_closed_book_prompt
from medrag_adaptive.policies.base import Policy


class ClosedBookPolicy(Policy):
    """P3: closed-book — answer without any retrieval."""

    name = "p3_closed_book"

    def answer(self, question: UnifiedQuestion) -> PolicyResult:
        prompt = build_closed_book_prompt(question.question_text, question.choices)
        text = self.llm.answer(prompt)
        return PolicyResult(
            question_id=question.question_id,
            policy_name=self.name,
            retrieval_triggered=False,
            answer_text=text,
        )
