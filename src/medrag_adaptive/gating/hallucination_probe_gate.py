"""
gating/hallucination_probe_gate.py — Draft-consistency gate (core novelty).

Replaces the deprecated verbalized-confidence gate. Instead of asking the model
to self-assess (which small models do poorly), it tests whether the model gives
the *same* answer when the question is framed two different ways. Inconsistency
across framings means the answer is not robustly held → RETRIEVE.

    draft_A = model(probe_prompt_A(question))
    draft_B = model(probe_prompt_B(question))

    MCQ  (letter_match):   retrieve ⟺ letter(A) ≠ letter(B)  (or either unparseable)
    Open (f1_threshold):   retrieve ⟺ token_f1(A, B) < threshold

Text-only: it calls draft() and reads the returned text, never the logit
buffer, so it is the *only* gate available on the low tier (logits_all=False).
It costs two draft calls per query; that overhead is reported explicitly.
Basis: self-consistency (Wang et al., 2023), used as a binary gate signal.
"""

from __future__ import annotations

from typing import Literal

from medrag_adaptive.data.schema import UnifiedQuestion
from medrag_adaptive.evaluation.scoring import extract_letter, token_f1
from medrag_adaptive.gating.base import Gate, GateDecision
from medrag_adaptive.models.base import LLMBackend
from medrag_adaptive.models.prompts import build_probe_prompt_a, build_probe_prompt_b


class HallucinationProbeGate(Gate):
    """Retrieve when two differently-framed drafts disagree."""

    name = "hallucination_probe"

    def __init__(
        self,
        agreement_mode: Literal["letter_match", "f1_threshold"] = "letter_match",
        f1_threshold: float = 0.7,
        max_tokens: int = 48,
    ) -> None:
        self.agreement_mode = agreement_mode
        self.f1_threshold = f1_threshold
        self.max_tokens = max_tokens

    def decide(self, question: UnifiedQuestion, llm: LLMBackend) -> GateDecision:
        prompt_a = build_probe_prompt_a(question.question_text, question.choices)
        prompt_b = build_probe_prompt_b(question.question_text, question.choices)
        draft_a, _ = llm.draft(prompt_a, max_tokens=self.max_tokens)
        draft_b, _ = llm.draft(prompt_b, max_tokens=self.max_tokens)

        # Choose agreement mode by question type unless explicitly overridden:
        # MCQ → letter match; open-ended → F1. Honour an explicit f1_threshold
        # request even on MCQ if the caller set it.
        use_letter = (
            self.agreement_mode == "letter_match" and question.is_multiple_choice()
        )

        if use_letter:
            la, lb = extract_letter(draft_a), extract_letter(draft_b)
            agreement = la is not None and lb is not None and la == lb
            signal = 1.0 if agreement else 0.0
            extra = {"letter_a": la, "letter_b": lb}
        else:
            f1 = token_f1(draft_a, draft_b)
            agreement = f1 >= self.f1_threshold
            signal = f1
            extra = {"f1": f1, "f1_threshold": self.f1_threshold}

        return GateDecision(
            name=self.name,
            retrieve=not agreement,
            signal_value=signal,
            details={
                "available": True,
                "agreement": agreement,
                "draft_a": draft_a,
                "draft_b": draft_b,
                "mode": "letter_match" if use_letter else "f1_threshold",
                **extra,
            },
        )
