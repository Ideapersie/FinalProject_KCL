"""
gating/margin_gate.py — Logit-margin retrieval gate (core novelty).

Measures the model's decisiveness from the gap between its top-1 and top-2
token probabilities. A small mean margin means the model is torn between
alternatives → RETRIEVE.

    margin_t = p_t^(1) - p_t^(2)          (per generated token t)
    M̄        = (1/N) Σ_t margin_t          (gate signal)
    retrieve  ⟺  M̄ < threshold

Unlike the entropy gate, this uses get_top2_logprobs() — the completion API's
top-k logprobs rather than the raw logit buffer — so it works even when
logits_all=False is unavailable... in practice both logit-based gates are
disabled on the low tier, but the margin gate's dependency is the lighter one.
Basis: TARG margin signal.

Input shape note: get_top2_logprobs() returns one dict per generated token,
mapping token-string -> log-probability, with (at least) the top-2 entries.
We take the two largest log-probs per dict, exponentiate to probabilities, and
difference them. This parser is validated against a real llama-cpp draft call
before the prototype run (the mock and the real API agree on this shape).
"""

from __future__ import annotations

import math
from typing import Dict, List

from medrag_adaptive.data.schema import UnifiedQuestion
from medrag_adaptive.gating.base import Gate, GateDecision
from medrag_adaptive.models.base import LLMBackend
from medrag_adaptive.models.prompts import build_draft_prompt


def _token_margin(token_logprobs: Dict[str, float]) -> float:
    """Top-1 minus top-2 probability for one token's logprob dict."""
    if not token_logprobs:
        return 0.0
    top = sorted(token_logprobs.values(), reverse=True)
    p1 = math.exp(top[0])
    p2 = math.exp(top[1]) if len(top) > 1 else 0.0
    return p1 - p2


class MarginGate(Gate):
    """Retrieve when the mean top-1/top-2 probability margin is small."""

    name = "margin"

    def __init__(self, threshold: float = 0.3, draft_max_tokens: int = 48) -> None:
        self.threshold = threshold
        self.draft_max_tokens = draft_max_tokens

    def decide(self, question: UnifiedQuestion, llm: LLMBackend) -> GateDecision:
        prompt = build_draft_prompt(question.question_text, question.choices)
        _text, top_logprobs = llm.get_top2_logprobs(
            prompt, max_tokens=self.draft_max_tokens
        )

        if not top_logprobs:
            return GateDecision(
                name=self.name,
                retrieve=True,
                signal_value=None,
                details={"available": False, "reason": "no logprobs returned"},
            )

        margins: List[float] = [_token_margin(t) for t in top_logprobs]
        mean_margin = float(sum(margins) / len(margins))

        return GateDecision(
            name=self.name,
            retrieve=mean_margin < self.threshold,
            signal_value=mean_margin,
            details={
                "available": True,
                "mean_margin": mean_margin,
                "threshold": self.threshold,
            },
        )
