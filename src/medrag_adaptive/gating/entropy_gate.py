"""
gating/entropy_gate.py — Token-entropy retrieval gate (core novelty).

Generates a short draft answer without retrieval, then measures the model's
uncertainty from the full-vocabulary logit distribution. Mean token entropy
above a threshold means the model is unsure → RETRIEVE.

    H(p_t) = -Σ_v p_t(v) · log p_t(v)        (per generated token t)
    H̄     = (1/N) Σ_t H(p_t)                 (gate signal)
    retrieve  ⟺  H̄ > threshold

The per-token entropy array H(p_t) is also returned in GateDecision.details
under "per_token_entropy" — this is the Token-Level Entropy Attribution
contribution: it shows *which* tokens of the draft the model is uncertain
about, not merely that it is uncertain overall.

Requires logits_all=True (the raw logit buffer). On the low tier
(logits_all=False) draft() returns None for the logits and the gate abstains,
so the ensemble can drop this member. Basis: TARG (arXiv 2025).
"""

from __future__ import annotations

import numpy as np

from medrag_adaptive.data.schema import UnifiedQuestion
from medrag_adaptive.gating.base import Gate, GateDecision
from medrag_adaptive.models.base import LLMBackend
from medrag_adaptive.models.prompts import build_draft_prompt


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax, numerically stabilised by max-subtraction."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _row_entropy(probs: np.ndarray) -> np.ndarray:
    """Per-row Shannon entropy in nats. Adds epsilon to avoid log(0)."""
    return -np.sum(probs * np.log(probs + 1e-12), axis=-1)


class EntropyGate(Gate):
    """Retrieve when mean draft-token entropy exceeds a threshold."""

    name = "entropy"

    def __init__(
        self,
        threshold: float = 2.5,
        draft_max_tokens: int = 48,
        keep_tokens: bool = False,
    ) -> None:
        self.threshold = threshold
        self.draft_max_tokens = draft_max_tokens
        # Observational only: when True the gate also records the draft's token
        # strings in details, for the demo UI's token-level heatmap. It never
        # changes the signal or the decision. Default False so every evaluation
        # run takes the identical code path it always has.
        self.keep_tokens = keep_tokens

    def decide(self, question: UnifiedQuestion, llm: LLMBackend) -> GateDecision:
        prompt = build_draft_prompt(question.question_text, question.choices)
        tokens = None
        if self.keep_tokens:
            _text, logits, tokens = llm.draft_with_tokens(
                prompt, max_tokens=self.draft_max_tokens
            )
        else:
            _text, logits = llm.draft(prompt, max_tokens=self.draft_max_tokens)

        if logits is None:
            # logits_all=False (low tier): this gate cannot run. Abstain.
            return GateDecision(
                name=self.name,
                retrieve=True,            # safe default if ever read directly
                signal_value=None,
                details={"available": False,
                         "reason": "logits unavailable (logits_all=False)"},
            )

        per_token = _row_entropy(_softmax(np.asarray(logits, dtype=np.float64)))
        mean_entropy = float(per_token.mean())

        return GateDecision(
            name=self.name,
            retrieve=mean_entropy > self.threshold,
            signal_value=mean_entropy,
            details={
                "available": True,
                "mean_entropy": mean_entropy,
                "threshold": self.threshold,
                "per_token_entropy": [float(h) for h in per_token],
                **({"draft_tokens": tokens} if self.keep_tokens else {}),
            },
        )
