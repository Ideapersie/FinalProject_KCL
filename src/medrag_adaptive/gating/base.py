"""
gating/base.py — Abstract base class for retrieval gates.

A gate is a training-free signal that decides, per query, whether the model
should retrieve external evidence or answer from parametric knowledge. Each
gate reads from the LLM backend (logits, top-2 logprobs, or draft agreement)
and returns a GateDecision.

The decision maps onto PolicyResult fields:
    gate_name         <- GateDecision.name
    gate_decision     <- "retrieve" if .retrieve else "skip"
    gate_signal_value <- GateDecision.signal_value

GateDecision.details carries richer, gate-specific payloads (per-token entropy
arrays, the two probe answers, per-member votes). The runner copies these into
RunRecord.qvault rather than adding typed schema columns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from medrag_adaptive.data.schema import UnifiedQuestion
from medrag_adaptive.models.base import LLMBackend


@dataclass
class GateDecision:
    """The outcome of a single gate evaluation on one question."""
    name:         str                       # "entropy" | "margin" | "hallucination_probe" | "ensemble"
    retrieve:     bool
    signal_value: Optional[float] = None    # scalar summary (mean entropy, mean margin, ...)
    details:      Dict[str, Any]  = field(default_factory=dict)  # -> RunRecord.qvault

    @property
    def decision_str(self) -> str:
        """The PolicyResult.gate_decision string form."""
        return "retrieve" if self.retrieve else "skip"


class Gate(ABC):
    """Abstract training-free retrieval gate."""

    name: str = "gate"

    @abstractmethod
    def decide(self, question: UnifiedQuestion, llm: LLMBackend) -> GateDecision:
        """Evaluate the gate signal for `question` and return a GateDecision."""
