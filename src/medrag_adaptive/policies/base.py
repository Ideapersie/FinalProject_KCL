"""
policies/base.py — Abstract base class for retrieval policies.

A policy answers one UnifiedQuestion and returns a PolicyResult. It composes
the three building blocks:

    LLMBackend  — generates the answer (always)
    Retriever   — fetches evidence  (P1/P2/P4 always; P5 conditionally; P3 never)
    Gate        — decides retrieve-or-skip per query (P5 only)

Policies fill the answer/retrieval/gate fields of PolicyResult. The profiler
wrapper fills latency_ns / energy_kwh / peak_memory_mb around the answer() call.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from medrag_adaptive.data.schema import PolicyResult, UnifiedQuestion
from medrag_adaptive.gating.base import Gate
from medrag_adaptive.models.base import LLMBackend
from medrag_adaptive.retrieval.base import Retriever


class Policy(ABC):
    """Abstract retrieval policy mapping a question to a PolicyResult."""

    name: str = "policy"

    def __init__(
        self,
        llm: LLMBackend,
        retriever: Optional[Retriever] = None,
        gate: Optional[Gate] = None,
        cite_sources: bool = False,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.gate = gate
        self.cite_sources = cite_sources

    @abstractmethod
    def answer(self, question: UnifiedQuestion) -> PolicyResult:
        """Run the policy on one question and return its raw result."""
