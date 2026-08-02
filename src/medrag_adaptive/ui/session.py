"""
ui/session.py — run orchestration for the demo app.

Owns the expensive singletons (one LLM backend, one retriever) and answers one
question at a time by running P5 and then P3. Both policies are built through
the existing factory, so the demo exercises the evaluation code path rather
than a parallel reimplementation of it — which is the only thing that makes
"this is what the gate really did" a true statement in a viva.

Knows nothing about HTML or Gradio: it returns a DemoResult, and app.py renders it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from medrag_adaptive.config import ProjectConfig
from medrag_adaptive.data.schema import Chunk, UnifiedQuestion
from medrag_adaptive.gating.base import Gate
from medrag_adaptive.gating.ensemble_gate import EnsembleGate
from medrag_adaptive.gating.entropy_gate import EntropyGate
from medrag_adaptive.models.base import LLMBackend
from medrag_adaptive.policies.factory import build_gate
from medrag_adaptive.policies.p3_closed_book import ClosedBookPolicy
from medrag_adaptive.policies.p5_gated import GatedPolicy
from medrag_adaptive.retrieval.base import Retriever
from medrag_adaptive.ui.attribution import AlignedDraft, align_draft


@dataclass
class DemoResult:
    """Everything app.py needs to render one submitted question."""
    verdict: str                       # "RETRIEVE" | "SKIP"
    gate_name: str
    signal_value: Optional[float]
    threshold: Optional[float]
    gate_details: Dict[str, Any]
    aligned: AlignedDraft
    p5_answer: str
    p3_answer: str
    answers_identical: bool
    chunks: List[Chunk]
    is_multiple_choice: bool
    query: str
    latency_s: float
    notes: List[str] = field(default_factory=list)


class _TopKRetriever(Retriever):
    """Pins top_k for a retriever called by a policy that does not pass one."""

    def __init__(self, inner: Retriever, top_k: int) -> None:
        self._inner = inner
        self._top_k = top_k

    def retrieve(self, query: str, top_k: int = 5) -> List[Chunk]:
        return self._inner.retrieve(query, top_k=self._top_k)

    def close(self) -> None:
        close = getattr(self._inner, "close", None)
        if close:
            close()


def _enable_token_capture(gate: Gate) -> None:
    """
    Switch on observational token capture wherever an entropy gate sits.

    Done here rather than in policies.factory so the factory — which every
    evaluation run goes through — is not touched at all.
    """
    if isinstance(gate, EntropyGate):
        gate.keep_tokens = True
    elif isinstance(gate, EnsembleGate):
        for member in gate.members:
            _enable_token_capture(member)


def _entropy_details(details: Dict[str, Any], gate_name: Optional[str]) -> Dict[str, Any]:
    """
    Pull the entropy payload out of a gate's details.

    Keyed off the gate NAME, not off which keys are present. An abstaining
    entropy gate emits neither `per_token_entropy` nor `draft_tokens`, so key
    sniffing would mistake its payload for an ensemble's and silently read the
    abstention as "available".
    """
    if gate_name == "entropy":
        return details
    if gate_name == "ensemble":
        return (details.get("members") or {}).get("entropy", {})
    return {}


def _threshold_for(cfg: ProjectConfig, gate_name: Optional[str]) -> Optional[float]:
    if gate_name == "entropy":
        return cfg.gate.entropy_threshold
    if gate_name == "margin":
        return cfg.gate.margin_threshold
    if gate_name == "ensemble":
        return float(cfg.gate.ensemble_min_votes)
    return None


class DemoSession:
    """Holds the model and index, and answers one question at a time."""

    def __init__(
        self,
        cfg: ProjectConfig,
        llm: Optional[LLMBackend],
        retriever: Optional[Retriever],
    ) -> None:
        self.cfg = cfg
        self.llm = llm
        self.retriever = retriever
        # The originally-loaded backend, kept so the UI can switch to an API
        # backend and back without paying to reload the GGUF.
        self.local_llm = llm

    # ── run one question ───────────────────────────────────────────

    def answer(
        self,
        question_text: str,
        choices: Optional[Dict[str, str]] = None,
        gate_type: Optional[str] = None,
        entropy_threshold: Optional[float] = None,
        margin_threshold: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> DemoResult:
        if self.llm is None:
            raise ValueError(
                "No model backend is loaded. Configure an API backend in "
                "Settings, or restart without --no-model."
            )
        if not question_text.strip():
            raise ValueError("Enter a question first.")

        cfg = self.cfg.model_copy(deep=True)
        if gate_type:
            cfg.gate.type = gate_type
        if entropy_threshold is not None:
            cfg.gate.entropy_threshold = entropy_threshold
        if margin_threshold is not None:
            cfg.gate.margin_threshold = margin_threshold

        question = UnifiedQuestion(
            question_id="ui",
            question_text=question_text.strip(),
            correct_answer="",                  # no ground truth in a live demo
            dataset_source="ui",
            choices=choices or None,
        )

        gate = build_gate(cfg)
        _enable_token_capture(gate)

        retriever = self.retriever
        if retriever is not None and top_k is not None:
            retriever = _TopKRetriever(retriever, top_k)

        t0 = time.perf_counter()
        p5 = GatedPolicy(
            llm=self.llm,
            retriever=retriever,
            gate=gate,
            cite_sources=cfg.policy.cite_sources,
        ).answer(question)

        notes: List[str] = []
        details = p5.gate_details or {}
        ent = _entropy_details(details, p5.gate_name)
        entropy_available = ent.get("available", True)

        if cfg.gate.type == "entropy" and not entropy_available:
            raise ValueError(
                "The entropy gate cannot run on this backend — it needs "
                "full-vocabulary logits, which this backend does not expose. "
                "Switch to the local GGUF backend, or pick the margin, probe "
                "or ensemble gate."
            )
        if cfg.gate.type == "ensemble" and not entropy_available:
            notes.append(
                "Entropy member abstained (no full-vocabulary logits on this "
                "backend); the ensemble voted on margin and probe only."
            )

        p3 = ClosedBookPolicy(llm=self.llm).answer(question)
        latency = time.perf_counter() - t0

        aligned = align_draft(ent.get("draft_tokens"), ent.get("per_token_entropy"))

        return DemoResult(
            verdict="RETRIEVE" if p5.retrieval_triggered else "SKIP",
            gate_name=p5.gate_name or cfg.gate.type,
            signal_value=p5.gate_signal_value,
            threshold=_threshold_for(cfg, p5.gate_name),
            gate_details=details,
            aligned=aligned,
            p5_answer=p5.answer_text,
            p3_answer=p3.answer_text,
            answers_identical=p5.answer_text.strip() == p3.answer_text.strip(),
            chunks=list(p5.retrieved_chunks),
            is_multiple_choice=question.is_multiple_choice(),
            query=question.question_text,
            latency_s=latency,
            notes=notes,
        )

    def close(self) -> None:
        if self.llm is not None:
            self.llm.close()
        if self.retriever is not None:
            self.retriever.close()
