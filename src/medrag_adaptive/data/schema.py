"""
schema.py — Central data contracts for the project.

Two primary dataclasses:
  - UnifiedQuestion : a single medical QA question, normalised from any source dataset.
  - RunRecord       : the result of running one policy on one question, including
                      profiling metrics and ground-truth scoring.

All other modules import from here. This file has no project-level imports,
so it can be imported anywhere without circular dependency risk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional


# ─────────────────────────────────────────────────────────────────
# Chunk — a retrieved document snippet
# ─────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """A single retrieved passage from the retrieval corpus."""
    chunk_id:   str
    source:     str           # e.g. "statpearls", "textbooks", "bnf"
    title:      str
    text:       str
    score:      float = 0.0   # retrieval score (BM25 / cosine / RRF)

    def to_context_string(self) -> str:
        """Format chunk for inclusion in an LLM prompt context block."""
        return f"[{self.source.upper()}] {self.title}\n{self.text}"


# ─────────────────────────────────────────────────────────────────
# Citation — a source cited in the answer
# ─────────────────────────────────────────────────────────────────

@dataclass
class Citation:
    """A source citation extracted from a policy answer."""
    chunk_id:   str
    source:     str
    title:      str


# ─────────────────────────────────────────────────────────────────
# UnifiedQuestion — normalised QA question
# ─────────────────────────────────────────────────────────────────

@dataclass
class UnifiedQuestion:
    """
    A single medical QA question normalised from any source dataset.

    For multiple-choice questions, `choices` is a dict like:
        {"A": "Penicillin", "B": "Amoxicillin", ...}
    and `correct_answer` is the letter key ("A", "B", etc.).

    For open-ended questions, `choices` is None and `correct_answer`
    is the expected answer string.
    """
    question_id:     str
    question_text:   str
    correct_answer:  str
    dataset_source:  str    # "mirage_mmlu", "mirage_medqa", "ragcare", "acute_care", etc.
    risk_level:      Literal["low", "medium", "high"] = "medium"
    choices:         Optional[Dict[str, str]] = None    # {letter: text} for MCQ
    specialty:       Optional[str] = None               # "cardiology", "emergency", etc.
    metadata:        Dict[str, Any] = field(default_factory=dict)

    def format_choices(self) -> str:
        """Return choices as a formatted string for prompt inclusion."""
        if not self.choices:
            return ""
        return "\n".join(f"{letter}. {text}" for letter, text in self.choices.items())

    def is_multiple_choice(self) -> bool:
        return self.choices is not None


# ─────────────────────────────────────────────────────────────────
# PolicyResult — raw output from a policy (before scoring)
# ─────────────────────────────────────────────────────────────────

@dataclass
class PolicyResult:
    """
    The output of running a retrieval policy on one question.
    Produced by a Policy.answer() call. Does not include ground truth.
    """
    question_id:        str
    policy_name:        str
    gate_name:          Optional[str]   = None   # "entropy" | "margin" | "verbalized" | None
    gate_decision:      Optional[str]   = None   # "retrieve" | "skip" | None
    gate_signal_value:  Optional[float] = None   # raw entropy or margin value
    retrieval_triggered: bool           = False
    retrieved_chunks:   List[Chunk]     = field(default_factory=list)
    answer_text:        str             = ""
    citations:          List[Citation]  = field(default_factory=list)
    # Profiling fields — populated by profiler.py wrapper
    latency_ns:         int             = 0
    energy_kwh:         float           = 0.0
    peak_memory_mb:     float           = 0.0


# ─────────────────────────────────────────────────────────────────
# RunRecord — PolicyResult + ground-truth scoring (for logging)
# ─────────────────────────────────────────────────────────────────

@dataclass
class RunRecord:
    """
    A complete per-query record written to the JSONL audit log.
    Merges PolicyResult with ground-truth scoring and experiment metadata.
    """
    # Identity
    question_id:        str
    dataset_source:     str
    risk_level:         str
    specialty:          Optional[str]

    # Policy
    policy_name:        str
    gate_name:          Optional[str]
    gate_decision:      Optional[str]
    gate_signal_value:  Optional[float]
    retrieval_triggered: bool

    # Answer
    answer_text:        str
    correct_answer:     str
    is_correct:         bool
    exact_match:        float           # 0.0 or 1.0
    f1_score:           float

    # Citations
    citations:          List[Dict[str, str]] = field(default_factory=list)
    citation_precision: Optional[float]      = None
    citation_recall:    Optional[float]      = None

    # Profiling
    latency_ns:         int   = 0
    energy_kwh:         float = 0.0
    peak_memory_mb:     float = 0.0

    # Experiment metadata
    model_name:         str = ""
    hardware_tier:      str = "medium"
    timestamp:          str = ""

    # qVault shared schema fields (cross-variant compatibility)
    qvault: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunRecord":
        return cls(**d)


# ─────────────────────────────────────────────────────────────────
# JSONL I/O helpers
# ─────────────────────────────────────────────────────────────────

def append_record(record: RunRecord, path: str | Path) -> None:
    """Append a single RunRecord to a JSONL file (creates file if absent)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(record.to_json() + "\n")


def load_records(path: str | Path) -> List[RunRecord]:
    """Load all RunRecords from a JSONL file."""
    p = Path(path)
    if not p.exists():
        return []
    records = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(RunRecord.from_dict(json.loads(line)))
    return records


def load_logged_ids(path: str | Path) -> set[str]:
    """Return the set of question_ids already written to a JSONL log (for resume support)."""
    p = Path(path)
    if not p.exists():
        return set()
    ids: set[str] = set()
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    ids.add(json.loads(line)["question_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return ids
