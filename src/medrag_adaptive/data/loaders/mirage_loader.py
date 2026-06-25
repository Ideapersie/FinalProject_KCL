"""
data/loaders/mirage_loader.py — MIRAGE benchmark loader.

MIRAGE (from the MedRAG toolkit) ships a single benchmark.json that is a dict
keyed by subset name (mmlu, medqa, medmcqa, pubmedqa, bioasq). Each subset is
a dict mapping question-id -> record:

    {
      "mmlu": {
        "mmlu_0": {
          "question": "....",
          "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
          "answer": "C"
        },
        ...
      },
      ...
    }

We also accept a flat {qid: record} file (a single pre-extracted subset).
Each record becomes a multiple-choice UnifiedQuestion; risk is tagged by the
keyword heuristic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from medrag_adaptive.data.loaders.base import cap, read_json, tag_risk
from medrag_adaptive.data.schema import UnifiedQuestion

# Subsets that exist in the standard MIRAGE benchmark.
MIRAGE_SUBSETS = ("mmlu", "medqa", "medmcqa", "pubmedqa", "bioasq")


def _record_to_question(qid: str, rec: Dict[str, Any], subset: str) -> UnifiedQuestion:
    options = rec.get("options") or rec.get("choices")
    choices: Optional[Dict[str, str]] = (
        {str(k): str(v) for k, v in options.items()} if options else None
    )
    return UnifiedQuestion(
        question_id=qid,
        question_text=rec["question"],
        correct_answer=str(rec["answer"]),
        dataset_source=f"mirage_{subset}",
        choices=choices,
        metadata={"subset": subset},
    )


def _is_subset_keyed(data: Dict[str, Any]) -> bool:
    """True if top-level keys look like MIRAGE subset names."""
    return any(k in MIRAGE_SUBSETS for k in data)


def load_mirage(
    path: str | Path,
    subsets: Optional[List[str]] = None,
    max_questions: Optional[int] = None,
) -> List[UnifiedQuestion]:
    """
    Load MIRAGE questions into a list of UnifiedQuestion.

    Args:
        path:          Path to benchmark.json (subset-keyed or flat qid-keyed).
        subsets:       Restrict to these subset names (default: all present).
        max_questions: Cap the total returned (after concatenating subsets).
    """
    data = read_json(path)
    questions: List[UnifiedQuestion] = []

    if _is_subset_keyed(data):
        wanted = subsets or [s for s in MIRAGE_SUBSETS if s in data]
        for subset in wanted:
            for qid, rec in data.get(subset, {}).items():
                questions.append(_record_to_question(qid, rec, subset))
    else:
        # Flat {qid: record} — treat as a single unnamed subset.
        subset = subsets[0] if subsets else "mmlu"
        for qid, rec in data.items():
            questions.append(_record_to_question(qid, rec, subset))

    tag_risk(questions)
    return cap(questions, max_questions)
