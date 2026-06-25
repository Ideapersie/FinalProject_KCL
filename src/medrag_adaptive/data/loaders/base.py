"""
data/loaders/base.py — Shared helpers for dataset loaders.

Each loader normalises a source dataset into a list of UnifiedQuestion. The
helpers here cover the parts every loader repeats: capping question count,
reading JSON/JSONL, and tagging risk via the keyword heuristic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from medrag_adaptive.data.schema import UnifiedQuestion


def cap(items: List[Any], max_questions: Optional[int]) -> List[Any]:
    """Return at most `max_questions` items (all of them if None or <= 0)."""
    if max_questions is None or max_questions <= 0:
        return items
    return items[:max_questions]


def read_json(path: str | Path) -> Any:
    """Load a single JSON document."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def read_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    """Yield records from a JSONL file, skipping blank lines."""
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def tag_risk(questions: List[UnifiedQuestion]) -> List[UnifiedQuestion]:
    """Apply the keyword risk heuristic in place, returning the same list."""
    # Imported here to avoid a circular import at module load.
    from medrag_adaptive.data.risk_tagger import classify_risk

    for q in questions:
        q.risk_level = classify_risk(q.question_text, q.specialty)
    return questions
