"""
data/loaders/ragcare_loader.py — RAGCare-QA loader.

RAGCare-QA is an open-ended medical QA set (from HuggingFace). Records carry a
question and a gold answer string (field names vary across exports), and may
carry a specialty/category. Open-ended means `choices` is None and
`correct_answer` is the gold string; scoring uses token-F1 rather than letter
match.

Accepts JSONL (one record per line) or a JSON list. Field names are resolved
flexibly so a HuggingFace `to_json` export loads without manual remapping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from medrag_adaptive.data.loaders.base import cap, read_json, read_jsonl, tag_risk
from medrag_adaptive.data.schema import UnifiedQuestion

_QUESTION_KEYS = ("question", "query", "input", "prompt")
_ANSWER_KEYS = ("answer", "gold", "output", "response", "target")
_ID_KEYS = ("id", "qid", "question_id", "_id")
_SPECIALTY_KEYS = ("specialty", "category", "topic", "domain")


def _first(rec: Dict[str, Any], keys: tuple[str, ...]) -> Optional[Any]:
    for k in keys:
        if k in rec and rec[k] not in (None, ""):
            return rec[k]
    return None


def _record_to_question(rec: Dict[str, Any], idx: int) -> UnifiedQuestion:
    qid = _first(rec, _ID_KEYS)
    return UnifiedQuestion(
        question_id=str(qid) if qid is not None else f"ragcare_{idx}",
        question_text=str(_first(rec, _QUESTION_KEYS) or ""),
        correct_answer=str(_first(rec, _ANSWER_KEYS) or ""),
        dataset_source="ragcare",
        choices=None,
        specialty=_first(rec, _SPECIALTY_KEYS),
    )


def load_ragcare(
    path: str | Path,
    max_questions: Optional[int] = None,
) -> List[UnifiedQuestion]:
    """Load RAGCare-QA into a list of open-ended UnifiedQuestion."""
    path = Path(path)
    if path.suffix == ".jsonl":
        records: List[Dict[str, Any]] = list(read_jsonl(path))
    else:
        data = read_json(path)
        records = data if isinstance(data, list) else list(data.values())

    questions = [_record_to_question(rec, i) for i, rec in enumerate(records)]
    tag_risk(questions)
    return cap(questions, max_questions)
