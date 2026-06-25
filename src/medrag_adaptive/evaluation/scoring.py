"""
evaluation/scoring.py — Answer scoring primitives.

Two scorers, matching the two question types:

    score_mcq  — multiple choice: extract the predicted option letter from the
                 model's free text and compare to the gold letter. Returns
                 (is_correct, exact_match) where exact_match is 1.0 / 0.0.
    token_f1   — open-ended: token-level F1 between prediction and gold answer.

token_f1 is also reused by the hallucination-probe gate (Week 2) to measure
agreement between two open-ended drafts.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional, Tuple

# Matches a leading option letter: "B", "B.", "B)", "(B)", "Answer: B", "The answer is B".
_LETTER_RE = re.compile(r"\b([A-E])\b")
_LEAD_RE = re.compile(r"^\s*\(?([A-E])\)?[\.\):\s]")


def extract_letter(text: str) -> Optional[str]:
    """Extract the predicted MCQ option letter from free-text, or None."""
    if not text:
        return None
    lead = _LEAD_RE.match(text)
    if lead:
        return lead.group(1).upper()
    # Common phrasings: "the answer is C", "Answer: C"
    m = re.search(r"answer\s*(?:is|:)?\s*\(?([A-E])\)?\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Fallback: first standalone capital letter A–E.
    m = _LETTER_RE.search(text.upper())
    return m.group(1) if m else None


def score_mcq(pred_text: str, correct_letter: str) -> Tuple[bool, float]:
    """Score an MCQ prediction. Returns (is_correct, exact_match in {0.0, 1.0})."""
    pred = extract_letter(pred_text)
    is_correct = pred is not None and pred == correct_letter.strip().upper()
    return is_correct, (1.0 if is_correct else 0.0)


def _normalise(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return text.split()


def token_f1(pred: str, gold: str) -> float:
    """Token-level F1 between two strings (SQuAD-style). 0.0 if either empty."""
    pred_toks = _normalise(pred)
    gold_toks = _normalise(gold)
    if not pred_toks or not gold_toks:
        return 0.0
    common = Counter(pred_toks) & Counter(gold_toks)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_toks)
    recall = overlap / len(gold_toks)
    return 2 * precision * recall / (precision + recall)
