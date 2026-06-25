"""
data/risk_tagger.py — Keyword heuristic clinical-risk classifier.

Assigns each question a clinical risk level (low / medium / high) used by the
safety-envelope analysis. This is a deliberately simple, training-free,
auditable heuristic — the report documents its rules and its limits.

    HIGH   — drug interactions, dosing, overdose, contraindications, acute
             emergencies: a wrong answer can directly harm a patient.
    MEDIUM — diagnosis, management, treatment selection: clinically important
             but less immediately dangerous.
    LOW    — anatomy, physiology, definitions, mechanisms: factual recall.

Precedence is HIGH > MEDIUM > LOW: any high-risk cue wins. Default is MEDIUM
(the conservative middle) when nothing matches.
"""

from __future__ import annotations

import re
from typing import List, Literal, Optional

RiskLevel = Literal["low", "medium", "high"]

# Word-boundary keyword groups. Order of checks encodes precedence.
_HIGH_TERMS: List[str] = [
    "interaction", "interact", "co-administ", "coadminist", "contraindicat",
    "dose", "dosing", "dosage", "overdose", "toxic", "toxicity", "poisoning",
    "anticoagul", "warfarin", "insulin", "chemotherap", "sedation",
    "emergenc", "acute", "resuscitat", "anaphyla", "arrest", "shock",
    "fatal", "lethal", "life-threatening", "adverse",
]
_MEDIUM_TERMS: List[str] = [
    "diagnos", "diagnosis", "manage", "management", "treat", "treatment",
    "therap", "first-line", "first line", "prognos", "differential",
    "indicat", "prescrib", "recommend",
]
_LOW_TERMS: List[str] = [
    "anatom", "innervat", "physiolog", "mechanism of action", "definition",
    "define", "structure", "located", "located in", "what is the function",
    "histolog", "embryolog",
]

_HIGH_SPECIALTIES = {"emergency", "pharmacology", "toxicology", "anesthesiology"}


def _matches(text: str, terms: List[str]) -> bool:
    return any(re.search(r"\b" + re.escape(t), text) for t in terms)


def classify_risk(question_text: str, specialty: Optional[str] = None) -> RiskLevel:
    """Classify a question's clinical risk from its text and optional specialty."""
    text = question_text.lower()
    spec = (specialty or "").lower()

    if _matches(text, _HIGH_TERMS) or spec in _HIGH_SPECIALTIES:
        return "high"
    if _matches(text, _LOW_TERMS):
        return "low"
    if _matches(text, _MEDIUM_TERMS):
        return "medium"
    return "medium"
