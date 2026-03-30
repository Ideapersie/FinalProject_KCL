"""
models/prompts.py — All prompt templates, centralised in one place.

Keeping templates here (rather than scattered in policy files) lets you:
  - Version-control prompt changes together
  - Reuse templates across policies
  - Swap templates for ablation studies without touching policy logic

Usage:
    from medrag_adaptive.models.prompts import build_closed_book_prompt
    prompt = build_closed_book_prompt(question, choices_text)
"""

from __future__ import annotations

from typing import Dict, Optional


# ─────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────

def _format_choices(choices: Optional[Dict[str, str]]) -> str:
    if not choices:
        return ""
    lines = "\n".join(f"{k}. {v}" for k, v in choices.items())
    return f"\nOptions:\n{lines}\n"


def _mcq_instruction(has_choices: bool) -> str:
    if has_choices:
        return "Answer with the letter only (A, B, C, D, or E)."
    return "Answer concisely and accurately."


# ─────────────────────────────────────────────────────────────────
# P3 — Closed-Book
# ─────────────────────────────────────────────────────────────────

CLOSED_BOOK_TEMPLATE = """\
[INST] You are a medical assistant with expert clinical knowledge.
Answer the following medical question accurately and concisely.

Question: {question}
{choices}
{instruction} [/INST]"""


def build_closed_book_prompt(
    question: str,
    choices: Optional[Dict[str, str]] = None,
) -> str:
    return CLOSED_BOOK_TEMPLATE.format(
        question=question,
        choices=_format_choices(choices),
        instruction=_mcq_instruction(choices is not None),
    )


# ─────────────────────────────────────────────────────────────────
# P1 / P2 / P4 / P5 — RAG prompt (with retrieved context)
# ─────────────────────────────────────────────────────────────────

RAG_TEMPLATE = """\
[INST] You are a medical assistant. Use the provided context to answer the question.
Only use information from the context; if the context does not contain the answer,
state that you are uncertain rather than guessing.

Context:
{context}

Question: {question}
{choices}
{instruction} [/INST]"""

RAG_TEMPLATE_WITH_CITATION = """\
[INST] You are a medical assistant. Use the provided context to answer the question.
Only use information from the context; if the context does not contain the answer,
state that you are uncertain rather than guessing.

Context:
{context}

Question: {question}
{choices}
{instruction}

After your answer, list the sources you used in this exact format:
SOURCES: [source_1_title], [source_2_title] [/INST]"""


def build_rag_prompt(
    question: str,
    context_chunks: list,          # List[Chunk] from schema.py
    choices: Optional[Dict[str, str]] = None,
    cite_sources: bool = False,
) -> str:
    context_text = "\n\n".join(chunk.to_context_string() for chunk in context_chunks)
    template = RAG_TEMPLATE_WITH_CITATION if cite_sources else RAG_TEMPLATE
    return template.format(
        context=context_text,
        question=question,
        choices=_format_choices(choices),
        instruction=_mcq_instruction(choices is not None),
    )


# ─────────────────────────────────────────────────────────────────
# P5 — Gate: verbalized confidence check
# ─────────────────────────────────────────────────────────────────

VERBALIZED_CONFIDENCE_TEMPLATE = """\
[INST] I am about to answer a medical question. Before answering, I need to assess \
my confidence in answering it accurately WITHOUT consulting external medical sources \
(textbooks, guidelines, drug formularies).

Question: {question}

How confident are you in answering this question accurately without external sources?
Respond with exactly one word: HIGH, MEDIUM, or LOW. [/INST]"""


def build_verbalized_confidence_prompt(question: str) -> str:
    return VERBALIZED_CONFIDENCE_TEMPLATE.format(question=question)


# ─────────────────────────────────────────────────────────────────
# P5 — Gate: draft prompt (for entropy / margin signal)
# ─────────────────────────────────────────────────────────────────

DRAFT_TEMPLATE = """\
[INST] You are a medical assistant. Answer the following question briefly.

Question: {question}
{choices}
Answer: [/INST]"""


def build_draft_prompt(
    question: str,
    choices: Optional[Dict[str, str]] = None,
) -> str:
    return DRAFT_TEMPLATE.format(
        question=question,
        choices=_format_choices(choices),
    )
