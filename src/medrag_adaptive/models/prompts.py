"""
models/prompts.py — All prompt templates, centralised in one place.

Keeping templates here (rather than scattered in policy files) lets you:
  - Version-control prompt changes together
  - Reuse templates across policies
  - Swap templates for ablation studies without touching policy logic

CHAT FORMATS (added for the multi-model scaling study)
------------------------------------------------------
Every instruction-tuned model is fine-tuned against ONE chat markup. Llama and
Mistral use `[INST] ... [/INST]`; Qwen2.5 uses `<|im_start|>role ... <|im_end|>`.
Handing a model the wrong markup does not error — it silently degrades, because
the model never saw that framing in training. Measuring "Qwen with Llama's
markup" and calling it "Qwen" would quietly corrupt the entire scaling curve, so
the wrapper is chosen per model rather than hard-coded.

The design separates the two concerns:

    prompt BODY     what we ask (model-independent; one definition, shared)
    chat WRAPPER    how this model expects to be addressed (model-specific)

`set_chat_format("qwen")` switches the wrapper globally; the bodies never change,
so all models are asked exactly the same question. The default is "llama", which
reproduces the original prompts byte-for-byte — every existing result and test
therefore remains valid.

Usage:
    from medrag_adaptive.models.prompts import build_closed_book_prompt, set_chat_format
    set_chat_format("qwen")          # once, at startup, from config
    prompt = build_closed_book_prompt(question, choices)
"""

from __future__ import annotations

from typing import Dict, Optional


# ─────────────────────────────────────────────────────────────────
# Chat formats
# ─────────────────────────────────────────────────────────────────
#
# Each entry wraps a system instruction + user turn in the markup the model was
# instruction-tuned on, and opens the assistant turn so the model continues it.

CHAT_FORMATS: Dict[str, Dict[str, str]] = {
    # Llama-3.x / Mistral-style. This is what the 3B baseline was run with; the
    # exact strings are preserved so existing logs remain reproducible.
    "llama": {
        "template": "[INST] {system}\n\n{user} [/INST]",
    },
    # Qwen2.5 ChatML. The trailing "<|im_start|>assistant\n" opens the reply turn.
    "qwen": {
        "template": (
            "<|im_start|>system\n{system}<|im_end|>\n"
            "<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
    },
    # No markup at all — for base (non-instruct) models, and useful for testing
    # that the body text itself is well-formed.
    "plain": {
        "template": "{system}\n\n{user}\n",
    },
}

_chat_format = "llama"          # default preserves the original behaviour


def set_chat_format(name: str) -> None:
    """Select the chat markup for the model being run. Call once, from config."""
    if name not in CHAT_FORMATS:
        raise ValueError(
            f"unknown chat_format {name!r}; known: {sorted(CHAT_FORMATS)}"
        )
    global _chat_format
    _chat_format = name


def get_chat_format() -> str:
    return _chat_format


def _wrap(system: str, user: str) -> str:
    """Apply the active model's chat markup to a system+user pair."""
    return CHAT_FORMATS[_chat_format]["template"].format(
        system=system.strip(), user=user.strip()
    )


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

CLOSED_BOOK_SYSTEM = """\
You are a medical assistant with expert clinical knowledge.
Answer the following medical question accurately and concisely."""

CLOSED_BOOK_USER = """\
Question: {question}
{choices}
{instruction}"""


def build_closed_book_prompt(
    question: str,
    choices: Optional[Dict[str, str]] = None,
) -> str:
    return _wrap(
        CLOSED_BOOK_SYSTEM,
        CLOSED_BOOK_USER.format(
            question=question,
            choices=_format_choices(choices),
            instruction=_mcq_instruction(choices is not None),
        ),
    )


# ─────────────────────────────────────────────────────────────────
# P1 / P2 / P4 / P5 — RAG prompt (with retrieved context)
# ─────────────────────────────────────────────────────────────────

RAG_SYSTEM = """\
You are a medical assistant. Use the provided context to answer the question.
Only use information from the context; if the context does not contain the answer,
state that you are uncertain rather than guessing."""

RAG_USER = """\
Context:
{context}

Question: {question}
{choices}
{instruction}"""

RAG_USER_WITH_CITATION = """\
Context:
{context}

Question: {question}
{choices}
{instruction}

After your answer, list the sources you used in this exact format:
SOURCES: [source_1_title], [source_2_title]"""


def build_rag_prompt(
    question: str,
    context_chunks: list,          # List[Chunk] from schema.py
    choices: Optional[Dict[str, str]] = None,
    cite_sources: bool = False,
) -> str:
    context_text = "\n\n".join(chunk.to_context_string() for chunk in context_chunks)
    user = RAG_USER_WITH_CITATION if cite_sources else RAG_USER
    return _wrap(
        RAG_SYSTEM,
        user.format(
            context=context_text,
            question=question,
            choices=_format_choices(choices),
            instruction=_mcq_instruction(choices is not None),
        ),
    )


# ─────────────────────────────────────────────────────────────────
# P5 — Gate: verbalized confidence check
# DEPRECATED: replaced by the hallucination-probe gate (build_probe_prompt_a/b).
# Retained only for the ablation study comparing probe vs verbalized.
# ─────────────────────────────────────────────────────────────────

VERBALIZED_CONFIDENCE_SYSTEM = """\
I am about to answer a medical question. Before answering, I need to assess \
my confidence in answering it accurately WITHOUT consulting external medical sources \
(textbooks, guidelines, drug formularies)."""

VERBALIZED_CONFIDENCE_USER = """\
Question: {question}

How confident are you in answering this question accurately without external sources?
Respond with exactly one word: HIGH, MEDIUM, or LOW."""


def build_verbalized_confidence_prompt(question: str) -> str:
    return _wrap(
        VERBALIZED_CONFIDENCE_SYSTEM,
        VERBALIZED_CONFIDENCE_USER.format(question=question),
    )


# ─────────────────────────────────────────────────────────────────
# P5 — Gate: draft prompt (for entropy / margin signal)
# ─────────────────────────────────────────────────────────────────

DRAFT_SYSTEM = "You are a medical assistant. Answer the following question briefly."

DRAFT_USER = """\
Question: {question}
{choices}
Answer:"""


def build_draft_prompt(
    question: str,
    choices: Optional[Dict[str, str]] = None,
) -> str:
    """The draft the entropy and margin gates measure.

    This prompt's continuation IS the gate signal: its per-token logit
    distribution is what entropy and margin are computed from. It therefore has
    to be phrased in the model's native chat markup, or the measured uncertainty
    is partly the model's confusion about the framing rather than about the
    medicine — which would silently invalidate every gate decision.
    """
    return _wrap(
        DRAFT_SYSTEM,
        DRAFT_USER.format(
            question=question,
            choices=_format_choices(choices),
        ),
    )


# ─────────────────────────────────────────────────────────────────
# P5 — Gate: hallucination probe (two framings of the same question)
# ─────────────────────────────────────────────────────────────────
# The probe gate generates two short drafts under different framings. If the
# extracted answers disagree, the model is unstable on this query → retrieve.
# Reuses the standard draft/answer machinery; no special confidence prompt.

PROBE_A_SYSTEM = "You are a medical assistant. Answer the following medical question briefly."

PROBE_A_USER = """\
Question: {question}
{choices}
{instruction}"""

PROBE_B_SYSTEM = "What is the correct answer to the following medical question? Be concise."

PROBE_B_USER = """\
{question}
{choices}
{instruction}"""


def build_probe_prompt_a(
    question: str,
    choices: Optional[Dict[str, str]] = None,
) -> str:
    return _wrap(
        PROBE_A_SYSTEM,
        PROBE_A_USER.format(
            question=question,
            choices=_format_choices(choices),
            instruction=_mcq_instruction(choices is not None),
        ),
    )


def build_probe_prompt_b(
    question: str,
    choices: Optional[Dict[str, str]] = None,
) -> str:
    return _wrap(
        PROBE_B_SYSTEM,
        PROBE_B_USER.format(
            question=question,
            choices=_format_choices(choices),
            instruction=_mcq_instruction(choices is not None),
        ),
    )
