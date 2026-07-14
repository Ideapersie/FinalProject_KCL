"""
tests/unit/test_prompts_chat_format.py — per-model chat markup.

Why this matters more than it looks: an instruction-tuned model is fine-tuned
against exactly one chat markup. Give Qwen2.5 Llama's `[INST]` markup and it does
not error — it silently answers worse, because it never saw that framing in
training. A scaling study run that way would be measuring "Qwen with the wrong
prompt" while reporting it as "Qwen", quietly corrupting the whole curve.

Two properties are therefore tested:

  1. BACKWARD COMPATIBILITY — under the default ("llama") every prompt is
     byte-identical to the pre-refactor version, so the existing 3B logs, figures
     and results remain valid. This is the assertion that makes the refactor safe.

  2. CONTENT INVARIANCE — switching format changes only the WRAPPER, never the
     question. If the models were asked different things, the comparison between
     them would be meaningless.
"""

from __future__ import annotations

import pytest

from medrag_adaptive.models import prompts as P
from medrag_adaptive.models.prompts import (
    build_closed_book_prompt,
    build_draft_prompt,
    build_probe_prompt_a,
    build_probe_prompt_b,
    build_rag_prompt,
    build_verbalized_confidence_prompt,
    set_chat_format,
    get_chat_format,
    CHAT_FORMATS,
)

QUESTION = "A patient develops anaphylaxis. What is the first-line drug?"
CHOICES = {"A": "Diphenhydramine", "B": "Epinephrine"}


class _Chunk:
    def __init__(self, text: str) -> None:
        self.text = text

    def to_context_string(self) -> str:
        return self.text


CHUNKS = [_Chunk("[TEXTBOOK] Anaphylaxis\nTreat with epinephrine 0.3-0.5 mL.")]


@pytest.fixture(autouse=True)
def _restore_default():
    """Chat format is module-global; never leak it between tests."""
    yield
    set_chat_format("llama")


def _all_builders():
    return [
        ("closed_book", lambda: build_closed_book_prompt(QUESTION, CHOICES)),
        ("closed_book_open", lambda: build_closed_book_prompt(QUESTION, None)),
        ("rag", lambda: build_rag_prompt(QUESTION, CHUNKS, CHOICES, False)),
        ("rag_cite", lambda: build_rag_prompt(QUESTION, CHUNKS, CHOICES, True)),
        ("draft", lambda: build_draft_prompt(QUESTION, CHOICES)),
        ("probe_a", lambda: build_probe_prompt_a(QUESTION, CHOICES)),
        ("probe_b", lambda: build_probe_prompt_b(QUESTION, CHOICES)),
        ("verbalized", lambda: build_verbalized_confidence_prompt(QUESTION)),
    ]


class TestBackwardCompatibility:
    """The default must reproduce the original Llama prompts exactly."""

    def test_default_is_llama(self):
        assert get_chat_format() == "llama"

    @pytest.mark.parametrize("name,build", _all_builders(), ids=lambda x: x if isinstance(x, str) else "")
    def test_llama_prompts_keep_inst_markup(self, name, build):
        text = build()
        assert text.startswith("[INST]"), f"{name} lost its [INST] opener"
        assert text.endswith("[/INST]"), f"{name} lost its [/INST] closer"
        assert "<|im_start|>" not in text


class TestQwenFormat:
    def test_uses_chatml_not_inst(self):
        set_chat_format("qwen")
        text = build_closed_book_prompt(QUESTION, CHOICES)
        assert "<|im_start|>system" in text
        assert "<|im_start|>user" in text
        assert "[INST]" not in text and "[/INST]" not in text

    def test_opens_the_assistant_turn(self):
        # Without this the model has not been handed the floor and will not reply.
        set_chat_format("qwen")
        assert build_draft_prompt(QUESTION, CHOICES).endswith("<|im_start|>assistant\n")

    @pytest.mark.parametrize("name,build", _all_builders(), ids=lambda x: x if isinstance(x, str) else "")
    def test_every_builder_switches_format(self, name, build):
        """No prompt may keep [INST] hard-coded — that is the bug this prevents."""
        set_chat_format("qwen")
        text = build()
        assert "[INST]" not in text, f"{name} still hard-codes Llama markup"


class TestContentInvariance:
    """Only the wrapper may change. The question asked must not."""

    @pytest.mark.parametrize("name,build", _all_builders(), ids=lambda x: x if isinstance(x, str) else "")
    def test_question_text_survives_every_format(self, name, build):
        for fmt in CHAT_FORMATS:
            set_chat_format(fmt)
            assert QUESTION in build(), f"{name} lost the question under {fmt}"

    def test_choices_survive_every_format(self):
        for fmt in CHAT_FORMATS:
            set_chat_format(fmt)
            text = build_closed_book_prompt(QUESTION, CHOICES)
            assert "Epinephrine" in text and "Diphenhydramine" in text

    def test_retrieved_context_survives_every_format(self):
        for fmt in CHAT_FORMATS:
            set_chat_format(fmt)
            assert "0.3-0.5 mL" in build_rag_prompt(QUESTION, CHUNKS, CHOICES)


class TestGuards:
    def test_unknown_format_raises(self):
        with pytest.raises(ValueError, match="unknown chat_format"):
            set_chat_format("gpt4")

    def test_known_formats_are_registered(self):
        assert {"llama", "qwen", "plain"} <= set(CHAT_FORMATS)
