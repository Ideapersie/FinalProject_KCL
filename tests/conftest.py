"""
tests/conftest.py — Shared pytest fixtures.

The key fixture here is MockLLMBackend: a drop-in replacement for LlamaBackend
that returns deterministic text and synthetic logit arrays without loading any
GGUF model files. All unit and integration tests use this mock so they run in
<1 second on any machine with no GPU or model files required.

Logit array conventions for MockLLMBackend:
  - HIGH_CONFIDENCE: peaked distribution → low entropy, high margin → gate returns SKIP
  - LOW_CONFIDENCE:  near-uniform distribution → high entropy, low margin → gate returns RETRIEVE
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pytest

from medrag_adaptive.data.schema import Chunk, UnifiedQuestion
from medrag_adaptive.models.base import LLMBackend


# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────

FIXTURES_DIR = Path(__file__).parent / "fixtures"
VOCAB_SIZE = 128          # small fake vocab for tests
N_DRAFT_TOKENS = 48


def _peaked_logits(n_tokens: int = N_DRAFT_TOKENS) -> np.ndarray:
    """
    Return a logit array where token 0 has a very high logit and all others
    are very low. After softmax: p_top1 ≈ 1.0, entropy ≈ 0.0.
    Simulates a highly confident model → gate should return SKIP.
    """
    logits = np.full((n_tokens, VOCAB_SIZE), -10.0, dtype=np.float32)
    logits[:, 0] = 10.0    # strong peak on token 0
    return logits


def _uniform_logits(n_tokens: int = N_DRAFT_TOKENS) -> np.ndarray:
    """
    Return a logit array with near-uniform distribution.
    After softmax: all p ≈ 1/VOCAB_SIZE, entropy ≈ log(VOCAB_SIZE).
    Simulates an uncertain model → gate should return RETRIEVE.
    """
    return np.zeros((n_tokens, VOCAB_SIZE), dtype=np.float32)


# ─────────────────────────────────────────────────────────────────
# MockLLMBackend
# ─────────────────────────────────────────────────────────────────

class MockLLMBackend(LLMBackend):
    """
    Mock LLM backend for testing.

    Args:
        answer_text:     Fixed string returned by answer().
        draft_text:      Fixed string returned by draft().
        confidence:      "high" → peaked logits (gate SKIP);
                         "low"  → uniform logits (gate RETRIEVE).
        verbalized_resp: Fixed response for verbalized confidence check
                         (used by VerbalisedGate tests).
    """

    def __init__(
        self,
        answer_text: str = "A",
        draft_text:  str = "A. Phrenic nerve",
        confidence:  str = "high",    # "high" | "low"
        verbalized_resp: str = "HIGH",
    ) -> None:
        self._answer_text = answer_text
        self._draft_text  = draft_text
        self._confidence  = confidence
        self._verbalized_resp = verbalized_resp

    def draft(
        self,
        prompt: str,
        max_tokens: int = N_DRAFT_TOKENS,
    ) -> Tuple[str, Optional[np.ndarray]]:
        if self._confidence == "high":
            logits = _peaked_logits(max_tokens)
        else:
            logits = _uniform_logits(max_tokens)
        return self._draft_text, logits

    def answer(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        # If the prompt contains the verbalized confidence template keywords,
        # return the configured verbalized response instead of the answer.
        if "HIGH, MEDIUM, or LOW" in prompt:
            return self._verbalized_resp
        return self._answer_text

    def get_top2_logprobs(
        self,
        prompt: str,
        max_tokens: int = N_DRAFT_TOKENS,
    ) -> Tuple[str, list]:
        """Return synthetic top-2 log-probs matching the confidence setting."""
        if self._confidence == "high":
            # Large gap between top-1 and top-2
            top_logprobs = [{"token_A": -0.01, "token_B": -5.0}] * max_tokens
        else:
            # Small gap — model is indecisive
            top_logprobs = [{"token_A": -0.69, "token_B": -0.70}] * max_tokens
        return self._draft_text, top_logprobs

    def close(self) -> None:
        pass


# ─────────────────────────────────────────────────────────────────
# Mock Retriever
# ─────────────────────────────────────────────────────────────────

class MockRetriever:
    """Returns the mock_corpus.jsonl chunks for any query."""

    def __init__(self) -> None:
        corpus_path = FIXTURES_DIR / "mock_corpus.jsonl"
        self._chunks: List[Chunk] = []
        if corpus_path.exists():
            with corpus_path.open() as fh:
                for line in fh:
                    d = json.loads(line)
                    self._chunks.append(Chunk(**d))

    def retrieve(self, query: str, top_k: int = 5) -> List[Chunk]:
        return self._chunks[:top_k]


# ─────────────────────────────────────────────────────────────────
# Pytest fixtures
# ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm_high() -> MockLLMBackend:
    """MockLLMBackend with high-confidence (peaked) logits → gate should SKIP."""
    return MockLLMBackend(answer_text="A", confidence="high")


@pytest.fixture
def mock_llm_low() -> MockLLMBackend:
    """MockLLMBackend with low-confidence (uniform) logits → gate should RETRIEVE."""
    return MockLLMBackend(answer_text="C", confidence="low")


@pytest.fixture
def mock_retriever() -> MockRetriever:
    return MockRetriever()


@pytest.fixture
def sample_questions() -> List[UnifiedQuestion]:
    """Load the 5-question fixture set."""
    path = FIXTURES_DIR / "sample_questions.jsonl"
    questions = []
    with path.open() as fh:
        for line in fh:
            d = json.loads(line)
            questions.append(UnifiedQuestion(**d))
    return questions


@pytest.fixture
def high_risk_question() -> UnifiedQuestion:
    """A high-risk drug interaction question (fixture_003)."""
    return UnifiedQuestion(
        question_id="fixture_003",
        question_text="Methotrexate and trimethoprim co-administration risks which complication?",
        correct_answer="C",
        dataset_source="fixture",
        risk_level="high",
        choices={"A": "Hepatotoxicity", "B": "Nephrotoxicity",
                 "C": "Severe bone marrow suppression", "D": "Peripheral neuropathy"},
        specialty="pharmacology",
    )


@pytest.fixture
def low_risk_question() -> UnifiedQuestion:
    """A low-risk anatomy question (fixture_001)."""
    return UnifiedQuestion(
        question_id="fixture_001",
        question_text="Which nerve innervates the diaphragm?",
        correct_answer="A",
        dataset_source="fixture",
        risk_level="low",
        choices={"A": "Phrenic nerve", "B": "Vagus nerve",
                 "C": "Intercostal nerve", "D": "Accessory nerve"},
        specialty="anatomy",
    )
