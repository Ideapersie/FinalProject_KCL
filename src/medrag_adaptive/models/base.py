"""
models/base.py — Abstract base class for LLM backends.

All concrete backends (LlamaBackend, OpenAIBackend) implement this interface
so that policies can be written against a single type, not a specific backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np


class LLMBackend(ABC):
    """
    Abstract LLM backend.

    Subclasses must implement:
      - draft()             : generate a short completion and return raw logits
      - answer()            : generate a full answer given an optional context
      - get_top2_logprobs() : generate a draft and return per-token top-2 logprobs
      - close()             : release any resources (model weights, file handles, etc.)
    """

    @abstractmethod
    def draft(
        self,
        prompt: str,
        max_tokens: int = 48,
    ) -> Tuple[str, Optional[np.ndarray]]:
        """
        Generate a short draft response (for gate signal computation).

        Args:
            prompt:     The full prompt string.
            max_tokens: Maximum number of tokens to generate.

        Returns:
            (text, logits)
            - text:   The generated text string.
            - logits: numpy array of shape [n_tokens_generated, vocab_size],
                      or None if logits are unavailable (e.g. low-tier config,
                      API backend).
        """

    def draft_with_tokens(
        self,
        prompt: str,
        max_tokens: int = 48,
    ) -> Tuple[str, Optional[np.ndarray], Optional[List[str]]]:
        """
        Like draft(), but also return the generated token STRINGS when the
        backend can supply them.

        Deliberately NOT abstract: the default delegates to draft() and reports
        tokens=None, so every existing backend — and every test mock — keeps
        satisfying the interface untouched. Only callers that want token-level
        display (currently just the demo UI) need a backend that overrides it.

        Args:
            prompt:     The full prompt string.
            max_tokens: Maximum number of tokens to generate.

        Returns:
            (text, logits, tokens)
            - tokens: one string per generated token, aligned index-for-index
                      with the rows of `logits`, or None if this backend cannot
                      report them.
        """
        text, logits = self.draft(prompt, max_tokens=max_tokens)
        return text, logits, None

    @abstractmethod
    def answer(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate a full answer.

        Args:
            prompt:     The full prompt string (may include retrieved context).
            max_tokens: Override max_new_tokens from config (optional).

        Returns:
            The generated answer string.
        """

    @abstractmethod
    def get_top2_logprobs(
        self,
        prompt: str,
        max_tokens: int = 48,
    ) -> Tuple[str, List[Dict[str, float]]]:
        """
        Generate a short draft and return per-token top-2 log-probabilities.

        Unlike draft(), this works even when full logits are unavailable
        (e.g. logits_all=False on the low tier), because it relies on the
        completion API's logprobs output rather than the raw _scores buffer.
        It is the signal source for the logit-margin gate.

        Args:
            prompt:     The full prompt string.
            max_tokens: Maximum number of tokens to generate.

        Returns:
            (text, top_logprobs)
            - text:         The generated text string.
            - top_logprobs: One dict per generated token, mapping token string
                            to its log-probability (top-2 entries each).
        """

    @abstractmethod
    def close(self) -> None:
        """Release model resources."""

    def __enter__(self) -> "LLMBackend":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
