"""
models/base.py — Abstract base class for LLM backends.

All concrete backends (LlamaBackend, OpenAIBackend) implement this interface
so that policies can be written against a single type, not a specific backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np


class LLMBackend(ABC):
    """
    Abstract LLM backend.

    Subclasses must implement:
      - draft()  : generate a short completion and return raw logits
      - answer() : generate a full answer given an optional context
      - close()  : release any resources (model weights, file handles, etc.)
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
    def close(self) -> None:
        """Release model resources."""

    def __enter__(self) -> "LLMBackend":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
