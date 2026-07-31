"""
models/openai_backend.py — LLMBackend over any OpenAI-compatible chat endpoint.

Covers the OpenAI API itself and every server that mimics it (Ollama, vLLM,
llama.cpp-server, LM Studio). Used only by the demo UI: every measurement in
the dissertation comes from LlamaBackend, because an HTTP API cannot supply the
full-vocabulary logits the entropy gate is defined on.

What survives the API boundary:
    answer()             exact
    get_top2_logprobs()  exact — the margin gate is unaffected
    draft_with_tokens()  token strings yes, logits no
    draft()              logits None → EntropyGate abstains, and EnsembleGate
                         already skips abstaining members (ensemble_gate.py:51)

Uses stdlib urllib rather than the `openai` package, to avoid adding a
dependency to a requirements.txt that is frozen for submission.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Tuple

import numpy as np

from medrag_adaptive.models.base import LLMBackend


class OpenAIBackend(LLMBackend):
    """Chat-completions backend exposing per-token top-k logprobs."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_new_tokens: int = 256,
        top_logprobs: int = 20,
        timeout: int = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._api_key = api_key
        self._temperature = temperature
        self._max_new_tokens = max_new_tokens
        self._top_logprobs = top_logprobs
        self._timeout = timeout
        # Top-k logprobs from the most recent draft call, so the UI can compute
        # a truncated entropy for display without paying for a second call.
        self.last_draft_topk: List[Dict[str, float]] = []

    # ── HTTP ───────────────────────────────────────────────────────

    def _post(self, payload: dict) -> dict:
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}),
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:400]
            raise RuntimeError(f"API returned HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach {self.base_url}: {exc.reason}") from exc

    def _complete(self, prompt: str, max_tokens: int, want_logprobs: bool) -> dict:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self._temperature,
            "max_tokens": max_tokens,
        }
        if want_logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = self._top_logprobs
        return self._post(payload)

    # ── parsing ────────────────────────────────────────────────────

    @staticmethod
    def _content(response: dict) -> str:
        return response["choices"][0]["message"]["content"] or ""

    @staticmethod
    def _logprob_entries(response: dict) -> List[dict]:
        logprobs = response["choices"][0].get("logprobs") or {}
        return logprobs.get("content") or []

    @staticmethod
    def _as_topk(entries: List[dict]) -> List[Dict[str, float]]:
        return [
            {alt["token"]: alt["logprob"] for alt in entry.get("top_logprobs", [])}
            for entry in entries
        ]

    # ── LLMBackend interface ───────────────────────────────────────

    def draft(
        self,
        prompt: str,
        max_tokens: int = 48,
    ) -> Tuple[str, Optional[np.ndarray]]:
        """No full-vocabulary logits exist over an API, so logits is always None."""
        text, _logits, _tokens = self.draft_with_tokens(prompt, max_tokens=max_tokens)
        return text, None

    def draft_with_tokens(
        self,
        prompt: str,
        max_tokens: int = 48,
    ) -> Tuple[str, Optional[np.ndarray], Optional[List[str]]]:
        response = self._complete(prompt, max_tokens, want_logprobs=True)
        entries = self._logprob_entries(response)
        self.last_draft_topk = self._as_topk(entries)
        tokens = [entry["token"] for entry in entries] or None
        return self._content(response), None, tokens

    def answer(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        n = max_tokens if max_tokens is not None else self._max_new_tokens
        return self._content(self._complete(prompt, n, want_logprobs=False))

    def get_top2_logprobs(
        self,
        prompt: str,
        max_tokens: int = 48,
    ) -> Tuple[str, List[Dict[str, float]]]:
        """
        Returns up to `top_logprobs` entries per token, not exactly two. The
        margin gate takes the two largest of whatever it is given, so a wider
        list is compatible — and those extra entries are what make a truncated
        entropy possible on this backend.
        """
        response = self._complete(prompt, max_tokens, want_logprobs=True)
        top = self._as_topk(self._logprob_entries(response))
        self.last_draft_topk = top
        return self._content(response), top

    def close(self) -> None:
        """Nothing to release — this backend holds no local resources."""
