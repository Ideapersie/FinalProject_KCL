"""
models/llama_backend.py — llama-cpp-python wrapper for CPU-only GGUF inference.

Key design notes:
  1. logits_all=True at Llama() construction is required for entropy + margin gates.
     It tells llama-cpp-python to retain the full logit array for every generated
     token position. This costs ~256 MB extra RAM on a 32K vocab / 2048 ctx model.
     On low-tier hardware (4 GB), set logits_all=False in the config — this disables
     entropy/margin gates but cuts RAM pressure significantly.

  2. Logit extraction for the entropy gate:
     After create_completion(), self._llm._scores is a numpy array of shape
     [n_ctx, vocab_size] containing raw logits (pre-softmax, float32).
     We slice [start_pos : end_pos] to get only the generated token positions.
     This is a semi-private attribute — document it clearly.

  3. Logit extraction for the margin gate:
     We use the logprobs=2 parameter in create_completion() which returns top-2
     log-probabilities per generated token in the response dict. No _scores access
     needed, which works even when logits_all=False.

  4. temperature=0.0 + seed=config.seed → deterministic greedy decode.
     Both are required: temperature controls sampling; seed controls initial state.

CLI usage (Sprint 1 milestone):
    python -m medrag_adaptive.models.llama_backend \
        --model models/llama-3.2-3b-q4_k_m.gguf \
        --prompt "What is the mechanism of action of metformin?"
"""

from __future__ import annotations

import argparse
import time
from typing import Optional, Tuple

import numpy as np

from medrag_adaptive.models.base import LLMBackend


class LlamaBackend(LLMBackend):
    """
    CPU-only LLM backend using llama-cpp-python.

    Args:
        gguf_path:      Path to the GGUF model file.
        n_ctx:          Context window size in tokens.
        n_threads:      Number of CPU threads for inference.
        n_batch:        Batch size for prompt evaluation.
        logits_all:     If True, retain logits for all token positions
                        (required for entropy gate; costs ~256 MB extra RAM).
        temperature:    Sampling temperature. Must be 0.0 for reproducibility.
        max_new_tokens: Default max tokens to generate.
        seed:           RNG seed for deterministic decoding.
        verbose:        If False, suppress llama.cpp console output.
    """

    def __init__(
        self,
        gguf_path: str,
        n_ctx: int = 2048,
        n_threads: int = 4,
        n_batch: int = 128,
        logits_all: bool = True,
        temperature: float = 0.0,
        max_new_tokens: int = 256,
        seed: int = 42,
        verbose: bool = False,
        chat_format: str = "llama",
        n_gpu_layers: int = 0,
    ) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ImportError(
                "llama-cpp-python is not installed. "
                "Run: pip install llama-cpp-python\n"
                "For CPU-only with OpenBLAS: "
                "CMAKE_ARGS='-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS' "
                "pip install llama-cpp-python"
            ) from exc

        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._logits_all = logits_all

        # Select the chat markup this model was tuned on, for every prompt built
        # from here on. Doing it at backend construction means a policy or gate
        # can never accidentally build a prompt in another model's format.
        from medrag_adaptive.models.prompts import set_chat_format
        set_chat_format(chat_format)

        self._llm = Llama(
            model_path=gguf_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_batch=n_batch,
            logits_all=logits_all,
            # 0 = CPU only (default). -1 = offload every layer to the GPU, used
            # for the cloud scaling runs. Under CUDA llama-cpp still exposes
            # `_scores` and `logprobs=2`, so the entropy and margin gates keep
            # working — the reason this project stays on llama-cpp.
            n_gpu_layers=n_gpu_layers,
            seed=seed,
            verbose=verbose,
        )

    # ── Public interface ──────────────────────────────────────────

    def draft(
        self,
        prompt: str,
        max_tokens: int = 48,
    ) -> Tuple[str, Optional[np.ndarray]]:
        """
        Generate a short draft response and return raw logits.

        Returns:
            (text, logits) where logits has shape [n_generated, vocab_size],
            or (text, None) if logits_all=False.
        """
        output = self._llm.create_completion(
            prompt,
            max_tokens=max_tokens,
            temperature=self._temperature,
            echo=False,
        )
        text = output["choices"][0]["text"]

        logits = None
        if self._logits_all:
            logits = self._extract_generation_logits(prompt, n_generated=max_tokens)

        return text, logits

    def answer(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate a full answer."""
        n = max_tokens if max_tokens is not None else self._max_new_tokens
        output = self._llm.create_completion(
            prompt,
            max_tokens=n,
            temperature=self._temperature,
            echo=False,
        )
        return output["choices"][0]["text"]

    def get_top2_logprobs(
        self,
        prompt: str,
        max_tokens: int = 48,
    ) -> Tuple[str, list]:
        """
        Generate a draft and return top-2 log-probs per token position.
        Used by the margin gate (works even when logits_all=False).

        Returns:
            (text, top_logprobs_list) where top_logprobs_list is a list of
            dicts mapping token string → log-probability.
        """
        output = self._llm.create_completion(
            prompt,
            max_tokens=max_tokens,
            temperature=self._temperature,
            logprobs=2,      # return top-2 log-probs per token
            echo=False,
        )
        choice = output["choices"][0]
        text = choice["text"]
        top_logprobs = choice.get("logprobs", {}).get("top_logprobs", [])
        return text, top_logprobs

    def close(self) -> None:
        """Release the llama model from memory."""
        del self._llm

    # ── Private helpers ───────────────────────────────────────────

    def _extract_generation_logits(
        self,
        prompt: str,
        n_generated: int,
    ) -> Optional[np.ndarray]:
        """
        Extract the logit matrix for the most recently generated tokens.

        llama-cpp-python stores a cumulative logit array in self._llm._scores
        with shape [n_ctx, vocab_size]. After a create_completion() call,
        the generated token logits occupy positions [n_prompt : n_prompt + n_gen].

        NOTE: _scores is a semi-private attribute. It is consistently available
        in llama-cpp-python >=0.2.x when logits_all=True, but this may change in
        future versions. If it breaks, fall back to the logprobs API instead.

        Returns:
            numpy array of shape [n_generated_actual, vocab_size] or None.
        """
        try:
            scores = self._llm._scores          # shape: [n_ctx, vocab_size]
            if scores is None or len(scores) == 0:
                return None

            # Determine how many tokens are in the prompt
            prompt_tokens = self._llm.tokenize(prompt.encode("utf-8"))
            n_prompt = len(prompt_tokens)

            # Slice out only the generated positions
            n_total = scores.shape[0]
            start = n_prompt
            end = min(start + n_generated, n_total)
            if start >= end:
                return None

            return np.array(scores[start:end], dtype=np.float32)
        except (AttributeError, Exception):
            # Graceful fallback — gate will get None and use safe default
            return None


# ─────────────────────────────────────────────────────────────────
# Factory from ProjectConfig
# ─────────────────────────────────────────────────────────────────

def llama_backend_from_config(config: "ProjectConfig") -> LlamaBackend:  # type: ignore[name-defined]
    """Construct a LlamaBackend from a ProjectConfig instance."""
    m = config.model
    hw = config.hardware
    return LlamaBackend(
        gguf_path=m.gguf_path,
        n_ctx=m.n_ctx,
        n_threads=hw.n_threads,
        n_batch=hw.n_batch,
        logits_all=m.logits_all,
        temperature=m.temperature,
        max_new_tokens=m.max_new_tokens,
        seed=config.seed,
        verbose=m.verbose,
        chat_format=m.chat_format,
        n_gpu_layers=m.n_gpu_layers,
    )


# ─────────────────────────────────────────────────────────────────
# Sprint 1 milestone CLI
# ─────────────────────────────────────────────────────────────────

def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Sprint 1 milestone — single closed-book inference on CPU"
    )
    parser.add_argument("--model",  required=True, help="Path to GGUF model file")
    parser.add_argument("--prompt", required=True, help="Prompt string")
    parser.add_argument("--n-ctx",      type=int, default=2048)
    parser.add_argument("--n-threads",  type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--no-logits", action="store_true",
                        help="Disable logits_all (saves RAM on low-tier)")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    backend = LlamaBackend(
        gguf_path=args.model,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        logits_all=not args.no_logits,
        max_new_tokens=args.max_tokens,
        seed=args.seed,
        verbose=False,
    )

    print(f"\nPrompt:\n{args.prompt}\n")
    t0 = time.perf_counter_ns()
    response = backend.answer(args.prompt)
    latency_ms = (time.perf_counter_ns() - t0) / 1_000_000

    print(f"Response:\n{response}")
    print(f"\nLatency: {latency_ms:.1f} ms")
    backend.close()


if __name__ == "__main__":
    _main()
