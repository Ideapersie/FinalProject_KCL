r"""
scripts/plot_entropy_attribution.py — Token-Level Entropy Attribution figure (A4).

The project's most distinctive asset: the entropy gate already computes a
*per-token* uncertainty array H(p_t) for its draft, but the logs store only the
floats, not the token strings they align to. This script re-runs the entropy
gate's draft ONCE on a chosen clinical question (a single LLM call, ~1 min),
captures the generated token STRINGS alongside their entropies, and renders a
coloured word-level attribution figure.

Why a live re-run rather than replay-from-logs: the logged per_token_entropy is
just a list of numbers with no token text, so it cannot be labelled. Here we
recompute the exact same H(p_t) = -Σ_v p_t(v) log p_t(v) from the model's
_scores buffer (identical to entropy_gate.py) but keep the token strings from
llama-cpp's logprobs output so each bar can be labelled with its word.

Output: results/figures/fig_entropy_attribution.{png,pdf}

Usage:
    python scripts/plot_entropy_attribution.py            # default anaphylaxis Q
    python scripts/plot_entropy_attribution.py --question "..." --choices A=.. B=.. ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

GGUF = "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
OUT = Path("results/figures/fig_entropy_attribution")
DRAFT_MAX_TOKENS = 48

# A high-yield clinical question: the drug NAME is the point of highest
# uncertainty — exactly what token-level attribution is meant to expose.
DEFAULT_Q = "A patient develops anaphylactic shock after a bee sting. What is the first-line drug?"
DEFAULT_CHOICES = {
    "A": "Diphenhydramine",
    "B": "Epinephrine",
    "C": "Hydrocortisone",
    "D": "Salbutamol",
}


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _row_entropy(probs: np.ndarray) -> np.ndarray:
    return -np.sum(probs * np.log(probs + 1e-12), axis=-1)


def run_draft(question: str, choices: dict):
    """Load the model, run the draft, return (tokens, per_token_entropy, threshold)."""
    from llama_cpp import Llama
    from medrag_adaptive.models.prompts import build_draft_prompt

    prompt = build_draft_prompt(question, choices)

    llm = Llama(
        model_path=GGUF,
        n_ctx=2048,
        n_threads=4,
        logits_all=True,          # required to read _scores for entropy
        verbose=False,
    )

    # logprobs=2 makes llama-cpp return the generated token STRINGS, which the
    # plain create_completion() text does not expose token-by-token.
    out = llm.create_completion(
        prompt,
        max_tokens=DRAFT_MAX_TOKENS,
        temperature=0.0,
        logprobs=2,
        echo=False,
    )
    choice = out["choices"][0]
    tokens = choice["logprobs"]["tokens"]           # list[str], one per generated token

    # Recompute per-token entropy exactly as entropy_gate.py does, from _scores.
    prompt_tokens = llm.tokenize(prompt.encode("utf-8"))
    n_prompt = len(prompt_tokens)
    scores = np.asarray(llm._scores, dtype=np.float64)
    start = n_prompt
    end = min(start + len(tokens), scores.shape[0])
    gen_logits = scores[start:end]
    per_token = _row_entropy(_softmax(gen_logits))

    # Align lengths defensively (tokenizer/scores off-by-one guard).
    n = min(len(tokens), len(per_token))
    tokens = tokens[:n]
    per_token = per_token[:n]

    threshold = 0.70   # calibrated entropy threshold (see calibration finding)
    llm.close() if hasattr(llm, "close") else None
    return tokens, per_token, threshold, out["choices"][0]["text"]


def plot(tokens, per_token, threshold, question, answer_text):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    # Clean token strings for display (strip leading spaces, show newlines).
    labels = [t.replace("\n", "\\n").strip() or "·" for t in tokens]

    mean_h = float(np.mean(per_token))
    norm = Normalize(vmin=0.0, vmax=max(1.8, per_token.max()))
    cmap = plt.get_cmap("YlOrRd")
    colors = cmap(norm(per_token))

    fig, ax = plt.subplots(figsize=(max(9, len(tokens) * 0.32), 4.2))
    x = np.arange(len(tokens))
    ax.bar(x, per_token, color=colors, edgecolor="none", width=0.85)
    ax.axhline(threshold, color="green", linestyle="--", linewidth=1.4,
               label=f"calibrated τ = {threshold:.2f}")
    ax.axhline(mean_h, color="black", linestyle=":", linewidth=1.2,
               label=f"mean H̄ = {mean_h:.2f}  → {'RETRIEVE' if mean_h > threshold else 'skip'}")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("token entropy H(p_t)  [nats]")
    ax.set_xlabel("draft token")
    ax.set_title("Token-Level Entropy Attribution — which draft tokens the model is unsure about",
                 fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.margins(x=0.01)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01, fraction=0.04)
    cbar.set_label("uncertainty", fontsize=8)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{OUT}.png", dpi=150)
    fig.savefig(f"{OUT}.pdf")
    plt.close(fig)
    print(f"[ok] {OUT}.png / .pdf")
    print(f"     question: {question}")
    print(f"     draft answer: {answer_text!r}")
    print(f"     mean entropy {mean_h:.3f}  (threshold {threshold:.2f})  "
          f"decision = {'RETRIEVE' if mean_h > threshold else 'skip'}")


def _parse_choices(items):
    if not items:
        return DEFAULT_CHOICES
    d = {}
    for it in items:
        k, _, v = it.partition("=")
        d[k.strip()] = v.strip()
    return d


def main():
    ap = argparse.ArgumentParser(description="Token-level entropy attribution figure.")
    ap.add_argument("--question", default=DEFAULT_Q)
    ap.add_argument("--choices", nargs="*", default=None,
                    help='e.g. A=Diphenhydramine B=Epinephrine ...')
    args = ap.parse_args()

    choices = _parse_choices(args.choices)
    tokens, per_token, thr, ans = run_draft(args.question, choices)
    plot(tokens, per_token, thr, args.question, ans)


if __name__ == "__main__":
    main()
