"""
ui/attribution.py — pure data→HTML helpers for the demo UI.

Imports no model, no retriever and no Gradio, so every function here is unit
testable without loading a GGUF or an index. app.py does the widget wiring;
this module decides what the pixels say.

Two attributions live here:
  - entropy: the draft's per-token H(p_t), rendered as shaded token chips.
  - retrieval: query terms highlighted inside each retrieved chunk.
"""

from __future__ import annotations

import html
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

# The retriever's own tokeniser. Importing it rather than re-implementing word
# splitting is the point: a highlighted term is then exactly a term BM25 scored
# on, not a plausible-looking approximation of one.
from medrag_adaptive.retrieval.bm25_retriever import _tokenize as _bm25_tokenize

# Low→high uncertainty. Same YlOrRd family as
# scripts/plot_entropy_attribution.py, so the live UI and the dissertation
# figure read as the same artefact.
_RAMP = ["#ffffb2", "#fed976", "#feb24c", "#fd8d3c", "#f03b20", "#bd0026"]

# From this ramp index up, the background is dark enough to need light text.
_DARK_FROM = 4

_WORD_SPLIT = re.compile(r"([A-Za-z0-9]+)")


# ─────────────────────────────────────────────────────────────────
# Draft alignment
# ─────────────────────────────────────────────────────────────────

@dataclass
class AlignedDraft:
    """Draft tokens paired with their entropies, plus what had to be dropped."""
    tokens: List[str]
    entropies: List[float]
    dropped_entropies: int
    dropped_tokens: int

    def __len__(self) -> int:
        return len(self.tokens)


def align_draft(
    tokens: Optional[Sequence[str]],
    entropies: Optional[Sequence[float]],
) -> AlignedDraft:
    """
    Pair tokens with entropies, truncating to the shorter of the two.

    The two can legitimately differ in length: EntropyGate slices the logit
    buffer by the REQUESTED token count, while the token list reflects what was
    actually generated. Truncation here is display-only — the gate's own H̄ is
    reported untouched in the verdict banner and is never recomputed from this.
    """
    toks = list(tokens or [])
    ents = [float(e) for e in (entropies or [])]
    n = min(len(toks), len(ents))
    return AlignedDraft(
        tokens=toks[:n],
        entropies=ents[:n],
        dropped_entropies=max(0, len(ents) - n),
        dropped_tokens=max(0, len(toks) - n),
    )


# ─────────────────────────────────────────────────────────────────
# Entropy heatmap
# ─────────────────────────────────────────────────────────────────

def _colour_index(value: float, vmax: float) -> int:
    if vmax <= 0:
        return 0
    frac = max(0.0, min(1.0, value / vmax))
    return min(len(_RAMP) - 1, int(frac * len(_RAMP)))


def _display_token(token: str) -> str:
    """Make whitespace visible without letting token text reach the DOM raw."""
    shown = token.replace("\n", "⏎").replace("\t", "⇥")
    if shown.strip() == "":
        shown = "␣" * max(1, len(shown))
    return html.escape(shown)


def render_entropy_html(
    aligned: AlignedDraft,
    threshold: float,
    vmax: Optional[float] = None,
) -> str:
    """Render the draft as coloured token chips — darkest is most uncertain."""
    if not aligned.tokens:
        return (
            '<div class="mr-note">No draft tokens were returned, so there is '
            "nothing to attribute. The gate decision above still stands.</div>"
        )

    ceiling = vmax if vmax is not None else max(max(aligned.entropies), threshold, 1e-9)

    chips = []
    for token, entropy in zip(aligned.tokens, aligned.entropies):
        idx = _colour_index(entropy, ceiling)
        fg = "#ffffff" if idx >= _DARK_FROM else "#111111"
        chips.append(
            f'<span class="mr-tok" style="background:{_RAMP[idx]};color:{fg}" '
            f'title="H = {entropy:.3f} nats">{_display_token(token)}</span>'
        )

    parts = [f'<div class="mr-heat">{"".join(chips)}</div>']

    if aligned.dropped_entropies:
        parts.append(
            f'<div class="mr-note">{aligned.dropped_entropies} entropy value(s) had '
            "no corresponding generated token and are not shown. The gate averaged "
            "over all of them; the H̄ in the verdict above is the gate's own value."
            "</div>"
        )
    if aligned.dropped_tokens:
        parts.append(
            f'<div class="mr-note">{aligned.dropped_tokens} token(s) had no '
            "corresponding entropy value and are not shown.</div>"
        )

    return "".join(parts)


# ─────────────────────────────────────────────────────────────────
# Retrieval term highlighting
# ─────────────────────────────────────────────────────────────────

def highlight_terms(text: str, query: str) -> str:
    """
    HTML-escape `text` and wrap every token that also appears in `query`.

    Tokenisation matches BM25Retriever exactly ([a-z0-9]+ over lowercased
    text), so a highlight means "this term contributed to the lexical score".
    Under vector or hybrid retrieval it means lexical overlap and nothing more;
    the caller is responsible for labelling the panel accordingly.
    """
    terms = set(_bm25_tokenize(query))
    if not terms:
        return html.escape(text)

    out: List[str] = []
    for part in _WORD_SPLIT.split(text):
        if part and _WORD_SPLIT.fullmatch(part) and part.lower() in terms:
            out.append(f'<mark class="mr-term">{html.escape(part)}</mark>')
        else:
            out.append(html.escape(part))
    return "".join(out)


def render_chunks_html(chunks: Sequence, query: str, lexical_only: bool) -> str:
    """Render retrieved chunks with matched query terms highlighted."""
    if not chunks:
        return '<div class="mr-note">No chunks retrieved.</div>'

    caveat = ""
    if lexical_only:
        caveat = (
            '<div class="mr-note">Retrieval used an embedding leg. Highlighting '
            "shows lexical overlap only — it does not explain the dense score.</div>"
        )

    cards = []
    for chunk in chunks:
        cards.append(
            '<div class="mr-chunk">'
            f'<div class="mr-chunk-head">{html.escape(chunk.title or "(untitled)")}'
            f'<span class="mr-badge">{html.escape(chunk.source)}</span>'
            f'<span class="mr-badge">score {chunk.score:.3f}</span></div>'
            f'<div class="mr-chunk-body">{highlight_terms(chunk.text, query)}</div>'
            "</div>"
        )
    return caveat + "".join(cards)


# ─────────────────────────────────────────────────────────────────
# Truncated (top-k) entropy — API backends only
# ─────────────────────────────────────────────────────────────────

@dataclass
class TruncatedEntropy:
    """Per-token and mean entropy computed over only the top-k logprobs."""
    per_token: List[float]
    mean: float


def truncated_entropy(
    top_logprobs: Optional[Sequence[Dict[str, float]]],
) -> Optional[TruncatedEntropy]:
    """
    H_topk = -Σ_{i≤k} p_i log p_i over the top-k entries an API returned.

    This is a LOWER BOUND on the true entropy: the discarded tail mass can only
    contribute further positive terms. The probabilities are deliberately NOT
    renormalised — renormalising would inflate the value and destroy the bound.
    It is not comparable to the calibrated τ, which was fitted on
    full-vocabulary entropy, and the UI must say so wherever it is displayed.
    """
    if not top_logprobs:
        return None

    per_token: List[float] = []
    for token_dist in top_logprobs:
        if not token_dist:
            continue
        probs = [math.exp(lp) for lp in token_dist.values()]
        per_token.append(-sum(p * math.log(p + 1e-12) for p in probs))

    if not per_token:
        return None
    return TruncatedEntropy(per_token=per_token, mean=sum(per_token) / len(per_token))
