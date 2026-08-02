# Adaptive-Gating Demo UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local Gradio app where a person types a medical question and sees the P5 gate's
decision, the draft tokens it was uncertain about, and the retrieved chunks with matched
query terms highlighted — driven by the same code the evaluation runs.

**Architecture:** In-process Gradio Blocks. `ui/attribution.py` holds pure
data→HTML functions (no model, no Gradio) and carries all the unit tests.
`ui/session.py` owns the backend + retriever singletons and runs P5 then P3 through the
existing `build_policy` factory. `ui/app.py` is Gradio wiring only. One new
`LLMBackend` method (`draft_with_tokens`) with a working default, so no existing call
site changes.

**Tech Stack:** Python 3.11, Gradio 4.x, llama-cpp-python 0.3.x, pydantic 2, pytest,
stdlib `urllib.request` for the OpenAI-compatible backend (no new dependency).

**Spec:** `docs/superpowers/specs/2026-07-26-adaptive-gating-demo-ui-design.md`

---

## Non-negotiable constraint

Submission is 6 Aug 2026. Every number in the report must survive this work unchanged.
The regression guard is Task 9: the existing test suite must pass with no modifications
to any existing test. If an existing test changes behaviour, **stop and report** — do not
adjust the test.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/medrag_adaptive/models/base.py` (modify) | Add non-abstract `draft_with_tokens()` defaulting to `draft()` + `tokens=None` |
| `src/medrag_adaptive/models/llama_backend.py` (modify) | Override `draft_with_tokens()` using `logprobs=2` |
| `src/medrag_adaptive/gating/entropy_gate.py` (modify) | `keep_tokens` flag → `details["draft_tokens"]` |
| `src/medrag_adaptive/models/openai_backend.py` (create) | OpenAI-compatible chat backend over stdlib HTTP |
| `src/medrag_adaptive/ui/attribution.py` (create) | Pure data→HTML: alignment, entropy heatmap, term highlighting, truncated entropy |
| `src/medrag_adaptive/ui/session.py` (create) | Owns backend+retriever; runs P5+P3; returns `DemoResult` |
| `src/medrag_adaptive/ui/app.py` (create) | Gradio Blocks, layout A |
| `scripts/run_demo.py` (create) | CLI launcher |
| `tests/unit/test_attribution.py` (create) | Tests for `attribution.py` |
| `tests/unit/test_ui_session.py` (create) | Tests for `session.py` against conftest mocks |
| `tests/unit/test_draft_with_tokens.py` (create) | Tests for the backend default + entropy gate flag |

---

## Task 1: `draft_with_tokens` plumbing

**Files:**
- Modify: `src/medrag_adaptive/models/base.py`
- Modify: `src/medrag_adaptive/models/llama_backend.py`
- Modify: `src/medrag_adaptive/gating/entropy_gate.py`
- Test: `tests/unit/test_draft_with_tokens.py`

- [ ] **Step 1: Write the failing test**

```python
"""tests/unit/test_draft_with_tokens.py — token-string plumbing for the demo UI.

The demo's entropy heatmap needs the draft's token STRINGS aligned to the
per-token entropies the gate already computes. These tests pin the two
guarantees that make that safe: backends that do not implement token capture
still work (default returns None), and EntropyGate's behaviour at
keep_tokens=False is byte-identical to before.
"""

from __future__ import annotations

from medrag_adaptive.gating.entropy_gate import EntropyGate
from tests.conftest import MockLLMBackend


def test_default_draft_with_tokens_returns_none_tokens(low_risk_question):
    """A backend that never heard of token capture still satisfies the interface."""
    llm = MockLLMBackend(confidence="high")
    text, logits, tokens = llm.draft_with_tokens("some prompt", max_tokens=8)
    assert isinstance(text, str)
    assert logits is not None
    assert tokens is None


def test_entropy_gate_default_does_not_request_tokens(low_risk_question):
    gate = EntropyGate(threshold=0.7)
    decision = gate.decide(low_risk_question, MockLLMBackend(confidence="high"))
    assert "draft_tokens" not in decision.details


def test_entropy_gate_keep_tokens_adds_key(low_risk_question):
    gate = EntropyGate(threshold=0.7, keep_tokens=True)
    decision = gate.decide(low_risk_question, MockLLMBackend(confidence="high"))
    assert "draft_tokens" in decision.details        # None on a mock, but present


def test_keep_tokens_does_not_change_the_signal(low_risk_question):
    """The flag must be observational only — same H̄, same decision."""
    llm = MockLLMBackend(confidence="low")
    plain = EntropyGate(threshold=0.7).decide(low_risk_question, llm)
    kept = EntropyGate(threshold=0.7, keep_tokens=True).decide(low_risk_question, llm)
    assert plain.signal_value == kept.signal_value
    assert plain.retrieve == kept.retrieve
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_draft_with_tokens.py -v`
Expected: FAIL — `AttributeError: 'MockLLMBackend' object has no attribute 'draft_with_tokens'`

- [ ] **Step 3: Add the default method to the ABC**

In `src/medrag_adaptive/models/base.py`, after the abstract `draft()` and before
`answer()`:

```python
    def draft_with_tokens(
        self,
        prompt: str,
        max_tokens: int = 48,
    ) -> Tuple[str, Optional[np.ndarray], Optional[List[str]]]:
        """
        Like draft(), but also return the generated token STRINGS when the
        backend can supply them.

        Deliberately NOT abstract: the default delegates to draft() and reports
        tokens=None, so every existing backend (and every test mock) satisfies
        the interface unchanged. Only callers that want token-level display —
        currently just the demo UI — need a backend that overrides this.

        Returns:
            (text, logits, tokens) where tokens is one string per generated
            token, aligned index-for-index with the rows of `logits`, or None
            if this backend cannot report them.
        """
        text, logits = self.draft(prompt, max_tokens=max_tokens)
        return text, logits, None
```

- [ ] **Step 4: Override it in LlamaBackend**

In `src/medrag_adaptive/models/llama_backend.py`, after `draft()`:

```python
    def draft_with_tokens(
        self,
        prompt: str,
        max_tokens: int = 48,
    ) -> Tuple[str, Optional[np.ndarray], Optional[list]]:
        """
        draft() plus the generated token strings, for the demo UI's heatmap.

        logprobs=2 is what makes llama-cpp return per-token strings; at
        temperature 0.0 it does not change which tokens are generated, so the
        draft is the same one the gate would have measured without it.

        Note the deliberate difference from draft(): the logit slice is bounded
        by the number of tokens ACTUALLY generated, not by max_tokens. draft()
        keeps its original max_tokens slicing because changing it would move H̄
        on every question and invalidate the shipped calibration.
        """
        output = self._llm.create_completion(
            prompt,
            max_tokens=max_tokens,
            temperature=self._temperature,
            logprobs=2,
            echo=False,
        )
        choice = output["choices"][0]
        text = choice["text"]
        tokens = (choice.get("logprobs") or {}).get("tokens")

        logits = None
        if self._logits_all:
            n_gen = len(tokens) if tokens else max_tokens
            logits = self._extract_generation_logits(prompt, n_generated=n_gen)

        return text, logits, tokens
```

- [ ] **Step 5: Add `keep_tokens` to EntropyGate**

In `src/medrag_adaptive/gating/entropy_gate.py`, replace `__init__` and the first two
lines of `decide`:

```python
    def __init__(
        self,
        threshold: float = 2.5,
        draft_max_tokens: int = 48,
        keep_tokens: bool = False,
    ) -> None:
        self.threshold = threshold
        self.draft_max_tokens = draft_max_tokens
        # Observational only: when True the gate also records the draft's token
        # strings in details, for the demo UI's token-level heatmap. It never
        # changes the signal or the decision. Default False so every evaluation
        # run takes the identical code path it always has.
        self.keep_tokens = keep_tokens

    def decide(self, question: UnifiedQuestion, llm: LLMBackend) -> GateDecision:
        prompt = build_draft_prompt(question.question_text, question.choices)
        tokens = None
        if self.keep_tokens:
            _text, logits, tokens = llm.draft_with_tokens(
                prompt, max_tokens=self.draft_max_tokens
            )
        else:
            _text, logits = llm.draft(prompt, max_tokens=self.draft_max_tokens)
```

and in the returned `details` dict, after `"per_token_entropy"`:

```python
                **({"draft_tokens": tokens} if self.keep_tokens else {}),
```

- [ ] **Step 6: Run the new tests**

Run: `python -m pytest tests/unit/test_draft_with_tokens.py -v`
Expected: 4 passed

- [ ] **Step 7: Run the gating tests as an immediate regression check**

Run: `python -m pytest tests/ -k "gate or gating" -v`
Expected: all pass, same count as before the change

- [ ] **Step 8: Commit**

```bash
git add src/medrag_adaptive/models/base.py src/medrag_adaptive/models/llama_backend.py \
        src/medrag_adaptive/gating/entropy_gate.py tests/unit/test_draft_with_tokens.py
git commit -m "feat(models): add opt-in draft_with_tokens for token-level attribution"
```

---

## Task 2: Draft alignment + entropy heatmap

**Files:**
- Create: `src/medrag_adaptive/ui/attribution.py`
- Test: `tests/unit/test_attribution.py`

- [ ] **Step 1: Write the failing test**

```python
"""tests/unit/test_attribution.py — pure rendering helpers for the demo UI.

No model, no retriever, no Gradio. Everything here is data in, HTML string out.
"""

from __future__ import annotations

from medrag_adaptive.ui.attribution import align_draft, render_entropy_html


def test_align_equal_lengths():
    a = align_draft(["a", "b"], [0.1, 0.2])
    assert a.tokens == ["a", "b"]
    assert a.entropies == [0.1, 0.2]
    assert a.dropped_entropies == 0
    assert a.dropped_tokens == 0


def test_align_more_entropies_than_tokens():
    """The gate slices _scores by max_tokens, so trailing entropies can have no token."""
    a = align_draft(["a"], [0.1, 0.2, 0.3])
    assert a.tokens == ["a"]
    assert a.entropies == [0.1]
    assert a.dropped_entropies == 2


def test_align_more_tokens_than_entropies():
    a = align_draft(["a", "b", "c"], [0.1])
    assert a.tokens == ["a"]
    assert a.dropped_tokens == 2


def test_align_handles_none_tokens():
    a = align_draft(None, [0.1, 0.2])
    assert a.tokens == []
    assert a.entropies == []
    assert a.dropped_entropies == 2


def test_render_escapes_html_in_tokens():
    html_out = render_entropy_html(align_draft(["<script>"], [0.5]), threshold=0.7)
    assert "<script>" not in html_out
    assert "&lt;script&gt;" in html_out


def test_render_reports_dropped_entropies():
    html_out = render_entropy_html(align_draft(["a"], [0.1, 0.2]), threshold=0.7)
    assert "2" in html_out
    assert "no corresponding generated token" in html_out


def test_render_empty_draft_is_not_a_crash():
    html_out = render_entropy_html(align_draft([], []), threshold=0.7)
    assert "no draft tokens" in html_out.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_attribution.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'medrag_adaptive.ui.attribution'`

- [ ] **Step 3: Write the implementation**

Create `src/medrag_adaptive/ui/attribution.py`:

```python
"""ui/attribution.py — pure data→HTML helpers for the demo UI.

Imports no model, no retriever and no Gradio, so every function here is unit
testable without loading a GGUF. app.py does the widget wiring; this module
decides what the pixels say.
"""

from __future__ import annotations

import html
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

# Low→high uncertainty. Same YlOrRd family as
# scripts/plot_entropy_attribution.py, so the live UI and the dissertation
# figure read as the same artefact.
_RAMP = ["#ffffb2", "#fed976", "#feb24c", "#fd8d3c", "#f03b20", "#bd0026"]

# Above this fraction of the ramp the background is dark enough to need light text.
_DARK_FROM = 4


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

    They can legitimately differ in length: EntropyGate slices the logit buffer
    by the REQUESTED token count while the token list reflects what was actually
    generated. Truncation here is display-only — the gate's own H̄ is reported
    untouched in the verdict banner and is never recomputed from this.
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
    """Render the draft as coloured token chips, darkest = most uncertain."""
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
            "over all of them; the H̄ in the verdict above is the gate's own value.</div>"
        )
    if aligned.dropped_tokens:
        parts.append(
            f'<div class="mr-note">{aligned.dropped_tokens} token(s) had no '
            "corresponding entropy value and are not shown.</div>"
        )

    return "".join(parts)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_attribution.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/medrag_adaptive/ui/attribution.py tests/unit/test_attribution.py
git commit -m "feat(ui): add draft alignment and entropy heatmap rendering"
```

---

## Task 3: Query-term highlighting

**Files:**
- Modify: `src/medrag_adaptive/ui/attribution.py`
- Test: `tests/unit/test_attribution.py`

- [ ] **Step 1: Write the failing test** (append to `tests/unit/test_attribution.py`)

```python
from medrag_adaptive.ui.attribution import highlight_terms


def test_highlight_marks_matching_terms():
    out = highlight_terms("Epinephrine treats anaphylaxis.", "anaphylaxis drug")
    assert "<mark" in out
    assert "anaphylaxis" in out


def test_highlight_is_case_insensitive():
    out = highlight_terms("EPINEPHRINE is first-line.", "epinephrine")
    assert "<mark" in out
    assert "EPINEPHRINE" in out          # original casing preserved in output


def test_highlight_respects_word_boundaries():
    """'pin' must not light up inside 'epinephrine'."""
    out = highlight_terms("epinephrine", "pin")
    assert "<mark" not in out


def test_highlight_hyphenated_query_matches_both_parts():
    """_tokenize splits on non-alphanumerics, so covid-19 is two terms."""
    out = highlight_terms("Guidance on COVID-19 vaccines.", "covid-19")
    assert out.count("<mark") == 2


def test_highlight_escapes_hostile_chunk_text():
    out = highlight_terms("<script>alert(1)</script> anaphylaxis", "anaphylaxis")
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


def test_highlight_empty_query_escapes_and_marks_nothing():
    out = highlight_terms("<b>text</b>", "")
    assert out == "&lt;b&gt;text&lt;/b&gt;"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_attribution.py -k highlight -v`
Expected: FAIL — `ImportError: cannot import name 'highlight_terms'`

- [ ] **Step 3: Write the implementation** (append to `attribution.py`)

```python
# The retriever's own tokeniser. Importing it rather than re-implementing word
# splitting is the point: highlighted terms are then exactly the terms BM25
# scored on, not a plausible-looking approximation of them.
from medrag_adaptive.retrieval.bm25_retriever import _tokenize as _bm25_tokenize

_WORD_SPLIT = re.compile(r"([A-Za-z0-9]+)")


def highlight_terms(text: str, query: str) -> str:
    """
    HTML-escape `text` and wrap every token that also appears in `query`.

    Tokenisation matches BM25Retriever exactly ([a-z0-9]+ over lowercased
    text), so a highlight means "this term contributed to the lexical score".
    On vector-only retrieval it means lexical overlap and nothing more — the
    caller is responsible for labelling the panel accordingly.
    """
    terms = set(_bm25_tokenize(query))
    if not terms:
        return html.escape(text)

    out: List[str] = []
    for part in _WORD_SPLIT.split(text):
        if part and part.lower() in terms and _WORD_SPLIT.fullmatch(part):
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_attribution.py -v`
Expected: 13 passed

- [ ] **Step 5: Commit**

```bash
git add src/medrag_adaptive/ui/attribution.py tests/unit/test_attribution.py
git commit -m "feat(ui): highlight BM25 query terms in retrieved chunks"
```

---

## Task 4: Truncated top-k entropy

**Files:**
- Modify: `src/medrag_adaptive/ui/attribution.py`
- Test: `tests/unit/test_attribution.py`

- [ ] **Step 1: Write the failing test** (append)

```python
import math

from medrag_adaptive.ui.attribution import truncated_entropy


def test_truncated_entropy_of_a_uniform_pair_is_ln2():
    """Two tokens at p=0.5 → H = ln 2 ≈ 0.6931 nats."""
    half = math.log(0.5)
    result = truncated_entropy([{"a": half, "b": half}])
    assert abs(result.mean - math.log(2)) < 1e-9
    assert len(result.per_token) == 1


def test_truncated_entropy_of_a_certain_token_is_near_zero():
    result = truncated_entropy([{"a": 0.0}])       # log p = 0 → p = 1
    assert abs(result.mean) < 1e-6


def test_truncated_entropy_averages_across_tokens():
    half = math.log(0.5)
    result = truncated_entropy([{"a": half, "b": half}, {"a": 0.0}])
    assert abs(result.mean - math.log(2) / 2) < 1e-9


def test_truncated_entropy_on_empty_input_is_none():
    assert truncated_entropy([]) is None
    assert truncated_entropy(None) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_attribution.py -k truncated -v`
Expected: FAIL — `ImportError: cannot import name 'truncated_entropy'`

- [ ] **Step 3: Write the implementation** (append)

```python
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
    add positive terms. The probabilities are deliberately NOT renormalised —
    renormalising would inflate the value and destroy the bound. It is not
    comparable to the calibrated τ, which was fitted on full-vocabulary entropy,
    and the UI must always say so where it is displayed.
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_attribution.py -v`
Expected: 17 passed

- [ ] **Step 5: Commit**

```bash
git add src/medrag_adaptive/ui/attribution.py tests/unit/test_attribution.py
git commit -m "feat(ui): add truncated top-k entropy for API backends"
```

---

## Task 5: DemoSession

**Files:**
- Create: `src/medrag_adaptive/ui/session.py`
- Test: `tests/unit/test_ui_session.py`

- [ ] **Step 1: Write the failing test**

```python
"""tests/unit/test_ui_session.py — the demo's run orchestration.

Runs entirely against the conftest mocks: no GGUF, no index, no network.
"""

from __future__ import annotations

import pytest

from medrag_adaptive.config import ProjectConfig
from medrag_adaptive.ui.session import DemoSession
from tests.conftest import MockLLMBackend, MockRetriever


def _cfg(gate_type: str = "entropy", entropy_threshold: float = 0.7) -> ProjectConfig:
    cfg = ProjectConfig()
    cfg.policy.name = "p5_gated"
    cfg.policy.retrieval_mode = "bm25"
    cfg.gate.type = gate_type
    cfg.gate.entropy_threshold = entropy_threshold
    return cfg


def test_low_confidence_retrieves_and_populates_chunks():
    session = DemoSession(_cfg(), MockLLMBackend(confidence="low"), MockRetriever())
    result = session.answer("Which nerve innervates the diaphragm?", choices=None)
    assert result.verdict == "RETRIEVE"
    assert result.chunks


def test_high_confidence_skips_and_leaves_chunks_empty():
    session = DemoSession(_cfg(), MockLLMBackend(confidence="high"), MockRetriever())
    result = session.answer("Which nerve innervates the diaphragm?", choices=None)
    assert result.verdict == "SKIP"
    assert result.chunks == []


def test_skip_path_flags_identical_answers():
    """On SKIP, P5 and P3 issue the same prompt — the UI must not imply a comparison."""
    session = DemoSession(_cfg(), MockLLMBackend(confidence="high"), MockRetriever())
    result = session.answer("Which nerve innervates the diaphragm?", choices=None)
    assert result.answers_identical is True


def test_entropy_gate_records_tokens_for_the_ui():
    session = DemoSession(_cfg(), MockLLMBackend(confidence="low"), MockRetriever())
    result = session.answer("q", choices=None)
    # MockLLMBackend inherits the default draft_with_tokens → tokens is None,
    # so the aligned draft is empty but the key must have been requested.
    assert result.aligned.dropped_entropies > 0


def test_choices_produce_an_mcq_question():
    session = DemoSession(_cfg(), MockLLMBackend(confidence="high"), MockRetriever())
    result = session.answer("q", choices={"A": "one", "B": "two"})
    assert result.is_multiple_choice is True


def test_sole_entropy_gate_without_logits_is_refused():
    """An entropy-only gate on a backend with no logits reports a default, not a
    measurement. Refuse rather than display it as a decision."""

    class NoLogitsBackend(MockLLMBackend):
        def draft(self, prompt, max_tokens=48):
            return self._draft_text, None

    session = DemoSession(_cfg(), NoLogitsBackend(), MockRetriever())
    with pytest.raises(ValueError, match="entropy gate cannot run"):
        session.answer("q", choices=None)


def test_top_k_override_is_passed_to_the_retriever():
    session = DemoSession(_cfg(), MockLLMBackend(confidence="low"), MockRetriever())
    result = session.answer("q", choices=None, top_k=2)
    assert len(result.chunks) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_ui_session.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'medrag_adaptive.ui.session'`

- [ ] **Step 3: Write the implementation**

Create `src/medrag_adaptive/ui/session.py`:

```python
"""ui/session.py — run orchestration for the demo app.

Owns the expensive singletons (one LLM backend, one retriever) and answers one
question at a time by running P5 and then P3. Both policies come from the
existing build_policy factory, so the demo exercises the evaluation code path
rather than a parallel reimplementation of it — which is the only thing that
makes "this is what the gate really did" a true statement.

Knows nothing about HTML or Gradio; it returns a DemoResult and app.py renders it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from medrag_adaptive.config import ProjectConfig
from medrag_adaptive.data.schema import Chunk, UnifiedQuestion
from medrag_adaptive.gating.base import Gate
from medrag_adaptive.gating.ensemble_gate import EnsembleGate
from medrag_adaptive.gating.entropy_gate import EntropyGate
from medrag_adaptive.models.base import LLMBackend
from medrag_adaptive.policies.factory import build_gate
from medrag_adaptive.policies.p3_closed_book import ClosedBookPolicy
from medrag_adaptive.policies.p5_gated import GatedPolicy
from medrag_adaptive.retrieval.base import Retriever
from medrag_adaptive.ui.attribution import AlignedDraft, align_draft


@dataclass
class DemoResult:
    """Everything app.py needs to render one submitted question."""
    verdict: str                       # "RETRIEVE" | "SKIP"
    gate_name: str
    signal_value: Optional[float]
    threshold: Optional[float]
    gate_details: Dict[str, Any]
    aligned: AlignedDraft
    p5_answer: str
    p3_answer: str
    answers_identical: bool
    chunks: List[Chunk]
    is_multiple_choice: bool
    query: str
    latency_s: float
    notes: List[str] = field(default_factory=list)


class _TopKRetriever(Retriever):
    """Pins top_k for retrievers called by a policy that does not pass one."""

    def __init__(self, inner, top_k: int) -> None:
        self._inner = inner
        self._top_k = top_k

    def retrieve(self, query: str, top_k: int = 5) -> List[Chunk]:
        return self._inner.retrieve(query, top_k=self._top_k)

    def close(self) -> None:
        close = getattr(self._inner, "close", None)
        if close:
            close()


def _enable_token_capture(gate: Gate) -> None:
    """Switch on observational token capture wherever an entropy gate sits.

    Done here rather than in policies.factory so the factory — which every
    evaluation run goes through — is not touched at all.
    """
    if isinstance(gate, EntropyGate):
        gate.keep_tokens = True
    elif isinstance(gate, EnsembleGate):
        for member in gate.members:
            _enable_token_capture(member)


def _entropy_details(details: Dict[str, Any]) -> Dict[str, Any]:
    """Pull the entropy member's details out of a possibly-ensemble payload."""
    if "per_token_entropy" in details:
        return details
    return (details.get("members") or {}).get("entropy", {})


class DemoSession:
    """Holds the model and index, and answers one question at a time."""

    def __init__(
        self,
        cfg: ProjectConfig,
        llm: LLMBackend,
        retriever: Optional[Retriever],
    ) -> None:
        self.cfg = cfg
        self.llm = llm
        self.retriever = retriever

    # ── run one question ───────────────────────────────────────────

    def answer(
        self,
        question_text: str,
        choices: Optional[Dict[str, str]] = None,
        gate_type: Optional[str] = None,
        entropy_threshold: Optional[float] = None,
        margin_threshold: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> DemoResult:
        if not question_text.strip():
            raise ValueError("Enter a question first.")

        cfg = self.cfg.model_copy(deep=True)
        if gate_type:
            cfg.gate.type = gate_type
        if entropy_threshold is not None:
            cfg.gate.entropy_threshold = entropy_threshold
        if margin_threshold is not None:
            cfg.gate.margin_threshold = margin_threshold

        question = UnifiedQuestion(
            question_id="ui",
            question_text=question_text.strip(),
            correct_answer="",                # no ground truth in a live demo
            dataset_source="ui",
            choices=choices or None,
        )

        gate = build_gate(cfg)
        _enable_token_capture(gate)

        retriever = self.retriever
        if retriever is not None and top_k is not None:
            retriever = _TopKRetriever(retriever, top_k)

        t0 = time.perf_counter()
        p5 = GatedPolicy(
            llm=self.llm, retriever=retriever, gate=gate,
            cite_sources=cfg.policy.cite_sources,
        ).answer(question)

        notes: List[str] = []
        details = p5.gate_details or {}
        ent = _entropy_details(details)

        if cfg.gate.type == "entropy" and not ent.get("available", True):
            raise ValueError(
                "The entropy gate cannot run on this backend — it needs "
                "full-vocabulary logits, which this backend does not expose. "
                "Switch to the local GGUF backend, or pick the margin, probe or "
                "ensemble gate."
            )
        if cfg.gate.type == "ensemble" and not ent.get("available", True):
            notes.append(
                "Entropy member abstained (no full-vocabulary logits on this "
                "backend); the ensemble voted on margin and probe only."
            )

        p3 = ClosedBookPolicy(llm=self.llm).answer(question)
        latency = time.perf_counter() - t0

        aligned = align_draft(ent.get("draft_tokens"), ent.get("per_token_entropy"))

        return DemoResult(
            verdict="RETRIEVE" if p5.retrieval_triggered else "SKIP",
            gate_name=p5.gate_name or cfg.gate.type,
            signal_value=p5.gate_signal_value,
            threshold=_threshold_for(cfg, p5.gate_name),
            gate_details=details,
            aligned=aligned,
            p5_answer=p5.answer_text,
            p3_answer=p3.answer_text,
            answers_identical=p5.answer_text.strip() == p3.answer_text.strip(),
            chunks=list(p5.retrieved_chunks),
            is_multiple_choice=question.is_multiple_choice(),
            query=question.question_text,
            latency_s=latency,
            notes=notes,
        )

    def close(self) -> None:
        self.llm.close()
        if self.retriever is not None:
            self.retriever.close()


def _threshold_for(cfg: ProjectConfig, gate_name: Optional[str]) -> Optional[float]:
    if gate_name == "entropy":
        return cfg.gate.entropy_threshold
    if gate_name == "margin":
        return cfg.gate.margin_threshold
    if gate_name == "ensemble":
        return float(cfg.gate.ensemble_min_votes)
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_ui_session.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/medrag_adaptive/ui/session.py tests/unit/test_ui_session.py
git commit -m "feat(ui): add DemoSession running P5 and P3 per question"
```

---

## Task 6: OpenAI-compatible backend

**Files:**
- Create: `src/medrag_adaptive/models/openai_backend.py`
- Test: `tests/unit/test_openai_backend.py`

- [ ] **Step 1: Write the failing test**

```python
"""tests/unit/test_openai_backend.py — response parsing, with no network.

The HTTP call is monkeypatched; what is under test is the parsing of an
OpenAI-shaped chat completion into the LLMBackend contract.
"""

from __future__ import annotations

import math

from medrag_adaptive.models.openai_backend import OpenAIBackend

_RESPONSE = {
    "choices": [{
        "message": {"content": "Epinephrine"},
        "logprobs": {"content": [
            {"token": "Epine", "logprob": -0.01,
             "top_logprobs": [{"token": "Epine", "logprob": -0.01},
                              {"token": "Diphen", "logprob": -4.6}]},
            {"token": "phrine", "logprob": -0.002,
             "top_logprobs": [{"token": "phrine", "logprob": -0.002},
                              {"token": "phrin", "logprob": -6.2}]},
        ]},
    }]
}


def _backend(monkeypatch):
    backend = OpenAIBackend(base_url="http://x/v1", model="m", api_key="k")
    monkeypatch.setattr(backend, "_post", lambda payload: _RESPONSE)
    return backend


def test_answer_returns_message_content(monkeypatch):
    assert _backend(monkeypatch).answer("prompt") == "Epinephrine"


def test_draft_reports_no_logits(monkeypatch):
    text, logits = _backend(monkeypatch).draft("prompt")
    assert text == "Epinephrine"
    assert logits is None          # no full-vocabulary logits over an API


def test_draft_with_tokens_returns_token_strings(monkeypatch):
    _text, _logits, tokens = _backend(monkeypatch).draft_with_tokens("prompt")
    assert tokens == ["Epine", "phrine"]


def test_get_top2_logprobs_shape_matches_margin_gate(monkeypatch):
    _text, top = _backend(monkeypatch).get_top2_logprobs("prompt")
    assert len(top) == 2
    assert abs(top[0]["Epine"] - (-0.01)) < 1e-9
    assert abs(top[0]["Diphen"] - (-4.6)) < 1e-9


def test_margin_gate_can_consume_this_backend(monkeypatch):
    from medrag_adaptive.data.schema import UnifiedQuestion
    from medrag_adaptive.gating.margin_gate import MarginGate

    q = UnifiedQuestion(question_id="q", question_text="t",
                        correct_answer="", dataset_source="ui")
    decision = MarginGate(threshold=0.3).decide(q, _backend(monkeypatch))
    assert decision.details["available"] is True
    assert decision.signal_value > 0.9      # near-certain draft → wide margin


def test_last_draft_topk_feeds_truncated_entropy(monkeypatch):
    backend = _backend(monkeypatch)
    backend.draft_with_tokens("prompt")
    assert len(backend.last_draft_topk) == 2
    assert math.isfinite(sum(backend.last_draft_topk[0].values()))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_openai_backend.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'medrag_adaptive.models.openai_backend'`

- [ ] **Step 3: Write the implementation**

Create `src/medrag_adaptive/models/openai_backend.py`:

```python
"""models/openai_backend.py — LLMBackend over any OpenAI-compatible chat endpoint.

Covers the OpenAI API itself and every server that mimics it (Ollama, vLLM,
llama.cpp-server, LM Studio). Used only by the demo UI: the dissertation's
measurements all come from LlamaBackend, because an API cannot supply the
full-vocabulary logits the entropy gate is defined on.

What survives the API boundary:
  answer()             exact
  get_top2_logprobs()  exact — the margin gate is unaffected
  draft_with_tokens()  token strings yes, logits no
  draft()              logits None → EntropyGate abstains, and EnsembleGate
                       already skips abstaining members

Uses stdlib urllib rather than the openai package to avoid adding a dependency
to a submission-frozen requirements.txt.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Tuple

import numpy as np

from medrag_adaptive.models.base import LLMBackend


class OpenAIBackend(LLMBackend):
    """Chat-completions backend with per-token top-k logprobs."""

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
        self.last_draft_topk = [
            {alt["token"]: alt["logprob"] for alt in entry.get("top_logprobs", [])}
            for entry in entries
        ]
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
        Returns up to top_logprobs entries per token, not exactly two. The
        margin gate takes the two largest, so a wider list is compatible — and
        the extra entries are what make a truncated entropy possible.
        """
        response = self._complete(prompt, max_tokens, want_logprobs=True)
        entries = self._logprob_entries(response)
        top = [
            {alt["token"]: alt["logprob"] for alt in entry.get("top_logprobs", [])}
            for entry in entries
        ]
        self.last_draft_topk = top
        return self._content(response), top

    def close(self) -> None:
        """Nothing to release — the backend holds no local resources."""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_openai_backend.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/medrag_adaptive/models/openai_backend.py tests/unit/test_openai_backend.py
git commit -m "feat(models): add OpenAI-compatible backend for the demo UI"
```

---

## Task 7: Gradio app

**Files:**
- Create: `src/medrag_adaptive/ui/app.py`

No unit tests: this file is widget wiring, and testing Gradio callbacks in-process buys
little. Its logic lives in `attribution.py` and `session.py`, both covered. Verification
is the manual smoke test in Task 9.

- [ ] **Step 1: Write the app**

Create `src/medrag_adaptive/ui/app.py`:

```python
"""ui/app.py — Gradio Blocks front end for the adaptive-gating demo.

Layout A from the design: a single scrolling column — settings accordion,
question, verdict banner, entropy heatmap, P5-vs-P3 answers, evidence. All
rendering logic lives in attribution.py and all orchestration in session.py;
this file only wires widgets to those.
"""

from __future__ import annotations

import html
from typing import Optional

import gradio as gr

from medrag_adaptive.models.openai_backend import OpenAIBackend
from medrag_adaptive.ui.attribution import (
    align_draft,
    render_chunks_html,
    render_entropy_html,
    truncated_entropy,
)
from medrag_adaptive.ui.session import DemoSession

CSS = """
.mr-heat { line-height: 2.1; }
.mr-tok { padding: 2px 4px; margin: 1px; border-radius: 3px;
          font-family: ui-monospace, monospace; font-size: 13px; }
.mr-note { font-size: 12px; opacity: .75; margin-top: 8px; font-style: italic; }
.mr-term { background: #ffe27a; color: #111; border-radius: 2px; }
.mr-chunk { border: 1px solid rgba(128,128,128,.35); border-radius: 6px;
            padding: 10px; margin-bottom: 8px; }
.mr-chunk-head { font-weight: 600; margin-bottom: 6px; }
.mr-chunk-body { font-size: 13px; line-height: 1.5; }
.mr-badge { display: inline-block; border: 1px solid rgba(128,128,128,.4);
            border-radius: 10px; padding: 0 7px; font-size: 11px;
            font-weight: 400; margin-left: 6px; }
.mr-verdict { padding: 10px 14px; border-radius: 6px; font-weight: 600;
              color: #fff; font-size: 15px; }
.mr-retrieve { background: #c0392b; }
.mr-skip { background: #27795b; }
"""


def _verdict_html(result) -> str:
    css_class = "mr-retrieve" if result.verdict == "RETRIEVE" else "mr-skip"
    signal = "n/a" if result.signal_value is None else f"{result.signal_value:.3f}"
    threshold = "n/a" if result.threshold is None else f"{result.threshold:.3f}"

    parts = [
        f'<div class="mr-verdict {css_class}">{result.verdict}'
        f" &nbsp;·&nbsp; {html.escape(result.gate_name)} gate"
        f" &nbsp;·&nbsp; signal {signal} vs τ {threshold}"
        f" &nbsp;·&nbsp; {result.latency_s:.1f}s</div>"
    ]

    votes = (result.gate_details or {}).get("votes")
    if votes:
        rendered = ", ".join(f"{name}: {vote}" for name, vote in votes.items())
        need = result.gate_details.get("min_votes")
        parts.append(
            f'<div class="mr-note">Member votes — {html.escape(rendered)} '
            f"(needs {need} to retrieve)</div>"
        )
    for note in result.notes:
        parts.append(f'<div class="mr-note">{html.escape(note)}</div>')
    return "".join(parts)


def build_app(session: DemoSession) -> gr.Blocks:
    with gr.Blocks(css=CSS, title="Adaptive Gating for Medical RAG") as demo:
        gr.Markdown(
            "## Adaptive gating for medical RAG\n"
            "Ask a clinical question. The gate decides whether the model needs "
            "to retrieve, and the panels below show why."
        )

        with gr.Accordion("Settings", open=False):
            with gr.Row():
                gate_type = gr.Dropdown(
                    ["ensemble", "entropy", "margin", "hallucination_probe"],
                    value=session.cfg.gate.type,
                    label="Gate",
                )
                top_k = gr.Slider(1, 10, value=session.cfg.retrieval.top_k,
                                  step=1, label="top-k chunks")
            with gr.Row():
                tau_h = gr.Slider(0.0, 3.0, value=session.cfg.gate.entropy_threshold,
                                  step=0.01, label="τ entropy (retrieve when H̄ >)")
                tau_m = gr.Slider(0.0, 1.0, value=session.cfg.gate.margin_threshold,
                                  step=0.01, label="τ margin (retrieve when M̄ <)")
            gr.Markdown("**Backend** — the local GGUF is loaded. Switch to an "
                        "OpenAI-compatible API below if you want.")
            with gr.Row():
                api_base = gr.Textbox(label="base_url",
                                      placeholder="https://api.openai.com/v1")
                api_model = gr.Textbox(label="model", placeholder="gpt-4o-mini")
                api_key = gr.Textbox(label="API key", type="password")
            with gr.Row():
                use_api = gr.Button("Use API backend")
                use_local = gr.Button("Use local GGUF")
            backend_status = gr.Markdown("Backend: **local GGUF**")

        question = gr.Textbox(
            label="Question", lines=3,
            placeholder="A patient develops anaphylactic shock after a bee sting. "
                        "What is the first-line drug?",
        )
        with gr.Row():
            choice_a = gr.Textbox(label="A", scale=1)
            choice_b = gr.Textbox(label="B", scale=1)
            choice_c = gr.Textbox(label="C", scale=1)
            choice_d = gr.Textbox(label="D", scale=1)
        gr.Markdown("*Leave the choices blank for a free-text question.*")
        submit = gr.Button("Run", variant="primary")

        verdict_out = gr.HTML()
        gr.Markdown("### Draft-token entropy")
        gr.Markdown(
            "Each chip is one token of the model's unretrieved draft, shaded by "
            "H(p_t). Darker means the model had less idea what came next."
        )
        heatmap_out = gr.HTML()
        with gr.Row():
            p5_out = gr.Textbox(label="P5 — gated", lines=6, interactive=False)
            p3_out = gr.Textbox(label="P3 — closed book", lines=6, interactive=False)
        identical_out = gr.Markdown()
        gr.Markdown("### Evidence")
        chunks_out = gr.HTML()

        # ── callbacks ──────────────────────────────────────────────

        def _run(q, a, b, c, d, gate, th, tm, k):
            choices = {k_: v.strip() for k_, v in
                       (("A", a), ("B", b), ("C", c), ("D", d)) if v and v.strip()}
            try:
                result = session.answer(
                    q, choices=choices or None, gate_type=gate,
                    entropy_threshold=th, margin_threshold=tm, top_k=int(k),
                )
            except Exception as exc:                     # surfaced, not swallowed
                return (f'<div class="mr-verdict mr-retrieve">Error</div>'
                        f'<div class="mr-note">{html.escape(str(exc))}</div>',
                        "", "", "", "", "")

            aligned = result.aligned
            note = ""
            if not aligned.tokens and isinstance(session.llm, OpenAIBackend):
                topk = truncated_entropy(session.llm.last_draft_topk)
                if topk:
                    aligned = align_draft(
                        [f"t{i}" for i in range(len(topk.per_token))], topk.per_token
                    )
                    note = ("Entropy computed from the API's top-k logprobs — a "
                            "truncated lower bound, not comparable to the calibrated τ.")

            heat = render_entropy_html(aligned, threshold=result.threshold or 0.0)
            if note:
                heat += f'<div class="mr-note">{html.escape(note)}</div>'

            identical = ""
            if result.answers_identical:
                identical = ("*The gate skipped retrieval, so both columns ran the "
                             "same closed-book prompt and are identical by construction.*")

            chunks_html = (
                render_chunks_html(
                    result.chunks, result.query,
                    lexical_only=session.cfg.policy.retrieval_mode != "bm25",
                )
                if result.chunks
                else '<div class="mr-note">The gate skipped retrieval, so no '
                     "evidence was fetched.</div>"
            )
            return (_verdict_html(result), heat, result.p5_answer,
                    result.p3_answer, identical, chunks_html)

        def _switch_api(base, model, key):
            if not base or not model:
                return "Backend unchanged — base_url and model are both required."
            session.llm = OpenAIBackend(base_url=base, model=model, api_key=key or None)
            return f"Backend: **API** — `{model}` at `{base}`"

        def _switch_local():
            if session.local_llm is None:
                return "No local GGUF was loaded at startup."
            session.llm = session.local_llm
            return "Backend: **local GGUF**"

        submit.click(
            _run,
            [question, choice_a, choice_b, choice_c, choice_d,
             gate_type, tau_h, tau_m, top_k],
            [verdict_out, heatmap_out, p5_out, p3_out, identical_out, chunks_out],
        )
        use_api.click(_switch_api, [api_base, api_model, api_key], backend_status)
        use_local.click(_switch_local, None, backend_status)

    # One Llama instance, not thread-safe: serialise submissions.
    demo.queue(default_concurrency_limit=1)
    return demo
```

- [ ] **Step 2: Add `local_llm` to DemoSession**

`app.py` calls `session.local_llm`. In `src/medrag_adaptive/ui/session.py`, at the end
of `DemoSession.__init__`:

```python
        # The originally-loaded backend, kept so the UI can switch to an API
        # backend and back without paying to reload the GGUF.
        self.local_llm = llm
```

- [ ] **Step 3: Verify the module imports cleanly**

Run: `python -c "import medrag_adaptive.ui.app; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add src/medrag_adaptive/ui/app.py src/medrag_adaptive/ui/session.py
git commit -m "feat(ui): add Gradio Blocks front end"
```

---

## Task 8: Launcher

**Files:**
- Create: `scripts/run_demo.py`

- [ ] **Step 1: Write the launcher**

```python
r"""scripts/run_demo.py — launch the adaptive-gating demo UI.

    python scripts/run_demo.py \
        --config configs/experiments/mirage_medcorp.yaml \
        --model models/Llama-3.2-3B-Instruct-Q4_K_M.gguf

--model is an explicit flag because the GGUF filename on disk differs from the
one in the YAML (a known trap in this repo).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive-gating demo UI")
    parser.add_argument("--config", default="configs/experiments/mirage_medcorp.yaml",
                        help="experiment YAML")
    parser.add_argument("--policy", default="configs/policies/p5_gated_entropy.yaml")
    parser.add_argument("--hardware", default="configs/hardware_medium.yaml")
    parser.add_argument("--model", default=None, help="override model.gguf_path")
    parser.add_argument("--bm25-index", default=None)
    parser.add_argument("--faiss-index", default=None)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-model", action="store_true",
                        help="start without a local GGUF (API backend only)")
    args = parser.parse_args()

    from medrag_adaptive.config import load_config
    from medrag_adaptive.retrieval.factory import build_retriever
    from medrag_adaptive.ui.app import build_app
    from medrag_adaptive.ui.session import DemoSession

    cfg = load_config(base="configs/base.yaml", hardware=args.hardware,
                      policy=args.policy, experiment=args.config)
    if args.model:
        cfg.model.gguf_path = args.model
    cfg.policy.name = "p5_gated"

    llm = None
    if not args.no_model:
        from medrag_adaptive.models.llama_backend import llama_backend_from_config
        print(f"Loading {cfg.model.gguf_path} ...")
        llm = llama_backend_from_config(cfg)

    try:
        retriever = build_retriever(cfg, bm25_index=args.bm25_index,
                                    faiss_index=args.faiss_index)
    except (FileNotFoundError, OSError) as exc:
        print(f"[warn] no retriever: {exc}\n       the SKIP path still works.")
        retriever = None

    session = DemoSession(cfg, llm, retriever)
    build_app(session).launch(server_name="127.0.0.1", server_port=args.port,
                              share=False, inbrowser=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the CLI parses**

Run: `python scripts/run_demo.py --help`
Expected: usage text listing every flag, exit 0

- [ ] **Step 3: Commit**

```bash
git add scripts/run_demo.py
git commit -m "feat(ui): add demo launcher script"
```

---

## Task 9: Regression guard and smoke test

**Files:** none modified

- [ ] **Step 1: Run the complete existing test suite**

Run: `python -m pytest tests/ -q`
Expected: every previously-passing test still passes, and no existing test file was
edited. **If any existing test fails or had to change, stop and report** — the whole
premise of this work is that reported numbers do not move.

- [ ] **Step 2: Confirm no existing test file was touched**

Run: `git diff --stat HEAD~8 -- tests/`
Expected: only the three new test files appear — `test_attribution.py`,
`test_ui_session.py`, `test_openai_backend.py`, `test_draft_with_tokens.py`.
`tests/conftest.py` must NOT appear.

- [ ] **Step 3: Manual smoke test**

Run:
```bash
python scripts/run_demo.py --model models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

Ask: *"A patient develops anaphylactic shock after a bee sting. What is the first-line
drug?"* with choices A=Diphenhydramine, B=Epinephrine, C=Hydrocortisone, D=Salbutamol.

Check:
1. A verdict banner appears with a numeric signal and τ.
2. The heatmap shows one chip per draft token, and hovering shows an H value.
3. If the gate retrieved, chunks appear with highlighted terms; if it skipped, the
   evidence panel says so and the two answers are flagged identical.
4. Note whether the "entropy values had no corresponding generated token" message
   appears — that is the §6 slicing question answering itself. Record the answer.

- [ ] **Step 4: Commit any fixes found during smoke testing**

```bash
git add -A
git commit -m "fix(ui): smoke-test corrections"
```

---

## Self-Review

**Spec coverage:** §3 architecture → Tasks 2/5/7/8. §4 shipped-code edits → Task 1.
§5 run sequence → Task 5. §6 slicing → `align_draft` (Task 2) + smoke step 3.4.
§7 panels → Task 7. §8 highlighting → Task 3. §9 API backend → Tasks 4 and 6.
§10 error handling → `_run` try/except and the launcher's retriever guard.
§11 testing → Tasks 1-6 and 9.

Not implemented from the spec, deliberately: per-stage profiling via
`evaluation/profiler.py` (§5 item 4). `DemoResult.latency_s` covers the whole submission
with `time.perf_counter`, which is what the banner shows; wiring `profiler.py` in would
add a codecarbon-adjacent dependency to an interactive path for no demo benefit. Note
this if the report claims otherwise.

**Type consistency:** `AlignedDraft(tokens, entropies, dropped_entropies, dropped_tokens)`
is constructed only by `align_draft` and consumed by `render_entropy_html` and
`DemoResult.aligned`. `TruncatedEntropy(per_token, mean)` is produced by
`truncated_entropy` and consumed in `app._run`. `DemoResult` field names used in
`_verdict_html` and `_run` all exist on the dataclass. `session.local_llm` is set in
Task 7 Step 2 and read in `app._switch_local`.
