# Adaptive-Gating Demo UI — Design

**Date:** 2026-07-26
**Status:** Approved (pending spec review)
**Context:** Dissertation submission 6 Aug 2026. This is a demo artefact, not a research
contribution. It must not perturb any number already in the report.

---

## 1. Purpose

A local Gradio app that lets a person type a medical question and watch the P5 gate
decide whether to retrieve — showing *which draft tokens the model was unsure about* and
*which query terms pulled each retrieved chunk*.

**Audience:** the viva demo and appendix screenshots. Single user, localhost, no deploy,
no auth, no sharing.

**Success criteria:**

1. An examiner types their own clinical question and sees a gate decision with its
   numeric justification within ~90 s on CPU.
2. The gate decision shown is produced by the same code the evaluation ran — not a
   reimplementation.
3. Every reported number in the dissertation is bit-identical before and after this work.

**Explicitly out of scope:** query-term retrieval-contribution bars (BM25 decomposition /
vector leave-one-out), the four-gates-side-by-side comparison panel, P1 in the answer
comparison, deployment, multi-user, authentication.

---

## 2. Decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | Local GGUF **and** OpenAI-compatible API backends | User drives both; OpenAI-compatible also covers Ollama / vLLM / llama.cpp-server |
| D2 | Layout A — single scrolling column, settings in a collapsed accordion | Crops into clean vertical appendix figures; least layout code |
| D3 | Panels: draft-token entropy heatmap + retrieved-chunk term highlighting | The two the user asked for; the other two cut under YAGNI |
| D4 | Each submit runs **P5 + P3** side by side | Shows what retrieval actually bought, for one extra LLM call |
| D5 | Token strings via a **new non-abstract `draft_with_tokens()`** on `LLMBackend`, not a kwarg on `draft()` | Adding a third tuple element to `draft()` would break all three existing call sites and the conftest mock. A new method with a working default breaks nothing |
| D6 | UI truncates to `min(len(tokens), len(entropies))` for display; **does not** change the gate's slicing | See §6 |
| D7 | API backends compute truncated `H_top20`, displayed with an explicit "not comparable to reported τ" label | Keeps the headline visualisation alive on the API path while staying honest |

---

## 3. Architecture

Five new files, three default-off edits to shipped code.

```
src/medrag_adaptive/ui/attribution.py      # pure functions, no LLM — unit-tested
src/medrag_adaptive/ui/session.py          # owns backend + retriever, runs P5 then P3
src/medrag_adaptive/ui/app.py              # Gradio Blocks, layout A
src/medrag_adaptive/models/openai_backend.py
scripts/run_demo.py                        # launcher
```

In-process Gradio Blocks. Rejected alternatives: a FastAPI service with a Gradio client
(buys deploy separation this project does not need), and hand-rolled HTML/JS (costs days
that belong to the report).

`session.py` builds its policies through the existing
`build_policy(cfg, llm, retriever)` factory. The demo therefore exercises the evaluation
code path rather than a parallel one — the claim that it shows real gate behaviour
depends on this.

### Unit boundaries

- **`attribution.py`** — takes data, returns HTML strings. Imports no model, no
  retriever, no Gradio. Fully testable without an LLM.
- **`session.py`** — owns the expensive singletons and the run sequence. Returns a
  `DemoResult` dataclass. Knows nothing about HTML.
- **`app.py`** — Gradio wiring only: widgets, callbacks, layout. Contains no entropy or
  highlighting logic.
- **`openai_backend.py`** — implements `LLMBackend`. Knows nothing about the UI.

---

## 4. Edits to shipped code

| File | Change | Default |
|---|---|---|
| `models/base.py` | **new non-abstract** `draft_with_tokens(prompt, max_tokens=48) -> (text, logits, tokens \| None)`, whose default implementation delegates to `self.draft()` and returns `tokens=None` | inherited by every backend |
| `models/llama_backend.py` | override `draft_with_tokens` — same `create_completion` call plus `logprobs=2`, returning `choices[0]["logprobs"]["tokens"]` | `draft()` untouched |
| `gating/entropy_gate.py` | `EntropyGate(..., keep_tokens: bool = False)`; when `True`, call `draft_with_tokens` and add `details["draft_tokens"]` | `False` |

Crucially, `draft()` keeps its two-element return. The three existing call sites
(`entropy_gate.py:55`, `hallucination_probe_gate.py:50-51`) and `MockLLMBackend` in
`tests/conftest.py` are **not touched at all**. `MockLLMBackend` inherits the default
`draft_with_tokens` for free, so UI tests can run against it.

At `keep_tokens=False` the entropy gate executes the identical code path it does today.

`temperature=0.0` means `logprobs=2` does not alter the greedy output.
`scripts/plot_entropy_attribution.py` already depends on this and produces the expected
draft.

Nothing else in `src/` changes.

---

## 5. Run sequence

On submit, `session.answer(question, choices, settings)`:

1. Build a `UnifiedQuestion` (`question_id = f"ui-{n}"`, `choices=None` for the
   free-text path).
2. `GatedPolicy.answer(q)` → `PolicyResult` carrying `gate_details`
   (`per_token_entropy`, `draft_tokens`, per-member votes, thresholds).
3. `ClosedBookPolicy.answer(q)` → the P3 text.
4. Assemble `DemoResult`: verdict, signal value, threshold, aligned token/entropy pairs,
   P5 text, P3 text, retrieved chunks (title, source, score, text), per-stage wall clock
   from the existing `evaluation/profiler.py`.

Cost: ~5 LLM calls for the ensemble gate plus the P5 answer, plus 1 for P3.

---

## 6. The `_scores` slicing mismatch — RESOLVED 2026-07-26

**Outcome: the concern was real, but in the opposite direction to the prediction below.**
Measured on the anaphylaxis question with the 3B: the draft produced 48 tokens but only
**47** entropies — the array is one *short*, not long.

Cause: `_extract_generation_logits` starts at `len(tokenize(prompt))`, one row too high,
because `tokenize()` prepends BOS and the completion's context does not. For that draft
`_scores` held 123 rows against a computed `n_prompt` of 76, so the prompt really
occupied 75. Confirmed by argmax matching under greedy decode (the argmax of a row must
be the token that row produced): start at `rows - n_generated` matched 48/48 tokens
(100%); start at `n_prompt` matched 0/48.

So the shipped gate's H̄ averages generated tokens **1…N−1, omitting token 0**.

| | H̄ | tokens | hottest tokens |
|---|---|---|---|
| Shipped gate | 0.7092 | 47 of 48 | `.` 2.95, `.` 2.59, ` to` 2.34 |
| Correctly aligned | 0.7628 | 48 of 48 | ` \n\n` 3.28, ` It` 2.95, ` Ep` 2.59 |

The old hottest-token list being punctuation is the signature of the off-by-one; aligned,
the drug-name token ` Ep` surfaces, which is what the attribution figure exists to show.

**Decision (user, 2026-07-26): the gate is NOT corrected, and the demo reports the
shipped H̄.** τ = 0.70 was calibrated on the shipped signal and 0.7092 sits 1.3% above it,
so re-slicing would move near-threshold decisions, change the realised retrieval budget,
and require re-running calibration and evaluation on both models. `draft_with_tokens`
therefore uses the shipped extractor and drops `tokens[0]` so the token list is
index-aligned with the 47 rows it returns. The heatmap renders exactly the tokens the
gate's average covers, and the UI says so. Written up in
`docs/limitations-entropy-offset.tex`.

---

### Original concern, retained for the record

`LlamaBackend.draft()` slices `_scores[n_prompt : n_prompt + max_tokens]` — by the
**requested** token count. `scripts/plot_entropy_attribution.py` slices by the
**actual** generated count. If a draft stops early and `_scores` is `n_ctx`-tall, the
gate averages entropy over rows that were never generated, inflating H̄.

Aligning tokens to entropies in the UI surfaces this: the two lengths will differ.

**This design does not change the gate's slicing.** Changing it would move H̄ on every
question, invalidating the shipped τ = 0.70 / 0.70 calibration and every result in the
report, eight days before submission.

Instead:

- The UI truncates to `min(len(tokens), len(entropies))` for display only.
- When the lengths differ, the UI shows a note stating that N entropy values had no
  corresponding generated token and were omitted from the display.
- The verdict banner always shows the gate's own H̄ — the untruncated one it actually
  decided on — never a recomputed display-only mean. The two must never be conflated.

If the lengths always match, the concern is disproved for free.

---

## 7. Panels (layout A, top to bottom)

1. **Settings accordion**, collapsed by default: backend radio (local GGUF |
   OpenAI-compatible, with base_url / model / API key fields), gate dropdown
   (entropy | margin | hallucination_probe | ensemble), τ_H and τ_M sliders seeded from
   the loaded config, retrieval mode (bm25 | vector | hybrid), top-k.
2. **Question** textarea, plus four optional choice fields. All blank → free-text path.
3. **Verdict banner** — `RETRIEVE — H̄ 1.04 > τ 0.70` or `SKIP — H̄ 0.41 ≤ τ 0.70`. For
   the ensemble gate, also the per-member votes and the min-votes rule.
4. **Entropy heatmap** — one coloured `gr.HTML` span per draft token, colour scaled to
   H(p_t), value on hover. HTML spans rather than matplotlib: they reflow, stay legible
   at any draft length, and screenshot cleanly.
5. **Answers**, two columns: P5 gated | P3 closed-book. On SKIP the two are identical by
   construction; the UI states this rather than implying a comparison happened.
6. **Evidence** — top-k chunks with title, source, score, and matched query terms
   highlighted. Hidden with an explanatory line when the gate chose SKIP.

---

## 8. Term highlighting

`attribution.py` tokenises the query with `bm25_retriever._tokenize` — the same
`[a-z0-9]+` regex the BM25 index was built with — so highlighted terms are exactly the
terms that contributed to the score. A separate word-splitter would produce
plausible-looking highlights that are not what retrieval saw.

Chunk text is HTML-escaped **before** highlight spans are injected. Corpus text is
untrusted input as far as the browser is concerned.

Limitation to note in the report: on `vector` or `hybrid` retrieval mode, lexical
highlighting shows lexical overlap only. It does not explain the embedding leg. The UI
labels the panel accordingly when the mode is not `bm25`.

---

## 9. API backend and truncated entropy

`OpenAIBackend` implements `LLMBackend` against any OpenAI-compatible `/completions`
endpoint (OpenAI, Ollama, vLLM, llama.cpp-server).

- `get_top2_logprobs()` — exact; the margin gate is unaffected.
- `answer()` — exact.
- `draft()` — returns `logits=None` (no full-vocab logits exist) plus the token strings.
- Truncated entropy: from the top-`k` logprobs the endpoint returns (typically 20),
  `H_topk = -Σ_{i≤k} p_i log p_i`. This is a **lower bound** on true entropy — it
  discards the tail mass.

Because `EntropyGate` reads `logits`, the truncated path cannot go through it unchanged.
`session.py` computes `H_topk` itself and renders it in the entropy panel, with a
persistent label: *"H from top-20 logprobs — truncated lower bound, not comparable to the
calibrated τ = 0.70"*.

**Gate selection on the API backend.** `EntropyGate` receiving `logits=None` returns
`retrieve=True` with `available: False`. `EnsembleGate` already skips any member whose
details say `available: False` (`ensemble_gate.py:51-53`) — it abstains rather than
voting — so the ensemble degrades to margin + probe with **no new logic in
`session.py`**. With two available members and `min_votes=2`, `degraded` stays `False`
and the standard majority rule applies. The verdict banner surfaces the abstention so a
viewer is not left thinking three gates voted.

One thing `session.py` must *not* do is set `cfg.model.logits_all=False` for the API
backend: `build_gate` (`policies/factory.py:68-70`) reads that flag and downgrades all
the way to probe-only, which would needlessly discard the margin gate — margin works
fine over an API, since it reads top-2 logprobs rather than the logit buffer. Leave the
flag `True` and let `EntropyGate` abstain at runtime.

If the user picks `entropy` as the *sole* gate while on the API backend, the UI refuses
the run with a message pointing at the local GGUF backend, rather than reporting a
`retrieve=True` that is a fallback default and not a measurement.

The three dissertation-reported gates on the local GGUF path are untouched by any of
this.

---

## 10. Error handling

| Failure | Behaviour |
|---|---|
| GGUF missing / load fails | Banner naming the path tried; app stays up |
| Index missing | Evidence panel explains; gate and closed-book still run |
| API unreachable / bad key | Banner with the HTTP status; no traceback in the UI |
| Draft returns zero tokens | Heatmap shows "no draft tokens"; verdict still rendered |
| Concurrent submits | `concurrency_limit=1` on the Gradio queue — one `Llama` instance, not thread-safe |

`share=False`. Localhost bind only. The API key field is a password-type input and is
never written to disk or logged.

---

## 11. Testing

`tests/unit/test_attribution.py` — no LLM required:

- token/entropy length mismatch in both directions, and the equal case
- empty token list
- highlighting: case-insensitivity, punctuation boundaries, no-match, repeated terms
- HTML escaping of hostile chunk text (`<script>`, `&`, quotes)
- truncated-entropy arithmetic on a hand-computed distribution

`tests/unit/test_ui_session.py` — against the existing `MockRetriever` and a mock backend
from `tests/conftest.py`: RETRIEVE path populates chunks, SKIP path leaves them empty and
sets the identical-answers flag.

Regression guard: the existing test suite must pass unchanged after the §4 edits. That
is the evidence that no reported number moved.

---

## 12. Launcher

```
python scripts/run_demo.py --config configs/experiments/mirage_medcorp.yaml \
                           --model models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
                           --port 7860
```

`--model` is an explicit flag because the GGUF filename on disk differs from the config's
(a known trap in this repo).
