# Qwen-7B Chapters, Re-run and Limitations — Implementation Plan

**Created:** 2026-07-30 · **Submission:** 6 Aug 2026 (7 days)

**Goal:** Add two report chapters (Qwen-7B findings; Qwen-vs-Llama comparison), generate
Qwen tables and figures, write the budget-mismatch and gate-saturation limitations, and
make the Qwen open-ended re-run feasible.

**Ordering principle:** the write-up depends only on data already on disk, so it is done
first and is never blocked. The re-run is the one risky item and is deliberately
sequenced so that a failure costs nothing already written.

---

## Measured facts this plan is built on

All figures below were computed from `results/raw_logs/` on 2026-07-30 and are the
numbers the chapters should quote.

### MCQ accuracy (200 MIRAGE questions, identical set for both models)

| Run | Acc | 95% Wilson CI | Retrieval |
|---|---|---|---|
| Llama-3.2-3B P3 closed-book | 0.620 | [0.551, 0.684] | 0% |
| Llama-3.2-3B P5 gated | 0.540 | [0.471, 0.608] | 49.5% |
| Llama-3.2-3B P1 always | 0.455 | [0.387, 0.524] | 100% |
| Llama-3.2-3B P4 hybrid | 0.455 | [0.387, 0.524] | 100% |
| Qwen2.5-7B P3 closed-book | 0.745 | [0.680, 0.800] | 0% |
| Qwen2.5-7B P5 gated | 0.720 | [0.654, 0.778] | 31.5% |

Paired McNemar, closed-book, n=200: both right 110, only-Llama 14, only-Qwen 39,
neither 37. χ² (continuity-corrected) = 10.87, p < 0.001.

### Open-ended (PubMedQA, mean token-F1 — binary accuracy is undefined here)

| Run | n | F1 | ±95% CI | Refusal |
|---|---|---|---|---|
| Llama P3 open (closed) | 200 | 0.1902 | ±0.0094 | 1.5% |
| Llama P5 open (pubmedqa corpus) | 200 | 0.1905 | ±0.0109 | 78.5% |
| Llama P1 open (medcorp) | 200 | 0.2200 | ±0.0122 | 90.0% |
| Llama P5 open (medcorp) | 200 | 0.2114 | ±0.0110 | 71.5% |
| Qwen P3 open (closed) | 200 | 0.2556 | ±0.0134 | 0.5% |
| Qwen P5 open (gated) | **35** | 0.2241 | ±0.0221 | 68.6% |

On the common 35 questions: Llama P5 0.2048 vs Qwen P5 0.2241 — gap +0.019 against CIs of
±0.023, i.e. **not significant**.

### Gate signal distributions (the saturation evidence)

| Run | member | τ | signal min / med / max | fires |
|---|---|---|---|---|
| Llama MCQ | entropy | 0.700 | 0.200 / 0.718 / 1.532 | 52.0% |
| Llama MCQ | margin | 0.700 | 0.462 / 0.695 / 0.892 | 53.5% |
| Llama MCQ | probe | — (letter_match) | — | 40.0% |
| Llama open | entropy | 0.700 | 0.324 / 0.877 / 1.744 | 75.5% |
| Llama open | margin | 0.700 | 0.346 / 0.624 / 0.853 | 79.0% |
| Llama open | probe | 0.700 (F1) | 0.071 / 0.509 / 1.000 | 87.0% |
| Qwen MCQ | entropy | 0.187 | 0.000 / 0.241 / 0.960 | 34.5% |
| Qwen MCQ | margin | 0.899 | 0.712 / 0.927 / 1.000 | 40.5% |
| Qwen MCQ | probe | — (letter_match) | — | 7.5% |
| **Qwen open** | **entropy** | **0.187** | **0.215** / 0.540 / 1.001 | **100%** |
| **Qwen open** | **margin** | **0.899** | 0.658 / 0.769 / **0.854** | **100%** |
| Qwen open | probe | 0.700 (F1) | 0.077 / 0.538 / 0.836 | 82.9% |

The two bolded rows are the finding: entropy fires when signal > τ and its **minimum**
(0.215) exceeds τ (0.187); margin fires when signal < τ and its **maximum** (0.854) is
below τ (0.899). Not one of the 35 questions falls on the other side of either threshold.
Both τ were fitted on MCQ drafts (`calib50` is MIRAGE MCQ) and applied unchanged to
open-ended drafts, whose signal distribution is different. Llama's 0.70/0.70 happens to
land inside its open-ended distributions, which is why it saturates less — a coincidence,
not a design.

Median-targeting τ for a 50% per-member budget on open-ended:
Qwen entropy **0.540**, margin **0.769**; Llama entropy **0.877**, margin **0.624**;
probe F1 τ ≈ **0.51** (Llama) / **0.54** (Qwen).

### Latency (the re-run blocker)

| Run | s/question | retrieve path | skip path |
|---|---|---|---|
| Qwen P3 MCQ | 0.1 | — | 0.1 |
| Qwen P3 open | 2.3 | — | 2.3 |
| Qwen P5 MCQ | 300.1 | 492.7 | **211.4** |
| Qwen P5 open | 531.6 | 531.6 | — (never skipped) |
| Llama P5 MCQ | 152.7 | 166.2 | 139.5 |
| Llama P5 open | 380.1 | 438.8 | 159.1 |

P5 adds four 48-token gate generations over P3. On the GPU those cost single-digit
seconds, yet the Qwen P5 **skip** path — no retrieval, no long prompt — costs 211 s.
Roughly 200 s per question is unaccounted for. At the observed rate a 200-question
open-ended run needs 29.5 h even at 30% retrieval, which is why the original run died at
35. **Hypothesis (unproven): `logits_all=True` makes llama-cpp materialise
full-vocabulary logits for every context position on each gate call — for Qwen's 152k
vocab at n_ctx 2048 that is ~1.2 GB per call, four times per question.** Task 1 tests
this before anything is built on it.

---

## Task 0: Decisions required before Task 3

- [ ] **Target retrieval budget for open-ended.** MCQ used ~50% (the 3B's realised rate).
      Open-ended could match that, or match the 3B's *open-ended* rate (79%). Matching 50%
      makes MCQ and open comparable within a model; matching 79% makes Llama and Qwen
      comparable within open-ended. Pick one and state it in the chapter.
- [ ] **Fate of the existing n=35 result.** Keep as a caveated preliminary row, or discard
      once the re-run lands. Recommend: keep in the chapter as the *saturation evidence*
      (it is the cleanest demonstration that τ did not transfer), and report accuracy from
      the re-run.
- [ ] **If Task 1 finds no fix**, choose: (a) 100-question stride-sampled re-run ≈ 12 h,
      (b) keep n=35 and re-sample it by stride so it is at least unbiased ≈ 5 h,
      (c) drop the Qwen gated-open row and report only Qwen P3 open (already complete at
      n=200).

---

## Task 1: Diagnose the P5 latency (blocks Task 3)

**Files:** Create `scripts/profile_p5_latency.py` · Test: manual, timing output

- [ ] **Step 1: Write a profiling harness that times each stage separately**

```python
r"""scripts/profile_p5_latency.py — where does a P5 question's time actually go?

P5 costs ~300 s/question on Qwen against 0.1 s for P3 on the same model and GPU,
including on the SKIP path where nothing is retrieved. Four 48-token gate
generations cannot account for that. This times each stage in isolation so the
cost is attributed rather than guessed at.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def stage(label, fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    dt = time.perf_counter() - t0
    print(f"{label:<44}{dt:8.2f} s")
    return out, dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-ctx", type=int, default=2048)
    ap.add_argument("--n-gpu-layers", type=int, default=-1)
    args = ap.parse_args()

    from medrag_adaptive.models.llama_backend import LlamaBackend
    from medrag_adaptive.models.prompts import build_closed_book_prompt, build_draft_prompt

    question = ("A patient develops anaphylactic shock after a bee sting. "
                "What is the first-line drug?")
    choices = {"A": "Diphenhydramine", "B": "Epinephrine",
               "C": "Hydrocortisone", "D": "Salbutamol"}

    for logits_all in (True, False):
        print(f"\n===== logits_all={logits_all} =====")
        _, load_s = stage("model load", LlamaBackend,
                          gguf_path=args.model, n_ctx=args.n_ctx, n_threads=4,
                          logits_all=logits_all, temperature=0.0,
                          max_new_tokens=256, seed=42, verbose=False,
                          n_gpu_layers=args.n_gpu_layers)
        llm = _
        draft_prompt = build_draft_prompt(question, choices)
        answer_prompt = build_closed_book_prompt(question, choices)

        stage("draft() 48 tok  [entropy gate]", llm.draft, draft_prompt, 48)
        stage("get_top2_logprobs() 48 tok [margin]", llm.get_top2_logprobs,
              draft_prompt, 48)
        stage("draft() 48 tok  [probe A]", llm.draft, draft_prompt, 48)
        stage("draft() 48 tok  [probe B]", llm.draft, draft_prompt, 48)
        stage("answer() closed-book", llm.answer, answer_prompt)
        llm.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it on the Qwen GGUF (Colab, GPU)**

Run:
```
python scripts/profile_p5_latency.py --model models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf
```
Expected: a per-stage breakdown for both `logits_all` settings. The hypothesis is
confirmed if the `logits_all=True` generations are one to two orders of magnitude slower
than the `logits_all=False` ones.

- [ ] **Step 3: Record the outcome in this plan and pick a branch**

- If `logits_all` is the cause → Task 1b.
- If the cost is in retrieval (`BM25Okapi.get_scores` is pure-Python and linear in corpus
  size; medcorp is large) → the fix is caching or swapping the BM25 implementation, and
  the skip-path cost is then *not* explained, so profile retrieval separately.
- If neither → do not proceed to Task 3; take Task 0 branch (c).

- [ ] **Step 4 (Task 1b, only if confirmed): reduce the logits cost without changing any signal**

The constraint is absolute: entropy is computed from the full-vocabulary softmax, so the
returned distribution must be numerically identical. Candidate: keep `logits_all=True`
only for the entropy gate's own call and use `logits_all=False` for the margin and two
probe calls, which read `logprobs` and never touch `_scores`. That is 1 expensive call per
question instead of 4.

**Verification gate:** re-run the 50-question Qwen calibration and confirm every logged
`mean_entropy` matches the previous run bit-for-bit. If any differs, revert — a faster
gate that changes the signal is worthless.

- [ ] **Step 5: Commit**

```bash
git add scripts/profile_p5_latency.py
git commit -m "perf: add P5 stage-level latency profiler"
```

---

## Task 2: Open-ended calibration tooling

**Files:** Modify `scripts/make_calibration_set.py` · Test:
`tests/unit/test_calibration_set.py`

`run_threshold_sweep.py` already replays open-ended logs (`--logs file.jsonl:OPEN`), so
only the *sampling* side is missing: `make_calibration_set.py` is MIRAGE-specific.

- [ ] **Step 1: Write the failing test**

```python
"""tests/unit/test_calibration_set.py — stride sampling over the open-ended set."""

from medrag_adaptive.evaluation.loading import stride_sample


def test_stride_sample_spans_the_whole_range():
    items = list(range(200))
    got = stride_sample(items, n=50)
    assert len(got) == 50
    assert got[0] == 0
    assert got[-1] > 150          # a prefix would end near 50


def test_stride_sample_n_larger_than_population():
    assert stride_sample([1, 2, 3], n=10) == [1, 2, 3]


def test_stride_sample_is_deterministic():
    items = list(range(200))
    assert stride_sample(items, n=40) == stride_sample(items, n=40)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/unit/test_calibration_set.py -v`
Expected: FAIL — `ImportError: cannot import name 'stride_sample'`

- [ ] **Step 3: Add `stride_sample` to `src/medrag_adaptive/evaluation/loading.py`**

```python
def stride_sample(items: list, n: int) -> list:
    """
    Take every k-th item so the subset spans the whole ordered population.

    A prefix is not a sample here: datasets are concatenated in a fixed order, so
    the first N questions are one or two subjects rather than a cross-section.
    Measured on the 3B's MCQ signals, a prefix-fitted threshold overshot the
    retrieval budget by 4-6.5pp and the error did not shrink with N, while a
    stride sample of 50 was within 1pp.
    """
    if n >= len(items):
        return list(items)
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]
```

- [ ] **Step 4: Generalise `make_calibration_set.py` to either dataset**

Add `--dataset {mirage,pubmedqa}` and `--out`, defaulting to the existing MIRAGE
behaviour so the current invocation is unchanged. For `pubmedqa` it reads
`data/raw/openqa/pubmedqa_labeled.jsonl` and writes
`data/raw/openqa/calib40_open.json`. Route both through `stride_sample` so the two
datasets cannot drift apart in sampling policy.

- [ ] **Step 5: Run tests and generate the open-ended calibration set**

Run: `python -m pytest tests/unit/test_calibration_set.py -v` → 3 passed
Run: `python scripts/make_calibration_set.py --dataset pubmedqa -n 40`
Expected: `data/raw/openqa/calib40_open.json` with 40 stride-sampled questions.

- [ ] **Step 6: Commit**

```bash
git add scripts/make_calibration_set.py src/medrag_adaptive/evaluation/loading.py \
        tests/unit/test_calibration_set.py
git commit -m "feat(calib): stride-sampled calibration sets for open-ended"
```

---

## Task 3: Re-run Qwen open-ended (gated by Tasks 1 and 2)

- [ ] **Step 1: Harvest open-ended gate signals on the 40-question calibration set**

Run P5 open on `calib40_open.json` with the *current* thresholds. The answers do not
matter; only `qvault.gate_details.signals` does.

- [ ] **Step 2: Fit open-ended thresholds by offline replay**

Run:
```
python scripts/run_threshold_sweep.py --logs results/raw_logs/p5_qwen7b_calib_open.jsonl:OPEN
```
Choose τ at the quantile that yields the Task 0 target budget. Starting estimates from
the existing 35 records: entropy **0.540**, margin **0.769**, probe F1 **0.54**.

- [ ] **Step 3: Write the fitted thresholds to a task-specific policy config**

New file `configs/policies/p5_gated_qwen_open.yaml`. Do **not** overwrite
`p5_gated_qwen.yaml` — the MCQ run's provenance depends on it.

- [ ] **Step 4: Sanity-check the budget before committing 200 questions**

Replay the fitted τ against the 35 existing records and confirm the predicted retrieval
rate is within a few points of target. If it still predicts ~100%, stop: the thresholds
are not the whole story and Task 0 branch (c) applies.

- [ ] **Step 5: Run the full 200-question open-ended evaluation**

Output `results/raw_logs/p5_qwen7b_open_v2.jsonl`. Keep the original 35-record file — it
is the evidence for the saturation finding and must not be overwritten.

- [ ] **Step 6: Verify the run completed and the budget landed**

Run the analysis from Task 4 and confirm n=200 and a retrieval rate near target.

---

## Task 4: Qwen tables

**Files:** Modify `scripts/generate_tables.py` · Output: `results/tables/*.tex`

`generate_tables.py` already scores MCQ by accuracy and open-ended by mean token-F1
(`generate_tables.py:133`), so no metric bug to fix — only Qwen rows to add.

- [ ] **Step 1: Add a Qwen block to `tab_main_results`** — the six MCQ rows and six
      open-ended rows from the Measured Facts tables above, with the Qwen open row
      carrying its `n` explicitly so a 35 is never mistaken for a 200.
- [ ] **Step 2: Add `tab_gate_saturation`** — the per-member τ / min / median / max /
      fire-rate table, both models, MCQ and open. This is the chapter's central exhibit.
- [ ] **Step 3: Add `tab_budget_match`** — realised retrieval rate per model per task
      against the intended budget, with the per-member breakdown showing the probe
      collapse (Llama 40% → Qwen 7.5% on MCQ).
- [ ] **Step 4: Extend `numbers()`** with Qwen macros (`\accQwenMcq`, `\accQwenOpen`,
      `\retrQwenMcq`, …) so the chapters never hard-code a number.
- [ ] **Step 5: Run `python scripts/generate_tables.py` and confirm the four .tex files
      regenerate without error.**
- [ ] **Step 6: Commit.**

---

## Task 5: Qwen figures

**Files:** Modify `scripts/generate_figures.py` · Output: `results/figures/*.{png,pdf}`

- [ ] **Step 1: `fig_scaling`** — accuracy vs retrieval budget with both models on one
      axis: P3 at 0%, P5 at its realised rate, P1 at 100%. Makes "Qwen is better
      everywhere and loses less to retrieval" visible in one panel.
- [ ] **Step 2: `fig_gate_saturation`** — per-member signal distributions (violin or box)
      with the τ line overlaid, four panels: {Llama, Qwen} × {MCQ, open}. The Qwen-open
      panel shows τ sitting entirely outside the data, which is the whole argument.
- [ ] **Step 3: `fig_signal_shift`** — entropy and margin distributions, MCQ vs
      open-ended, same model. Shows *why* an MCQ-fitted τ cannot transfer to open-ended.
- [ ] **Step 4: Extend `fig_pareto` and `fig_retrieval` with Qwen series** (existing
      figures currently plot the 3B only).
- [ ] **Step 5: Run `python scripts/generate_figures.py`; confirm every figure writes both
      .png and .pdf.**
- [ ] **Step 6: Commit.**

---

## Task 6: Chapter — Qwen-7B findings

**Files:** Create `report/chapters/qwen_scaling.tex` · Modify `report/main.tex`

- [ ] **Step 1: Write the chapter** with these sections:
      1. Purpose and setup — identical corpus, identical questions, identical policies;
         only the model changes, so any difference is attributable to scale.
      2. Closed-book capability — 0.745 vs 0.620, McNemar p<0.001.
      3. Calibration non-transfer — the 3B's τ made Qwen barely retrieve (2% on the
         calibration run); prospective confirmation, not a post-hoc excuse.
      4. Re-calibration and the realised MCQ budget — 31.5% against a 50% target, with
         the probe collapse as the cause.
      5. Gate saturation on open-ended — τ outside the data range entirely.
      6. Open-ended results, with the n caveat carried in every sentence that cites them.
- [ ] **Step 2: Add `\input{chapters/qwen_scaling}` to `report/main.tex`** after
      `chapters/evaluation` and before `chapters/discussion`.
- [ ] **Step 3: Confirm every number comes from a `numbers.tex` macro, not a literal.**
- [ ] **Step 4: Compile and check for undefined references.**
- [ ] **Step 5: Commit.**

---

## Task 7: Chapter — Qwen vs Llama comparison

**Files:** Create `report/chapters/model_comparison.tex` · Modify `report/main.tex`

- [ ] **Step 1: Write the chapter** with these sections:
      1. What is and is not comparable — same corpus and questions, but **different
         realised retrieval budgets**, which is the honest caveat that must lead.
      2. MCQ: paired analysis, the 39/14 split, McNemar.
      3. Open-ended: common-35 comparison, gap +0.019 against ±0.023 CIs → not
         significant; state that plainly rather than implying a trend.
      4. Does scale change the *value of gating*? Llama loses 8.0pp from P3 to P5, Qwen
         2.5pp — but at different budgets, so this is suggestive, not established.
      5. Cost: the P5 latency finding, and what it implies for deployability.
- [ ] **Step 2: Add `\input{chapters/model_comparison}` to `report/main.tex`.**
- [ ] **Step 3: Cross-reference the saturation table and scaling figure.**
- [ ] **Step 4: Compile; check floats land and references resolve.**
- [ ] **Step 5: Commit.**

---

## Task 8: Limitations — budget mismatch and gate saturation

**Files:** Modify `report/chapters/discussion.tex`

- [ ] **Step 1: Write the budget-mismatch limitation.** Llama 49.5% vs Qwen 31.5% on MCQ.
      The calibration rule targeted the 3B's per-gate budget, and entropy (34.5%) and
      margin (40.5%) landed near target, but the probe is threshold-free on MCQ
      (`letter_match`) and cannot be recalibrated: it fell from 40% on Llama to 7.5% on
      Qwen because the larger model self-agrees more. Majority-2-of-3 then cannot reach
      the intended budget. Consequence: "selective retrieval beats always-retrieve at a
      matched budget" is demonstrated *within* each model but **not across** them, since
      the two budgets differ by 18 points.
- [ ] **Step 2: Write the gate-saturation limitation.** On open-ended, MCQ-fitted τ put
      both logit gates entirely outside their signal range for Qwen — every question
      retrieved. P5 open therefore degenerates to P1 open, and no open-ended P5-vs-P1
      claim can be made from that run. Llama saturates less only because its τ happens to
      fall inside its open-ended distribution. State the general lesson: **gate thresholds
      are task-specific, not just model-specific**, and calibrating on MCQ and deploying
      on open-ended is unsound.
- [ ] **Step 3: Note the open-ended metric's insensitivity.** Every run scores F1 0.19-0.26
      while refusal behaviour ranges from 0.5% to 90%. Refusing costs almost nothing
      (Llama P1 open: 0.2173 refusing vs 0.2435 answering) because token-F1 against a long
      `long_answer` gold rewards any fluent medical prose. Open-ended conclusions are
      therefore weakly supported in **either** direction — this is a measurement
      limitation, not a null result.
- [ ] **Step 4: Note the corpus mismatch.** PubMedQA asks about specific studies;
      StatPearls + Textbooks does not contain them, which is why retrieval *raises*
      refusal from ~1% to 90%.
- [ ] **Step 5: Cross-reference `docs/limitations-entropy-offset.tex`** (the entropy
      off-by-one), which is already drafted and belongs in the same section.
- [ ] **Step 6: Compile and commit.**

---

## Suggested order

1. **Tasks 4, 5, 8** — depend only on existing data, deliver guaranteed value. Do first.
2. **Tasks 6, 7** — chapters, written against the n=35 caveat.
3. **Task 1** — profiling; cheap, and decides whether the re-run is possible at all.
4. **Tasks 2, 3** — calibration and re-run, only if Task 1 finds a fix.
5. If Task 3 lands, update the affected numbers in Tasks 4-7. Every number lives in a
   `numbers.tex` macro precisely so this is a regeneration rather than a rewrite.

**Hard cutoff: 3 Aug.** If the re-run has not completed by then, take Task 0 branch (c),
keep the n=35 row with its caveat, and spend the remaining days on the write-up.
