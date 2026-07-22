# Qwen-7B scaling run — step-by-step

**Goal:** turn "selective beats always-retrieve" and "calibration doesn't transfer"
from single-model observations into two-model claims, and find out where the
accuracy curve sits relative to the clinical safety bars (low 70 / med 80 / high 90).

**Scope tonight:** Qwen2.5-7B-Instruct Q4_K_M, policies **P3 + P5**, **MCQ + open**.
Four 200-question runs plus one 50-question calibration run.

**Timeline:** setup ~45 min, runs ~4–5 h. Fits one Colab session if it does not
disconnect.

---

## Phase 0 — DO THIS FIRST (do not wait, it is the critical path)

**0.1 Start the corpus upload now.** It is 412 MB and nothing else can proceed
without it.

Upload `data/corpora/medcorp_tp.jsonl` to Google Drive at exactly:

```
MyDrive/medrag/data/corpora/medcorp_tp.jsonl
```

Use a **personal Google account**, not the KCL one — Workspace for Education
accounts frequently block Drive mounting inside Colab, and you do not want to
discover that at midnight. Everything else in this plan assumes that account.

Upload the corpus only. The prebuilt indexes are 1.8 GB and are cheaper to rebuild
on the T4 (~25 min) than to transfer.

**0.2 While it uploads — push the repo.** This is the blocker that would otherwise
waste the whole evening: **Colab clones from GitHub**, and every Qwen prerequisite
is currently either uncommitted or unpushed. You have **42 unpushed commits**, and
`configs/models/`, the notebook, and the smoke test are untracked. A clone right now
gets a repo with no Qwen support at all.

`data/raw/` is gitignored, so the calibration set needs `-f`:

```bash
git add configs/models/ notebooks/ \
        scripts/smoke_test_model.py scripts/make_calibration_set.py \
        scripts/analyse_chunk_relevance.py scripts/run_gate_variants.py \
        src/medrag_adaptive/models/prompts.py \
        report/main.tex report/references.bib QWEN_RUN_PLAN.md
git add -f data/raw/mirage/calib50.json data/raw/openqa/pubmedqa_labeled.jsonl
git commit
git push origin feat/p5-gated-prototype
```

`pubmedqa_labeled.jsonl` (0.1 MB) is the open-ended dataset — without it the two
open runs die immediately on Colab.

**0.3 Verify the push.** Open
`https://github.com/Ideapersie/FinalProject_KCL/tree/feat/p5-gated-prototype` and
confirm `configs/models/qwen7b.yaml` and `data/raw/mirage/calib50.json` are visible.
If they are not there, Colab will not get them.

**0.4** Colab clones the **default branch**. If that is `main`, either merge this
branch to main or edit the clone cell to
`git clone -b feat/p5-gated-prototype ...`.

---

## Phase 1 — Colab session (~45 min)

Open `notebooks/colab_scaling.ipynb` in Colab. Run cells in order.

| Step | Cell | What it does | Expected |
|---|---|---|---|
| 1 | GPU check | `nvidia-smi` | **T4, ~15 GB free.** No GPU → Runtime → Change runtime type → T4 |
| 2 | Mount Drive | mounts + asserts corpus present | corpus ~412 MB |
| 3 | Clone + install | CUDA build of llama-cpp | `llama-cpp-python 0.3.28`, GPU offload `True` |
| 4 | Model download | resolves the real GGUF filename, then downloads | ~4.5 GB, cached to Drive |
| 5 | Index rebuild | embeds 426K chunks on GPU, builds FAISS + BM25 | ~25 min, cached to Drive |

**Watch for:**

- **GPU offload `False`** → the CUDA build fell back to CPU. Re-run the install cell.
  Do not continue; CPU would take days.
- **Assemble stage OOM** (step 5). Colab free has ~12.7 GB RAM and the assemble stage
  holds embeddings + chunks + BM25 at once. It fits on your 8 GB laptop so it should
  fit here, but if it dies: Runtime → Change runtime type → High-RAM (Pro only), or
  upload the prebuilt `indexes/` from your machine instead and let the restore path
  pick them up.
- The install pins **llama-cpp-python 0.3.28** deliberately. The entropy gate reads
  `_scores`, a semi-private attribute, and the margin gate needs `logprobs=2`. A newer
  release that moves either one does not raise — the gates return `None` and P5 quietly
  degrades to retrieve-everything. Do not "upgrade to fix" anything.

---

## Phase 2 — Smoke test (~5 min). **The gate. Nothing long runs until this passes.**

Four checks, each guarding a failure that does not raise an error:

1. **Chat format** — Qwen uses ChatML `<|im_start|>`, not Llama `[INST]`. Wrong markup
   produces fluent but worse text; you would be measuring the prompt, not the model.
2. **Answers parse** — `extract_letter` finds a letter in ≥80% of outputs. If not,
   accuracy reads ~0% and looks like a bad model rather than a bad parser.
3. **Gates fire** — entropy and margin return real numbers, i.e. `logits_all` survived
   the CUDA build.
4. **Signals span τ** — the observed range brackets the threshold.

**Stop conditions:**

- Any `[FAIL]` → do not start the runs. Fix it or stop for the night.
- `[WARN] tau outside range` → **expected and fine.** That is the calibration
  non-transfer finding; step 6b fixes it automatically.

---

## Phase 3 — Calibrate BEFORE the long runs (~15 min)

This is the part that changed, and it matters.

τ = 0.70/0.70 was fitted to the **3B**. If it does not transfer to Qwen and you only
find out in the morning, P5 will have spent the night collapsed onto P1 (retrieve
everything) or P3 (retrieve nothing) — four hours measuring nothing.

So a 50-question P5 run harvests Qwen's signal distribution, and the thresholds are
refit offline in seconds.

**How the refit works.** Not by copying the 3B's τ, and not by the `p75/p25` hint the
sweep script prints — that hint was the *plan's starting point*, not what the 3B
shipped. Measured on your real 3B log:

| | τ_H/τ_M | entropy | margin | probe | ensemble |
|---|---|---|---|---|---|
| shipped 3B | 0.70/0.70 | 52% | 54% | 40% | **50%** |
| p75/p25 hint | 0.919/0.635 | 25% | 25% | 40% | **24%** |

Using p75/p25 would give Qwen **half** the 3B's retrieval budget, confounding model
scale with how often each model is allowed to retrieve — and "selective beats
always-retrieve" is a claim about accuracy *at a given budget*, so it would no longer
be comparable.

Instead each gate's τ is set so it **fires at the same rate it did on the 3B**
(entropy 52%, margin 54%). Budget held constant, scale the only variable. Non-transfer
then becomes a sharper claim: not "τ stopped working" but "τ had to move *this far* to
buy the same budget."

Calibration uses `data/raw/mirage/calib50.json`, a **stride sample** (every 4th of the
200). Your evaluation set is 200 MMLU questions ordered by subject, so a prefix is one
or two subjects, not a sample. Verified on the 3B's own signals: a prefix overshoots
the budget by ~6pp and the error does **not** shrink with N; the stride sample lands
within ~1pp.

**Read the per-gate table it prints.** Specifically the probe row. The probe has no
threshold on MCQ (`letter_match`: two drafts disagree → retrieve), so it cannot be
recalibrated. If Qwen self-agrees more than the 3B did, the probe goes quiet and
majority-2-of-3 silently reduces to "entropy AND margin" — two gates that already
agree 82% of the time. The cell warns if the probe drops below 5% or exceeds 95%.
**That is a finding about gate ensembles at scale, not a bug** — write it up rather
than fixing it.

The cell writes `configs/policies/p5_gated_qwen.yaml` and copies it to Drive (the
Colab checkout is wiped when the VM recycles).

---

## Phase 4 — The runs (~4–5 h)

Order is deliberate — cheap runs first, so a disconnect still leaves something usable:

| # | Run | Est. | Why it matters |
|---|---|---|---|
| 1 | P3 closed-book MCQ | ~30 min | **The new ceiling.** Does 7B clear the 70% low-risk bar? |
| 2 | P3 closed-book open | ~45 min | Open-ended baseline |
| 3 | P5 gated MCQ | ~70 min | Does *selective beats always* survive the scale change? |
| 4 | P5 gated open | ~90 min | Where retrieval genuinely helps |

Logs write **straight to Drive**, and every run skips question-ids already present.
After any disconnect: re-run the notebook top to bottom. Completed runs are skipped,
the partial one resumes. A disconnect costs minutes, not the run.

**Overnight, free tier:** the session dies if the tab closes or the machine sleeps.
Leave the tab open, disable sleep, stay on AC. Expect to re-run the cell in the
morning to finish whatever was in flight — that is normal, not a failure.

If you have Colab Pro, enable background execution.

---

## Phase 5 — Morning (~20 min)

1. Download `MyDrive/medrag/logs/*.jsonl` → `results/raw_logs/`.
   Also grab `p5_gated_qwen.yaml` → `configs/policies/`.
2. Sanity check every run is complete:
   ```bash
   wc -l results/raw_logs/*qwen7b*.jsonl     # expect 200 each
   ```
3. Post-hoc calibration table (notebook step 8, or locally):
   ```bash
   python scripts/run_threshold_sweep.py \
     --logs results/raw_logs/p5_qwen7b_mcq.jsonl:QWEN_MCQ \
            results/raw_logs/p5_qwen7b_open.jsonl:QWEN_OPEN
   ```
   Confirms the 50-question subset was representative.
4. Regenerate numbers and figures:
   ```bash
   python scripts/generate_tables.py
   python scripts/generate_figures.py
   python -m pytest tests/
   ```
   **`generate_tables.py` and `generate_figures.py` currently only know about the 3B
   logs.** They will need a Qwen row / scaling figure added — that is Friday's first
   job, before writing §5.10.

---

## What each outcome means for the write-up

Every result is publishable whichever way it lands — say so in the chapter.

| Question | If yes | If no |
|---|---|---|
| Does P5 > P1/P4 hold at 7B? | Central claim generalises beyond one model | Gating is model-scale-dependent — a sharper, more interesting finding |
| Does τ=0.70 fail on Qwen? | Calibration non-transfer confirmed **prospectively** — the strongest form | Thresholds are more portable than claimed; report honestly |
| Does 7B clear the 70% low-risk bar? | Quantifies the scale at which low-risk deployment becomes viable | Model-capability-ceiling finding is **strengthened**, not weakened |
| Does the probe stay live? | 3-gate ensemble is justified at scale | Ensembles degrade as models improve — a genuinely novel observation |

---

## Abort rules

- **Smoke test fails** → stop. Do not burn the night on a broken pipeline.
- **Index rebuild OOMs twice** → stop, upload prebuilt indexes tomorrow instead.
- **Not started by ~01:00** → stop. A finished single-model thesis beats a half-run
  two-model one. Delete Ch5 §5.10 and caveat in Discussion.
- **Hard cutoff: 27 July.** If Qwen is not done and analysed by then, cut it entirely
  and spend the remaining days on the write-up. The write-up is where the marks are.

**Do not run 14B tonight.** Only consider it if 7B lands clean with days to spare;
7B alone already gives the scaling point.
