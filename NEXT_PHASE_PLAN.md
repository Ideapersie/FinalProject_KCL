# Next Phase Plan — Calibrate Gates → Add Retrieval Types → Full Open-Ended Eval

Three sequential phases, each gated on the previous working. Builds on the P5
prototype (branch `feat/p5-gated-prototype`). Test-first throughout; nothing
committed without explicit ask.

**Why this order:** calibration makes the gate actually fire (otherwise the new
retrievers are never exercised by P5); retrieval types must work before scaling;
open-ended eval needs both a working pipeline and an open-ended dataset.

---

## Phase 1 — Calibrate gate thresholds (make P5 fire correctly)

**Goal:** move entropy/margin off their floor so the ensemble genuinely votes,
then re-run P5 and show retrieval rate > 0% with accuracy moving toward the P1
ceiling.

### 1.0 Prerequisite — run P1 (the missing ceiling)
- Run P1 (always-retrieve, BM25) on the same 200 MMLU Qs → `p1_mirage200.jsonl`.
- Without P1 there is no ceiling to calibrate *toward*. ~25 min, 1 call/query.

### 1.1 Build the calibration sweep script
- `scripts/run_threshold_sweep.py`: for each threshold in a grid, evaluate the
  gate decision **offline** against already-logged signals where possible, or
  re-run P5 per threshold on a subset.
- Efficient approach: the existing `p5_mirage200.jsonl` already stores every
  gate signal in `qvault.gate_details.signals`. We can **replay** entropy/margin
  decisions at any threshold *without re-calling the LLM* — only the final
  retrieve/skip branch changes the answer, so re-run only the queries whose
  decision flips. This turns a many-hour sweep into minutes for the gate-vote
  analysis, with a smaller confirmation run for accuracy.

### 1.2 Fix the sweep grid to the observed ranges
- Current `configs/experiments/threshold_sweep.yaml` grid is [1.0–5.0] nats —
  above this model's max entropy (1.59). Replace with model-calibrated grids:
  - entropy τ_H grid: [0.5, 0.7, 0.9, 1.1, 1.3] (observed 0.27–1.59)
  - margin  τ_M grid: [0.50, 0.60, 0.65, 0.70, 0.75] (observed 0.45–0.89)
- Add a margin sweep config (only entropy exists today).

### 1.3 Pick operating points
- For each gate, choose τ at a sensible percentile of the observed signal
  distribution (starting points: entropy p75 ≈ 0.9, margin p25 ≈ 0.64), then
  validate against accuracy + retrieval-rate trade-off.
- Decision rule to report: τ that maximises accuracy-per-retrieval, or that hits
  a target retrieval rate (40–70% per the project goal).

### 1.4 Re-run P5 calibrated + ablation
- Re-run P5 (calibrated τ) on 200 MMLU Qs → compare to P1/P3.
- Ablation: each gate solo vs the ensemble (does the probe catch what entropy/
  margin miss?). Log per-gate agreement.

**Deliverables:** P1 baseline, calibration curves, calibrated P5 result,
gate ablation table. Update `week2_gating.tex` (replace the zero-retrieval
table with calibrated numbers + the calibration story as a finding).

**Exit criterion:** P5 retrieval rate in a sensible band (not 0%, not 100%),
accuracy ≥ P3 and approaching P1.

---

## Phase 2 — Implement the other retrieval types

**Goal:** semantic + hybrid retrieval working, so P1/P4/P5 can use more than
lexical BM25. Only proceed once Phase 1 shows the gate working.

### 2.1 Vector retriever (semantic search)
- `src/medrag_adaptive/retrieval/vector_retriever.py`:
  - embed chunks with `sentence-transformers/all-MiniLM-L6-v2` (already in config)
  - FAISS `IndexFlatIP` (cosine after L2-normalise) — exact, deterministic, CPU
  - `from_chunks()` (build) + `from_index()` (load) mirroring BM25Retriever
  - real-time: embed the query, search FAISS, return top-k Chunks
- Extend `scripts/build_indexes.py` to build the FAISS index alongside BM25.
- **Tests:** semantic match a paraphrase BM25 would miss ("heart attack" →
  "myocardial infarction" chunk); save/load roundtrip.

### 2.2 Hybrid retriever (RRF fusion)
- `src/medrag_adaptive/retrieval/hybrid_retriever.py`:
  - run BM25 + FAISS, fuse rankings via Reciprocal Rank Fusion
    `score(d) = Σ 1/(k + rank_i(d))`, k=60 (config `rrf_k`)
  - return top-k by fused score
- **Tests:** a doc ranked mid by both retrievers rises via fusion; RRF formula.

### 2.3 Wire retrieval_mode through the factory
- `factory.build_policy` reads `cfg.policy.retrieval_mode`
  (bm25 | vector | hybrid) and constructs the right retriever — today it's
  handed a BM25 retriever regardless.
- P1/P5 then honour `retrieval_mode: hybrid` from their YAMLs.
- **Tests:** factory builds the retriever named by the config.

### 2.4 Implement P2 and P4 (need the new retrievers)
- `p2_always_retrieve_cite.py`: P1 + citation extraction (parse SOURCES line,
  match to retrieved chunks → citation precision/recall).
- `p4_hybrid.py`: routes/uses the hybrid retriever explicitly.

**Deliverables:** vector + hybrid retrievers, P2/P4, all retrieval modes
selectable. New `week3_retrieval.tex` section (BM25 vs vector vs hybrid, with
a retrieval-quality comparison on the pilot corpus).

**Exit criterion:** P1 runs identically under bm25/vector/hybrid modes (same
harness), and hybrid measurably changes which chunks are retrieved vs BM25.

---

## Phase 3 — Full dataset incl. open-ended (paragraph) Q&A

**Goal:** move beyond 200 MMLU MCQs to (a) more MIRAGE subsets and (b) genuine
open-ended paragraph Q&A scored by token-F1, not letter match.

### 3.0 IMPORTANT scope finding — MIRAGE is all MCQ
- All 5 MIRAGE subsets (mmlu, medqa, medmcqa, pubmedqa, bioasq) ship with
  `options` → they are multiple-choice. **MIRAGE alone gives no paragraph Q&A.**
- True open-ended (free-text answer) requires a different dataset.

### 3.1 Get an open-ended dataset
- `scripts/download_datasets.py` already targets **RAGCare-QA** (HuggingFace,
  open-ended) — but it is **not yet downloaded**. Run the downloader.
- `load_ragcare` loader already exists (choices=None, gold string, token-F1).
- Alternative/supplement if RAGCare is unsuitable: an open-ended split of
  MedQA, or the synthetic acute-care set (`synthetic_loader.py`, ⬜ to build).
- **Supervisor question:** is token-F1 an acceptable open-ended accuracy proxy,
  or do they want LLM-as-judge / clinical review? (Affects scoring design.)

### 3.2 Verify the open-ended path end-to-end
- The pipeline already branches on `is_multiple_choice()`:
  scoring uses `token_f1`; the hallucination probe uses its `f1_threshold` mode.
  Confirm both work on real open-ended data (currently only unit-tested on mocks).
- Open-ended changes gate behaviour: longer drafts → different entropy/margin
  distributions → **may need separate calibration** (re-run a mini-sweep on
  open-ended). Likely a finding in itself.

### 3.3 Scale up the question set
- Run P1/P3/P5(calibrated) across multiple MIRAGE subsets (stratified) +
  the open-ended set, all 3 retrieval modes where relevant.
- Stratify by risk level for the safety-envelope analysis (current 200 are
  skewed: only 6 high-risk).
- Use the full real corpus here if Phase 2 corpus work is done; else flag
  pilot-corpus caveat.

**Deliverables:** open-ended results, MCQ-vs-open-ended comparison, per-subset
breakdown, risk-stratified results feeding the safety envelopes.

**Exit criterion:** P1/P3/P5 results on both MCQ and open-ended, scored
correctly per type, with calibrated gates.

---

## Cross-cutting / deferred (not in these three phases)
- `evaluation/harness.py` formal class, `metrics.py` (ECE, citation P/R),
  `safety_envelope.py`, `generate_figures.py` — pull in as Phase 3 needs them.
- `attention_entropy.py`, Gradio UI (P6), P7 multi-agent, Vercel demo — stretch.

## Open questions for the supervisor (raise before Phase 3)
1. Token-F1 vs LLM-as-judge for open-ended scoring?
2. Full MedCorp (~427K chunks) on CPU, or a justified subset? (gates Phase 2 corpus)
3. One global threshold per model, or per-dataset / per-risk calibration?
4. Which stretch goals (P7, Vercel, attention layer) are graded vs cuttable?

## Findings logged during execution

- **P1 (always-retrieve, pilot corpus) = 17.0% accuracy** — far BELOW P3's 62%.
  Cause (diagnosed, not a bug): the 200-chunk pilot corpus rarely contains the
  answer to an MMLU question, and the RAG prompt instructs the model to use only
  the context and state uncertainty otherwise. So on 128/200 questions the model
  correctly refused ("the context does not provide information... I am
  uncertain"), producing no parseable letter → scored wrong. This is the honest
  consequence of retrieving irrelevant context, and it strongly motivates Phase
  2/3 (semantic retrieval + the full MedCorp corpus). It also flags that
  always-retrieve can *hurt* when the corpus is weak — itself a result worth
  reporting (retrieval is not free; bad retrieval is worse than none).
- **Implication for calibration (Phase 1):** until the real corpus is in place,
  the P1 ceiling is artificially depressed; calibrate P5 against P3 (the
  meaningful reference here) and re-establish the P1 ceiling on the full corpus.
- **Scoring robustness:** verbose RAG answers stress the MCQ letter extractor;
  on the real corpus, confirm extraction handles "The correct answer is X."
  phrasings (currently it does, but abstentions legitimately return None).

## Risks
- Replay-based sweep assumes stored signals are complete — verify qvault has
  entropy + margin signals for every record before relying on offline replay.
- Open-ended entropy/margin ranges differ from MCQ → may need its own
  calibration; budget for a second mini-sweep.
- FAISS + sentence-transformers RAM on full corpus may exceed the medium tier;
  measure on the pilot first, extrapolate.
```
