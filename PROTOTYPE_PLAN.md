# Prototype Plan — First Working P5 Gated Pipeline

**Goal:** P5 (Selective/Gated Retrieval) runs end-to-end on real MIRAGE questions
with the local Llama-3.2-3B model, producing a JSONL log with gate decisions,
retrieval rate, accuracy, and latency — directly comparable to the existing
P3=62% baseline.

**Scope decisions (locked):**
- BM25-only retrieval (no FAISS/hybrid yet — deferred to a follow-up phase).
- Full 3-gate ensemble: entropy + margin + hallucination-probe, majority vote (≥2).
- Test-first: every component gets unit tests against `MockLLMBackend` before
  any real-model run.
- `report/` LaTeX explanation updated after the prototype runs (per the
  CLAUDE_CODE_CONTEXT.md standing instruction to explain each change).

---

## What already exists (no work needed)
- `gating/base.py` — `Gate` ABC + `GateDecision` (done)
- `scoring.extract_letter`, `scoring.token_f1` — reused by gates (done)
- `prompts.build_probe_prompt_a/b`, `build_draft_prompt` (done)
- `MockLLMBackend` — peaked/uniform logits + `get_top2_logprobs` (done)
- `LLMBackend.draft()` / `get_top2_logprobs()` on the ABC (done)
- `run_experiment.py` driver with profiling + resume (done; needs factory wiring)
- Llama-3.2-3B GGUF on disk; MIRAGE benchmark.json on disk

---

## Phase A — The three gates (core novelty)

### A1. `gating/entropy_gate.py`
- `EntropyGate(threshold, draft_max_tokens)` implementing `decide()`.
- Call `llm.draft(prompt)` → `(text, logits)`; softmax per row; compute
  per-token entropy `H_t = -Σ p·log p`; mean over tokens.
- `retrieve = mean_H > threshold`.
- `GateDecision.details["per_token_entropy"] = [...]` — this array IS the
  Token-Level Entropy Attribution novelty; store it for later visualisation.
- If `logits is None` (logits_all=False / low tier): return abstain signal so
  the ensemble can drop this member. (`details["available"]=False`.)
- **Tests:** `mock_llm_high` → SKIP (low entropy); `mock_llm_low` → RETRIEVE;
  per-token array length == draft tokens; None-logits path abstains.

### A2. `gating/margin_gate.py`
- `MarginGate(threshold, draft_max_tokens)`.
- Call `llm.get_top2_logprobs(prompt)` → per-token top-2; convert logprobs to
  probs; margin_t = p_top1 - p_top2; mean.
- `retrieve = mean_margin < threshold` (small margin = indecisive = retrieve).
- `details["mean_margin"]`, `signal_value = mean_margin`.
- **Tests:** high-confidence mock (big gap) → SKIP; low-confidence (tiny gap)
  → RETRIEVE.

### A3. `gating/hallucination_probe_gate.py`
- `HallucinationProbeGate(agreement_mode, f1_threshold, max_tokens)`.
- Build prompt A/B via `build_probe_prompt_a/b`; two `llm.draft()` calls.
- MCQ (`letter_match`): `extract_letter` on each; `retrieve = letters differ
  or either None`.
- Open-ended (`f1_threshold`): `token_f1(draft_a, draft_b)`;
  `retrieve = f1 < threshold`.
- `details["draft_a"]`, `["draft_b"]`, `["agreement"]`.
- Works with logits_all=False (text-only) — the only low-tier gate.
- **Tests:** mock returning same letter both prompts → SKIP; a mock variant
  returning different letters → RETRIEVE; open-ended F1 path both branches.

### A4. `gating/ensemble_gate.py` (or fold into factory)
- `EnsembleGate(members: list[Gate], min_votes)`.
- Run each member's `decide()`; collect votes from members reporting
  `available`; `retrieve = (#RETRIEVE votes >= min_votes)`.
- `details["votes"] = {name: decision_str}`, `["members_available"]`.
- Edge case: if fewer available members than min_votes (low tier with only
  probe), fall back to "retrieve if any available member says retrieve" and
  record the degraded mode in details.
- **Tests:** 2-of-3 retrieve → RETRIEVE; 1-of-3 → SKIP; degraded single-member.

---

## Phase B — Gated policy + factory

### B1. `policies/p5_gated.py`
- `GatedPolicy(llm, retriever, gate, cite_sources)` implementing `answer()`.
- Flow: `decision = gate.decide(question, llm)`.
  - If `decision.retrieve`: `chunks = retriever.retrieve(...)`; build RAG
    prompt; `answer()`. `retrieval_triggered=True`.
  - Else: closed-book prompt; `answer()`. `retrieval_triggered=False`,
    `chunks=[]`.
- Populate `PolicyResult.gate_name/gate_decision/gate_signal_value`; pass
  `decision.details` through so the runner writes it to `RunRecord.qvault`.
- **Tests:** high-confidence mock → no retrieval, no chunks; low-confidence
  mock → retrieval + chunks present; gate fields populated on result.

### B2. `policies/factory.py`
- `build_policy(cfg, llm, retriever) -> Policy`.
- Map `cfg.policy.name`: p1→AlwaysRetrieve, p3→ClosedBook, p5→GatedPolicy.
- For p5: read `cfg.gate.type`. If `ensemble`, construct members from
  `cfg.gate.ensemble_members` with thresholds from cfg; wrap in EnsembleGate
  with `cfg.gate.ensemble_min_votes`. If single type, construct that one gate.
- Centralises construction so `run_experiment.py` stops hand-wiring.
- **Tests:** factory returns correct policy class per config; p5 ensemble has
  3 members; low-tier config (`type=hallucination_probe`) yields probe-only.

### B3. Wire factory into `run_experiment.py`
- Replace `build_policy()` body with a call to `factory.build_policy`.
- Keep the CLI and resume logic unchanged.
- The runner already copies `gate_signal_value` to qvault; extend to copy the
  full `decision.details` dict (per-token entropy, votes, drafts).
- **Tests:** existing `test_runner_resume.py` still green; add a p5 smoke test
  with mock llm end-to-end producing a RunRecord with gate fields.

---

## Phase C — First real prototype run + verification

### C1. Build a small BM25 index for the prototype
- The gated policy needs a real retriever for the RETRIEVE branch. Two options:
  - **C1a (fast):** build a BM25 index from the MIRAGE question contexts /
    a small StatPearls sample via a minimal `scripts/build_indexes.py` (BM25
    leg only). Enough to exercise the retrieve path.
  - **C1b (minimal):** if no corpus is ready, stand up a tiny fixture corpus so
    the prototype runs; flag the real-corpus index as a Phase D task.
- Decision needed from you (see Open Questions).

### C2. Run P5 on 50–100 MIRAGE questions
- `python scripts/run_experiment.py --policy configs/policies/p5_gated_ensemble.yaml
   --dataset data/raw/mirage/benchmark.json --bm25-index <idx>
   --max-questions 100 --output results/raw_logs/p5_mirage100.jsonl`
- Note: a config `configs/policies/p5_gated_ensemble.yaml` may need creating
  (CLAUDE_CODE_CONTEXT §13 calls for renaming the entropy variant to ensemble).

### C3. Compute prototype metrics
- Accuracy, retrieval rate (fraction RETRIEVE), mean latency, per-gate vote
  agreement. Compare against P3=62% floor and (once run) P1 ceiling.
- Quick script or inline analysis; write numbers into the report.

### C4. Update LaTeX + progress log
- Add a `report/week2_gating.tex` section documenting the three gates, the
  ensemble vote, the token-level entropy attribution, and the prototype
  result table.
- Update `CLAUDE_CODE_CONTEXT.md` §21 progress log and the memory file.

---

## Deferred (explicitly NOT in this prototype)
- FAISS `vector_retriever.py` + `hybrid_retriever.py` + RRF — Phase D.
- P2 (citation) and P4 (hybrid routing) policies — Phase D.
- `attention_entropy.py` interpretability layer — Week 3.
- `evaluation/harness.py` formal class — the existing `run_experiment.py`
  driver is sufficient for the prototype; a refactor into a `BenchmarkHarness`
  class can come when multi-policy/multi-tier orchestration is needed.
- Gradio UI (P6), safety-envelope analysis, figures.

---

## Risks
- **Margin gate logprobs format:** real llama-cpp `logprobs` dict shape may
  differ from the mock. Mitigation: validate against one real `draft()` call
  early in Phase A2; adjust parser.
- **Probe latency:** 2× draft calls per query. Mitigation: keep
  `draft_max_tokens` small (48); report overhead explicitly.
- **No real corpus indexed yet:** the RETRIEVE branch needs chunks. Resolved by
  the C1 decision below.
- **Per-token entropy array size in JSONL:** storing 48 floats × many queries
  inflates logs. Acceptable at prototype scale; revisit for full runs.

---

## Resolved Decisions
1. **Corpus (C1):** No real chunks downloaded. Build a small curated pilot
   corpus (~200 medical chunks, StatPearls/textbook-style, like
   `mock_corpus.jsonl`) committed to repo at e.g.
   `data/corpora/pilot_corpus.jsonl`. `build_indexes.py` (BM25 leg only)
   pickles it. Full MedCorp/StatPearls download + chunker deferred to Phase D.
   Report flags retrieval-quality numbers as pilot-only until real corpus lands.
2. **Question count (C2):** 200 MIRAGE questions — same set as the existing
   P3=62% run, so P3 vs P5 compare directly. Output
   `results/raw_logs/p5_mirage200.jsonl`.
3. **Config (C3):** Reuse `configs/policies/p5_gated_entropy.yaml`; set
   `gate.type: ensemble`. No new ensemble YAML file.
```
