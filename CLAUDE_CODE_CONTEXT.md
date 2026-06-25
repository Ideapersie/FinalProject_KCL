# qVault Framework — Claude Code Session Context (v2)
## Adaptive Retrieval Gating for Safety-Aware Medical RAG Under Resource Constraints

> **Student:** Tharit Mohamed Hussain Khan | **ID:** 22058691
> **Institution:** King's College London, Department of Informatics, MSc Artificial Intelligence
> **Module:** 7CCSMPRJ — Individual Project
> **Supervisor:** Dr Hanqi Yan
> **Framework:** qVault, Variant 5 — Retrieval Policy Evaluation under Constraints
> **Session started:** June 2026
> **Submission target:** ~July 21, 2026
> **Report word limit:** 15,000 words maximum

---

## How to Use This Document

This is the single source of truth for the project in Claude Code. It covers:
- What is already implemented in the repo (Section 3)
- What has changed since v1 of this document (Section 4)
- The revised 3-gate ensemble with Hallucination Probe replacing Verbalized Confidence
- The Attention Entropy interpretability layer (new, novel)
- Token-level entropy attribution (new, novel)
- The interactive demo vision (Vercel-hosted, API key input, optional PDF)
- The 6-week compressed plan updated for current repo state
- Full technical spec, file structure, and build order

**Start every Claude Code session by referencing this file.**

make sure to also be more explanatory in this project than usual, I would like proper explanations on why each component was implemented the way it is and how it connects to other parts, what role does it play, and how it plays in the bigger picture of the project. Do this after every single changes made. 

---

## 1. Project Summary

The project investigates **when retrieval is necessary — and when it can be safely skipped** — in a resource-constrained, offline medical question-answering (QA) system.

The core idea is **adaptive retrieval gating**: using lightweight, training-free confidence signals from a local LLM's own logit distributions to decide per-query whether to retrieve external medical evidence or answer from parametric knowledge.

**The research gap:** No prior work systematically evaluates adaptive gating on hospital-critical medical QA under CPU-only resource constraints. Self-RAG, TARG, and Adaptive-RAG addressed general-domain QA. MIRAGE benchmarked medical RAG for always-retrieve pipelines. This project sits at the intersection.

**Novel contributions (updated):**
1. **Safety Envelopes** — for each clinical risk level (Low/Medium/High), the minimum retrieval policy that maintains clinically acceptable accuracy
2. **Hardware-Aware Policy Escalation** — mapping policies onto real NHS and Thai hospital hardware profiles
3. **Hallucination Probe Gate** — a new gating signal based on draft consistency (replaces verbalized confidence)
4. **Token-Level Entropy Attribution** — per-token uncertainty maps showing *which parts* of a query the model is uncertain about, not just whether to retrieve
5. **Attention Entropy Interpretability Layer** — visualising where the model focuses attention during generation, producing interpretable heatmaps for clinical users

---

## 2. Rubric Weights (KCL 7CCSMPRJ)

| Section | Weight | What Distinction Requires |
|---------|--------|--------------------------|
| **Implementation/Technical Achievement** | **25%** | Systematic planned approach, novel solution, testing/verification, deep methodological knowledge |
| **Evaluation** | **20%** | Critically compare results expected from theory with results obtained in practice |
| **Specification & Design** | **15%** | Compare alternatives, explain rationale, distinctly novel attempt, shows research potential |
| **Introduction** | **10%** | Place project in wider economic/subject context, natural progression from broad to specific |
| **Literature Review** | **10%** | Penetrating analysis, new insights not immediately identifiable from literature |
| **General Scholarship** | **10%** | Professional presentation, grammatical style, correct bibliographic conventions, critical reflection |
| **LSEP Issues** | **10%** | All relevant professional standards, legislation, public well-being, sustainability, IP, trustworthiness |
| **Presentation** | Separate | Live demo + slides covering complete project scope, carefully selected scenarios |

**Implementation + Evaluation = 45% of marks.**

---

## 3. Current Repo State (What Is Already Built)

The GitHub repo `FinalProject_KCL/` has a solid scaffold. Here is exactly what exists and what doesn't.

### ✅ Fully Implemented

| File | What It Does | Status |
|------|-------------|--------|
| `configs/base.yaml` | Shared defaults: seed, paths, model, retrieval, gate, evaluation, profiling | Complete, well-commented |
| `configs/hardware_low.yaml` | 4GB tier overrides: 1B model, BM25-only, logits_all=false | Complete |
| `configs/hardware_medium.yaml` | 8GB tier overrides: 3B model, hybrid retrieval | Complete |
| `configs/hardware_high.yaml` | 16GB+ tier overrides: Phi-3.5-mini, expanded context | Complete |
| `configs/policies/*.yaml` | All 6 policy configs (P1–P6) including 3 P5 gate variants | Complete |
| `configs/experiments/*.yaml` | 4 experiment configs: threshold_sweep, full_mirage, ragcare, acute_care | Complete |
| `src/medrag_adaptive/config.py` | YAML deep-merge + Pydantic `ProjectConfig` with all nested models | Complete, tested |
| `src/medrag_adaptive/data/schema.py` | `UnifiedQuestion`, `RunRecord`, `PolicyResult`, `Chunk`, `Citation`, JSONL I/O | Complete |
| `src/medrag_adaptive/models/base.py` | `LLMBackend` ABC with `draft()`, `answer()`, `close()` | Complete |
| `src/medrag_adaptive/models/llama_backend.py` | Full llama-cpp-python wrapper: logit extraction via `_scores`, `get_top2_logprobs()`, CLI test | Complete |
| `src/medrag_adaptive/models/prompts.py` | All prompt templates: closed-book, RAG, RAG+citation, verbalized confidence, draft | Complete (needs hallucination probe prompt added) |
| `scripts/setup_environment.sh` | Full setup: venv, deps, model download, verification, unit tests | Complete |
| `pyproject.toml` | Package config with all deps, editable install, pytest config | Complete |
| `requirements.txt` | Pinned dependencies | Complete |
| `tests/conftest.py` | `MockLLMBackend` (peaked + uniform logits), `MockRetriever`, fixtures | Complete |
| `tests/fixtures/sample_questions.jsonl` | 5 test questions (low/medium/high risk) | Complete |
| `tests/fixtures/mock_corpus.jsonl` | 5 mock retrieval chunks | Complete |
| `tests/unit/test_config.py` | 12 tests covering deep-merge, config loading, all hardware/policy overrides | Complete |
| All `__init__.py` files | Package structure for all modules | Complete |

### 🔲 Not Yet Implemented (Empty Modules or Missing)

| Module | What It Needs | Priority |
|--------|--------------|----------|
| `src/medrag_adaptive/gating/base.py` | `Gate` ABC + `GateDecision` enum | Week 2 |
| `src/medrag_adaptive/gating/entropy_gate.py` | Token entropy gate + token-level attribution | Week 2 |
| `src/medrag_adaptive/gating/margin_gate.py` | Logit margin gate | Week 2 |
| `src/medrag_adaptive/gating/hallucination_probe_gate.py` | **NEW** — replaces verbalized confidence | Week 2 |
| `src/medrag_adaptive/gating/attention_entropy.py` | **NEW** — interpretability layer, not a gate | Week 2–3 |
| `src/medrag_adaptive/retrieval/bm25_retriever.py` | BM25 index + search over StatPearls chunks | Week 2 |
| `src/medrag_adaptive/retrieval/vector_retriever.py` | FAISS index + sentence-transformer search | Week 2 |
| `src/medrag_adaptive/retrieval/hybrid_retriever.py` | RRF fusion of BM25 + FAISS | Week 2 |
| `src/medrag_adaptive/retrieval/medrag_corpus.py` | MedRAG corpus loader/chunker | Week 1 |
| `src/medrag_adaptive/policies/base.py` | `Policy` ABC + `PolicyResult` integration | Week 1 |
| `src/medrag_adaptive/policies/p1_always_retrieve.py` | Always-retrieve policy | Week 1 |
| `src/medrag_adaptive/policies/p2_always_retrieve_cite.py` | P1 + citation extraction | Week 2 |
| `src/medrag_adaptive/policies/p3_closed_book.py` | Closed-book policy | Week 1 |
| `src/medrag_adaptive/policies/p4_hybrid.py` | Hybrid routing policy | Week 2 |
| `src/medrag_adaptive/policies/p5_gated.py` | Gated policy composing gate + retriever + LLM | Week 2 |
| `src/medrag_adaptive/policies/p6_user_toggle.py` | Gradio UI toggle | Week 3 |
| `src/medrag_adaptive/policies/factory.py` | `PolicyFactory.build(cfg) → Policy` | Week 2 |
| `src/medrag_adaptive/data/loaders/mirage_loader.py` | MIRAGE benchmark parser | Week 1 |
| `src/medrag_adaptive/data/loaders/ragcare_loader.py` | RAGCare-QA parser | Week 1 |
| `src/medrag_adaptive/data/loaders/synthetic_loader.py` | Synthetic acute-care loader | Week 3 |
| `src/medrag_adaptive/data/risk_tagger.py` | Keyword heuristic risk classifier | Week 1 |
| `src/medrag_adaptive/evaluation/metrics.py` | EM, F1, accuracy, ECE, citation P/R | Week 3 |
| `src/medrag_adaptive/evaluation/profiler.py` | Latency + energy + memory wrappers | Week 1 |
| `src/medrag_adaptive/evaluation/harness.py` | `BenchmarkHarness` — sequential, resumable | Week 3 |
| `src/medrag_adaptive/evaluation/safety_envelope.py` | Per-risk-level minimum policy analysis | Week 3 |
| `src/medrag_adaptive/evaluation/results.py` | `ExperimentSummary` aggregation | Week 4 |
| `src/medrag_adaptive/ui/gradio_app.py` | P6 Fast/Sourced toggle interface | Week 3 |
| `scripts/download_datasets.py` | MIRAGE, RAGCare-QA download script | Week 1 |
| `scripts/build_indexes.py` | BM25 pickle + FAISS index builder | Week 2 |
| `scripts/run_threshold_sweep.py` | Gate threshold calibration | Week 2 |
| `scripts/run_full_experiment.py` | Main experiment driver | Week 3 |
| `scripts/generate_figures.py` | Pareto plots, heatmaps, safety envelopes | Week 4 |

---

## 4. What Changed Since v1 (Critical Updates)

### 4.1 Hallucination Probe Gate Replaces Verbalized Confidence

**Why:** The verbalized confidence gate was the weakest signal. It required a full extra LLM call, wasn't grounded in the model's actual internal state, and self-reported confidence is notoriously miscalibrated in small models. It added nothing that the other gates didn't already capture better.

**What replaces it:** The **Hallucination Probe Gate** generates two short draft answers to the same question using slightly different prompt framings. If the drafts disagree (different answer letters or contradictory claims), the model is unstable on this query = retrieve. If they agree = skip.

```
Draft A: "Answer the following medical question briefly: {question}"
Draft B: "What is the correct answer to: {question}"

If draft_A_answer ≠ draft_B_answer → RETRIEVE (model is unstable)
If draft_A_answer == draft_B_answer → SKIP (model is consistent)
```

**Implementation:** Two `draft()` calls with different prompt templates, then a lightweight agreement detector (for MCQ: compare extracted answer letters; for open-ended: compute token-level F1 between the two drafts, retrieve if F1 < 0.7).

**Trade-off:** 2× LLM calls like verbalized, but directly tests whether the model *knows* the answer rather than asking it to self-assess. Document this latency overhead explicitly in the report.

**Basis:** Related to self-consistency approaches (Wang et al., 2023) but applied as a binary gate signal rather than majority voting over many samples.

### 4.2 Token-Level Entropy Attribution (New Novel Feature)

Instead of just computing mean entropy across all draft tokens (the existing entropy gate), also produce a **per-token entropy map** showing which tokens in the generated draft have high uncertainty.

For multi-part medical questions, this reveals *which part* the model is uncertain about. Example: "What is the dosage of metformin for a patient with stage 3 CKD?" — the model may be confident about "metformin" but uncertain about "stage 3 CKD dosing", and the entropy heatmap would show this.

**Implementation:** Already have the logits from `entropy_gate.py`. Compute `H(pₜ)` per token position `t` instead of just `H̄`. Return both the mean (for the gate decision) and the per-token array (for visualisation). The Gradio UI and the demo app can render this as a colour-coded heatmap over the draft text.

**This is novel:** No existing adaptive retrieval work produces token-level uncertainty attribution. TARG computes mean entropy only. Self-RAG uses binary reflection tokens. This granular view is especially valuable in healthcare where knowing *what* the model doesn't know is as important as knowing *that* it doesn't know.

### 4.3 Attention Entropy Interpretability Layer (New Novel Feature)

During draft generation, extract **attention weight distributions** from the model's layers and compute entropy over the attention heads. When the model is confident, attention concentrates on specific input tokens. When uncertain, attention spreads uniformly.

**Implementation approach:**
- llama-cpp-python does not natively expose attention weights, but they can be extracted via the `llama_get_logits()` C API or by patching the model's forward pass
- Alternative (simpler): use the `logits_all=True` scores to compute a proxy — the gradient of the output logits with respect to input token embeddings gives saliency maps
- Produce a heatmap: input question tokens on one axis, attention concentration on the other
- Integrate into the Gradio UI and the Vercel demo as a visualisation panel

**If attention extraction proves too complex for llama-cpp-python:** Fall back to input saliency via logit perturbation (mask each input token, re-run draft, measure output change). This is slower but guaranteed to work. Document the choice in the report.

### 4.4 Interactive Demo App (Vercel-Hosted)

**Vision:** A public-facing web app where users can:
1. Enter a medical question (text input)
2. Optionally upload a PDF medical report for context-specific Q&A
3. Choose an API backend (enter their own OpenAI/compatible API key, or use a provided default key with rate limiting)
4. See the full pipeline visualised: gate signals firing, token-level entropy heatmap, attention map, retrieval decision, retrieved chunks, final answer with citations

**Tech stack:**
- Frontend: React/Next.js on Vercel
- Backend: FastAPI on a cloud VM (or Railway/Render) running the full pipeline
- PDF handling: `PyMuPDF` or `pdfplumber` for text extraction, chunked and fed into the retrieval context
- API options: GPT-4o-mini (OpenAI), DeepSeek-V2 (cheaper Chinese alternative), or local GGUF via the backend
- The demo is separate from the core benchmark — it's a showcase artifact, not the evaluation system

**Scope for 6-week plan:** Build a minimal version in Week 3 (Gradio locally), polish into Vercel app in Week 5–6 if time permits. The Gradio version is the MVP; the Vercel version is a stretch goal.

### 4.5 Report Structure Changes

- **Background → Related Works**: Rename the section to "Related Works" and include the frameworks/pipelines that inspired the gating mechanisms (TARG, Self-RAG, Adaptive-RAG), showing how this project differentiates from them
- **Motivation moves into Introduction/Overview**: Fold the personal motivation (Thailand, LMIC healthcare) into Section 1.1 rather than a separate 1.4 subsection. Makes the opening more compelling.
- **Remove personal metrics from overview**: The internship stats (85% accuracy, 80% time reduction from KPIT) are fine in a motivation paragraph but don't belong in the project overview section
- **15,000 word maximum**: All chapters must fit within this. Budget roughly: Introduction (1,500), Related Works (2,500), Specification & Design (2,000), Implementation (4,000), Evaluation (3,000), LSEP (800), Conclusion (700) = ~14,500 with buffer
- **Interpretability emphasis**: The report should have a dedicated subsection in Implementation covering the token-level entropy attribution and attention entropy visualisation, positioned as novel contributions that differentiate this from existing adaptive retrieval work

---

## 5. The Seven Retrieval Policies (Updated)

| Policy | Name | Description | Hardware | Role |
|--------|------|-------------|----------|------|
| **P1** | Always-Retrieve | BM25 + FAISS every query, no gate | Medium+ | Accuracy ceiling baseline |
| **P2** | Always-Retrieve + Citation | P1 + mandatory source citation extraction | Medium+ | Provenance baseline |
| **P3** | Closed-Book | No retrieval; parametric knowledge only | Any | Speed/energy floor baseline |
| **P4** | Hybrid Retrieval | Routes queries to BM25 or FAISS based on type; combines via RRF | Medium+ | Balanced routing |
| **P5** | Selective/Gated | Training-free gate (entropy + margin + hallucination probe) decides per-query | Any | **Core novelty** |
| **P6** | User-Choice Toggle | User picks "Fast Answer" vs "Sourced Answer" in Gradio UI | Any | HCI component |
| **P7** | RAG-Gated Multi-Agent | P5 gate triggers retrieval; multiple specialist agents review; meta-agent synthesises | Higher only | Extension |

---

## 6. Gating Mechanisms — P5 (Revised Ensemble)

The P5 gate uses three training-free signals combined via majority vote. **The verbalized confidence gate has been replaced with the Hallucination Probe Gate.**

### Gate 1: Token Entropy Gate

Generate a 32–64 token draft without retrieval. Compute mean token entropy:

```
H̄ = (1/N) Σₜ H(pₜ)    where H(pₜ) = -Σᵥ pₜ(v) · log pₜ(v)
```

If `H̄ > τ_H` (default: 1.5) → **RETRIEVE**

**Additionally:** compute per-token entropy `H(pₜ)` and return as an array for the **Token-Level Entropy Attribution** visualisation. This is the interpretability feature — the gate decision uses the mean, but the per-token values are displayed to the user.

Basis: TARG (arXiv 2511.09803, 2025)

### Gate 2: Logit Margin Gate

From the draft's logits, compute the gap between top-1 and top-2 softmax probabilities:

```
M̄ = (1/N) Σₜ (pₜ⁽¹⁾ - pₜ⁽²⁾)
```

If `M̄ < τ_M` (default: 0.25) → **RETRIEVE** (model is indecisive)

Basis: TARG margin signal

### Gate 3: Hallucination Probe Gate (NEW — replaces Verbalized Confidence)

Generate two short drafts using different prompt framings:

```
Prompt A: "Answer the following medical question briefly: {question}"
Prompt B: "What is the correct answer to: {question}"
```

Compare the two draft answers:
- **MCQ:** Extract answer letter from each draft. If letters differ → **RETRIEVE**
- **Open-ended:** Compute token-level F1 between drafts. If F1 < 0.7 → **RETRIEVE**
- If both drafts agree → **SKIP**

**Why this is better than verbalized confidence:**
- Tests actual knowledge consistency, not self-reported confidence
- Grounded in model behaviour, not metacognition (which small models are bad at)
- Catches cases where the model confidently gives a *wrong* answer — verbalized would say HIGH, but the probe detects instability across prompt variations
- Related to self-consistency (Wang et al., 2023) but used as a binary gate signal

**Trade-off:** 2× LLM calls (same cost as verbalized). Document this overhead.

### Final Decision: Majority Vote

```
g = RETRIEVE  if  Σᵢ 1[gateᵢ = RETRIEVE] ≥ 2
g = SKIP      otherwise
```

where `i ∈ {entropy, margin, hallucination_probe}`

### Interpretability Layer: Attention Entropy (Not a Gate — Visualisation Only)

During draft generation, compute attention entropy to produce a heatmap showing where the model focuses. This is rendered in the UI but does not affect the gate decision.

**Primary approach:** Input saliency via logit perturbation — mask each input token, re-run a short draft, measure the change in output logit distribution. Tokens whose masking causes large output changes are "important". Display as a coloured overlay on the question text.

**Fallback:** If perturbation is too slow, use the token-level entropy from Gate 1 as a proxy for the attention visualisation.

---

## 7. Hardware Tiers

| Tier | RAM | Maps To | Model | Context | Retrieval |
|------|-----|---------|-------|---------|-----------|
| **Low** | 4 GB | Thai rural HPH, old NHS community | Llama 3.2-1B Q4_K_M | `n_ctx=1024` | BM25-only, `logits_all=false` |
| **Medium** | 8 GB | NHS community workstations | Llama 3.2-3B Q4_K_M | `n_ctx=2048` | BM25 + FAISS hybrid |
| **Higher** | 16 GB+ | NHS Link 4, Thai urban | Phi-3.5-mini Q5_K_M | `n_ctx=4096` | Full MedCorp + RRF + P7 MAS |

**Low tier note:** With `logits_all=false`, entropy and margin gates are unavailable. Only the hallucination probe gate works (it uses `answer()` not `_scores`). On low tier, P5 falls back to hallucination-probe-only gating. Document this as a finding.

---

## 8. Datasets

| Dataset | Size | Risk Level | Source | Priority |
|---------|------|-----------|--------|---------|
| **MIRAGE** | 7,663 Qs | Low–Medium | MedRAG GitHub | **Primary** |
| **RAGCare-QA** | 420 Qs | Low–High | HuggingFace | **Primary** |
| **DxBench** | 1,148 Qs | Medium–High | MedMASLab repo | **Primary** |
| **MedBullets** | 308 Qs | Low–Medium | MedMASLab repo | Secondary |
| **Synthetic Acute-Care** | ~150 Qs | **High** | Self-built from BNF/WHO/NICE | **Primary for P7** |

---

## 9. Retrieval Corpora

Default for Low/Medium tier: **StatPearls + Textbooks (~427K snippets)**. Higher tier: full MedCorp via RRF.

Chunk size: 256 tokens. Via MedRAG toolkit.

---

## 10. Tech Stack

### Install

```bash
pip install -e ".[dev]"
# Or manually:
pip install llama-cpp-python rank-bm25 faiss-cpu sentence-transformers
pip install codecarbon psutil pandas matplotlib seaborn gradio pyyaml pydantic
```

### GGUF Models

```bash
# Primary — Llama 3.2-3B Q4_K_M (~2.5 GB)
huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF \
  --include "Llama-3.2-3B-Instruct-Q4_K_M.gguf" --local-dir ./models

# Low tier fallback
huggingface-cli download bartowski/Llama-3.2-1B-Instruct-GGUF \
  --include "Llama-3.2-1B-Instruct-Q4_K_M.gguf" --local-dir ./models

# Higher tier
huggingface-cli download microsoft/Phi-3.5-mini-instruct-gguf \
  --include "Phi-3.5-mini-instruct-Q5_K_M.gguf" --local-dir ./models
```

---

## 11. New Files to Create (Not in Repo Yet)

These files need to be added to the existing repo structure:

```
src/medrag_adaptive/
├── gating/
│   ├── base.py                      # Gate ABC + GateDecision enum
│   ├── entropy_gate.py              # Token entropy + per-token attribution
│   ├── margin_gate.py               # Logit margin gate
│   ├── hallucination_probe_gate.py  # NEW — draft consistency gate
│   └── attention_entropy.py         # NEW — interpretability layer (not a gate)
├── retrieval/
│   ├── base.py                      # Retriever ABC
│   ├── bm25_retriever.py            # rank_bm25 wrapper
│   ├── vector_retriever.py          # FAISS + sentence-transformers
│   ├── hybrid_retriever.py          # RRF fusion
│   └── medrag_corpus.py             # Corpus loader/chunker
├── policies/
│   ├── base.py                      # Policy ABC
│   ├── p1_always_retrieve.py
│   ├── p2_always_retrieve_cite.py
│   ├── p3_closed_book.py
│   ├── p4_hybrid.py
│   ├── p5_gated.py                  # Composes 3 gates + retriever + LLM
│   ├── p6_user_toggle.py
│   ├── p7_multi_agent.py            # Extension
│   └── factory.py                   # PolicyFactory.build(cfg) → Policy
├── data/
│   ├── loaders/
│   │   ├── mirage_loader.py
│   │   ├── ragcare_loader.py
│   │   ├── dxbench_loader.py
│   │   └── synthetic_loader.py
│   ├── risk_tagger.py
│   └── synthetic_builder.py
├── evaluation/
│   ├── metrics.py                   # EM, F1, accuracy, ECE, citation P/R
│   ├── profiler.py                  # Latency + energy + memory wrappers
│   ├── harness.py                   # BenchmarkHarness (sequential, resumable)
│   ├── safety_envelope.py           # Per-risk minimum policy analysis
│   └── results.py                   # ExperimentSummary aggregation
├── ui/
│   └── gradio_app.py               # P6 toggle + entropy heatmap + attention viz
└── demo/                            # NEW — Vercel demo app (stretch goal)
    ├── api_backend.py               # FastAPI wrapper for the pipeline
    ├── pdf_handler.py               # PDF text extraction for context Q&A
    └── README.md                    # Demo deployment instructions

scripts/
├── download_datasets.py
├── build_indexes.py
├── run_threshold_sweep.py
├── run_full_experiment.py
├── run_all_experiments.sh
└── generate_figures.py
```

---

## 12. Prompt Templates to Add

The existing `prompts.py` needs these additions:

### Hallucination Probe — Prompt A

```python
HALLUCINATION_PROBE_A = """\
[INST] You are a medical assistant. Answer the following medical question briefly.

Question: {question}
{choices}
{instruction} [/INST]"""
```

### Hallucination Probe — Prompt B

```python
HALLUCINATION_PROBE_B = """\
[INST] What is the correct answer to the following medical question? Be concise.

{question}
{choices}
{instruction} [/INST]"""
```

### Remove Verbalized Confidence Prompt

The `VERBALIZED_CONFIDENCE_TEMPLATE` and `build_verbalized_confidence_prompt()` in `prompts.py` should be deprecated (keep in file but mark as unused). The hallucination probe does not need a special confidence-check prompt — it reuses the standard draft/answer prompts with different framings.

---

## 13. YAML Config Updates Needed

### Update `configs/base.yaml` gate section:

```yaml
gate:
  type: ensemble                  # entropy | margin | hallucination_probe | ensemble
  entropy_threshold: 1.5          # τ_H — retrieve if mean entropy > threshold
  margin_threshold: 0.25          # τ_M — retrieve if mean margin < threshold
  draft_max_tokens: 48
  # Hallucination probe settings
  hallucination_probe:
    enabled: true
    agreement_mode: letter_match  # letter_match (MCQ) | f1_threshold (open-ended)
    f1_threshold: 0.7             # for open-ended: retrieve if F1 < this
  # Ensemble mode: majority vote across all enabled gates
  ensemble_threshold: 2           # retrieve if >= N gates vote RETRIEVE
```

### Update `configs/hardware_low.yaml`:

```yaml
gate:
  type: hallucination_probe       # only gate available when logits_all=false
  hallucination_probe:
    enabled: true
```

### Update `configs/policies/p5_gated_entropy.yaml` → rename to `p5_gated_ensemble.yaml`:

```yaml
policy:
  name: p5_gated
  description: "Training-free gating with 3-gate ensemble: entropy + margin + hallucination probe."
  retrieval_mode: hybrid
  cite_sources: false

gate:
  type: ensemble
  entropy_threshold: 1.5
  margin_threshold: 0.25
  draft_max_tokens: 48
  hallucination_probe:
    enabled: true
    agreement_mode: letter_match
```

### Remove `configs/policies/p5_gated_verbalized.yaml`

Replace with `configs/policies/p5_gated_hallucination.yaml` for standalone hallucination-probe-only experiments.

---

## 14. Experiment Logging Schema (Updated)

```json
{
    "qid": "mirage_001",
    "policy": "p5",
    "tier": "medium",
    "model": "Llama-3.2-3B-Instruct-Q4_K_M",
    "gate_entropy": 1.82,
    "gate_entropy_per_token": [0.5, 1.2, 2.1, ...],
    "gate_margin": 0.19,
    "gate_hallucination_probe": {
        "draft_a_answer": "B",
        "draft_b_answer": "C",
        "agreement": false
    },
    "gate_decision": "retrieve",
    "gate_votes": {"entropy": "retrieve", "margin": "retrieve", "hallucination_probe": "retrieve"},
    "retrieved": true,
    "num_chunks": 3,
    "correct": true,
    "predicted": "B",
    "latency_ms": 842.3,
    "energy_kwh": 0.000023,
    "risk_level": "medium",
    "dataset": "mirage",
    "seed": 42,
    "attention_entropy_available": true
}
```

The `gate_entropy_per_token` array is new — it stores the per-token entropy values for the interpretability visualisation. Only populated when entropy gate is active.

---

## 15. Implementation Priority Order (Updated for Current Repo State)

Since the scaffold, config system, LLM backend, schema, and tests are already built, the build order starts at data loading.

### Phase 1: Data Pipeline (Week 1, Days 1–3)

1. `src/medrag_adaptive/data/loaders/mirage_loader.py` — parse MIRAGE benchmark.json into `UnifiedQuestion` list
2. `src/medrag_adaptive/data/loaders/ragcare_loader.py` — parse RAGCare-QA from HuggingFace
3. `src/medrag_adaptive/data/risk_tagger.py` — keyword heuristic classifier
4. `src/medrag_adaptive/evaluation/profiler.py` — energy/latency/memory wrapper
5. `scripts/download_datasets.py` — automated dataset download

### Phase 2: Baseline Policies (Week 1, Days 4–7)

6. `src/medrag_adaptive/policies/base.py` — Policy ABC
7. `src/medrag_adaptive/policies/p3_closed_book.py` — simplest policy
8. `src/medrag_adaptive/retrieval/bm25_retriever.py` — BM25 index over StatPearls
9. `src/medrag_adaptive/policies/p1_always_retrieve.py` — BM25-only for now
10. Run P1 + P3 on 200 MIRAGE questions with full logging

### Phase 3: Retrieval Engine (Week 2, Days 1–3)

11. `src/medrag_adaptive/retrieval/vector_retriever.py` — FAISS + MiniLM
12. `src/medrag_adaptive/retrieval/hybrid_retriever.py` — RRF fusion
13. `scripts/build_indexes.py` — build both indexes
14. `src/medrag_adaptive/policies/p2_always_retrieve_cite.py`
15. `src/medrag_adaptive/policies/p4_hybrid.py`

### Phase 4: Gates — Core Novelty (Week 2, Days 3–7)

16. `src/medrag_adaptive/gating/base.py` — Gate ABC + GateDecision
17. `src/medrag_adaptive/gating/entropy_gate.py` — including per-token attribution
18. `src/medrag_adaptive/gating/margin_gate.py`
19. `src/medrag_adaptive/gating/hallucination_probe_gate.py` — NEW
20. `src/medrag_adaptive/policies/p5_gated.py` — compose gates + majority vote
21. `src/medrag_adaptive/policies/factory.py` — PolicyFactory
22. Threshold calibration sweep on 500 questions

### Phase 5: Interpretability Layer (Week 3, Days 1–2)

23. `src/medrag_adaptive/gating/attention_entropy.py` — saliency/attention maps
24. Integrate token-level entropy + attention maps into Gradio UI

### Phase 6: Full Experiments (Week 3, Days 3–7)

25. `src/medrag_adaptive/evaluation/metrics.py`
26. `src/medrag_adaptive/evaluation/harness.py`
27. `scripts/run_full_experiment.py`
28. Run P1–P5 on all datasets × 3 tiers
29. `src/medrag_adaptive/evaluation/safety_envelope.py`
30. `src/medrag_adaptive/ui/gradio_app.py` — P6 with visualisations

### Phase 7: Analysis + Writing (Weeks 4–5)

31. `scripts/generate_figures.py`
32. `src/medrag_adaptive/evaluation/results.py`
33. P7 multi-agent extension (if time)
34. All report chapters

### Phase 8: Demo + Polish (Week 5–6)

35. Demo app (Vercel) — stretch goal
36. Report revisions, presentation, submission

---

## 16. The 6-Week Compressed Plan (Updated)

### Week 1 (June 9–15): Data Pipeline + Baselines (P1, P3)

**Purpose:** The scaffold is already built. This week fills it with data and gets the first real results.

**Tasks:**
1. Run `scripts/setup_environment.sh` — verify everything installs
2. Implement `mirage_loader.py` and `ragcare_loader.py`
3. Implement `risk_tagger.py`
4. Implement `profiler.py` (energy/latency wrapper)
5. Implement `Policy` ABC and `p3_closed_book.py`
6. Implement `bm25_retriever.py` and `p1_always_retrieve.py` (BM25-only)
7. Run P1 + P3 on 200 MIRAGE questions. Log results.
8. Draft Specification & Design chapter while experiments run

**Deliverables:** P1+P3 running, 200-question results, Spec/Design draft

**Milestone:** M11 — environment + baseline inference

---

### Week 2 (June 16–22): Retrieval Engine + All 3 Gates + P5

**Purpose:** Build the core novelty. The hallucination probe gate and token-level entropy attribution are what make this project unique.

**Tasks:**
1. Build FAISS index, implement `vector_retriever.py` and `hybrid_retriever.py`
2. Implement `entropy_gate.py` with per-token attribution output
3. Implement `margin_gate.py`
4. Implement `hallucination_probe_gate.py` (two-draft consistency check)
5. Implement `p5_gated.py` with majority vote ensemble
6. Implement P2 and P4
7. Implement `PolicyFactory`
8. Run threshold calibration on 500-question subset

**Risk flag:** If hallucination probe is not working by Wednesday, fall back to a simplified version (just compare first extracted letter from two `answer()` calls). Refine later.

**Deliverables:** All 3 gates tested, P1–P5 operational, threshold curves

**Milestone:** M12 — retrieval engine + gating

---

### Week 3 (June 23–29): Full Experiments + Interpretability + P6 UI

**Purpose:** Generate all experimental evidence and build the interpretability visualisations.

**Tasks:**
1. Implement `attention_entropy.py` (input saliency or attention extraction)
2. Run P1–P5 on full MIRAGE × 3 tiers (overnight run)
3. Run P1–P5 on RAGCare-QA and DxBench
4. Compute safety envelopes
5. Implement `gradio_app.py` with: question input, gate visualisation panel (token entropy heatmap, attention map), retrieval decision display, answer + citations
6. Generate Pareto front plots per tier
7. Implement P6 user toggle in Gradio

**Deliverables:** Full results, safety envelopes, Pareto plots, Gradio UI with interpretability

**Milestones:** M13 (full experiments), M14 (safety envelopes)

---

### Week 4 (June 30–July 6): Analysis + Core Report Chapters

**Purpose:** Convert raw results into findings. Write Introduction, Related Works, LSEP.

**Tasks:**
1. Per-tier recommendation tables
2. Gate failure analysis (false-skips, false-retrievals)
3. Gate ablation study (each gate solo vs ensemble)
4. Hallucination probe effectiveness analysis (how often does it catch errors the other gates miss?)
5. Design P7 multi-agent prompts, unit-test on 5 questions
6. Write **Introduction** (~1,500 words) — motivation folded into overview, no separate section
7. Write **Related Works** (~2,500 words) — renamed from "Background", includes framework comparisons
8. Write **LSEP** (~800 words)

**Deliverables:** Analysis tables, failure cases, 3 report chapters drafted

---

### Week 5 (July 7–13): P7 + Implementation Chapter + Evaluation Chapter

**Purpose:** The two highest-weighted chapters (45% combined). Complete report draft to supervisor.

**Tasks:**
1. Run P7 experiments on Higher Tier (200 high-risk questions + DxBench)
2. Write **Implementation** (~4,000 words) — include dedicated subsection on token-level entropy attribution and attention entropy as novel interpretability features
3. Write **Evaluation** (~3,000 words) — answer each RQ with evidence, compare simulated vs real results
4. Finalise **Spec & Design** (~2,000 words)
5. Assemble complete first draft, send to supervisor
6. If time: begin Vercel demo app (FastAPI backend + React frontend)

**Deliverables:** Complete report draft to supervisor, P7 results

**Milestones:** M15 (P7), M16 partial (first draft)

---

### Week 6 (July 14–20): Polish + Demo + Presentation + Submission

**Purpose:** Quality, not content. Supervisor feedback, demo rehearsal, formatting.

**Tasks:**
1. Incorporate supervisor feedback
2. Prepare live demo (3 scenarios showing gate decisions + interpretability visualisations)
3. Prepare presentation slides (12–15)
4. Final proofreading (15,000 word limit check, bibliography, figures)
5. Clean code repo (README, reproducibility scripts)
6. If Vercel demo is viable: deploy; otherwise, Gradio local demo is the fallback
7. Submit report, deliver presentation

**Deliverables:** Final report, presentation, demo, clean repo

**Milestone:** M16 — project submitted

---

## 17. Key Design Decisions — Do Not Change

- Seven-policy structure (P1–P7) is fixed
- Three hardware tiers map to real hospital contexts
- Safety envelope concept is the primary novel contribution
- P7 is extension, not core — do not let it delay P1–P6
- All experiments use fixed seed 42
- All results log to JSONL with full schema
- Hardware tiers set via YAML config — no hardcoded paths
- **Hallucination probe replaces verbalized confidence** — do not revert
- **Token-level entropy attribution is a novel contribution** — ensure it's prominent in the report

---

## 18. Simulated Preliminary Results (Baseline for Comparison)

These are from the simulation notebook. Compare real results against these in the Evaluation chapter.

| Policy | Simulated Accuracy | Simulated Latency | Retrieval Rate |
|--------|-------------------|------------------|----------------|
| P1 (Always-Retrieve) | 0.773 | 1,454 ms | 100% |
| P3 (Closed-Book) | 0.570 | 350 ms | 0% |
| P5 (Gated) | 0.694 | 795 ms | 37.9% |

P5 achieved 89.8% of P1 accuracy with 45.3% less latency and 37.9% retrieval rate in simulation.

---

## 19. Risk Register (Updated)

| Risk | Prob. | Impact | Mitigation |
|------|-------|--------|-----------|
| GGUF model too slow on Low Tier | Med | High | Fall back to 1B model. Report as finding. |
| Hallucination probe adds too much latency | Med | Med | Profile 2× draft cost. If >2s/query, reduce draft_max_tokens to 24. |
| Attention extraction not possible in llama-cpp-python | High | Med | Fall back to input saliency via logit perturbation. Document the choice. |
| Gate thresholds don't transfer across question types | High | Med | This is a finding. Report per-type optimal thresholds. |
| P7 too slow even on Higher Tier | Med | Low | P7 is extension. Report latency multiplier. |
| 15k word limit too tight for all content | Med | Med | Prioritise Implementation + Evaluation. Move supplementary material to appendix or GitHub. |
| Vercel demo not completed in time | High | Low | Gradio local demo is the fallback. Demo is a stretch goal. |

---

## 20. References (28 papers — same as v1, add self-consistency)

All 28 papers from v1 remain. Add:

29. Wang et al. "Self-Consistency Improves Chain of Thought Reasoning in Language Models" — ICLR 2023. Basis for the hallucination probe gate's consistency-checking approach, adapted from majority-vote sampling to binary agreement detection.

---

---

## 21. Progress Log

### June 12, 2026 (Session 1 — v1 baseline)
Repo scaffold complete: config, schema, LLM backend (`llama_backend.py`), prompts, all `__init__.py`, `pyproject.toml`, `requirements.txt`, `setup_environment.sh`, `tests/conftest.py`, `tests/fixtures/`, `tests/unit/test_config.py`. No policies, gates, retrievers, or experiments implemented.

### June 18, 2026 (Session 2 — v2 update)
**Status: End of Week 2. Behind on Week 2 targets.**

**Completed since v1:**

Phase 1 (Week 1, Days 1–3) ✅
- `src/medrag_adaptive/data/loaders/base.py`
- `src/medrag_adaptive/data/loaders/mirage_loader.py`
- `src/medrag_adaptive/data/loaders/ragcare_loader.py`
- `src/medrag_adaptive/data/risk_tagger.py`
- `src/medrag_adaptive/evaluation/profiler.py`
- `src/medrag_adaptive/evaluation/scoring.py`
- `scripts/download_datasets.py`
- `scripts/run_experiment.py` (partial runner)

Phase 2 (Week 1, Days 4–7) ✅
- `src/medrag_adaptive/policies/base.py`
- `src/medrag_adaptive/policies/p1_always_retrieve.py`
- `src/medrag_adaptive/policies/p3_closed_book.py`
- `src/medrag_adaptive/retrieval/base.py`
- `src/medrag_adaptive/retrieval/bm25_retriever.py`
- `src/medrag_adaptive/gating/base.py` (stub only — Gate ABC exists, no concrete gates)

Tests added:
- `tests/unit/test_bm25_retriever.py`
- `tests/unit/test_loaders.py`
- `tests/unit/test_policies_baseline.py`
- `tests/unit/test_profiler.py`
- `tests/unit/test_runner_resume.py`
- `tests/unit/test_scoring.py`

**Still missing (Week 2 targets not yet done):**

Phase 3 — Retrieval engine:
- `src/medrag_adaptive/retrieval/vector_retriever.py` (FAISS + MiniLM)
- `src/medrag_adaptive/retrieval/hybrid_retriever.py` (RRF fusion)
- `scripts/build_indexes.py`
- `src/medrag_adaptive/policies/p2_always_retrieve_cite.py`
- `src/medrag_adaptive/policies/p4_hybrid.py`

Phase 4 — Core novelty (gates):
- `src/medrag_adaptive/gating/entropy_gate.py` (+ per-token attribution)
- `src/medrag_adaptive/gating/margin_gate.py`
- `src/medrag_adaptive/gating/hallucination_probe_gate.py`
- `src/medrag_adaptive/policies/p5_gated.py`
- `src/medrag_adaptive/policies/factory.py`

Phase 5–8 — Not started:
- `evaluation/harness.py`, `evaluation/metrics.py`, `evaluation/safety_envelope.py`, `evaluation/results.py`
- `gating/attention_entropy.py`
- `ui/gradio_app.py`
- All analysis scripts, figures, report chapters, demo app

**No real experiments run yet** — no data downloaded, no indexes built, no GGUF model on disk. Next priority: run `scripts/setup_environment.sh`, download model and datasets, then complete Phase 3 + 4 immediately.

*Last updated: June 18, 2026.*
