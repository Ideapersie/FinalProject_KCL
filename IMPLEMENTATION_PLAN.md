# Implementation Plan
## "When to Retrieve? Adaptive Gating Policies for Offline Medical RAG in Hospital-Critical Scenarios"
### KCL MSc AI — Variant 5 Final Project

---

## Project Goal

Build and benchmark **6 retrieval policies** for medical QA on CPU-only hardware.
The core novelty is **Policy P5 (Selective/Gated Retrieval)** — three training-free gating mechanisms that decide *per-query* whether to retrieve or answer from parametric knowledge.

Target grade: **Distinction (80–100)**

---

## Directory Structure

```
FinalProject_KCL/
├── IMPLEMENTATION_PLAN.md           ← this file
├── Variant5_Hospital_Critical_Care_Project_Plan.md
├── pyproject.toml
├── requirements.txt
├── README.md
│
├── configs/
│   ├── base.yaml                    shared defaults (seed, paths, model, retrieval, gate, evaluation)
│   ├── hardware_low.yaml            4 GB / dual-core overrides
│   ├── hardware_medium.yaml         8 GB / quad-core overrides
│   ├── hardware_high.yaml           16 GB+ / 6-core overrides
│   ├── policies/
│   │   ├── p1_always_retrieve.yaml
│   │   ├── p2_always_retrieve_cite.yaml
│   │   ├── p3_closed_book.yaml
│   │   ├── p4_hybrid.yaml
│   │   ├── p5_gated_entropy.yaml
│   │   ├── p5_gated_margin.yaml
│   │   ├── p5_gated_verbalized.yaml
│   │   └── p6_user_toggle.yaml
│   └── experiments/
│       ├── threshold_sweep.yaml
│       ├── full_mirage.yaml
│       ├── ragcare.yaml
│       └── acute_care.yaml
│
├── src/medrag_adaptive/             installable Python package
│   ├── __init__.py
│   ├── config.py                    YAML load + Pydantic ProjectConfig
│   ├── models/
│   │   ├── base.py                  LLMBackend ABC
│   │   ├── llama_backend.py         llama-cpp-python wrapper + logit extraction
│   │   ├── openai_backend.py        GPT-4o-mini reference (API only)
│   │   └── prompts.py               all prompt templates (centralised)
│   ├── retrieval/
│   │   ├── base.py                  Retriever ABC
│   │   ├── bm25_retriever.py        rank_bm25 in-memory
│   │   ├── vector_retriever.py      FAISS + sentence-transformers
│   │   ├── hybrid_retriever.py      RRF fusion
│   │   └── medrag_corpus.py         MedRAG corpus loader
│   ├── gating/
│   │   ├── base.py                  Gate ABC + GateDecision enum
│   │   ├── entropy_gate.py          Token entropy gate (core novelty)
│   │   ├── margin_gate.py           Logit margin gate (core novelty)
│   │   └── verbalized_gate.py       Verbalized confidence gate
│   ├── policies/
│   │   ├── base.py                  Policy ABC + PolicyResult dataclass
│   │   ├── p1_always_retrieve.py
│   │   ├── p2_always_retrieve_cite.py
│   │   ├── p3_closed_book.py
│   │   ├── p4_hybrid.py
│   │   ├── p5_gated.py              composes gate + retriever + LLM
│   │   ├── p6_user_toggle.py
│   │   └── factory.py               PolicyFactory.build(cfg) → Policy
│   ├── data/
│   │   ├── schema.py                UnifiedQuestion + RunRecord dataclasses
│   │   ├── loaders/
│   │   │   ├── mirage_loader.py
│   │   │   ├── ragcare_loader.py
│   │   │   ├── ehrdsqa_loader.py    (stub until PhysioNet approval)
│   │   │   └── synthetic_loader.py
│   │   ├── risk_tagger.py           keyword heuristics + manual overrides
│   │   └── synthetic_builder.py    builds acute-care question set
│   ├── evaluation/
│   │   ├── metrics.py               EM, F1, accuracy, ECE, citation P/R
│   │   ├── profiler.py              latency + energy + memory wrappers
│   │   ├── harness.py               BenchmarkHarness (sequential, resumable)
│   │   ├── safety_envelope.py       per-risk-level minimum policy analysis
│   │   └── results.py               RunRecord + ExperimentSummary + JSONL I/O
│   └── ui/
│       └── gradio_app.py            P6 Fast/Sourced toggle interface
│
├── scripts/
│   ├── setup_environment.sh         install deps, create dirs, download first model
│   ├── download_datasets.py         MIRAGE, RAGCare-QA, MedRAG corpora
│   ├── build_indexes.py             build BM25 pickle + FAISS index (idempotent)
│   ├── run_threshold_sweep.py       Sprint 4 calibration
│   ├── run_full_experiment.py       main experiment driver (reads experiment YAML)
│   ├── run_all_experiments.sh       orchestrates all experiments end-to-end
│   └── generate_figures.py         Pareto plots, heatmaps, safety envelope tables
│
├── data/                            (all git-ignored for large files)
│   ├── raw/mirage/
│   ├── raw/ragcare_qa/
│   ├── raw/acute_care_raw/
│   ├── processed/                   unified JSONL files
│   └── corpora/                     statpearls/, textbooks/, bnf_chunks/, who_mhgap_chunks/
│
├── indexes/                         (git-ignored) BM25 pkl + FAISS .index files
├── models/                          (git-ignored) GGUF model files
├── results/
│   ├── raw_logs/                    per-query JSONL audit logs
│   ├── aggregated/                  summary CSVs + JSONs
│   └── figures/                     PNG/PDF plots
│
└── tests/
    ├── conftest.py                  MockLLMBackend (no real model needed)
    ├── fixtures/                    sample_questions.jsonl, mock_corpus.jsonl
    ├── unit/                        fast tests, no model
    ├── integration/                 end-to-end pipeline tests, no model
    ├── smoke/                       @pytest.mark.slow — requires real GGUF
    └── regression/                  after Sprint 5 — fixed expected accuracy
```

---

## Sprint Roadmap

| Sprint | Weeks | Focus | Milestone |
|--------|-------|-------|-----------|
| **1** | 1–2 | Foundation: scaffold, config, schema, LLM backend | Single closed-book inference on CPU |
| **2** | 3–4 | Baseline policies (P1, P3) + data pipeline + profiler | P1+P3 on 100 MIRAGE Qs with full logging |
| **3** | 5–6 | Retrieval engine (BM25, FAISS, RRF) + P2, P4 | All 4 non-gated policies runnable |
| **4** | 7–8 | **Core novelty**: 3 gating mechanisms + P5 + calibration | Gate reduces retrieval 40–70% on 500-Q subset |
| **5** | 9–10 | Full experiments: all 6 policies × 3 tiers × 3 datasets + P6 | All results JSONL files produced |
| **6** | 11–12 | Analysis, Pareto curves, safety envelopes, user study | All figures + tables ready |
| **7** | 13–14 | Report writing + reproducibility packaging | First complete draft |
| **8** | 15–16 | Polish, presentation, submission | Submitted |

---

## The 6 Policies

| ID | Name | Retrieval | Novelty |
|----|------|-----------|---------|
| P1 | Always-Retrieve | BM25 + vector every query | Baseline |
| P2 | Always-Retrieve + Citation | P1 + source citation extraction | Baseline |
| P3 | Closed-Book | None — parametric knowledge only | Baseline |
| P4 | Hybrid Retrieval | BM25 vs. vector routing + RRF | Baseline |
| **P5** | **Selective/Gated** | **Training-free gate decides per-query** | **CORE NOVELTY** |
| P6 | User-Choice Toggle | User selects Fast vs. Sourced (Gradio) | HCI component |

---

## Core Novelty: 3 Gating Mechanisms (P5)

### Token Entropy Gate
- Generate draft answer (32–64 tokens) without retrieval
- Compute mean token entropy from full-vocab logits: `H = -Σ p·log(p)`
- `mean_entropy > threshold` → **RETRIEVE**; else → **SKIP**
- Access via `llm._scores` numpy array (requires `logits_all=True`)
- Basis: TARG (arXiv 2025)

### Logit Margin Gate
- Compute gap: `margin = p_top1 - p_top2` per generated token
- `mean_margin < threshold` → **RETRIEVE** (model is indecisive); else → **SKIP**
- Access via `logprobs=2` in `create_completion` response
- Basis: TARG margin signal

### Verbalized Confidence Gate
- Prompt model: *"How confident are you answering this WITHOUT external sources? (HIGH/MEDIUM/LOW)"*
- `MEDIUM` or `LOW` → **RETRIEVE**; `HIGH` → **SKIP**
- Parse with `re.search(r'\b(HIGH|MEDIUM|LOW)\b', response.upper())`
- Default to `RETRIEVE` on parse failure (safe default)
- Note: 2× LLM calls per query — report latency overhead explicitly
- Basis: Kadavath et al. 2022

---

## Critical Technical Decisions

1. **Logit extraction**: `logits_all=True` at `Llama()` construction (+256 MB RAM). Entropy gate uses `llm._scores` (full vocab). Margin gate uses `logprobs=2` API param. Low tier: `logits_all=False` → verbalized gate only.

2. **BM25**: Use `rank_bm25` (not Whoosh — last release 2013, uncertain Python 3.12 compat). Stays in-memory after pickle load.

3. **FAISS**: Use `IndexFlatIP` (exact inner product / cosine after L2 normalisation). Deterministic. ~650 MB RAM, <100 ms per query.

4. **Determinism**: `temperature=0.0` + `seed=config.seed` at model construction. Both required.

5. **Energy measurement**: `codecarbon.EmissionsTracker(tracking_mode="process")` per-query, not global tracker. Linux: accesses Intel RAPL hardware counters directly.

6. **Sequential eval**: No parallelism in `BenchmarkHarness` — concurrent queries corrupt codecarbon readings.

7. **Resume support**: Harness appends to JSONL and skips already-logged `question_id` values. Required because full experiments take hours on CPU.

---

## Datasets

| Dataset | Size | Risk Tagging | Source |
|---------|------|-------------|--------|
| MIRAGE | 7,663 Qs | Keyword heuristics + manual overrides | MedRAG GitHub |
| RAGCare-QA | 420 Qs | complexity field: Basic→low, Intermediate→med, Advanced→high | HuggingFace |
| Synthetic Acute-Care | ~100–150 Qs | All high risk | Manual from BNF/WHO/NICE |
| EHR-DS-QA | TBD | — | PhysioNet (conditional) |

---

## Retrieval Corpora

| Corpus | Size | Use |
|--------|------|-----|
| StatPearls | 301K snippets | Clinical decision support |
| Medical Textbooks | 126K snippets | Foundational knowledge |
| BNF Open Data | Custom chunks | Drug interactions / dosing (high-risk) |
| WHO mhGAP | Custom chunks | Mental health / emergency (high-risk) |
| NICE Guidelines | Custom chunks | UK clinical protocols |

---

## Evaluation Metrics

- **Accuracy** (multi-choice): extract letter A–E via regex
- **Exact Match (EM)** + **F1** (token overlap): for open-ended questions
- **Expected Calibration Error (ECE)**: 10-bin histogram; gate signal vs. accuracy
- **Retrieval Rate**: fraction of queries where retrieval triggered
- **Latency**: `perf_counter_ns`, report p50 + p95
- **Energy**: kWh via codecarbon (Intel RAPL on Linux)
- **Peak Memory**: `tracemalloc` reset per-query
- **Citation Precision/Recall**: cited sources ∩ retrieved chunks (ALCE framework)

---

## Safety Envelopes (Key Output)

For each `(risk_level, policy)` pair, compute whether accuracy ≥ threshold:
- Low risk: ≥ 70% accuracy
- Medium risk: ≥ 80% accuracy
- High risk: ≥ 90% accuracy

Output: table of least-expensive policy meeting threshold at each risk level per hardware tier.
**This "safety envelope" concept is the publishable contribution of this project.**

---

## How to Run (once implemented)

```bash
# 1. Setup
bash scripts/setup_environment.sh

# 2. Download data
python scripts/download_datasets.py

# 3. Build indexes (run once)
python scripts/build_indexes.py

# 4. Run a single experiment
python scripts/run_full_experiment.py --config configs/experiments/full_mirage.yaml

# 5. Run all experiments
bash scripts/run_all_experiments.sh

# 6. Generate figures
python scripts/generate_figures.py

# 7. Tests
pytest tests/unit/                    # <5s, no model needed
pytest tests/unit/ tests/integration/ # <30s, no model needed
pytest -m slow                        # requires GGUF files
```

---

## Key References

- TARG (arXiv 2025) — training-free adaptive retrieval gating with entropy/margin signals
- Self-RAG (Asai et al., ICLR 2024) — reflection tokens for adaptive retrieval
- MIRAGE (Xiong et al., ACL 2024) — 7,663-question medical benchmark + MedRAG toolkit
- Kadavath et al. (arXiv 2022) — LLMs know what they don't know (verbalized confidence)
- Guo et al. (ICML 2017) — calibration of neural networks (ECE metric)
- Adaptive-RAG (Jeong et al., NAACL 2024) — query complexity routing
- ALCE (Gao et al., EMNLP 2023) — attribution in language models
- Wilkins et al. (HotCarbon 2024) — offline energy-optimal LLM serving
