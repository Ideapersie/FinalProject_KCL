# Variant 5: Adaptive Retrieval Gating for Hospital-Critical Medical QA Under Resource Constraints

## Project Title
**When to Retrieve? Adaptive Gating Policies for Offline Medical RAG in Hospital-Critical Scenarios**

---

## 1. Project Overview

### 1.1 What This Project Is

This project investigates **when retrieval is necessary** — and when it can be safely skipped — in a resource-constrained, offline medical question-answering (QA) system. Rather than always retrieving external knowledge (expensive) or always relying on the model's internal knowledge (risky in medicine), the system uses **lightweight gating signals** (entropy, logit margins, verbalized confidence) to decide per-query whether retrieval is needed.

The critical innovation is applying this to **hospital-critical medical domains** — emergency triage, drug interactions, acute care protocols, and clinical decision support — where:

- **Getting it wrong is dangerous**: a missed retrieval on a drug interaction question could mean a hallucinated answer in a life-critical context.
- **Resources are scarce**: NHS training wards, rural clinics, and LMIC hospitals often lack reliable internet, GPU hardware, or cloud budgets.
- **Provenance matters**: every answer that draws on retrieved evidence must cite its source; every answer from internal knowledge must be flagged as such.

The project produces a **reproducible benchmark harness**, **trade-off curves** (accuracy vs. latency vs. energy), and **evidence-backed policy recommendations** that map hardware tiers to the safest and most efficient retrieval strategy for different medical question types.

### 1.2 Why This Matters (Rubric Alignment)

| Rubric Section | Weight | How This Project Addresses It |
|---|---|---|
| **Introduction** | 10% | Positions work at the intersection of three active research areas: adaptive RAG, medical AI safety, and edge deployment. Connects to NHS training, LMIC healthcare, and the qVault offline appliance vision. |
| **Literature Review** | 10% | Synthesises Self-RAG, TARG, MIRAGE, MedRAG, Adaptive-RAG, and clinical QA benchmarks (emrQA, EHRNoteQA) into a coherent gap analysis: no prior work benchmarks adaptive gating on hospital-critical medical QA under CPU-only constraints. |
| **Specification & Design** | 15% | Compares six retrieval policies and three gating mechanisms across three hardware tiers. Designs a novel "safety envelope" concept mapping question risk-level to minimum retrieval requirements. |
| **Implementation** | 25% | Builds a complete benchmark harness: data loading, policy engine, gating module, inference pipeline, energy logging, and evaluation scripts. All runnable offline on CPU with deterministic seeds. |
| **Evaluation** | 20% | Multi-dimensional evaluation: accuracy, citation compliance, p95 latency, energy per query, calibration error (ECE). Produces trade-off Pareto curves and per-tier recommendation tables. Compares theory (gating should save resources) vs. practice (where does it fail in medical contexts?). |
| **Legal/Social/Ethical** | 10% | Privacy (offline-only, no data leaves device), medical AI safety (hallucination risk when skipping retrieval), NHS AI governance frameworks, GDPR compliance for clinical data, environmental sustainability (energy tracking). |
| **General Scholarship** | 10% | Clear structure, reproducible scripts, consistent citation format, critical reflection on limitations. |

### 1.3 What Makes This Hospital-Critical (Not Just Clinic Exam Questions)

The standard MIRAGE benchmark uses five medical QA datasets that are largely **exam-style** (MMLU-Med, MedQA-US, MedMCQA) or **research-oriented** (PubMedQA, BioASQ). These test textbook knowledge recall.

This project extends the evaluation to **hospital-critical scenarios** by:

1. **Risk-stratifying MIRAGE questions**: Tagging each question by clinical urgency (routine knowledge vs. acute decision-making vs. life-critical drug/dosing).
2. **Adding a clinical guidelines corpus**: Supplementing MedRAG's corpora with open-access emergency protocols, BNF-style drug interaction references, and WHO acute care guidelines.
3. **Adding hospital-oriented QA subsets**: Incorporating questions from EHR-DS-QA (discharge summary QA), RAGCare-QA (cardiology, neurology, oncology with complexity tiers), and constructing a small synthetic set of acute-care scenarios (drug interactions, contraindications, emergency triage decisions).
4. **Measuring "safe-to-skip" vs. "must-retrieve" boundaries**: The key research question becomes — for which medical question types is it *safe* to rely on the model's internal knowledge, and for which must the system *always* retrieve and cite?

---

## 2. Research Questions

**RQ1 (Primary):** How do different retrieval policies (always-retrieve, closed-book, hybrid, selective/gated, user-toggle) compare on accuracy, latency, energy, and citation compliance for hospital-critical medical QA on CPU-only hardware?

**RQ2:** Can training-free gating signals (token entropy, logit margin, verbalized confidence) reliably identify medical questions where retrieval is necessary vs. safely skippable?

**RQ3:** How do these trade-offs shift across hardware tiers (Low: 4GB RAM laptop / Medium: 8GB / Higher: 16GB+)?

**RQ4:** What are the "safety envelopes" — the minimum retrieval requirements per medical question risk-level that maintain clinically acceptable accuracy?

---

## 3. Six Retrieval Policies Under Test

| # | Policy | Description | Expected Strength | Expected Weakness |
|---|--------|-------------|-------------------|-------------------|
| P1 | **Always-Retrieve** | Every query triggers full retrieval pipeline (keyword + vector). Baseline. | Highest accuracy, full provenance | Highest latency and energy cost |
| P2 | **Always-Retrieve + Citation** | P1 plus mandatory source citation in every answer. Provenance baseline. | Auditability | Even higher latency (citation extraction overhead) |
| P3 | **Closed-Book** | No retrieval; model answers from internal parametric knowledge only. | Fastest, lowest energy | Highest hallucination risk; unacceptable for critical medical queries |
| P4 | **Hybrid Retrieval** | Routes to BM25 (keyword) for factual lookup, vector for conceptual queries, based on query-type classifier. | Balanced | Routing errors; may miss when vector is needed |
| P5 | **Selective/Gated Retrieval** | Training-free gate (entropy/margin/confidence) decides per-query whether to retrieve. **Core novelty.** | Best accuracy-efficiency trade-off | Gate calibration critical; medical domain may require conservative thresholds |
| P6 | **User-Choice Toggle** | User selects "Fast Answer" (closed-book) vs. "Sourced Answer" (full retrieval + citation). | User agency; good for training contexts | Requires UI; user may choose poorly in critical situations |

---

## 4. Three Gating Mechanisms (for Policy P5)

These are the **training-free adaptive signals** that decide whether to retrieve for a given query:

### 4.1 Token Entropy Gate
- Generate a short draft answer (32-64 tokens) without retrieval.
- Compute mean token entropy from the draft's logits.
- If entropy > threshold → the model is uncertain → trigger retrieval.
- **Basis:** TARG (Training-free Adaptive Retrieval Gating, 2025).

### 4.2 Logit Margin Gate
- From the draft's prefix logits, compute the gap between top-1 and top-2 token probabilities.
- Small margin → model is indecisive → trigger retrieval.
- **Basis:** TARG margin signal; shown to be robust as models become instruction-tuned.

### 4.3 Verbalized Confidence Gate
- Prompt the model: "How confident are you in answering this question without external sources? Reply HIGH/MEDIUM/LOW."
- If MEDIUM or LOW → trigger retrieval.
- **Basis:** Kadavath et al. 2022 (Language Models Know What They Don't Know).

**Key hypothesis for medical domain:** In hospital-critical contexts, the entropy and margin gates will need **more conservative thresholds** (i.e., retrieve more often) than in general QA, because the cost of a wrong answer is asymmetric — a missed drug interaction is far worse than unnecessary retrieval latency.

---

## 5. Datasets & Corpora

### 5.1 Evaluation Question Sets

| Dataset | Size | Type | Hospital Relevance | Access |
|---------|------|------|-------------------|--------|
| **MIRAGE Benchmark** (Xiong et al. 2024) | 7,663 Qs | Multi-choice medical QA across 5 datasets | Foundation benchmark — includes MMLU-Med (clinical knowledge, anatomy, pharmacology), MedQA-US (USMLE-style), MedMCQA (Indian medical exams), PubMedQA, BioASQ | Open — GitHub + Google Drive |
| **RAGCare-QA** (2025) | 420 Qs | Medical QA across 6 specialties (Cardiology, Neurology, Oncology, Gastroenterology, Endocrinology, Family Medicine) with 3 complexity tiers (Basic/Intermediate/Advanced) | Directly maps to hospital department specialities; complexity tiers parallel question risk levels | Open — HuggingFace |
| **EHR-DS-QA** (Kotschenreuther 2024) | Synthetic QA from discharge summaries | QA pairs from MIMIC-IV discharge notes | Hospital discharge — diagnosis, medications, post-discharge instructions | PhysioNet (requires credentialing) |
| **Synthetic Acute-Care Set** (self-constructed) | ~100-150 Qs | Drug interactions, contraindications, emergency triage, dosing decisions | **Highest hospital criticality** — constructed from BNF, WHO mhGAP, open formularies | Self-built from open-access sources |

**Total evaluation corpus: ~8,300+ questions** spanning from routine medical knowledge to hospital-critical acute care.

### 5.2 Retrieval Corpora (via MedRAG toolkit)

| Corpus | Description | Size | Best For |
|--------|-------------|------|----------|
| **PubMed** | Biomedical abstracts | ~24M snippets | Research-oriented questions (PubMedQA, BioASQ) |
| **StatPearls** | Clinical decision support articles | ~301K snippets | Clinical knowledge, treatment protocols |
| **Medical Textbooks** | Domain-specific knowledge | ~126K snippets | Foundational medical knowledge |
| **Wikipedia** | General knowledge | ~34M snippets | General context, broad medical topics |
| **MedCorp** (combined) | All four above via RRF fusion | ~58M snippets | Best overall performance per MIRAGE results |

**Hospital-critical extension:** Supplement with:
- **BNF Open Data** (drug interactions, contraindications) — stored as local text chunks.
- **WHO mhGAP Intervention Guide** (mental health acute care) — PDF parsed to text.
- **NICE Clinical Guidelines** (selected acute care pathways) — open-access PDFs.

### 5.3 Risk Stratification of Questions

Each question in the evaluation set is tagged with a **clinical risk level**:

| Risk Level | Description | Example | Retrieval Expectation |
|-----------|-------------|---------|----------------------|
| **Low** | Textbook recall, anatomy, basic science | "Which nerve innervates the diaphragm?" | Model may answer correctly from parametric knowledge |
| **Medium** | Clinical reasoning, differential diagnosis | "Patient presents with chest pain and ST elevation — most likely diagnosis?" | Retrieval recommended but model may have strong internal knowledge |
| **High** | Drug interactions, dosing, contraindications, emergency protocols | "Is methotrexate contraindicated with trimethoprim?" | **Must retrieve.** Hallucination here is clinically dangerous. |

This stratification enables the "safety envelope" analysis: at each risk level, what is the minimum retrieval policy that maintains acceptable accuracy?

---

## 6. Tech Stack

### 6.1 Core Infrastructure

| Component | Tool | Purpose |
|-----------|------|---------|
| **Language** | Python 3.10+ | Primary development language |
| **LLM Inference** | `llama-cpp-python` (llama.cpp bindings) | CPU-only inference of GGUF quantized models |
| **Models** | Llama 3.2-3B (Q4_K_M), Qwen2.5-3B (Q4_K_M), Phi-3.5-mini-3.8B (Q4_K_M) | Small models that fit in 4-8GB RAM |
| **Baseline (closed-source)** | GPT-4o-mini via API (for baseline comparison only, not the constrained system) | Upper bound reference — *not* the deployed system |
| **Quantization** | GGUF Q4_K_M / Q5_K_M formats | 4-bit and 5-bit quantization for memory efficiency |

### 6.2 Retrieval Stack

| Component | Tool | Purpose |
|-----------|------|---------|
| **Keyword Retrieval** | `rank_bm25` or `Whoosh` | BM25 term-based retrieval (no neural components) |
| **Vector Retrieval** | FAISS (CPU mode) with `sentence-transformers` | Dense retrieval with compact embeddings |
| **Embedding Model** | `all-MiniLM-L6-v2` (22M params, 384-dim) or `MedCPT` (domain-specific) | Lightweight embeddings suitable for CPU |
| **Hybrid Fusion** | Reciprocal Rank Fusion (RRF) | Combines BM25 + vector scores |
| **Corpus Management** | MedRAG toolkit | Pre-built corpus access and retrieval pipeline |

### 6.3 Evaluation & Benchmarking

| Component | Tool | Purpose |
|-----------|------|---------|
| **Energy Measurement** | `codecarbon` + Intel RAPL (where available) | Track energy consumption per query |
| **Latency Profiling** | Python `time.perf_counter_ns()` + custom p95 calculation | Measure end-to-end and per-component latency |
| **Memory Profiling** | `psutil` + `tracemalloc` | Track RAM usage across hardware tiers |
| **Accuracy Metrics** | Exact Match (EM), F1, Accuracy (multi-choice) | Standard QA evaluation |
| **Calibration Metrics** | Expected Calibration Error (ECE) | Measure how well confidence scores predict correctness |
| **Citation Metrics** | Citation Precision, Citation Recall (ALCE framework) | Measure provenance quality |
| **Experiment Tracking** | JSON logs + pandas DataFrames | Reproducible experiment records |
| **Visualization** | `matplotlib` + `seaborn` | Trade-off curves, Pareto fronts, heatmaps |

### 6.4 Interface & Integration

| Component | Tool | Purpose |
|-----------|------|---------|
| **UI** | Gradio (lightweight) | "Fast Answer" vs. "Sourced Answer" toggle for P6 evaluation |
| **Config Management** | YAML config files | Hardware profiles, policy parameters, thresholds |
| **qVault Integration** | Shared JSON schema (Variant 1 anchors, Variant 2 context payload, Variant 4 citation contract) | Ensures composability with other qVault variants |
| **Version Control** | Git | All code, configs, and experiment logs |

### 6.5 Hardware Tiers for Testing

| Tier | Spec | Representative Device | Model Fit |
|------|------|-----------------------|-----------|
| **Low** | 4GB RAM, dual-core CPU | Chromebook / old laptop | Llama 3.2-1B (Q4), BM25-only retrieval |
| **Medium** | 8GB RAM, quad-core CPU | Standard student laptop | Llama 3.2-3B (Q4_K_M), hybrid retrieval |
| **Higher** | 16GB+ RAM, 6+ core CPU | Developer workstation | Phi-3.5-mini (Q5_K_M), full MedCorp + RRF |

---

## 7. Research Papers (Categorised)

### 7.1 Adaptive Retrieval & Gating (Core Methodology)

| # | Paper | Venue | Key Contribution | Relevance |
|---|-------|-------|-----------------|-----------|
| 1 | Asai et al. "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection" | ICLR 2024 (Oral) | Reflection tokens for adaptive retrieval; retrieval on-demand | Foundational framework for adaptive retrieval gating |
| 2 | "TARG: Training-free Adaptive Retrieval Gating" | arXiv 2025 (2511.09803) | Entropy/margin/variance signals for zero-cost retrieval decision | **Direct methodological basis** for gating mechanisms P5 |
| 3 | Jeong et al. "Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity" | NAACL 2024 | Routes queries by complexity to different RAG strategies | Supports hybrid routing policy P4 |
| 4 | SmartRAG (ICLR 2025) | ICLR 2025 | RL-trained policy for when to retrieve; shows retrieval % reduction | Evidence that selective retrieval matches always-retrieve accuracy |
| 5 | Mallen et al. "When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories" | ACL 2023 | Shows LLMs struggle with rare/long-tail knowledge; retrieval helps most there | Justifies gating: skip retrieval for common knowledge, retrieve for rare facts |

### 7.2 Medical RAG & Benchmarking (Domain Foundation)

| # | Paper | Venue | Key Contribution | Relevance |
|---|-------|-------|-----------------|-----------|
| 6 | Xiong et al. "Benchmarking Retrieval-Augmented Generation for Medicine" (MIRAGE) | ACL Findings 2024 | 7,663-question benchmark; MedRAG toolkit; log-linear scaling; lost-in-the-middle effect | **Primary benchmark and toolkit** for this project |
| 7 | Xiong et al. "Improving Retrieval-Augmented Generation in Medicine with Iterative Follow-up Questions" (i-MedRAG) | PSB 2025 | Iterative retrieval for complex medical queries | Informs multi-step retrieval strategies |
| 8 | Singhal et al. "Large Language Models Encode Clinical Knowledge" (Med-PaLM) | Nature 2023 | First LLM evaluation on medical QA benchmarks | Baseline for model capabilities in medical domain |
| 9 | Jin et al. "MedQA: A Large-scale Open-domain Question Answering Dataset from Medical Exams" | Applied Sciences 2021 | USMLE-style QA dataset used in MIRAGE | Core evaluation data |
| 10 | "RAGCare-QA: A benchmark dataset for evaluating RAG pipelines in theoretical medical knowledge" | ScienceDirect 2025 | 420 Qs across 6 specialties with complexity tiers and RAG-type annotation | **Hospital-department-oriented** evaluation data |

### 7.3 Clinical QA & Hospital-Critical Datasets

| # | Paper | Venue | Key Contribution | Relevance |
|---|-------|-------|-----------------|-----------|
| 11 | Pampari et al. "emrQA: A Large Corpus for Question Answering on Electronic Medical Records" | EMNLP 2018 | 400K+ QA pairs from clinical notes (i2b2) | Established clinical note QA methodology |
| 12 | Kweon et al. "EHRNoteQA: An LLM Benchmark for Real-World Clinical Practice Using Discharge Summaries" | NeurIPS 2024 D&B | Clinician-verified QA from MIMIC-IV discharge summaries; multi-note reasoning | Hospital-grounded QA that goes beyond exam questions |
| 13 | Kotschenreuther "EHR-DS-QA: Synthetic QA from Medical Discharge Summaries" | PhysioNet 2024 | Synthetic QA for discharge diagnosis, medications, post-discharge instructions | Directly hospital-critical: discharge decision support |
| 14 | Jin et al. "Retrieval-Augmented Generation for AI-Generated Content: A Survey" | npj Digital Medicine 2024 | Context window sizing effects; BriefContext map-reduce for long documents | Informs context window sizing policy experiments |

### 7.4 Energy, Efficiency & Constrained Deployment

| # | Paper | Venue | Key Contribution | Relevance |
|---|-------|-------|-----------------|-----------|
| 15 | Wilkins et al. "Offline Energy-Optimal LLM Serving" | HotCarbon 2024 | Energy profiling of LLM inference; scheduling for energy budgets | **Methodological basis** for energy-aware policy P3/P5 |
| 16 | Strubell et al. "Energy and Policy Considerations for Deep Learning in NLP" | ACL 2019 | Seminal work on computational cost of NLP; CO2 equivalence reporting | Framework for eco-benchmarking in report |
| 17 | Qin et al. "Edge RAG: Online-Indexed Edge Retrieval-Augmented Generation" | arXiv 2024 (2405.04700) | RAG deployment on edge/resource-constrained devices | Architectural inspiration for offline deployment |

### 7.5 Confidence Calibration & Safety

| # | Paper | Venue | Key Contribution | Relevance |
|---|-------|-------|-----------------|-----------|
| 18 | Kadavath et al. "Language Models (Mostly) Know What They Don't Know" | arXiv 2022 | LLMs can self-assess confidence; verbalized confidence as a signal | Basis for verbalized confidence gate |
| 19 | Guo et al. "On Calibration of Modern Neural Networks" | ICML 2017 | Expected Calibration Error (ECE) metric; temperature scaling | Calibration evaluation methodology |
| 20 | Gao et al. "ALCE: Enabling Attribution in Language Models" | EMNLP 2023 | Citation precision/recall metrics for attributed generation | Citation quality evaluation framework |

### 7.6 Surveys & Background

| # | Paper | Venue | Key Contribution | Relevance |
|---|-------|-------|-----------------|-----------|
| 21 | Fan et al. "A Survey on RAG Meeting LLMs" | KDD 2024 | Comprehensive RAG taxonomy: naive, advanced, modular | Positions project within RAG landscape |
| 22 | "Retrieval-Augmented Generation in Biomedicine: A Comprehensive Review" | arXiv 2025 | Biomedical RAG survey covering KG-RAG, multimodal, evaluation | Domain-specific survey for literature review |
| 23 | "Privacy Challenges and Solutions in RAG-Enhanced Healthcare" | arXiv 2025 (2511.11347) | Privacy risks in medical RAG; data localisation strategies | Directly supports Legal/Social/Ethical section |
| 24 | Robertson & Zaragoza "The Probabilistic Relevance Framework: BM25 and Beyond" | Foundations and Trends in IR 2009 | BM25 algorithm specification | Technical foundation for keyword retrieval component |

---

## 8. Bi-Weekly Timeline (16 Weeks / ~4 Months)

> **Assumption:** Project runs approximately mid-February to mid-June 2026, with report submission and presentation in June/July.

---

### Sprint 1: Weeks 1-2 (Mid Feb — End Feb)
**Theme: Foundation & Literature**

| Task | Deliverable | Rubric Target |
|------|------------|---------------|
| Deep-read papers #1-8, #15-16, #21-23 | Annotated bibliography (25+ sources) | Literature Review (10%) |
| Clone MedRAG repo; download MIRAGE benchmark.json | Local copy of 7,663 questions | — |
| Set up Python environment: `llama-cpp-python`, `rank_bm25`, `faiss-cpu`, `sentence-transformers`, `codecarbon` | Working `requirements.txt` + install script | Implementation (25%) |
| Download first GGUF model: Llama 3.2-3B Q4_K_M | Verified CPU inference (test prompt → response) | — |
| Draft Introduction chapter outline | 1-page outline with context positioning | Introduction (10%) |

**Milestone:** Can run a single question through closed-book LLM inference on CPU.

---

### Sprint 2: Weeks 3-4 (Early Mar — Mid Mar)
**Theme: Baseline Policies + Data Preparation**

| Task | Deliverable | Rubric Target |
|------|------------|---------------|
| Implement Policy P1 (Always-Retrieve) using MedRAG toolkit | Working retrieval → LLM pipeline | Implementation (25%) |
| Implement Policy P3 (Closed-Book) with deterministic seeds | Baseline closed-book inference | Implementation (25%) |
| Download RAGCare-QA from HuggingFace; parse into unified format | Unified JSON question format across datasets | Implementation (25%) |
| Risk-stratify MIRAGE questions (Low/Medium/High) using keyword heuristics + manual spot-check | Tagged question set with risk labels | Spec & Design (15%) |
| Begin constructing synthetic acute-care question set from BNF/WHO open sources | Draft of ~50 acute-care questions | Spec & Design (15%) |
| Set up hardware profiling: `codecarbon`, `psutil`, `tracemalloc` wrappers | Energy/memory logging per query | Implementation (25%) |

**Milestone:** Can run P1 and P3 on 100 MIRAGE questions and log accuracy + latency + energy.

---

### Sprint 3: Weeks 5-6 (Mid Mar — End Mar)
**Theme: Retrieval Engine + Hybrid Policy**

| Task | Deliverable | Rubric Target |
|------|------------|---------------|
| Implement BM25 retrieval index over StatPearls + Textbooks corpus | Working keyword retrieval returning top-k chunks with anchors | Implementation (25%) |
| Implement FAISS vector index with `all-MiniLM-L6-v2` embeddings | Working dense retrieval | Implementation (25%) |
| Implement RRF hybrid fusion (P4: Hybrid Retrieval) | Combined retrieval with query-type routing | Implementation (25%) |
| Implement Policy P2 (Always-Retrieve + Citation extraction) | Citation extraction from model outputs | Implementation (25%) |
| Complete synthetic acute-care question set (~100-150 Qs) | Finalised acute-care evaluation set | Spec & Design (15%) |
| Draft Specification & Design chapter: architecture diagram, policy comparison rationale, hardware tier definitions | 3-4 page design chapter draft | Spec & Design (15%) |

**Milestone:** All four non-gated policies (P1-P4) runnable. Full retrieval pipeline operational.

---

### Sprint 4: Weeks 7-8 (Early Apr — Mid Apr)
**Theme: Gating Mechanisms (Core Novelty)**

| Task | Deliverable | Rubric Target |
|------|------------|---------------|
| Implement Token Entropy Gate: draft generation → entropy computation → threshold decision | Working entropy gate module | Implementation (25%) |
| Implement Logit Margin Gate: prefix logits → top-1/top-2 gap → threshold decision | Working margin gate module | Implementation (25%) |
| Implement Verbalized Confidence Gate: confidence prompt → parse HIGH/MEDIUM/LOW → decision | Working confidence gate module | Implementation (25%) |
| Combine into Policy P5 (Selective/Gated Retrieval) with configurable gate type and threshold | Unified gating policy with YAML config | Implementation (25%) |
| Run threshold sweep experiments: vary gate threshold on 500-question subset, plot accuracy vs. retrieval % | Threshold calibration curves | Evaluation (20%) |
| Read papers #18-19 (calibration); compute ECE for each gate type | Calibration analysis | Evaluation (20%) |

**Milestone:** Core novelty implemented. Can demonstrate that gating reduces retrieval by 40-70% while maintaining accuracy on general medical questions.

---

### Sprint 5: Weeks 9-10 (Mid Apr — End Apr)
**Theme: Full-Scale Experiments + Medical Risk Analysis**

| Task | Deliverable | Rubric Target |
|------|------------|---------------|
| Run all 6 policies on full MIRAGE benchmark (7,663 Qs) across all 3 hardware tiers | Raw results: accuracy, latency, energy, citation compliance per {policy × tier × dataset} | Evaluation (20%) |
| Run all 6 policies on RAGCare-QA (420 Qs) stratified by specialty and complexity | Per-specialty results | Evaluation (20%) |
| Run all 6 policies on synthetic acute-care set (100-150 Qs) | Hospital-critical results | Evaluation (20%) |
| Compute "safety envelopes": at each risk level (Low/Medium/High), what is the minimum retrieval policy that maintains ≥X% accuracy? | Safety envelope table and visualisation | Evaluation (20%) — **key novel contribution** |
| Implement Policy P6 (User-Choice Toggle) in Gradio UI | Working "Fast vs. Sourced" interface | Implementation (25%) |
| Begin second model evaluation: repeat key experiments with Qwen2.5-3B Q4_K_M | Cross-model comparison | Evaluation (20%) |

**Milestone:** Complete experimental results across all policies, datasets, and tiers.

---

### Sprint 6: Weeks 11-12 (Early May — Mid May)
**Theme: Analysis, Visualisation & User Study**

| Task | Deliverable | Rubric Target |
|------|------------|---------------|
| Generate Pareto front plots: accuracy vs. latency, accuracy vs. energy, for each tier | Publication-quality figures | Evaluation (20%) |
| Generate per-tier recommendation tables: "At Low tier, use P5-margin with threshold 0.3; at Higher tier, use P4-hybrid" | Tiered recommendation table | Evaluation (20%) |
| Analyse where gating **fails** on medical questions: identify question types where gate incorrectly skips retrieval | Failure case analysis with examples | Evaluation (20%) |
| Small user study (5-10 participants): P6 toggle interface. Measure: perceived helpfulness, trust, time-to-answer, "regret" (when fast answer was wrong) | User study results + qualitative feedback | Evaluation (20%) |
| Draft Evaluation chapter with all results, tables, figures | 8-10 page evaluation chapter | Evaluation (20%) |
| Draft Legal/Social/Ethical chapter: GDPR, NHS AI governance, medical AI safety, environmental impact, accessibility | 3-4 page LSEP chapter | Legal/Social/Ethical (10%) |

**Milestone:** All analysis complete. Report writing in progress.

---

### Sprint 7: Weeks 13-14 (Mid May — End May)
**Theme: Report Writing & Refinement**

| Task | Deliverable | Rubric Target |
|------|------------|---------------|
| Write complete Literature Review chapter | 5-6 pages, 25+ references | Literature Review (10%) |
| Write complete Implementation chapter: architecture, code walkthrough, testing approach | 8-10 pages | Implementation (25%) |
| Finalise Introduction with full context positioning | 2-3 pages | Introduction (10%) |
| Write Conclusion: key findings, limitations, future work (NHS pilot study, multi-model ensemble, fine-tuned gates) | 2-3 pages | General Scholarship (10%) |
| Package reproducibility artifacts: `run_all_experiments.sh`, `requirements.txt`, YAML configs, README | Reproducible codebase | General Scholarship (10%) |
| First complete draft of full report | ~40-50 page report | All sections |

**Milestone:** Complete first draft ready for review.

---

### Sprint 8: Weeks 15-16 (Early Jun — Mid Jun)
**Theme: Polish, Presentation & Submission**

| Task | Deliverable | Rubric Target |
|------|------------|---------------|
| Supervisor feedback incorporation | Revised report | All sections |
| Prepare live demonstration: run P5 gating on 3 medical questions showing gate decision in real-time | Demo script with pre-selected scenarios | Presentation |
| Prepare presentation slides (15-20 slides): problem → method → results → demo → conclusions | Slide deck | Presentation |
| Rehearse presentation; prepare for Q&A on: "Why not fine-tune the gate?", "How would this work with real patient data?", "What about multi-modal inputs?" | Q&A preparation notes | Presentation |
| Final report polish: formatting, bibliography check, figure quality, abstract | Submission-ready report | General Scholarship (10%) |
| Export audit logs and experiment records as JSON/CSV | Compliance artifacts for qVault | Implementation (25%) |

**Milestone:** Report submitted. Presentation delivered.

---

## 9. Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| GGUF model too slow on Low-tier hardware | Medium | High | Fall back to Llama 3.2-1B (Q4); reduce context window; report the failure as a finding |
| MedRAG corpus too large for local storage | Medium | Medium | Use StatPearls + Textbooks only (~427K snippets); document the trade-off |
| Gating thresholds don't transfer across medical question types | High | Medium | **This is a finding, not a failure.** Report per-question-type optimal thresholds |
| PhysioNet credentialing takes too long (EHR-DS-QA) | Medium | Low | Proceed with MIRAGE + RAGCare-QA + synthetic set; add EHR data if approved in time |
| Energy measurement unreliable on macOS (no RAPL) | Medium | Low | Use `codecarbon` software estimates; document the limitation; test on Linux VM if possible |
| Insufficient time for user study | Low | Medium | Reduce to 3-5 participants; focus on qualitative insights rather than statistical significance |

---

## 10. Key Outputs (Mapping to Variant 5 Deliverables)

| Variant 5 Required Output | This Project's Deliverable |
|---|---|
| Reproducible benchmark harness | Python package with `run_all_experiments.sh`, YAML configs, deterministic seeds |
| Trade-off curves | Pareto fronts: accuracy vs. latency, accuracy vs. energy, per hardware tier |
| Operating envelopes | Per-tier, per-risk-level "safety envelope" tables |
| Policy recommendations | Tiered recommendation table mapping {hardware × risk-level → best policy} |
| Audit logs | JSON/CSV logs of every query: policy used, retrieval decision, latency, energy, accuracy |
| qVault integration | Shared schema compliance: Variant 1 anchors, Variant 2 context payload, Variant 4 citation contract, Variant 5 fields |

---

## 11. How This Targets Distinction (80-100)

1. **Novel contribution:** First systematic benchmark of adaptive retrieval gating on hospital-critical medical QA under CPU-only constraints. The "safety envelope" concept — mapping question risk to minimum retrieval — is a publishable finding.

2. **Systematic evaluation:** Six policies × three gates × three tiers × three datasets = comprehensive multi-dimensional analysis that goes far beyond a single-machine snapshot.

3. **Real-world relevance:** Directly addresses NHS training, LMIC deployment, and the qVault offline appliance vision. The Legal/Social/Ethical section writes itself: privacy, medical AI safety, energy sustainability.

4. **Technical depth:** Combines information retrieval (BM25, FAISS, RRF), NLP (prompt engineering, confidence calibration), systems engineering (energy profiling, hardware-aware configs), and HCI (user-toggle study).

5. **Critical reflection:** The project is designed to find *where gating fails* in medical contexts, not just where it succeeds. This honest evaluation of limitations is what distinguishes Distinction from Merit.
