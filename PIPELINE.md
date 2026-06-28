# End-to-End Run Pipeline

How one experiment run flows, from raw data to a scored JSONL record. Reflects
the code as built on branch `feat/p5-gated-prototype` (2026-06-26).

Legend: ✅ built · ⬜ planned/not built.

---

## 0. Stages at a glance

```
RAW DATA → LOAD/NORMALISE → RISK-TAG → [per question:] POLICY → (GATE?) → (RETRIEVE?) → ANSWER → SCORE → LOG
                                                         └─ P5 only ─┘   └ P1/P5-retrieve ┘
```

One run = one `(policy, dataset)` pair, executed by `scripts/run_experiment.py`.

---

## 1. Corpus side — building the retrieval index (offline, once)

This is separate from the question side. It prepares what retrieval searches over.

```
data/corpora/pilot_corpus.jsonl        ✅  (200 curated medical chunks)
        │   scripts/build_pilot_corpus.py wrote these
        ▼
scripts/build_indexes.py
        │   loads each line → Chunk(chunk_id, source, title, text, score)
        │   BM25Retriever.from_chunks(chunks):
        │       tokenise "title + text" per chunk (regex [a-z0-9]+, lowercased)
        │       build rank_bm25.BM25Okapi over the token lists
        ▼
indexes/bm25_pilot.pkl                 ✅  (pickled: {chunks, bm25})  — in-memory, no DB
```

### Chunking note
There is **no runtime document chunker yet**. The pilot corpus is *pre-chunked*
(each JSONL line is already one passage). The real MedRAG/StatPearls chunker
(256-token windows) is ⬜ Phase D, alongside the FAISS vector index.

### What does NOT exist yet
- ⬜ `vector_retriever.py` — FAISS + sentence-transformer embeddings (semantic search)
- ⬜ `hybrid_retriever.py` — RRF fusion of BM25 + FAISS
- ⬜ real ~427K MedCorp chunking pipeline

So today retrieval = **BM25 lexical only**, over the 200-chunk pilot corpus.

---

## 2. Question side — loading and normalising (once per run)

`run_experiment.py → load_questions()` dispatches by dataset name:

```
data/raw/mirage/benchmark.json         ✅
        │   load_mirage(path, max_questions=200)
        │     subset-keyed JSON {mmlu, medqa, medmcqa, pubmedqa, bioasq}
        │     each record → UnifiedQuestion(
        │        question_id, question_text, correct_answer,
        │        dataset_source="mirage_<subset>", choices={A:..,B:..}, ...)
        ▼
   tag_risk(questions)                  ✅
        │   risk_tagger.classify_risk(text, specialty):
        │     HIGH  : dose/overdose/interaction/contraindication/emergency...
        │     LOW   : anatomy/physiology/definition/mechanism...
        │     MEDIUM: diagnosis/treatment/management  (default)
        ▼
   cap to max_questions (200)
        ▼
   List[UnifiedQuestion]   (each has risk_level set)
```

RAGCare-QA loader (`load_ragcare`) ✅ exists for open-ended questions (choices=None,
gold answer string, scored by token-F1) but the runs so far use MIRAGE (MCQ).

The runner then skips any `question_id` already in the output JSONL (**resume**),
and loops over the rest.

---

## 3. Per-question flow — depends on the POLICY

`policy = factory.build_policy(cfg, llm, retriever)` picks the class from
`cfg.policy.name`. Each policy's `answer(question)` does:

### P3 — Closed-Book ✅  (no retrieval, no gate)
```
build_closed_book_prompt(question, choices)
        ▼
llm.answer(prompt)                      # 1 LLM call
        ▼
PolicyResult(retrieval_triggered=False, retrieved_chunks=[])
```

### P1 — Always-Retrieve ✅  (retrieval every query, no gate)
```
chunks = retriever.retrieve(question_text, top_k=5)     # BM25 lexical
        │   get BM25 scores for all 200 chunks, take top-5
        ▼
build_rag_prompt(question, chunks, choices)
        │   stuffs "[SOURCE] title\n text" of each chunk into context block
        ▼
llm.answer(prompt)                      # 1 LLM call
        ▼
PolicyResult(retrieval_triggered=True, retrieved_chunks=chunks)
```

### P5 — Gated ✅  (gate decides retrieve vs closed-book)
```
decision = gate.decide(question, llm)   # see §4 — costs extra LLM calls
        │
        ├── decision.retrieve == True  → behave like P1
        │       chunks = retriever.retrieve(question_text)   # BM25
        │       build_rag_prompt(...) ; llm.answer(...)
        │       retrieval_triggered=True
        │
        └── decision.retrieve == False → behave like P3
                build_closed_book_prompt(...) ; llm.answer(...)
                retrieval_triggered=False, chunks=[]
        ▼
PolicyResult( + gate_name, gate_decision, gate_signal_value,
              gate_details = decision.details )
```

### Policy → retrieval method map
| Policy | Retrieves? | Method | Built |
|---|---|---|---|
| P1 | every query | BM25 lexical (config says hybrid, but vector leg missing) | ✅ |
| P3 | never | — | ✅ |
| P5 | only when gate says RETRIEVE | BM25 lexical (same retriever as P1) | ✅ |
| P2 | every query + citations | BM25 + citation extraction | ⬜ |
| P4 | every query, routed | BM25↔vector + RRF | ⬜ |
| P6 | user toggle | none / retrieve | ⬜ |

**Today every retrieving policy uses BM25.** Semantic/vector retrieval is not
wired, so `retrieval_mode: hybrid` in the YAMLs currently resolves to BM25.

---

## 4. The gate — only triggered by P5

`cfg.gate.type` (after the factory's hardware check) selects which gate runs.
On the medium tier with logits enabled, it's the **ensemble**:

```
EnsembleGate.decide(question, llm):
    for each member in [EntropyGate, MarginGate, HallucinationProbeGate]:
        member.decide(question, llm)
    majority vote (retrieve if >= 2 of available members vote RETRIEVE)
```

### Gate 1 — EntropyGate ✅
```
prompt = build_draft_prompt(question, choices)
text, logits = llm.draft(prompt, max_tokens=48)      # 1 LLM call, needs logits_all=True
    softmax(logits) per token → H(p_t) = -Σ p log p
    mean_H = mean over tokens
    retrieve = mean_H > τ_H (2.5)
    details["per_token_entropy"] = [...]   # ← interpretability output
    if logits is None (low tier): abstain (available=False)
```

### Gate 2 — MarginGate ✅
```
text, top_logprobs = llm.get_top2_logprobs(prompt, 48)   # 1 LLM call (logprobs=2)
    per token: margin = p_top1 - p_top2
    mean_M = mean
    retrieve = mean_M < τ_M (0.3)
```

### Gate 3 — HallucinationProbeGate ✅  (text-only; the only low-tier gate)
```
draft_a = llm.draft(build_probe_prompt_a(question))      # 1 LLM call
draft_b = llm.draft(build_probe_prompt_b(question))      # 1 LLM call
    MCQ : extract_letter(a) vs extract_letter(b) → differ ⇒ retrieve
    open: token_f1(a, b) < 0.7 ⇒ retrieve
```

### Ensemble vote ✅
```
count RETRIEVE votes among members with available=True
    normal : retrieve = votes >= min_votes (2)
    degraded (fewer available than min_votes, e.g. low tier probe-only):
             retrieve = votes >= 1     ; details["degraded"]=True
```

### Gate cost
P5 issues **~4 LLM calls/query** (entropy/margin draft + 2 probe drafts +
final answer) vs P3's 1 — the source of P5's latency overhead (143.7s vs 7.42s).

### Hardware-aware gate selection (factory)
```
if cfg.model.logits_all == False (low tier):
        gate = HallucinationProbeGate     # entropy+margin need logits → dropped
else:
        gate = EnsembleGate([entropy, margin, probe], min_votes=2)
```

---

## 5. Answering

Whichever branch ran, the final answer is one call:
```
LlamaBackend.answer(prompt)
    create_completion(prompt, temperature=0.0, max_tokens=...)   # greedy, seed=42
    → answer text  (e.g. " Answer: A.")
```
Deterministic: temperature 0 + fixed seed ⇒ reproducible.

---

## 6. Profiling (wraps the whole `answer()` call)

```
profile_call(lambda: policy.answer(q)):
    perf_counter_ns       → latency_ns
    tracemalloc           → peak_memory_mb
    codecarbon (optional) → energy_kwh
```
Sequential only — no parallelism (concurrent queries corrupt energy readings).

---

## 7. Scoring + record assembly

`score_and_record(question, result, cfg)`:
```
if MCQ:   is_correct, em = score_mcq(answer_text, correct_letter)
              extract_letter() handles "A", "A.", "(A)", "answer is A", ...
          f1 = em
else:     f1 = token_f1(answer_text, gold) ; is_correct = f1 >= 0.5

RunRecord(
   question_id, dataset_source, risk_level, specialty,
   policy_name, gate_name, gate_decision, gate_signal_value, retrieval_triggered,
   answer_text, correct_answer, is_correct, exact_match, f1_score,
   citations, latency_ns, energy_kwh, peak_memory_mb,
   model_name, hardware_tier, timestamp,
   qvault = { gate_signal_value, gate_details },   # per-token entropy, votes, probe drafts
)
        ▼
append_record(record, output.jsonl)     # one self-describing line per query
```

---

## 8. Aggregate evaluation (post-run, manual today)

Reads the JSONL back and computes:
- accuracy = mean(is_correct)
- retrieval rate = mean(retrieval_triggered)
- latency p50/p95, mean energy, per-risk-level accuracy
- (P5) per-gate vote counts + signal distributions from qvault.gate_details

⬜ Not yet a formal module: `evaluation/harness.py` (orchestration class),
`metrics.py` (ECE, citation P/R), `safety_envelope.py`, `generate_figures.py`.

---

## 9. Concrete trace — one P5 question from the real run

```
Q anatomy-000 (MIRAGE mmlu, risk=medium, MCQ A–D)
 ├ gate.decide:
 │   entropy: draft → mean_H 0.73  (τ 2.5) → SKIP
 │   margin : draft → mean_M 0.74  (τ 0.3) → SKIP
 │   probe  : draft_a, draft_b → same letter → agree → SKIP
 │   votes {entropy:skip, margin:skip, probe:skip} → 0 retrieve → SKIP
 ├ skip branch → build_closed_book_prompt → llm.answer → "Answer: A."
 ├ score_mcq("Answer: A.", "A") → correct
 └ RunRecord(... gate_decision="skip", retrieval_triggered=false, is_correct=true)
```
This pattern held for all 200 — gate skipped every time at default thresholds
(the calibration finding). Retrieval path (BM25) is wired and tested, but the
gate never exercised it on this run.
```
