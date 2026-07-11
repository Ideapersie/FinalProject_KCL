# Real Corpus Build Plan — MedRAG / StatPearls for Adaptive Gating

**Goal:** replace the 200-chunk pilot corpus with a real medical corpus so the
gates retrieve genuine evidence, so P1/P4/P5 accuracy reflects real retrieval
quality (not pilot noise), and so the safety-envelope analysis has meaningful data.

**Why now:** the headline finding — calibrated P5 dropped 62%→45.5% on retrieval —
is a *corpus* artifact, not a gate defect. The pilot is 30 curated passages + 170
filler ("general principles of X"), so retrieval almost always injects noise. Until
the corpus contains the answers, the gate cannot be fairly evaluated. This is the
single critical-path dependency for the whole results chapter.

---

## Hard constraints (measured, not assumed)

| Constraint | Value | Implication |
|---|---|---|
| Total RAM | 16.8 GB | 8 GB tier assumption is conservative; some headroom |
| **Free RAM right now** | **~1.6 GB** | Cannot hold a large FAISS index + model + embedder simultaneously as-is |
| CPU | quad-core, CPU-only | Embedding the full corpus is the time bottleneck |
| FAISS index type | `IndexFlatIP` (exact) | RAM = n_chunks × 384 dims × 4 bytes. Full MedCorp (~427K) ≈ 655 MB just for vectors |
| Embedding model | all-MiniLM-L6-v2 (384-d) | ~14 MB model; ~well under 1 GB working set |
| llama-3.2-3B Q4 | ~2.3 GB resident | Runs concurrently with retrieval during eval |

**Key RAM math:** full MedCorp FAISS (655 MB) + 3B model (2.3 GB) + Python/embedder
overhead (~1 GB) ≈ **4 GB working set**. Fits in 16.8 GB total, but NOT in the
current 1.6 GB free — need to close other apps, or the index must be built in a
separate process from the eval run (it already is).

---

## Corpus options (pick scale to fit CPU/RAM budget)

### ⚠️ HF path verification (done 2026-07-02) — StatPearls is DEAD on HF
Probed each MedRAG sub-corpus on the Hub before committing (lesson from RAGCare):

| HF path | Status | Schema |
|---|---|---|
| `MedRAG/textbooks` | **OK** (~126K) | `id, title, content, contents` |
| `MedRAG/pubmed` | OK but ~23.9M — too big for CPU | `id, title, content, contents, PMID` |
| `MedRAG/statpearls` | **DEAD** — `EmptyDatasetError`, no data files | — |
| `MedRAG/MedCorp`, `MedRAG/statpearls_v2`, `ncbi/statpearls` | do not exist | — |

**Consequence:** the ideal "StatPearls + Textbooks ≈ 427K" is NOT directly
downloadable from HF right now. StatPearls (the best MIRAGE match) needs a
different source. Three ways to recover it:

1. **MedRAG GitHub download script** (canonical). The Teddy-XiongGZ/MIRAGE /
   MedRAG repo fetches StatPearls from NCBI Bookshelf directly and chunks it.
   More setup (NCBI download + their chunker), but gives real StatPearls.
2. **Textbooks-only (~126K)** — works on HF today. Real corpus, smaller.
   FAISS ≈ 194 MB. Safe, immediate fallback.
3. **Textbooks + capped PubMed slice** (e.g. first ~300K of PubMed) — works, but
   PubMed = research abstracts, a weaker match for MIRAGE clinical MCQs than
   StatPearls/textbooks. Use only if more volume is wanted.

### Recommended path given the above
- **If StatPearls via GitHub/NCBI is feasible:** Textbooks (HF) + StatPearls
  (GitHub) ≈ 427K — the original full-MedCorp goal. Best quality.
- **If not (or to move fast tonight):** **Textbooks-only (~126K)** is a real,
  clean, immediately-available corpus and a large step up from the 200-chunk
  pilot. Can add StatPearls later without re-architecting.

**RAM by scale (FAISS IndexFlatIP = n × 384 × 4 bytes):**
- 427K → ~655 MB · 301K → ~460 MB · 126K (textbooks) → ~194 MB

---

## Build steps

### 1. Corpus loader — `retrieval/medrag_corpus.py` (does not exist yet)
- Stream MedRAG sub-corpora from HuggingFace (`datasets.load_dataset`, streaming
  mode to avoid loading all rows into RAM at once).
- Normalise each snippet into the existing `Chunk` schema
  (`chunk_id`, `source`, `title`, `text`) — same shape the pilot uses, so BM25 /
  FAISS builders and citation paths need no change.
- Write to `data/corpora/medcorp.jsonl` (git-ignored, large).
- **Test:** loads N rows, every row is a valid `Chunk`, ids unique.

### 2. Extend `build_indexes.py` for scale
- BM25 leg: `rank_bm25` over the full snippet list — RAM-heavy at 427K; measure,
  and if it exceeds budget, either shard or fall back to Tier A/B.
- FAISS leg: embed in **batches** (e.g. 512) with a progress counter; the current
  builder likely embeds in one call — must batch to bound peak RAM and to be
  resumable. Persist embeddings so a crash doesn't re-embed from zero.
- Time estimate (CPU, MiniLM): ~500–1500 snippets/sec on quad-core →
  **~5–15 min for 427K**. Not the bottleneck; RAM is.
- Output: `indexes/bm25_medcorp.pkl`, `indexes/faiss_medcorp/`.

### 3. Sanity-check retrieval quality BEFORE any LLM run
- Script: for a sample of MIRAGE questions, print top-5 retrieved chunks.
- **Gate criterion:** the answer-bearing passage should appear for clearly
  factual questions (e.g. "phrenic nerve → diaphragm"). If retrieval is still
  junk, fix here — do NOT spend LLM hours on a bad index.

### 4. Re-run policies on the real corpus (the payoff)
- P1, P4, P5 (calibrated) on 200 MIRAGE MCQs + 200 PubMedQA open, real corpus.
- **Re-check calibration:** retrieval changes the *answer* distribution, not the
  gate's draft signals (gate drafts are pre-retrieval), so τ should transfer — but
  verify retrieval rate is still in-band and re-sweep if not.
- Expected outcome (hypothesis to test): P1 recovers well above 17%; calibrated
  P5 now ≥ P3 (retrieval finally helps), so the gate's accuracy/cost trade-off is
  finally measured fairly.

### 5. Regenerate figures + safety envelope
- `generate_figures.py` re-run on new logs (no code change — reads logs).
- `safety_envelope` now has a corpus where policies can clear the per-risk bars →
  the headline contribution produces a non-empty table.

---

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| 427K FAISS + model exceeds free RAM | Build index in separate process (already the design); close other apps; or drop to Tier A/B |
| BM25 in-memory too large at 427K | Measure; shard or use Tier A/B; BM25 pickle for 427K may be ~hundreds of MB |
| Embedding time on CPU | Batched + resumable; one-time ~5–15 min cost |
| MedRAG HF path/download fails (as RAGCare did) | Verify `MedRAG/statpearls` resolves before committing; else use the MedRAG GitHub chunk files directly |
| Calibration drifts on real corpus | Re-run offline sweep from new logged signals (cheap) |
| Runs get killed (~10 min harness cap) | Same resumable-relaunch pattern already proven; or user runs the .bat in a real terminal |

---

## Supervisor questions to confirm before the full build
1. **Corpus scope** — full StatPearls+Textbooks (427K), StatPearls-only (301K), or
   a justified topic-filtered subset on this hardware? (RAM-bound decision.)
2. Is a focused subset acceptable if full MedCorp is infeasible on the laptop, or
   should this move to a machine with more RAM / a cloud VM?
3. Retrieval-quality bar: is top-k containing the answer passage the acceptance
   criterion, or do they want a formal retrieval-recall number (needs gold
   passage labels, which MIRAGE MCQs don't ship)?

---

## Suggested order of execution (once scope confirmed)
1. `medrag_corpus.py` + test  →  2. verify HF download of chosen sub-corpus
3. batch/resumable index build  →  4. retrieval sanity check (STOP if junk)
5. re-run P1/P4/P5 (relaunch pattern)  →  6. regenerate figures + safety envelope
7. update `week3_policies.tex` results with real-corpus numbers.
