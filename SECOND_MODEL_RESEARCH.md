# Second-Model Research — Options, Pros & Cons

**Purpose:** the real corpus is built and the gate works, but no policy clears the
safety bars (best = P3 closed-book 62% vs low-risk bar 70%). This is now a
*model-capability* ceiling. A second, stronger model tests two things:
1. can a better model clear the safety envelope's accuracy bars?
2. do the two key findings (**selective > always-retrieve**, and
   **calibration doesn't transfer**) generalise across models, turning them from
   single-model observations into general properties?

**Decision is the user's — this document only lays out options. Confirm later.**

---

## Hard requirements (non-negotiable for this codebase)

| Requirement | Why | Consequence if unmet |
|---|---|---|
| **GGUF format** | The backend is `llama-cpp-python`; it loads GGUF only | Can't use it without a new backend |
| **`logits_all=True` capable** | Entropy + margin gates read full per-token logits | Any GGUF model supports this — always met |
| **Fits ~8 GB RAM at Q4** | CPU-only student laptop, medium tier | OOM / swapping |
| **Deterministic greedy** | temp 0 + seed for reproducibility | Non-reproducible results |
| **Instruction-tuned** | Answers MCQ + open-ended in a followable format | Poor extraction, low scores |

Note: the codebase *already* references two other models — `phi-3.5-mini`
(high tier) and `llama-3.2-1b` (low tier) — so the config + factory paths for a
second model largely exist.

---

## Candidate models

### A. Phi-3.5-mini-instruct (3.8B) — RECOMMENDED
- **Already in the config** (`hardware_high.yaml`), so least integration work.
- Microsoft; small but punches far above its weight on reasoning/knowledge
  benchmarks — designed for exactly the "small model, strong quality" niche.
- **Pros:** different architecture + training data from Llama → clean test of
  model-agnosticism; similar size (~3.8B vs 3B) so RAM/latency comparable; strong
  MMLU/medical performance for its size; GGUF widely available (Q4_K_M ~2.3 GB).
- **Cons:** only modestly larger than the current 3B, so the accuracy *ceiling*
  lift may be limited — good for the "does it generalise" question, less decisive
  for "does it clear the safety bar".
- **Best for:** proving the findings generalise across architectures at low cost.

### B. Qwen2.5-7B-Instruct (7B) — best accuracy-ceiling test
- **Pros:** meaningfully larger (7B) → the model most likely to actually lift the
  closed-book ceiling toward the 70% bar; excellent on medical/reasoning
  benchmarks; strong GGUF support; different family (Alibaba) → also tests
  agnosticism.
- **Cons:** Q4_K_M ~4.5 GB resident + FAISS 654 MB + BM25 → tight on 8 GB, likely
  needs apps closed / the high tier; ~2× slower per token than 3B on CPU, so P5's
  already-long runs (5 LLM calls/query) get longer.
- **Best for:** answering "can a stronger model clear the safety envelope?"

### C. BioMistral-7B / medical-tuned 7B — domain specialist
- **Pros:** explicitly fine-tuned on medical corpora (PubMed etc.) → potentially
  the highest medical accuracy per parameter; directly on-topic for a medical-RAG
  dissertation; GGUF available.
- **Cons:** medical fine-tunes can be weaker at instruction-following/format
  adherence (hurts MCQ letter extraction); 7B RAM/latency cost as in (B); quality
  of specific GGUF quantisations varies — needs verifying.
- **Best for:** a domain-specialist comparison point ("does medical fine-tuning
  beat general capability?") — a nice extra axis if time allows.

### D. Llama-3.2-1B (already the low tier) — the cheap contrast
- **Pros:** already configured; near-zero integration; provides the *downward*
  comparison (does the gate still work on a weaker model, on the probe-only low
  tier where entropy/margin are disabled?).
- **Cons:** will *lower* accuracy, not raise it — does nothing for the safety
  envelope. Only useful as a tier-comparison data point.
- **Best for:** completing the hardware-tier story (low/medium/high), not for
  clearing safety bars.

---

## Recommendation matrix

| Goal | Best pick |
|---|---|
| Prove findings generalise, cheapest | **Phi-3.5-mini (A)** |
| Actually clear the safety bars | **Qwen2.5-7B (B)** |
| Medical-specialist axis | BioMistral-7B (C) |
| Complete the tier story | Llama-3.2-1B (D) — already available |

**Suggested plan (if budget allows two):** Phi-3.5-mini for the agnosticism test
(cheap, already configured) **plus** Qwen2.5-7B for the ceiling test (the one that
might clear the safety bar). Run only P3 + P5 on the second model(s), MCQ + open —
not all six policies — to keep CPU cost bounded. Each new model needs its own
offline threshold calibration (cheap: replay from logged signals).

---

## Cost reality check (CPU, per second model)
- P5 makes ~5 LLM calls/query; on the current 3B that is ~150--380 s/query.
- A 7B model roughly doubles per-token time → P5 runs could hit ~5--10 min/query
  → a 200-question P5 run is many hours. The resumable self-healing batch
  (`run_overnight.bat`) already handles this, but budget the wall-clock time.
- Phi-3.5-mini (~3.8B) is close to current cost; Qwen-7B is the expensive one.

## Open questions for the user / supervisor
1. One second model or two (agnosticism + ceiling)?
2. If a 7B is chosen, is closing other apps / using the high tier acceptable, or
   should the heavy runs move to a machine with more RAM?
3. Is a medical-specialist model (BioMistral) wanted as an extra axis, or keep the
   comparison to general-capability models?
