# Cloud GPU Plan — Scaling the Second-Model Runs

**Question being answered:** should the bigger-model policy runs go on a rented
cloud GPU instead of the local CPU? This is a concrete setup + cost plan so the
decision can be made on facts. **No commitment yet.**

**Framing (important):** this is an *additive scaling tier*, not a relocation. The
local CPU-only results stay the primary contribution — they are what makes
adaptive gating meaningful (offline, hospital-deployable, resource-constrained).
The cloud runs answer a complementary question: *do the findings hold on a larger
model, and can a stronger model clear the safety-accuracy bars?* Report it as a
scaling study, not as the main result.

---

## Why this is technically cheap for THIS codebase

The backend is `llama-cpp-python` (`Llama()`, `create_completion`, `_scores`,
`logprobs=2`). llama-cpp has a CUDA build: adding **`n_gpu_layers=-1`** to the
`Llama()` constructor offloads all layers to the GPU. That is essentially the only
code change — the gate signals still come from the same `_scores` array (entropy)
and `logprobs=2` response (margin).

**This is the decisive reason to stay on llama-cpp rather than switch to vLLM/TGI:**
those serve faster but expose full per-token logits awkwardly or not at all, which
would break the entropy and margin gates. llama-cpp + CUDA keeps the gates working
unchanged.

### Minimal code change
```python
# models/llama_backend.py — add one config-driven arg to Llama(...)
self._llm = Llama(
    model_path=gguf_path,
    logits_all=logits_all,
    n_gpu_layers=n_gpu_layers,   # NEW: -1 = all layers on GPU; 0 = CPU (default)
    ...
)
```
Expose `n_gpu_layers` in the hardware config (default 0, set -1 on a cloud tier).
Everything else — retrieval, gates, policies, scoring, logging — is unchanged.

---

## What must move to the cloud box

| Artifact | Size | Move or rebuild? |
|---|---:|---|
| `data/corpora/medcorp_tp.jsonl` | 413 MB | **Upload this**, rebuild indexes on-cloud (fast on GPU box) — smaller transfer than the indexes |
| `indexes/faiss_medcorp_tp/` | 1.7 GB | OR upload directly to skip rebuild |
| `indexes/bm25_medcorp_tp.pkl` | 787 MB | (part of above) |
| GGUF model | — | **Download on the box** from HuggingFace, don't upload |
| Code (`src/`, `scripts/`, `configs/`) | tiny | `git clone` on the box |

**Recommended:** `git clone` the repo on the box, upload only the 413 MB corpus
JSONL, rebuild indexes there (embedding 426K on a GPU is minutes). Avoids the
2.5 GB index upload.

---

## Provider options

| Provider | GPU (example) | ~Price/hr | Notes |
|---|---|---:|---|
| **RunPod** | RTX 4090 (24 GB) | ~$0.35–0.70 | Cheap, per-second billing, easy templates. Good default. |
| **Vast.ai** | RTX 3090/4090 | ~$0.20–0.50 | Cheapest (marketplace), variable reliability |
| **Lambda** | A10 / A100 | ~$0.60–1.30 | Clean, reliable, pricier |
| **Google Colab Pro+** | A100 (spot) | ~$10/mo | Notebook UX, session limits, not ideal for long batch |
| **Kaggle** | free T4/P100 | free | 30 hr/wk quota; viable for a modest run if patient |

For a 7B GGUF at Q4, a **24 GB card (RTX 4090)** is ample and cheapest-good. A100
(40–80 GB) only needed for ~70B models.

---

## Concrete run plan (once a provider is picked)

1. **Spin up** a CUDA GPU box (RTX 4090, 24 GB).
2. **Install:** `pip install llama-cpp-python` with CUDA
   (`CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python`) + repo deps.
3. **`git clone`** the repo; **upload** `medcorp_tp.jsonl` (413 MB via `scp`/rsync).
4. **Rebuild indexes** on the box: `python scripts/build_medcorp.py --stage assemble`
   (or re-embed — minutes on GPU).
5. **Download** the chosen GGUF (e.g. Qwen2.5-7B-Instruct-Q4_K_M).
6. **Add a `hardware_cloud.yaml`** tier: `n_gpu_layers: -1`, `logits_all: true`,
   the new model path.
7. **Calibrate** the new model offline (a short P5 run to log signals, then
   `run_threshold_sweep.py` — same as before, cheap).
8. **Run** the policies via the existing `run_overnight.bat`-equivalent shell
   script (write a `.sh` version — cloud box is Linux). On GPU there are no ~10-min
   kills, so it runs straight through.
9. **Download** the `results/raw_logs/*.jsonl` back (small), regenerate figures
   locally.

---

## Cost estimate

- Full run: P1/P3/P4/P5 × {MCQ, open} on one 7B model.
- On a 4090, ~1–5 s/query even for P5's 5 calls → a 200-q P5 run ~10–30 min;
  lighter policies faster. All policies+types ≈ **2–5 GPU-hours**.
- Two models (Phi-3.5 + Qwen-7B) ≈ **5–10 GPU-hours**.
- At ~$0.50/hr → **$3–5 total**, plus ~1–2 hr setup (billed) → call it **$5–10**.

Effectively negligible versus weeks of local CPU grinding.

---

## Pros / cons summary

**Pros**
- Runs a genuinely capable model → chance to clear the safety-accuracy bars.
- 10–50× faster → all six policies feasible, no kill/relaunch grind.
- 7B/13B fit VRAM trivially; no local RAM fight.
- Tiny code change (llama-cpp CUDA); gates keep working.
- Cheap ($5–10).

**Cons**
- Must protect the thesis framing (cloud = scaling tier, not the main result).
- Setup overhead (CUDA llama-cpp build can be fiddly; ~1 hr).
- Data upload (413 MB) + result download.
- Ongoing cost if left running — **remember to shut the box down** (per-second
  billing means an idle box still bills).
- Reproducibility note for the report: GPU vs CPU floating-point can differ
  slightly; keep temp 0 + seed and note the hardware.

---

## Recommendation
**Yes, use cloud for the bigger-model scaling study — via llama-cpp + CUDA, as an
added `hardware_cloud` tier, keeping the local CPU results as the primary
contribution.** Pick RunPod RTX 4090 as the default. Start with one model
(Qwen2.5-7B — the ceiling test) to see if it clears the safety bar; add Phi-3.5
only if a second data point is wanted.

## Decision points for the user
1. One cloud model (Qwen-7B) or two (add Phi-3.5)?
2. RunPod (easy) vs Vast (cheapest) vs Kaggle (free, slower)?
3. OK to keep the framing as "scaling study, local CPU stays primary"? (protects
   the dissertation's novelty)
