# Progress Check 4 — Talking Points for Supervisor Meeting

**Student:** Tharit Mohamed Hussain Khan (22058691) · KCL MSc AI · 7CCSMPRJ
**Date prepared:** 2026-06-26
**Branch:** `feat/p5-gated-prototype`

---

## 1. One-line summary

The core novelty — the 3-gate adaptive retrieval ensemble (P5) — is **built,
tested, and running end-to-end** on the real local model. The first full run
produced a **genuine empirical finding**: literature-default gate thresholds do
not transfer to a small quantised medical model, which directly motivates the
calibration work. The prototype did exactly what a prototype should: it
validated the mechanism *and* surfaced the next research question.

---

## 2. What's done since last check (headline achievements)

- **Three training-free gates implemented** (the core contribution):
  - Token Entropy gate (+ per-token entropy attribution — a novel interpretability output)
  - Logit Margin gate (parser validated against the real llama-cpp output, not just mocks)
  - Hallucination Probe gate (two-framing consistency; replaces verbalized confidence)
- **Ensemble gate** — majority vote (2-of-3), with hardware-aware degraded mode
- **P5 gated policy + policy factory** — gate decides retrieve vs closed-book per query
- **Pilot retrieval corpus + BM25 index** for the RETRIEVE path
- **First full P5 run** on 200 MIRAGE questions, directly comparable to the P3 baseline
- **86/86 unit tests passing** in ~0.5s, no model download needed
- **Two dissertation chapter drafts written** (`report/week1_implementation.tex`,
  `report/week2_gating.tex`)

---

## 3. The key results to present

### 3a. Baseline (Week 1) — establishes the floor

| Metric | P3 (closed-book), 200 MIRAGE Qs |
|---|---|
| Accuracy | **62.0%** |
| Latency p50 / p95 | 7.42 s / 9.25 s |
| Mean energy / query | 6.77×10⁻⁴ kWh |

**Talking point:** 62% closed-book accuracy proves the quantised 3B model holds
real medical knowledge — so "when can we skip retrieval?" is a *meaningful*
question, not a rhetorical one. If closed-book were near chance, gating would be
pointless.

### 3b. P5 prototype (Week 2) — the central finding

| Metric | P3 (closed-book) | P5 (gated, default τ) |
|---|---|---|
| Accuracy | 62.0% | **62.0%** (identical) |
| Retrieval rate | 0% | **0%** |
| Latency p50 | 7.42 s | 143.7 s |
| Mean energy / query | 6.77×10⁻⁴ kWh | 1.20×10⁻³ kWh |

**Why identical accuracy:** the gate skipped all 200 questions, so every P5
answer was generated the same way as P3 → same outputs → same accuracy. This
*proves the skip-path is wired correctly* (a bug would make accuracy differ, not
match exactly).

**Why P5 is slower:** P3 = 1 LLM call/query; P5 = ~4 (entropy/margin draft +
2 probe drafts + final answer). It paid the full gating cost and — because the
gate never fired — answered closed-book anyway.

### 3c. The per-gate trace — this is the most important slide

| Gate | Signal range observed | Default threshold | Retrieve votes |
|---|---|---|---|
| Entropy | 0.27–1.59 nats | trigger if > 2.5 | **0 / 200** |
| Margin | 0.45–0.89 | trigger if < 0.3 | **0 / 200** |
| Hallucination probe | (threshold-free) | — | **80 / 200** |

**The finding:** entropy maxes at 1.59 but needs >2.5 to fire; margin never
drops below 0.45 but needs <0.3. Both logit-based gates are structurally pinned
to SKIP. Only the probe ever voted retrieve (80/200), but a lone vote loses a
2-of-3 majority. **Default thresholds — tuned on large general-domain models —
do not transfer to a 4-bit 3B medical model.**

**Calibration target (from observed percentiles):** entropy ≈ 0.9 (p75),
margin ≈ 0.64 (p25). This is the Sprint 4 sweep.

---

## 4. Code snippets to show

### 4a. Entropy gate — the signal + the novel per-token attribution

```python
# src/medrag_adaptive/gating/entropy_gate.py
per_token = _row_entropy(_softmax(np.asarray(logits, dtype=np.float64)))
mean_entropy = float(per_token.mean())

return GateDecision(
    name=self.name,
    retrieve=mean_entropy > self.threshold,
    signal_value=mean_entropy,
    details={
        "available": True,
        "mean_entropy": mean_entropy,
        "threshold": self.threshold,
        "per_token_entropy": [float(h) for h in per_token],  # <- novelty
    },
)
```
**Point:** the gate decision uses the mean, but we also keep the *per-token*
entropy array — this localises uncertainty to specific tokens of a question, an
interpretability output no prior adaptive-retrieval work produces.

### 4b. Ensemble — majority vote with availability accounting

```python
# src/medrag_adaptive/gating/ensemble_gate.py
degraded = available < self.min_votes
if degraded:
    # low tier: only the probe runs; strict majority unreachable
    retrieve = retrieve_votes >= 1
else:
    retrieve = retrieve_votes >= self.min_votes
```
**Point:** abstaining gates (entropy/margin on the low tier where logits are off)
are excluded from the vote, and the ensemble falls back gracefully — a
hardware-aware design decision, logged as `degraded` for auditability.

### 4c. Hardware-aware downgrade in the factory

```python
# src/medrag_adaptive/policies/factory.py
logits_available = cfg.model.logits_all
if not logits_available:
    # entropy + margin need the raw logit buffer; only the probe can run
    return _build_single_gate("hallucination_probe", cfg)
```
**Point:** a policy config can't request a gate the hardware physically can't
run. The gate that actually executes is constrained by the hardware tier, not
just the config file — this underpins the "Hardware-Aware Policy Escalation"
contribution.

---

## 5. What this proves about engineering rigour (for the marks)

- **Test-first throughout:** every gate and policy built against a mock backend
  before any real run. 86 passing tests, run in 0.5s with no model needed.
- **Validated assumptions against reality:** the margin-gate logprob parser was
  checked against the real llama-cpp output, confirming the mock is faithful.
- **Full auditability:** every gate decision (per-token entropy, all votes, both
  probe drafts) is logged per query to JSONL — every result is reconstructable.
- **Resumable runs:** the 200-Q run is restartable from where it stopped, needed
  because each run is hours on CPU.

---

## 6. Honest limitations to raise (shows critical reflection)

- **Pilot corpus, not full MedCorp:** the RETRIEVE-path corpus is ~200 curated
  chunks; retrieval-*quality* numbers are provisional. Full StatPearls/MedCorp
  is the next retrieval task. (Retrieval-*rate* and gate-vote findings are not
  affected by corpus size.)
- **Gate never fired at default thresholds:** so P5's accuracy/latency *benefit*
  is not yet demonstrated — only the mechanism is. Calibration unlocks this.
- **Latency overhead is real:** 3 gating drafts/query is expensive on CPU;
  motivates calibration and the low-tier probe-only configuration.
- **High-risk stratum is tiny** (6/200 questions) — the full experiments need
  risk-stratified sampling for the safety-envelope analysis.

---

## 7. Immediate next steps (what I'll do before next check)

1. **Threshold calibration sweep** — sweep τ_H, τ_M over observed signal
   distributions; pick model-specific operating points; re-run P5.
2. **Re-run P5 calibrated** → expect retrieval rate > 0% and accuracy moving
   toward the P1 ceiling.
3. **Gate ablation** — each gate solo vs the ensemble (does the probe catch
   errors the logit gates miss?).
4. **FAISS + hybrid retriever + real corpus** (Sprint 3 completion) for P2/P4.

---

## 8. Questions for the supervisor

- Is the zero-retrieval threshold-transfer result worth foregrounding as a named
  finding in the Evaluation chapter, or treating as a calibration footnote?
- For calibration: tune thresholds per-dataset, per-risk-level, or one global
  operating point per model? (Affects how much of the sweep to run.)
- Priority for the next two weeks: calibration + ablation (deepen the novelty),
  or FAISS/hybrid + P2/P4 (broaden policy coverage)?
```
