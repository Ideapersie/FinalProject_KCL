# Honest Feedback & One-Month Plan — Adaptive Retrieval Gating Dissertation

*Written 2026-07-06. ~1 month to deadline (target ~21 July, plus buffer). Weighting: 20% of final grade. Reader: you (MSc student, final dissertation). Also written with an eye to the "could this become a medtech startup?" question you raised.*

---

## 1. The honest headline

**This is a genuinely good project with one uncomfortable result you must handle deliberately, not hide.**

You have real novelty (training-free 3-gate ensemble + token-entropy attribution), clean and well-tested engineering, and — crucially — a *chain of honest empirical findings* that most MSc dissertations never produce. The calibration-non-transfer story and the "gated retrieval inherits corpus quality" story are the kind of insight that comes from actually running experiments and thinking about the results, not from a tutorial. That is worth a lot.

**But:** your best accuracy is 62% (closed-book), and no policy clears even the 70% low-risk safety bar. You already see this. The instinct to "get the accuracy up so it's usable in a hospital" is the right instinct commercially and the *wrong* instinct for the dissertation with a month left. More on this below — it's the single most important strategic point in this document.

**Grade read:** as it stands, this is comfortably a Merit and within reach of a Distinction. The gap to Distinction is **not** more accuracy — it is (a) framing the low accuracy as a deliberate, well-analysed finding rather than a shortfall, (b) finishing the analysis chapters (safety envelope, calibration, ablation) that show depth, and (c) the write-up quality (which is already strong in the drafts I've read).

---

## 2. What's genuinely strong (bank these, don't over-polish)

- **Real, defensible novelty.** Training-free gating that needs no fine-tuning is a real contribution, and the 3-signal ensemble with a hallucination probe (instead of unreliable verbalized confidence) is a thoughtful design, well justified in `week2_gating.tex`.
- **Token-Level Entropy Attribution.** This is your most *distinctive* and least-exploited asset. It localises *which token* of a clinical question the model is unsure about. Right now it's logged but not analysed or visualised. This is low-hanging fruit for both marks and startup appeal (see §5).
- **Engineering discipline.** 106 tests, mock backend, resumable runs, 4-file config, one-source-of-truth figures. Examiners notice reproducibility. This is Distinction-level rigour.
- **The findings chain.** Four honest results, each building on the last:
  1. Default thresholds physically can't fire on a quantised model (calibration non-transfer).
  2. Gated retrieval inherits corpus quality (P5 dropped on the pilot corpus).
  3. On a real corpus, selective retrieval beats always-retrieve (P5 54% vs P1/P4 46% at half the retrieval).
  4. On MCQ, closed-book still wins → a model-capability ceiling, not a method failure.
- **Write-up quality.** The LaTeX chapters are formal, motivated, and honest. This is already better than typical.

---

## 3. The uncomfortable truth about accuracy — read this twice

**62% is not a failure of your method. It is a property of a 4-bit quantised 3B model on a hard medical benchmark. Do not spend your last month chasing accuracy.**

Here is the trap: you are (rightly, as a founder) drawn to "make it hospital-usable." But:

- Clinical safety needs ~90%+ on high-risk questions. No 3B model — and honestly no single small model — gets there on MIRAGE. Chasing it will consume your month and still fall short, and a half-finished accuracy chase reads *worse* than a well-analysed limitation.
- **Your dissertation's contribution is the gating method and the safety-aware analysis, not a deployable clinical product.** Examiners grade the former. Conflating the two is the most common way strong projects lose marks.

**Reframe the low accuracy as a first-class result.** Your safety-envelope table showing "no policy clears the bar at this model scale" is a *finding*, not an embarrassment — it quantifies exactly how far small-model medical RAG is from clinical safety, which is genuinely useful and honest. Then the second-model / cloud work becomes "how much does scale close the gap?" — a scaling study with a clear narrative, rather than a desperate accuracy grind.

**One caveat where accuracy IS worth a little effort:** cheap correctness leaks you already identified — the MCQ letter extractor losing valid answers behind preambles ("Note: I'll follow the format…", reasoning-before-answer). Fixing that is a few hours and recovers *real* accuracy you are currently throwing away. Do that. It's different from chasing model capability.

---

## 4. Gaps to close (ranked by marks-per-hour, one-month budget)

### Tier 1 — do these, highest return, all achievable
1. **Gate ablation** (½–1 day). Each gate solo vs the ensemble, on data you already have (replay from logged signals — no new LLM runs). This *defends your central novelty*: does the 3-gate ensemble actually beat any single gate? Without it, a examiner asks "why three?" and you have no answer. Highest marks-per-hour in the whole project.
2. **Fix the MCQ answer extractor** (few hours). Recovers real accuracy currently lost to formatting. Re-score existing logs (no re-runs needed for scoring-only).
3. **Finish the safety-envelope analysis + figure** (1 day). Code exists. Frame the empty envelope as the honest "distance to clinical safety" finding. This is your stated publishable contribution — it must appear as a proper table + discussion.
4. **Token-Entropy Attribution: one worked visualisation** (1 day). You have the per-token arrays logged. Produce *one* figure showing, on a real clinical question, which tokens the model was uncertain about. This is your most distinctive contribution and currently invisible in the report. Huge distinctiveness-per-hour.

### Tier 2 — strong if time allows
5. **Second model** (Phi-3.5 local, or Qwen-7B on cloud per your plans). Turns "calibration doesn't transfer" and "selective beats always" from single-model observations into *general* claims. Cloud (llama-cpp+CUDA) makes this fast and cheap (~$5–10) — your `CLOUD_GPU_PLAN.md` is sound. Do **P3+P5 only**, both question types.
6. **Risk-stratified results** (½ day). You have `risk_level` tags but only 6 high-risk questions — too few. Either pull more high-risk questions or explicitly caveat. The safety-envelope story needs a believable high-risk sample.

### Tier 3 — cut without guilt if the calendar tightens
- P6 Gradio toggle, P7 multi-agent, attention-entropy layer, the OpenAI reference backend. These are stretch goals. A *finished* core beats a sprawling half-built one. Examiners reward depth over breadth.

---

## 5. The startup angle — honest take

You asked about converting this to a medtech startup. Honest founder-to-founder read:

- **The gating idea has real commercial legs — but not as "an AI that answers medical questions."** That market is brutal (regulation, liability, incumbents, and you'll never win on raw accuracy vs frontier models).
- **Where it's genuinely differentiated:** *cost-and-safety-aware routing.* "Retrieve/compute only when the model is uncertain" is a real, sellable efficiency+safety layer — for on-prem/offline deployments (hospitals with data-sovereignty rules, low-connectivity settings, edge devices). That's exactly your CPU-only, offline framing. Lean into it.
- **Token-Entropy Attribution is your most startup-relevant asset.** "Show clinicians *which part* of an answer the model was unsure about" is an *interpretability/trust* feature. Trust and auditability sell in regulated healthcare far more than a 3% accuracy bump. If you build that one visualisation well, it doubles as a dissertation figure *and* a demo slide.
- **What would make it a real product (post-dissertation, not now):** a proper corpus (licensed clinical content), a bigger base model, a human-in-the-loop UI, and — the hard part — a regulatory/liability story. None of that belongs in the next month; note it in Future Work.

**For the dissertation, the startup framing helps in exactly one place: the motivation/introduction and the future-work sections.** It gives your "why offline, why safety-aware, why cost-gated" a real-world anchor. Don't let it distort the technical contribution.

---

## 6. Suggested one-month schedule

Assumes ~4 weeks. Adjust to your real submission date.

**Week 1 — lock the core results (highest-value technical work)**
- Fix MCQ extractor, re-score logs.
- Gate ablation (replay-based) + table.
- Safety-envelope table + figure + discussion.
- Token-Entropy Attribution: one clean visualisation.

**Week 2 — the scaling study (if doing the second model)**
- Cloud box (or Phi-3.5 local): P3+P5, MCQ+open, calibrate the new model.
- Cross-model comparison: does "selective beats always" and "calibration doesn't transfer" hold?
- If skipping the second model, spend this week deepening analysis + starting write-up.

**Week 3 — write the report body**
- You have week1–4 chapters drafted. Now: Introduction (with the offline/safety/cost motivation), Background/Related Work (TARG, Self-RAG, Adaptive-RAG, MIRAGE, ALCE — you have the refs), and the LSEP chapter (medical AI is rich here: liability, bias, data governance — KCL requires it and it's easy marks if done thoughtfully).
- Consolidate the weekly `.tex` files into coherent Methodology / Implementation / Evaluation chapters (right now they read as sprint logs; the final report needs them re-threaded as chapters).

**Week 4 — polish + buffer**
- Discussion + Conclusion + Future Work (this is where the startup vision goes).
- Proofread, tighten figures, check every number against logs.
- Buffer for the inevitable overruns.

---

## 7. The three things that most move your grade

If you do nothing else from this document, do these:

1. **Reframe low accuracy as a deliberate, quantified finding** (the safety-envelope "distance to clinical safety") — not a shortfall to apologise for. This single reframing is the difference between "the accuracy is disappointing" and "the analysis is mature."
2. **Do the gate ablation.** It defends your central novel claim and is nearly free (replay from logs).
3. **Ship one Token-Entropy Attribution visualisation.** It's your most distinctive asset and it's currently invisible.

Everything else is upside. The project is in good shape — the risk now is misallocating the last month chasing accuracy instead of finishing the analysis and the write-up. Resist it.
