# Dissertation Feedback & Change List

*Written 2026-07-15. Full-history review: project plan, all planning MDs, memory state, raw logs, generated tables, and the four weekly LaTeX chapters. Companion to the earlier `HONEST_FEEDBACK.md` (2026-07-06) — this file does not repeat it; it extends it with what changed since, plus concrete fixes.*

---

## 0. TL;DR — the three things that matter most now

1. **Ship-blocking LaTeX bug.** `week1/2/3.tex` reference `ccPthreeMcq`, `etPthreeMcq`, `1PthreeOpen` — these are `\accPthreeMcq`, `\retPthreeMcq`, `\f1PthreeOpen` with the leading `\a`/`\r`/`\f` deleted (a bad find-replace ate the command backslash+first letter). They render as literal text, not the calibrated numbers. **Fix before anything else** (§3.1).
2. **The report is still four sprint logs, not a thesis.** `main.tex` still `\input`s week1–4 as chapters. The best marks come from re-threading these into Methodology / Implementation / Evaluation and *adding the missing chapters* (Intro, Background, Discussion, LSEP, Conclusion). The skeleton I've written (`report/main_skeleton.tex`) is the target structure.
3. **You have MORE story than the report currently tells.** Three strong findings from July that are in memory/logs but NOT in any chapter: the retrieval-distraction analysis (retrieval breaks 46, fixes 13), the gate's 19/19 protective skip, and the chunk-relevance verdict. These are your best evaluation material and they're invisible. Get them into Ch5 (§4).

Grade read is unchanged from `HONEST_FEEDBACK.md`: solid Merit, Distinction reachable. The gap is write-up completeness + framing, not more experiments.

---

## 1. Project trajectory — what actually happened vs the plan

The original `Variant5_..._Project_Plan.md` was ambitious: 6 policies × 3 gates × 3 hardware tiers × 4 datasets + user study + Gradio UI + qVault integration. Reality (from logs + memory) converged on a tighter, more defensible core:

| Planned | Delivered | Verdict |
|---|---|---|
| 6 policies (P1–P6) | P1, P2, P3, P4, P5 built; **P6 toggle / P7 multi-agent cut** | Correct call. Depth > breadth. |
| 3 gates incl. verbalized confidence | entropy + margin + **hallucination probe** (replaced verbalized confidence) | Good design change, well justified. |
| 3 hardware tiers, physically tested | tiers exist in *config* (logits_all downgrade); only **medium tier actually run** | Honest gap — see §5. |
| 4 datasets incl. EHR-DS-QA, synthetic acute-care | MIRAGE (MMLU 200 MCQ) + **PubMedQA** (RAGCare path was dead) | Substitution is fine and documented. |
| StatPearls/MedCorp corpus | **Textbooks + 300K PubMed = 425,847 chunks** built | Delivered, this was the critical unblock. |
| Second model cross-validation | **prereqs done, not yet run** (Qwen 7B/14B Colab notebook ready) | The main open item. |
| User study (5–10 participants) | cut | Fine to cut; note in limitations. |

**The narrative arc is genuinely good** and is the spine of the thesis:
1. Default thresholds (from full-precision literature) physically can't fire on a quantised 3B → **calibration non-transfer**.
2. Calibrated gate fires, but P5 accuracy *drops* on the pilot corpus → **gated retrieval inherits corpus quality**.
3. On the real 426K corpus, P5 (54.5%) beats P1/P4 (45.5%) at half the retrieval → **selective beats always-retrieve**.
4. Closed-book P3 (62%) still tops MCQ → **model-capability ceiling**, not method failure.
5. (Newer, unreported) Retrieval is actively *distracting* on questions the model knows → the gate's skip is protective exactly where retrieval is poison.

That is a real chain of empirical reasoning. Most MSc dissertations don't have one. Protect it.

---

## 2. What's genuinely strong (bank it, stop polishing)

- **Reproducibility discipline.** Single-source-of-truth numbers (`numbers.tex` generated from logs), a drift-guard regression test that fails the build on a hand-edited literal, paired-run verification (P3 corpus-independence test), 180+ passing unit tests, resumable runs. Examiners reward this heavily and most students can't demonstrate it. Make it explicit in the Implementation chapter and the LSEP/scholarship framing.
- **Honest negative results.** The pilot-corpus collapse, the empty safety envelope, the gate-variants refutation (your own hypothesis disproven), the Newcombe CI crossing zero. Every one is written up as a finding, not hidden. This is the single biggest signal of research maturity.
- **The calibration-non-transfer result** is publishable-flavoured and cheap to defend (offline replay from logged signals).
- **Design judgement.** Replacing verbalized confidence with the hallucination probe, keeping the risk tagger rule-based (auditable) not learned, RRF for scale-free fusion — each is a defensible decision with a stated reason.

---

## 3. Bugs & correctness issues to fix

### 3.1 (BLOCKER) Broken macro references in week1–3
`ccPthreeMcq` / `etPthreeMcq` / `1PthreeOpen` are `\accPthreeMcq` / `\retPthreeMcq` / `\f1PthreeOpen` with the command head stripped. Locations (from grep):
- `week1_implementation.tex`: lines 297, 321, 326 (`ccPthreeMcq`)
- `week2_gating.tex`: line 217 (×2), 249
- `week3_policies.tex`: lines 164, 174 (`ccPthreeMcq`), 174–175 (`etPthreeMcq`), 178 (`1PthreeOpen`), 178 (`etPthreeOpen`)

Fix: restore the leading `\a` / `\r` / `\f`. Then grep the whole `report/` tree for any bare `[a-z]P(one|three|four|five)` token that isn't preceded by `\` — that's the signature of the same corruption. Recommend adding a build check: `grep -nE '(^|[^\\])(cc|et|at)P(one|three|four|five)' report/*.tex` should return nothing.

### 3.2 Stale hard-coded numbers that survived the numbers.tex migration
Some cells are still literals instead of macros, so they *won't* be caught by the drift guard and will silently rot:
- `week1` Table `tab:p3-baseline`: latency `7.42 s`, `9.25 s`, energy `6.77e-4 kWh` — hard-coded.
- `week2` Table `tab:p5-prototype`: `143.7 s`, `154.1 s`, energy figures, `Entropy 0.27–1.59`, `Margin 0.45–0.89` — hard-coded.
- `week3` Table `tab:week3-results`: `17.0%`, `45.5%`, `142 s`, `0.191`, `79%` — hard-coded (these are *pilot/uncalibrated* numbers, intentionally frozen for contrast, but that intent isn't machine-checked).
- `week4`: `380 s` open latency, `23 s`, `27 s`, `426,000`, `654 MB`, `787 MB` — hard-coded.

Action: either macro-ise them too, or add a comment block listing which literals are *deliberately frozen historical values* so a future you doesn't "fix" them. Right now the reader can't tell a frozen-on-purpose number from one that rotted.

### 3.3 references.bib is a stub with `[TODO verify]`
Only 8 entries, several with `[TODO authors]` and `arXiv:2025.xxxxx [TODO verify id]`. This will produce broken citations and looks unfinished. Needs: real MIRAGE (Xiong 2024 ACL Findings), Self-RAG (Asai ICLR 2024), Adaptive-RAG (Jeong NAACL 2024), TARG (verify the arXiv id — 2511.09803 per the plan), ALCE (Gao EMNLP 2023), Kadavath 2022, Guo 2017 (ECE), Robertson & Zaragoza (BM25), llama.cpp, Llama-3 model card, PubMedQA (Jin 2019), MiniLM/sentence-transformers, FAISS. ~15–20 solid refs. Budget half a day.

### 3.4 week2 placeholder caveat is now false
`week2_gating.tex` header comment says the prototype table is "a PLACEHOLDER pending completion of the 200-question P5 run." That run is long done. Remove the stale comment so a reader (or examiner glancing at source) isn't misled.

---

## 4. Missing analysis that's already in your data (highest marks-per-hour)

These exist in `raw_logs/` + memory but appear in NO chapter. Each is a paragraph + at most one figure you can generate from existing logs — no new LLM runs.

1. **Retrieval-distraction analysis** (P1-vs-P3 paired, `p1_medcorp_mcq.jsonl`): retrieval **breaks 46** correct answers, **fixes 13**, net −33/200 (−16.5%), harmful 3.5:1. On the broken 46, P1 answers are ~15× longer (274 vs 18 chars) and 26% are refusals. This *explains* why closed-book wins MCQ — it's not "the model is bad," it's "retrieval distracts a model that already knew." This is your mechanism, and it's missing.
2. **The killer protective-skip stat**: on those same 46 poison questions, when P5's gate **skipped**, it scored **19/19 = 100%**; when it retrieved, 3/27 = 11%. This is the strongest single sentence you can write for the gate's value proposition. It's nowhere in the report.
3. **Chunk-relevance verdict** (`p1_medcorp_mcq_chunks.jsonl` + `analyse_chunk_relevance.py`): chunks are equally on-topic in broken vs correct groups (~0.14 overlap) but **2.4× less answer-bearing** on broken questions (13% vs 31% gold-in-chunk). Nuanced "both hypotheses partly right" finding — retriever fetches relevant-but-not-decisive text, model can't distinguish. Distinguishes retriever fault from model fault. Strong Ch5/Discussion material.
4. **Gate-variants ablation** (`gate_variants_mcq.json`): every probe-boosting variant retrieves MORE and scores WORSE than the shipped majority-2of3. Your hypothesis (probe should carry more weight) was *refuted*. A refuted hypothesis honestly reported is Distinction-signal. Currently only in memory.
5. **Token-entropy attribution figure** (`fig_entropy_attribution.png` exists): the honest finding that high entropy sits on *filler/phrasing* tokens, not clinical-content tokens — so raw entropy measures phrasing freedom, not clinical uncertainty. A real limitation of the method, and your most distinctive asset. The figure exists; the *analysis* isn't written.

**If you do only five things in Ch5, do these five.** They're all replay/analysis, zero GPU.

---

## 5. Validity threats to state explicitly (don't let the examiner find them first)

- **n=200, single seed, single dataset subset (MMLU only for MCQ).** The headline P5-vs-P1 gap (+9.0pp) has a 95% CI of [−0.8, 18.5] that *crosses zero*. You already say this in week4 — good. Make sure the abstract and conclusion don't overclaim; "consistent with an improvement" not "improves."
- **Hardware tiers not physically benchmarked.** Only medium tier was run. The low/high tiers exist as config downgrades. Reframe as "the *mechanism* for tier-adaptive degradation is implemented and unit-tested; empirical per-tier latency/energy on real 4GB/16GB devices is future work." Don't imply you measured three devices.
- **High-risk stratum n=6.** Wilson CI ±30pp — cannot distinguish policies. You have this (`tab_risk.tex`). State plainly that the safety-envelope high-risk conclusion is under-powered.
- **Single model.** Everything is one quantised Llama-3.2-3B. "Calibration doesn't transfer" and "selective beats always" are single-model observations until the second model runs. Either run Qwen (prereqs are done) or caveat hard in Discussion.
- **Energy via codecarbon software estimate**, no RAPL guarantee on this box. Report as estimate, note the limitation.
- **PubMedQA token-F1 is low in absolute terms** (0.19–0.22). Explain that paragraph-vs-paragraph token-F1 is a harsh metric and you report *relative* differences, not absolute quality.

---

## 6. Second model — decision needed

Prereqs are all done (memory A9): `prompts.py` refactored to `set_chat_format(llama|qwen|plain)`, verified byte-identical for existing Llama logs so current results stay valid; `config.py` model layer; `--model-config` flag; Qwen 7B/14B YAMLs; `smoke_test_model.py` gate; Colab notebook. Two blockers noted: (a) notebook has `YOUR_USERNAME` placeholder in the clone URL; (b) smoke test MUST pass before the long run.

**Recommendation given the deadline (memory says target ~21 Jul / 7 Aug):** run **Qwen-7B, P3+P5 only, MCQ+open**, one model. That's enough to turn "selective beats always" and "calibration non-transfer" from single-model into two-model claims — the highest-value generalisation. Skip 14B unless 7B runs clean with time to spare. If the calendar tightens, cut the second model entirely and caveat; a *finished* single-model thesis beats a half-run two-model one.

---

## 7. Chapter-by-chapter state & what each needs

| Chapter | Source | State | Action |
|---|---|---|---|
| Abstract | — | placeholder | Write last. ~250w. |
| 1 Introduction | plan §1 | not written | offline/safety/cost motivation + RQs + contributions. ~1,500w. |
| 2 Background | plan §7 refs | not written | TARG, Self-RAG, Adaptive-RAG, MIRAGE, ALCE, calibration. Gap analysis. ~2,500w. |
| 3 Methodology/Design | week2 | drafted, sprint-voice | Re-thread: gate math, ensemble, safety-envelope concept. Strip "Week 2" framing. ~2,500w. |
| 4 Implementation | week1 + week3 build | drafted, sprint-voice | Merge config/schema/retrieval/corpus-build. ~2,000w. |
| 5 Evaluation | week3+week4 results + §4 above | partial — **missing the §4 findings** | The marks. Add distraction, protective-skip, chunk-relevance, ablation, attribution. ~3,000w. |
| 6 Discussion | — | not written | What the findings mean; startup/deployment angle; validity threats (§5). ~1,500w. |
| 7 LSEP | plan §1.2 | not written | Medical AI liability, GDPR/data-sovereignty (offline framing helps here), bias, energy. KCL requires it, easy marks. ~1,200w. |
| 8 Conclusion + Future Work | — | not written | Findings recap, second model, real corpus, bigger model, HITL UI. ~900w. |

Total target ~15,000w. Current drafts ≈6,000 real words → roughly on track once the four new chapters are written.

**Re-threading note:** the weekly chapters read as chronological sprint logs ("Week 2 delivered X but left three gaps..."). A thesis chapter is organised by *concept*, not *time*. When you migrate week2→Ch3, drop "this sprint," "Week 1," "the prototype" and present the gate design as a finished thing. Keep the reasoning; change the tense and framing.

---

## 8. Suggested order of work (deadline-aware, ~1 week to target)

1. **Day 0 (½ day):** Fix §3.1 macro bug + §3.4 stale comment. Finish references.bib (§3.3). Compile clean in Overleaf. *Now the report at least builds correctly.*
2. **Day 1–2:** Write Ch5 Evaluation properly — fold in all five §4 findings. This is where the marks are; do it while fresh.
3. **Day 2:** Re-thread week2→Ch3, week1+week3→Ch4 (mostly cut/retense, not new writing).
4. **Day 3:** Ch1 Intro + Ch2 Background (you have the refs from the plan).
5. **Day 4:** Ch6 Discussion + Ch7 LSEP.
6. **Day 5:** Ch8 Conclusion + Abstract. Proofread every number against logs.
7. **Buffer / optional:** second model (Qwen-7B) if days 1–5 land early; otherwise caveat.

Do NOT spend the week chasing accuracy. 62% is a property of a 4-bit 3B model, not a bug. The reframe — "quantified distance to clinical safety" — is the mature move and it's already written into week4.

---

## 9. Small things

- `main.tex` still `\input`s the four `weekN` files. Once re-threaded, delete those inputs and the temporary comment block. Keep them in git history, not in the build.
- `\graphicspath` is set to `{../}{../results/figures/}` — figures are referenced as `results/figures/fig_x`. Works, but confirm in Overleaf (path resolution differs from local).
- No local TeX on the machine (per memory) — you're compiling in Overleaf. Make sure `numbers.tex` and `results/tables/*.tex` are uploaded, or the macros are undefined and the build dies. Consider committing a snapshot of these generated files specifically for the Overleaf upload.
- `bash.exe.stackdump` in repo root — stray crash artifact, gitignore or delete.
- The abstract should lead with the *finding* (selective beats always-retrieve at half the cost; closed-book ceiling quantifies distance to clinical safety), not the method. Examiners read it first.
