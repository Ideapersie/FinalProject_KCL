r"""
scripts/generate_tables.py — Emit every LaTeX table and number in the report.

This is the other half of the "one source of truth" discipline. `generate_figures.py`
makes the plots; this makes the tables AND the numbers that appear in the prose.

The prose problem it solves: a table can be regenerated, but a sentence that says
"P5 reached 54\%" is just characters — nothing recomputes it, so it silently rots
the moment the scorer changes (which is exactly what happened: the report said 54%
while the logs said 54.5%). The fix is `numbers.tex`, a file of LaTeX macros:

    \newcommand{\accPfiveMcq}{54.5\%}

The chapter writes `\accPfiveMcq{}` and never a literal. A scorer change then
propagates into the *sentences*, not just the tables.

Everything is read through `evaluation.loading.load_run`, so tables, figures and
prose are computed from the same bytes by the same code path. Drift becomes
structurally impossible rather than merely currently-absent, and
`tests/regression/test_no_number_drift.py` fails the build if anyone hand-edits an
emitted file.

ROUNDING — one rule, applied everywhere (a number must not print as 50% in a table
and 49.5% in a macro):
    percentages  1 d.p.    token-F1  3 d.p.    latency  0 d.p. seconds

MACRO NAMES contain no digits: a LaTeX control sequence is letters only, so
\f1PfiveOpen parses as \f followed by the text '1PfiveOpen' and fails at both
definition and use. Token-F1 macros are therefore spelled \fOnePfiveOpen.

Usage:
    python scripts/generate_tables.py
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medrag_adaptive.evaluation.loading import load_run, load_run_with_report
from medrag_adaptive.evaluation.metrics import (
    aggregate_summary,
    proportion_diff_interval,
    risk_stratified_summary,
    DEFAULT_THRESHOLDS,
)

RAW = Path("results/raw_logs")
OUT = Path("results/tables")

# The four policies on the real corpus. P3 is daggered everywhere: it is
# closed-book, so its log is corpus-independent (verified: identical question-ids
# in identical order, zero retrievals). See test_p3_corpus_independence.py.
MCQ = {
    "P1 always-retrieve": "p1_medcorp_mcq.jsonl",
    "P4 hybrid":          "p4_medcorp_mcq.jsonl",
    "P3 closed-book$^\\dagger$": "p3_mirage200.jsonl",
    "P5 gated (calib.)":  "p5_medcorp_mcq.jsonl",
}
OPEN = {
    "P1 always-retrieve": "p1_medcorp_open.jsonl",
    "P3 closed-book$^\\dagger$": "pubmedqa_p3_open.jsonl",
    "P5 gated (calib.)":  "p5_medcorp_open.jsonl",
}

# ── Qwen2.5-7B: the scaling arm ────────────────────────────────────────
# Identical corpus, identical questions, identical policies; only the model
# changes, so a difference is attributable to scale and nothing else.
#
# Two open-ended P5 runs exist and serve different purposes, DO NOT conflate them:
#   p5_qwen7b_open_v2.jsonl (n=200) — the PROPER run, open-ended thresholds refitted
#       to a matched budget (entropy 0.568, margin 0.752, probe f1 0.538). This is
#       the scaling result.
#   p5_qwen7b_open.jsonl    (n=35)  — P5-open run using the MCQ thresholds unchanged.
#       Truncated at 35 because it saturated: every question retrieved (ensemble
#       100%), P5 collapsed into P1. Kept ONLY as the gate-saturation exhibit, and
#       any table that prints it must print its n and the "MCQ tau" caveat.
QWEN_MCQ = {
    "P1 always-retrieve": "p1_qwen7b_mcq.jsonl",   # graceful-skipped until it lands
    "P3 closed-book$^\\dagger$": "p3_qwen7b_mcq.jsonl",
    "P5 gated (recalib.)": "p5_qwen7b_mcq.jsonl",
}
QWEN_OPEN = {
    "P1 always-retrieve": "p1_qwen7b_open.jsonl",  # graceful-skipped until it lands
    "P3 closed-book$^\\dagger$": "p3_qwen7b_open.jsonl",
    "P5 gated (recalib.)": "p5_qwen7b_open_v2.jsonl",
}

# ── Qwen2.5-14B: the third scaling point ──────────────────────────────
# Same corpus, questions and policies again; only scale changes. P5-MCQ latency
# alone is read from the clean-profiled re-run (p5_qwen7b_mcq_clean is the 7B analogue);
# the 14B runs were profiled clean from the start (energy + memory tracking off).
QWEN14_MCQ = {
    "P1 always-retrieve": "p1_qwen14b_mcq.jsonl",
    "P3 closed-book$^\\dagger$": "p3_qwen14b_mcq.jsonl",
    "P5 gated (recalib.)": "p5_qwen14b_mcq.jsonl",
}
QWEN14_OPEN = {
    "P1 always-retrieve": "p1_qwen14b_open.jsonl",
    "P3 closed-book$^\\dagger$": "p3_qwen14b_open.jsonl",
    "P5 gated (recalib.)": "p5_qwen14b_open.jsonl",
}

# The three-model scaling ladder, in order. Used by every cross-model table/figure.
SCALES = [
    ("Llama-3.2-3B", MCQ,        OPEN),
    ("Qwen2.5-7B",   QWEN_MCQ,   QWEN_OPEN),
    ("Qwen2.5-14B",  QWEN14_MCQ, QWEN14_OPEN),
]

# The truncated MCQ-tau open run — saturation evidence only, never the scaling result.
QWEN_OPEN_SATURATED = "p5_qwen7b_open.jsonl"

# Per-member gate payloads, for the saturation and budget tables. Keyed by the
# member name used in GateDecision.details["members"]. The two Qwen open rows are
# the before/after of task-type recalibration: the MCQ-tau run saturates, the
# refitted run lands on budget.
GATE_RUNS = [
    ("Llama-3.2-3B",  "MCQ",  "p5_medcorp_mcq.jsonl"),
    ("Llama-3.2-3B",  "open", "p5_medcorp_open.jsonl"),
    ("Qwen2.5-7B",    "MCQ",  "p5_qwen7b_mcq.jsonl"),
    ("Qwen2.5-7B",    "open (MCQ $\\tau$)", QWEN_OPEN_SATURATED),
    ("Qwen2.5-7B",    "open (recalib.)",    "p5_qwen7b_open_v2.jsonl"),
    ("Qwen2.5-14B",   "MCQ",  "p5_qwen14b_mcq.jsonl"),
    ("Qwen2.5-14B",   "open", "p5_qwen14b_open.jsonl"),
]


def _present(mapping: dict) -> dict:
    """Drop entries whose log file is not on disk yet (e.g. P1 before it lands),
    so the tables regenerate cleanly both before and after the runs arrive."""
    return {k: v for k, v in mapping.items() if (RAW / v).exists()}


# ── formatting: the single rounding rule ──────────────────────────────

def pct(x: float) -> str:
    return "n/a" if x != x else f"{100 * x:.1f}\\%"


def f1(x: float) -> str:
    return f"{x:.3f}"


def secs(x: float) -> str:
    # 0 d.p. is the house rule, but it prints "0 s" for the GPU closed-book runs
    # (Qwen P3 MCQ is 0.1 s/question), which reads as a missing measurement rather
    # than a fast one. Sub-10-second values therefore keep one decimal.
    return f"{x:.1f}\\,s" if x < 10 else f"{x:.0f}\\,s"


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _header(script_rel: str) -> str:
    return (
        f"% AUTO-GENERATED by {script_rel} -- DO NOT EDIT.\n"
        f"% Regenerate with: python {script_rel}\n"
        f"% Every number is computed from results/raw_logs/ via evaluation.loading.\n"
        f"% git {_git_sha()} | {datetime.now():%Y-%m-%d %H:%M}\n"
    )


def write(name: str, body: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(_header("scripts/generate_tables.py") + body,
                            encoding="utf-8")
    print(f"[ok] {OUT / name}")


# ── tables ─────────────────────────────────────────────────────────────

def tab_main_results() -> None:
    """The core results table: all policies, both question types."""
    rows = []
    for label, fn in MCQ.items():
        s = aggregate_summary(load_run(RAW / fn))
        rows.append(f"    {label} & MCQ & {pct(s.accuracy)} & {pct(s.retrieval_rate)} "
                    f"& {secs(s.mean_latency_s)} \\\\")
    rows.append("    \\midrule")
    for label, fn in OPEN.items():
        s = aggregate_summary(load_run(RAW / fn))
        rows.append(f"    {label} & open & {f1(s.mean_f1)} & {pct(s.retrieval_rate)} "
                    f"& {secs(s.mean_latency_s)} \\\\")

    body = f"""\\begin{{table}}[t]
  \\centering
  \\begin{{tabular}}{{llccc}}
    \\toprule
    Policy & Type & Accuracy / F1 & Retrieval & Latency/q \\\\
    \\midrule
{chr(10).join(rows)}
    \\bottomrule
  \\end{{tabular}}
  \\caption{{All policies on the real MedCorp corpus (425{{,}}847 chunks), quantised
  Llama-3.2-3B. Multiple-choice is scored by accuracy; open-ended by mean token-F1
  (binary accuracy is not defined for open-ended answers -- see
  Section~\\ref{{sec:eval-ablation}}). $^{{\\dagger}}$P3 never retrieves, so its result is
  corpus-independent; it is a paired run over the identical question set.}}
  \\label{{tab:main-results}}
\\end{{table}}
"""
    write("tab_main_results.tex", body)


def tab_risk() -> None:
    """Risk-stratified accuracy with Wilson intervals — the small-n caveat."""
    rows = []
    for label, fn in MCQ.items():
        out = risk_stratified_summary(load_run(RAW / fn))
        cells = []
        for risk in ("low", "medium", "high"):
            c = out[risk]
            cells.append(f"{pct(c['accuracy'])} [{100*c['ci_low']:.0f}, {100*c['ci_high']:.0f}]")
        rows.append(f"    {label} & " + " & ".join(cells) + " \\\\")

    ns = risk_stratified_summary(load_run(RAW / MCQ["P5 gated (calib.)"]))
    body = f"""\\begin{{table}}[t]
  \\centering
  \\begin{{tabular}}{{lccc}}
    \\toprule
    & \\multicolumn{{3}}{{c}}{{Accuracy \\% [95\\% Wilson CI]}} \\\\
    \\cmidrule(lr){{2-4}}
    Policy & Low ($n={ns['low']['n']}$) & Medium ($n={ns['medium']['n']}$) & High ($n={ns['high']['n']}$) \\\\
    \\midrule
{chr(10).join(rows)}
    \\bottomrule
  \\end{{tabular}}
  \\caption{{Multiple-choice accuracy by clinical risk level, with 95\\% Wilson score
  intervals. The high-risk stratum contains only $n={ns['high']['n']}$ questions, giving intervals
  roughly $\\pm$30 percentage points wide: \\emph{{no comparison between policies at high
  risk is statistically meaningful in this evaluation}}. The medium stratum
  ($n={ns['medium']['n']}$) is where the headline comparison actually rests, and even there the
  intervals overlap.}}
  \\label{{tab:risk}}
\\end{{table}}
"""
    write("tab_risk.tex", body)


def tab_setup() -> None:
    """Policy definitions — the ruler, stated before anything is measured."""
    body = """\\begin{table}[t]
  \\centering
  \\begin{tabular}{llll}
    \\toprule
    Policy & Retrieval & Gate & Role in the evaluation \\\\
    \\midrule
    P1 always-retrieve & every query & n/a & retrieval baseline \\\\
    P4 hybrid & every query & n/a & BM25\\,+\\,dense fusion (RRF) \\\\
    P3 closed-book$^{\\dagger}$ & never & n/a & parametric-knowledge reference \\\\
    P5 gated & selective & 3-gate ensemble & the proposed method \\\\
    \\bottomrule
  \\end{tabular}
  \\caption{The four policies compared. P5 decides per query whether to retrieve,
  using a majority vote ($\\geq$2 of 3) over the token-entropy, logit-margin and
  hallucination-probe gates. $^{\\dagger}$P3 never retrieves and is therefore
  corpus-independent.}
  \\label{tab:setup}
\\end{table}
"""
    write("tab_setup.tex", body)


# ── scaling arm: Qwen2.5-7B ────────────────────────────────────────────

def _member_stats(records) -> dict:
    """Per-member gate signal, threshold and fire-rate from the ensemble payload.

    Reads GateDecision.details as the runner copied it into RunRecord.qvault.
    Each member reports its own signal under its own key, and the fire condition
    differs by member (entropy fires above tau, margin and probe below), so the
    comparison has to be done per member rather than generically.
    """
    out: dict = {}
    for rec in records:
        det = ((rec.get("qvault") or {}).get("gate_details") or {})
        for name, d in (det.get("members") or {}).items():
            slot = out.setdefault(name, {"sig": [], "thr": set(), "fire": 0, "n": 0})
            slot["n"] += 1
            if name == "entropy":
                v, t = d.get("mean_entropy"), d.get("threshold")
                fired = v is not None and t is not None and v > t
            elif name == "margin":
                v, t = d.get("mean_margin"), d.get("threshold")
                fired = v is not None and t is not None and v < t
            else:
                v, t = d.get("f1"), d.get("f1_threshold")
                fired = not d.get("agreement", True)
            if v is not None:
                slot["sig"].append(float(v))
            if t is not None:
                slot["thr"].add(round(float(t), 3))
            slot["fire"] += int(bool(fired))
    return out


def tab_scaling_results() -> None:
    """Both models, both question types — the scaling comparison."""
    rows = []
    for model, mcq, opn in SCALES:
        rows.append(f"    \\multicolumn{{6}}{{l}}{{\\textit{{{model}}}}} \\\\")
        for label, fn in _present(mcq).items():
            s = aggregate_summary(load_run(RAW / fn))
            rows.append(f"    \\quad {label} & MCQ & ${s.n}$ & {pct(s.accuracy)} "
                        f"& {pct(s.retrieval_rate)} & {secs(s.mean_latency_s)} \\\\")
        for label, fn in _present(opn).items():
            s = aggregate_summary(load_run(RAW / fn))
            n_cell = f"${s.n}$" if s.n >= 200 else f"$\\mathbf{{{s.n}}}$"
            rows.append(f"    \\quad {label} & open & {n_cell} & {f1(s.mean_f1)} "
                        f"& {pct(s.retrieval_rate)} & {secs(s.mean_latency_s)} \\\\")
        if model != SCALES[-1][0]:
            rows.append("    \\midrule")

    body = f"""\\begin{{table}}[t]
  \\centering
  \\begin{{tabular}}{{llcccc}}
    \\toprule
    Policy & Type & $n$ & Accuracy / F1 & Retrieval & Latency/q \\\\
    \\midrule
{chr(10).join(rows)}
    \\bottomrule
  \\end{{tabular}}
  \\caption{{The scaling arm. Identical corpus, identical questions and identical
  policy definitions across both models, so a difference is attributable to model
  scale. Multiple-choice is scored by accuracy, open-ended by mean token-F1.
  $^{{\\dagger}}$P3 never retrieves and is corpus-independent. The Qwen open-ended
  P5 row uses thresholds refitted for open-ended drafts (a single global operating
  point per task type); reusing the multiple-choice thresholds instead saturates the
  gate (Table~\\ref{{tab:gate-saturation}}). Retrieval budgets are \\emph{{not}}
  matched across models; see Table~\\ref{{tab:budget-match}}.}}
  \\label{{tab:scaling-results}}
\\end{{table}}
"""
    write("tab_scaling_results.tex", body)


def tab_gate_saturation() -> None:
    """Per-member signal ranges against their thresholds — the saturation exhibit."""
    order = ["entropy", "margin", "hallucination_probe"]
    pretty = {"entropy": "entropy", "margin": "margin",
              "hallucination_probe": "probe"}
    rows = []
    for model, kind, fn in GATE_RUNS:
        stats = _member_stats(load_run(RAW / fn))
        first = True
        for m in order:
            s = stats.get(m)
            if not s:
                continue
            label = f"{model} / {kind}" if first else ""
            first = False
            thr = ", ".join(f"{t:.3f}" for t in sorted(s["thr"])) or "n/a"
            if s["sig"]:
                lo, med, hi = (min(s["sig"]),
                               sorted(s["sig"])[len(s["sig"]) // 2],
                               max(s["sig"]))
                rng = f"{lo:.3f} / {med:.3f} / {hi:.3f}"
            else:
                rng = "n/a (threshold-free)"
            rate = s["fire"] / s["n"]
            cell = f"\\textbf{{{pct(rate)}}}" if rate in (0.0, 1.0) else pct(rate)
            rows.append(f"    {label} & {pretty[m]} & {thr} & {rng} & {cell} \\\\")
        rows.append("    \\midrule")
    rows = rows[:-1]

    body = f"""\\begin{{table}}[t]
  \\centering
  \\small
  \\begin{{tabular}}{{lllcc}}
    \\toprule
    Model / task & Gate & $\\tau$ & signal min / med / max & fires \\\\
    \\midrule
{chr(10).join(rows)}
    \\bottomrule
  \\end{{tabular}}
  \\caption{{Gate saturation. The entropy gate fires when its signal exceeds
  $\\tau$; the margin and probe gates fire when theirs fall below it. For
  Qwen on open-ended questions the entropy signal's \\emph{{minimum}} lies above
  its threshold and the margin signal's \\emph{{maximum}} lies below its own, so
  every question retrieves and P5 degenerates into P1. Both thresholds were fitted
  on multiple-choice drafts and reused unchanged on open-ended drafts, whose signal
  distributions differ; the probe is threshold-free on MCQ (letter agreement) and
  so reports no range there. Llama saturates less only because its $\\tau=0.70$
  happens to fall inside its open-ended distribution.}}
  \\label{{tab:gate-saturation}}
\\end{{table}}
"""
    write("tab_gate_saturation.tex", body)


def tab_budget_match() -> None:
    """Realised retrieval budget per model — the comparability condition."""
    rows = []
    for model, kind, fn in GATE_RUNS:
        recs = load_run(RAW / fn)
        s = aggregate_summary(recs)
        stats = _member_stats(recs)
        cells = []
        for m in ("entropy", "margin", "hallucination_probe"):
            st = stats.get(m)
            cells.append(pct(st["fire"] / st["n"]) if st else "n/a")
        rows.append(f"    {model} & {kind} & " + " & ".join(cells)
                    + f" & {pct(s.retrieval_rate)} \\\\")

    body = f"""\\begin{{table}}[t]
  \\centering
  \\begin{{tabular}}{{llcccc}}
    \\toprule
    & & \\multicolumn{{3}}{{c}}{{per-member vote rate}} & \\\\
    \\cmidrule(lr){{3-5}}
    Model & Task & entropy & margin & probe & ensemble \\\\
    \\midrule
{chr(10).join(rows)}
    \\bottomrule
  \\end{{tabular}}
  \\caption{{Realised retrieval budgets. Thresholds for Qwen were refitted to
  reproduce the 3B's per-gate budget, and the two tunable members landed close to
  target. The probe could not follow: on multiple-choice it is threshold-free
  (two drafts agree or they do not), so it cannot be recalibrated, and the larger
  model self-agrees far more often. With one member near-silent, a
  majority-of-three vote cannot reach the intended rate. The consequence is that
  the two models operate at different budgets, so ``selective retrieval beats
  always-retrieve at a matched budget'' is established \\emph{{within}} each model
  but not \\emph{{across}} them.}}
  \\label{{tab:budget-match}}
\\end{{table}}
"""
    write("tab_budget_match.tex", body)


# ── numbers.tex: the macros the prose uses ─────────────────────────────

def numbers() -> None:
    """Emit every number the chapter cites, as a LaTeX macro.

    The chapter must never contain a literal figure. This is what makes a scorer
    change propagate into the sentences rather than silently rotting them.
    """
    lines = []

    def mac(name: str, value: str) -> None:
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    # MCQ headline numbers
    summaries = {}
    for key, fn in [("Pone", "p1_medcorp_mcq.jsonl"), ("Pfour", "p4_medcorp_mcq.jsonl"),
                    ("Pthree", "p3_mirage200.jsonl"), ("Pfive", "p5_medcorp_mcq.jsonl")]:
        s = aggregate_summary(load_run(RAW / fn))
        summaries[key] = s
        mac(f"acc{key}Mcq", pct(s.accuracy))
        mac(f"ret{key}Mcq", pct(s.retrieval_rate))
        mac(f"lat{key}Mcq", secs(s.mean_latency_s))

    # Open-ended headline numbers
    for key, fn in [("Pone", "p1_medcorp_open.jsonl"), ("Pthree", "pubmedqa_p3_open.jsonl"),
                    ("Pfive", "p5_medcorp_open.jsonl")]:
        s = aggregate_summary(load_run(RAW / fn))
        mac(f"fOne{key}Open", f1(s.mean_f1))
        mac(f"ret{key}Open", pct(s.retrieval_rate))

    # THE headline claim, computed with its uncertainty rather than asserted.
    p5, p1 = summaries["Pfive"], summaries["Pone"]
    diff, lo, hi = proportion_diff_interval(
        round(p5.accuracy * p5.n), p5.n, round(p1.accuracy * p1.n), p1.n
    )
    mac("diffPfivePone", f"{100 * diff:.1f}")
    mac("diffPfivePoneCI", f"[{100 * lo:.1f}, {100 * hi:.1f}]")

    # Provenance: how many records the corrected extractor changed.
    _, rep = load_run_with_report(RAW / "p5_medcorp_mcq.jsonl")
    mac("rescoredPfive", str(rep.rescored))
    mac("nQuestions", str(rep.n))

    # Safety bars.
    for risk, bar in DEFAULT_THRESHOLDS.items():
        mac(f"bar{risk.capitalize()}", f"{100 * bar:.0f}\\%")

    # Distance to the nearest safety bar (P3 is the best policy on MCQ).
    gap = DEFAULT_THRESHOLDS["low"] - summaries["Pthree"].accuracy
    mac("gapToLowBar", f"{100 * gap:.1f}")

    # ── the scaling arm ────────────────────────────────────────────────
    # Accuracy and retrieval come from the original runs; the 7B P5-MCQ LATENCY comes
    # from the clean-profiled re-run (the original had memory tracking on and its
    # latency is unusable — accuracy is unaffected and stays on the original).
    LAT_OVERRIDE = {"p5_qwen7b_mcq.jsonl": "p5_qwen7b_mcq_clean.jsonl"}
    qwen = {}
    for key, fn in [("QwenPthree", "p3_qwen7b_mcq.jsonl"),
                    ("QwenPfive", "p5_qwen7b_mcq.jsonl")]:
        s = aggregate_summary(load_run(RAW / fn))
        qwen[key] = s
        mac(f"acc{key}Mcq", pct(s.accuracy))
        mac(f"ret{key}Mcq", pct(s.retrieval_rate))
        lat_fn = LAT_OVERRIDE.get(fn, fn)
        lat_s = aggregate_summary(load_run(RAW / lat_fn)) if (RAW / lat_fn).exists() else s
        mac(f"lat{key}Mcq", secs(lat_s.mean_latency_s))

    # Open-ended headline uses the PROPER run (v2, n=200, open-refit thresholds),
    # not the truncated MCQ-tau saturation run.
    for key, fn in [("QwenPthree", "p3_qwen7b_open.jsonl"),
                    ("QwenPfive", "p5_qwen7b_open_v2.jsonl")]:
        s = aggregate_summary(load_run(RAW / fn))
        mac(f"fOne{key}Open", f1(s.mean_f1))
        mac(f"ret{key}Open", pct(s.retrieval_rate))
        mac(f"n{key}Open", str(s.n))

    # Qwen P1 always-retrieve (both types) — the missing arm of "selective beats
    # always" at 7B. Graceful: emitted only once the runs land, so the build works
    # before and after. The chapter guards its P1 sentences with \ifdefined.
    if (RAW / "p1_qwen7b_mcq.jsonl").exists():
        s = aggregate_summary(load_run(RAW / "p1_qwen7b_mcq.jsonl"))
        mac("accQwenPoneMcq", pct(s.accuracy))
        mac("retQwenPoneMcq", pct(s.retrieval_rate))
    if (RAW / "p1_qwen7b_open.jsonl").exists():
        s = aggregate_summary(load_run(RAW / "p1_qwen7b_open.jsonl"))
        mac("fOneQwenPoneOpen", f1(s.mean_f1))
        mac("retQwenPoneOpen", pct(s.retrieval_rate))

    # ── the third scaling point: Qwen2.5-14B (graceful) ────────────────
    q14 = {}
    for key, fn in [("QwenFourteenPone", "p1_qwen14b_mcq.jsonl"),
                    ("QwenFourteenPthree", "p3_qwen14b_mcq.jsonl"),
                    ("QwenFourteenPfive", "p5_qwen14b_mcq.jsonl")]:
        if (RAW / fn).exists():
            s = aggregate_summary(load_run(RAW / fn))
            q14[key] = s
            mac(f"acc{key}Mcq", pct(s.accuracy))
            mac(f"ret{key}Mcq", pct(s.retrieval_rate))
            mac(f"lat{key}Mcq", secs(s.mean_latency_s))
    for key, fn in [("QwenFourteenPone", "p1_qwen14b_open.jsonl"),
                    ("QwenFourteenPthree", "p3_qwen14b_open.jsonl"),
                    ("QwenFourteenPfive", "p5_qwen14b_open.jsonl")]:
        if (RAW / fn).exists():
            s = aggregate_summary(load_run(RAW / fn))
            mac(f"fOne{key}Open", f1(s.mean_f1))
            mac(f"ret{key}Open", pct(s.retrieval_rate))

    # Selective-beats-always at 7B (P5 vs P1), with its interval. The 3B analogue is
    # diffPfivePone; the 14B is diffFourteenPfivePone below. All three cross-checked
    # together are what carry the claim.
    if (RAW / "p1_qwen7b_mcq.jsonl").exists():
        p5q, p1q = qwen["QwenPfive"], aggregate_summary(load_run(RAW / "p1_qwen7b_mcq.jsonl"))
        d, lo, hi = proportion_diff_interval(
            round(p5q.accuracy * p5q.n), p5q.n, round(p1q.accuracy * p1q.n), p1q.n)
        mac("diffQwenPfivePone", f"{100 * d:.1f}")
        mac("diffQwenPfivePoneCI", f"[{100 * lo:.1f}, {100 * hi:.1f}]")

    # Closed-book plateau: the 7B->14B gain is a fraction of the 3B->7B gain.
    if "QwenFourteenPthree" in q14:
        g1 = 100 * (qwen["QwenPthree"].accuracy - summaries["Pthree"].accuracy)   # 3B->7B
        g2 = 100 * (q14["QwenFourteenPthree"].accuracy - qwen["QwenPthree"].accuracy)  # 7B->14B
        mac("scaleGainSmall", f"{g1:.1f}")
        mac("scaleGainLarge", f"{g2:.1f}")

    # Selective-beats-always at 14B, with its interval — significant for the first time.
    if "QwenFourteenPfive" in q14 and "QwenFourteenPone" in q14:
        p5x, p1x = q14["QwenFourteenPfive"], q14["QwenFourteenPone"]
        d, lo, hi = proportion_diff_interval(
            round(p5x.accuracy * p5x.n), p5x.n, round(p1x.accuracy * p1x.n), p1x.n)
        mac("diffFourteenPfivePone", f"{100 * d:.1f}")
        mac("diffFourteenPfivePoneCI", f"[{100 * lo:.1f}, {100 * hi:.1f}]")

    # ── corpus-lacks-info vs distraction: paired P1-vs-P3 break/fix per model ──
    # BREAK = P3 correct, P1 wrong (retrieval broke it); FIX = the reverse. FIX>0
    # everywhere shows the corpus supplies answers; BREAK rising with scale shows
    # the mechanism is distraction, not corpus absence.
    def _break_fix(p3_fn, p1_fn):
        p3 = {r["question_id"]: bool(r["is_correct"]) for r in load_run(RAW / p3_fn)}
        p1 = {r["question_id"]: bool(r["is_correct"]) for r in load_run(RAW / p1_fn)}
        ids = set(p3) & set(p1)
        brk = sum(1 for i in ids if p3[i] and not p1[i])
        fix = sum(1 for i in ids if not p3[i] and p1[i])
        return brk, fix
    bf_specs = [("Llama", "p3_mirage200.jsonl", "p1_medcorp_mcq.jsonl"),
                ("QwenSeven", "p3_qwen7b_mcq.jsonl", "p1_qwen7b_mcq.jsonl"),
                ("QwenFourteen", "p3_qwen14b_mcq.jsonl", "p1_qwen14b_mcq.jsonl")]
    for tag, p3_fn, p1_fn in bf_specs:
        if (RAW / p3_fn).exists() and (RAW / p1_fn).exists():
            brk, fix = _break_fix(p3_fn, p1_fn)
            mac(f"break{tag}", str(brk))
            mac(f"fix{tag}", str(fix))
            mac(f"harmRatio{tag}", f"{brk/fix:.1f}" if fix else "$\\infty$")

    # Closed-book scale gap, with its interval — the one clean cross-model claim.
    q3, l3 = qwen["QwenPthree"], summaries["Pthree"]
    diff, lo, hi = proportion_diff_interval(
        round(q3.accuracy * q3.n), q3.n, round(l3.accuracy * l3.n), l3.n
    )
    mac("diffQwenLlamaMcq", f"{100 * diff:.1f}")
    mac("diffQwenLlamaMcqCI", f"[{100 * lo:.1f}, {100 * hi:.1f}]")

    # Paired McNemar on the identical 200 closed-book questions. A paired test is
    # the right one here: both models answered the same items, so the unpaired
    # interval above discards the pairing and is needlessly wide.
    l_by_id = {r["question_id"]: bool(r["is_correct"])
               for r in load_run(RAW / "p3_mirage200.jsonl")}
    q_by_id = {r["question_id"]: bool(r["is_correct"])
               for r in load_run(RAW / "p3_qwen7b_mcq.jsonl")}
    shared = set(l_by_id) & set(q_by_id)
    only_l = sum(1 for i in shared if l_by_id[i] and not q_by_id[i])
    only_q = sum(1 for i in shared if q_by_id[i] and not l_by_id[i])
    both = sum(1 for i in shared if q_by_id[i] and l_by_id[i])
    chi = ((abs(only_l - only_q) - 1) ** 2 / (only_l + only_q)) if (only_l + only_q) else 0.0
    mac("mcnemarOnlyLlama", str(only_l))
    mac("mcnemarOnlyQwen", str(only_q))
    mac("mcnemarBoth", str(both))
    mac("mcnemarNeither", str(len(shared) - both - only_l - only_q))
    mac("mcnemarChi", f"{chi:.2f}")
    mac("mcnemarN", str(len(shared)))

    # Gate saturation, stated as the numbers that make it undeniable.
    qo = _member_stats(load_run(RAW / "p5_qwen7b_open.jsonl"))
    if qo.get("entropy") and qo["entropy"]["sig"]:
        mac("qwenOpenEntropyMin", f"{min(qo['entropy']['sig']):.3f}")
        mac("qwenOpenEntropyTau", f"{sorted(qo['entropy']['thr'])[0]:.3f}")
    if qo.get("margin") and qo["margin"]["sig"]:
        mac("qwenOpenMarginMax", f"{max(qo['margin']['sig']):.3f}")
        mac("qwenOpenMarginTau", f"{sorted(qo['margin']['thr'])[0]:.3f}")

    # Probe collapse across scale — the cause of the budget miss.
    lm = _member_stats(load_run(RAW / "p5_medcorp_mcq.jsonl")).get("hallucination_probe")
    qm = _member_stats(load_run(RAW / "p5_qwen7b_mcq.jsonl")).get("hallucination_probe")
    if lm:
        mac("probeFireLlamaMcq", pct(lm["fire"] / lm["n"]))
    if qm:
        mac("probeFireQwenMcq", pct(qm["fire"] / qm["n"]))

    # What the 3B's own tau did on Qwen — the prospective non-transfer result.
    calib = aggregate_summary(load_run(RAW / "p5_qwen7b_calib.jsonl"))
    mac("retQwenCalibAtLlamaTau", pct(calib.retrieval_rate))

    # The budget gap that blocks the cross-model claim.
    mac("budgetGapMcq",
        f"{100 * (summaries['Pfive'].retrieval_rate - qwen['QwenPfive'].retrieval_rate):.1f}")

    # Cost of gating within each model (closed-book minus gated).
    mac("gatingCostLlamaMcq",
        f"{100 * (summaries['Pthree'].accuracy - summaries['Pfive'].accuracy):.1f}")
    mac("gatingCostQwenMcq",
        f"{100 * (qwen['QwenPthree'].accuracy - qwen['QwenPfive'].accuracy):.1f}")

    # Open-ended, restricted to the questions Qwen's truncated run actually covered.
    # Comparing the full 200 against a 35-question prefix would compare two
    # different question sets and attribute the difference to the model.
    qwen_open = load_run(RAW / "p5_qwen7b_open.jsonl")
    common = {r["question_id"] for r in qwen_open}

    def _f1_on(path, ids=None):
        sel = [r for r in load_run(RAW / path)
               if ids is None or r["question_id"] in ids]
        vals = [float(r.get("f1_score") or 0.0) for r in sel]
        if not vals:
            return 0.0, 0.0, 0
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1) if len(vals) > 1 else 0.0
        return mean, 1.96 * (var ** 0.5) / (len(vals) ** 0.5), len(vals)

    l_sub, l_ci, _ = _f1_on("p5_medcorp_open.jsonl", common)
    q_sub, q_ci, _ = _f1_on("p5_qwen7b_open.jsonl", common)
    mac("fOnePfiveOpenCommon", f1(l_sub))
    mac("fOneQwenPfiveOpenCommon", f1(q_sub))
    mac("ciPfiveOpenCommon", f1(l_ci))
    mac("ciQwenPfiveOpenCommon", f1(q_ci))
    mac("gapOpenCommon", f1(abs(q_sub - l_sub)))

    # How much harder the truncated prefix is than the full benchmark, measured
    # on Qwen's own COMPLETED closed-book run so the model is held constant.
    q3_full, _, _ = _f1_on("p3_qwen7b_open.jsonl")
    q3_sub, _, _ = _f1_on("p3_qwen7b_open.jsonl", common)
    mac("subsetBiasOpen", f1(q3_full - q3_sub))

    write("numbers.tex", "\n".join(lines) + "\n")


def main() -> None:
    tab_setup()
    tab_main_results()
    tab_risk()
    tab_scaling_results()
    tab_gate_saturation()
    tab_budget_match()
    numbers()
    print(f"\nTables written to {OUT}/")


if __name__ == "__main__":
    main()
