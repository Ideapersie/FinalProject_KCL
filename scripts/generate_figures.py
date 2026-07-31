"""
scripts/generate_figures.py — Build the report figures from the raw JSONL logs.

Reads results/raw_logs/*.jsonl and writes PNG+PDF to results/figures/. Every
figure is derived only from logged records via evaluation.metrics, so the plots
and the report tables can never disagree. Four figures:

  fig_pareto        accuracy vs mean latency per policy (the cost/quality frontier)
  fig_retrieval     retrieval rate uncalibrated vs calibrated (the calibration story)
  fig_signals       gate-signal distributions vs the (impossible) default thresholds
  fig_mcq_vs_open   P3/P5 on MCQ (accuracy) and open-ended (token-F1) side by side

Run:  python scripts/generate_figures.py
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")  # headless: write files, open no window
import matplotlib.pyplot as plt

from medrag_adaptive.evaluation.metrics import aggregate_summary
from medrag_adaptive.evaluation.loading import load_run

RAW = Path("results/raw_logs")
FIG = Path("results/figures")


def load(name: str) -> List[dict]:
    """Read a log through the canonical loader — the single source of truth.

    `load_run` re-scores MCQ records with the current extractor and voids the
    unusable open-ended `is_correct`, so every figure here plots exactly the
    numbers the tables and the prose report. See `evaluation/loading.py`.
    """
    return load_run(RAW / name)


def _save(fig, stem: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{stem}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[ok] {stem}.png / .pdf")


# ── fig 1: Pareto — accuracy vs latency (MCQ policies) ───────────────

def fig_pareto() -> None:
    points = {
        "P1 always (pilot)":  ("p1_mirage200.jsonl", "tab:red"),
        "P3 closed-book":     ("p3_mirage200.jsonl", "tab:green"),
        "P5 gated (uncalib)": ("p5_mirage200.jsonl", "tab:gray"),
        "P5 gated (calib)":   ("p5_mirage200_calibrated.jsonl", "tab:blue"),
    }
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for label, (fname, colour) in points.items():
        try:
            s = aggregate_summary(load(fname))
        except FileNotFoundError:
            continue
        ax.scatter(s.mean_latency_s, s.accuracy * 100, s=90, color=colour, zorder=3)
        ax.annotate(label, (s.mean_latency_s, s.accuracy * 100),
                    textcoords="offset points", xytext=(8, 4), fontsize=8)
    ax.set_xlabel("Mean latency per query (s)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs latency on 200 MIRAGE MCQs (pilot corpus)")
    ax.grid(True, alpha=0.3)
    _save(fig, "fig_pareto")


# ── fig 2: retrieval rate uncalibrated vs calibrated ─────────────────

def fig_retrieval() -> None:
    bars = {
        "P5 MCQ\n(uncalib)":  "p5_mirage200.jsonl",
        "P5 MCQ\n(calib)":    "p5_mirage200_calibrated.jsonl",
        "P5 open\n(calib)":   "pubmedqa_p5_open.jsonl",
    }
    labels, rates = [], []
    for label, fname in bars.items():
        try:
            rates.append(aggregate_summary(load(fname)).retrieval_rate * 100)
            labels.append(label)
        except FileNotFoundError:
            continue
    fig, ax = plt.subplots(figsize=(5.5, 4))
    colours = ["tab:gray", "tab:blue", "tab:cyan"][: len(labels)]
    ax.bar(labels, rates, color=colours)
    for i, r in enumerate(rates):
        ax.text(i, r + 1.5, f"{r:.0f}%", ha="center", fontsize=9)
    ax.axhspan(40, 70, alpha=0.12, color="green")  # target operating band
    ax.set_ylabel("Retrieval rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Gate retrieval rate: default vs calibrated thresholds")
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, "fig_retrieval")


# ── fig 3: gate-signal distributions vs default thresholds ───────────

def _signals(fname: str, key: str) -> List[float]:
    out = []
    for rec in load(fname):
        s = rec.get("qvault", {}).get("gate_details", {}).get("signals", {})
        if key in s:
            out.append(s[key])
    return out


def fig_signals() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, key, default, calib, name in [
        (axes[0], "entropy", 2.5, 0.70, "Entropy (nats)"),
        (axes[1], "margin", 0.30, 0.70, "Margin (prob gap)"),
    ]:
        vals = _signals("p5_mirage200.jsonl", key)
        if vals:
            ax.hist(vals, bins=20, color="tab:blue", alpha=0.7)
        ax.axvline(default, color="tab:red", ls="--", label=f"default τ={default}")
        ax.axvline(calib, color="tab:green", ls="-", label=f"calibrated τ={calib}")
        ax.set_title(name)
        ax.set_ylabel("count")
        ax.legend(fontsize=7)
    fig.suptitle("Why defaults never fired: signal ranges vs thresholds (MCQ)")
    fig.tight_layout()
    _save(fig, "fig_signals")


# ── fig 4: MCQ vs open-ended for P3 / P5 ─────────────────────────────

def fig_mcq_vs_open() -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.8))
    # MCQ accuracy
    mcq = {"P3": "p3_mirage200.jsonl", "P5 (calib)": "p5_mirage200_calibrated.jsonl"}
    m_lab, m_val = [], []
    for k, f in mcq.items():
        try:
            m_val.append(aggregate_summary(load(f)).accuracy * 100); m_lab.append(k)
        except FileNotFoundError:
            pass
    a1.bar(m_lab, m_val, color=["tab:green", "tab:blue"][: len(m_lab)])
    for i, v in enumerate(m_val):
        a1.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=9)
    a1.set_ylabel("Accuracy (%)"); a1.set_title("MCQ (MIRAGE)"); a1.set_ylim(0, 80)
    # open-ended F1
    opn = {"P3": "pubmedqa_p3_open.jsonl", "P5 (calib)": "pubmedqa_p5_open.jsonl"}
    o_lab, o_val = [], []
    for k, f in opn.items():
        try:
            o_val.append(aggregate_summary(load(f)).mean_f1); o_lab.append(k)
        except FileNotFoundError:
            pass
    a2.bar(o_lab, o_val, color=["tab:green", "tab:blue"][: len(o_lab)])
    for i, v in enumerate(o_val):
        a2.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    a2.set_ylabel("Mean token-F1"); a2.set_title("Open-ended (PubMedQA)")
    fig.suptitle("P3 vs calibrated P5 by question type (pilot corpus)")
    fig.tight_layout()
    _save(fig, "fig_mcq_vs_open")


# ── fig 5: pilot vs real corpus (the corpus-quality finding) ─────────

def fig_corpus_effect() -> None:
    """Accuracy/F1 before (pilot) and after (MedCorp) the real corpus."""
    # (label, pilot_file, medcorp_file, metric)
    groups = [
        ("P1 MCQ",  "p1_mirage200.jsonl",       "p1_medcorp_mcq.jsonl",  "acc"),
        ("P5 MCQ",  "p5_mirage200_calibrated.jsonl", "p5_medcorp_mcq.jsonl", "acc"),
        ("P5 open", "pubmedqa_p5_open.jsonl",    "p5_medcorp_open.jsonl",  "f1"),
    ]
    fig, (a_acc, a_f1) = plt.subplots(1, 2, figsize=(9, 3.8))
    import numpy as np
    acc_labels, acc_pilot, acc_med = [], [], []
    f1_labels, f1_pilot, f1_med = [], [], []
    for label, pf, mf, metric in groups:
        try:
            sp, sm = aggregate_summary(load(pf)), aggregate_summary(load(mf))
        except FileNotFoundError:
            continue
        if metric == "acc":
            acc_labels.append(label); acc_pilot.append(sp.accuracy * 100); acc_med.append(sm.accuracy * 100)
        else:
            f1_labels.append(label); f1_pilot.append(sp.mean_f1); f1_med.append(sm.mean_f1)
    for ax, labs, pilot, med, ylab in [
        (a_acc, acc_labels, acc_pilot, acc_med, "Accuracy (%)"),
        (a_f1, f1_labels, f1_pilot, f1_med, "Mean token-F1"),
    ]:
        x = np.arange(len(labs)); w = 0.38
        ax.bar(x - w/2, pilot, w, label="pilot (200)", color="tab:gray")
        ax.bar(x + w/2, med, w, label="MedCorp (425K)", color="tab:blue")
        ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_ylabel(ylab); ax.legend(fontsize=7)
    fig.suptitle("Effect of the real corpus: pilot vs MedCorp")
    fig.tight_layout()
    _save(fig, "fig_corpus_effect")


# ── fig 6: real-corpus policy comparison (selective beats always) ────

def fig_medcorp_policies() -> None:
    mcq = {"P1 always": "p1_medcorp_mcq.jsonl", "P4 hybrid": "p4_medcorp_mcq.jsonl",
           "P3 closed": "p3_mirage200.jsonl", "P5 gated": "p5_medcorp_mcq.jsonl"}
    labels, accs, rets = [], [], []
    for k, f in mcq.items():
        try:
            s = aggregate_summary(load(f))
        except FileNotFoundError:
            continue
        labels.append(k); accs.append(s.accuracy * 100); rets.append(s.retrieval_rate * 100)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    colours = ["tab:red", "tab:orange", "tab:green", "tab:blue"][: len(labels)]
    bars = ax.bar(labels, accs, color=colours)
    for b, a, r in zip(bars, accs, rets):
        ax.text(b.get_x() + b.get_width()/2, a + 1, f"{a:.0f}%\n(ret {r:.0f}%)",
                ha="center", fontsize=8)
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 75)
    ax.set_title("MCQ accuracy by policy on the real corpus (MedCorp)")
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, "fig_medcorp_policies")


# ── fig 7: safety envelope — accuracy by risk vs the safety bars ─────

def fig_safety_envelope() -> None:
    """Per-policy accuracy at each risk level against the clinical safety bars."""
    import numpy as np
    from collections import defaultdict
    policies = {"P1 always": "p1_medcorp_mcq.jsonl", "P4 hybrid": "p4_medcorp_mcq.jsonl",
                "P3 closed": "p3_mirage200.jsonl", "P5 gated": "p5_medcorp_mcq.jsonl"}
    risks = ["low", "medium", "high"]
    bars = {"low": 70, "medium": 80, "high": 90}
    # acc[policy][risk]
    acc: Dict[str, Dict[str, float]] = {}
    for pol, f in policies.items():
        try:
            rows = load(f)
        except FileNotFoundError:
            continue
        byr = defaultdict(list)
        for r in rows:
            byr[r.get("risk_level")].append(r)
        acc[pol] = {rk: (sum(x["is_correct"] for x in byr[rk]) / len(byr[rk]) * 100
                         if byr.get(rk) else 0.0) for rk in risks}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(risks)); w = 0.2
    colours = {"P1 always": "tab:red", "P4 hybrid": "tab:orange",
               "P3 closed": "tab:green", "P5 gated": "tab:blue"}
    for i, (pol, per_risk) in enumerate(acc.items()):
        ax.bar(x + (i - 1.5) * w, [per_risk[r] for r in risks], w,
               label=pol, color=colours.get(pol))
    # safety bars as horizontal target lines
    for j, r in enumerate(risks):
        ax.hlines(bars[r], x[j] - 2*w, x[j] + 2*w, colors="black", linestyles="--", lw=1.5)
        ax.text(x[j], bars[r] + 1, f"bar {bars[r]}%", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f"{r}\nrisk" for r in risks])
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 100)
    ax.set_title("Safety envelope: accuracy by risk level vs clinical bars (real corpus, MCQ)")
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, "fig_safety_envelope")


# ── the scaling arm: Qwen2.5-7B vs Llama-3.2-3B ──────────────────────
#
# Two-series categorical palette, fixed order (Llama, Qwen), never cycled.
# Validated for colour-vision deficiency rather than eyeballed: worst adjacent
# separation is deutan dE 11.0, normal-vision dE 25.8, both above the floor.
# The red/green pairing used elsewhere in this file is not CVD-safe and is not
# reused here.
C_LLAMA = "#0072B2"   # blue
C_QWEN = "#D55E00"    # vermillion
C_TASK_MCQ = "#0072B2"
C_TASK_OPEN = "#D55E00"

MEMBER_LABEL = {"entropy": "entropy  (fires above $\\tau$)",
                "margin": "margin  (fires below $\\tau$)",
                "hallucination_probe": "probe  (fires below $\\tau$)"}


def _member_signals(fname: str) -> Dict[str, dict]:
    """Per-member signal values and threshold, from the ensemble payload.

    Reads `members` rather than `signals` because only the former carries each
    member's own threshold, and the saturation figure is precisely about where
    the threshold sits relative to the signal.
    """
    out: Dict[str, dict] = {}
    for rec in load(fname):
        det = (rec.get("qvault") or {}).get("gate_details") or {}
        for name, d in (det.get("members") or {}).items():
            slot = out.setdefault(name, {"sig": [], "tau": None})
            v = (d.get("mean_entropy") if name == "entropy" else
                 d.get("mean_margin") if name == "margin" else d.get("f1"))
            t = (d.get("threshold") if name in ("entropy", "margin")
                 else d.get("f1_threshold"))
            if v is not None:
                slot["sig"].append(float(v))
            if t is not None:
                slot["tau"] = float(t)
    return out


def fig_scaling() -> None:
    """Accuracy against the retrieval budget actually spent, for both models.

    One axis, one measure. The x-position is the realised retrieval rate rather
    than a policy name, which is what makes the budget mismatch visible: the two
    P5 points do not sit above one another.
    """
    series = [
        ("Llama-3.2-3B", C_LLAMA, [
            ("P3", "p3_mirage200.jsonl"),
            ("P5", "p5_medcorp_mcq.jsonl"),
            ("P1", "p1_medcorp_mcq.jsonl"),
        ]),
        ("Qwen2.5-7B", C_QWEN, [
            ("P3", "p3_qwen7b_mcq.jsonl"),
            ("P5", "p5_qwen7b_mcq.jsonl"),
            ("P1", "p1_qwen7b_mcq.jsonl"),
        ]),
    ]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for label, colour, runs in series:
        xs, ys = [], []
        for name, fn in runs:
            s = aggregate_summary(load(fn))
            xs.append(100 * s.retrieval_rate)
            ys.append(100 * s.accuracy)
            ax.annotate(name, (xs[-1], ys[-1]), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=8, color="#333333")
        ax.plot(xs, ys, "-o", color=colour, label=label, linewidth=2,
                markersize=8, markeredgecolor="white", markeredgewidth=1.5)

    ax.set_xlabel("retrieval budget actually spent (\\% of queries)"
                  if plt.rcParams["text.usetex"] else
                  "retrieval budget actually spent (% of queries)")
    ax.set_ylabel("MCQ accuracy (%)")
    ax.set_title("Retrieval costs accuracy on this corpus — for both model scales")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    _save(fig, "fig_scaling")


def fig_gate_saturation() -> None:
    """Where each gate's threshold sits relative to the signal it thresholds.

    One panel per member, because the three signals are in different units and
    must not share an axis. A threshold drawn outside its own range bar is a
    gate that fires on every question, or on none.
    """
    runs = [
        ("Llama / MCQ", "p5_medcorp_mcq.jsonl", C_LLAMA),
        ("Llama / open", "p5_medcorp_open.jsonl", C_LLAMA),
        ("Qwen / MCQ", "p5_qwen7b_mcq.jsonl", C_QWEN),
        ("Qwen / open", "p5_qwen7b_open.jsonl", C_QWEN),
    ]
    members = ["entropy", "margin", "hallucination_probe"]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, member in zip(axes, members):
        ypos, labels = [], []
        for i, (label, fn, colour) in enumerate(runs):
            st = _member_signals(fn).get(member)
            y = len(runs) - i
            ypos.append(y)
            labels.append(label)
            if not st or not st["sig"]:
                ax.text(0.5, y, "threshold-free", transform=ax.get_yaxis_transform(),
                        va="center", ha="center", fontsize=7, color="#888888",
                        style="italic")
                continue
            lo, hi = min(st["sig"]), max(st["sig"])
            med = sorted(st["sig"])[len(st["sig"]) // 2]
            # plot() rather than hlines(): LineCollection has no solid_capstyle,
            # and a rounded data-end is the house mark spec.
            ax.plot([lo, hi], [y, y], color=colour, linewidth=6, alpha=0.35,
                    solid_capstyle="round", zorder=2)
            ax.plot([med], [y], "o", color=colour, markersize=7,
                    markeredgecolor="white", markeredgewidth=1.2, zorder=3)
            if st["tau"] is not None:
                outside = not (lo <= st["tau"] <= hi)
                ax.plot([st["tau"]], [y], marker="|", markersize=16, zorder=4,
                        color="#111111" if not outside else "#B00020",
                        markeredgewidth=2.2)
                if outside:
                    ax.annotate("$\\tau$ outside range", (st["tau"], y),
                                textcoords="offset points", xytext=(6, 10),
                                fontsize=7, color="#B00020")
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(MEMBER_LABEL[member], fontsize=9)
        ax.grid(True, axis="x", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        # Fixed y-limits, not autoscale: the probe panel has data on only two of
        # the four rows, so autoscaling would drop the empty rows off the axes —
        # taking their tick labels and their "threshold-free" note with them.
        ax.set_ylim(0.4, len(runs) + 0.6)

    # Identity is not carried by colour alone: every row is labelled on the axis.
    fig.suptitle("Gate thresholds against the signals they threshold "
                 "(bar = min–max, dot = median, tick = $\\tau$)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "fig_gate_saturation")


def fig_signal_shift() -> None:
    """Why an MCQ-fitted threshold cannot be reused on open-ended questions.

    Same model, same gate, two task types. The distributions barely overlap, so a
    threshold placed on one lands in the wrong part of the other.
    """
    panels = [
        ("Llama-3.2-3B", "entropy", "p5_medcorp_mcq.jsonl", "p5_medcorp_open.jsonl"),
        ("Llama-3.2-3B", "margin", "p5_medcorp_mcq.jsonl", "p5_medcorp_open.jsonl"),
        ("Qwen2.5-7B", "entropy", "p5_qwen7b_mcq.jsonl", "p5_qwen7b_open.jsonl"),
        ("Qwen2.5-7B", "margin", "p5_qwen7b_mcq.jsonl", "p5_qwen7b_open.jsonl"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.0))
    for ax, (model, member, mcq_fn, open_fn) in zip(axes.ravel(), panels):
        mcq = _member_signals(mcq_fn).get(member, {"sig": [], "tau": None})
        opn = _member_signals(open_fn).get(member, {"sig": [], "tau": None})
        bins = 18
        if mcq["sig"]:
            ax.hist(mcq["sig"], bins=bins, color=C_TASK_MCQ, alpha=0.55,
                    label="MCQ", density=True)
        if opn["sig"]:
            ax.hist(opn["sig"], bins=bins, color=C_TASK_OPEN, alpha=0.55,
                    label="open-ended", density=True)
        if mcq["tau"] is not None:
            ax.axvline(mcq["tau"], color="#111111", linestyle="--", linewidth=1.6,
                       label=f"$\\tau$ fitted on MCQ = {mcq['tau']:.3f}")
        ax.set_title(f"{model} — {member}", fontsize=9)
        ax.set_ylabel("density", fontsize=8)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Gate signals move with the task, so an MCQ-fitted $\\tau$ "
                 "does not transfer to open-ended", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, "fig_signal_shift")


def main() -> None:
    fig_pareto()
    fig_retrieval()
    fig_signals()
    fig_mcq_vs_open()
    fig_corpus_effect()
    fig_medcorp_policies()
    fig_safety_envelope()
    fig_scaling()
    fig_gate_saturation()
    fig_signal_shift()
    print(f"\nFigures written to {FIG}/")


if __name__ == "__main__":
    main()
