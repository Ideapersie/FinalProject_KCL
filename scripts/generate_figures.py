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


def main() -> None:
    fig_pareto()
    fig_retrieval()
    fig_signals()
    fig_mcq_vs_open()
    fig_corpus_effect()
    fig_medcorp_policies()
    fig_safety_envelope()
    print(f"\nFigures written to {FIG}/")


if __name__ == "__main__":
    main()
