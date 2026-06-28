"""
scripts/run_threshold_sweep.py — Offline gate-threshold calibration via replay.

The P5 gate fired 0% retrieval at its default thresholds (entropy tau_H=2.5,
margin tau_M=0.3) because those defaults were never calibrated to *this* quantized
3B model. Re-running P5 per candidate threshold would cost hours of LLM time; we
don't need to. Every P5 run already logs the raw per-gate signal for every query
in `qvault.gate_details.signals` (entropy nats, margin probability gap, probe
token-F1). The retrieve/skip decision is a pure function of those signals and the
thresholds, so we **replay** the ensemble vote at any threshold offline — no LLM.

This script:
  - loads the logged signals from one or more P5 run logs (MCQ and/or open-ended),
  - sweeps entropy and margin thresholds over model-calibrated grids,
  - reports, per question type, the per-gate retrieval rate and the ensemble
    (majority >= min_votes) retrieval rate at each operating point,
  - prints suggested operating points at percentiles of the observed signal
    distribution (entropy p75, margin p25) — the starting points from the plan.

Output is a table the writeup uses directly as the calibration curve; the chosen
tau values then go into the P5 config for a single calibrated re-run.

Usage:
    python scripts/run_threshold_sweep.py \
        --logs results/raw_logs/p5_mirage200.jsonl:MCQ \
               results/raw_logs/pubmedqa_p5_open.jsonl:OPEN
"""

from __future__ import annotations

import argparse
import io
import json
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Console may default to cp1252 on Windows; force UTF-8 so em-dashes print.
sys.stdout.reconfigure(encoding="utf-8")

# Model-calibrated grids (observed ranges, not the impossible 1.0–5.0 defaults).
ENTROPY_GRID = [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.3]
MARGIN_GRID = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
PROBE_THRESHOLD = 0.7   # probe votes retrieve when draft-agreement F1 < this
MIN_VOTES = 2           # ensemble majority


def load_signals(spec: str) -> Tuple[str, List[Dict[str, float]]]:
    """Parse a `path:LABEL` spec, return (label, list of per-query signal dicts)."""
    path_str, _, label = spec.partition(":")
    label = label or Path(path_str).stem
    sigs: List[Dict[str, float]] = []
    for line in io.open(path_str, encoding="utf-8"):
        rec = json.loads(line)
        s = rec.get("qvault", {}).get("gate_details", {}).get("signals")
        if s and {"entropy", "margin", "hallucination_probe"} <= set(s):
            sigs.append(s)
    return label, sigs


def _pct(values: List[float], q: float) -> float:
    xs = sorted(values)
    if not xs:
        return float("nan")
    i = min(int(q * len(xs)), len(xs) - 1)
    return xs[i]


def gate_votes(sig: Dict[str, float], tau_h: float, tau_m: float) -> Dict[str, bool]:
    """Replay each gate's retrieve vote from a logged signal triple."""
    return {
        "entropy": sig["entropy"] > tau_h,            # high entropy -> retrieve
        "margin": sig["margin"] < tau_m,              # small margin -> retrieve
        "hallucination_probe": sig["hallucination_probe"] < PROBE_THRESHOLD,
    }


def ensemble_rate(sigs: List[Dict[str, float]], tau_h: float, tau_m: float) -> Dict[str, float]:
    """Per-gate + ensemble retrieval rate over a set of signals at (tau_H, tau_M)."""
    n = len(sigs)
    counts = {"entropy": 0, "margin": 0, "hallucination_probe": 0, "ensemble": 0}
    for sig in sigs:
        votes = gate_votes(sig, tau_h, tau_m)
        for g, v in votes.items():
            counts[g] += int(v)
        if sum(votes.values()) >= MIN_VOTES:
            counts["ensemble"] += 1
    return {g: c / n for g, c in counts.items()}


def describe(label: str, sigs: List[Dict[str, float]]) -> None:
    print(f"\n=== {label}: {len(sigs)} queries — signal distribution ===")
    for key in ("entropy", "margin", "hallucination_probe"):
        xs = [s[key] for s in sigs]
        print(f"  {key:20s} min={min(xs):.3f} p25={_pct(xs,0.25):.3f} "
              f"median={st.median(xs):.3f} p75={_pct(xs,0.75):.3f} max={max(xs):.3f}")
    # Suggested operating points (plan): entropy p75, margin p25.
    ent = [s["entropy"] for s in sigs]
    mar = [s["margin"] for s in sigs]
    print(f"  -> suggested tau_H (entropy p75) = {_pct(ent,0.75):.3f}")
    print(f"  -> suggested tau_M (margin  p25) = {_pct(mar,0.25):.3f}")


def sweep(label: str, sigs: List[Dict[str, float]]) -> None:
    print(f"\n=== {label}: ensemble retrieval-rate sweep (min_votes={MIN_VOTES}) ===")
    print("    rows = tau_H (entropy), cols = tau_M (margin); cell = ensemble retrieval rate")
    header = "  tau_H \\ tau_M | " + " ".join(f"{m:>5.2f}" for m in MARGIN_GRID)
    print(header)
    for tau_h in ENTROPY_GRID:
        cells = []
        for tau_m in MARGIN_GRID:
            r = ensemble_rate(sigs, tau_h, tau_m)["ensemble"]
            cells.append(f"{r:>5.0%}")
        print(f"  {tau_h:>9.2f} | " + " ".join(cells))
    # Per-gate solo rates at the suggested point for the ablation table.
    ent = [s["entropy"] for s in sigs]
    mar = [s["margin"] for s in sigs]
    th, tm = _pct(ent, 0.75), _pct(mar, 0.25)
    rates = ensemble_rate(sigs, th, tm)
    print(f"\n  At suggested (tau_H={th:.2f}, tau_M={tm:.2f}):")
    for g in ("entropy", "margin", "hallucination_probe", "ensemble"):
        print(f"    {g:20s} retrieval rate = {rates[g]:.0%}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline gate-threshold calibration.")
    ap.add_argument("--logs", nargs="+", required=True,
                    help="one or more `path:LABEL` P5 run logs")
    args = ap.parse_args()
    for spec in args.logs:
        label, sigs = load_signals(spec)
        if not sigs:
            print(f"[warn] {spec}: no usable signals")
            continue
        describe(label, sigs)
        sweep(label, sigs)


if __name__ == "__main__":
    main()
