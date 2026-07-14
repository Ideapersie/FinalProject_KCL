"""
scripts/run_gate_ablation.py — Ablate the P5 gate ensemble from logged signals.

Answers the question an examiner will ask about the core novelty: "why three
gates rather than one?" It replays each single gate's retrieve/skip decision and
the ensemble's decision from the per-query signals already stored in
`qvault.gate_details` — so it needs NO new model runs.

It reports, per gate and for the ensemble:
  - retrieval rate  (how often that gate/ensemble would retrieve),
  - pairwise agreement between gates (do they even measure the same thing?),
  - swing analysis    (how often each gate is the deciding vote in the ensemble),
  - decision-consistent accuracy: on the subset of queries where a single gate's
    decision matches the ensemble's, the accuracy is exactly the logged accuracy;
    where a single gate would FLIP the decision we do not have that branch's
    answer, so those are reported as a flagged "would-flip" fraction rather than
    silently guessed. This keeps the ablation honest about what the logs support.

Usage:
    python scripts/run_gate_ablation.py --logs results/raw_logs/p5_medcorp_mcq.jsonl
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medrag_adaptive.evaluation.loading import load_run

sys.stdout.reconfigure(encoding="utf-8")

GATES = ["entropy", "margin", "hallucination_probe"]


def load(path: str) -> List[dict]:
    """Read through the canonical loader so the ablation's accuracy figures
    reconcile exactly with the tables and prose. See `evaluation/loading.py`."""
    return load_run(path)


def _gate_retrieves(votes: Dict[str, str], gate: str) -> bool:
    return votes.get(gate) == "retrieve"


def ablate(records: List[dict]) -> None:
    recs = [r for r in records
            if r.get("qvault", {}).get("gate_details", {}).get("votes")]
    n = len(recs)
    if not n:
        print("no gate_details.votes in these records"); return

    # Per-gate + ensemble retrieval rate.
    print(f"\n=== Gate ablation over {n} queries ===\n")
    print("Retrieval rate per gate (solo) and ensemble:")
    ens_ret = sum(1 for r in recs if r["retrieval_triggered"]) / n
    for g in GATES:
        rate = sum(_gate_retrieves(r["qvault"]["gate_details"]["votes"], g) for r in recs) / n
        print(f"  {g:20s} {rate:5.0%}")
    print(f"  {'ENSEMBLE (>=2)':20s} {ens_ret:5.0%}")

    # Pairwise agreement between gates' decisions.
    print("\nPairwise decision agreement (do gates agree on retrieve/skip?):")
    for a, b in combinations(GATES, 2):
        agree = sum(
            _gate_retrieves(r["qvault"]["gate_details"]["votes"], a)
            == _gate_retrieves(r["qvault"]["gate_details"]["votes"], b)
            for r in recs
        ) / n
        print(f"  {a} vs {b:20s} {agree:5.0%}")

    # Swing analysis: how often is each gate the deciding vote? (removing it
    # would change the >=2 majority outcome.)
    print("\nSwing analysis (gate is the deciding vote in the 2-of-3 majority):")
    for g in GATES:
        swing = 0
        for r in recs:
            votes = r["qvault"]["gate_details"]["votes"]
            total = sum(votes.get(x) == "retrieve" for x in GATES)
            with_g = _gate_retrieves(votes, g)
            without = total - (1 if with_g else 0)
            # ensemble decision flips if removing this gate crosses the >=2 line
            ens_with = total >= 2
            ens_without = without >= 2
            if ens_with != ens_without:
                swing += 1
        print(f"  {g:20s} decides {swing/n:5.0%} of queries")

    # Decision-consistent accuracy: where a solo gate == ensemble decision, the
    # logged accuracy applies; where it would flip, flag it.
    #
    # On OPEN-ENDED runs this is refused outright. The canonical loader voids
    # `is_correct` there because the logged flag is a soft floor (f1 > 0) AND is
    # inconsistent across runs (p3=0/200, p5=200/200). Emitting it would print
    # "100% accuracy" for every gate — a meaningless artifact that would be a
    # serious error to report. Mean token-F1 is the only defensible open-ended
    # metric; see the evaluation chapter.
    scorable = [r for r in recs if r.get("is_correct") is not None]
    if not scorable:
        print("\nSolo-gate decision-consistent accuracy: n/a (open-ended run).")
        print("  Refusing to compute: `is_correct` is a soft floor (f1>0) and is")
        print("  inconsistent across logs, so any accuracy from it is meaningless.")
        print("  Report mean token-F1 instead.")
        return

    print("\nSolo-gate vs ensemble decision consistency + accuracy where consistent:")
    ens_acc = sum(bool(r["is_correct"]) for r in recs) / n
    print(f"  {'ENSEMBLE':20s} acc {ens_acc:5.0%}  (all {n} queries)")
    for g in GATES:
        consistent = [r for r in recs
                      if _gate_retrieves(r["qvault"]["gate_details"]["votes"], g)
                      == r["retrieval_triggered"]]
        flips = n - len(consistent)
        acc = (sum(bool(r["is_correct"]) for r in consistent) / len(consistent)
               if consistent else 0.0)
        print(f"  {g:20s} acc {acc:5.0%}  on {len(consistent)}/{n} decision-consistent "
              f"(would flip {flips/n:.0%}, re-answer needed for those)")

    print("\nInterpretation: high pairwise disagreement + non-trivial swing shares "
          "\nmean the gates measure different things and the ensemble is not "
          "\nredundant — each contributes decisions the others would not make.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Ablate the P5 gate ensemble from logs.")
    ap.add_argument("--logs", nargs="+", required=True, help="P5 run log(s)")
    args = ap.parse_args()
    for path in args.logs:
        print(f"\n########## {path} ##########")
        ablate(load(path))


if __name__ == "__main__":
    main()
