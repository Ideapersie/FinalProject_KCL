"""
evaluation/metrics.py — Turn per-query logs into report tables.

Operates on the RunRecord JSONL shape as plain dicts so it works directly on the
audit logs without re-instantiating dataclasses. Four outputs:

  aggregate_summary          accuracy / retrieval-rate / mean latency+energy for a
                             set of records (one policy×dataset×tier group).
  expected_calibration_error does the gate signal track correctness? (10-bin ECE).
  citation_precision_recall  of the sources the answer cited, how many were
                             actually retrieved (precision) and of the retrieved,
                             how many were cited (recall) — the P2 attribution metric.
  safety_envelope            THE headline contribution: for each risk level, the
                             cheapest policy whose accuracy clears the per-risk bar.

These are intentionally dependency-light (stdlib only) and pure functions, so the
report figures and the unit tests share exactly one implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

Record = Dict   # a RunRecord serialised to a dict (one JSONL line)


# ─────────────────────────────────────────────────────────────────────
# Small-sample confidence intervals (used for the risk-stratified caveat)
# ─────────────────────────────────────────────────────────────────────

def wilson_interval(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95%% Wilson score interval for a binomial proportion.

    Used for the risk-stratified accuracy cells, where n is tiny (e.g. only 6
    high-risk questions). The normal-approximation interval is invalid at that
    n; the Wilson interval stays inside [0,1] and does not collapse to a point
    when the count is 0 or n. Returned as (low, high). n==0 -> (0.0, 1.0).
    """
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, centre - half), min(1.0, centre + half)


def risk_stratified_summary(records: Sequence[Record]) -> Dict[str, dict]:
    """Per-risk accuracy with n and a 95%% Wilson interval, for one policy's records.

    Makes the small-n problem explicit in the numbers themselves: the high-risk
    cell (typically n=6) comes back with a wide interval, so the report can state
    the point estimate *and* its uncertainty rather than over-claiming.
    """
    out: Dict[str, dict] = {}
    for risk in ("low", "medium", "high"):
        recs = [r for r in records if r.get("risk_level") == risk]
        n = len(recs)
        succ = sum(1 for r in recs if r.get("is_correct"))
        lo, hi = wilson_interval(succ, n)
        out[risk] = {
            "n": n,
            "accuracy": (succ / n) if n else float("nan"),
            "ci_low": lo,
            "ci_high": hi,
        }
    return out


# ─────────────────────────────────────────────────────────────────────
# Group summary
# ─────────────────────────────────────────────────────────────────────

@dataclass
class PolicySummary:
    """Aggregate metrics for one group of records (e.g. one policy on one set)."""
    n: int
    accuracy: float
    retrieval_rate: float
    mean_f1: float
    mean_latency_s: float
    mean_energy_kwh: float


def aggregate_summary(records: Sequence[Record]) -> PolicySummary:
    """Mean accuracy / retrieval-rate / latency / energy over `records`."""
    n = len(records)
    if n == 0:
        return PolicySummary(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    acc = sum(bool(r.get("is_correct")) for r in records) / n
    ret = sum(bool(r.get("retrieval_triggered")) for r in records) / n
    f1 = sum(float(r.get("f1_score") or 0.0) for r in records) / n
    lat = sum(float(r.get("latency_ns") or 0) for r in records) / n / 1e9
    eng = sum(float(r.get("energy_kwh") or 0.0) for r in records) / n
    return PolicySummary(n, acc, ret, f1, lat, eng)


# ─────────────────────────────────────────────────────────────────────
# Expected Calibration Error
# ─────────────────────────────────────────────────────────────────────

def expected_calibration_error(records: Sequence[Record], n_bins: int = 10) -> float:
    """
    ECE between the gate signal (used as a confidence proxy) and correctness.

    Records with no gate_signal_value are skipped (closed-book has no signal).
    The signal is treated as a confidence in [0,1]; each record falls in one of
    n_bins. ECE = Σ_b (|b|/N) · |acc(b) − conf(b)|. Returns 0.0 when no record
    carries a usable signal (nothing to calibrate).
    """
    scored = [r for r in records if r.get("gate_signal_value") is not None]
    if not scored:
        return 0.0
    n = len(scored)
    bins: List[List[Tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for r in scored:
        conf = min(max(float(r["gate_signal_value"]), 0.0), 1.0)
        idx = min(int(conf * n_bins), n_bins - 1)
        bins[idx].append((conf, bool(r.get("is_correct"))))

    ece = 0.0
    for b in bins:
        if not b:
            continue
        acc = sum(correct for _, correct in b) / len(b)
        conf = sum(c for c, _ in b) / len(b)
        ece += (len(b) / n) * abs(acc - conf)
    return ece


# ─────────────────────────────────────────────────────────────────────
# Citation precision / recall
# ─────────────────────────────────────────────────────────────────────

def _retrieved_ids(rec: Record) -> List[str]:
    """Retrieved chunk ids — top-level field, else from qvault, else []."""
    if rec.get("retrieved_chunk_ids"):
        return list(rec["retrieved_chunk_ids"])
    qv = rec.get("qvault") or {}
    return list(qv.get("retrieved_chunk_ids", []))


def citation_precision_recall(rec: Record) -> Tuple[float, float]:
    """
    Precision = |cited ∩ retrieved| / |cited|   (did it cite real sources?)
    Recall    = |cited ∩ retrieved| / |retrieved| (did it cite what it had?)

    Both 0.0 when there are no citations. Recall is 0.0 when nothing was retrieved.
    """
    cited = {c.get("chunk_id") for c in (rec.get("citations") or []) if c.get("chunk_id")}
    if not cited:
        return 0.0, 0.0
    retrieved = set(_retrieved_ids(rec))
    hit = len(cited & retrieved)
    precision = hit / len(cited)
    recall = hit / len(retrieved) if retrieved else 0.0
    return precision, recall


# ─────────────────────────────────────────────────────────────────────
# Safety envelope (headline contribution)
# ─────────────────────────────────────────────────────────────────────

# Default per-risk accuracy bars (project spec): low 0.70, med 0.80, high 0.90.
DEFAULT_THRESHOLDS = {"low": 0.70, "medium": 0.80, "high": 0.90}


@dataclass
class EnvelopeEntry:
    """The chosen policy for one risk level and why."""
    policy_name: str
    accuracy: float
    mean_latency_s: float
    n: int


def safety_envelope(
    records: Sequence[Record],
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Optional[EnvelopeEntry]]:
    """
    For each risk level, choose the *cheapest* policy whose accuracy on that
    risk level clears the bar. "Cheapest" = lowest mean latency (the CPU cost
    proxy). Returns {risk_level: EnvelopeEntry | None}; None when no policy on
    that risk level meets the threshold.

    This is the project's publishable output: it answers "what is the least
    expensive policy I can safely run at each clinical risk level?" — directly
    coupling the accuracy/cost trade-off to a safety requirement.
    """
    thr = thresholds if thresholds is not None else DEFAULT_THRESHOLDS
    out: Dict[str, Optional[EnvelopeEntry]] = {}

    for risk, bar in thr.items():
        # group this risk level's records by policy
        by_policy: Dict[str, List[Record]] = {}
        for r in records:
            if r.get("risk_level") == risk:
                by_policy.setdefault(r.get("policy_name", "?"), []).append(r)

        candidates: List[EnvelopeEntry] = []
        for policy, recs in by_policy.items():
            s = aggregate_summary(recs)
            if s.accuracy >= bar:
                candidates.append(
                    EnvelopeEntry(policy, s.accuracy, s.mean_latency_s, s.n)
                )

        # cheapest passing policy = lowest mean latency
        out[risk] = min(candidates, key=lambda e: e.mean_latency_s) if candidates else None

    return out
