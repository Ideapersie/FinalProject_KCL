r"""
scripts/make_calibration_set.py — build a stride-sampled calibration subset.

Why this exists
---------------
Gate thresholds are fitted by replaying logged signals, but harvesting those
signals costs LLM time: P5 makes ~5 generations per question. Calibrating a new
model on all 200 evaluation questions would cost as much as the evaluation run
itself, which defeats the point of doing it *before* the long run.

The obvious shortcut — take the first N questions — is wrong here. `load_mirage`
concatenates subsets in a fixed order and the evaluation set is 200 MMLU
questions ordered by subject, so a prefix is one or two subjects rather than a
sample of the benchmark. Measured on the 3B's own signals
(`results/raw_logs/p5_medcorp_mcq.jsonl`), fitting tau on a prefix and applying it
to all 200 overshoots the retrieval budget, and the error does *not* shrink with
N because more prefix is still the same subjects:

    N=20 prefix -> +4.0pp     N=20 stride -> +1.5pp
    N=40 prefix -> +6.5pp     N=40 stride -> +3.0pp
    N=50 prefix -> +6.5pp     N=50 stride -> +1.0pp

A stride sample (every k-th question) spans every subject and tracks the full-set
operating point closely. N=50 is the default: ~1pp budget error for ~12 minutes of
7B GPU time.

Getting the budget right matters more than it looks. Retrieval rate is the x-axis
of the whole comparison — "selective beats always-retrieve" is a claim about
accuracy *at a given retrieval budget*. If Qwen is calibrated to a different
budget than the 3B, the cross-model comparison confounds model scale with how
often each model was allowed to retrieve.

Usage:
    python scripts/make_calibration_set.py            # -> data/raw/mirage/calib50.json
    python scripts/make_calibration_set.py -n 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medrag_adaptive.data.loaders.mirage_loader import MIRAGE_SUBSETS


def main() -> None:
    ap = argparse.ArgumentParser(description="Stride-sample a calibration subset.")
    ap.add_argument("--benchmark", default="data/raw/mirage/benchmark.json")
    ap.add_argument("--output", default="data/raw/mirage/calib50.json")
    ap.add_argument("-n", type=int, default=50, help="calibration questions")
    ap.add_argument("--pool", type=int, default=200,
                    help="size of the evaluation pool to sample from")
    args = ap.parse_args()

    data = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))

    # Reproduce load_mirage's ordering exactly, so the pool this samples from is
    # the same 200 questions the evaluation runs use.
    ordered: list[tuple[str, str, dict]] = []
    for subset in MIRAGE_SUBSETS:
        for qid, rec in data.get(subset, {}).items():
            ordered.append((subset, qid, rec))

    pool = ordered[: args.pool]
    if len(pool) < args.pool:
        sys.exit(f"benchmark has only {len(pool)} questions, need {args.pool}")

    stride = len(pool) // args.n
    sampled = pool[::stride][: args.n]

    out: dict[str, dict] = {}
    for subset, qid, rec in sampled:
        out.setdefault(subset, {})[qid] = rec

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=1), encoding="utf-8")

    print(f"pool          {len(pool)} questions ({', '.join(sorted({s for s,_,_ in pool}))})")
    print(f"stride        every {stride}")
    print(f"sampled       {len(sampled)} questions -> {args.output}")
    for subset, recs in out.items():
        print(f"  {subset:10s} {len(recs)}")

    # The sample must not silently drop questions when reloaded.
    from medrag_adaptive.data.loaders.mirage_loader import load_mirage
    reloaded = load_mirage(args.output)
    assert len(reloaded) == len(sampled), (
        f"round-trip lost questions: wrote {len(sampled)}, loader returned {len(reloaded)}"
    )
    print(f"round-trip OK — loader returns {len(reloaded)} questions")


if __name__ == "__main__":
    main()
