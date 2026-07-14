"""
evaluation/loading.py — The one canonical way to read a run log.

Every downstream artefact (figures, tables, ablation, and therefore every number
in the report) must read logs through `load_run`. That is what makes the
"one source of truth" claim true rather than merely aspirational: there is a
single code path from the bytes on disk to any number that reaches the page.

Two corrections are applied *on read*. The raw JSONL files are never modified —
they are the audit trail, and an examiner asking "are these the raw logs?" must
get an unqualified yes.

1. MCQ records are RE-SCORED with the current `score_mcq`.

   The answer-extractor was hardened after an audit found that a leading option
   letter could be mistaken for the model's choice when the model echoed the
   choice list back before stating its answer ("A. ... B. ... The correct answer
   is B."). The logs still carry the `is_correct` computed by the *old*
   extractor. Re-scoring on read means the figures, tables and prose all reflect
   the corrected extractor without rewriting history.

2. Open-ended records have `is_correct` forced to None.

   For open-ended questions the logged `is_correct` is unusable. It was written
   as a soft floor (`f1 > 0`), but it is also *internally inconsistent* across
   runs: pubmedqa_p3_open records 0/200 while p1/p5_medcorp_open record 200/200.
   Any accuracy averaged from it would be meaningless. Setting it to None makes
   it impossible to report by accident — `aggregate_summary` skips None — and
   forces open-ended results to be reported as mean token-F1, which is the only
   defensible metric for them.
"""

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

from medrag_adaptive.evaluation.scoring import score_mcq

Record = Dict

# A gold answer that is a bare option letter marks a multiple-choice record.
_MCQ_GOLD = re.compile(r"^\s*[A-E]\s*$")


@dataclass
class LoadReport:
    """What `load_run` changed on the way in — reportable provenance."""
    path: str
    n: int
    kind: str            # "mcq" | "open" | "empty"
    rescored: int        # MCQ records whose is_correct flipped vs the log
    nulled: int          # open-ended records whose is_correct was voided


def is_mcq_record(rec: Record) -> bool:
    """True when the gold answer is a bare option letter (A-E)."""
    gold = rec.get("correct_answer")
    return isinstance(gold, str) and bool(_MCQ_GOLD.match(gold))


def load_run(path: Union[str, Path]) -> List[Record]:
    """Read a run log, applying the on-read corrections. See module docstring."""
    records, _ = load_run_with_report(path)
    return records


def load_run_with_report(path: Union[str, Path]) -> tuple[List[Record], LoadReport]:
    """As `load_run`, but also return what was corrected.

    Used by the report to state provenance as a checked fact ("re-scoring changed
    1 of 200 records") rather than an unverified claim.
    """
    path = Path(path)
    records: List[Record] = []
    rescored = 0
    nulled = 0

    with io.open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            if is_mcq_record(rec):
                is_correct, exact_match = score_mcq(
                    rec.get("answer_text", ""), rec["correct_answer"]
                )
                if bool(rec.get("is_correct")) != is_correct:
                    rescored += 1
                rec["is_correct"] = is_correct
                rec["exact_match"] = exact_match
            else:
                # Open-ended: the logged is_correct is unusable (see docstring).
                if rec.get("is_correct") is not None:
                    nulled += 1
                rec["is_correct"] = None

            records.append(rec)

    if not records:
        kind = "empty"
    elif is_mcq_record(records[0]):
        kind = "mcq"
    else:
        kind = "open"

    report = LoadReport(
        path=str(path), n=len(records), kind=kind,
        rescored=rescored, nulled=nulled,
    )
    return records, report
