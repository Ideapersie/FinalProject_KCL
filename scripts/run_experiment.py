"""
scripts/run_experiment.py — Sequential, resumable experiment driver.

Runs one policy over one dataset, profiling each query and appending a
RunRecord to a JSONL audit log. Resumable: question-ids already present in the
output file are skipped, so an interrupted run continues where it left off.

Week 1 wires P1 (always-retrieve, BM25) and P3 (closed-book) directly. From
Week 2 the PolicyFactory replaces the direct instantiation below.

Usage:
    python scripts/run_experiment.py \
        --policy configs/policies/p3_closed_book.yaml \
        --dataset data/raw/mirage/benchmark.json \
        --bm25-index indexes/bm25_combined.pkl \
        --max-questions 200 \
        --output results/raw_logs/p3_mirage.jsonl
"""

from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path
from typing import List, Optional

from medrag_adaptive.config import ProjectConfig, load_config
from medrag_adaptive.data.loaders.mirage_loader import load_mirage
from medrag_adaptive.data.loaders.ragcare_loader import load_ragcare
from medrag_adaptive.data.schema import (
    RunRecord,
    UnifiedQuestion,
    append_record,
    load_logged_ids,
)
from medrag_adaptive.evaluation.profiler import profile_call
from medrag_adaptive.evaluation.scoring import score_mcq, token_f1
from medrag_adaptive.models.base import LLMBackend
from medrag_adaptive.policies.base import Policy
from medrag_adaptive.policies.factory import build_policy
from medrag_adaptive.retrieval.base import Retriever
from medrag_adaptive.retrieval.bm25_retriever import BM25Retriever


# ── data loading ───────────────────────────────────────────────────

def load_questions(dataset_path: str, cfg: ProjectConfig) -> List[UnifiedQuestion]:
    """Dispatch to the right loader by configured dataset name."""
    name = cfg.dataset.lower()
    max_q = cfg.evaluation.max_questions
    if name.startswith("mirage"):
        return load_mirage(dataset_path, max_questions=max_q)
    if name.startswith("ragcare"):
        return load_ragcare(dataset_path, max_questions=max_q)
    raise ValueError(f"Unknown dataset '{cfg.dataset}'")


# ── policy wiring: delegated to policies.factory.build_policy ───────
# (The factory picks the policy class and assembles the P5 gate ensemble,
#  downgrading to probe-only when the hardware tier disables logits.)


# ── scoring + record assembly ──────────────────────────────────────

def score_and_record(question, result, cfg) -> RunRecord:
    if question.is_multiple_choice():
        is_correct, em = score_mcq(result.answer_text, question.correct_answer)
        f1 = em
    else:
        f1 = token_f1(result.answer_text, question.correct_answer)
        is_correct = f1 >= 0.5
        em = 1.0 if f1 == 1.0 else 0.0

    qvault = {}
    if result.gate_signal_value is not None:
        qvault["gate_signal_value"] = result.gate_signal_value
    # Full gate decision payload (per-token entropy, member votes, probe drafts)
    # for auditability and the interpretability visualisations.
    if result.gate_details:
        qvault["gate_details"] = result.gate_details

    return RunRecord(
        question_id=question.question_id,
        dataset_source=question.dataset_source,
        risk_level=question.risk_level,
        specialty=question.specialty,
        policy_name=result.policy_name,
        gate_name=result.gate_name,
        gate_decision=result.gate_decision,
        gate_signal_value=result.gate_signal_value,
        retrieval_triggered=result.retrieval_triggered,
        answer_text=result.answer_text,
        correct_answer=question.correct_answer,
        is_correct=is_correct,
        exact_match=em,
        f1_score=f1,
        citations=[{"chunk_id": c.chunk_id, "source": c.source, "title": c.title}
                   for c in result.citations],
        latency_ns=result.latency_ns,
        energy_kwh=result.energy_kwh,
        peak_memory_mb=result.peak_memory_mb,
        model_name=cfg.model.name,
        hardware_tier=cfg.hardware.tier,
        timestamp=_dt.datetime.now(_dt.timezone.utc).isoformat(),
        qvault=qvault,
    )


def run(
    cfg: ProjectConfig,
    dataset_path: str,
    output: str,
    llm: LLMBackend,
    retriever: Optional[Retriever] = None,
) -> int:
    """Run the configured policy over the dataset. Returns count of new records."""
    questions = load_questions(dataset_path, cfg)
    policy = build_policy(cfg, llm, retriever)
    seen = load_logged_ids(output)

    written = 0
    for q in questions:
        if q.question_id in seen:
            continue
        result, metrics = profile_call(
            lambda: policy.answer(q),
            track_latency=cfg.profiling.track_latency,
            track_memory=cfg.profiling.track_memory,
            track_energy=cfg.profiling.track_energy,
            country_iso=cfg.profiling.codecarbon_country_iso_code,
        )
        result.latency_ns = metrics.latency_ns
        result.energy_kwh = metrics.energy_kwh
        result.peak_memory_mb = metrics.peak_memory_mb
        append_record(score_and_record(q, result, cfg), output)
        written += 1
    return written


# ── CLI ────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Run one policy over one dataset.")
    ap.add_argument("--base", default="configs/base.yaml")
    ap.add_argument("--hardware", default=None)
    ap.add_argument("--policy", required=True, help="policy YAML")
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--dataset", required=True, help="path to benchmark file")
    ap.add_argument("--bm25-index", default=None, help="BM25 pickle for retrieval policies")
    ap.add_argument("--model", default=None, help="override gguf_path from config")
    ap.add_argument("--max-questions", type=int, default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cfg = load_config(
        base=args.base, hardware=args.hardware,
        policy=args.policy, experiment=args.experiment,
    )
    if args.max_questions is not None:
        cfg.evaluation.max_questions = args.max_questions
    if args.model is not None:
        cfg.model.gguf_path = args.model

    from medrag_adaptive.models.llama_backend import llama_backend_from_config
    retriever = BM25Retriever.from_index(args.bm25_index) if args.bm25_index else None

    with llama_backend_from_config(cfg) as llm:
        n = run(cfg, args.dataset, args.output, llm, retriever)
    print(f"Wrote {n} new records to {args.output}")


if __name__ == "__main__":
    main()
