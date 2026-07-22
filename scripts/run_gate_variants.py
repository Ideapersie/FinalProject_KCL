r"""
scripts/run_gate_variants.py — A8: fix the ensemble's design flaw and measure it.

THE FLAW (found by the ablation, and it is the author's own design):

  entropy and margin agree 82% of the time, because they read the SAME logit
  distribution -- they are near-collinear, not independent. The hallucination
  probe is the only genuinely independent signal (it agrees with entropy just
  55%), yet it swings the ensemble decision only 6% of the time versus entropy's
  24% and margin's 26%.

  A flat 2-of-3 majority vote over two correlated members plus one independent
  member therefore systematically DROWNS OUT the independent signal. The ensemble
  is effectively two signals wearing three hats.

THE FIX, AND WHY IT IS CHEAP TO MEASURE:

  A variant's retrieve/skip DECISION replays for free from the gate signals
  already in the logs -- no gate re-run is needed, because each gate's vote is
  recorded per query. Only the ANSWER is missing, and only on the queries where
  the variant's decision differs from the one that was actually taken. Those are
  the only queries that need the model.

  On p5_medcorp_mcq that is ~25 of 200 queries for the probe-weighted variant, so
  a genuinely MEASURED accuracy for the fix costs ~17 minutes rather than a full
  8.5-hour re-run. This turns "here is a flaw someone should fix" into "here is a
  flaw, we fixed it, and here is the measured gain" -- which is the difference
  between describing a limitation and doing something about it.

Determinism note: the model runs greedy (T=0), so re-answering a query whose
decision did NOT flip must reproduce the logged answer exactly. `--verify` checks
this on a sample and is the sanity check that the re-answer path is faithful.

Usage:
    python scripts/run_gate_variants.py --logs results/raw_logs/p5_medcorp_mcq.jsonl \
        --dataset data/processed/mirage_200.jsonl \
        --bm25-index indexes/bm25_medcorp_tp.pkl \
        --faiss-index indexes/faiss_medcorp_tp \
        --output results/raw_logs/gate_variants_mcq.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medrag_adaptive.config import load_config
from medrag_adaptive.data.loaders.mirage_loader import load_mirage
from medrag_adaptive.evaluation.loading import load_run
from medrag_adaptive.evaluation.scoring import score_mcq
from medrag_adaptive.models.prompts import build_rag_prompt, build_closed_book_prompt

GATES = ["entropy", "margin", "hallucination_probe"]


# ── the variants ───────────────────────────────────────────────────────
# Each maps the logged per-gate votes to a retrieve/skip decision.

def _n_retrieve(votes: Dict[str, str]) -> int:
    return sum(votes.get(g) == "retrieve" for g in GATES)


VARIANTS: Dict[str, Callable[[Dict[str, str]], bool]] = {
    # The shipped ensemble: flat majority. Two correlated gates can outvote the
    # independent one.
    "majority_2of3": lambda v: _n_retrieve(v) >= 2,

    # Give the independent signal the weight its independence earns: the probe
    # alone can carry the decision, and entropy+margin (which are ~the same
    # signal) must BOTH fire to do so.
    "probe_weighted": lambda v: (
        (2 if v.get("hallucination_probe") == "retrieve" else 0)
        + sum(v.get(g) == "retrieve" for g in ("entropy", "margin"))
    ) >= 2,

    # The same idea stated as a rule rather than a weight.
    "probe_or_both_logit": lambda v: (
        v.get("hallucination_probe") == "retrieve"
        or (v.get("entropy") == "retrieve" and v.get("margin") == "retrieve")
    ),

    # Drop the redundant member entirely: if margin ~= entropy, keep one.
    "drop_margin": lambda v: (
        v.get("entropy") == "retrieve" or v.get("hallucination_probe") == "retrieve"
    ),
}


def variant_decisions(records: List[dict], rule: Callable) -> List[bool]:
    return [rule(r["qvault"]["gate_details"]["votes"]) for r in records]


# ── re-answering only the flipped queries ──────────────────────────────

def answer_one(question, retrieve: bool, llm, retriever) -> str:
    """Answer a single question under a forced retrieve/skip decision.

    The gates are NOT re-run: the variant has already decided, from the logged
    signals, what the decision is. Only the answer is regenerated.
    """
    if retrieve:
        chunks = retriever.retrieve(question.question_text)
        prompt = build_rag_prompt(question.question_text, chunks,
                                  choices=question.choices, cite_sources=False)
    else:
        # The same prompt P3 (closed-book) uses, so a "skip" here is exactly the
        # skip branch the gated policy would have taken.
        prompt = build_closed_book_prompt(question.question_text, question.choices)
    return llm.answer(prompt)


def evaluate_variant(
    name: str, records: List[dict], questions: Dict[str, object],
    llm, retriever, cache: Dict[tuple, str],
) -> dict:
    """Measure a variant's true accuracy, re-answering only what flipped."""
    rule = VARIANTS[name]
    decisions = variant_decisions(records, rule)

    n = len(records)
    correct = 0
    reanswered = 0
    t0 = time.time()

    for rec, want_retrieve in zip(records, decisions):
        qid = rec["question_id"]
        had_retrieve = bool(rec["retrieval_triggered"])

        if want_retrieve == had_retrieve:
            # Decision unchanged -> the logged answer is exactly what this variant
            # would have produced. No model call needed.
            correct += bool(rec["is_correct"])
            continue

        # Decision flipped -> we do not have this branch's answer. Generate it.
        key = (qid, want_retrieve)
        if key not in cache:
            cache[key] = answer_one(questions[qid], want_retrieve, llm, retriever)
            reanswered += 1
        is_correct, _ = score_mcq(cache[key], rec["correct_answer"])
        correct += is_correct

    return {
        "variant": name,
        "n": n,
        "accuracy": correct / n,
        "retrieval_rate": sum(decisions) / n,
        "reanswered": reanswered,
        "seconds": round(time.time() - t0, 1),
    }


def verify_determinism(records, questions, llm, retriever, k: int = 3) -> None:
    """Re-answer a few UNFLIPPED queries; greedy decoding must reproduce the log.

    If this fails, the re-answer path differs from the original run (different
    prompt, retriever, or sampling) and every variant number below is untrustworthy.
    """
    print("\n[verify] re-answering unflipped queries; must match the log exactly...")
    checked = 0
    for rec in records:
        if checked >= k:
            break
        qid = rec["question_id"]
        if qid not in questions:
            continue
        got = answer_one(questions[qid], bool(rec["retrieval_triggered"]), llm, retriever)
        same = got.strip() == rec["answer_text"].strip()
        status = "OK " if same else "MISMATCH"
        print(f"  [{status}] {qid}")
        if not same:
            print(f"      logged: {rec['answer_text'].strip()[:70]!r}")
            print(f"      re-ran: {got.strip()[:70]!r}")
        checked += 1


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure gate-ensemble variants (A8).")
    ap.add_argument("--logs", required=True, help="a P5 MCQ run log")
    ap.add_argument("--dataset", required=True, help="the questions that log covers")
    ap.add_argument("--bm25-index", required=True)
    ap.add_argument("--faiss-index", required=True)
    ap.add_argument("--output", required=True, help="JSON results")
    ap.add_argument("--base", default="configs/base.yaml")
    ap.add_argument("--policy", default="configs/policies/p5_gated_entropy.yaml")
    ap.add_argument("--model", default="models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
                    help="GGUF path (configs/base.yaml carries a stale filename)")
    ap.add_argument("--verify", action="store_true",
                    help="check the re-answer path reproduces logged answers")
    args = ap.parse_args()

    records = [r for r in load_run(args.logs)
               if r.get("qvault", {}).get("gate_details", {}).get("votes")]
    if not records:
        sys.exit("no gate votes in that log -- is it a P5 run?")

    qs = {q.question_id: q for q in load_mirage(args.dataset)}
    missing = [r["question_id"] for r in records if r["question_id"] not in qs]
    if missing:
        sys.exit(f"{len(missing)} logged questions are not in --dataset (e.g. {missing[0]})")

    cfg = load_config(base=args.base, policy=args.policy)
    cfg.model.gguf_path = args.model

    from medrag_adaptive.models.llama_backend import llama_backend_from_config
    from medrag_adaptive.retrieval.factory import build_retriever

    retriever = build_retriever(cfg, bm25_index=args.bm25_index,
                                faiss_index=args.faiss_index)

    # Cache: the same (question, decision) pair is re-answered at most once even
    # if several variants need it.
    cache: Dict[tuple, str] = {}
    results = []

    with llama_backend_from_config(cfg) as llm:
        if args.verify:
            verify_determinism(records, qs, llm, retriever)

        # Cost preview before spending any time.
        print(f"\nRe-answers needed per variant (of {len(records)} queries):")
        for name in VARIANTS:
            flips = sum(
                1 for r, d in zip(records, variant_decisions(records, VARIANTS[name]))
                if d != bool(r["retrieval_triggered"])
            )
            print(f"  {name:22s} {flips:3d} flipped")

        for name in VARIANTS:
            print(f"\n[{name}] measuring...")
            res = evaluate_variant(name, records, qs, llm, retriever, cache)
            results.append(res)
            print(f"  accuracy {res['accuracy']:.1%}  retrieval {res['retrieval_rate']:.1%}"
                  f"  ({res['reanswered']} re-answered, {res['seconds']:.0f}s)")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")

    base = next(r for r in results if r["variant"] == "majority_2of3")
    print("\n=== summary (vs the shipped 2-of-3 majority) ===")
    for r in sorted(results, key=lambda x: -x["accuracy"]):
        delta = r["accuracy"] - base["accuracy"]
        mark = "  <- shipped" if r["variant"] == "majority_2of3" else ""
        print(f"  {r['variant']:22s} acc {r['accuracy']:6.1%} ({delta:+.1%})"
              f"  retrieval {r['retrieval_rate']:5.1%}{mark}")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
