r"""
scripts/analyse_chunk_relevance.py — why does retrieval HURT on multiple choice?

The measured fact this exists to explain: on MIRAGE MCQ, retrieval breaks 46
previously-correct answers and fixes only 13 (net -33 of 200, i.e. -16.5%). Yet
the SAME retriever on the SAME corpus HELPS on open-ended (F1 0.211 vs 0.190).
Same components, opposite sign. Something about MCQ specifically is going wrong.

There are only two possible explanations, and they demand opposite remedies:

  (a) RETRIEVER FAULT — the passages fetched are irrelevant. The model is misled
      by junk. Remedy: fix retrieval (better chunking, reranking, top-k).

  (b) MODEL FAULT (distraction) — the passages are relevant, but a 3B model
      handed a wall of related-but-not-decisive prose stops recalling the fact it
      already knew and starts pattern-matching the text. Remedy: a stronger model,
      or don't retrieve when the model already knows (i.e. the gate).

Distinguishing them is the whole point: (a) means the project should invest in the
retriever, (b) means the gate is the right idea and the ceiling is model capacity.
Guessing is not acceptable; this script measures it.

METHOD. For the questions where P3 (closed-book) is RIGHT but P1 (retrieve) is
WRONG -- retrieval actively destroyed a correct answer -- score the chunks P1
was given against the question and its gold answer:

  gold_hit      does any retrieved chunk contain the gold answer text?
  lexical_overlap  token-F1 between the question and the chunk (on-topic-ness)
  answer_overlap   token-F1 between the gold answer and the chunk

If chunks score LOW on relevance, it is (a): the retriever failed.
If chunks score HIGH -- relevant passages, still wrong answer -- it is (b):
distraction, and the retriever is exonerated.

REQUIRES a P1 run logged WITH retrieved chunks. The original P1 log predates chunk
logging (`qvault.retrieved_chunks` is absent), so re-run P1 first:

    python scripts/run_experiment.py --policy configs/policies/p1_always_retrieve.yaml \
        --experiment configs/experiments/mirage_medcorp.yaml \
        --dataset data/raw/mirage/benchmark.json \
        --bm25-index indexes/bm25_medcorp_tp.pkl --faiss-index indexes/faiss_medcorp_tp \
        --model models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
        --output results/raw_logs/p1_medcorp_mcq_chunks.jsonl

Then:
    python scripts/analyse_chunk_relevance.py \
        --p1 results/raw_logs/p1_medcorp_mcq_chunks.jsonl \
        --p3 results/raw_logs/p3_mirage200.jsonl \
        --dataset data/raw/mirage/benchmark.json
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Dict, List

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medrag_adaptive.data.loaders.mirage_loader import load_mirage
from medrag_adaptive.evaluation.loading import load_run
from medrag_adaptive.evaluation.scoring import token_f1


def _chunks(rec: dict) -> List[dict]:
    return rec.get("qvault", {}).get("retrieved_chunks", []) or []


def _gold_text(question) -> str:
    """The text of the correct option (not the letter) -- what a useful chunk
    would have to contain."""
    if question.choices:
        return question.choices.get(question.correct_answer.strip().upper(), "")
    return question.correct_answer


def score_group(recs: List[dict], questions: Dict[str, object], label: str) -> dict:
    """Relevance of what was retrieved, over one group of questions."""
    gold_hits, q_overlaps, a_overlaps, best_a = [], [], [], []

    for r in recs:
        q = questions.get(r["question_id"])
        if q is None:
            continue
        cs = _chunks(r)
        if not cs:
            continue
        gold = _gold_text(q)
        gold_lc = gold.lower().strip()

        # Does ANY chunk literally contain the gold answer text?
        hit = any(gold_lc and gold_lc in c["text"].lower() for c in cs)
        gold_hits.append(hit)

        # How on-topic are the chunks (question overlap), and how answer-bearing
        # (gold-answer overlap)?
        q_overlaps.append(statistics.mean(
            token_f1(c["text"], q.question_text) for c in cs))
        a_scores = [token_f1(c["text"], gold) for c in cs]
        a_overlaps.append(statistics.mean(a_scores))
        best_a.append(max(a_scores))

    n = len(gold_hits)
    if not n:
        return {"label": label, "n": 0}
    return {
        "label": label,
        "n": n,
        "gold_hit_rate": sum(gold_hits) / n,
        "mean_question_overlap": statistics.mean(q_overlaps),
        "mean_answer_overlap": statistics.mean(a_overlaps),
        "mean_best_answer_overlap": statistics.mean(best_a),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Is retrieval irrelevant, or just distracting?")
    ap.add_argument("--p1", required=True, help="P1 log WITH qvault.retrieved_chunks")
    ap.add_argument("--p3", required=True, help="P3 closed-book log")
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()

    p1 = {r["question_id"]: r for r in load_run(args.p1)}
    p3 = {r["question_id"]: r for r in load_run(args.p3)}
    questions = {q.question_id: q for q in load_mirage(args.dataset)}

    have_chunks = sum(1 for r in p1.values() if _chunks(r))
    if not have_chunks:
        sys.exit(
            "This P1 log has no qvault.retrieved_chunks -- it predates chunk "
            "logging. Re-run P1 (see this script's docstring); the run_experiment "
            "change is already in place."
        )
    print(f"P1 records with logged chunks: {have_chunks}/{len(p1)}")

    # The decisive group: retrieval turned a right answer into a wrong one.
    broke = [q for q in p1 if q in p3 and p3[q]["is_correct"] and not p1[q]["is_correct"]]
    fixed = [q for q in p1 if q in p3 and not p3[q]["is_correct"] and p1[q]["is_correct"]]
    both_ok = [q for q in p1 if q in p3 and p3[q]["is_correct"] and p1[q]["is_correct"]]

    print(f"\nretrieval BROKE a correct answer : {len(broke)}")
    print(f"retrieval FIXED a wrong answer   : {len(fixed)}")
    print(f"both correct (retrieval harmless): {len(both_ok)}")

    groups = [
        score_group([p1[q] for q in broke], questions, "retrieval BROKE it"),
        score_group([p1[q] for q in fixed], questions, "retrieval FIXED it"),
        score_group([p1[q] for q in both_ok], questions, "both correct"),
    ]

    print("\n=== relevance of the chunks P1 was given ===")
    print(f"{'group':22s} {'n':>4s} {'gold in chunk':>14s} {'q-overlap':>10s} "
          f"{'a-overlap':>10s} {'best a-ovl':>11s}")
    for g in groups:
        if not g["n"]:
            continue
        print(f"{g['label']:22s} {g['n']:4d} {g['gold_hit_rate']:13.0%} "
              f"{g['mean_question_overlap']:10.3f} {g['mean_answer_overlap']:10.3f} "
              f"{g['mean_best_answer_overlap']:11.3f}")

    # The verdict. Two axes must be read TOGETHER, because they can disagree --
    # and on this data they do, which is the whole finding:
    #
    #   question-overlap = is the chunk ON-TOPIC?
    #   gold-in-chunk    = does the chunk actually CONTAIN THE ANSWER?
    #
    # A chunk can be perfectly on-topic and still not answer the question. That
    # combination is the worst case for a small model: it looks authoritative,
    # it is about the right subject, and it does not contain the fact -- so the
    # model reasons from it instead of recalling what it already knew.
    b = next(g for g in groups if g["label"].startswith("retrieval BROKE"))
    ok = next((g for g in groups if g["label"] == "both correct"), None)
    print("\n=== interpretation ===")
    if ok and ok["n"]:
        print("  chunks on BROKEN questions vs chunks on questions that survived:")
        print(f"    on-topic (q-overlap) : {b['mean_question_overlap']:.3f}  vs  "
              f"{ok['mean_question_overlap']:.3f}")
        print(f"    answer-bearing       : {b['gold_hit_rate']:.0%}  vs  "
              f"{ok['gold_hit_rate']:.0%}")
        print()

        on_topic_same = abs(b["mean_question_overlap"] - ok["mean_question_overlap"]) < 0.05
        answer_gap = ok["gold_hit_rate"] - b["gold_hit_rate"]

        if on_topic_same and answer_gap > 0.10:
            print("  MIXED, and the mixture is the finding:")
            print()
            print("  The chunks are just as ON-TOPIC on the questions retrieval breaks as")
            print("  on the ones it does not -- the retriever is not fetching junk. But")
            print("  they are far less often ANSWER-BEARING "
                  f"({b['gold_hit_rate']:.0%} vs {ok['gold_hit_rate']:.0%}).")
            print()
            print("  So the model is handed passages that are topically relevant and")
            print("  factually useless, and -- lacking the capacity to tell 'related' from")
            print("  'decisive' -- it abandons the answer it already knew and reasons from")
            print("  the passage instead. Both explanations hold a piece of the truth:")
            print()
            print("    * retrieval quality IS a real limit: only a minority of retrievals")
            print("      contain the answer at all, so there is headroom in the retriever.")
            print("    * but the FAILURE MODE is distraction, not irrelevance. Better")
            print("      chunks would raise the ceiling; they would not stop a small model")
            print("      being derailed by on-topic text that does not answer the question.")
            print()
            print("  Remedies, in order of leverage: (1) do not retrieve when the model")
            print("  already knows -- which is exactly what the gate does, and why its SKIP")
            print("  decisions are so protective; (2) a model able to ignore unhelpful")
            print("  context, i.e. scale.")
        elif on_topic_same:
            print("  DISTRACTION (hypothesis b). The chunks are as relevant on the broken")
            print("  questions as elsewhere, so the retriever is not the differentiator.")
            print("  Better retrieval would not fix this; not retrieving would.")
        else:
            print("  RETRIEVER FAULT (hypothesis a). The chunks on broken questions are")
            print("  markedly less relevant. Improving chunking / reranking / top-k is")
            print("  likely to recover accuracy.")

    print("\n=== examples: relevant chunk retrieved, answer still wrong ===")
    shown = 0
    for qid in broke:
        if shown >= 3:
            break
        q = questions.get(qid)
        cs = _chunks(p1[qid])
        if not q or not cs:
            continue
        gold = _gold_text(q)
        best = max(cs, key=lambda c: token_f1(c["text"], gold))
        print(f"\n[{qid}] gold = {p1[qid]['correct_answer']}. {gold!r}")
        print(f"  best chunk (a-overlap {token_f1(best['text'], gold):.2f}) "
              f"from {best['source']}/{best['title'][:40]!r}:")
        print(f"    {best['text'][:200].strip()!r}")
        print(f"  P3 (no retrieval, RIGHT): {p3[qid]['answer_text'].strip()[:60]!r}")
        print(f"  P1 (with chunks, WRONG) : {p1[qid]['answer_text'].strip()[:90]!r}")
        shown += 1


if __name__ == "__main__":
    main()
