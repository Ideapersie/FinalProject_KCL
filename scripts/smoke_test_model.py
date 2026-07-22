r"""
scripts/smoke_test_model.py — prove a new model works BEFORE spending 20 hours on it.

The failure this exists to prevent: a model given the wrong chat markup does not
crash. Qwen2.5 handed Llama's `[INST]` markup will still emit fluent text — just
worse text, and possibly text the answer-extractor cannot parse. Discovering that
after a 22-hour 14B run, with the deadline three weeks out, is the single worst
outcome available in this project. So nothing long starts until this passes.

Four checks, each of which has a specific way of silently ruining a run:

  1. CHAT FORMAT     the prompt carries the markup this model was tuned on.
                     Wrong markup => degraded answers reported as the model's.

  2. ANSWERS PARSE   `extract_letter` finds a letter in the model's output.
                     If it cannot, accuracy reads ~0% and looks like a bad model
                     rather than a bad parser.

  3. GATES FIRE      entropy and margin come back as real numbers, which means
                     `logits_all` survived and the GPU build still exposes
                     `_scores`. If these are None the gates are silently dead and
                     P5 degenerates to a retrieve-everything policy.

  4. SIGNALS IN RANGE  the gate signals actually span the calibrated threshold.
                     If EVERY query lands on one side of tau, the gate never
                     fires and P5 collapses onto P1 or P3 — exactly the
                     calibration-non-transfer failure seen on the 3B. This is not
                     an error (it is a finding), but it must be caught BEFORE the
                     long run so the thresholds can be recalibrated first.

Usage:
    python scripts/smoke_test_model.py \
        --model-config configs/models/qwen7b.yaml \
        --dataset data/raw/mirage/benchmark.json \
        --bm25-index indexes/bm25_medcorp_tp.pkl \
        --faiss-index indexes/faiss_medcorp_tp \
        -n 20
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medrag_adaptive.config import load_config
from medrag_adaptive.data.loaders.mirage_loader import load_mirage
from medrag_adaptive.evaluation.scoring import extract_letter, score_mcq
from medrag_adaptive.models.prompts import build_closed_book_prompt, get_chat_format

PASS = "  [PASS]"
FAIL = "  [FAIL]"
WARN = "  [WARN]"


def main() -> None:
    ap = argparse.ArgumentParser(description="Smoke-test a model before a long run.")
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--bm25-index", default=None)
    ap.add_argument("--faiss-index", default=None)
    ap.add_argument("--base", default="configs/base.yaml")
    ap.add_argument("--policy", default="configs/policies/p5_gated_entropy.yaml")
    ap.add_argument("-n", type=int, default=20, help="questions to test")
    args = ap.parse_args()

    cfg = load_config(base=args.base, model=args.model_config, policy=args.policy)
    failures: list[str] = []

    print(f"\n=== smoke test: {cfg.model.name} ===")
    print(f"  gguf         {cfg.model.gguf_path}")
    print(f"  chat_format  {cfg.model.chat_format}")
    print(f"  n_gpu_layers {cfg.model.n_gpu_layers}")
    print(f"  logits_all   {cfg.model.logits_all}")

    if not Path(cfg.model.gguf_path).exists():
        sys.exit(f"\nGGUF not found: {cfg.model.gguf_path}\nDownload it first.")

    questions = load_mirage(args.dataset)[: args.n]
    print(f"  questions    {len(questions)}")

    from medrag_adaptive.models.llama_backend import llama_backend_from_config
    from medrag_adaptive.retrieval.factory import build_retriever
    from medrag_adaptive.policies.factory import build_gate

    retriever = None
    if args.bm25_index or args.faiss_index:
        retriever = build_retriever(cfg, bm25_index=args.bm25_index,
                                    faiss_index=args.faiss_index)

    with llama_backend_from_config(cfg) as llm:

        # ── 1. chat format actually applied ───────────────────────────
        print("\n[1/4] chat format")
        probe = build_closed_book_prompt(questions[0].question_text,
                                         questions[0].choices)
        active = get_chat_format()
        if active != cfg.model.chat_format:
            failures.append("backend did not apply the configured chat_format")
            print(f"{FAIL} active={active!r}, expected {cfg.model.chat_format!r}")
        elif cfg.model.chat_format == "qwen" and "<|im_start|>" not in probe:
            failures.append("qwen configured but prompt has no ChatML markup")
            print(f"{FAIL} no <|im_start|> in the prompt")
        elif cfg.model.chat_format == "llama" and "[INST]" not in probe:
            failures.append("llama configured but prompt has no [INST] markup")
            print(f"{FAIL} no [INST] in the prompt")
        else:
            print(f"{PASS} prompts use {active!r} markup")

        # ── 2. answers parse ──────────────────────────────────────────
        print(f"\n[2/4] answers parse ({len(questions)} questions)")
        parsed, correct = 0, 0
        samples = []
        for q in questions:
            text = llm.answer(build_closed_book_prompt(q.question_text, q.choices))
            letter = extract_letter(text)
            if letter:
                parsed += 1
            if score_mcq(text, q.correct_answer)[0]:
                correct += 1
            if len(samples) < 3:
                samples.append((q.question_id, q.correct_answer, letter,
                                text.strip()[:60]))

        rate = parsed / len(questions)
        if rate < 0.8:
            failures.append(f"only {rate:.0%} of answers yielded a letter")
            print(f"{FAIL} extract_letter found a letter in only {parsed}/{len(questions)}")
            print("       -> the chat format or the answer prompt is wrong for this model")
        else:
            print(f"{PASS} {parsed}/{len(questions)} parsed ({rate:.0%})")
        print(f"       closed-book accuracy on this sample: {correct}/{len(questions)}"
              f" ({correct/len(questions):.0%})")
        for qid, gold, got, txt in samples:
            print(f"       [{qid}] gold={gold} got={got} :: {txt!r}")

        # ── 3. gates produce real signals ─────────────────────────────
        print("\n[3/4] gate signals")
        gate = build_gate(cfg)
        ent, mar = [], []
        votes_retrieve = 0
        for q in questions:
            d = gate.decide(q, llm)
            sig = (d.details or {}).get("signals", {})
            if sig.get("entropy") is not None:
                ent.append(float(sig["entropy"]))
            if sig.get("margin") is not None:
                mar.append(float(sig["margin"]))
            votes_retrieve += bool(d.retrieve)

        if not ent:
            failures.append("entropy gate produced no signal (logits_all lost?)")
            print(f"{FAIL} no entropy signal — is logits_all still true under CUDA?")
        if not mar:
            failures.append("margin gate produced no signal (logprobs lost?)")
            print(f"{FAIL} no margin signal — is logprobs=2 still available?")
        if ent and mar:
            print(f"{PASS} entropy and margin both produced signals")

        # ── 4. signals span the threshold ─────────────────────────────
        print("\n[4/4] signal range vs calibrated thresholds")
        tau_h = cfg.gate.entropy_threshold
        tau_m = cfg.gate.margin_threshold
        for name, vals, tau in (("entropy", ent, tau_h), ("margin", mar, tau_m)):
            if not vals:
                continue
            lo, hi = min(vals), max(vals)
            above = sum(v > tau for v in vals)
            print(f"       {name:8s} [{lo:.3f}, {hi:.3f}]  mean {statistics.mean(vals):.3f}"
                  f"  tau={tau}  ({above}/{len(vals)} above)")
            if tau < lo or tau > hi:
                print(f"{WARN} tau={tau} is OUTSIDE the observed {name} range.")
                print(f"       The gate can never fire on this model with these")
                print(f"       thresholds — exactly the calibration-non-transfer")
                print(f"       failure seen on the 3B. RECALIBRATE before the long run")
                print(f"       (this is a finding, not a bug).")

        retr_rate = votes_retrieve / len(questions)
        print(f"       ensemble retrieval rate on this sample: {retr_rate:.0%}")
        if retr_rate in (0.0, 1.0):
            print(f"{WARN} the gate is degenerate ({retr_rate:.0%}) — P5 would collapse")
            print(f"       onto {'P3 closed-book' if retr_rate == 0 else 'P1 always-retrieve'}."
                  f" Recalibrate the thresholds first.")

    # ── verdict ───────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    if failures:
        print("SMOKE TEST FAILED — do NOT start the long run:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("SMOKE TEST PASSED — the model is safe to run.")
    print("(Check any [WARN] above: thresholds may still need recalibrating,")
    print(" which is cheap and offline — but the run itself will work.)")


if __name__ == "__main__":
    main()
