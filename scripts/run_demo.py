r"""
scripts/run_demo.py — launch the adaptive-gating demo UI.

    python scripts/run_demo.py

Every path it needs is defaulted for this machine, so the demo starts with no
flags at all. Two of those defaults exist because the config files cannot carry
them:

  * --model: the GGUF filename on disk differs from the one named in the YAML.
  * --bm25-index / --faiss-index: base.yaml names the never-built `*_combined`
    indexes, so without an override build_retriever fails and the app silently
    boots with no retriever — every RETRIEVE verdict would then show an error
    instead of evidence.

Speed flags (--threads, --n-batch, --max-new-tokens) default well above the
hardware_medium evaluation tier. That tier is what the reported latencies were
measured on and is deliberately left alone; a live demo on a 16-core laptop has
no reason to run at 4 threads and wait 150 s per question. None of the three
changes a gate signal: the draft is greedy and its length is set separately by
gate.draft_max_tokens.

Requires gradio, which is declared in requirements.txt but is not needed by any
evaluation code path — install it into a virtualenv rather than the environment
the submission runs in.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DEFAULT_MODEL = "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
DEFAULT_BM25 = "indexes/bm25_medcorp_tp.pkl"
DEFAULT_FAISS = "indexes/faiss_medcorp_tp"


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive-gating demo UI")
    parser.add_argument("--config", default="configs/experiments/mirage_medcorp.yaml",
                        help="experiment YAML")
    parser.add_argument("--policy", default="configs/policies/p5_gated_entropy.yaml")
    parser.add_argument("--hardware", default="configs/hardware_medium.yaml")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="override model.gguf_path")
    parser.add_argument("--bm25-index", default=DEFAULT_BM25)
    parser.add_argument("--faiss-index", default=DEFAULT_FAISS)
    parser.add_argument("--threads", type=int, default=12,
                        help="inference threads (evaluation tier uses 4)")
    parser.add_argument("--n-batch", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=64,
                        help="final answer length; does not affect gate signals")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-model", action="store_true",
                        help="start without a local GGUF (API backend only)")
    args = parser.parse_args()

    from medrag_adaptive.config import load_config
    from medrag_adaptive.retrieval.factory import build_retriever
    from medrag_adaptive.ui.app import launch_app
    from medrag_adaptive.ui.session import DemoSession

    cfg = load_config(base="configs/base.yaml", hardware=args.hardware,
                      policy=args.policy, experiment=args.config)
    if args.model:
        cfg.model.gguf_path = args.model
    cfg.hardware.n_threads = args.threads
    cfg.hardware.n_batch = args.n_batch
    cfg.model.max_new_tokens = args.max_new_tokens
    cfg.policy.name = "p5_gated"

    llm = None
    if not args.no_model:
        from medrag_adaptive.models.llama_backend import llama_backend_from_config
        print(f"Loading {cfg.model.gguf_path} ...")
        t0 = time.perf_counter()
        llm = llama_backend_from_config(cfg)
        print(f"  model ready in {time.perf_counter() - t0:.1f}s")

    # Deliberately broad: a missing index raises FileNotFoundError, but a missing
    # faiss or sentence-transformers install raises ImportError, and a corrupt
    # pickle raises almost anything. None of it should stop the app booting —
    # the gate and closed-book paths work without a retriever.
    try:
        print(f"Loading indexes {args.bm25_index} + {args.faiss_index} ...")
        t0 = time.perf_counter()
        retriever = build_retriever(cfg, bm25_index=args.bm25_index,
                                    faiss_index=args.faiss_index)
        print(f"  indexes ready in {time.perf_counter() - t0:.1f}s")
    except Exception as exc:
        print(f"[warn] no retriever available: {type(exc).__name__}: {exc}")
        print("       the SKIP path still works; RETRIEVE will report the error.")
        retriever = None

    # Warm up before the browser opens. The first generation of a llama.cpp
    # session pays for KV-cache allocation and the first prompt eval, which
    # would otherwise land on the first question asked in front of an audience.
    if llm is not None:
        print("Warming up ...")
        t0 = time.perf_counter()
        llm.draft("Question: What is the largest organ in the body?\nAnswer:",
                  max_tokens=8)
        print(f"  warm in {time.perf_counter() - t0:.1f}s")

    session = DemoSession(cfg, llm, retriever)
    print(f"Ready on http://127.0.0.1:{args.port}")
    launch_app(session, port=args.port)


if __name__ == "__main__":
    main()
