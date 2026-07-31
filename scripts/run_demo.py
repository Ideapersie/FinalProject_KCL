r"""
scripts/run_demo.py — launch the adaptive-gating demo UI.

    python scripts/run_demo.py \
        --config configs/experiments/mirage_medcorp.yaml \
        --model models/Llama-3.2-3B-Instruct-Q4_K_M.gguf

--model is an explicit flag because the GGUF filename on disk differs from the
one named in the YAML (a known trap in this repo).

Requires gradio, which is declared in requirements.txt but is not needed by any
evaluation code path — install it into a virtualenv rather than the environment
the submission runs in.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive-gating demo UI")
    parser.add_argument("--config", default="configs/experiments/mirage_medcorp.yaml",
                        help="experiment YAML")
    parser.add_argument("--policy", default="configs/policies/p5_gated_entropy.yaml")
    parser.add_argument("--hardware", default="configs/hardware_medium.yaml")
    parser.add_argument("--model", default=None, help="override model.gguf_path")
    parser.add_argument("--bm25-index", default=None)
    parser.add_argument("--faiss-index", default=None)
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
    cfg.policy.name = "p5_gated"

    llm = None
    if not args.no_model:
        from medrag_adaptive.models.llama_backend import llama_backend_from_config
        print(f"Loading {cfg.model.gguf_path} ...")
        llm = llama_backend_from_config(cfg)

    # Deliberately broad: a missing index raises FileNotFoundError, but a missing
    # faiss or sentence-transformers install raises ImportError, and a corrupt
    # pickle raises almost anything. None of it should stop the app booting —
    # the gate and closed-book paths work without a retriever.
    try:
        retriever = build_retriever(cfg, bm25_index=args.bm25_index,
                                    faiss_index=args.faiss_index)
    except Exception as exc:
        print(f"[warn] no retriever available: {type(exc).__name__}: {exc}")
        print("       the SKIP path still works; RETRIEVE will report the error.")
        retriever = None

    session = DemoSession(cfg, llm, retriever)
    launch_app(session, port=args.port)


if __name__ == "__main__":
    main()
