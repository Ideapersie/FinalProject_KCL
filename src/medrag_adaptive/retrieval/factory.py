"""
retrieval/factory.py — Build a Retriever from config + index paths.

Selects the retriever implementation from cfg.policy.retrieval_mode:
    bm25   → BM25Retriever         (lexical; loads a BM25 pickle)
    vector → VectorRetriever       (dense; loads a FAISS index dir)
    hybrid → HybridRetriever       (RRF over BM25 + vector)

Index locations come from cfg.retrieval (bm25_index, faiss_index). The caller
may override them; missing indexes raise a clear error so a run fails fast
rather than silently degrading to the wrong retriever.

Keeping this separate from policies.factory means the policy factory stays
about policies and gates, and retriever construction (with its FAISS/embedding
imports) is loaded only when actually needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from medrag_adaptive.config import ProjectConfig
from medrag_adaptive.retrieval.base import Retriever
from medrag_adaptive.retrieval.bm25_retriever import BM25Retriever


def build_retriever(
    cfg: ProjectConfig,
    bm25_index: Optional[str] = None,
    faiss_index: Optional[str] = None,
) -> Optional[Retriever]:
    """
    Construct the retriever named by cfg.policy.retrieval_mode.

    Returns None for retrieval_mode == "none" (e.g. closed-book P3).
    """
    mode = cfg.policy.retrieval_mode
    bm25_path = bm25_index or cfg.retrieval.bm25_index
    faiss_path = faiss_index or cfg.retrieval.faiss_index

    if mode == "none":
        return None

    if mode == "bm25":
        return BM25Retriever.from_index(bm25_path)

    if mode == "vector":
        from medrag_adaptive.retrieval.vector_retriever import VectorRetriever
        return VectorRetriever.from_index(faiss_path)

    if mode == "hybrid":
        from medrag_adaptive.retrieval.vector_retriever import VectorRetriever
        from medrag_adaptive.retrieval.hybrid_retriever import HybridRetriever
        bm25 = BM25Retriever.from_index(bm25_path)
        vector = VectorRetriever.from_index(faiss_path)
        return HybridRetriever(bm25, vector, rrf_k=cfg.retrieval.rrf_k)

    raise ValueError(f"Unknown retrieval_mode '{mode}'")
