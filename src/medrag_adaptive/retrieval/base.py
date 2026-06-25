"""
retrieval/base.py — Abstract base class for retrievers.

Every concrete retriever (BM25, vector/FAISS, hybrid/RRF) implements this
interface so that policies can hold a single `Retriever` type regardless of
the underlying index. The signature matches `MockRetriever` in
tests/conftest.py exactly, so the test fixtures are drop-in substitutes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from medrag_adaptive.data.schema import Chunk


class Retriever(ABC):
    """Abstract retriever returning scored Chunks for a text query."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Chunk]:
        """
        Return up to `top_k` Chunks ranked best-first for `query`.

        Each returned Chunk has its `.score` set to the retriever's
        relevance score (BM25 weight, cosine similarity, or RRF score).
        """

    def close(self) -> None:
        """Release any resources (memory-mapped index, model). Default no-op."""
