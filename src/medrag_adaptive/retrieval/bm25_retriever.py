"""
retrieval/bm25_retriever.py — Lexical BM25 retriever.

Wraps rank_bm25.BM25Okapi over a corpus of Chunks. BM25 is the lexical leg of
the hybrid retriever and the only retriever available on the low tier (no
embedding model). Two constructors:

    from_chunks(chunks)  — build in memory (tests, small corpora).
    from_index(path)     — load a pickle built by scripts/build_indexes.py
                           (the real StatPearls + Textbooks corpus).

retrieve() returns the top_k Chunks with `.score` set to the BM25 score.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import List

from rank_bm25 import BM25Okapi

from medrag_adaptive.data.schema import Chunk
from medrag_adaptive.retrieval.base import Retriever

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever(Retriever):
    """BM25 lexical retriever over a Chunk corpus."""

    def __init__(self, chunks: List[Chunk], bm25: BM25Okapi) -> None:
        self._chunks = chunks
        self._bm25 = bm25

    # ── constructors ───────────────────────────────────────────────

    @classmethod
    def from_chunks(cls, chunks: List[Chunk]) -> "BM25Retriever":
        if not chunks:
            raise ValueError("BM25Retriever requires a non-empty corpus")
        tokenized = [_tokenize(f"{c.title} {c.text}") for c in chunks]
        return cls(chunks, BM25Okapi(tokenized))

    @classmethod
    def from_index(cls, path: str | Path) -> "BM25Retriever":
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        return cls(payload["chunks"], payload["bm25"])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({"chunks": self._chunks, "bm25": self._bm25}, fh)

    # ── retrieval ──────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> List[Chunk]:
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(
            range(len(self._chunks)), key=lambda i: scores[i], reverse=True
        )[:top_k]
        results: List[Chunk] = []
        for i in ranked:
            c = self._chunks[i]
            results.append(
                Chunk(chunk_id=c.chunk_id, source=c.source, title=c.title,
                      text=c.text, score=float(scores[i]))
            )
        return results
