"""
retrieval/vector_retriever.py — Dense (semantic) retriever: FAISS + MiniLM.

Where BM25 matches on shared words, the vector retriever matches on meaning:
it embeds chunks and queries into a shared vector space with a sentence-
transformer and ranks by cosine similarity, so "heart attack" can retrieve a
chunk about "myocardial infarction" that BM25 would miss.

Index: FAISS IndexFlatIP (exact inner product). After L2-normalising the
embeddings, inner product equals cosine similarity. IndexFlatIP is exact (no
approximation), deterministic, and CPU-only — the right choice for a
reproducible, offline, resource-constrained system.

Embedding model: sentence-transformers/all-MiniLM-L6-v2 (384-dim, ~80 MB),
read from cfg.retrieval.embedding_model. Loaded lazily so importing this module
costs nothing until a retriever is actually built.

Two constructors mirror BM25Retriever:
    from_chunks(chunks, model_name)  — embed + build in memory (tests, small).
    from_index(dir_path)             — load a FAISS index + chunk metadata
                                       built by scripts/build_indexes.py.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np

from medrag_adaptive.data.schema import Chunk
from medrag_adaptive.retrieval.base import Retriever

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _load_model(model_name: str):
    """Lazily import and construct the sentence-transformer (CPU)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device="cpu")


def _embed(model, texts: List[str]) -> np.ndarray:
    """Embed texts and L2-normalise so inner product == cosine similarity."""
    import faiss

    vecs = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=False,   # we normalise explicitly below
        show_progress_bar=False,
    ).astype("float32")
    faiss.normalize_L2(vecs)
    return vecs


class VectorRetriever(Retriever):
    """Dense retriever over a FAISS IndexFlatIP of MiniLM embeddings."""

    def __init__(self, chunks: List[Chunk], index, model_name: str,
                 model=None) -> None:
        self._chunks = chunks
        self._index = index
        self._model_name = model_name
        self._model = model            # may be None until first query

    # ── constructors ───────────────────────────────────────────────

    @classmethod
    def from_chunks(cls, chunks: List[Chunk],
                    model_name: str = _DEFAULT_MODEL) -> "VectorRetriever":
        import faiss

        if not chunks:
            raise ValueError("VectorRetriever requires a non-empty corpus")
        model = _load_model(model_name)
        embeddings = _embed(model, [f"{c.title} {c.text}" for c in chunks])
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return cls(chunks, index, model_name, model)

    @classmethod
    def from_index(cls, dir_path: str | Path) -> "VectorRetriever":
        import faiss

        d = Path(dir_path)
        index = faiss.read_index(str(d / "faiss.index"))
        with open(d / "chunks.pkl", "rb") as fh:
            payload = pickle.load(fh)
        return cls(payload["chunks"], index, payload["model_name"], model=None)

    def save(self, dir_path: str | Path) -> None:
        import faiss

        d = Path(dir_path)
        d.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(d / "faiss.index"))
        with open(d / "chunks.pkl", "wb") as fh:
            pickle.dump({"chunks": self._chunks, "model_name": self._model_name}, fh)

    # ── retrieval ──────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> List[Chunk]:
        if self._model is None:               # lazy-load on first query after from_index
            self._model = _load_model(self._model_name)
        q = _embed(self._model, [query])      # shape [1, dim], normalised
        k = min(top_k, len(self._chunks))
        scores, idxs = self._index.search(q, k)
        results: List[Chunk] = []
        for score, i in zip(scores[0], idxs[0]):
            if i < 0:
                continue
            c = self._chunks[i]
            results.append(Chunk(chunk_id=c.chunk_id, source=c.source,
                                 title=c.title, text=c.text, score=float(score)))
        return results
