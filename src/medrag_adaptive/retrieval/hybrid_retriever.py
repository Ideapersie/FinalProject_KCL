"""
retrieval/hybrid_retriever.py — Reciprocal Rank Fusion of BM25 + vector.

Lexical (BM25) and dense (vector) retrieval have complementary strengths: BM25
nails exact terminology and rare tokens (drug names, abbreviations); the vector
retriever captures paraphrase and synonymy. The hybrid retriever runs both and
fuses their rankings with Reciprocal Rank Fusion (RRF):

    score(d) = Σ_i  1 / (k + rank_i(d))

where rank_i(d) is d's 1-based rank in retriever i's list (only over the lists
where d appears) and k is a smoothing constant (default 60, the value from the
original RRF paper). RRF needs no score normalisation across retrievers — it
uses only ranks — which is why it is robust to BM25 and cosine living on
different scales.

Each retriever is queried for an expanded candidate pool (top_k * pool_factor)
so that documents ranked highly by only one retriever can still surface.
"""

from __future__ import annotations

from typing import Dict, List

from medrag_adaptive.data.schema import Chunk
from medrag_adaptive.retrieval.base import Retriever


class HybridRetriever(Retriever):
    """Fuse two retrievers' rankings via Reciprocal Rank Fusion."""

    def __init__(self, bm25: Retriever, vector: Retriever,
                 rrf_k: int = 60, pool_factor: int = 4) -> None:
        self._bm25 = bm25
        self._vector = vector
        self._rrf_k = rrf_k
        self._pool_factor = pool_factor

    def retrieve(self, query: str, top_k: int = 5) -> List[Chunk]:
        pool = top_k * self._pool_factor
        bm25_hits = self._bm25.retrieve(query, top_k=pool)
        vector_hits = self._vector.retrieve(query, top_k=pool)

        fused: Dict[str, float] = {}
        chunk_by_id: Dict[str, Chunk] = {}

        for hits in (bm25_hits, vector_hits):
            for rank, chunk in enumerate(hits, start=1):
                fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) + \
                    1.0 / (self._rrf_k + rank)
                chunk_by_id.setdefault(chunk.chunk_id, chunk)

        ranked_ids = sorted(fused, key=lambda cid: fused[cid], reverse=True)[:top_k]
        results: List[Chunk] = []
        for cid in ranked_ids:
            c = chunk_by_id[cid]
            results.append(Chunk(chunk_id=c.chunk_id, source=c.source,
                                 title=c.title, text=c.text, score=fused[cid]))
        return results
