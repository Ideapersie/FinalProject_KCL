"""
tests/unit/test_vector_hybrid.py — Vector and hybrid retriever tests.

The vector tests load the real MiniLM model (small, CPU) and are marked slow so
they can be skipped in the fast loop. The hybrid RRF-fusion tests use stub
retrievers and need no model, so they always run fast.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest

from medrag_adaptive.data.schema import Chunk
from medrag_adaptive.retrieval.base import Retriever
from medrag_adaptive.retrieval.hybrid_retriever import HybridRetriever

CORPUS = Path(__file__).parent.parent / "fixtures" / "mock_corpus.jsonl"


@pytest.fixture
def corpus_chunks() -> List[Chunk]:
    chunks = []
    with CORPUS.open() as fh:
        for line in fh:
            chunks.append(Chunk(**json.loads(line)))
    return chunks


# ─────────────────────────────────────────────────────────────────
# Hybrid retriever — RRF fusion (no model needed; stub retrievers)
# ─────────────────────────────────────────────────────────────────

class _StubRetriever(Retriever):
    """Returns a fixed, pre-ranked chunk list regardless of query."""

    def __init__(self, chunks: List[Chunk]) -> None:
        self._chunks = chunks

    def retrieve(self, query: str, top_k: int = 5) -> List[Chunk]:
        return self._chunks[:top_k]


def _mk(cid: str) -> Chunk:
    return Chunk(chunk_id=cid, source="s", title=cid, text=cid)


class TestHybridRRF:
    def test_consensus_doc_ranks_first(self):
        # C appears high in BOTH lists → should win after fusion.
        bm25 = _StubRetriever([_mk("A"), _mk("C"), _mk("B")])
        vec = _StubRetriever([_mk("C"), _mk("D"), _mk("E")])
        h = HybridRetriever(bm25, vec, rrf_k=60, pool_factor=4)
        results = h.retrieve("q", top_k=3)
        assert results[0].chunk_id == "C"

    def test_union_of_sources(self):
        bm25 = _StubRetriever([_mk("A"), _mk("B")])
        vec = _StubRetriever([_mk("C"), _mk("D")])
        h = HybridRetriever(bm25, vec)
        ids = {c.chunk_id for c in h.retrieve("q", top_k=4)}
        assert ids == {"A", "B", "C", "D"}

    def test_rrf_score_formula(self):
        # Single doc ranked 1st in both: score = 1/(60+1) + 1/(60+1).
        bm25 = _StubRetriever([_mk("X")])
        vec = _StubRetriever([_mk("X")])
        h = HybridRetriever(bm25, vec, rrf_k=60)
        r = h.retrieve("q", top_k=1)
        assert r[0].score == pytest.approx(2 / 61)

    def test_respects_top_k(self):
        bm25 = _StubRetriever([_mk(c) for c in "ABCDE"])
        vec = _StubRetriever([_mk(c) for c in "FGHIJ"])
        h = HybridRetriever(bm25, vec)
        assert len(h.retrieve("q", top_k=3)) == 3


# ─────────────────────────────────────────────────────────────────
# Vector retriever — real MiniLM (slow)
# ─────────────────────────────────────────────────────────────────

@pytest.mark.slow
class TestVectorRetriever:
    def test_semantic_match_beats_lexical_gap(self, corpus_chunks):
        """A paraphrase with no shared keywords still retrieves the right chunk."""
        from medrag_adaptive.retrieval.vector_retriever import VectorRetriever
        r = VectorRetriever.from_chunks(corpus_chunks)
        # mock_002 is the metformin/diabetes chunk; query avoids its exact words.
        results = r.retrieve("first-line drug for high blood sugar", top_k=2)
        assert any(c.chunk_id == "mock_002" for c in results)

    def test_respects_top_k(self, corpus_chunks):
        from medrag_adaptive.retrieval.vector_retriever import VectorRetriever
        r = VectorRetriever.from_chunks(corpus_chunks)
        assert len(r.retrieve("anything", top_k=2)) == 2

    def test_save_and_load_roundtrip(self, corpus_chunks, tmp_path):
        from medrag_adaptive.retrieval.vector_retriever import VectorRetriever
        r = VectorRetriever.from_chunks(corpus_chunks)
        r.save(tmp_path / "vec")
        loaded = VectorRetriever.from_index(tmp_path / "vec")
        results = loaded.retrieve("phrenic nerve", top_k=1)
        assert len(results) == 1
        assert results[0].score > 0
