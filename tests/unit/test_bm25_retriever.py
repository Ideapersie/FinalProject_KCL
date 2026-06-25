"""Unit tests for the BM25 retriever over the mock corpus."""

import json
from pathlib import Path

import pytest

from medrag_adaptive.data.schema import Chunk
from medrag_adaptive.retrieval.bm25_retriever import BM25Retriever

CORPUS = Path(__file__).parent.parent / "fixtures" / "mock_corpus.jsonl"


@pytest.fixture
def corpus_chunks():
    chunks = []
    with CORPUS.open() as fh:
        for line in fh:
            chunks.append(Chunk(**json.loads(line)))
    return chunks


def test_relevant_chunk_ranks_first(corpus_chunks):
    r = BM25Retriever.from_chunks(corpus_chunks)
    results = r.retrieve("phrenic nerve diaphragm", top_k=3)
    assert results[0].chunk_id == "mock_001"   # phrenic-nerve chunk
    assert results[0].score > 0


def test_respects_top_k(corpus_chunks):
    r = BM25Retriever.from_chunks(corpus_chunks)
    assert len(r.retrieve("anything", top_k=2)) == 2


def test_warfarin_query_finds_warfarin_chunk(corpus_chunks):
    r = BM25Retriever.from_chunks(corpus_chunks)
    results = r.retrieve("warfarin ciprofloxacin interaction", top_k=1)
    assert results[0].chunk_id == "mock_005"


def test_empty_corpus_raises():
    with pytest.raises(ValueError):
        BM25Retriever.from_chunks([])


def test_save_and_load_roundtrip(corpus_chunks, tmp_path):
    idx = tmp_path / "bm25.pkl"
    BM25Retriever.from_chunks(corpus_chunks).save(idx)
    loaded = BM25Retriever.from_index(idx)
    results = loaded.retrieve("phrenic nerve", top_k=1)
    assert results[0].chunk_id == "mock_001"
