"""
scripts/build_indexes.py — Build retrieval indexes from a chunk corpus.

Phase C scope: BM25 leg only. Reads a JSONL corpus of Chunks and pickles a
BM25Retriever to the configured index path. The FAISS vector leg and RRF hybrid
index are Phase D and are not built here.

Idempotent: skips an existing index unless --force is given.

Usage:
    python scripts/build_indexes.py \
        --corpus data/corpora/pilot_corpus.jsonl \
        --bm25-out indexes/bm25_pilot.pkl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from medrag_adaptive.data.schema import Chunk
from medrag_adaptive.retrieval.bm25_retriever import BM25Retriever


def _load_chunks(corpus_path: str | Path) -> List[Chunk]:
    chunks: List[Chunk] = []
    with open(corpus_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            chunks.append(Chunk(
                chunk_id=d["chunk_id"],
                source=d["source"],
                title=d["title"],
                text=d["text"],
                score=float(d.get("score", 0.0)),
            ))
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser(description="Build BM25 index from a chunk corpus.")
    ap.add_argument("--corpus", required=True, help="JSONL corpus of chunks")
    ap.add_argument("--bm25-out", default="indexes/bm25_pilot.pkl")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = Path(args.bm25_out)
    if out.exists() and not args.force:
        print(f"[skip] BM25 index already at {out} (use --force to rebuild)")
        return

    chunks = _load_chunks(args.corpus)
    if not chunks:
        raise SystemExit(f"No chunks loaded from {args.corpus}")

    retriever = BM25Retriever.from_chunks(chunks)
    retriever.save(out)
    print(f"[ok  ] BM25 index built from {len(chunks)} chunks -> {out}")


if __name__ == "__main__":
    main()
