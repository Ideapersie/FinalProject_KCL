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


def _build_bm25(chunks, out: Path, force: bool) -> None:
    if out.exists() and not force:
        print(f"[skip] BM25 index already at {out} (use --force to rebuild)")
        return
    BM25Retriever.from_chunks(chunks).save(out)
    print(f"[ok  ] BM25 index built from {len(chunks)} chunks -> {out}")


def _build_faiss(chunks, out_dir: Path, model_name: str, force: bool) -> None:
    if (out_dir / "faiss.index").exists() and not force:
        print(f"[skip] FAISS index already at {out_dir} (use --force to rebuild)")
        return
    # Imported here so the BM25-only path never needs FAISS / sentence-transformers.
    from medrag_adaptive.retrieval.vector_retriever import VectorRetriever
    VectorRetriever.from_chunks(chunks, model_name=model_name).save(out_dir)
    print(f"[ok  ] FAISS index built from {len(chunks)} chunks -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build BM25 and/or FAISS index from a corpus.")
    ap.add_argument("--corpus", required=True, help="JSONL corpus of chunks")
    ap.add_argument("--bm25-out", default="indexes/bm25_pilot.pkl")
    ap.add_argument("--faiss-out", default="indexes/faiss_pilot",
                    help="directory for the FAISS index + chunk metadata")
    ap.add_argument("--which", choices=["bm25", "faiss", "both"], default="both")
    ap.add_argument("--embedding-model",
                    default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    chunks = _load_chunks(args.corpus)
    if not chunks:
        raise SystemExit(f"No chunks loaded from {args.corpus}")

    if args.which in ("bm25", "both"):
        _build_bm25(chunks, Path(args.bm25_out), args.force)
    if args.which in ("faiss", "both"):
        _build_faiss(chunks, Path(args.faiss_out), args.embedding_model, args.force)


if __name__ == "__main__":
    main()
