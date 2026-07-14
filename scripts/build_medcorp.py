"""
scripts/build_medcorp.py — Download + index a real MedRAG corpus, resumable.

Standalone, single-process, NO LLM. Safe to launch and leave overnight: every
stage checkpoints to disk, so if the machine sleeps, the process is killed, or it
crashes, re-running continues from the last completed shard rather than restarting.

Stages (each idempotent):
  1. download   stream chosen MedRAG sub-corpora from HF -> data/corpora/medcorp.jsonl
  2. embed      embed chunks in fixed-size shards, each saved to
                indexes/<name>/_shards/emb_XXXX.npy (skips shards already present)
  3. assemble   concatenate shards -> FAISS IndexFlatIP + chunks.pkl (the layout
                VectorRetriever.from_index expects), and build the BM25 pickle.

Because embedding is the long stage and is shard-checkpointed, an interrupted run
loses at most one shard (default 2000 chunks, seconds of work).

Sources (verified 2026-07-02):
  textbooks  -> MedRAG/textbooks   (OK on HF, ~126K)
  pubmed     -> MedRAG/pubmed       (OK on HF, ~23.9M — only with an explicit --limit)
  statpearls -> NOT on HF (EmptyDatasetError); supply via --extra-jsonl if fetched
                from the MedRAG GitHub/NCBI route.

Usage (does nothing until you pass --sources; no auto-download on import):
  python scripts/build_medcorp.py --sources textbooks \
      --name medcorp_textbooks
  python scripts/build_medcorp.py --sources textbooks pubmed --limit-pubmed 300000 \
      --name medcorp_tp
  # add StatPearls fetched separately as JSONL of {id,title,content}:
  python scripts/build_medcorp.py --sources textbooks \
      --extra-jsonl data/corpora/statpearls.jsonl --name medcorp_full
"""

from __future__ import annotations

import argparse
import io
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional

sys.stdout.reconfigure(encoding="utf-8")

CORPORA = Path("data/corpora")
INDEXES = Path("indexes")

# Verified-good HF paths. statpearls intentionally absent (dead on HF).
HF_PATHS = {
    "textbooks": "MedRAG/textbooks",
    "pubmed": "MedRAG/pubmed",
}


def log(msg: str, logfile: Optional[Path]) -> None:
    print(msg, flush=True)
    if logfile:
        with logfile.open("a", encoding="utf-8") as fh:
            fh.write(msg + "\n")


# ── stage 1: download -> unified JSONL ───────────────────────────────

def _rows_from_hf(source: str, limit: Optional[int]) -> Iterator[Dict]:
    from datasets import load_dataset

    ds = load_dataset(HF_PATHS[source], split="train", streaming=True)
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        # MedRAG schema: id, title, content, contents. `content` is the passage.
        yield {
            "chunk_id": f"{source}_{row.get('id', i)}",
            "source": source,
            "title": row.get("title", "") or "",
            "text": row.get("content") or row.get("contents") or "",
            "score": 0.0,
        }


def _rows_from_jsonl(path: Path, source: str) -> Iterator[Dict]:
    for i, line in enumerate(io.open(path, encoding="utf-8")):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        yield {
            "chunk_id": d.get("chunk_id") or f"{source}_{d.get('id', i)}",
            "source": d.get("source", source),
            "title": d.get("title", "") or "",
            "text": d.get("text") or d.get("content") or d.get("contents") or "",
            "score": 0.0,
        }


def stage_download(sources: List[str], limit_pubmed: Optional[int],
                   extra_jsonl: Optional[Path], out: Path,
                   logfile: Path) -> int:
    """Write the unified corpus JSONL. Resumable: if out exists, reuse it."""
    if out.exists():
        n = sum(1 for _ in io.open(out, encoding="utf-8"))
        log(f"[skip] corpus already at {out} ({n} chunks)", logfile)
        return n
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".partial")
    n = 0
    with tmp.open("w", encoding="utf-8") as fh:
        for source in sources:
            limit = limit_pubmed if source == "pubmed" else None
            log(f"[get ] {source} (limit={limit}) ...", logfile)
            for rec in _rows_from_hf(source, limit):
                if rec["text"]:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n += 1
                    if n % 20000 == 0:
                        log(f"       {n} chunks written", logfile)
        if extra_jsonl:
            log(f"[get ] extra {extra_jsonl} ...", logfile)
            for rec in _rows_from_jsonl(extra_jsonl, "statpearls"):
                if rec["text"]:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n += 1
    tmp.rename(out)   # atomic: only a complete download becomes the real file
    log(f"[ok  ] corpus written: {n} chunks -> {out}", logfile)
    return n


# ── stage 2: embed in shards (resumable) ─────────────────────────────

def _iter_chunks(corpus: Path) -> Iterator[Dict]:
    for line in io.open(corpus, encoding="utf-8"):
        line = line.strip()
        if line:
            yield json.loads(line)


def stage_embed(corpus: Path, work: Path, shard_size: int,
                model_name: str, logfile: Path) -> int:
    """Embed chunks into per-shard .npy files. Skips shards already on disk."""
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer

    shard_dir = work / "_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    all_chunks = list(_iter_chunks(corpus))
    total = len(all_chunks)
    n_shards = (total + shard_size - 1) // shard_size
    log(f"[embed] {total} chunks -> {n_shards} shards of {shard_size}", logfile)

    model = None  # lazy: don't load the model if every shard is already done
    for s in range(n_shards):
        shard_path = shard_dir / f"emb_{s:05d}.npy"
        if shard_path.exists():
            continue  # resume: already embedded
        if model is None:
            # Use the GPU when there is one. On the CPU-only laptop this stays
            # "cpu" exactly as before; on a Colab T4 it cuts embedding 426K
            # chunks from roughly an hour to ~10 minutes, which matters because
            # a free Colab session is capped at ~12 h.
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
            log(f"[embed] embedding on {device}", logfile)
            model = SentenceTransformer(model_name, device=device)
        lo, hi = s * shard_size, min((s + 1) * shard_size, total)
        texts = [f"{c['title']} {c['text']}" for c in all_chunks[lo:hi]]
        vecs = model.encode(texts, convert_to_numpy=True,
                            normalize_embeddings=False,
                            show_progress_bar=False).astype("float32")
        faiss.normalize_L2(vecs)
        np.save(shard_path, vecs)
        log(f"[embed] shard {s+1}/{n_shards} ({hi}/{total})", logfile)
    return total


# ── stage 3: assemble FAISS + chunks.pkl + BM25 ──────────────────────

def stage_assemble(corpus: Path, work: Path, model_name: str,
                   bm25_out: Path, logfile: Path) -> None:
    import numpy as np
    import faiss
    from medrag_adaptive.data.schema import Chunk
    from medrag_adaptive.retrieval.bm25_retriever import BM25Retriever

    shard_dir = work / "_shards"
    shard_files = sorted(shard_dir.glob("emb_*.npy"))
    if not shard_files:
        raise SystemExit("no embedding shards found — run embed stage first")

    log(f"[asm ] concatenating {len(shard_files)} shards", logfile)
    mats = [np.load(f) for f in shard_files]
    embeddings = np.vstack(mats)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(work / "faiss.index"))
    log(f"[asm ] FAISS index: {index.ntotal} vecs x {dim}d -> {work/'faiss.index'}", logfile)

    chunks = [Chunk(chunk_id=c["chunk_id"], source=c["source"],
                    title=c["title"], text=c["text"], score=0.0)
              for c in _iter_chunks(corpus)]
    if len(chunks) != index.ntotal:
        raise SystemExit(f"chunk/vector count mismatch: {len(chunks)} vs {index.ntotal}")
    with open(work / "chunks.pkl", "wb") as fh:
        pickle.dump({"chunks": chunks, "model_name": model_name}, fh)
    log(f"[asm ] chunks.pkl written ({len(chunks)} chunks)", logfile)

    # BM25 leg (in-memory; may be heavy at 400K — measured, not assumed)
    if bm25_out.exists():
        log(f"[skip] BM25 already at {bm25_out}", logfile)
    else:
        bm25_out.parent.mkdir(parents=True, exist_ok=True)
        BM25Retriever.from_chunks(chunks).save(bm25_out)
        log(f"[asm ] BM25 index -> {bm25_out}", logfile)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a real MedRAG corpus index (resumable).")
    ap.add_argument("--sources", nargs="+", default=[],
                    choices=list(HF_PATHS), help="HF sub-corpora to include")
    ap.add_argument("--extra-jsonl", type=Path, default=None,
                    help="extra corpus JSONL (e.g. StatPearls fetched from GitHub/NCBI)")
    ap.add_argument("--limit-pubmed", type=int, default=None,
                    help="cap pubmed rows (it is 23.9M)")
    ap.add_argument("--name", default="medcorp", help="index dir name under indexes/")
    ap.add_argument("--shard-size", type=int, default=2000)
    ap.add_argument("--embedding-model",
                    default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--stage", choices=["all", "download", "embed", "assemble"],
                    default="all")
    args = ap.parse_args()

    if not args.sources and not args.extra_jsonl:
        raise SystemExit("nothing to build: pass --sources and/or --extra-jsonl")

    work = INDEXES / f"faiss_{args.name}"
    work.mkdir(parents=True, exist_ok=True)
    logfile = work / "build.log"
    corpus = CORPORA / f"{args.name}.jsonl"
    bm25_out = INDEXES / f"bm25_{args.name}.pkl"

    log(f"=== build_medcorp {args.name} stage={args.stage} ===", logfile)
    if args.stage in ("all", "download"):
        stage_download(args.sources, args.limit_pubmed, args.extra_jsonl, corpus, logfile)
    if args.stage in ("all", "embed"):
        stage_embed(corpus, work, args.shard_size, args.embedding_model, logfile)
    if args.stage in ("all", "assemble"):
        stage_assemble(corpus, work, args.embedding_model, bm25_out, logfile)
    log(f"=== done: index at {work}, bm25 at {bm25_out} ===", logfile)


if __name__ == "__main__":
    main()
