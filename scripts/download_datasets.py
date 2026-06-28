"""
scripts/download_datasets.py — Fetch MIRAGE and the open-ended QA set into data/raw/.

Idempotent: skips files that already exist unless --force is given. MIRAGE's
benchmark.json comes from the MedRAG repository; the open-ended set comes from
the HuggingFace Hub. Network access is required.

Open-ended source: the project originally targeted RAGCare-QA, but that Hub path
(`RAGCare/RAGCare-QA`) no longer resolves (DatasetNotFoundError). We substitute
**PubMedQA** (`qiaojin/PubMedQA`, config `pqa_labeled`): 1,000 expert-annotated
biomedical research questions, each with a paragraph `long_answer` (the abstract
conclusion). This is genuine free-text / paragraph Q&A — exactly what Phase 3
needs to move past MIRAGE's all-MCQ subsets — and loads without trust_remote_code.
Records are saved as JSONL with `question` + `long_answer`, which the existing
open-ended loader (`load_ragcare`, key-flexible) reads directly; scoring is
token-F1. The `--only ragcare` flag is kept as an alias for the open-ended set.

Usage:
    python scripts/download_datasets.py                 # both, skip existing
    python scripts/download_datasets.py --only mirage   # one dataset
    python scripts/download_datasets.py --only ragcare  # open-ended (PubMedQA)
    python scripts/download_datasets.py --dry-run       # show plan, fetch nothing
    python scripts/download_datasets.py --limit 200     # cap open-ended rows saved
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

RAW = Path("data/raw")

# MedRAG MIRAGE benchmark.json (subset-keyed: mmlu/medqa/medmcqa/pubmedqa/bioasq).
MIRAGE_URL = "https://raw.githubusercontent.com/Teddy-XiongGZ/MIRAGE/main/benchmark.json"
MIRAGE_OUT = RAW / "mirage" / "benchmark.json"

# Open-ended QA on the HuggingFace Hub. RAGCare-QA's path is dead, so we use
# PubMedQA's expert-labeled split (pqa_labeled) as the paragraph-answer source.
OPENQA_HF = "qiaojin/PubMedQA"
OPENQA_CONFIG = "pqa_labeled"
OPENQA_SPLIT = "train"  # pqa_labeled ships a single 1,000-row split named "train"
OPENQA_OUT = RAW / "openqa" / "pubmedqa_labeled.jsonl"


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def fetch_mirage(force: bool, dry_run: bool) -> None:
    if MIRAGE_OUT.exists() and not force:
        print(f"[skip] MIRAGE already at {MIRAGE_OUT}")
        return
    print(f"[get ] MIRAGE {MIRAGE_URL} -> {MIRAGE_OUT}")
    if dry_run:
        return
    _ensure_parent(MIRAGE_OUT)
    urllib.request.urlretrieve(MIRAGE_URL, MIRAGE_OUT)
    print(f"[ok  ] MIRAGE saved ({MIRAGE_OUT.stat().st_size} bytes)")


def fetch_openqa(force: bool, dry_run: bool, limit: int | None) -> None:
    if OPENQA_OUT.exists() and not force:
        print(f"[skip] open-ended QA already at {OPENQA_OUT}")
        return
    print(f"[get ] open-ended QA {OPENQA_HF}:{OPENQA_CONFIG} -> {OPENQA_OUT}")
    if dry_run:
        return
    from datasets import load_dataset

    ds = load_dataset(OPENQA_HF, OPENQA_CONFIG, split=OPENQA_SPLIT)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    _ensure_parent(OPENQA_OUT)
    n = 0
    with OPENQA_OUT.open("w", encoding="utf-8") as fh:
        for row in ds:
            # Normalise to the open-ended loader's shape: a question + a gold
            # paragraph answer. PubMedQA's `long_answer` is the abstract
            # conclusion; `final_decision` (yes/no/maybe) is kept for reference.
            rec = {
                "id": row.get("pubid", n),
                "question": row.get("question", ""),
                "long_answer": row.get("long_answer", ""),
                "final_decision": row.get("final_decision", ""),
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"[ok  ] open-ended QA saved ({n} rows)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Download MIRAGE + RAGCare-QA.")
    ap.add_argument("--only", choices=["mirage", "ragcare"], default=None)
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    ap.add_argument("--dry-run", action="store_true", help="print plan, fetch nothing")
    ap.add_argument("--limit", type=int, default=None, help="cap open-ended rows saved")
    args = ap.parse_args()

    if args.only in (None, "mirage"):
        fetch_mirage(args.force, args.dry_run)
    if args.only in (None, "ragcare"):
        fetch_openqa(args.force, args.dry_run, args.limit)


if __name__ == "__main__":
    main()
