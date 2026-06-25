"""
scripts/download_datasets.py — Fetch MIRAGE and RAGCare-QA into data/raw/.

Idempotent: skips files that already exist unless --force is given. MIRAGE's
benchmark.json comes from the MedRAG repository; RAGCare-QA comes from the
HuggingFace Hub. Network access is required.

Usage:
    python scripts/download_datasets.py                 # both, skip existing
    python scripts/download_datasets.py --only mirage   # one dataset
    python scripts/download_datasets.py --dry-run       # show plan, fetch nothing
    python scripts/download_datasets.py --limit 200     # cap RAGCare rows saved
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

# RAGCare-QA on the HuggingFace Hub (resolved via the datasets library).
RAGCARE_HF = "RAGCare/RAGCare-QA"
RAGCARE_OUT = RAW / "ragcare" / "ragcare_qa.jsonl"


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


def fetch_ragcare(force: bool, dry_run: bool, limit: int | None) -> None:
    if RAGCARE_OUT.exists() and not force:
        print(f"[skip] RAGCare already at {RAGCARE_OUT}")
        return
    print(f"[get ] RAGCare {RAGCARE_HF} -> {RAGCARE_OUT}")
    if dry_run:
        return
    from datasets import load_dataset

    ds = load_dataset(RAGCARE_HF, split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    _ensure_parent(RAGCARE_OUT)
    with RAGCARE_OUT.open("w", encoding="utf-8") as fh:
        for row in ds:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[ok  ] RAGCare saved ({len(ds)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Download MIRAGE + RAGCare-QA.")
    ap.add_argument("--only", choices=["mirage", "ragcare"], default=None)
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    ap.add_argument("--dry-run", action="store_true", help="print plan, fetch nothing")
    ap.add_argument("--limit", type=int, default=None, help="cap RAGCare rows saved")
    args = ap.parse_args()

    if args.only in (None, "mirage"):
        fetch_mirage(args.force, args.dry_run)
    if args.only in (None, "ragcare"):
        fetch_ragcare(args.force, args.dry_run, args.limit)


if __name__ == "__main__":
    main()
