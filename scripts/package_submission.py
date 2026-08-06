"""
scripts/package_submission.py — build a clean source-code submission archive.

Includes everything an examiner needs to read the code, run the tests, and
regenerate every reported number from the committed logs. Excludes the multi-GB
rebuildable artifacts (weights, indexes, corpus), reference PDFs, caches, and the
internal working notes / AI-context files that are not part of the deliverable.

Usage:
    python scripts/package_submission.py           # -> submission.zip at repo root
    python scripts/package_submission.py -o out.zip

Verify afterwards (the archive is self-sufficient for these two):
    unzip -q submission.zip -d /tmp/sub && cd /tmp/sub/<top>
    pip install -e ".[dev]" && python -m pytest -q
    python scripts/generate_tables.py && python scripts/generate_figures.py
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOP = "medrag_adaptive_submission"   # single top-level folder inside the zip

# Directories included in full (minus the exclusions below).
INCLUDE_DIRS = [
    "src", "scripts", "configs", "tests", "notebooks", "report", "results", "docs",
]
# Sub-trees included from an otherwise-excluded parent (data/ is mostly the corpus).
INCLUDE_SUBTREES = ["data/raw"]
# Individual files at the repo root.
INCLUDE_FILES = [
    "README.md", "requirements.txt", "pyproject.toml", "LICENSE", ".gitignore",
]

# Directory names pruned anywhere in the tree (caches, VCS, AI tooling, unrelated).
EXCLUDE_DIR_NAMES = {
    "__pycache__", ".git", ".venv", "venv", "env", ".pytest_cache", ".ruff_cache",
    ".mypy_cache", ".ipynb_checkpoints", "node_modules", ".claude", "superpowers",
    "_shards",                      # embed checkpoints under indexes (belt-and-braces)
    "design_handoff_portfolio",
}
# File glob patterns excluded anywhere.
EXCLUDE_FILE_GLOBS = [
    "*.gguf", "*.stackdump", "*.pyc", "*.log", "*.faiss", "*.index",
    "bm25_*.pkl", "faiss.index", "chunks.pkl",
]
# Specific paths (repo-relative, posix) that must never ship.
EXCLUDE_PATHS = set()


def _excluded_file(rel_posix: str) -> bool:
    name = rel_posix.rsplit("/", 1)[-1]
    if rel_posix in EXCLUDE_PATHS:
        return True
    return any(fnmatch.fnmatch(name, g) for g in EXCLUDE_FILE_GLOBS)


def _iter_tree(base: Path):
    """Yield files under `base`, pruning excluded directory names."""
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIR_NAMES]
        for fn in filenames:
            p = Path(dirpath) / fn
            rel = p.relative_to(ROOT).as_posix()
            if not _excluded_file(rel):
                yield p, rel


def collect() -> list[tuple[Path, str]]:
    seen: dict[str, Path] = {}
    for d in INCLUDE_DIRS + INCLUDE_SUBTREES:
        base = ROOT / d
        if not base.exists():
            print(f"  [warn] include path missing, skipped: {d}")
            continue
        for p, rel in _iter_tree(base):
            seen[rel] = p
    for f in INCLUDE_FILES:
        p = ROOT / f
        if p.exists():
            seen[f] = p
        else:
            print(f"  [warn] include file missing, skipped: {f}")
    return sorted(((p, rel) for rel, p in seen.items()), key=lambda t: t[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default="submission.zip")
    args = ap.parse_args()
    out = (ROOT / args.output).resolve()

    items = collect()
    raw_bytes = sum(p.stat().st_size for p, _ in items)

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p, rel in items:
            z.write(p, arcname=f"{TOP}/{rel}")

    by_top: dict[str, list[int]] = {}
    for p, rel in items:
        key = rel.split("/", 1)[0]
        by_top.setdefault(key, []).append(p.stat().st_size)

    print(f"\nwrote {out.name}")
    print(f"  files      : {len(items)}")
    print(f"  uncompressed: {raw_bytes/1e6:.1f} MB")
    print(f"  zip size   : {out.stat().st_size/1e6:.1f} MB")
    print("  by top-level:")
    for k in sorted(by_top, key=lambda k: -sum(by_top[k])):
        sz = sum(by_top[k]) / 1e6
        print(f"    {k:16s} {len(by_top[k]):4d} files  {sz:7.1f} MB")


if __name__ == "__main__":
    main()
