r"""
scripts/_count.py — tiny gate for the overnight batch loop.

Exits 0 if <jsonl> has at least <n> lines, else exits 1. The run_overnight.bat
loop uses this to decide whether a policy run finished (200 records) or needs
relaunching: `python scripts\_count.py out.jsonl 200 || goto step`.

Kept separate (not inline) because cmd.exe has no clean way to count file lines
and branch on it; this makes the batch loop robust to the ~per-run process kills.
"""

import io
import sys

path, need = sys.argv[1], int(sys.argv[2])
try:
    n = sum(1 for _ in io.open(path, encoding="utf-8"))
except FileNotFoundError:
    n = 0
print(f"{path}: {n}/{need}")
sys.exit(0 if n >= need else 1)
