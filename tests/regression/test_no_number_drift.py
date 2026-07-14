"""
tests/regression/test_no_number_drift.py — make number drift impossible.

The dissertation claims "one source of truth: every number is generated from the
logs." That claim was, until this test existed, enforced only by discipline — and
discipline failed: the report said P5 scored 54% while the logs said 54.5%,
because the answer-extractor was fixed in code but the emitted artefacts were
never regenerated.

This test regenerates every table and macro in memory and asserts they match what
is on disk. Hand-edit a number in results/tables/, or change the scorer without
re-running `scripts/generate_tables.py`, and the suite goes red.

It is deliberately a *regression* test, not a unit test: it reads the real logs.
It is skipped (not failed) when the logs are absent, so a fresh clone without the
run artefacts can still run the rest of the suite.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results" / "tables"
RAW = ROOT / "results" / "raw_logs"

pytestmark = pytest.mark.skipif(
    not (RAW / "p5_medcorp_mcq.jsonl").exists(),
    reason="run logs not present in this checkout",
)


def _load_generator():
    """Import scripts/generate_tables.py (not an installed package)."""
    spec = importlib.util.spec_from_file_location(
        "generate_tables", ROOT / "scripts" / "generate_tables.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["generate_tables"] = mod
    spec.loader.exec_module(mod)
    return mod


def _strip_header(text: str) -> str:
    """Drop the AUTO-GENERATED banner (it carries a timestamp and git SHA)."""
    return "\n".join(
        l for l in text.splitlines() if not l.startswith("%")
    ).strip()


@pytest.fixture(scope="module")
def regenerated(tmp_path_factory):
    """Regenerate every artefact into a temp dir, leaving the real ones alone."""
    gen = _load_generator()
    out = tmp_path_factory.mktemp("tables")
    gen.OUT = out
    gen.main()
    return out


@pytest.mark.parametrize("name", [
    "tab_setup.tex",
    "tab_main_results.tex",
    "tab_risk.tex",
    "numbers.tex",
])
def test_emitted_file_matches_logs(regenerated, name):
    """Every emitted file must equal what the logs currently produce."""
    on_disk = TABLES / name
    assert on_disk.exists(), (
        f"{name} missing — run: python scripts/generate_tables.py"
    )
    expected = _strip_header((regenerated / name).read_text(encoding="utf-8"))
    actual = _strip_header(on_disk.read_text(encoding="utf-8"))
    assert actual == expected, (
        f"{name} is out of date with the logs (or was hand-edited).\n"
        f"Regenerate with: python scripts/generate_tables.py"
    )


def test_macros_carry_the_corrected_scores(regenerated):
    """Guards the specific drift that motivated this test.

    P5's MCQ accuracy must be the RE-SCORED 54.5%, not the stale logged 54.0%.
    If someone reverts the extractor fix, or bypasses the canonical loader, this
    is the assertion that catches it.
    """
    text = (regenerated / "numbers.tex").read_text(encoding="utf-8")
    m = re.search(r"\\newcommand\{\\accPfiveMcq\}\{([\d.]+)\\%\}", text)
    assert m, "accPfiveMcq macro not emitted"
    assert float(m.group(1)) == pytest.approx(54.5), (
        "P5 MCQ accuracy is not the re-scored value — is the canonical loader "
        "still being used? See evaluation/loading.py."
    )


def test_no_literal_result_numbers_in_report():
    """Result numbers must come from macros, never be typed as literals.

    This is the guard that would have caught the original drift: the chapters
    hard-coded "54\\%" while the logs said 54.5\\%, and nothing noticed. Any
    headline result number typed directly into a chapter is a future drift bug,
    so they are banned outright — use the \\newcommand macros from numbers.tex.

    NOTE on comment-stripping: a LaTeX comment starts at an *unescaped* `%`, but
    the numbers we are hunting are written `46\\%` — so naively splitting on `%`
    deletes the very text being searched for. (That bug made an earlier version of
    this test pass vacuously.) Escaped percents are masked out first.
    """
    # Only numbers that are REGENERATED from the current logs are banned. Figures
    # from the superseded pilot-corpus runs (e.g. the 45.5% / 0.191 pilot P5) are
    # historical: they are not recomputed by generate_tables.py, so a literal is
    # the honest representation and a macro would be a lie about their provenance.
    # Those cells are daggered in the tables instead.
    banned = [
        "54.5", "54\\%", "46\\%",   # P5 / P1 MedCorp MCQ accuracy
        "62.0",                      # P3 closed-book accuracy
        "0.220", "0.211",            # P1 / P5 open-ended F1 (MedCorp)
    ]

    offenders = []
    for tex in sorted((ROOT / "report").glob("*.tex")):
        lines = []
        for line in tex.read_text(encoding="utf-8").splitlines():
            masked = line.replace("\\%", "\x00")      # protect escaped percents
            code = masked.split("%")[0]               # now strip real comments
            lines.append(code.replace("\x00", "\\%"))  # restore
        body = "\n".join(lines)
        for stale in banned:
            if stale in body:
                offenders.append(f"{tex.name}: literal '{stale}'")

    assert not offenders, (
        "Hard-coded result numbers found. Replace them with macros from "
        "results/tables/numbers.tex so they cannot drift from the logs:\n  "
        + "\n  ".join(offenders)
    )
