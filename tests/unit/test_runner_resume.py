"""Resume-safety test for scripts/run_experiment.py (mock-backed, no model)."""

import sys
from pathlib import Path

import pytest

# scripts/ is not a package; add it to the path so we can import the runner.
SCRIPTS = Path(__file__).parent.parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_experiment  # noqa: E402

from medrag_adaptive.config import load_config  # noqa: E402
from medrag_adaptive.data.schema import load_records  # noqa: E402


@pytest.fixture
def cfg():
    c = load_config(policy="configs/policies/p3_closed_book.yaml")
    c.dataset = "mirage"
    c.profiling.track_energy = False
    return c


def test_runner_writes_then_resumes(cfg, mock_llm_high, tmp_path):
    dataset = str(Path(__file__).parent.parent / "fixtures" / "mirage_sample.json")
    out = str(tmp_path / "p3.jsonl")

    first = run_experiment.run(cfg, dataset, out, mock_llm_high)
    assert first == 3                       # 3 questions in the fixture

    # Second run must skip all already-logged ids → zero new records.
    second = run_experiment.run(cfg, dataset, out, mock_llm_high)
    assert second == 0

    records = load_records(out)
    ids = [r.question_id for r in records]
    assert len(ids) == len(set(ids)) == 3   # no duplicates


def test_runner_scores_mcq(cfg, mock_llm_high, tmp_path):
    dataset = str(Path(__file__).parent.parent / "fixtures" / "mirage_sample.json")
    out = str(tmp_path / "p3.jsonl")
    run_experiment.run(cfg, dataset, out, mock_llm_high)
    records = load_records(out)
    # mock_llm_high answers "A"; mmlu_0 gold is "B" → scored incorrect, not crashing.
    rec = next(r for r in records if r.question_id == "mmlu_0")
    assert rec.is_correct is False
    assert rec.retrieval_triggered is False
    assert rec.latency_ns > 0
