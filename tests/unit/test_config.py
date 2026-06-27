"""
tests/unit/test_config.py — Unit tests for config.py.

Tests that the YAML deep-merge, Pydantic validation, and field accessors
work correctly. No model files or external services required.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from medrag_adaptive.config import load_config, ProjectConfig, _deep_merge


# ─────────────────────────────────────────────────────────────────
# Helper: write a temp YAML file
# ─────────────────────────────────────────────────────────────────

def _write_yaml(data: dict, path: Path) -> None:
    with path.open("w") as fh:
        yaml.dump(data, fh)


# ─────────────────────────────────────────────────────────────────
# _deep_merge tests
# ─────────────────────────────────────────────────────────────────

class TestDeepMerge:
    def test_simple_override(self) -> None:
        base = {"a": 1, "b": 2}
        override = {"b": 99}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 99}

    def test_nested_override(self) -> None:
        base = {"model": {"n_ctx": 2048, "temperature": 0.0}}
        override = {"model": {"n_ctx": 1024}}
        result = _deep_merge(base, override)
        assert result["model"]["n_ctx"] == 1024
        assert result["model"]["temperature"] == 0.0   # preserved

    def test_new_key_added(self) -> None:
        base = {"a": 1}
        override = {"b": 2}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 2}

    def test_base_not_mutated(self) -> None:
        base = {"a": {"x": 1}}
        override = {"a": {"x": 99}}
        _deep_merge(base, override)
        assert base["a"]["x"] == 1     # original unchanged

    def test_list_overrides_fully(self) -> None:
        # Lists are replaced wholesale, not merged element-by-element
        base = {"corpora": ["statpearls", "textbooks"]}
        override = {"corpora": ["statpearls"]}
        result = _deep_merge(base, override)
        assert result["corpora"] == ["statpearls"]


# ─────────────────────────────────────────────────────────────────
# load_config tests
# ─────────────────────────────────────────────────────────────────

class TestLoadConfig:
    def test_loads_real_base_yaml(self) -> None:
        cfg = load_config(base="configs/base.yaml")
        assert isinstance(cfg, ProjectConfig)
        assert cfg.seed == 42
        assert cfg.hardware.tier == "medium"
        assert cfg.model.temperature == 0.0
        assert cfg.model.logits_all is True

    def test_hardware_override(self) -> None:
        cfg = load_config(
            base="configs/base.yaml",
            hardware="configs/hardware_low.yaml",
        )
        assert cfg.hardware.tier == "low"
        assert cfg.hardware.n_threads == 2
        assert cfg.model.logits_all is False
        # Low tier falls back to the probe-only gate (no raw logits available).
        assert cfg.gate.type == "hallucination_probe"

    def test_medium_hardware(self) -> None:
        cfg = load_config(
            base="configs/base.yaml",
            hardware="configs/hardware_medium.yaml",
        )
        assert cfg.hardware.tier == "medium"
        assert cfg.model.logits_all is True

    def test_high_hardware(self) -> None:
        cfg = load_config(
            base="configs/base.yaml",
            hardware="configs/hardware_high.yaml",
        )
        assert cfg.hardware.tier == "high"
        assert cfg.hardware.n_threads == 8

    def test_policy_override(self) -> None:
        cfg = load_config(
            base="configs/base.yaml",
            policy="configs/policies/p3_closed_book.yaml",
        )
        assert cfg.policy.name == "p3_closed_book"
        assert cfg.policy.retrieval_mode == "none"

    def test_p5_gated_policy(self) -> None:
        cfg = load_config(
            base="configs/base.yaml",
            policy="configs/policies/p5_gated_entropy.yaml",
        )
        assert cfg.policy.name == "p5_gated"
        # Canonical P5 now runs the 3-gate ensemble (entropy + margin + probe).
        assert cfg.gate.type == "ensemble"
        assert cfg.gate.ensemble_min_votes == 2
        assert "hallucination_probe" in cfg.gate.ensemble_members
        assert cfg.gate.entropy_threshold == pytest.approx(2.5)

    def test_missing_base_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(base="configs/nonexistent.yaml")

    def test_custom_yaml_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "base.yaml"
            override_path = Path(tmpdir) / "override.yaml"
            _write_yaml({"seed": 42, "hardware": {"tier": "medium", "n_threads": 4, "n_batch": 128}}, base_path)
            _write_yaml({"seed": 99}, override_path)
            cfg = load_config(base=base_path, experiment=override_path)
            assert cfg.seed == 99

    def test_paths_accessible(self) -> None:
        cfg = load_config(base="configs/base.yaml")
        assert cfg.paths.data_raw == "data/raw"
        assert cfg.paths.corpora == "data/corpora"
        assert cfg.paths.models == "models"

    def test_evaluation_defaults(self) -> None:
        cfg = load_config(base="configs/base.yaml")
        assert cfg.evaluation.max_questions is None
        assert cfg.evaluation.ece_n_bins == 10
        assert cfg.evaluation.safety_thresholds.high == pytest.approx(0.90)

    def test_retrieval_defaults(self) -> None:
        cfg = load_config(base="configs/base.yaml")
        assert cfg.retrieval.top_k == 5
        assert "statpearls" in cfg.retrieval.corpora
        assert cfg.retrieval.rrf_k == 60
