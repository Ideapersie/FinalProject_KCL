"""
config.py — Project configuration loader.

Usage:
    cfg = load_config(
        base="configs/base.yaml",
        hardware="configs/hardware_medium.yaml",
        policy="configs/policies/p5_gated_entropy.yaml",
        experiment="configs/experiments/full_mirage.yaml",
    )
    print(cfg.model.gguf_path)

The four YAML files are deep-merged in order (base → hardware → policy → experiment),
so later files override earlier ones. The merged dict is validated into a
ProjectConfig Pydantic model, which catches missing or mis-typed fields early.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────────────────────────
# Nested config models
# ─────────────────────────────────────────────────────────────────

class PathsConfig(BaseModel):
    data_raw:       str = "data/raw"
    data_processed: str = "data/processed"
    corpora:        str = "data/corpora"
    indexes:        str = "indexes"
    models:         str = "models"
    results:        str = "results/raw_logs"
    aggregated:     str = "results/aggregated"
    figures:        str = "results/figures"


class HardwareConfig(BaseModel):
    tier:      Literal["low", "medium", "high"] = "medium"
    n_threads: int = 4
    n_batch:   int = 128


class ModelConfig(BaseModel):
    name:           str   = "llama-3.2-3b"
    gguf_path:      str   = "models/llama-3.2-3b-q4_k_m.gguf"
    n_ctx:          int   = 2048
    # logits_all=True is required for entropy + margin gates.
    # Set False on low-tier hardware to save ~256 MB RAM
    # (disables entropy/margin gates; verbalized gate still works).
    logits_all:     bool  = True
    temperature:    float = 0.0
    max_new_tokens: int   = 256
    verbose:        bool  = False

    @field_validator("temperature")
    @classmethod
    def temperature_is_zero_for_reproducibility(cls, v: float) -> float:
        if v != 0.0:
            import warnings
            warnings.warn(
                f"temperature={v} — non-zero temperature breaks reproducibility. "
                "Set temperature=0.0 for evaluation runs.",
                stacklevel=2,
            )
        return v


class VerbalisedLabels(BaseModel):
    retrieve: List[str] = ["MEDIUM", "LOW"]
    skip:     List[str] = ["HIGH"]


class HallucinationProbeConfig(BaseModel):
    """Settings for the draft-consistency (hallucination probe) gate."""
    enabled:        bool = True
    # letter_match for MCQ (compare extracted option letters);
    # f1_threshold for open-ended (retrieve if token-F1 between drafts < threshold).
    agreement_mode: Literal["letter_match", "f1_threshold"] = "letter_match"
    f1_threshold:   float = 0.7
    max_tokens:     int   = 48


class GateConfig(BaseModel):
    # entropy | margin | hallucination_probe : single-signal gates.
    # ensemble : majority vote over `ensemble_members`.
    # verbalized : deprecated (replaced by hallucination_probe); kept for ablation.
    type:                  Literal[
        "entropy", "margin", "hallucination_probe", "ensemble", "verbalized"
    ] = "entropy"
    entropy_threshold:     float = 2.5
    margin_threshold:      float = 0.3
    draft_max_tokens:      int   = 48
    # Ensemble: which single gates vote, and how many votes trigger retrieval.
    ensemble_members:      List[str] = Field(
        default_factory=lambda: ["entropy", "margin", "hallucination_probe"]
    )
    ensemble_min_votes:    int   = 2
    hallucination_probe:   HallucinationProbeConfig = Field(
        default_factory=HallucinationProbeConfig
    )
    # Deprecated verbalized-confidence settings (retained for ablation only).
    verbalized_labels:     VerbalisedLabels = Field(default_factory=VerbalisedLabels)
    verbalized_max_tokens: int   = 10


class RetrievalConfig(BaseModel):
    bm25_index:         str       = "indexes/bm25_combined.pkl"
    faiss_index:        str       = "indexes/faiss_combined.index"
    embedding_model:    str       = "sentence-transformers/all-MiniLM-L6-v2"
    top_k:              int       = 5
    corpora:            List[str] = Field(default_factory=lambda: ["statpearls", "textbooks"])
    rrf_k:              int       = 60
    bm25_route_patterns: List[str] = Field(default_factory=list)
    bm25_weight:        float     = 0.5
    vector_weight:      float     = 0.5


class SafetyThresholds(BaseModel):
    low:    float = 0.70
    medium: float = 0.80
    high:   float = 0.90


class EvaluationConfig(BaseModel):
    batch_size:     int           = 1
    max_questions:  Optional[int] = None
    datasets:       List[str]     = Field(default_factory=lambda: ["mirage", "ragcare", "acute_care"])
    compute_ece:    bool          = True
    ece_n_bins:     int           = 10
    safety_thresholds: SafetyThresholds = Field(default_factory=SafetyThresholds)


class ProfilingConfig(BaseModel):
    track_energy:               bool = True
    track_memory:               bool = True
    track_latency:              bool = True
    codecarbon_country_iso_code: str = "GBR"


class PolicyConfig(BaseModel):
    name:           str           = "p3_closed_book"
    description:    str           = ""
    retrieval_mode: str           = "none"
    cite_sources:   bool          = False
    bm25_weight:    float         = 0.5
    vector_weight:  float         = 0.5


# ─────────────────────────────────────────────────────────────────
# Top-level config
# ─────────────────────────────────────────────────────────────────

class ProjectConfig(BaseModel):
    seed:       int              = 42
    paths:      PathsConfig      = Field(default_factory=PathsConfig)
    hardware:   HardwareConfig   = Field(default_factory=HardwareConfig)
    model:      ModelConfig      = Field(default_factory=ModelConfig)
    gate:       GateConfig       = Field(default_factory=GateConfig)
    retrieval:  RetrievalConfig  = Field(default_factory=RetrievalConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    profiling:  ProfilingConfig  = Field(default_factory=ProfilingConfig)
    policy:     PolicyConfig     = Field(default_factory=PolicyConfig)

    # Experiment-level overrides (set by experiment YAML)
    experiment_name:  str           = "unnamed"
    dataset:          str           = "mirage"
    output:           Optional[str] = None


# ─────────────────────────────────────────────────────────────────
# Merge helpers
# ─────────────────────────────────────────────────────────────────

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge `override` into `base`, returning a new dict."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p.resolve()}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data or {}


# ─────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────

def load_config(
    base:       str | Path = "configs/base.yaml",
    hardware:   str | Path | None = None,
    policy:     str | Path | None = None,
    experiment: str | Path | None = None,
) -> ProjectConfig:
    """
    Load and merge YAML config files in order:
        base → hardware → policy → experiment

    Later files override earlier ones via deep-merge.
    The result is validated into a ProjectConfig Pydantic model.

    Args:
        base:       Path to base.yaml (required).
        hardware:   Path to hardware_*.yaml (optional).
        policy:     Path to policies/*.yaml (optional).
        experiment: Path to experiments/*.yaml (optional).

    Returns:
        Validated ProjectConfig instance.
    """
    merged: Dict[str, Any] = _load_yaml(base)

    for extra in (hardware, policy, experiment):
        if extra is not None:
            merged = _deep_merge(merged, _load_yaml(extra))

    merged = _normalise_experiment_keys(merged)
    return ProjectConfig.model_validate(merged)


def _normalise_experiment_keys(merged: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reconcile the convenience shape used by experiment YAMLs with the nested
    ProjectConfig schema. Experiment files use a few flat top-level keys for
    brevity; map them onto their nested homes before validation:

        name              -> experiment_name
        max_questions     -> evaluation.max_questions
        policy: <string>  -> policy.name
        hardware_config   -> merge in the referenced hardware YAML
    """
    if "name" in merged:
        merged["experiment_name"] = merged.pop("name")

    # A referenced hardware file is merged as a (low-precedence) base so that
    # explicit keys already in `merged` still win.
    hw_ref = merged.pop("hardware_config", None)
    if hw_ref:
        merged = _deep_merge(_load_yaml(hw_ref), merged)

    if "max_questions" in merged:
        merged.setdefault("evaluation", {})
        merged["evaluation"]["max_questions"] = merged.pop("max_questions")

    # `policy:` may be a string shorthand for the policy name, or a full object.
    if isinstance(merged.get("policy"), str):
        merged["policy"] = {"name": merged["policy"]}

    return merged
