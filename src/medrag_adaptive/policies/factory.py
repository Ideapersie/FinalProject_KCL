"""
policies/factory.py — Construct a Policy (and its gate) from a ProjectConfig.

Centralises the wiring that the run driver used to do by hand, so every entry
point builds policies identically. `build_policy` reads cfg.policy.name to pick
the policy class, and (for P5) cfg.gate to assemble the gate.

Gate assembly:
  - cfg.gate.type in {entropy, margin, hallucination_probe} → that single gate.
  - cfg.gate.type == "ensemble" → an EnsembleGate over cfg.gate.ensemble_members,
    each constructed with thresholds from cfg.gate, voting with
    cfg.gate.ensemble_min_votes.
  - cfg.gate.type == "verbalized" is deprecated and not built here.

The low tier sets cfg.gate.type = hallucination_probe (probe-only), so the same
P5 policy file yields an ensemble on medium/high and a probe-only gate on low.
"""

from __future__ import annotations

from typing import Optional

from medrag_adaptive.config import ProjectConfig
from medrag_adaptive.gating.base import Gate
from medrag_adaptive.gating.entropy_gate import EntropyGate
from medrag_adaptive.gating.margin_gate import MarginGate
from medrag_adaptive.gating.hallucination_probe_gate import HallucinationProbeGate
from medrag_adaptive.gating.ensemble_gate import EnsembleGate
from medrag_adaptive.models.base import LLMBackend
from medrag_adaptive.policies.base import Policy
from medrag_adaptive.policies.p1_always_retrieve import AlwaysRetrievePolicy
from medrag_adaptive.policies.p2_always_retrieve_cite import AlwaysRetrieveCitePolicy
from medrag_adaptive.policies.p3_closed_book import ClosedBookPolicy
from medrag_adaptive.policies.p4_hybrid import HybridRetrievalPolicy
from medrag_adaptive.policies.p5_gated import GatedPolicy
from medrag_adaptive.retrieval.base import Retriever


def _build_single_gate(gate_type: str, cfg: ProjectConfig) -> Gate:
    g = cfg.gate
    if gate_type == "entropy":
        return EntropyGate(threshold=g.entropy_threshold,
                           draft_max_tokens=g.draft_max_tokens)
    if gate_type == "margin":
        return MarginGate(threshold=g.margin_threshold,
                          draft_max_tokens=g.draft_max_tokens)
    if gate_type == "hallucination_probe":
        return HallucinationProbeGate(
            agreement_mode=g.hallucination_probe.agreement_mode,
            f1_threshold=g.hallucination_probe.f1_threshold,
            max_tokens=g.hallucination_probe.max_tokens,
        )
    raise ValueError(f"Unknown gate type '{gate_type}'")


def build_gate(cfg: ProjectConfig) -> Gate:
    """
    Construct the gate described by cfg.gate.

    Hardware constraint: when the model runs with logits_all=False (the low
    tier), the entropy and margin gates have no logit buffer to read. Rather
    than let a policy YAML request an impossible gate, we downgrade to a
    probe-only gate here. This makes the low-tier fallback a property of the
    hardware config, independent of which policy file is loaded.
    """
    logits_available = cfg.model.logits_all

    if not logits_available:
        # Only the (text-only) hallucination probe can run.
        return _build_single_gate("hallucination_probe", cfg)

    if cfg.gate.type == "ensemble":
        members = [_build_single_gate(name, cfg) for name in cfg.gate.ensemble_members]
        return EnsembleGate(members=members, min_votes=cfg.gate.ensemble_min_votes)
    if cfg.gate.type == "verbalized":
        raise ValueError("verbalized gate is deprecated; use hallucination_probe")
    return _build_single_gate(cfg.gate.type, cfg)


def build_policy(
    cfg: ProjectConfig,
    llm: LLMBackend,
    retriever: Optional[Retriever] = None,
) -> Policy:
    """Build the policy named by cfg.policy.name, wiring gate and retriever."""
    name = cfg.policy.name

    if name == "p3_closed_book":
        return ClosedBookPolicy(llm=llm)

    if name == "p1_always_retrieve":
        if retriever is None:
            raise ValueError(f"{name} requires a retriever")
        return AlwaysRetrievePolicy(
            llm=llm, retriever=retriever, cite_sources=cfg.policy.cite_sources
        )

    if name == "p2_always_retrieve_cite":
        if retriever is None:
            raise ValueError(f"{name} requires a retriever")
        return AlwaysRetrieveCitePolicy(
            llm=llm, retriever=retriever, cite_sources=True
        )

    if name == "p4_hybrid":
        if retriever is None:
            raise ValueError(f"{name} requires a retriever")
        return HybridRetrievalPolicy(
            llm=llm, retriever=retriever, cite_sources=cfg.policy.cite_sources
        )

    if name in ("p5_gated", "p5_gated_entropy", "p5_gated_margin",
                "p5_gated_verbalized", "p5_gated_ensemble"):
        if retriever is None:
            raise ValueError(f"{name} requires a retriever for its RETRIEVE path")
        return GatedPolicy(
            llm=llm, retriever=retriever, gate=build_gate(cfg),
            cite_sources=cfg.policy.cite_sources,
        )

    raise ValueError(f"Policy '{name}' is not known to the factory")
