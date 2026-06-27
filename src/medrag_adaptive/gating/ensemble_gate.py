"""
gating/ensemble_gate.py — Majority-vote gate ensemble (P5 core).

Combines the entropy, margin and hallucination-probe gates into one decision by
majority vote. Each member returns a GateDecision; only members reporting
`available=True` in their details cast a vote (the logit-based gates abstain on
the low tier). Retrieve when the number of RETRIEVE votes meets `min_votes`.

    retrieve  ⟺  #{members voting RETRIEVE} ≥ min_votes

Degraded mode: when fewer members are available than `min_votes` can be reached
with (e.g. the low tier where only the probe runs), the strict threshold is
unreachable, so the ensemble falls back to "retrieve if any available member
votes RETRIEVE" and flags `degraded=True`. This makes the low-tier behaviour a
documented finding rather than a silent failure.

The per-member votes and signals are recorded in details so the full decision
is auditable in the JSONL log (RunRecord.qvault).
"""

from __future__ import annotations

from typing import List

from medrag_adaptive.data.schema import UnifiedQuestion
from medrag_adaptive.gating.base import Gate, GateDecision
from medrag_adaptive.models.base import LLMBackend


class EnsembleGate(Gate):
    """Majority-vote over a list of member gates."""

    name = "ensemble"

    def __init__(self, members: List[Gate], min_votes: int = 2) -> None:
        if not members:
            raise ValueError("EnsembleGate requires at least one member gate")
        self.members = members
        self.min_votes = min_votes

    def decide(self, question: UnifiedQuestion, llm: LLMBackend) -> GateDecision:
        votes = {}
        signals = {}
        member_details = {}
        retrieve_votes = 0
        available = 0

        for member in self.members:
            decision = member.decide(question, llm)
            member_details[member.name] = decision.details
            if not decision.details.get("available", True):
                votes[member.name] = "abstain"
                continue
            available += 1
            signals[member.name] = decision.signal_value
            votes[member.name] = decision.decision_str
            if decision.retrieve:
                retrieve_votes += 1

        degraded = available < self.min_votes
        if degraded:
            # Strict majority unreachable; retrieve if any available member did.
            retrieve = retrieve_votes >= 1
        else:
            retrieve = retrieve_votes >= self.min_votes

        return GateDecision(
            name=self.name,
            retrieve=retrieve,
            signal_value=float(retrieve_votes),
            details={
                "available": True,
                "votes": votes,
                "signals": signals,
                "retrieve_votes": retrieve_votes,
                "members_available": available,
                "min_votes": self.min_votes,
                "degraded": degraded,
                "members": member_details,
            },
        )
