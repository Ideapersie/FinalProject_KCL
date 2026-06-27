"""
policies/p4_hybrid.py — P4: hybrid-retrieval policy.

P4 always retrieves, like P1, but is bound to the hybrid (BM25 + vector, RRF-
fused) retriever rather than a single-source one. Mechanically the answering
step is identical to P1 — retrieve, build a RAG prompt, answer — so the policy
subclasses the always-retrieve behaviour and differs only in the retriever it
is constructed with (the factory supplies a HybridRetriever for this policy).
Keeping it a distinct class makes the policy name explicit in logs and gives a
home for any future BM25/vector routing logic.
"""

from __future__ import annotations

from medrag_adaptive.policies.p1_always_retrieve import AlwaysRetrievePolicy


class HybridRetrievalPolicy(AlwaysRetrievePolicy):
    """P4: always retrieve using the hybrid (RRF) retriever."""

    name = "p4_hybrid"
