"""
policies/p2_always_retrieve_cite.py — P2: always-retrieve with citations.

P2 is P1 plus provenance: the RAG-with-citation prompt asks the model to list
the sources it used in a trailing "SOURCES: title, title" line, and this policy
parses that line back into Citation objects by matching the cited titles to the
retrieved chunks. The matched citations populate PolicyResult.citations, from
which citation precision/recall (cited ∩ retrieved) are later computed — the
provenance baseline against which gated policies are judged on attribution.

Title matching is deliberately lenient (case-insensitive substring) because
small models paraphrase titles; an exact match would under-count valid
citations. Citations that match no retrieved chunk are dropped (a cited source
the model invented is not counted as a true citation).
"""

from __future__ import annotations

import re
from typing import List

from medrag_adaptive.data.schema import Chunk, Citation, PolicyResult, UnifiedQuestion
from medrag_adaptive.models.prompts import build_rag_prompt
from medrag_adaptive.policies.base import Policy

_SOURCES_RE = re.compile(r"SOURCES?\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)


def _parse_citations(answer: str, chunks: List[Chunk]) -> List[Citation]:
    """Match titles named in the answer's SOURCES line to retrieved chunks."""
    m = _SOURCES_RE.search(answer)
    if not m:
        return []
    cited_blob = m.group(1).lower()
    citations: List[Citation] = []
    seen = set()
    for c in chunks:
        title = c.title.lower().strip("[] ")
        if title and title in cited_blob and c.chunk_id not in seen:
            citations.append(Citation(chunk_id=c.chunk_id, source=c.source, title=c.title))
            seen.add(c.chunk_id)
    return citations


class AlwaysRetrieveCitePolicy(Policy):
    """P2: always retrieve, answer with context, extract source citations."""

    name = "p2_always_retrieve_cite"

    def answer(self, question: UnifiedQuestion) -> PolicyResult:
        if self.retriever is None:
            raise ValueError(f"{self.name} requires a retriever")

        chunks = self.retriever.retrieve(question.question_text)
        prompt = build_rag_prompt(
            question.question_text, chunks,
            choices=question.choices, cite_sources=True,
        )
        text = self.llm.answer(prompt)
        citations = _parse_citations(text, chunks)
        return PolicyResult(
            question_id=question.question_id,
            policy_name=self.name,
            retrieval_triggered=True,
            retrieved_chunks=chunks,
            answer_text=text,
            citations=citations,
        )
