"""
tests/unit/test_attribution.py — pure rendering helpers for the demo UI.

No model, no retriever, no Gradio: everything here is data in, HTML string out,
so the whole file runs in milliseconds with no GGUF and no index on disk.
"""

from __future__ import annotations

import math

from medrag_adaptive.data.schema import Chunk
from medrag_adaptive.ui.attribution import (
    align_draft,
    highlight_terms,
    render_chunks_html,
    render_entropy_html,
    truncated_entropy,
)


# ── alignment ──────────────────────────────────────────────────────

def test_align_equal_lengths():
    a = align_draft(["a", "b"], [0.1, 0.2])
    assert a.tokens == ["a", "b"]
    assert a.entropies == [0.1, 0.2]
    assert a.dropped_entropies == 0
    assert a.dropped_tokens == 0


def test_align_more_entropies_than_tokens():
    """The gate slices _scores by max_tokens, so trailing entropies can have no token."""
    a = align_draft(["a"], [0.1, 0.2, 0.3])
    assert a.tokens == ["a"]
    assert a.entropies == [0.1]
    assert a.dropped_entropies == 2


def test_align_more_tokens_than_entropies():
    a = align_draft(["a", "b", "c"], [0.1])
    assert a.tokens == ["a"]
    assert a.dropped_tokens == 2


def test_align_handles_none_tokens():
    a = align_draft(None, [0.1, 0.2])
    assert a.tokens == []
    assert a.entropies == []
    assert a.dropped_entropies == 2


# ── entropy heatmap ────────────────────────────────────────────────

def test_render_escapes_html_in_tokens():
    out = render_entropy_html(align_draft(["<script>"], [0.5]), threshold=0.7)
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


def test_render_reports_dropped_entropies():
    out = render_entropy_html(align_draft(["a"], [0.1, 0.2]), threshold=0.7)
    assert "2" in out
    assert "no corresponding generated token" in out


def test_render_empty_draft_is_not_a_crash():
    out = render_entropy_html(align_draft([], []), threshold=0.7)
    assert "no draft tokens" in out.lower()


def test_render_shades_higher_entropy_darker():
    out = render_entropy_html(align_draft(["lo", "hi"], [0.0, 2.0]), threshold=0.7)
    assert out.index("#ffffb2") < out.index("#bd0026")


# ── term highlighting ──────────────────────────────────────────────

def test_highlight_marks_matching_terms():
    out = highlight_terms("Epinephrine treats anaphylaxis.", "anaphylaxis drug")
    assert "<mark" in out
    assert "anaphylaxis" in out


def test_highlight_is_case_insensitive():
    out = highlight_terms("EPINEPHRINE is first-line.", "epinephrine")
    assert "<mark" in out
    assert "EPINEPHRINE" in out          # original casing preserved


def test_highlight_respects_word_boundaries():
    """'pin' must not light up inside 'epinephrine'."""
    out = highlight_terms("epinephrine", "pin")
    assert "<mark" not in out


def test_highlight_hyphenated_query_matches_both_parts():
    """_tokenize splits on non-alphanumerics, so covid-19 is two terms."""
    out = highlight_terms("Guidance on COVID-19 vaccines.", "covid-19")
    assert out.count("<mark") == 2


def test_highlight_escapes_hostile_chunk_text():
    out = highlight_terms("<script>alert(1)</script> anaphylaxis", "anaphylaxis")
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


def test_highlight_empty_query_escapes_and_marks_nothing():
    assert highlight_terms("<b>text</b>", "") == "&lt;b&gt;text&lt;/b&gt;"


def test_render_chunks_shows_score_and_source():
    chunk = Chunk(chunk_id="c1", source="statpearls", title="Anaphylaxis",
                  text="Give epinephrine.", score=18.4)
    out = render_chunks_html([chunk], "epinephrine", lexical_only=False)
    assert "statpearls" in out
    assert "18.400" in out
    assert "<mark" in out


def test_render_chunks_warns_when_retrieval_was_not_purely_lexical():
    chunk = Chunk(chunk_id="c1", source="s", title="t", text="body", score=1.0)
    out = render_chunks_html([chunk], "body", lexical_only=True)
    assert "lexical overlap only" in out


def test_render_chunks_empty_list():
    assert "No chunks retrieved" in render_chunks_html([], "q", lexical_only=False)


# ── truncated top-k entropy ────────────────────────────────────────

def test_truncated_entropy_of_a_uniform_pair_is_ln2():
    """Two tokens at p=0.5 → H = ln 2 ≈ 0.6931 nats."""
    half = math.log(0.5)
    result = truncated_entropy([{"a": half, "b": half}])
    assert abs(result.mean - math.log(2)) < 1e-9
    assert len(result.per_token) == 1


def test_truncated_entropy_of_a_certain_token_is_near_zero():
    result = truncated_entropy([{"a": 0.0}])       # log p = 0 → p = 1
    assert abs(result.mean) < 1e-6


def test_truncated_entropy_averages_across_tokens():
    half = math.log(0.5)
    result = truncated_entropy([{"a": half, "b": half}, {"a": 0.0}])
    assert abs(result.mean - math.log(2) / 2) < 1e-9


def test_truncated_entropy_on_empty_input_is_none():
    assert truncated_entropy([]) is None
    assert truncated_entropy(None) is None
