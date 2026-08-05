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
    render_agreement_html,
    render_chunks_html,
    render_entropy_html,
    render_gate_table_html,
    truncated_entropy,
)

# Shape copied from a real logged ensemble payload (results/raw_logs/
# p5_medcorp_mcq.jsonl, qvault.gate_details) so the renderer is tested against
# the keys the gates actually emit rather than keys invented for the test.
ENSEMBLE_DETAILS = {
    "votes": {"entropy": "retrieve", "margin": "skip", "hallucination_probe": "skip"},
    "retrieve_votes": 1,
    "min_votes": 2,
    "members": {
        "entropy": {"available": True, "mean_entropy": 0.734, "threshold": 0.7,
                    "per_token_entropy": [0.5, 0.9]},
        "margin": {"available": True, "mean_margin": 0.740, "threshold": 0.7},
        "hallucination_probe": {"available": True, "agreement": True,
                                "mode": "letter_match"},
    },
}


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


# ── gate signal table ──────────────────────────────────────────────

def test_gate_table_lists_every_member_with_its_signal_and_vote():
    out = render_gate_table_html(ENSEMBLE_DETAILS, "ensemble")
    for name in ("entropy", "margin", "hallucination_probe"):
        assert name in out
    assert "0.734" in out and "0.740" in out
    assert "retrieve" in out and "skip" in out


def test_gate_table_shows_each_rule_direction():
    """The two thresholds fire in opposite directions; the table must say which."""
    out = render_gate_table_html(ENSEMBLE_DETAILS, "ensemble")
    assert "retrieve if &gt; 0.700" in out
    assert "retrieve if &lt; 0.700" in out


def test_gate_table_probe_has_no_threshold():
    out = render_gate_table_html(ENSEMBLE_DETAILS, "ensemble")
    assert "retrieve if they disagree" in out


def test_gate_table_marks_an_abstaining_member():
    """An API backend cannot supply full-vocabulary logits; entropy abstains."""
    details = {
        "votes": {"margin": "skip"},
        "members": {
            "entropy": {"available": False},
            "margin": {"available": True, "mean_margin": 0.8, "threshold": 0.7},
        },
    }
    out = render_gate_table_html(details, "ensemble")
    assert "abstained" in out


def test_gate_table_single_gate_uses_the_payload_itself():
    """A non-ensemble run has no `members` key — the details ARE the member."""
    out = render_gate_table_html(
        {"available": True, "mean_entropy": 0.42, "threshold": 0.7}, "entropy"
    )
    assert "0.420" in out
    assert "margin" not in out


# ── answer agreement strip ─────────────────────────────────────────

def test_agreement_is_green_only_when_both_policies_are_right():
    out = render_agreement_html("C", "C", "C", True, True)
    assert "mr-agree-good" in out
    assert "both correct" in out


def test_agreement_is_not_green_when_both_agree_but_are_wrong():
    """Two identical wrong answers is the failure mode, not a success."""
    out = render_agreement_html("A", "A", "C", False, False)
    assert "mr-agree-good" not in out
    assert "mr-agree-bad" in out
    assert "both wrong" in out


def test_agreement_names_which_policy_was_right_when_they_split():
    gated = render_agreement_html("C", "A", "C", True, False)
    assert "mr-agree-split" in gated
    assert "only the gated policy is correct" in gated

    closed = render_agreement_html("A", "C", "C", False, True)
    assert "only closed-book is correct" in closed


def test_agreement_without_a_gold_letter_renders_nothing():
    assert render_agreement_html("C", "C", None, None, None) == ""


def test_agreement_shows_a_placeholder_when_no_letter_was_extracted():
    out = render_agreement_html(None, "C", "C", False, True)
    assert "P5 ?" in out


def test_gate_table_of_empty_details_renders_nothing():
    assert render_gate_table_html({}, "entropy") == ""
    assert render_gate_table_html({"available": True}, "hybrid_unknown_gate") == ""


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
