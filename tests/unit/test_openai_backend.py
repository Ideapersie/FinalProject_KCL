"""
tests/unit/test_openai_backend.py — response parsing, with no network.

The HTTP call is monkeypatched out; what is under test is the translation of an
OpenAI-shaped chat completion into the LLMBackend contract the gates consume.
"""

from __future__ import annotations

import math

from medrag_adaptive.models.openai_backend import OpenAIBackend

_RESPONSE = {
    "choices": [{
        "message": {"content": "Epinephrine"},
        "logprobs": {"content": [
            {"token": "Epine", "logprob": -0.01,
             "top_logprobs": [{"token": "Epine", "logprob": -0.01},
                              {"token": "Diphen", "logprob": -4.6}]},
            {"token": "phrine", "logprob": -0.002,
             "top_logprobs": [{"token": "phrine", "logprob": -0.002},
                              {"token": "phrin", "logprob": -6.2}]},
        ]},
    }]
}


def _backend(monkeypatch) -> OpenAIBackend:
    backend = OpenAIBackend(base_url="http://x/v1", model="m", api_key="k")
    monkeypatch.setattr(backend, "_post", lambda payload: _RESPONSE)
    return backend


def test_answer_returns_message_content(monkeypatch):
    assert _backend(monkeypatch).answer("prompt") == "Epinephrine"


def test_draft_reports_no_logits(monkeypatch):
    text, logits = _backend(monkeypatch).draft("prompt")
    assert text == "Epinephrine"
    assert logits is None            # no full-vocabulary logits exist over an API


def test_draft_with_tokens_returns_token_strings(monkeypatch):
    _text, _logits, tokens = _backend(monkeypatch).draft_with_tokens("prompt")
    assert tokens == ["Epine", "phrine"]


def test_get_top2_logprobs_shape_matches_margin_gate(monkeypatch):
    _text, top = _backend(monkeypatch).get_top2_logprobs("prompt")
    assert len(top) == 2
    assert abs(top[0]["Epine"] - (-0.01)) < 1e-9
    assert abs(top[0]["Diphen"] - (-4.6)) < 1e-9


def test_margin_gate_can_consume_this_backend(monkeypatch):
    from medrag_adaptive.data.schema import UnifiedQuestion
    from medrag_adaptive.gating.margin_gate import MarginGate

    question = UnifiedQuestion(question_id="q", question_text="t",
                               correct_answer="", dataset_source="ui")
    decision = MarginGate(threshold=0.3).decide(question, _backend(monkeypatch))
    assert decision.details["available"] is True
    assert decision.signal_value > 0.9        # near-certain draft → wide margin


def test_entropy_gate_abstains_on_this_backend(monkeypatch):
    """No logits → the gate must abstain, so the ensemble can skip it."""
    from medrag_adaptive.data.schema import UnifiedQuestion
    from medrag_adaptive.gating.entropy_gate import EntropyGate

    question = UnifiedQuestion(question_id="q", question_text="t",
                               correct_answer="", dataset_source="ui")
    decision = EntropyGate(threshold=0.7).decide(question, _backend(monkeypatch))
    assert decision.details["available"] is False


def test_last_draft_topk_feeds_truncated_entropy(monkeypatch):
    backend = _backend(monkeypatch)
    backend.draft_with_tokens("prompt")
    assert len(backend.last_draft_topk) == 2
    assert math.isfinite(sum(backend.last_draft_topk[0].values()))


def test_payload_requests_logprobs_only_when_needed(monkeypatch):
    """answer() must not pay for logprobs it never reads."""
    seen = []
    backend = OpenAIBackend(base_url="http://x/v1", model="m")
    monkeypatch.setattr(backend, "_post",
                        lambda payload: (seen.append(payload), _RESPONSE)[1])
    backend.answer("prompt")
    backend.draft_with_tokens("prompt")
    assert "logprobs" not in seen[0]
    assert seen[1]["logprobs"] is True
    assert seen[1]["top_logprobs"] == 20
