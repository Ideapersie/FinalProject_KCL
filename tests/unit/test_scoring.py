"""Unit tests for evaluation/scoring.py."""

import pytest

from medrag_adaptive.evaluation.scoring import (
    extract_letter,
    score_mcq,
    token_f1,
)


@pytest.mark.parametrize("text,expected", [
    ("A. Phrenic nerve", "A"),
    ("B) Metformin", "B"),
    ("(C)", "C"),
    ("The answer is D.", "D"),
    ("Answer: E", "E"),
    ("Phrenic nerve", None),
    ("", None),
])
def test_extract_letter(text, expected):
    assert extract_letter(text) == expected


def test_score_mcq_correct():
    assert score_mcq("A. Phrenic nerve", "A") == (True, 1.0)


def test_score_mcq_wrong():
    assert score_mcq("B. Vagus nerve", "A") == (False, 0.0)


def test_score_mcq_case_insensitive_gold():
    assert score_mcq("c) foo", "C") == (True, 1.0)


def test_score_mcq_unparseable():
    assert score_mcq("I am not sure", "A") == (False, 0.0)


def test_token_f1_identical():
    assert token_f1("aspirin inhibits cox", "aspirin inhibits cox") == 1.0


def test_token_f1_disjoint():
    assert token_f1("foo bar", "baz qux") == 0.0


def test_token_f1_partial_between_0_and_1():
    score = token_f1("aspirin inhibits cox enzyme", "aspirin inhibits cox")
    assert 0.0 < score < 1.0


def test_token_f1_empty():
    assert token_f1("", "something") == 0.0


def test_token_f1_ignores_punctuation_and_case():
    assert token_f1("Aspirin, inhibits COX!", "aspirin inhibits cox") == 1.0
