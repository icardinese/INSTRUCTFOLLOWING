"""Tests for evals/caveman/judge.py's conciseness addition. Uses a fake client shaped like the
OpenAI SDK's response object -- no real API key or network call, consistent with this file having
zero live-API test coverage by design (see ARCHITECTURE.md / the project handoff on why).

The fake client is driven by a LIST of replies consumed in call order, not a dict keyed by model
name -- JUDGE_MODEL and CONCISENESS_JUDGE_MODEL are currently the SAME string ("gpt-4o-mini", per
project decision to keep API cost down), so a model-keyed fake would silently collapse two
different expected replies into one and mask real bugs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from evals.caveman.judge import CONCISENESS_JUDGE_MODEL, JUDGE_MODEL, SCORE_FIELDS, _call_judge_json, score_response


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    """replies: a list consumed one-per-call, in call order. Each entry is either a raw string
    (the model's message content) or an Exception instance (raised instead of returned)."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def create(self, model, max_tokens, temperature, messages):
        self.calls.append({"model": model, "messages": messages})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return _FakeResponse(reply)


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


class _FakeClient:
    def __init__(self, replies):
        self.chat = _FakeChat(_FakeCompletions(replies))


ROW = {"code": "def add(a, b):\n    return a + b", "reference_explanation": "Adds two numbers."}


def test_score_fields_includes_conciseness_alongside_correct_and_coherent():
    assert SCORE_FIELDS == ["correct", "coherent", "conciseness"]


def test_call_judge_json_parses_response_and_ignores_surrounding_text():
    client = _FakeClient(['sure, here you go: {"correct": 2, "coherent": true} thanks'])
    result = _call_judge_json(client, JUDGE_MODEL, "irrelevant prompt")
    assert result == {"correct": 2, "coherent": True}


def test_call_judge_json_retries_then_raises_on_persistent_failure():
    client = _FakeClient([RuntimeError("boom"), RuntimeError("boom again")])
    with pytest.raises(RuntimeError):
        _call_judge_json(client, JUDGE_MODEL, "irrelevant prompt", n_attempts=2)
    assert len(client.chat.completions.calls) == 2


def test_call_judge_json_succeeds_after_one_retry():
    client = _FakeClient([RuntimeError("transient"), '{"correct": 1, "coherent": false}'])
    result = _call_judge_json(client, JUDGE_MODEL, "irrelevant prompt", n_attempts=3)
    assert result == {"correct": 1, "coherent": False}


def test_score_response_makes_two_independent_calls_for_correctness_and_conciseness(monkeypatch):
    """Two calls happen (correctness/coherence, then conciseness), in that order, regardless of
    whether JUDGE_MODEL and CONCISENESS_JUDGE_MODEL happen to be the same model string."""
    client = _FakeClient(['{"correct": 2, "coherent": true}', '{"conciseness": 1}'])
    import evals.caveman.judge as judge_module
    monkeypatch.setattr(judge_module, "_get_client", lambda: client)

    result = score_response(ROW, "adds a and b together")
    assert result == {"correct": 2, "coherent": True, "conciseness": 1}

    models_called = [c["model"] for c in client.chat.completions.calls]
    assert models_called == [JUDGE_MODEL, CONCISENESS_JUDGE_MODEL]
    assert len(client.chat.completions.calls) == 2, "must be two separate API calls, not one combined rubric"


def test_conciseness_prompt_does_not_require_reference_explanation():
    """The conciseness rubric only formats {code} and {candidate} -- it must not reference
    {reference_explanation}, since grading conciseness against a reference wording would
    conflate 'matches the reference' with 'isn't padded' (see module docstring)."""
    import evals.caveman.judge as judge_module
    formatted = judge_module.CONCISENESS_RUBRIC.format(code=ROW["code"], candidate="adds a and b")
    assert "{reference_explanation}" not in formatted
