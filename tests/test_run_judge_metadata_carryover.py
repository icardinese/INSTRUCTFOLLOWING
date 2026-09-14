"""Tests for evals/run_judge.py's generic {cond}_* metadata carry-over (tokens,
participation_ratio, and any future field) -- including the real collision hazard: condition
names that are prefixes of other condition names (e.g. "psr" vs "psr_proper").
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import evals.run_judge as run_judge


class _FakeAdapter:
    def __init__(self, results_dir):
        self.RESULTS_DIR = results_dir

    def load_rows(self, split):
        return [{"id": "0", "text": "irrelevant"}]


class _FakeEvalAdapter:
    SCORE_FIELDS = ["correct"]

    @staticmethod
    def score_response(row, response):
        return {"correct": 1 if "good" in response else 0}


def test_tokens_and_participation_ratio_are_carried_forward(tmp_path, monkeypatch):
    monkeypatch.setattr(run_judge, "get_adapter", lambda task: _FakeAdapter(tmp_path))
    monkeypatch.setattr(run_judge, "get_eval_adapter", lambda task: _FakeEvalAdapter())

    gen_row = {
        "id": "0", "psr_conceptor_matrix_response": "good response",
        "psr_conceptor_matrix_tokens": 5, "psr_conceptor_matrix_participation_ratio": 3.7,
    }
    (tmp_path / "generations_test.jsonl").write_text(json.dumps(gen_row) + "\n")

    run_judge.main("caveman", "test")

    out_rows = [json.loads(l) for l in (tmp_path / "judged_test.jsonl").open()]
    assert len(out_rows) == 1
    row = out_rows[0]
    assert row["psr_conceptor_matrix_correct"] == 1
    assert row["psr_conceptor_matrix_tokens"] == 5
    assert row["psr_conceptor_matrix_participation_ratio"] == 3.7


def test_prefix_overlapping_condition_names_do_not_cross_contaminate(tmp_path, monkeypatch):
    """'psr' is a prefix of 'psr_proper' -- psr's own tokens field must not accidentally get
    claimed by (or claim) psr_proper's fields, and vice versa."""
    monkeypatch.setattr(run_judge, "get_adapter", lambda task: _FakeAdapter(tmp_path))
    monkeypatch.setattr(run_judge, "get_eval_adapter", lambda task: _FakeEvalAdapter())

    gen_row = {
        "id": "0",
        "psr_response": "ok", "psr_tokens": 10,
        "psr_proper_response": "good", "psr_proper_tokens": 20,
    }
    (tmp_path / "generations_test.jsonl").write_text(json.dumps(gen_row) + "\n")

    run_judge.main("caveman", "test")

    row = json.loads((tmp_path / "judged_test.jsonl").open().readline())
    assert row["psr_tokens"] == 10
    assert row["psr_proper_tokens"] == 20
    assert row["psr_correct"] == 0
    assert row["psr_proper_correct"] == 1
    # no cross-contamination: psr's score must not have been computed from psr_proper's response
    assert "psr_response" not in row and "psr_proper_response" not in row, "raw response text should never leak into judged output"
