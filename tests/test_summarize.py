"""Tests for evals/summarize.py's summarize() -- particularly the generalized metadata-suffix
handling (participation_ratio alongside the pre-existing tokens case) and that it doesn't
misattribute fields across condition names that share a prefix.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.summarize import parse_condition_and_field, summarize

SCORE_FIELDS = ["correct", "coherent"]


def test_parse_condition_and_field_splits_correctly_with_underscored_condition_names():
    assert parse_condition_and_field("prompt_psr_proper_correct", SCORE_FIELDS) == ("prompt_psr_proper", "correct")
    assert parse_condition_and_field("psr_conceptor_matrix_coherent", SCORE_FIELDS) == ("psr_conceptor_matrix", "coherent")
    assert parse_condition_and_field("psr_conceptor_matrix_tokens", SCORE_FIELDS) is None


def test_summarize_aggregates_tokens_and_score_fields():
    rows = [
        {"id": "0", "const_correct": 2, "const_coherent": True, "const_tokens": 10},
        {"id": "1", "const_correct": 0, "const_coherent": False, "const_tokens": 20},
    ]
    summary = summarize(rows, SCORE_FIELDS)
    assert summary["const"]["correct"] == 1.0
    assert summary["const"]["avg_tokens"] == 15.0


def test_summarize_aggregates_participation_ratio_for_matrix_conditions_only():
    rows = [
        {"id": "0", "psr_conceptor_matrix_correct": 2, "psr_conceptor_matrix_tokens": 10,
         "psr_conceptor_matrix_participation_ratio": 3.5, "psr_proper_correct": 1, "psr_proper_tokens": 8},
        {"id": "1", "psr_conceptor_matrix_correct": 0, "psr_conceptor_matrix_tokens": 12,
         "psr_conceptor_matrix_participation_ratio": 3.5, "psr_proper_correct": 2, "psr_proper_tokens": 9},
    ]
    summary = summarize(rows, SCORE_FIELDS)
    assert summary["psr_conceptor_matrix"]["participation_ratio"] == 3.5
    assert "participation_ratio" not in summary["psr_proper"], "psr_proper never has this metadata -- must not appear at all"


def test_summarize_does_not_cross_contaminate_prefix_sharing_condition_names():
    """'psr' and 'psr_proper' share a prefix -- their tokens/participation_ratio must stay
    correctly attributed to the exact condition name in the key, not the shorter one."""
    rows = [
        {"id": "0", "psr_correct": 1, "psr_tokens": 5,
         "psr_proper_correct": 2, "psr_proper_tokens": 15, "psr_proper_participation_ratio": 4.0},
    ]
    summary = summarize(rows, SCORE_FIELDS)
    assert summary["psr"]["avg_tokens"] == 5
    assert "participation_ratio" not in summary["psr"]
    assert summary["psr_proper"]["avg_tokens"] == 15
    assert summary["psr_proper"]["participation_ratio"] == 4.0
