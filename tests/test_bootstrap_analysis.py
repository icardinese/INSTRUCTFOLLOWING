"""Tests for evals/bootstrap_analysis.py -- mainly that collect_values_by_cond_field now also
picks up metadata fields (avg_tokens, participation_ratio) via the shared
evals.summarize.collect_raw_values_by_cond_field, not just judge score fields.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.bootstrap_analysis import bootstrap_ci, collect_values_by_cond_field, paired_bootstrap_diff

SCORE_FIELDS = ["correct"]


def test_collect_values_by_cond_field_includes_metadata_fields():
    rows = [
        {"id": "0", "psr_conceptor_matrix_correct": 1, "psr_conceptor_matrix_tokens": 10,
         "psr_conceptor_matrix_participation_ratio": 2.0},
        {"id": "1", "psr_conceptor_matrix_correct": 0, "psr_conceptor_matrix_tokens": 20,
         "psr_conceptor_matrix_participation_ratio": 2.0},
    ]
    values = collect_values_by_cond_field(rows, SCORE_FIELDS)
    assert values["psr_conceptor_matrix"]["correct"] == [1, 0]
    assert values["psr_conceptor_matrix"]["avg_tokens"] == [10, 20]
    assert values["psr_conceptor_matrix"]["participation_ratio"] == [2.0, 2.0]


def test_bootstrap_ci_contains_the_point_estimate():
    point, lo, hi = bootstrap_ci([1, 1, 1, 0, 0], n_bootstrap=500)
    assert abs(point - 0.6) < 1e-9
    assert lo <= point <= hi


def test_paired_bootstrap_diff_excludes_zero_for_a_clear_difference():
    a = [1] * 20
    b = [0] * 20
    point, lo, hi = paired_bootstrap_diff(a, b, n_bootstrap=500)
    assert point == 1.0
    assert lo > 0, "a CI for an obviously real, maximal difference should exclude zero"
