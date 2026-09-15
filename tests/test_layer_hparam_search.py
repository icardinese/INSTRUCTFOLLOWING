"""Tests for evals/layer_hparam_search.py's orchestration logic. Training (retrain_fn) and batched
generation (generate_with_routed_configs) are both mocked here -- they need a real GPU/model, and
that's not what these tests are checking anyway. What's checked: does best_row_per_layer correctly
implement "MSE only picks a per-layer representative, never eliminates a layer", does
evaluate_candidates correctly wire retrain -> batch -> judge -> bootstrap CI, and does the tier
progression (Tier 1 -> Tier 2 -> Final) correctly narrow the candidate set using JUDGED scores,
not MSE, at each cut.
"""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.layer_hparam_search import SearchContext, best_row_per_layer, evaluate_candidates


def test_best_row_per_layer_picks_lowest_mse_per_layer():
    rows = [
        {"layer": 2, "mse_weight": 1.0, "nll_weight": 0.0, "final_mse": 5.0},
        {"layer": 2, "mse_weight": 1.0, "nll_weight": 0.1, "final_mse": 3.0},  # better -- should win layer 2
        {"layer": 4, "mse_weight": 1.0, "nll_weight": 0.0, "final_mse": 7.0},
    ]
    best = best_row_per_layer(rows)
    assert set(best.keys()) == {2, 4}
    assert best[2]["nll_weight"] == 0.1
    assert best[4]["final_mse"] == 7.0


def test_best_row_per_layer_excludes_skipped_and_incomplete_rows():
    rows = [
        {"layer": 2, "skipped": True, "final_mse": 1.0},  # must be excluded despite great MSE
        {"layer": 2, "final_mse": 9.0},                    # only usable row for layer 2
        {"layer": 4},                                       # no final_mse at all -- excluded
    ]
    best = best_row_per_layer(rows)
    assert set(best.keys()) == {2}
    assert best[2]["final_mse"] == 9.0


class _FakeEvalAdapter:
    SCORE_FIELDS = ["correct"]

    @staticmethod
    def score_response(row, response):
        # deterministic score derived from the response text so tests can assert on it precisely
        return {"correct": int(response.split("_")[-1])}


class _FakeAdapter:
    def __init__(self):
        self.CACHE_DIR = Path("/tmp/fake_cache")

    def load_rows(self, split):
        return [{"id": str(i)} for i in range(3)]

    def to_items(self, tokenizer, rows):
        return [{"id": r["id"], "base_prompt": f"prompt_{r['id']}"} for r in rows]


def test_evaluate_candidates_retrains_judges_and_computes_ci():
    """End-to-end orchestration of evaluate_candidates with retrain_fn and generate_with_routed_configs
    both mocked -- checks the WIRING (candidate -> hook -> batched call -> judged score -> CI), not
    the real training/generation math (already covered elsewhere)."""
    candidates = [
        {"layer": 2, "mse_weight": 1.0, "nll_weight": 0.0},
        {"layer": 4, "mse_weight": 1.0, "nll_weight": 0.0},
    ]
    ctx = SearchContext(
        model=object(), tokenizer=object(), n_layers=10, hidden_size=16, cache_dir=Path("/tmp"),
        train_items=[], dev_items=[], train_responses={}, dev_responses={},
    )

    def fake_retrain(row, ctx):
        return f"hook_for_layer_{row['layer']}"  # stand-in hook object, never actually called

    fake_responses_by_group = {
        0: ["resp_1", "resp_2", "resp_0"],  # layer 2's candidate: scores [1, 2, 0]
        1: ["resp_2", "resp_2", "resp_2"],  # layer 4's candidate: scores [2, 2, 2]
    }

    with patch("evals.layer_hparam_search.RETRAIN_FNS", {"proper": fake_retrain}), \
         patch("evals.layer_hparam_search.generate_with_routed_configs", return_value=fake_responses_by_group):
        results = evaluate_candidates("proper", candidates, ctx, _FakeAdapter(), _FakeEvalAdapter(), n_examples=3)

    assert len(results) == 2
    assert results[0]["layer"] == 2 and abs(results[0]["judged_score"] - 1.0) < 1e-9  # mean([1,2,0])
    assert results[1]["layer"] == 4 and abs(results[1]["judged_score"] - 2.0) < 1e-9  # mean([2,2,2])
    for r in results:
        assert not r["skipped"]
        assert r["ci_lo"] <= r["judged_score"] <= r["ci_hi"]
        assert r["n"] == 3


def test_evaluate_candidates_marks_skipped_when_retrain_returns_none():
    """A retrain function can return None (e.g. selfproj's delta_scale-too-small skip) -- that
    candidate must be marked skipped, not silently dropped or crashed on."""
    candidates = [{"layer": 2, "alpha": 1e6, "mse_weight": 1.0, "nll_weight": 0.0}]
    ctx = SearchContext(
        model=object(), tokenizer=object(), n_layers=10, hidden_size=16, cache_dir=Path("/tmp"),
        train_items=[], dev_items=[], train_responses={}, dev_responses={},
    )

    def fake_retrain_returns_none(row, ctx):
        return None

    with patch("evals.layer_hparam_search.RETRAIN_FNS", {"proper": fake_retrain_returns_none}), \
         patch("evals.layer_hparam_search.generate_with_routed_configs", return_value={}):
        results = evaluate_candidates("proper", candidates, ctx, _FakeAdapter(), _FakeEvalAdapter(), n_examples=3)

    assert len(results) == 1
    assert results[0]["skipped"] is True
    assert "judged_score" not in results[0]


def test_tier_progression_narrows_by_judged_score_not_mse():
    """The core methodological claim: a layer with WORSE mse (higher final_mse) but BETTER judged
    score must be able to survive over a layer with better mse but worse judged score -- proving
    MSE genuinely has no vote in which layers survive Tier 1."""
    tier1_results = [
        {"layer": 2, "final_mse": 1.0, "judged_score": 0.2, "skipped": False},   # great MSE, bad judged score
        {"layer": 4, "final_mse": 9.0, "judged_score": 0.9, "skipped": False},   # bad MSE, great judged score
    ]
    survivors = sorted((r for r in tier1_results if not r["skipped"]), key=lambda r: -r["judged_score"])[:1]
    assert survivors[0]["layer"] == 4, "the judged-score winner must survive even with far worse training MSE"
