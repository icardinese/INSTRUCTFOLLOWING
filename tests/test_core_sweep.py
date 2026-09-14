"""Tests for core/sweep.py -- pure-Python logic (resumability + best-tracking), no torch or any
real model needed since train_fn is an opaque callable as far as this file is concerned.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.sweep import run_grid_sweep


def test_runs_every_point_and_writes_jsonl(tmp_path):
    calls = []

    def train_fn(point):
        calls.append(point)
        return {"final_mse": point["layer"] * 1.0}

    grid = [{"layer": l} for l in [2, 4, 6]]
    results, best = run_grid_sweep(grid, train_fn, tmp_path / "sweep.jsonl", key_fields=["layer"])

    assert len(calls) == 3
    assert len(results) == 3
    assert best == {"layer": 2, "final_mse": 2.0}
    with (tmp_path / "sweep.jsonl").open() as f:
        assert sum(1 for _ in f) == 3


def test_resumes_and_does_not_recall_train_fn_for_done_points(tmp_path):
    out_path = tmp_path / "sweep.jsonl"
    out_path.write_text(json.dumps({"layer": 2, "final_mse": 99.0}) + "\n")

    calls = []

    def train_fn(point):
        calls.append(point)
        return {"final_mse": 1.0}

    results, best = run_grid_sweep([{"layer": 2}, {"layer": 4}], train_fn, out_path, key_fields=["layer"])

    assert calls == [{"layer": 4}], "layer=2 was already done and must not be recomputed"
    assert len(results) == 2
    # the pre-existing (stale-looking) row for layer=2 is kept as-is, not overwritten
    assert any(r["layer"] == 2 and r["final_mse"] == 99.0 for r in results)


def test_best_selection_respects_minimize_flag():
    def train_fn(point):
        return {"accuracy": point["layer"] * 0.1}

    grid = [{"layer": l} for l in [2, 4, 6]]

    def _run(tmp_path, minimize):
        return run_grid_sweep(grid, train_fn, tmp_path / "sweep.jsonl", key_fields=["layer"],
                               select_best_by="accuracy", minimize=minimize)

    import tempfile
    with tempfile.TemporaryDirectory() as d1:
        _, best_min = _run(Path(d1), minimize=True)
        assert best_min["layer"] == 2
    with tempfile.TemporaryDirectory() as d2:
        _, best_max = _run(Path(d2), minimize=False)
        assert best_max["layer"] == 6


def test_skipped_points_are_written_but_excluded_from_best(tmp_path):
    def train_fn(point):
        if point["alpha"] == 1.0:
            return {"skipped": True}
        return {"final_mse": 10.0}

    grid = [{"alpha": 1.0}, {"alpha": 2.0}]
    results, best = run_grid_sweep(grid, train_fn, tmp_path / "sweep.jsonl", key_fields=["alpha"])

    assert len(results) == 2, "skipped points are still recorded"
    assert best["alpha"] == 2.0, "the skipped point must never be selected as best"


def test_multi_field_key_disambiguates_points_that_share_one_field(tmp_path):
    """layer=2,alpha=1.0 and layer=2,alpha=2.0 must be treated as DIFFERENT points -- a bug that
    only matched on `layer` would wrongly treat the second as already-done after the first ran."""
    calls = []

    def train_fn(point):
        calls.append((point["layer"], point["alpha"]))
        return {"final_mse": 1.0}

    grid = [{"layer": 2, "alpha": 1.0}, {"layer": 2, "alpha": 2.0}]
    run_grid_sweep(grid, train_fn, tmp_path / "sweep.jsonl", key_fields=["layer", "alpha"])
    assert set(calls) == {(2, 1.0), (2, 2.0)}
