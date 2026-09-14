"""Tests for evals/plotting.py -- checks each plot function actually produces a real, non-empty
PNG from synthetic data, and that edge cases (no usable rows, all points skipped) return None
instead of crashing or writing a blank/garbage file.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.plotting import (
    discover_group_keys,
    plot_compression_quality_frontier,
    plot_everything,
    plot_layer_sweep,
    plot_participation_ratio_vs_metric,
    plot_score_comparison,
    plot_sweep_file_everything,
)

SCORE_FIELDS = ["correct", "coherent"]


def _judged_rows(n=30, seed=0):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        rows.append({
            "id": str(i),
            "prompt_correct": rng.choice([0, 1, 2]), "prompt_coherent": True, "prompt_tokens": rng.randint(40, 60),
            "psr_proper_correct": rng.choice([0, 1, 2]), "psr_proper_coherent": True, "psr_proper_tokens": rng.randint(10, 20),
        })
    return rows


def test_plot_score_comparison_writes_one_png_per_field(tmp_path):
    rows = _judged_rows()
    saved = plot_score_comparison(rows, SCORE_FIELDS, tmp_path)
    assert len(saved) == 2  # "correct" and "coherent" both have data for both conditions
    for path in saved:
        assert path.exists()
        assert path.stat().st_size > 0


def test_plot_score_comparison_skips_fields_with_no_data():
    rows = [{"id": "0", "prompt_correct": 1}]  # no "coherent" anywhere
    saved = plot_score_comparison(rows, SCORE_FIELDS, Path("/tmp/plotting_test_no_data"))
    assert len(saved) == 1  # only "correct" produced a plot


def test_plot_compression_quality_frontier_writes_a_png(tmp_path):
    rows = _judged_rows()
    path = plot_compression_quality_frontier(rows, SCORE_FIELDS, tmp_path)
    assert path is not None and path.exists() and path.stat().st_size > 0


def test_plot_compression_quality_frontier_returns_none_without_token_data():
    rows = [{"id": "0", "prompt_correct": 1}]  # no "*_tokens" field at all
    path = plot_compression_quality_frontier(rows, SCORE_FIELDS, Path("/tmp/plotting_test_no_tokens"))
    assert path is None


def test_plot_layer_sweep_single_line(tmp_path):
    sweep_rows = [{"layer": l, "final_mse": 30 - l, "skipped": False} for l in [2, 6, 10, 14]]
    out = plot_layer_sweep(sweep_rows, tmp_path / "sweep.png")
    assert out is not None and out.exists() and out.stat().st_size > 0


def test_plot_layer_sweep_multiple_groups(tmp_path):
    sweep_rows = [
        {"layer": l, "alpha": a, "final_mse": 30 - l + a, "skipped": False}
        for l in [2, 6, 10] for a in [1.0, 4.0]
    ]
    out = plot_layer_sweep(sweep_rows, tmp_path / "sweep_grouped.png", group_key="alpha")
    assert out is not None and out.exists()


def test_plot_layer_sweep_excludes_skipped_rows_and_returns_none_if_all_skipped(tmp_path):
    all_skipped = [{"layer": l, "skipped": True} for l in [2, 6, 10]]
    assert plot_layer_sweep(all_skipped, tmp_path / "sweep_none.png") is None

    mixed = [{"layer": 2, "final_mse": 5.0, "skipped": False}, {"layer": 6, "skipped": True}]
    out = plot_layer_sweep(mixed, tmp_path / "sweep_mixed.png")
    assert out is not None  # at least one usable point


def test_plot_participation_ratio_vs_metric_writes_a_png(tmp_path):
    sweep_rows = [
        {"layer": l, "alpha": a, "final_mse": 10 - a, "participation_ratio": a * 2, "skipped": False}
        for l in [2, 6] for a in [1.0, 2.0, 4.0]
    ]
    out = plot_participation_ratio_vs_metric(sweep_rows, tmp_path / "pr_scatter.png")
    assert out is not None and out.exists() and out.stat().st_size > 0


def test_plot_participation_ratio_vs_metric_returns_none_without_pr_field(tmp_path):
    sweep_rows = [{"layer": 2, "final_mse": 5.0, "skipped": False}]
    out = plot_participation_ratio_vs_metric(sweep_rows, tmp_path / "pr_scatter_none.png")
    assert out is None


def test_plot_judged_results_produces_extra_conciseness_frontier_when_field_present(tmp_path, monkeypatch):
    import evals.plotting as plotting_module
    import adapters.registry as adapters_registry
    import evals.registry as evals_registry

    class _FakeAdapter:
        RESULTS_DIR = tmp_path

    class _FakeEvalAdapter:
        SCORE_FIELDS = ["correct", "coherent", "conciseness"]

    monkeypatch.setattr(adapters_registry, "get_adapter", lambda task: _FakeAdapter())
    monkeypatch.setattr(evals_registry, "get_eval_adapter", lambda task: _FakeEvalAdapter())

    rows = [
        {"id": "0", "prompt_correct": 2, "prompt_coherent": True, "prompt_conciseness": 0, "prompt_tokens": 90},
        {"id": "1", "prompt_correct": 1, "prompt_coherent": True, "prompt_conciseness": 1, "prompt_tokens": 95},
        {"id": "2", "psr_proper_correct": 1, "psr_proper_coherent": True, "psr_proper_conciseness": 2, "psr_proper_tokens": 15},
        {"id": "3", "psr_proper_correct": 2, "psr_proper_coherent": True, "psr_proper_conciseness": 2, "psr_proper_tokens": 18},
    ]
    (tmp_path / "judged_test.jsonl").write_text("\n".join(__import__("json").dumps(r) for r in rows) + "\n")

    saved = plotting_module.plot_judged_results("caveman", "test")
    names = {p.name for p in saved}
    assert "compression_quality_frontier_conciseness.png" in names
    assert "compression_quality_frontier_correct.png" in names


def test_discover_group_keys_finds_hyperparameters_and_excludes_fixed_fields():
    rows = [
        {"layer": 2, "alpha": 1.0, "nll_weight": 0.0, "final_mse": 5.0, "baseline_mse": 6.0,
         "final_nll": 1.0, "baseline_nll": 1.2, "participation_ratio": 3.0, "skipped": False},
    ]
    assert discover_group_keys(rows) == ["alpha", "nll_weight"]


def test_discover_group_keys_is_empty_for_a_sweep_with_only_layer():
    rows = [{"layer": 2, "final_mse": 5.0, "skipped": False}]
    assert discover_group_keys(rows) == []


def test_plot_sweep_file_everything_covers_every_metric_and_hyperparameter(tmp_path):
    """A conceptor/matrix-shaped sweep (layer x alpha x nll_weight, + participation_ratio, +
    both final_mse and final_nll) should produce: per metric (2) -> 1 ungrouped + 2 grouped
    (alpha, nll_weight) + 1 PR-vs-metric = 4 plots/metric x 2 metrics = 8 plots total."""
    rows = [
        {"layer": l, "alpha": a, "nll_weight": w, "final_mse": 30 - l + a + w, "final_nll": 2 - w,
         "baseline_mse": 40.0, "baseline_nll": 3.0, "participation_ratio": a, "skipped": False}
        for l in [2, 6, 10] for a in [1.0, 4.0] for w in [0.0, 0.1]
    ]
    sweep_path = tmp_path / "psr_conceptor_matrix_sweep.jsonl"
    sweep_path.write_text("\n".join(__import__("json").dumps(r) for r in rows) + "\n")

    saved = plot_sweep_file_everything(sweep_path, tmp_path / "plots")
    assert len(saved) == 8
    names = {p.name for p in saved}
    assert "psr_conceptor_matrix_sweep_final_mse_by_layer.png" in names
    assert "psr_conceptor_matrix_sweep_final_mse_by_layer_grouped_alpha.png" in names
    assert "psr_conceptor_matrix_sweep_final_mse_by_layer_grouped_nll_weight.png" in names
    assert "psr_conceptor_matrix_sweep_pr_vs_final_mse.png" in names
    assert "psr_conceptor_matrix_sweep_pr_vs_final_nll.png" in names


def test_plot_sweep_file_everything_handles_proper_shaped_sweep_without_pr(tmp_path):
    """A proper-shaped sweep (layer x nll_weight only, no participation_ratio, no final_nll if
    a caller only logged mse) should still produce plots for whatever metrics/keys ARE present,
    with no crash from the missing fields."""
    rows = [{"layer": l, "nll_weight": w, "final_mse": 30 - l + w, "skipped": False}
            for l in [2, 6, 10] for w in [0.0, 0.1]]
    sweep_path = tmp_path / "psr_proper_sweep.jsonl"
    sweep_path.write_text("\n".join(__import__("json").dumps(r) for r in rows) + "\n")

    saved = plot_sweep_file_everything(sweep_path, tmp_path / "plots")
    # final_mse only (no final_nll present) -> 1 ungrouped + 1 grouped(nll_weight), no PR plot
    assert len(saved) == 2


def test_plot_sweep_file_everything_empty_file_returns_empty_list(tmp_path):
    sweep_path = tmp_path / "empty_sweep.jsonl"
    sweep_path.write_text("")
    assert plot_sweep_file_everything(sweep_path, tmp_path / "plots") == []


def test_plot_everything_combines_judged_and_every_sweep_file(tmp_path, monkeypatch):
    import evals.plotting as plotting_module
    import adapters.registry as adapters_registry
    import evals.registry as evals_registry

    class _FakeAdapter:
        RESULTS_DIR = tmp_path

    class _FakeEvalAdapter:
        SCORE_FIELDS = ["correct", "coherent", "conciseness"]

    monkeypatch.setattr(adapters_registry, "get_adapter", lambda task: _FakeAdapter())
    monkeypatch.setattr(evals_registry, "get_eval_adapter", lambda task: _FakeEvalAdapter())

    judged_rows = [
        {"id": "0", "prompt_correct": 2, "prompt_coherent": True, "prompt_conciseness": 0, "prompt_tokens": 90},
        {"id": "1", "psr_proper_correct": 1, "psr_proper_coherent": True, "psr_proper_conciseness": 2, "psr_proper_tokens": 15},
    ]
    (tmp_path / "judged_test.jsonl").write_text("\n".join(__import__("json").dumps(r) for r in judged_rows) + "\n")

    proper_rows = [{"layer": l, "nll_weight": w, "final_mse": 30 - l + w, "skipped": False}
                   for l in [2, 6] for w in [0.0, 0.1]]
    (tmp_path / "psr_proper_sweep.jsonl").write_text("\n".join(__import__("json").dumps(r) for r in proper_rows) + "\n")

    matrix_rows = [
        {"layer": l, "alpha": a, "final_mse": 30 - l + a, "participation_ratio": a, "skipped": False}
        for l in [2, 6] for a in [1.0, 4.0]
    ]
    (tmp_path / "psr_conceptor_matrix_sweep.jsonl").write_text("\n".join(__import__("json").dumps(r) for r in matrix_rows) + "\n")

    saved = plotting_module.plot_everything("caveman", "test")
    # judged: 3 score-comparison + 2 frontier = 5. proper sweep: 2 (1 ungrouped + 1 grouped).
    # matrix sweep: 1 ungrouped + 1 grouped(alpha) + 1 PR-scatter = 3. Total = 10.
    assert len(saved) == 10
    for p in saved:
        assert p.exists() and p.stat().st_size > 0
