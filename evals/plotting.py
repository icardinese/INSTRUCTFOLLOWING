"""Visualizations for judged results and for layer/hyperparameter sweeps. Task-agnostic (works off
score_fields, never a hardcoded caveman/ifeval branch) and method-agnostic (sweep plots take plain
rows-of-dicts with a "layer" key and whatever metric/hyperparameter keys the caller names -- they
don't know or care which steering method produced them).

This fills the "plotting regression" flagged in the project handoff (evals/summarize.py used to
only print a table) and goes past the original scatter-plot-with-legend it replaces: four plot
kinds, not one, because the actual ask is exploratory ("as many datapoints as possible, so a new
method idea might fall out"), not just a paper figure.

Reuses rather than re-derives: bootstrap_ci (evals/bootstrap_analysis.py) for every error bar, and
collect_raw_values_by_cond_field (evals/summarize.py) for every condition/field lookup -- so a plot
can never silently disagree with the numbers evals/summarize.py or evals/bootstrap_analysis.py
already printed for the same data.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless -- this runs on a GPU pod / CI, never an interactive display
import matplotlib.pyplot as plt

from evals.bootstrap_analysis import bootstrap_ci
from evals.summarize import collect_raw_values_by_cond_field


def _save(fig, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_score_comparison(rows: list[dict], score_fields: list[str], out_dir: Path) -> list[Path]:
    """One bar chart per score field: one bar per condition, with a 95% bootstrap CI whisker
    (point estimate +/- the same [lo, hi] evals/bootstrap_analysis.py's CLI output already prints).
    Returns the list of saved PNG paths, one per field that had at least one condition with data."""
    values_by_cond = collect_raw_values_by_cond_field(rows, score_fields)
    saved = []
    for field in score_fields:
        conds = sorted(c for c in values_by_cond if field in values_by_cond[c])
        if not conds:
            continue
        points, err_lo, err_hi = [], [], []
        for cond in conds:
            point, lo, hi = bootstrap_ci(values_by_cond[cond][field])
            points.append(point)
            err_lo.append(point - lo)
            err_hi.append(hi - point)

        fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(conds)), 4.5))
        ax.bar(conds, points, yerr=[err_lo, err_hi], capsize=4, color="#4C72B0")
        ax.set_ylabel(field)
        ax.set_title(f"{field} by condition (95% bootstrap CI, n={len(rows)})")
        ax.tick_params(axis="x", rotation=45)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        fig.tight_layout()
        saved.append(_save(fig, out_dir / f"score_comparison_{field}.png"))
    return saved


def plot_compression_quality_frontier(rows: list[dict], score_fields: list[str], out_dir: Path, quality_field: str | None = None) -> Path | None:
    """Scatter of avg_tokens (x) vs a quality field (y), one point per condition, both axes with
    95% bootstrap CI error bars (reusing the exact same avg_tokens list summarize.py already
    aggregates as a mean, and score field list every other plot here uses). This is the
    "how much correctness did we give up for how much compression" plot -- the one most directly
    aimed at spotting a new steering-method idea, not just reporting a number for a paper."""
    quality_field = quality_field or (score_fields[0] if score_fields else None)
    if quality_field is None:
        return None
    values_by_cond = collect_raw_values_by_cond_field(rows, score_fields)
    conds = sorted(c for c in values_by_cond if quality_field in values_by_cond[c] and "avg_tokens" in values_by_cond[c])
    if not conds:
        return None

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for cond in conds:
        x, x_lo, x_hi = bootstrap_ci(values_by_cond[cond]["avg_tokens"])
        y, y_lo, y_hi = bootstrap_ci(values_by_cond[cond][quality_field])
        ax.errorbar(x, y, xerr=[[x - x_lo], [x_hi - x]], yerr=[[y - y_lo], [y_hi - y]], fmt="o", capsize=4, markersize=7)
        ax.annotate(cond, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("avg_tokens (lower = more compressed)")
    ax.set_ylabel(quality_field)
    ax.set_title(f"compression vs. {quality_field} tradeoff (95% CI both axes, n={len(rows)})")
    fig.tight_layout()
    return _save(fig, out_dir / f"compression_quality_frontier_{quality_field}.png")


def plot_layer_sweep(sweep_rows: list[dict], out_path: Path, metric_key: str = "final_mse", group_key: str | None = None, layer_key: str = "layer") -> Path | None:
    """Line plot of metric_key vs. layer_key from a sweep JSONL (any variant's
    results/<task>/psr_<variant>_sweep.jsonl) -- one line per distinct group_key value if given
    (e.g. "alpha" or "nll_weight"), else a single line. Rows with skipped=True (selfproj's
    delta_scale-too-small case) or missing metric_key are dropped, not plotted as zero."""
    usable = [r for r in sweep_rows if not r.get("skipped", False) and metric_key in r and layer_key in r]
    if not usable:
        return None

    fig, ax = plt.subplots(figsize=(7, 5))
    if group_key is None:
        pts = sorted(usable, key=lambda r: r[layer_key])
        ax.plot([r[layer_key] for r in pts], [r[metric_key] for r in pts], marker="o")
    else:
        groups = sorted({r[group_key] for r in usable})
        for g in groups:
            pts = sorted((r for r in usable if r[group_key] == g), key=lambda r: r[layer_key])
            ax.plot([r[layer_key] for r in pts], [r[metric_key] for r in pts], marker="o", label=f"{group_key}={g}")
        ax.legend(fontsize=8)
    ax.set_xlabel(layer_key)
    ax.set_ylabel(metric_key)
    ax.set_title(f"{metric_key} vs. {layer_key}" + (f" (by {group_key})" if group_key else ""))
    fig.tight_layout()
    return _save(fig, out_path)


def plot_participation_ratio_vs_metric(sweep_rows: list[dict], out_path: Path, metric_key: str = "final_mse", pr_key: str = "participation_ratio", color_by: str | None = "layer") -> Path | None:
    """Scatter of participation_ratio (x) vs. metric_key (y) across every USABLE grid point in a
    matrix-based method's sweep results -- the cheap, dense dataset (no generation/judging needed,
    just dev-set forward passes already computed during the sweep) that directly answers "does
    accuracy [here, dev MSE -- lower is better] increase or decrease as the conceptor's effective
    dimensionality changes". Points are colored by color_by (default "layer") if that key is
    present, so a layer-driven confound is at least visible rather than hidden inside one blob."""
    usable = [r for r in sweep_rows if not r.get("skipped", False) and metric_key in r and pr_key in r]
    if not usable:
        return None

    fig, ax = plt.subplots(figsize=(7, 5.5))
    xs = [r[pr_key] for r in usable]
    ys = [r[metric_key] for r in usable]
    if color_by and all(color_by in r for r in usable):
        colors = [r[color_by] for r in usable]
        sc = ax.scatter(xs, ys, c=colors, cmap="viridis")
        fig.colorbar(sc, ax=ax, label=color_by)
    else:
        ax.scatter(xs, ys)
    ax.set_xlabel(f"{pr_key} (effective dimensionality of C)")
    ax.set_ylabel(metric_key)
    ax.set_title(f"{metric_key} vs. {pr_key} across sweep grid points (n={len(usable)})")
    fig.tight_layout()
    return _save(fig, out_path)


def plot_judged_results(task: str, split: str) -> list[Path]:
    """CLI-friendly entry point for the judged-data plots (score comparison + compression/quality
    frontier) -- mirrors evals/summarize.py's own --task/--split contract exactly. If the task's
    judge exposes a "conciseness" field (currently only caveman's does -- see
    evals/caveman/judge.py), an EXTRA frontier plot is produced against it specifically, since
    that's the direct comparison this feature exists for: raw avg_tokens is a proxy for
    terseness, conciseness is an actual LLM-judged score of whether the wording itself is padded
    -- the two can and do diverge (same token count, different amount of hedging/repetition)."""
    from adapters.registry import get_adapter
    from evals.registry import get_eval_adapter

    adapter = get_adapter(task)
    eval_adapter = get_eval_adapter(task)
    with (adapter.RESULTS_DIR / f"judged_{split}.jsonl").open() as f:
        rows = [json.loads(line) for line in f]

    out_dir = adapter.RESULTS_DIR / "plots"
    saved = plot_score_comparison(rows, eval_adapter.SCORE_FIELDS, out_dir)
    frontier = plot_compression_quality_frontier(rows, eval_adapter.SCORE_FIELDS, out_dir)
    if frontier:
        saved.append(frontier)
    if "conciseness" in eval_adapter.SCORE_FIELDS:
        conciseness_frontier = plot_compression_quality_frontier(rows, eval_adapter.SCORE_FIELDS, out_dir, quality_field="conciseness")
        if conciseness_frontier:
            saved.append(conciseness_frontier)
    return saved


def plot_sweep_file(sweep_path: Path, out_dir: Path, metric_key: str = "final_mse", group_key: str | None = None) -> list[Path]:
    """CLI-friendly entry point for a single sweep JSONL -- always produces the layer-sweep plot,
    and additionally the participation_ratio scatter if the file actually has that field (proper's
    and fixed-vector conceptor's sweeps won't; conceptor/matrix's and conceptor/selfproj's will)."""
    with sweep_path.open() as f:
        rows = [json.loads(line) for line in f]

    saved = []
    layer_plot = plot_layer_sweep(rows, out_dir / f"{sweep_path.stem}_layer_sweep.png", metric_key=metric_key, group_key=group_key)
    if layer_plot:
        saved.append(layer_plot)
    if any("participation_ratio" in r for r in rows):
        pr_plot = plot_participation_ratio_vs_metric(rows, out_dir / f"{sweep_path.stem}_pr_vs_{metric_key}.png", metric_key=metric_key)
        if pr_plot:
            saved.append(pr_plot)
    return saved


# Keys every sweep row can carry that are NOT themselves a swept hyperparameter -- everything
# else present in a sweep JSONL's rows is treated as a candidate group_key for
# plot_sweep_file_everything, so a FUTURE variant's new hyperparameter (added to its own grid
# dict) gets its own grouped plot automatically, with no change needed here.
_NON_HYPERPARAM_SWEEP_KEYS = {
    "layer", "skipped", "baseline_mse", "baseline_nll", "final_mse", "final_nll",
    "participation_ratio", "seed", "delta_scale",
}
_CANDIDATE_SWEEP_METRICS = ["final_mse", "final_nll"]


def discover_group_keys(sweep_rows: list[dict]) -> list[str]:
    """Every key present in the sweep's rows, minus the fixed non-hyperparameter set above,
    restricted to keys that actually hold a plain scalar somewhere (str/int/float/bool) -- so a
    key that only ever held a skipped-point placeholder or similar doesn't produce a useless plot."""
    all_keys = set()
    for row in sweep_rows:
        all_keys.update(row.keys())
    candidates = all_keys - _NON_HYPERPARAM_SWEEP_KEYS
    return sorted(k for k in candidates if any(isinstance(row.get(k), (int, float, str, bool)) for row in sweep_rows))


def plot_sweep_file_everything(sweep_path: Path, out_dir: Path) -> list[Path]:
    """Maximum-coverage version of plot_sweep_file: for every metric actually present
    (final_mse and/or final_nll) and every discovered hyperparameter, produces an ungrouped
    layer-sweep plot AND one grouped-by-that-hyperparameter plot, plus a PR-vs-metric scatter per
    metric if participation_ratio is present. This is what plot_everything() calls per sweep file
    -- breadth over curation, on purpose (see project ask: "as many datapoints as possible")."""
    with sweep_path.open() as f:
        rows = [json.loads(line) for line in f]
    if not rows:
        return []

    metrics = [m for m in _CANDIDATE_SWEEP_METRICS if any(m in r for r in rows)]
    group_keys = discover_group_keys(rows)
    has_pr = any("participation_ratio" in r for r in rows)

    saved = []
    for metric in metrics:
        ungrouped = plot_layer_sweep(rows, out_dir / f"{sweep_path.stem}_{metric}_by_layer.png", metric_key=metric)
        if ungrouped:
            saved.append(ungrouped)
        for group_key in group_keys:
            grouped = plot_layer_sweep(
                rows, out_dir / f"{sweep_path.stem}_{metric}_by_layer_grouped_{group_key}.png",
                metric_key=metric, group_key=group_key,
            )
            if grouped:
                saved.append(grouped)
        if has_pr:
            pr_plot = plot_participation_ratio_vs_metric(rows, out_dir / f"{sweep_path.stem}_pr_vs_{metric}.png", metric_key=metric)
            if pr_plot:
                saved.append(pr_plot)
    return saved


def plot_everything(task: str, split: str = "test") -> list[Path]:
    """The "generate every plot we reasonably can" entry point: judged-result plots (if
    judged_{split}.jsonl exists) plus plot_sweep_file_everything for every *_sweep.jsonl found in
    results/<task>/ (however many variants have been swept so far -- gracefully however many or
    few that is, no hardcoded list of expected filenames)."""
    from adapters.registry import get_adapter
    adapter = get_adapter(task)

    saved = []
    judged_path = adapter.RESULTS_DIR / f"judged_{split}.jsonl"
    if judged_path.exists():
        saved.extend(plot_judged_results(task, split))
    else:
        print(f"NOTE: {judged_path} not found -- skipping score-comparison/frontier plots")

    out_dir = adapter.RESULTS_DIR / "plots"
    sweep_files = sorted(adapter.RESULTS_DIR.glob("*_sweep.jsonl"))
    if not sweep_files:
        print(f"NOTE: no *_sweep.jsonl files found in {adapter.RESULTS_DIR} -- skipping sweep plots")
    for sweep_path in sweep_files:
        saved.extend(plot_sweep_file_everything(sweep_path, out_dir))
    return saved


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    judged = sub.add_parser("judged", help="score comparison + compression/quality frontier from judged_{split}.jsonl")
    judged.add_argument("--task", required=True, choices=["caveman", "ifeval"])
    judged.add_argument("--split", default="test")

    sweep = sub.add_parser("sweep", help="layer-sweep (+ PR-vs-metric, if present) plots from a sweep JSONL")
    sweep.add_argument("--file", required=True, help="path to a *_sweep.jsonl file")
    sweep.add_argument("--out-dir", required=True)
    sweep.add_argument("--metric", default="final_mse")
    sweep.add_argument("--group-by", default=None, help='e.g. "alpha" or "nll_weight"')

    everything = sub.add_parser("everything", help="maximum coverage: judged plots + every discovered *_sweep.jsonl, every metric, every hyperparameter grouping")
    everything.add_argument("--task", required=True, choices=["caveman", "ifeval"])
    everything.add_argument("--split", default="test")

    args = parser.parse_args()
    if args.mode == "judged":
        paths = plot_judged_results(args.task, args.split)
    elif args.mode == "sweep":
        paths = plot_sweep_file(Path(args.file), Path(args.out_dir), args.metric, args.group_by)
    else:
        paths = plot_everything(args.task, args.split)
    print(f"\n{len(paths)} plot(s) written:")
    for p in paths:
        print(f"  {p}")
