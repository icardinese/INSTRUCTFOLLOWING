"""Analysis + plots for the all-layer results, recomputed from results/<task>/
all_layer_variants_eval.json. NO GPU AND NO API CALLS -- that file stores the raw per-example
scores, so every statistic here is derived from disk.

Exists for two reasons:

1. CORRECTS AN INVERTED SIGN. The 2026-09-18 run's printed verdicts were all backwards:
   scripts/eval_all_layer_variants.py called paired_bootstrap_diff(prompt, candidate), which
   returns mean(prompt) - mean(candidate), so a positive diff meant the CANDIDATE WAS WORSE -- but
   the verdict string said "BEATS Prompt". The magnitudes were right, only the labels lied. This
   script recomputes with candidate FIRST, so positive = candidate better, and prints the verdicts
   that actually follow from the numbers. Rerunning the GPU work is not necessary or useful.

2. PLOTS THAT SHOW THE EFFECT. evals/plotting.py's score_comparison_* bars have three problems
   that make real differences invisible (all three visible in score_comparison_correct.png):
     - y-axis anchored at 0 while every score sits in [1.6, 2.0], so 100% of the variation is
       squeezed into the top tenth of the figure and every bar looks identical;
     - conditions sorted ALPHABETICALLY, which interleaves baselines, steering-alone and
       Prompt+steering instead of grouping them;
     - raw column names ("prompt_psr_proper") as tick labels at an angle that still collides.
   The plots here fix all three: limits from the data, explicit logical ordering, readable labels.

The frontier plot is the important one for the paper -- this task has two axes that trade off
(correctness, brevity), and a single bar chart on either axis alone can't show a tradeoff.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adapters.registry import get_adapter
from evals.bootstrap_analysis import bootstrap_ci, paired_bootstrap_diff

# Logical display order and short labels. Baselines, then steering-alone, then Prompt+steering --
# so the eye moves through the three regimes instead of alphabetical noise.
ORDER = [
    ("base", "Base", "baseline"),
    ("prompt", "Prompt", "baseline"),
    ("psr_all_layer_probe_mse_alone", "A-PSR (MSE)", "alone"),
    ("psr_all_layer_probe_nll_alone", "A-PSR (NLL)", "alone"),
    ("psr_multi_gate_probe_mse_alone", "Multi-Gate (MSE)", "alone"),
    ("psr_multi_gate_probe_nll_alone", "Multi-Gate (NLL)", "alone"),
    ("psr_multi_gate_conceptor_probe_mse_alone", "MG-Conceptor (MSE)", "alone"),
    ("psr_multi_gate_conceptor_probe_nll_alone", "MG-Conceptor (NLL)", "alone"),
    ("psr_all_layer_probe_mse_combined", "P+A-PSR (MSE)", "combined"),
    ("psr_all_layer_probe_nll_combined", "P+A-PSR (NLL)", "combined"),
    ("psr_multi_gate_probe_mse_combined", "P+Multi-Gate (MSE)", "combined"),
    ("psr_multi_gate_probe_nll_combined", "P+Multi-Gate (NLL)", "combined"),
    ("psr_multi_gate_conceptor_probe_mse_combined", "P+MG-Conceptor (MSE)", "combined"),
    ("psr_multi_gate_conceptor_probe_nll_combined", "P+MG-Conceptor (NLL)", "combined"),
]
GROUP_COLORS = {"baseline": "#6c757d", "alone": "#2a6f97", "combined": "#c1440e"}


def load(task: str) -> dict:
    adapter = get_adapter(task)
    path = adapter.RESULTS_DIR / "all_layer_variants_eval.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run scripts/eval_all_layer_variants.py first")
    with path.open() as f:
        return json.load(f), adapter


def main(task: str) -> None:
    payload, adapter = load(task)
    n = payload["n"]
    res = payload["results"]
    present = [(k, label, grp) for k, label, grp in ORDER if k in res]

    rows = []
    for key, label, grp in present:
        r = res[key]
        pt, lo, hi = bootstrap_ci(r["scores"])
        rows.append({
            "key": key, "label": label, "group": grp,
            "correct": pt, "correct_lo": lo, "correct_hi": hi,
            "avg_tokens": r["avg_tokens"], "coherent_rate": r["coherent_rate"],
        })

    print(f"\n=== all-layer results, n={n} (recomputed from disk) ===")
    print(f"{'condition':<22}{'avg_tokens':>12}{'correct':>10}{'  95% CI':>20}{'coherent':>11}")
    print("-" * 75)
    for r in rows:
        ci = f"[{r['correct_lo']:.3f}, {r['correct_hi']:.3f}]"
        print(f"{r['label']:<22}{r['avg_tokens']:>12.1f}{r['correct']:>10.3f}{ci:>20}{r['coherent_rate']:>11.2f}")

    print(f"\n=== paired vs Prompt alone, CORRECTED SIGN (candidate - prompt; positive = better) ===")
    prompt_scores = res["prompt"]["scores"]
    prompt_tokens = res["prompt"]["avg_tokens"]
    for key, label, grp in present:
        if grp == "baseline":
            continue
        diff, lo, hi = paired_bootstrap_diff(res[key]["scores"], prompt_scores)
        verdict = "BEATS Prompt" if lo > 0 else "WORSE than Prompt" if hi < 0 else "ties Prompt"
        tok = res[key]["avg_tokens"]
        tok_note = "fewer tokens" if tok < prompt_tokens else "MORE tokens"
        print(f"  {label:<22} correctness {diff:+.4f} [{lo:+.4f}, {hi:+.4f}] {verdict:<18}"
              f" | {tok:.1f} vs {prompt_tokens:.1f} tok ({tok_note})")

    plots_dir = adapter.RESULTS_DIR.parent.parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # --- Frontier: the paper's key figure. Two trading-off axes need a 2D plot. ---
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for grp in ("baseline", "alone", "combined"):
        pts = [r for r in rows if r["group"] == grp]
        if not pts:
            continue
        ax.errorbar(
            [p["avg_tokens"] for p in pts], [p["correct"] for p in pts],
            yerr=[[p["correct"] - p["correct_lo"] for p in pts], [p["correct_hi"] - p["correct"] for p in pts]],
            fmt="o", ms=9, capsize=3, lw=1, color=GROUP_COLORS[grp],
            label={"baseline": "Baselines", "alone": "Steering alone", "combined": "Prompt + steering"}[grp],
        )
    # Label placement: alternate above/below within each group, ordered by x. Point-distance
    # collision tests don't work here because what actually overlaps is the label TEXT (which is
    # much wider than the gap between two points at similar correctness), and its width in data
    # units isn't known before rendering. Strict alternation guarantees no two x-adjacent labels
    # ever share a row, which is sufficient regardless of text width.
    for grp in ("baseline", "alone", "combined"):
        grp_rows = sorted([r for r in rows if r["group"] == grp], key=lambda r: r["avg_tokens"])
        for i, r in enumerate(grp_rows):
            r["_dy"] = 6 if i % 2 == 0 else -14
    for r in rows:
        ax.annotate(r["label"], (r["avg_tokens"], r["correct"]), textcoords="offset points",
                    xytext=(8, r["_dy"]), fontsize=8.5)
    # Headroom so the rightmost label (Base, at the token ceiling) doesn't run off the axes.
    ax.set_xlim(min(r["avg_tokens"] for r in rows) - 14, max(r["avg_tokens"] for r in rows) + 28)
    # Reference lines make "better than Prompt on this axis" readable at a glance.
    pr = next(r for r in rows if r["key"] == "prompt")
    ax.axhline(pr["correct"], ls="--", lw=0.9, color="#adb5bd", zorder=0)
    ax.axvline(pr["avg_tokens"], ls="--", lw=0.9, color="#adb5bd", zorder=0)
    ax.annotate("better\n(fewer tokens, more correct)", (pr["avg_tokens"], pr["correct"]),
                textcoords="offset points", xytext=(-135, 18), fontsize=8, color="#495057")
    ax.set_xlabel("avg response tokens  (lower is better)")
    ax.set_ylabel("correctness, judged 0-2  (higher is better)")
    ax.set_title(f"Accuracy / brevity frontier, all-layer methods (n={n})")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(alpha=0.25, lw=0.6)
    fig.tight_layout()
    p1 = plots_dir / "all_layer_frontier.png"
    fig.savefig(p1, dpi=200)
    plt.close(fig)

    # --- Bars, with the three fixes described in the module docstring ---
    for metric, ylabel, fname in [
        ("correct", "correctness (judged 0-2)", "all_layer_correct_bars.png"),
        ("avg_tokens", "avg response tokens", "all_layer_tokens_bars.png"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5.2))
        xs = range(len(rows))
        vals = [r[metric] for r in rows]
        colors = [GROUP_COLORS[r["group"]] for r in rows]
        if metric == "correct":
            err = [[r["correct"] - r["correct_lo"] for r in rows], [r["correct_hi"] - r["correct"] for r in rows]]
            ax.bar(xs, vals, color=colors, yerr=err, capsize=3, error_kw={"lw": 1})
            # Limits from the DATA, not anchored at 0 -- this is the single change that makes the
            # differences visible at all (scores live in a narrow band near the ceiling).
            lo = min(r["correct_lo"] for r in rows)
            hi = max(r["correct_hi"] for r in rows)
            pad = (hi - lo) * 0.18 or 0.05
            ax.set_ylim(max(0, lo - pad), min(2.0, hi + pad))
        else:
            ax.bar(xs, vals, color=colors)
            ax.set_ylim(0, max(vals) * 1.12)
        ref = next(r for r in rows if r["key"] == "prompt")[metric]
        ax.axhline(ref, ls="--", lw=1, color="#212529", zorder=3)
        ax.annotate("Prompt", (-0.45, ref), textcoords="offset points", xytext=(2, 3),
                    fontsize=8.5, color="#212529")
        ax.set_xticks(list(xs))
        ax.set_xticklabels([r["label"] for r in rows], rotation=32, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} by condition (95% bootstrap CI, n={n})" if metric == "correct"
                      else f"{ylabel} by condition (n={n})")
        ax.grid(axis="y", alpha=0.25, lw=0.6)
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in GROUP_COLORS.values()]
        # lower left: the Prompt reference line and the tallest bars both live in the upper area,
        # so a default ("best") placement collides with the line's label.
        ax.legend(handles, ["Baseline", "Steering alone", "Prompt + steering"],
                   frameon=False, fontsize=9, loc="lower left")
        fig.tight_layout()
        fig.savefig(plots_dir / fname, dpi=200)
        plt.close(fig)

    print(f"\nwrote {p1}")
    print(f"wrote {plots_dir / 'all_layer_correct_bars.png'}")
    print(f"wrote {plots_dir / 'all_layer_tokens_bars.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="caveman", choices=["caveman", "ifeval"])
    args = parser.parse_args()
    main(args.task)
