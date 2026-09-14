"""Bootstrap CIs for every condition x score field, auto-detected the same way summarize.py does.
The bootstrap math itself (bootstrap_ci, paired_bootstrap_diff) is unchanged from its original,
already-verified form -- only the condition/field detection is new, generalized off SCORE_FIELDS
instead of a hardcoded per-task condition list.
"""
import argparse
import json
import random

from adapters.registry import get_adapter
from core.reproducibility import set_seed
from evals.registry import get_eval_adapter
from evals.summarize import parse_condition_and_field

N_BOOTSTRAP = 2000
CI_LOW, CI_HIGH = 2.5, 97.5


def bootstrap_ci(values: list, n_bootstrap: int = N_BOOTSTRAP) -> tuple[float, float, float]:
    n = len(values)
    point = sum(values) / n
    boot_means = []
    for _ in range(n_bootstrap):
        sample = [values[random.randrange(n)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    lo = boot_means[int(n_bootstrap * CI_LOW / 100)]
    hi = boot_means[int(n_bootstrap * CI_HIGH / 100)]
    return point, lo, hi


def paired_bootstrap_diff(values_a: list, values_b: list, n_bootstrap: int = N_BOOTSTRAP) -> tuple[float, float, float]:
    """CI on (mean(a) - mean(b)), resampling the SAME indices for both -- correct here since every
    condition is evaluated on the exact same underlying rows."""
    n = len(values_a)
    assert len(values_b) == n
    point = sum(values_a) / n - sum(values_b) / n
    diffs = []
    for _ in range(n_bootstrap):
        idx = [random.randrange(n) for _ in range(n)]
        a = sum(values_a[i] for i in idx) / n
        b = sum(values_b[i] for i in idx) / n
        diffs.append(a - b)
    diffs.sort()
    lo = diffs[int(n_bootstrap * CI_LOW / 100)]
    hi = diffs[int(n_bootstrap * CI_HIGH / 100)]
    return point, lo, hi


def collect_values_by_cond_field(rows: list[dict], score_fields: list[str]) -> dict:
    values = {}
    for row in rows:
        for key, value in row.items():
            if key == "id":
                continue
            parsed = parse_condition_and_field(key, score_fields)
            if parsed:
                cond, field = parsed
                values.setdefault(cond, {}).setdefault(field, []).append(value)
    return values


def main(task: str, split: str, compare: str | None, seed: int = 42) -> None:
    set_seed(seed)
    adapter = get_adapter(task)
    eval_adapter = get_eval_adapter(task)

    with (adapter.RESULTS_DIR / f"judged_{split}.jsonl").open() as f:
        rows = [json.loads(line) for line in f]

    values = collect_values_by_cond_field(rows, eval_adapter.SCORE_FIELDS)

    print(f"{len(rows)} rows, 95% bootstrap CIs\n")
    for cond in sorted(values):
        for field in eval_adapter.SCORE_FIELDS:
            if field not in values[cond]:
                continue
            point, lo, hi = bootstrap_ci(values[cond][field])
            print(f"{cond:<28}{field:<26}{point:>8.3f}  [{lo:.3f}, {hi:.3f}]")

    if compare:
        cond_a, cond_b = compare.split(",")
        print(f"\n--- paired comparison: {cond_a} vs {cond_b} ---")
        for field in eval_adapter.SCORE_FIELDS:
            if field not in values.get(cond_a, {}) or field not in values.get(cond_b, {}):
                continue
            point, lo, hi = paired_bootstrap_diff(values[cond_a][field], values[cond_b][field])
            flag = "  <-- excludes 0, likely real" if (lo > 0 or hi < 0) else "  (includes 0, could be noise)"
            print(f"{field:<26}diff={point:>8.3f}  [{lo:.3f}, {hi:.3f}]{flag}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["caveman", "ifeval"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--compare", default=None, help="e.g. 'prompt_psr_proper,prompt_psr_conceptor'")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.task, args.split, args.compare, args.seed)
