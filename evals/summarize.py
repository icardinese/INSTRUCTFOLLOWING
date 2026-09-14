"""Summarizes judged_{split}.jsonl: for every condition, the mean of every score field (correctness,
coherence, tokens, whatever the task's judge returns) plus average token count and, for matrix-based
conditions, participation ratio. Task-agnostic -- never touched when adding a new task, same as
run_judge.py.
"""
import argparse
import json
from collections import defaultdict

from adapters.registry import get_adapter
from evals.registry import get_eval_adapter

# Raw {cond}_{suffix} key -> the display field name it's aggregated under. Distinct from
# score_fields (which come from the task's judge and vary per task) -- these are METHOD-level
# metadata generate.py can attach to any condition, for any task, so they're fixed here rather
# than sourced from the eval adapter. "_tokens" existed before this; "_participation_ratio" is new
# (see steering/psr/conceptor/rank_diagnostic.py) and, unlike every other aggregated field, is a
# per-checkpoint CONSTANT rather than a real per-row average -- averaging N identical values still
# gives the right number, so it reuses the same aggregation path rather than needing a special case.
_METADATA_SUFFIXES = {"tokens": "avg_tokens", "participation_ratio": "participation_ratio"}


def parse_condition_and_field(key: str, score_fields: list[str]) -> tuple[str, str] | None:
    """key looks like "{cond}_{field}" where BOTH cond and field may contain underscores
    ("prompt_psr_proper_correct") -- try each known field name as a suffix, longest first so
    "n_total" doesn't accidentally match inside a field that happens to end similarly."""
    for field in sorted(score_fields, key=len, reverse=True):
        suffix = f"_{field}"
        if key.endswith(suffix):
            return key[: -len(suffix)], field
    return None


def collect_raw_values_by_cond_field(rows: list[dict], score_fields: list[str]) -> dict[str, dict[str, list]]:
    """{cond: {field: [raw values across rows]}} -- the shared per-row collection step every
    aggregation in this project needs (means here, bootstrap CIs in bootstrap_analysis.py, raw
    scatter/bar data in plotting.py). Factored out so those three don't each re-derive the same
    "{cond}_{field}" / "{cond}_{metadata_suffix}" parsing independently and risk drifting apart."""
    values_by_cond = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for key, value in row.items():
            if key == "id":
                continue
            parsed = parse_condition_and_field(key, score_fields)
            if parsed:
                cond, field = parsed
                values_by_cond[cond][field].append(value)
                continue
            for suffix, display_field in sorted(_METADATA_SUFFIXES.items(), key=lambda kv: len(kv[0]), reverse=True):
                if key.endswith(f"_{suffix}"):
                    cond = key[: -len(f"_{suffix}")]
                    values_by_cond[cond][display_field].append(value)
                    break
    return values_by_cond


def summarize(rows: list[dict], score_fields: list[str]) -> dict[str, dict]:
    values_by_cond = collect_raw_values_by_cond_field(rows, score_fields)
    summary = {}
    for cond, fields in values_by_cond.items():
        summary[cond] = {
            field: sum(vals) / len(vals) for field, vals in fields.items()
        }
    return summary


def main(task: str, split: str) -> None:
    adapter = get_adapter(task)
    eval_adapter = get_eval_adapter(task)

    with (adapter.RESULTS_DIR / f"judged_{split}.jsonl").open() as f:
        rows = [json.loads(line) for line in f]

    summary = summarize(rows, eval_adapter.SCORE_FIELDS)
    display_fields = ["avg_tokens", *eval_adapter.SCORE_FIELDS, "participation_ratio"]

    print(f"{'condition':<28}" + "".join(f"{field:>20}" for field in display_fields))
    for cond in sorted(summary):
        s = summary[cond]
        print(f"{cond:<28}" + "".join(f"{s.get(field, float('nan')):>20.3f}" for field in display_fields))

    out_path = adapter.RESULTS_DIR / f"summary_{split}.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["caveman", "ifeval"])
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    main(args.task, args.split)
