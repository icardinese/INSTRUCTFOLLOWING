"""Summarizes judged_{split}.jsonl: for every condition, the mean of every score field (correctness,
coherence, tokens, whatever the task's judge returns) plus average token count. Task-agnostic --
never touched when adding a new task, same as run_judge.py.
"""
import argparse
import json
from collections import defaultdict

from adapters.registry import get_adapter
from evals.registry import get_eval_adapter


def parse_condition_and_field(key: str, score_fields: list[str]) -> tuple[str, str] | None:
    """key looks like "{cond}_{field}" where BOTH cond and field may contain underscores
    ("prompt_psr_proper_correct") -- try each known field name as a suffix, longest first so
    "n_total" doesn't accidentally match inside a field that happens to end similarly."""
    for field in sorted(score_fields, key=len, reverse=True):
        suffix = f"_{field}"
        if key.endswith(suffix):
            return key[: -len(suffix)], field
    return None


def summarize(rows: list[dict], score_fields: list[str]) -> dict[str, dict]:
    values_by_cond = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for key, value in row.items():
            if key == "id":
                continue
            parsed = parse_condition_and_field(key, score_fields)
            if parsed:
                cond, field = parsed
                values_by_cond[cond][field].append(value)
            elif key.endswith("_tokens"):
                cond = key[: -len("_tokens")]
                values_by_cond[cond]["avg_tokens"].append(value)

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

    print(f"{'condition':<28}" + "".join(f"{field:>16}" for field in ["avg_tokens", *eval_adapter.SCORE_FIELDS]))
    for cond in sorted(summary):
        s = summary[cond]
        print(f"{cond:<28}" + "".join(f"{s.get(field, float('nan')):>16.3f}" for field in ["avg_tokens", *eval_adapter.SCORE_FIELDS]))

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
