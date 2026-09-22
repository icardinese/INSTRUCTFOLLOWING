"""Reads a generations file, scores every {cond}_response column via the task's eval adapter,
writes one judged row per id with {cond}_{score_key} columns -- same shape convention every prior
judge script in this project has used. Adding a new task never touches this file.

Also carries forward every OTHER {cond}_* metadata field generate.py attached (currently
{cond}_tokens always, and {cond}_participation_ratio for matrix-based conditions) -- generically,
by key prefix, not a hardcoded field list. This means a future metadata field generate.py starts
attaching (to any condition, for any reason) automatically survives into judged_{split}.jsonl,
and therefore into summarize.py/bootstrap_analysis.py/plotting.py, with zero changes needed here.
"""
import argparse
import json

from adapters.registry import get_adapter
from evals.registry import get_eval_adapter
from adapters.registry import TASK_CHOICES


def main(task: str, split: str) -> None:
    adapter = get_adapter(task)
    eval_adapter = get_eval_adapter(task)

    gen_path = adapter.RESULTS_DIR / f"generations_{split}.jsonl"
    rows_by_id = {row["id"]: row for row in adapter.load_rows(split)}

    with gen_path.open() as f:
        gen_rows = [json.loads(line) for line in f]

    out_rows = []
    for i, gen_row in enumerate(gen_rows):
        row = rows_by_id.get(gen_row["id"])
        if row is None:
            print(f"WARNING: id {gen_row['id']} in generations but not in data, skipping")
            continue

        out_row = {"id": gen_row["id"]}
        conditions = sorted((k[: -len("_response")] for k in gen_row if k.endswith("_response")), key=len, reverse=True)
        claimed_keys = set()
        for cond in conditions:
            scores = eval_adapter.score_response(row, gen_row[f"{cond}_response"])
            for score_key, value in scores.items():
                out_row[f"{cond}_{score_key}"] = value
            # Longest-name-first + claimed_keys avoids a real ambiguity: condition names can be
            # prefixes of other condition names (e.g. "psr" vs "psr_proper" vs
            # "psr_conceptor_matrix"), so a naive `key.startswith(f"{cond}_")` for the SHORTER name
            # would also match the LONGER name's own fields (e.g. cond="psr" matching
            # "psr_proper_tokens"). Processing longest-first and marking each field "claimed" the
            # first time it's assigned means the longer, more specific condition name always wins
            # -- same trick evals/summarize.py's parse_condition_and_field already uses for the
            # analogous score-field-suffix ambiguity. Excluding every "_response" key outright
            # (not just this cond's own) matters for the same reason: "psr_proper_response" starts
            # with "psr_" too, and raw response text should never leak into judged output under
            # any condition's name.
            for key, value in gen_row.items():
                if key.startswith(f"{cond}_") and not key.endswith("_response") and key not in claimed_keys:
                    out_row[key] = value
                    claimed_keys.add(key)
        out_rows.append(out_row)

        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{len(gen_rows)} judged")

    out_path = adapter.RESULTS_DIR / f"judged_{split}.jsonl"
    with out_path.open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(out_rows)} rows to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=TASK_CHOICES)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    main(args.task, args.split)
