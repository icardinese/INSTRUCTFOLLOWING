"""Reads a generations file, scores every {cond}_response column via the task's eval adapter,
writes one judged row per id with {cond}_{score_key} columns -- same shape convention every prior
judge script in this project has used. Adding a new task never touches this file.
"""
import argparse
import json

from adapters.registry import get_adapter
from evals.registry import get_eval_adapter


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
        conditions = sorted(k[: -len("_response")] for k in gen_row if k.endswith("_response"))
        for cond in conditions:
            scores = eval_adapter.score_response(row, gen_row[f"{cond}_response"])
            for score_key, value in scores.items():
                out_row[f"{cond}_{score_key}"] = value
            if f"{cond}_tokens" in gen_row:
                out_row[f"{cond}_tokens"] = gen_row[f"{cond}_tokens"]
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
    parser.add_argument("--task", required=True, choices=["caveman", "ifeval"])
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    main(args.task, args.split)
