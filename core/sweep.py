"""Generic (layer x hyperparameter) grid-sweep driver. Lives in core/, not steering/ or src/,
because it knows nothing about a specific TASK (it never touches an adapter) or a specific METHOD
(it never touches a gate, a conceptor, or a direction) -- it only knows how to iterate a list of
grid-point dicts, call an opaque `train_fn(point) -> dict` for each one, and handle the two pieces
of bookkeeping every sweep in this project has needed so far:

1. Resumability at the grid-point level (same discipline as src/generate.py's already_done_ids and
   src/const/sweep.py's `done` set -- a sweep can be long, and a crash halfway through shouldn't
   mean starting over).
2. Tracking and returning the single best point by some scalar metric, then leaving it to the
   CALLER to decide what "best" gets saved as a checkpoint -- this file never calls torch.save,
   since checkpoint FORMAT is exactly the thing that differs per method (see
   steering/psr/training_loop.py's train_gate docstring making the same point about checkpointing).

Before this file existed, this exact loop-plus-best-tracking shape was written inline, once, inside
src/psr/conceptor/selfproj/train.py (which already swept alpha, just not layer) -- this factors
that shape out so proper/conceptor/matrix/selfproj's NEW layer(+hyperparameter) sweeps
(src/psr/<variant>/train.py's `sweep()` functions) all reuse it instead of re-deriving it four
times with four subtly different resumability bugs waiting to happen.
"""
import json
from pathlib import Path
from typing import Callable


def _point_key(point: dict, key_fields: list[str]) -> tuple:
    """A grid point's identity for resumability purposes -- only key_fields (e.g. ["layer", "alpha",
    "nll_weight"]) participate, not every field train_fn might have added to its result, so a
    result dict with extra diagnostic fields (participation_ratio, etc.) doesn't break matching."""
    return tuple(point[f] for f in key_fields)


def run_grid_sweep(
    grid_points: list[dict],
    train_fn: Callable[[dict], dict],
    out_path: Path,
    key_fields: list[str],
    select_best_by: str = "final_mse",
    minimize: bool = True,
) -> tuple[list[dict], dict | None]:
    """For each point in grid_points (a dict of hyperparameter values, e.g. {"layer": 14, "alpha": 4.0,
    "nll_weight": 0.0}), calls train_fn(point) and expects back a dict of results merged with the
    point's own fields (train_fn is responsible for including select_best_by in what it returns;
    this function does the merging: `{**point, **train_fn(point)}`).

    Writes every result (in point-completion order) to out_path as JSONL, incrementally, after each
    point -- so a crash mid-sweep loses at most the one in-flight point, not the whole run. Resumes
    automatically: any point already present in out_path (matched on key_fields) is skipped and its
    already-written result is kept instead of being recomputed.

    Returns (all_results, best_result_or_None). best_result is the single row (from ALL rows, not
    just this session's new ones) with the extreme select_best_by value -- minimize=True picks the
    smallest (e.g. final_mse), minimize=False the largest (e.g. a correctness/accuracy metric).
    Rows a train_fn explicitly marked {"skipped": True} (see selfproj's existing "delta_scale ~0"
    skip case) are excluded from best-tracking but still written, matching selfproj/train.py's
    original behavior of logging skipped points without letting them win "best"."""
    results = []
    done = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                results.append(row)
                done.add(_point_key(row, key_fields))
        print(f">>> resuming sweep: {len(done)} grid point(s) already in {out_path}")

    for point in grid_points:
        point_id = _point_key(point, key_fields)
        if point_id in done:
            continue
        result = {**point, **train_fn(point)}
        results.append(result)
        with out_path.open("a") as f:
            f.write(json.dumps(result) + "\n")

    candidates = [r for r in results if not r.get("skipped", False) and select_best_by in r]
    if not candidates:
        return results, None
    best = min(candidates, key=lambda r: r[select_best_by]) if minimize else max(candidates, key=lambda r: r[select_best_by])
    return results, best
