"""Systematic cosine similarity across every direction vector this project has, computed from real
checkpoints/cached pooled activations on disk -- nothing here is copied from a conversation or
reconstructed by hand. Run this on the actual training machine (needs real torch + the model, for
const/stolfo's on-the-fly direction recompute -- cheap, since pool_prompt_last_token's cache from
the tiered searches means no new forward passes are needed for those two).

Sources, and exactly how each is obtained:
  - psr_proper / psr_conceptor: {variant}_probe_tiered_winner.pt if it exists (the REAL corrected
    tiered-search winner's checkpoint), else falls back to {variant}_probe.pt with a warning (the
    sweep()-selected checkpoint, NOT necessarily the tiered search's actual winner -- see this
    project's own history for why that distinction matters).
  - s_psr / a_psr: psr_probe.pt / a_psr_probe.pt's own "direction" field directly.
  - const / stolfo: recomputed from {variant}_tiered_search.jsonl's real Final winner's layer,
    via the SAME compute_diff_mean_direction / compute_target_projection functions the search
    itself uses (steering/const/direction.py, steering/stolfo/direction.py) -- not reimplemented
    here, imported directly, so this can't silently drift from what the search actually did.

Writes results/<task>/cosine_similarity_matrix.{csv,json}.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.registry import get_adapter
from evals.layer_hparam_search import build_search_context, load_existing_tiered_results
from steering.const.direction import compute_diff_mean_direction
from steering.psr.data import load_or_pool_prompt_last_token
from steering.stolfo.direction import compute_target_projection


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    return (a @ b / (a.norm() * b.norm())).item()


def load_probe_direction(adapter, name: str, prefer_tiered_winner: bool) -> tuple[torch.Tensor, str]:
    """Returns (direction, source_note). Checks {name}_tiered_winner.pt first if prefer_tiered_winner,
    else just {name}.pt -- see module docstring."""
    candidates = [f"{name}_tiered_winner.pt", f"{name}.pt"] if prefer_tiered_winner else [f"{name}.pt"]
    for fname in candidates:
        path = adapter.RESULTS_DIR / fname
        if path.exists():
            ckpt = torch.load(path, map_location="cpu")
            note = fname
            if fname == f"{name}.pt" and prefer_tiered_winner:
                note += "  ** WARNING: no _tiered_winner.pt found -- this is the sweep()-selected " \
                        "checkpoint, may NOT be the real tiered-search Final winner **"
            return ckpt["direction"], note
    raise FileNotFoundError(f"neither {' nor '.join(candidates)} found in {adapter.RESULTS_DIR}")


def recompute_const_or_stolfo_direction(adapter, ctx, variant: str) -> tuple[torch.Tensor, str, dict]:
    tiered_path = adapter.RESULTS_DIR / f"{variant}_tiered_search.jsonl"
    existing = load_existing_tiered_results(tiered_path)
    if "final" not in existing:
        raise FileNotFoundError(f"{tiered_path} has no Final row -- run its tiered search first.")
    winner = existing["final"][0]
    layer = winner["layer"]
    base_pool, instr_pool = load_or_pool_prompt_last_token(ctx.model, ctx.tokenizer, ctx.train_items, layer, ctx.cache_dir)
    direction = compute_diff_mean_direction(base_pool, instr_pool)
    extra = {"layer": layer}
    if variant == "const":
        extra["coeff"] = winner.get("coeff")
    else:
        extra["target_projection"] = compute_target_projection(instr_pool, direction)
    return direction, f"{variant}_tiered_search.jsonl Final (layer={layer})", extra


def main(task: str) -> None:
    adapter = get_adapter(task)
    vectors: dict[str, torch.Tensor] = {}
    sources: dict[str, str] = {}
    extras: dict[str, dict] = {}

    for name in ("psr_proper_probe", "psr_conceptor_probe"):
        try:
            v, src = load_probe_direction(adapter, name, prefer_tiered_winner=True)
            vectors[name], sources[name] = v, src
        except FileNotFoundError as e:
            print(f"SKIP {name}: {e}")

    for name in ("psr_probe", "a_psr_probe"):
        try:
            v, src = load_probe_direction(adapter, name, prefer_tiered_winner=False)
            vectors[name], sources[name] = v, src
        except FileNotFoundError as e:
            print(f"SKIP {name}: {e}")

    need_model = any((adapter.RESULTS_DIR / f"{v}_tiered_search.jsonl").exists() for v in ("const", "stolfo"))
    if need_model:
        print("loading model for const/stolfo's on-the-fly direction recompute (cheap -- pooled activations already cached) ...")
        ctx = build_search_context(adapter, seed=42)
        for variant in ("const", "stolfo"):
            try:
                v, src, extra = recompute_const_or_stolfo_direction(adapter, ctx, variant)
                vectors[variant], sources[variant], extras[variant] = v, src, extra
            except FileNotFoundError as e:
                print(f"SKIP {variant}: {e}")

    if len(vectors) < 2:
        print("Fewer than 2 vectors available -- nothing to compare. Run the relevant searches first.")
        return

    names = list(vectors.keys())
    n = len(names)
    matrix = [[cos(vectors[a], vectors[b]) for b in names] for a in names]
    d = next(iter(vectors.values())).shape[0]

    print(f"\n{n} vectors loaded (d={d}):")
    for name in names:
        print(f"  {name:<22} source={sources[name]}  ||v||={vectors[name].double().norm().item():.4f}")

    print(f"\n{'':<22}" + "".join(f"{n2:>16}" for n2 in names))
    for i, a in enumerate(names):
        print(f"{a:<22}" + "".join(f"{matrix[i][j]:>16.4f}" for j in range(n)))

    out_csv = adapter.RESULTS_DIR / "cosine_similarity_matrix.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + names)
        for i, a in enumerate(names):
            w.writerow([a] + [f"{matrix[i][j]:.4f}" for j in range(n)])

    out_json = adapter.RESULTS_DIR / "cosine_similarity_matrix.json"
    with out_json.open("w") as f:
        json.dump({
            "task": task, "dim": d,
            "note": f"random-vector noise floor for d={d}: mean|cos|~{1/(d**0.5)*0.8:.4f}ish, "
                     f"treat anything near that as no real alignment",
            "sources": sources, "extras": extras,
            "norms": {k: v.double().norm().item() for k, v in vectors.items()},
            "names": names, "matrix": matrix,
        }, f, indent=2)
    print(f"\nwrote {out_csv}\nwrote {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="caveman", choices=["caveman", "ifeval"])
    args = parser.parse_args()
    main(args.task)
