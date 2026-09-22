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
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.registry import get_adapter
from evals.layer_hparam_search import build_search_context, load_existing_tiered_results
from steering.const.direction import compute_diff_mean_direction
from steering.psr.data import load_or_pool_prompt_last_token
from steering.stolfo.direction import compute_target_projection
from adapters.registry import TASK_CHOICES


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    return (a @ b / (a.norm() * b.norm())).item()


def load_probe_direction(adapter, name: str, prefer_tiered_winner: bool) -> tuple[torch.Tensor, str]:
    """Returns (direction, source_note). Checks {name}_tiered_winner.pt first if prefer_tiered_winner,
    else just {name}.pt -- see module docstring.

    Handles the SINGLE-LAYER checkpoint shape only: a top-level "direction" (d,) plus "layer".
    Multi-layer checkpoints store "directions" (a {layer: (d,)} dict) and "layer_indices" instead,
    with NO top-level "direction" key -- reading those unconditionally as ckpt["direction"] is what
    raised KeyError on a_psr_probe.pt on 2026-09-18. Those go through
    load_multi_layer_directions below; this function raises a clear error rather than guessing."""
    candidates = [f"{name}_tiered_winner.pt", f"{name}.pt"] if prefer_tiered_winner else [f"{name}.pt"]
    for fname in candidates:
        path = adapter.RESULTS_DIR / fname
        if path.exists():
            ckpt = torch.load(path, map_location="cpu")
            if "direction" not in ckpt:
                raise KeyError(
                    f"{fname} has no top-level 'direction' (keys: {sorted(ckpt)}). If it has "
                    f"'directions'/'layer_indices' it is a multi-layer checkpoint -- add it to "
                    f"MULTI_LAYER_CHECKPOINTS instead of SINGLE_LAYER_CHECKPOINTS."
                )
            note = fname
            if fname == f"{name}.pt" and prefer_tiered_winner:
                note += "  ** WARNING: no _tiered_winner.pt found -- this is the sweep()-selected " \
                        "checkpoint, may NOT be the real tiered-search Final winner **"
            return ckpt["direction"], note
    raise FileNotFoundError(f"neither {' nor '.join(candidates)} found in {adapter.RESULTS_DIR}")


def load_multi_layer_directions(adapter, name: str) -> tuple[dict[int, torch.Tensor], str]:
    """Returns ({layer: direction}, source_note) for a multi-layer checkpoint -- the old A-PSR
    baseline (a_psr_probe.pt: 'directions' + 'layer_indices' + 'probes') and every all-layer
    checkpoint written by src/psr/all_layer/train.py ('directions' + 'gates' + 'layer_indices').
    Both shapes expose 'directions' as a {layer: tensor} dict, which is the only part this needs."""
    path = adapter.RESULTS_DIR / f"{name}.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    ckpt = torch.load(path, map_location="cpu")
    if "directions" not in ckpt:
        raise KeyError(f"{name}.pt has no 'directions' (keys: {sorted(ckpt)})")
    dirs = {int(l): v for l, v in ckpt["directions"].items()}
    src = f"{name}.pt ({len(dirs)} layers"
    if ckpt.get("direction_source"):
        src += f", direction_source={ckpt['direction_source']}"
    if ckpt.get("alpha") is not None:
        src += f", alpha={ckpt['alpha']}"
    return dirs, src + ")"


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


REFERENCE_LAYER = int(os.environ.get("COSINE_REFERENCE_LAYER", 14))  # where the single-layer
# methods (proper, conceptor, S-PSR) actually live, so multi-layer checkpoints are sliced here to
# make the cross-method matrix an apples-to-apples comparison at one depth.
MULTI_LAYER_CHECKPOINTS = [
    "a_psr_probe",                       # the OLD mislabeled A-PSR (N independently-trained probes)
    "psr_all_layer_probe_mse",           # faithful A-PSR, gradient-trained directions
    "psr_all_layer_probe_nll",
    "psr_multi_gate_probe_mse",          # Multi-Gate ablation, fixed diff-in-means
    "psr_multi_gate_probe_nll",
    "psr_multi_gate_conceptor_probe_mse",  # Multi-Gate with conceptor-projected directions
    "psr_multi_gate_conceptor_probe_nll",
]


def main(task: str) -> None:
    adapter = get_adapter(task)
    vectors: dict[str, torch.Tensor] = {}
    sources: dict[str, str] = {}
    extras: dict[str, dict] = {}
    per_layer: dict[str, dict[int, torch.Tensor]] = {}

    for name in ("psr_proper_probe", "psr_conceptor_probe"):
        try:
            v, src = load_probe_direction(adapter, name, prefer_tiered_winner=True)
            vectors[name], sources[name] = v, src
        except (FileNotFoundError, KeyError) as e:
            print(f"SKIP {name}: {e}")

    for name in ("psr_probe",):
        try:
            v, src = load_probe_direction(adapter, name, prefer_tiered_winner=False)
            vectors[name], sources[name] = v, src
        except (FileNotFoundError, KeyError) as e:
            print(f"SKIP {name}: {e}")

    # Multi-layer checkpoints: keep the full per-layer dict (for the depth profile below) AND
    # slice REFERENCE_LAYER into the cross-method matrix.
    for name in MULTI_LAYER_CHECKPOINTS:
        try:
            dirs, src = load_multi_layer_directions(adapter, name)
        except (FileNotFoundError, KeyError) as e:
            print(f"SKIP {name}: {e}")
            continue
        per_layer[name] = dirs
        if REFERENCE_LAYER in dirs:
            key = f"{name}@L{REFERENCE_LAYER}"
            vectors[key], sources[key] = dirs[REFERENCE_LAYER], src
        else:
            print(f"NOTE {name}: no layer {REFERENCE_LAYER} (has {sorted(dirs)[:6]}...) -- "
                  f"included in the depth profile but not the cross-method matrix")

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

    # --- Depth profile: for each pair of multi-layer checkpoints that share layers, cosine at
    # EVERY layer. This is the figure that speaks to the 2026-09-18 finding that Multi-Gate
    # (fixed diff-in-means) roughly matches faithful A-PSR (gradient-trained) -- if the trained
    # directions turn out to be highly aligned with DiM at most depths, that alignment IS the
    # explanation; if they're near-orthogonal yet perform the same, the direction genuinely
    # doesn't matter much once there's a trained gate at every layer. Either way it's a real
    # result, and it can't be read off the single-layer matrix above.
    depth_profile = {}
    names_ml = sorted(per_layer)
    for i, a in enumerate(names_ml):
        for b in names_ml[i + 1:]:
            shared = sorted(set(per_layer[a]) & set(per_layer[b]))
            if len(shared) < 2:
                continue
            depth_profile[f"{a} vs {b}"] = {
                "layers": shared,
                "cosine": [round(cos(per_layer[a][l], per_layer[b][l]), 4) for l in shared],
            }

    if depth_profile:
        print(f"\n=== per-layer cosine (multi-layer checkpoints, {len(depth_profile)} pairs) ===")
        for pair, prof in depth_profile.items():
            vals = prof["cosine"]
            mean_abs = sum(abs(v) for v in vals) / len(vals)
            print(f"\n  {pair}")
            print(f"    mean|cos|={mean_abs:.4f}  min={min(vals):+.4f}  max={max(vals):+.4f}")
            print(f"    by layer: " + " ".join(f"{l}:{v:+.3f}" for l, v in zip(prof["layers"], vals)))

        csv_depth = adapter.RESULTS_DIR / "cosine_similarity_by_layer.csv"
        with csv_depth.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pair", "layer", "cosine"])
            for pair, prof in depth_profile.items():
                for l, v in zip(prof["layers"], prof["cosine"]):
                    w.writerow([pair, l, f"{v:.4f}"])
        print(f"\nwrote {csv_depth}")

    with out_json.open("w") as f:
        json.dump({
            "task": task, "dim": d,
            "note": f"random-vector noise floor for d={d}: mean|cos|~{1/(d**0.5)*0.8:.4f}ish, "
                     f"treat anything near that as no real alignment",
            "sources": sources, "extras": extras,
            "reference_layer": REFERENCE_LAYER, "depth_profile": depth_profile,
            "norms": {k: v.double().norm().item() for k, v in vectors.items()},
            "names": names, "matrix": matrix,
        }, f, indent=2)
    print(f"\nwrote {out_csv}\nwrote {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="caveman", choices=TASK_CHOICES)
    args = parser.parse_args()
    main(args.task)
