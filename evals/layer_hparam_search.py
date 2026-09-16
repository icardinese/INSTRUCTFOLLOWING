"""Tiered (successive-halving, Li et al. 2018) layer + hyperparameter search: REAL judged
evaluation decides which layers/configs survive, not training MSE -- MSE only ever picks WHICH
hyperparameter combo represents a layer before it gets judged, it never eliminates a layer outright.
Follows this project's registry conventions (--task, get_adapter, get_eval_adapter) same as every
other evals/ script, and reuses evals.bootstrap_analysis.bootstrap_ci for CIs rather than
re-deriving statistics a third time.

Uses the DEV split for every search tier, test split ONLY for the final confirmation run -- search
decisions must not be made on the same data the reported numbers come from, or the eventual winner
is optimistically biased by however much the search itself overfit to that split.

Three tiers, same shape, different candidate sets and sample sizes -- see evaluate_candidates():
  Tier 1 (layer sweep):  one MSE-best hyperparameter combo per layer, small n. Real judged
                         evidence decides which layers survive -- MSE never eliminates a layer.
  Tier 2 (hyperparameter sweep, layer(s) fixed to Tier 1's survivors): the full alpha x
                         loss-config grid at just those layers, small n.
  Final: the single overall winner, full sample size, for the number that actually gets reported.

Every tier's candidates -- however many layers or hyperparameter combos -- are retrained (cheap:
pooled activations are already cached from the original sweep) and then evaluated in AS FEW
generate() calls as fit in one batch, via steering.batch_routing.generate_with_routed_configs --
see BATCHED_STEERING.md for why batching heterogeneous configs together is exact, not approximate.
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from adapters.registry import get_adapter
from core.generation_cache import load_or_compute_responses
from core.model_common import generate_response, load_model, num_layers
from evals.bootstrap_analysis import bootstrap_ci
from evals.registry import get_eval_adapter
from steering.batch_routing import generate_with_routed_configs
from steering.psr.gate import GateState

DEFAULT_TIER1_N = 20
DEFAULT_TIER2_N = 20
DEFAULT_FINAL_N = 180
DEFAULT_TOP_K_LAYERS = 3


@dataclass
class SearchContext:
    """Everything every retrain function needs, bundled once instead of threaded through 8
    positional args at every call site."""
    model: object
    tokenizer: object
    n_layers: int
    hidden_size: int
    cache_dir: Path
    train_items: list
    dev_items: list
    train_responses: dict
    dev_responses: dict
    seed: int = 42


def _retrain_proper(row: dict, ctx: SearchContext):
    from src.psr.proper.train import train_one_config
    from steering.psr.gate import make_inference_hook

    result = train_one_config(
        ctx.model, ctx.tokenizer, row["layer"], ctx.seed, ctx.n_layers, ctx.hidden_size, "cuda",
        ctx.train_items, ctx.dev_items, ctx.train_responses, ctx.dev_responses,
        mse_weight=row["mse_weight"], nll_weight=row["nll_weight"],
    )
    gate = GateState(result["weight"].to("cuda"), result["bias"].to("cuda"), result["coeff_bias"].to("cuda"))
    return make_inference_hook(gate, result["direction"].to("cuda"))


def _retrain_conceptor(row: dict, ctx: SearchContext):
    from src.psr.conceptor.train import train_one_config
    from steering.psr.gate import make_inference_hook

    result = train_one_config(
        ctx.model, ctx.tokenizer, row["layer"], row["alpha"], ctx.seed, ctx.n_layers, "cuda",
        ctx.train_items, ctx.dev_items, ctx.train_responses, ctx.dev_responses, ctx.cache_dir,
        mse_weight=row["mse_weight"], nll_weight=row["nll_weight"],
    )
    gate = GateState(result["weight"].to("cuda"), result["bias"].to("cuda"), result["coeff_bias"].to("cuda"))
    return make_inference_hook(gate, result["direction"].to("cuda"))


def _retrain_conceptor_matrix(row: dict, ctx: SearchContext):
    from src.psr.conceptor.matrix.train import train_one_config
    from steering.psr.conceptor.matrix.logic import make_inference_hook

    result = train_one_config(
        ctx.model, ctx.tokenizer, row["layer"], row["alpha"], ctx.seed, ctx.n_layers, "cuda",
        ctx.train_items, ctx.dev_items, ctx.train_responses, ctx.dev_responses, ctx.cache_dir,
        mse_weight=row["mse_weight"], nll_weight=row["nll_weight"],
    )
    gate = GateState(result["weight"].to("cuda"), result["bias"].to("cuda"), result["coeff_bias"].to("cuda"))
    return make_inference_hook(gate, result["conceptor"].to("cuda"), result["mu_instr"].to("cuda"), result["delta_scale"].to("cuda"))


def _retrain_conceptor_selfproj(row: dict, ctx: SearchContext):
    from src.psr.conceptor.selfproj.train import train_one_config
    from steering.psr.conceptor.selfproj.logic import make_inference_hook

    result = train_one_config(
        ctx.model, ctx.tokenizer, row["layer"], row["alpha"], ctx.seed, ctx.n_layers, "cuda",
        ctx.train_items, ctx.dev_items, ctx.train_responses, ctx.dev_responses, ctx.cache_dir,
        mse_weight=row["mse_weight"], nll_weight=row["nll_weight"],
    )
    if result.get("skipped"):
        return None  # delta_scale collapsed at this alpha -- same skip condition the sweep used
    gate = GateState(result["weight"].to("cuda"), result["bias"].to("cuda"), result["coeff_bias"].to("cuda"))
    return make_inference_hook(gate, result["conceptor"].to("cuda"), result["delta_scale_tensor"].to("cuda"))


# One retrain function per variant -- genuinely different math per variant (fixed-vector vs.
# matrix vs. self-projection), so this mirrors src/generate.py's own per-variant loader functions
# (load_gate_condition vs. load_matrix_condition vs. load_selfproj_condition) rather than forcing
# a false uniformity across methods that don't actually share a signature.
RETRAIN_FNS = {
    "proper": _retrain_proper,
    "conceptor": _retrain_conceptor,
    "conceptor_matrix": _retrain_conceptor_matrix,
    "conceptor_selfproj": _retrain_conceptor_selfproj,
}


def load_sweep_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def best_row_per_layer(rows: list[dict]) -> dict[int, dict]:
    """Tier 1's candidate set: the single MSE-best (non-skipped) row at each layer. This is the
    ONLY place MSE gets to make a decision -- picking among near-equivalent hyperparameter combos
    AT a layer, never deciding which layer survives (that's Tier 1's judged evaluation's job)."""
    best: dict[int, dict] = {}
    for row in rows:
        if row.get("skipped", False) or "final_mse" not in row:
            continue
        layer = row["layer"]
        if layer not in best or row["final_mse"] < best[layer]["final_mse"]:
            best[layer] = row
    return best


def evaluate_candidates(
    variant: str,
    candidates: list[dict],
    ctx: SearchContext,
    adapter,
    eval_adapter,
    n_examples: int,
    split: str = "dev",
) -> list[dict]:
    """Retrains every candidate, batches ALL of them into as few generate() calls as one batch can
    hold (grouped by layer internally by generate_with_routed_configs), judges n_examples real
    responses per candidate, and returns each candidate augmented with judged_score/ci_lo/ci_hi.

    candidates: each a dict with at least {"layer", "mse_weight", "nll_weight"} and whatever
    else that variant's retrain function needs (e.g. "alpha"). Usually rows pulled straight from
    a sweep JSONL (best_row_per_layer's output) or a hand-built hyperparameter grid for Tier 2.
    """
    retrain_fn = RETRAIN_FNS[variant]
    rows = adapter.load_rows(split)[:n_examples]
    items = adapter.to_items(ctx.tokenizer, rows)
    prompts = [item["base_prompt"] for item in items]

    hook_fns_by_group, layer_by_group, prompts_by_group = {}, {}, {}
    skipped_candidates = []
    for i, candidate in enumerate(candidates):
        group_id = i
        hook_fn = retrain_fn(candidate, ctx)
        if hook_fn is None:
            skipped_candidates.append(candidate)
            continue
        hook_fns_by_group[group_id] = hook_fn
        layer_by_group[group_id] = candidate["layer"]
        prompts_by_group[group_id] = prompts

    responses_by_group = generate_with_routed_configs(
        ctx.model, ctx.tokenizer, prompts_by_group, hook_fns_by_group, layer_by_group,
    )

    primary_field = eval_adapter.SCORE_FIELDS[0]
    results = []
    for i, candidate in enumerate(candidates):
        if i not in responses_by_group:
            results.append({**candidate, "skipped": True})
            continue
        scores = [eval_adapter.score_response(row, resp)[primary_field] for row, resp in zip(rows, responses_by_group[i])]
        point, lo, hi = bootstrap_ci(scores)
        results.append({**candidate, "skipped": False, "judged_score": point, "ci_lo": lo, "ci_hi": hi, "n": len(scores)})
    return results


def load_existing_tiered_results(out_path: Path) -> dict[str, list[dict]]:
    """{"tier1": [...], "tier2": [...], "final": [...]} from a previous (possibly interrupted) run
    of this exact variant, or {} if out_path doesn't exist yet. Each row has its "tier" key
    stripped back off -- the on-disk format tags every row with its tier, but callers that
    reconstruct e.g. tier1_results from this want the same shape evaluate_candidates() returns."""
    if not out_path.exists():
        return {}
    by_tier: dict[str, list[dict]] = {}
    with out_path.open() as f:
        for line in f:
            row = json.loads(line)
            tier = row.pop("tier")
            by_tier.setdefault(tier, []).append(row)
    return by_tier


def run_tiered_search(
    task: str,
    variant: str,
    tier1_n: int = DEFAULT_TIER1_N,
    tier2_n: int = DEFAULT_TIER2_N,
    final_n: int = DEFAULT_FINAL_N,
    top_k_layers: int = DEFAULT_TOP_K_LAYERS,
    seed: int = 42,
) -> None:
    adapter = get_adapter(task)
    eval_adapter = get_eval_adapter(task)
    sweep_path = adapter.RESULTS_DIR / f"psr_{variant}_sweep.jsonl"
    if not sweep_path.exists():
        raise FileNotFoundError(f"{sweep_path} not found -- run that variant's --sweep first")

    out_path = adapter.RESULTS_DIR / f"{variant}_tiered_search.jsonl"
    existing = load_existing_tiered_results(out_path)
    if "final" in existing:
        print(f"{out_path} already has a Final result for '{variant}' -- fully done, skipping. "
              f"Delete {out_path} (or just its 'final' row) to force a rerun.")
        return

    device = "cuda"
    model, tokenizer = load_model(device, model_name=adapter.MODEL_NAME)
    for p in model.parameters():
        p.requires_grad_(False)

    train_items = adapter.to_items(tokenizer, adapter.load_rows("train"))
    dev_items = adapter.to_items(tokenizer, adapter.load_rows("dev"))
    train_responses = load_or_compute_responses(model, tokenizer, train_items, adapter.CACHE_DIR / "teacher_responses_train.json", generate_response)
    dev_responses = load_or_compute_responses(model, tokenizer, dev_items, adapter.CACHE_DIR / "teacher_responses_dev.json", generate_response)
    ctx = SearchContext(
        model=model, tokenizer=tokenizer, n_layers=num_layers(model), hidden_size=model.config.hidden_size,
        cache_dir=adapter.CACHE_DIR, train_items=train_items, dev_items=dev_items,
        train_responses=train_responses, dev_responses=dev_responses, seed=seed,
    )

    all_results = [{"tier": tier, **r} for tier, rows in existing.items() for r in rows]

    def write(tier: str, rows: list[dict]) -> None:
        for r in rows:
            all_results.append({"tier": tier, **r})
        with out_path.open("w") as f:
            for r in all_results:
                f.write(json.dumps(r) + "\n")

    sweep_rows = load_sweep_rows(sweep_path)

    if "tier1" in existing:
        print(f"Tier 1 already complete ({len(existing['tier1'])} rows) -- reusing from {out_path}")
        tier1_results = existing["tier1"]
    else:
        print(f"== Tier 1: layer sweep, n={tier1_n}, real judged evaluation decides survivors ==")
        tier1_candidates = list(best_row_per_layer(sweep_rows).values())
        print(f"{len(tier1_candidates)} layers to evaluate (MSE-best hyperparameter combo per layer)")
        tier1_results = evaluate_candidates(variant, tier1_candidates, ctx, adapter, eval_adapter, tier1_n, split="dev")
        write("tier1", tier1_results)

    survivors = sorted((r for r in tier1_results if not r["skipped"]), key=lambda r: -r["judged_score"])[:top_k_layers]
    if not survivors:
        print("WARNING: every Tier 1 candidate was skipped -- nothing to search further")
        return
    print(f"Tier 1 survivors (top {top_k_layers} by judged score): {[s['layer'] for s in survivors]}")

    if "tier2" in existing:
        print(f"Tier 2 already complete ({len(existing['tier2'])} rows) -- reusing from {out_path}")
        tier2_results = existing["tier2"]
    else:
        print(f"\n== Tier 2: hyperparameter sweep at surviving layer(s), n={tier2_n} ==")
        survivor_layers = {s["layer"] for s in survivors}
        tier2_candidates = [r for r in sweep_rows if r.get("layer") in survivor_layers and not r.get("skipped", False) and "final_mse" in r]
        print(f"{len(tier2_candidates)} (layer, hyperparameter) combos to evaluate at the surviving layer(s)")
        tier2_results = evaluate_candidates(variant, tier2_candidates, ctx, adapter, eval_adapter, tier2_n, split="dev")
        write("tier2", tier2_results)

    winner = max((r for r in tier2_results if not r["skipped"]), key=lambda r: r["judged_score"], default=None)
    if winner is None:
        print("WARNING: every Tier 2 candidate was skipped -- no final confirmation run")
        return
    print(f"Tier 2 winner: layer={winner['layer']}, {({k: v for k, v in winner.items() if k not in ('tier',)})}")

    print(f"\n== Final: full-size confirmation run, n={final_n}, TEST split ==")
    final_results = evaluate_candidates(variant, [winner], ctx, adapter, eval_adapter, final_n, split="test")
    write("final", final_results)
    print(f"Final judged score: {final_results[0]['judged_score']:.4f} "
          f"[{final_results[0]['ci_lo']:.4f}, {final_results[0]['ci_hi']:.4f}] (95% CI, n={final_n})")
    print(f"\nWrote full tiered search log to {out_path}")


def summarize_all_variants(task: str) -> dict:
    """Reads every {variant}_tiered_search.jsonl that exists yet for this task, pulls out each
    one's Final row (the real, full-n, test-split, judged-evidence winner), prints a comparison
    table, and writes results/<task>/tiered_search_summary.json. Safe to call any time, including
    mid-overnight-run -- a variant that hasn't reached Final yet is just reported as "not done"
    rather than causing an error, so this doubles as a progress check, not only a final report."""
    adapter = get_adapter(task)
    summary = {}
    for path in sorted(adapter.RESULTS_DIR.glob("*_tiered_search.jsonl")):
        variant = path.stem.removesuffix("_tiered_search")
        existing = load_existing_tiered_results(path)
        if "final" not in existing:
            summary[variant] = {"status": f"not yet at Final (tiers done: {sorted(existing.keys())})"}
            continue
        row = existing["final"][0]
        summary[variant] = {
            "status": "done", "layer": row["layer"], "judged_score": row["judged_score"],
            "ci_lo": row["ci_lo"], "ci_hi": row["ci_hi"], "n": row["n"],
        }

    print(f"{'variant':<22}{'status':<45}{'layer':>8}{'judged_score':>15}{'95% CI':>20}")
    for variant, s in summary.items():
        if s["status"] != "done":
            print(f"{variant:<22}{s['status']:<45}")
            continue
        ci = f"[{s['ci_lo']:.3f}, {s['ci_hi']:.3f}]"
        print(f"{variant:<22}{'done':<45}{s['layer']:>8}{s['judged_score']:>15.4f}{ci:>20}")

    out_path = adapter.RESULTS_DIR / "tiered_search_summary.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {out_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["caveman", "ifeval"])
    parser.add_argument("--variant", choices=list(RETRAIN_FNS.keys()), help="omit with --summarize to just aggregate every variant's results")
    parser.add_argument("--summarize", action="store_true", help="skip searching -- just read whatever *_tiered_search.jsonl files exist and print/write a comparison")
    parser.add_argument("--tier1-n", type=int, default=DEFAULT_TIER1_N)
    parser.add_argument("--tier2-n", type=int, default=DEFAULT_TIER2_N)
    parser.add_argument("--final-n", type=int, default=DEFAULT_FINAL_N)
    parser.add_argument("--top-k-layers", type=int, default=DEFAULT_TOP_K_LAYERS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.summarize:
        summarize_all_variants(args.task)
    else:
        if not args.variant:
            parser.error("--variant is required unless --summarize is given")
        run_tiered_search(args.task, args.variant, args.tier1_n, args.tier2_n, args.final_n, args.top_k_layers, args.seed)
