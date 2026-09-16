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
from typing import Any

import torch

from adapters.registry import get_adapter
from core.generation_cache import load_or_compute_responses
from core.model_common import generate_response, load_model, num_layers
from evals.bootstrap_analysis import bootstrap_ci, paired_bootstrap_diff
from evals.registry import get_eval_adapter
from steering.batch_routing import generate_with_routed_configs
from steering.psr.gate import GateState

DEFAULT_TIER1_N = 20
DEFAULT_TIER2_N = 20
DEFAULT_FINAL_N = 180
DEFAULT_TOP_K_LAYERS = 3
# Rows (candidates x prompts) allowed in one generate_with_routed_configs call. Batching
# heterogeneous configs together is mathematically exact (see BATCHED_STEERING.md) -- but nothing
# bounds how much GPU memory ONE call needs, and MLP intermediate activations scale with
# batch_size x seq_len. Confirmed by a REAL CUDA OOM on Qwen2.5-7B-Instruct at 300 rows (15
# hyperparameter combos x 20 prompts, one Tier 2 call) on an 80GB A100. This caps it by splitting
# into multiple sequential batched calls instead -- still batches as much as fits at once, just
# not an unbounded amount. 60 is a conservative starting point, not a measured ceiling; lower it
# if you still see OOMs, raise it if you want to verify more headroom is actually available.
DEFAULT_MAX_BATCH_ROWS = 60


def _chunk_groups_by_row_budget(prompts_by_group: dict[Any, list[str]], max_rows: int) -> list[list[Any]]:
    """Greedily packs group_ids into chunks whose total prompt-row count stays under max_rows.
    A single group bigger than max_rows on its own (e.g. Final's one config x 180 prompts) still
    gets its own chunk -- that's no worse than every pre-batching call already was, not a
    regression, just not further reducible."""
    chunks: list[list[Any]] = []
    current_chunk: list[Any] = []
    current_size = 0
    for group_id, prompts in prompts_by_group.items():
        size = len(prompts)
        if current_chunk and current_size + size > max_rows:
            chunks.append(current_chunk)
            current_chunk, current_size = [], 0
        current_chunk.append(group_id)
        current_size += size
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


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


_PROMPT_BASELINE_GROUP = "__prompt_baseline__"
# The ceiling value of eval_adapter.SCORE_FIELDS[0], needed only for fully_correct_rate (a
# REPORTED-only metric, never used in selection -- see select_constrained_survivors, which is
# untouched by this). Caveman's "correct" field is 0|1|2 (see evals/caveman/judge.py's RUBRIC);
# ifeval's "follow_all_instructions" is already binary (0|1). Neither judge module currently
# declares its own scale, so this is a parameter rather than something introspected automatically.
DEFAULT_PRIMARY_FIELD_MAX = 2


def _fully_correct_rate(scores: list[float], primary_field_max: float) -> float:
    """Fraction of examples scoring exactly at the ceiling -- the metric your own draft paper's
    Figure 1 actually plots ("Fully-Correct Rate"), distinct from the MEAN score that drives
    selection (mean_score treats a 1 as "half credit"; this treats anything short of the ceiling
    as equally "not fully correct," same as the paper's own y-axis does)."""
    return sum(1 for s in scores if s == primary_field_max) / len(scores)


def _avg_tokens(ctx: SearchContext, responses: list[str]) -> float:
    """Raw response length -- the blunt, always-available proxy for terseness, reported
    alongside the judged conciseness score (which is what actually drives selection) rather than
    instead of it. Kept separate deliberately: a response can be short by accident or short
    because the wording is genuinely tight, and conflating the two was the whole reason the
    judged conciseness field exists in the first place (see evals/caveman/judge.py)."""
    return sum(len(ctx.tokenizer(resp)["input_ids"]) for resp in responses) / len(responses)


def mark_pareto_frontier(results: list[dict]) -> list[dict]:
    """Attaches on_pareto_frontier: bool to every non-skipped result -- True if no OTHER result
    is both >= it on judged_score (correctness) AND >= it on optimize_score (conciseness), with
    at least one strictly greater. Purely informational: never read by select_constrained_survivors,
    never affects which candidate wins -- it's here so the full correctness/conciseness tradeoff
    shape for a tier survives in the log even when you'd have picked a different point off it than
    the automated selection did. Skipped candidates get on_pareto_frontier=None (not comparable)."""
    usable = [r for r in results if not r.get("skipped", False)]
    frontier_ids = set()
    for i, r in enumerate(usable):
        dominated = False
        for j, other in enumerate(usable):
            if i == j:
                continue
            if (other["judged_score"] >= r["judged_score"] and other["optimize_score"] >= r["optimize_score"]
                    and (other["judged_score"] > r["judged_score"] or other["optimize_score"] > r["optimize_score"])):
                dominated = True
                break
        if not dominated:
            frontier_ids.add(id(r))
    return [
        {**r, "on_pareto_frontier": (id(r) in frontier_ids) if not r.get("skipped", False) else None}
        for r in results
    ]


def evaluate_candidates(
    variant: str,
    candidates: list[dict],
    ctx: SearchContext,
    adapter,
    eval_adapter,
    n_examples: int,
    split: str = "dev",
    max_batch_rows: int = DEFAULT_MAX_BATCH_ROWS,
    optimize_field: str | None = None,
    primary_field_max: float = DEFAULT_PRIMARY_FIELD_MAX,
) -> list[dict]:
    """Retrains every candidate, then batches them into generate_with_routed_configs calls --
    as many candidates per call as fit within max_batch_rows total prompt-rows, chunked into
    multiple sequential calls if there are more than that (see DEFAULT_MAX_BATCH_ROWS's docstring
    for why this cap exists -- it's a fix for a real OOM, not a hypothetical precaution).

    ALSO generates a Prompt-alone (no steering, the real instructed prompt, not the base one)
    baseline in the SAME batched call, on the SAME rows -- not read from some other run's
    aggregate file, specifically so it's PAIRED to every candidate's scores here (paired
    comparisons need the same underlying examples; an old summary_test.json's mean was very
    likely computed on a different split/n and can't be validly paired against anything). This is
    what lets select_constrained_survivors enforce "not significantly worse than Prompt alone" --
    your own project's own central question (finding #5) -- not just "not worse than the best
    candidate in this grid," which says nothing about whether ANY of them actually beat prompting.

    Judges n_examples real responses per candidate on TWO axes that drive selection --
    `judged_score` (eval_adapter.SCORE_FIELDS[0], e.g. "correct" -- the GATE) and `optimize_score`
    (the task's own "conciseness" judged field if its judge provides one, e.g. caveman, else
    negative average response token count) -- PLUS two more that are computed and logged but
    never fed into any selection decision: `fully_correct_rate` (fraction scoring at the ceiling,
    matching your paper's own Figure 1 metric) and `avg_tokens` (raw response length). All four
    come from ONE score_response call and ONE tokenization pass per response, not four, to avoid
    quadrupling judge API cost. The same four are ALSO computed for the Prompt-alone baseline and
    attached to every result as prompt_fully_correct_rate/prompt_avg_tokens, so every comparison
    has both sides without a second run. `correctness_scores` (the raw, unaggregated per-example
    list) is kept on each result specifically so select_constrained_survivors can run a PAIRED
    comparison against the reference candidate later. Every non-skipped result also gets
    on_pareto_frontier (see mark_pareto_frontier) attached before returning.

    candidates: each a dict with at least {"layer", "mse_weight", "nll_weight"} and whatever
    else that variant's retrain function needs (e.g. "alpha"). Usually rows pulled straight from
    a sweep JSONL (best_row_per_layer's output) or a hand-built hyperparameter grid for Tier 2.
    """
    retrain_fn = RETRAIN_FNS[variant]
    rows = adapter.load_rows(split)[:n_examples]
    items = adapter.to_items(ctx.tokenizer, rows)
    prompts = [item["base_prompt"] for item in items]

    hook_fns_by_group, layer_by_group, prompts_by_group = {}, {}, {}
    for i, candidate in enumerate(candidates):
        group_id = i
        hook_fn = retrain_fn(candidate, ctx)
        prompts_by_group[group_id] = prompts
        layer_by_group[group_id] = candidate["layer"]
        if hook_fn is not None:
            hook_fns_by_group[group_id] = hook_fn
        # A None hook_fn (e.g. selfproj's delta_scale-too-small skip) still gets a prompts/layer
        # entry so it occupies a slot in chunking math, but generate_with_routed_configs treats a
        # group missing from hook_fns_by_group as "no correction" -- see below, its response is
        # discarded rather than judged, since "no correction" isn't what this candidate meant.
        if hook_fn is None:
            del prompts_by_group[group_id]  # don't waste batch rows generating something we'll discard

    # The Prompt-alone baseline: the actual instructed prompt, no hook, and deliberately no entry
    # in layer_by_group at all -- generate_with_routed_configs treats a group absent from
    # layer_by_group as never touched by any layer's routed hook, which is exactly "no steering,"
    # the same mechanism already used for baseline rows elsewhere in this file.
    terse_prompts = [item["terse_prompt"] for item in items]
    prompts_by_group[_PROMPT_BASELINE_GROUP] = terse_prompts

    responses_by_group: dict[Any, list[str]] = {}
    for chunk_group_ids in _chunk_groups_by_row_budget(prompts_by_group, max_batch_rows):
        chunk_prompts = {g: prompts_by_group[g] for g in chunk_group_ids}
        chunk_hooks = {g: hook_fns_by_group[g] for g in chunk_group_ids if g in hook_fns_by_group}
        chunk_layers = {g: layer_by_group[g] for g in chunk_group_ids if g in layer_by_group}
        chunk_responses = generate_with_routed_configs(ctx.model, ctx.tokenizer, chunk_prompts, chunk_hooks, chunk_layers)
        responses_by_group.update(chunk_responses)

    primary_field = eval_adapter.SCORE_FIELDS[0]
    resolved_optimize_field = optimize_field
    if resolved_optimize_field is None and "conciseness" in eval_adapter.SCORE_FIELDS:
        resolved_optimize_field = "conciseness"

    prompt_responses = responses_by_group[_PROMPT_BASELINE_GROUP]
    prompt_baseline_scores = [eval_adapter.score_response(row, resp)[primary_field] for row, resp in zip(rows, prompt_responses)]
    prompt_fully_correct_rate = _fully_correct_rate(prompt_baseline_scores, primary_field_max)
    prompt_avg_tokens = _avg_tokens(ctx, prompt_responses)

    results = []
    for i, candidate in enumerate(candidates):
        if i not in responses_by_group:
            results.append({**candidate, "skipped": True})
            continue
        responses = responses_by_group[i]
        score_dicts = [eval_adapter.score_response(row, resp) for row, resp in zip(rows, responses)]
        correctness_scores = [d[primary_field] for d in score_dicts]
        point, lo, hi = bootstrap_ci(correctness_scores)

        if resolved_optimize_field is not None:
            optimize_scores = [d[resolved_optimize_field] for d in score_dicts]
        else:
            optimize_scores = [-len(ctx.tokenizer(resp)["input_ids"]) for resp in responses]  # fewer tokens = better, so negate
        optimize_point, optimize_lo, optimize_hi = bootstrap_ci(optimize_scores)

        results.append({
            **candidate, "skipped": False, "n": len(responses),
            "judged_score": point, "ci_lo": lo, "ci_hi": hi, "correctness_scores": correctness_scores,
            "optimize_score": optimize_point, "optimize_ci_lo": optimize_lo, "optimize_ci_hi": optimize_hi,
            "fully_correct_rate": _fully_correct_rate(correctness_scores, primary_field_max),
            "avg_tokens": _avg_tokens(ctx, responses),
            "prompt_baseline_scores": prompt_baseline_scores,
            "prompt_fully_correct_rate": prompt_fully_correct_rate,
            "prompt_avg_tokens": prompt_avg_tokens,
        })
    return mark_pareto_frontier(results)


def select_constrained_survivors(results: list[dict], top_k: int) -> list[dict]:
    """Constrained optimization, not a single blended metric, with TWO gates that are NOT
    symmetric in how strictly they're enforced:

    (1) HARD gate -- not statistically significantly worse than the best-in-grid candidate
        (paired bootstrap comparison, evals.bootstrap_analysis.paired_bootstrap_diff). This one
        genuinely eliminates candidates: don't let within-grid noise pick something needlessly
        worse than its peers.
    (2) SOFT gate -- not statistically significantly worse than Prompt-alone (same paired-
        comparison logic, against `prompt_baseline_scores`). Preferred, not absolute: if there's
        a genuine accuracy/conciseness tradeoff (the whole premise of this project), EVERY
        candidate could legitimately lose to Prompt, and a hard floor here would mean this
        function returns nothing, forever, which is useless. So: if at least one gate-(1)
        survivor also clears gate (2), rank THOSE by optimize_score (conciseness) as before. If
        NONE do, fall back to ranking gate-(1) survivors by correctness alone instead (conciseness
        is not a safe thing to optimize for once nothing has established a real accuracy floor),
        and flag it -- callers can tell a fallback happened via each returned result's
        `prompt_floor_fallback` key.
    """
    usable = [r for r in results if not r.get("skipped", False)]
    if not usable:
        return []
    reference = max(usable, key=lambda r: r["judged_score"])

    grid_gate_survivors = []
    for r in usable:
        if r is not reference:
            _, lo, _ = paired_bootstrap_diff(reference["correctness_scores"], r["correctness_scores"])
            if lo > 0:  # CI on (reference - r) is entirely positive -- r is significantly worse than the grid's best
                continue
        grid_gate_survivors.append(r)

    prompt_gate_survivors = []
    for r in grid_gate_survivors:
        prompt_baseline_scores = r.get("prompt_baseline_scores")
        if prompt_baseline_scores is None:
            prompt_gate_survivors.append(r)  # no baseline available to compare against -- can't gate, let it through
            continue
        _, lo, _ = paired_bootstrap_diff(prompt_baseline_scores, r["correctness_scores"])
        if lo > 0:  # CI on (prompt - r) is entirely positive -- r is significantly worse than Prompt alone
            continue
        prompt_gate_survivors.append(r)

    if prompt_gate_survivors:
        winners = sorted(prompt_gate_survivors, key=lambda r: -r["optimize_score"])[:top_k]
        return [{**r, "prompt_floor_fallback": False} for r in winners]

    print("WARNING: no candidate beat or tied Prompt alone -- falling back to the highest "
          "correctness among the grid's survivors (conciseness is NOT used to rank in this "
          "fallback, since it isn't safe to optimize for terseness with no accuracy floor met).")
    winners = sorted(grid_gate_survivors, key=lambda r: -r["judged_score"])[:top_k]
    return [{**r, "prompt_floor_fallback": True} for r in winners]


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
    max_batch_rows: int = DEFAULT_MAX_BATCH_ROWS,
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
        tier1_results = evaluate_candidates(variant, tier1_candidates, ctx, adapter, eval_adapter, tier1_n, split="dev", max_batch_rows=max_batch_rows)
        write("tier1", tier1_results)

    survivors = select_constrained_survivors(tier1_results, top_k_layers)
    if not survivors:
        print("WARNING: every Tier 1 candidate was skipped -- nothing to search further")
        return
    if survivors[0]["prompt_floor_fallback"]:
        print(f"Tier 1 survivors (FALLBACK -- none beat/tied Prompt alone, ranked by correctness "
              f"instead of conciseness): {[s['layer'] for s in survivors]}")
    else:
        print(f"Tier 1 survivors (top {top_k_layers} by conciseness, among those beating/tying "
              f"Prompt and not significantly less correct than the grid's best): {[s['layer'] for s in survivors]}")

    if "tier2" in existing:
        print(f"Tier 2 already complete ({len(existing['tier2'])} rows) -- reusing from {out_path}")
        tier2_results = existing["tier2"]
    else:
        print(f"\n== Tier 2: hyperparameter sweep at surviving layer(s), n={tier2_n} ==")
        survivor_layers = {s["layer"] for s in survivors}
        tier2_candidates = [r for r in sweep_rows if r.get("layer") in survivor_layers and not r.get("skipped", False) and "final_mse" in r]
        print(f"{len(tier2_candidates)} (layer, hyperparameter) combos to evaluate at the surviving layer(s)")
        tier2_results = evaluate_candidates(variant, tier2_candidates, ctx, adapter, eval_adapter, tier2_n, split="dev", max_batch_rows=max_batch_rows)
        write("tier2", tier2_results)

    winner_list = select_constrained_survivors(tier2_results, top_k=1)
    winner = winner_list[0] if winner_list else None
    if winner is None:
        print("WARNING: every Tier 2 candidate was skipped -- no final confirmation run")
        return
    fallback_note = " -- FALLBACK, does not beat/tie Prompt alone" if winner["prompt_floor_fallback"] else ""
    print(f"Tier 2 winner{fallback_note}: layer={winner['layer']}, "
          f"{({k: v for k, v in winner.items() if k not in ('tier', 'correctness_scores', 'prompt_baseline_scores')})}")

    print(f"\n== Final: full-size confirmation run, n={final_n}, TEST split ==")
    final_results = evaluate_candidates(variant, [winner], ctx, adapter, eval_adapter, final_n, split="test", max_batch_rows=max_batch_rows)
    write("final", final_results)
    r = final_results[0]
    print(f"Final correctness (judged_score, mean 0-2): {r['judged_score']:.4f} [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] (95% CI, n={final_n})")
    print(f"Final fully_correct_rate (matches the paper's Figure 1 metric): {r['fully_correct_rate']:.4f}  "
          f"(Prompt alone: {r['prompt_fully_correct_rate']:.4f})")
    print(f"Final optimize_score (mean judged conciseness or -avg_tokens): {r['optimize_score']:.4f} "
          f"[{r['optimize_ci_lo']:.4f}, {r['optimize_ci_hi']:.4f}]")
    print(f"Final avg_tokens: {r['avg_tokens']:.1f}  (Prompt alone: {r['prompt_avg_tokens']:.1f})")

    prompt_scores = r.get("prompt_baseline_scores")
    if prompt_scores is not None:
        diff, diff_lo, diff_hi = paired_bootstrap_diff(r["correctness_scores"], prompt_scores)
        verdict = "SIGNIFICANTLY BEATS Prompt alone" if diff_lo > 0 else \
                  "SIGNIFICANTLY WORSE than Prompt alone" if diff_hi < 0 else \
                  "not significantly different from Prompt alone"
        print(f"vs. Prompt alone (paired, n={final_n}): diff={diff:.4f} [{diff_lo:.4f}, {diff_hi:.4f}] -- {verdict}")

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
        vs_prompt = None
        prompt_scores = row.get("prompt_baseline_scores")
        if prompt_scores is not None and "correctness_scores" in row:
            diff, lo, hi = paired_bootstrap_diff(row["correctness_scores"], prompt_scores)
            vs_prompt = "beats Prompt" if lo > 0 else "loses to Prompt" if hi < 0 else "ties Prompt"
        summary[variant] = {
            "status": "done", "layer": row["layer"], "judged_score": row["judged_score"],
            "ci_lo": row["ci_lo"], "ci_hi": row["ci_hi"],
            "optimize_score": row.get("optimize_score"), "optimize_ci_lo": row.get("optimize_ci_lo"),
            "optimize_ci_hi": row.get("optimize_ci_hi"), "vs_prompt": vs_prompt, "n": row["n"],
        }

    print(f"{'variant':<22}{'status':<45}{'layer':>8}{'correctness':>14}{'conciseness':>14}{'vs Prompt':>16}")
    for variant, s in summary.items():
        if s["status"] != "done":
            print(f"{variant:<22}{s['status']:<45}")
            continue
        correctness = f"{s['judged_score']:.3f}"
        conciseness = f"{s['optimize_score']:.3f}" if s.get("optimize_score") is not None else "n/a"
        vs_prompt_str = s.get("vs_prompt") or "n/a"
        print(f"{variant:<22}{'done':<45}{s['layer']:>8}{correctness:>14}{conciseness:>14}{vs_prompt_str:>16}")

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
    parser.add_argument("--max-batch-rows", type=int, default=DEFAULT_MAX_BATCH_ROWS,
                        help="cap on rows (candidates x prompts) per generate() call -- lower this if you hit CUDA OOM")
    args = parser.parse_args()
    if args.summarize:
        summarize_all_variants(args.task)
    else:
        if not args.variant:
            parser.error("--variant is required unless --summarize is given")
        run_tiered_search(args.task, args.variant, args.tier1_n, args.tier2_n, args.final_n, args.top_k_layers, args.seed, args.max_batch_rows)
