"""Surface ablation with the configuration held FIXED.

Evaluates a method at its own already-selected (layer, coefficient) under both intervention
surfaces -- all positions vs. response-only -- so the single thing that differs between the two
numbers is where the intervention lands.

WHY NOT REUSE THE TIERED SEARCH. Running `--variant const_resp` through
evals/layer_hparam_search.py re-searches the whole 13-layer x 10-coefficient grid: 130 candidates
x 20 examples x 2 judge calls = 5200 API calls, hours at any safe concurrency. Worse, it answers
a different question. A fresh search picks the best config FOR the response-only surface, which
may be a different layer than the all-positions winner -- and then the gap between the two numbers
confounds surface with configuration. That already happened: Stolfo's all-positions search won at
layer 16, its response-only search won at layer 18, so those two results are not a clean surface
comparison.

This script instead reads the parent variant's Final row, rebuilds that EXACT config, and
generates under both surfaces. Cost is 2 x n generations (~360) and no training at all, so it runs
in minutes rather than hours -- and the difference it reports is attributable to the surface
alone.

Both surfaces are regenerated here rather than reusing the parent's stored alone number, because
the stored one came from the tiered search's own generation pass; regenerating both in one run on
identical rows keeps the pair exactly comparable and lets the paired bootstrap be valid.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.registry import get_adapter
from core.generation_cache import load_or_compute_responses
from core.model_common import generate_response, load_model, num_layers, token_count
from evals.bootstrap_analysis import bootstrap_ci, paired_bootstrap_diff
from evals.layer_hparam_search import (
    DEFAULT_MAX_BATCH_ROWS, SearchContext, _score_all_concurrently, load_existing_tiered_results,
)
from evals.registry import get_eval_adapter
from steering.batch_routing import generate_batched_uniform

# variant -> (tiered-search stems to look for, functional form)
PARENTS = {
    "const":  (["nogate_dim", "const"], "additive"),
    "stolfo": (["nogate_clamp", "stolfo"], "clamp"),
}


def _find_final(results_dir: Path, stems: list[str]) -> dict:
    for stem in stems:
        p = results_dir / f"{stem}_tiered_search.jsonl"
        if p.exists():
            finals = load_existing_tiered_results(p).get("final")
            if finals:
                return finals[0]
    raise FileNotFoundError(
        f"no tiered-search Final found (looked for {', '.join(s + '_tiered_search.jsonl' for s in stems)})")


def build_hooks(form: str, winner: dict, ctx: SearchContext):
    """Returns {"all_positions": hook, "response_only": hook} at the winner's exact config."""
    from steering.const.direction import compute_diff_mean_direction
    from steering.psr.data import load_or_pool_prompt_last_token

    layer = winner["layer"]
    base_pool, instr_pool = load_or_pool_prompt_last_token(
        ctx.model, ctx.tokenizer, ctx.train_items, layer, ctx.cache_dir)
    direction = compute_diff_mean_direction(base_pool, instr_pool)

    if form == "clamp":
        from steering.clamp.hooks import make_clamp_hook
        from steering.stolfo.direction import compute_target_projection
        target = compute_target_projection(instr_pool, direction)
        return {
            "all_positions": make_clamp_hook(direction, target, response_only=False),
            "response_only": make_clamp_hook(direction, target, response_only=True),
        }, f"layer={layer} target={target:.4f}"

    from steering.const.hooks import make_const_hook
    coeff = winner["coeff"]
    base_hook = make_const_hook(direction, coeff)

    def response_only(hidden):
        if hidden.shape[1] > 1:   # prefill -- identical guard to make_inference_hook's
            return hidden
        return base_hook(hidden)

    return {"all_positions": base_hook, "response_only": response_only}, f"layer={layer} coeff={coeff}"


def main(task: str, variants: list[str], n: int, seed: int, max_batch_rows: int,
         skip_judge: bool = False) -> None:
    adapter = get_adapter(task)
    eval_adapter = get_eval_adapter(task)
    primary = eval_adapter.SCORE_FIELDS[0]

    model, tokenizer = load_model("cuda", model_name=adapter.MODEL_NAME)
    for p in model.parameters():
        p.requires_grad_(False)
    train_items = adapter.to_items(tokenizer, adapter.load_rows("train"))
    ctx = SearchContext(
        model=model, tokenizer=tokenizer, n_layers=num_layers(model),
        hidden_size=model.config.hidden_size, cache_dir=adapter.CACHE_DIR,
        train_items=train_items, dev_items=[], train_responses={}, dev_responses={}, seed=seed,
    )

    test_rows = adapter.load_rows("test")[:n]
    test_items = adapter.to_items(tokenizer, test_rows)
    base_prompts = [it["base_prompt"] for it in test_items]

    out_all = {}
    for variant in variants:
        stems, form = PARENTS[variant]
        winner = _find_final(adapter.RESULTS_DIR, stems)
        hooks, cfg = build_hooks(form, winner, ctx)
        print(f"\n{variant} ({form}) at its own Final config: {cfg}")

        res = {}
        for surface, hook in hooks.items():
            with torch.no_grad():
                responses = generate_batched_uniform(
                    model, tokenizer, base_prompts, hooks={winner["layer"]: hook},
                    max_batch_rows=max_batch_rows)
            # avg_tokens costs NOTHING -- it's the local tokenizer, no API. It is also the primary
            # length metric for this project, so the headline surface result is fully obtainable
            # with zero judge budget. Correctness is the only part that needs the API.
            tokens = sum(token_count(tokenizer, r) for r in responses) / len(responses)
            entry = {"avg_tokens": tokens, "responses": responses}

            if skip_judge:
                print(f"  {surface:<15} {tokens:6.1f} tok   (judging skipped)")
            else:
                dicts = _score_all_concurrently(eval_adapter, test_rows, responses)
                scores = [d[primary] for d in dicts]
                pt, lo, hi = bootstrap_ci(scores)
                entry.update(correct=pt, ci_lo=lo, ci_hi=hi, scores=scores,
                              coherent_rate=sum(1.0 if d["coherent"] else 0.0 for d in dicts) / len(dicts))
                print(f"  {surface:<15} {tokens:6.1f} tok / {pt:.3f} [{lo:.3f}, {hi:.3f}]")
            res[surface] = entry

        delta_tok = res["response_only"]["avg_tokens"] - res["all_positions"]["avg_tokens"]
        print(f"  SURFACE EFFECT: {delta_tok:+.1f} tokens when restricted to response-only")
        if not skip_judge:
            d, dlo, dhi = paired_bootstrap_diff(
                res["response_only"]["scores"], res["all_positions"]["scores"])
            print(f"                  correctness {d:+.4f} [{dlo:+.4f}, {dhi:+.4f}]")

        out_all[variant] = {"config": cfg, "layer": winner["layer"], "form": form, "n": n,
                             "judged": not skip_judge,
                             "surfaces": res, "token_delta_response_only_minus_all": delta_tok}

    path = adapter.RESULTS_DIR / "surface_ablation.json"
    with path.open("w") as f:
        json.dump(out_all, f, indent=2)
    print(f"\nwrote {path}")
    if skip_judge:
        print("Raw responses are saved in that file, so judging can be added later without "
               "regenerating anything:\n  python3 scripts/surface_ablation.py --judge-saved")


def judge_saved(task: str) -> None:
    """Scores the responses already stored in surface_ablation.json. Lets the expensive-but-free
    part (generation) and the cheap-but-rate-limited part (judging) happen in separate sessions,
    which matters when a daily request cap is the binding constraint rather than GPU time."""
    adapter = get_adapter(task)
    eval_adapter = get_eval_adapter(task)
    primary = eval_adapter.SCORE_FIELDS[0]
    path = adapter.RESULTS_DIR / "surface_ablation.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run with --skip-judge first to generate.")
    with path.open() as f:
        payload = json.load(f)
    # No model load here at all: the responses are already on disk and judging is pure API work.
    test_rows = None

    for variant, v in payload.items():
        if v.get("judged"):
            print(f"{variant}: already judged, skipping")
            continue
        if test_rows is None:
            test_rows = adapter.load_rows("test")[: v["n"]]
        print(f"\n{variant} ({v['config']})")
        for surface, entry in v["surfaces"].items():
            if "scores" in entry:
                continue
            dicts = _score_all_concurrently(eval_adapter, test_rows, entry["responses"])
            scores = [d[primary] for d in dicts]
            pt, lo, hi = bootstrap_ci(scores)
            entry.update(correct=pt, ci_lo=lo, ci_hi=hi, scores=scores,
                          coherent_rate=sum(1.0 if d["coherent"] else 0.0 for d in dicts) / len(dicts))
            print(f"  {surface:<15} {entry['avg_tokens']:6.1f} tok / {pt:.3f} [{lo:.3f}, {hi:.3f}]")
        if all("scores" in e for e in v["surfaces"].values()):
            d, dlo, dhi = paired_bootstrap_diff(
                v["surfaces"]["response_only"]["scores"], v["surfaces"]["all_positions"]["scores"])
            print(f"  correctness effect: {d:+.4f} [{dlo:+.4f}, {dhi:+.4f}]")
            v["judged"] = True
        with path.open("w") as f:
            json.dump(payload, f, indent=2)
    print(f"\nupdated {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="caveman", choices=["caveman", "ifeval"])
    ap.add_argument("--variants", default="stolfo,const",
                     help=f"comma-separated, from: {', '.join(PARENTS)}")
    ap.add_argument("--n", type=int, default=180)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-batch-rows", type=int, default=DEFAULT_MAX_BATCH_ROWS)
    ap.add_argument("--skip-judge", action="store_true",
                     help="generate and report avg_tokens only -- ZERO API calls. Responses are "
                          "saved so --judge-saved can add correctness later.")
    ap.add_argument("--judge-saved", action="store_true",
                     help="score responses already saved by a previous --skip-judge run")
    a = ap.parse_args()
    if a.judge_saved:
        judge_saved(a.task)
    else:
        main(a.task, [v.strip() for v in a.variants.split(",")], a.n, a.seed, a.max_batch_rows,
             skip_judge=a.skip_judge)
