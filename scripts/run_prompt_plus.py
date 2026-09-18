"""Generate the +Prompt (joint intervention) condition for any variant that already has a
tiered-search Final row.

Supersedes scripts/prompt_plus_steer_missing_piece.py, which hardcoded ("proper", "conceptor")
and carried a block of reference numbers copied from the old caveman-steer repo. This takes the
variant list on the command line and derives every number from THIS repo's files.

WHY A SEPARATE RUN IS NEEDED AT ALL. evals/layer_hparam_search.py only ever steers `base_prompt`
(see its evaluate_candidates: `prompts = [item["base_prompt"] ...]`, with `terse_prompt` used only
for the un-steered Prompt baseline). So every tiered search produces the ALONE condition and
nothing else. The +Prompt condition -- which is what the paper's joint-intervention claim is
actually about -- has to be generated separately, which is why it was missing for most methods.

WHAT IS REUSED, NOT REGENERATED. The Final row already stores, at n=180 on the test split and
paired to the same rows:
    avg_tokens + correctness_scores        -> the alone condition
    prompt_avg_tokens + prompt_baseline_scores -> Prompt alone
so only the combined generation is new. That keeps this ~180 generations per variant instead of
~540, and keeps the paired bootstrap valid because all three conditions share identical rows.

Works for trained and training-free variants alike: RETRAIN_FNS already contains entries for
"sg", "conceptor", "proper", "conceptor_matrix", "conceptor_selfproj", "const" and "stolfo", and
each returns a ready-to-use inference hook from that variant's own Final hyperparameters. For
const/stolfo "retraining" is just recomputing a closed-form direction, so those are cheap.
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
    DEFAULT_MAX_BATCH_ROWS, RETRAIN_FNS, SearchContext, _score_all_concurrently,
    load_existing_tiered_results,
)
from evals.registry import get_eval_adapter
from steering.batch_routing import generate_batched_uniform

DEFAULT_N = 180


def _tiered_path(results_dir: Path, variant: str) -> Path:
    """Accepts both the legacy and the post-migration filename. scripts/migrate_naming.py runs in
    copy mode by default, so during the transition BOTH exist; after a --mode move only the new
    one does. Checking new-first means this keeps working either way without a flag."""
    new_stem = {
        "sg": "sg_dim", "conceptor": "sg_conc", "proper": "sg_gradient_trained",
        "conceptor_matrix": "sg_conc_matrix", "conceptor_selfproj": "sg_conc_selfproj",
        "const": "nogate_dim", "stolfo": "nogate_clamp",
    }.get(variant, variant)
    for stem in (new_stem, variant):
        p = results_dir / f"{stem}_tiered_search.jsonl"
        if p.exists():
            return p
    raise FileNotFoundError(
        f"no tiered search for '{variant}' (looked for {new_stem}_tiered_search.jsonl and "
        f"{variant}_tiered_search.jsonl in {results_dir}). Run its layer search first."
    )


def _out_path(results_dir: Path, variant: str) -> Path:
    stem = {
        "sg": "sg_dim", "conceptor": "sg_conc", "proper": "sg_gradient_trained",
        "conceptor_matrix": "sg_conc_matrix", "conceptor_selfproj": "sg_conc_selfproj",
        "const": "nogate_dim", "stolfo": "nogate_clamp",
    }.get(variant, variant)
    return results_dir / f"{stem}_prompt_plus.json"


def build_context(adapter, seed: int) -> SearchContext:
    model, tokenizer = load_model("cuda", model_name=adapter.MODEL_NAME)
    for p in model.parameters():
        p.requires_grad_(False)
    train_items = adapter.to_items(tokenizer, adapter.load_rows("train"))
    dev_items = adapter.to_items(tokenizer, adapter.load_rows("dev"))
    train_responses = load_or_compute_responses(model, tokenizer, train_items, adapter.CACHE_DIR / "teacher_responses_train.json", generate_response)
    dev_responses = load_or_compute_responses(model, tokenizer, dev_items, adapter.CACHE_DIR / "teacher_responses_dev.json", generate_response)
    return SearchContext(
        model=model, tokenizer=tokenizer, n_layers=num_layers(model),
        hidden_size=model.config.hidden_size, cache_dir=adapter.CACHE_DIR,
        train_items=train_items, dev_items=dev_items,
        train_responses=train_responses, dev_responses=dev_responses, seed=seed,
    )


def run_variant(variant: str, adapter, eval_adapter, ctx: SearchContext, n: int, max_batch_rows: int) -> dict:
    winner = load_existing_tiered_results(_tiered_path(adapter.RESULTS_DIR, variant)).get("final")
    if not winner:
        raise RuntimeError(f"{variant}: tiered search has no 'final' row -- it didn't finish.")
    winner = winner[0]
    cfg = f"layer={winner['layer']}"
    for k in ("alpha", "coeff", "mse_weight", "nll_weight"):
        if winner.get(k) is not None:
            cfg += f" {k}={winner[k]}"
    print(f"\n{variant}: {cfg}")

    if variant not in RETRAIN_FNS:
        raise KeyError(f"{variant} not in RETRAIN_FNS ({sorted(RETRAIN_FNS)})")
    hook_fn = RETRAIN_FNS[variant](winner, ctx)
    if hook_fn is None:
        raise RuntimeError(f"{variant}: its winning config is a 'skipped' point -- can't generate.")

    # Same [:n] slice the Final run used, so the rows are identical and the paired tests below are
    # valid against the stored alone/prompt-alone scores rather than a fresh, differently-sampled set.
    test_rows = adapter.load_rows("test")[:n]
    test_items = adapter.to_items(ctx.tokenizer, test_rows)
    terse_prompts = [it["terse_prompt"] for it in test_items]

    print(f"  generating Prompt+Steer, n={n} (batched) ...")
    with torch.no_grad():
        responses = generate_batched_uniform(
            ctx.model, ctx.tokenizer, terse_prompts,
            hooks={winner["layer"]: hook_fn}, max_batch_rows=max_batch_rows,
        )

    primary = eval_adapter.SCORE_FIELDS[0]
    score_dicts = _score_all_concurrently(eval_adapter, test_rows, responses)
    combined_scores = [d[primary] for d in score_dicts]
    combined_tokens = sum(token_count(ctx.tokenizer, r) for r in responses) / len(responses)
    coherent = sum(1.0 if d["coherent"] else 0.0 for d in score_dicts) / len(score_dicts)

    out = {
        "variant": variant, "n": n, "config": cfg, "layer": winner["layer"],
        "prompt_plus_steer_avg_tokens": combined_tokens,
        "prompt_plus_steer_correctness_scores": combined_scores,
        "prompt_plus_steer_coherent_rate": coherent,
        # Carried over from the Final row so a downstream reader has all three conditions in one
        # file without needing to re-open the tiered search.
        "steer_alone_avg_tokens": winner.get("avg_tokens"),
        "steer_alone_correctness_scores": winner.get("correctness_scores"),
        "prompt_alone_avg_tokens": winner.get("prompt_avg_tokens"),
        "prompt_alone_correctness_scores": winner.get("prompt_baseline_scores"),
    }
    if "conciseness" in eval_adapter.SCORE_FIELDS:
        out["prompt_plus_steer_conciseness_scores"] = [d["conciseness"] for d in score_dicts]

    path = _out_path(adapter.RESULTS_DIR, variant)
    with path.open("w") as f:
        json.dump(out, f, indent=2)

    pt, lo, hi = bootstrap_ci(combined_scores)
    print(f"  +Prompt: {combined_tokens:.1f} tok / {pt:.3f} [{lo:.3f}, {hi:.3f}]  coherent={coherent:.2f}")
    if out["prompt_alone_correctness_scores"]:
        d, dlo, dhi = paired_bootstrap_diff(combined_scores, out["prompt_alone_correctness_scores"])
        verdict = "BEATS Prompt" if dlo > 0 else "WORSE than Prompt" if dhi < 0 else "ties Prompt"
        print(f"  vs Prompt alone ({out['prompt_alone_avg_tokens']:.1f} tok): "
              f"{d:+.4f} [{dlo:+.4f}, {dhi:+.4f}] -- {verdict}")
    print(f"  wrote {path}")
    return out


def main(task: str, variants: list[str], n: int, seed: int, max_batch_rows: int) -> None:
    adapter = get_adapter(task)
    eval_adapter = get_eval_adapter(task)
    print(f"loading model once for {len(variants)} variant(s): {', '.join(variants)}")
    ctx = build_context(adapter, seed)

    failures = []
    for v in variants:
        try:
            run_variant(v, adapter, eval_adapter, ctx, n, max_batch_rows)
        except Exception as e:
            print(f"  FAILED {v}: {type(e).__name__}: {e}")
            failures.append(v)

    print(f"\ndone. {len(variants) - len(failures)}/{len(variants)} succeeded"
          + (f"; failed: {', '.join(failures)}" if failures else ""))
    print("run scripts/inventory.py to see the updated grid")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="caveman", choices=["caveman", "ifeval"])
    ap.add_argument("--variants", required=True,
                     help="comma-separated, e.g. 'stolfo,const,sg'. Must be keys of RETRAIN_FNS.")
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-batch-rows", type=int, default=DEFAULT_MAX_BATCH_ROWS)
    a = ap.parse_args()
    main(a.task, [v.strip() for v in a.variants.split(",") if v.strip()], a.n, a.seed, a.max_batch_rows)
